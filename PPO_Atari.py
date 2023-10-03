import math
import random
import time

import gym
from gym.wrappers import AtariPreprocessing
#from gym.wrappers import Monitor
import numpy as np
import wandb


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical


import argparse
import os

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-f') #dummy argument for interactive interpreter
    parser.add_argument("--exp-name", type=str, default = "PPO", help = "experiment name")
    parser.add_argument("--env-name", type=str, default = "BreakoutNoFrameskip-v4", help = "environment name")
    parser.add_argument("--lr", type=float, default = 2.5e-4, help = "learning rate of the NN")
    parser.add_argument("--seed", type=int, default=1, help = "seed of the experiment")
    parser.add_argument("--max-step", type=int, default=10000000, help = "the maximum step of the experiment")
    parser.add_argument("--wandb-track", type=bool, default=False, help="the flag of tracking the experiment data")
    parser.add_argument("--seed-value", type=int, default=1, help="seed of the experiment")

    # Agent specific arguments
    parser.add_argument("--num-env", type=int, default=8, help = "the number of parallel environments")
    parser.add_argument("--horizon", type=int, default=128, help = "the number of step in each rollout")
    parser.add_argument("--lam", type=float, default=0.95, help = "the lambda for estimating generalized advantage")
    parser.add_argument("--gamma", type=float, default=0.99, help = "the gamma for estimating generalizedd advantage")
    parser.add_argument("--num-epoch", type=int, default=4, help = "the number of epoch in each parameter update")
    parser.add_argument("--size-miniBatch", type=int, default=256, help = "the size of minibatch")
    parser.add_argument("--clip-coef", type=float, default=0.1, help = "the surrogate clipping coefficient controlingt how much the policy distribution can be updated from the old policy distribution")
    parser.add_argument("--entro-coef", type=float, default=0.01, help = "the entropy coefficient controlling the exploration rate of the agent")
    parser.add_argument("--vf-coef", type=float, default=0.5, help = "the value function coefficient controlling the weight of value function loss in the total loss")
    parser.add_argument("--anneal-lr", type=bool, default=True, help="flag of using annealing learning rate")

    args = parser.parse_args()
    args.batch_size = args.num_env * args.horizon
    args.num_miniBatch = args.batch_size // args.size_miniBatch

    return args

def make_env(vis, env_name, seed):
    def _thunk():
        env = gym.make(env_name)
        # revcording the agent performance
        if vis: env = gym.wrappers.RecordVideo(env, f"videos")
        env = AtariPreprocessing(env, screen_size=84, terminal_on_life_loss=True,grayscale_obs=True, frame_skip=4, noop_max=30)
        env = gym.wrappers.FrameStack(env,4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _thunk


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def test_env(vis = False, model = None):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if model is not None:
            with torch.no_grad():
                dist, _ = model(state)
                action = dist.sample().cpu().numpy()[0]
        else:
            action = np.random.randint(4)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean = 0, std = 0.1)
        nn.init.constant_(m.bias, 0.1)

#PPO network
class Actor_Critic(nn.Module):
    def __init__(self, num_inputs_layer, num_outputs, action_type = 'continuous', std = 0.0):
        super(Actor_Critic, self).__init__()

        self.action_type = action_type

        self.shared_net = nn.Sequential(
            nn.Conv2d(num_inputs_layer, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.critic = nn.Linear(512, 1)

        self.actor = nn.Linear(512, num_outputs)

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)


    def forward(self, x):
        x = self.shared_net(x/255.0)
        value = self.critic(x)
         #critic net outputs value of a state
        if self.action_type == 'continuous':               #continuous action space
            mu = self.actor(x)                             #action net ouputs the mean of the pi(a | x)
            std = self.log_std.exp().expand_as(mu)         #get a covariance matrix
            dist = Normal(mu, std)                         # pi(a | x) = NN(x) + sigma     sigma = 0
        elif self.action_type == 'discrete':               #discrete action space
            logits = self.actor(x)
            dist = Categorical(logits = logits)
        return dist, value
    
    def get_value(self, x):
        x = self.shared_net(x/255.0)
        value = self.critic(x)
        return value


def compute_gae(gamma, lam, next_value, rewards, masks, values):


    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae               # Advantage
        returns.insert(0, gae + values[step])                       # Advantage + value = return
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)

        #return one mini_batch
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]   #how does the mini_batch be divided?



def ppo_update(clip_coef, vf_coef, entro_coef, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages):

    losses = torch.zeros(ppo_epochs)
    actor_losses = torch.zeros(ppo_epochs)
    critic_losses = torch.zeros(ppo_epochs)

    for i in range(ppo_epochs):
        # train on mini_batch
        j = 0
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()             #entropy of policy distribution: encourage explooration
            if model.action_type == 'discrete':
                action = action.squeeze(1)
            new_log_probs = dist.log_prob(action)       #log new_pi(a | s)
            if model.action_type == 'discrete':
                new_log_probs = new_log_probs.unsqueeze(1)


            log_ratio = (new_log_probs - old_log_probs)
            ratio = log_ratio.exp()    #new_pi(a | s) / old_pi(a | s)

            #if i == 0 and j == 0:
            #  print(ratio)
            # calculate approximate Kl between old pi and new pi
            #with torch.no_grad():
            #  old_approx_kl = (-log_ratio).mean()
            #  approx_kl = ((ratio - 1) - log_ratio).mean()

            surr1 = ratio * advantage                        #new_pi(a | s) / old_pi(a | s) * A
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()      #MSE between true return and value function

            actor_losses[i] = actor_loss
            critic_losses[i] = critic_loss


            loss = vf_coef * critic_loss + actor_loss - entro_coef * entropy
            losses[i] = loss

            j += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses.mean(), actor_losses.mean(), critic_losses.mean()

  #print(f'Loss in step {frame_idx} is: {losses.mean()}')
  #print(f'KL in step {frame_idx} is: {approx_kl}')


if __name__ == "__main__":
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    vis = False
    # seed the experiment
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    torch.manual_seed(args.seed_value)


    #create enviroment with different seeds
    envs = [make_env(vis, args.env_name, seed=args.seed_value+i) for i in range(args.num_env)]
    envs = gym.vector.SyncVectorEnv(envs)
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30)
    env = gym.wrappers.FrameStack(env,4)

    num_inputs_layer = envs.single_observation_space.shape[0]
    num_outputs = envs.single_action_space.n

    action_type = 'discrete'

    threshold_reward = 500




    model = Actor_Critic(num_inputs_layer, num_outputs,  action_type=action_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    #initial wandb track
    print(args.wandb_track)
    if args.wandb_track:
        wandb.init(
            project = "PPO_Atari",
            config=vars(args),
            name = f"{args.exp_name}_{args.env_name}_{int(time.time())}",
            save_code=True
        )
        wandb.define_metric("global_step")

    max_step = args.max_step
    step = 0
    test_rewards = []
    best_reward = 0
    current_reward_envs = np.zeros(args.num_env)

    max_updates = max_step // args.batch_size
    num_updates = 0
    
    #main loop
    state = envs.reset()
    early_stop = False

    while step < max_step and not early_stop:

        log_probs  = []
        values     = []
        states     = []
        actions    = []
        rewards    = []
        masks      = []
        entropy    = 0

        # set annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - num_updates/max_updates
            lr = frac * args.lr
            optimizer.param_groups[0]["lr"] = lr

        # generate num_envs trajectories with num_steps of old policy
        for _ in range(args.horizon):
            state = torch.FloatTensor(state).to(device)    # state: (num_env)x4x84x84
            with torch.no_grad():
                dist, value = model(state)

            # sample a torch action from policy pi(a|s)
            action = dist.sample()                         # action: (num_env)x1 (continuous) 8 (discrete)
            log_prob = dist.log_prob(action)               # log_prob: (num_env)x1 (continuous) 8 (discrete)
            #entropy += dist.entropy().mean()


            # interact to the envs
            next_state, reward, done, info = envs.step(action.cpu().numpy())   #send action to cpu and convert it to numpy
            
            
            #current_reward_envs = current_reward_envs + reward
            for env_index in range(args.num_env):
                if not done[env_index]: 
                    current_reward_envs[env_index] = current_reward_envs[env_index] + reward[env_index]
                else:
                    print(f"global step = {step}, episodic reward = {current_reward_envs[env_index]}")
                    
                    if args.wandb_track:
                        wandb.define_metric("Episodic_reward", step_metric="Global_step")
                        wandb.log(
                        {"Episodic_reward": current_reward_envs[env_index],
                        "  Global_step": step}
                    )
                    if current_reward_envs[env_index] >= threshold_reward: early_stop = True
                    current_reward_envs[env_index] = 0

                #if reward[env_index] > 0: print(f"global step = {step}, current reward = #{reward}, current episodic return = {current_reward_envs}")

            # reshape the log_pro to the shape (num_env)x1
            if model.action_type == 'discrete':
                log_prob = log_prob.reshape(-1,1)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))       # rewards: (num_env)x1
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))       # masks: (num_env)x1

            # reshape the action to the shape (num_env)x1
            states.append(state)
            if model.action_type == 'discrete':
                action = action.reshape(-1,1)
            actions.append(action)

            state = next_state
            step += 1 * args.num_env
            # Evaluation over 1 episodes
            '''
            if step % 5000 == 0:
                test_reward = np.mean([test_env(model=model) for _ in range(1)])
                print(f"evaluation at step {step}, episodic reward = {test_reward}")
                if test_reward > best_reward:
                    best_reward = test_reward
                    torch.save(model.state_dict(), "best_model.pt")
                test_rewards.append(test_reward)
                #plot(step, test_rewards)
                if args.wandb_track:
                    wandb.define_metric("Test_reward", step_metric="Global_step")
                    wandb.log(
                        {"Test_reward": test_reward,
                        "  Global_step": step}
                    )
                    if test_reward >= threshold_reward: early_stop = True

            '''

            next_state = torch.FloatTensor(next_state).to(device)
        with torch.no_grad():
            next_value = model.get_value(next_state)

            # compputing generalized advantage estimation
            returns = compute_gae(args.gamma, args.lam, next_value, rewards, masks, values)


        # convert (num_step)x(num_env) list to (num_step*num_env)x1 tensor for batch
        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values


        loss, actor_loss, critic_loss = ppo_update(args.clip_coef, args.vf_coef, args.entro_coef, args.num_epoch, args.size_miniBatch, states, actions, log_probs, returns, advantage)
        num_updates += 1

        if args.wandb_track:
            wandb.define_metric("Total_loss", step_metric="Global_step")
            wandb.define_metric("Actor_loss", step_metric="Global_step")
            wandb.define_metric("Critic_loss", step_metric="Global_step")
            wandb.define_metric("Learning_rate", step_metric="Global_step")
            wandb.log(
            {
              "Total_loss": loss,
              "Actor_loss": actor_loss,
              "Critic_loss": critic_loss,
              "Learning_rate": lr,
              "Global_step": step
            }
        )

    if args.wandb_track:
        wandb.finish()