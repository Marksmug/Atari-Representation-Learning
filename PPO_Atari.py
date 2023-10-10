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

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

import argparse
import os

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-f') #dummy argument for interactive interpreter
    parser.add_argument("--exp-name", type=str, default = "PPO", help = "experiment name")
    parser.add_argument("--env-name", type=str, default = "BreakoutNoFrameskip-v4", help = "environment name")
    parser.add_argument("--lr", type=float, default = 2.5e-4, help = "learning rate of the NN")
    parser.add_argument("--max-step", type=int, default=10000000, help = "the maximum step of the experiment")
    parser.add_argument("--wandb-track", type=bool, default=False, help="the flag of tracking the experiment data")
    parser.add_argument("--seed-value", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--norm-adv", type=bool, default=True, help="the flag of normalizing the advantage")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")

    # Agent specific arguments
    parser.add_argument("--num-env", type=int, default=1, help = "the number of parallel environments")
    parser.add_argument("--horizon", type=int, default=8, help = "the number of step in each rollout")
    parser.add_argument("--lam", type=float, default=0.95, help = "the lambda for estimating generalized advantage")
    parser.add_argument("--gamma", type=float, default=0.99, help = "the gamma for estimating generalizedd advantage")
    parser.add_argument("--num-epoch", type=int, default=4, help = "the number of epoch in each parameter update")
    parser.add_argument("--size-miniBatch", type=int, default=256, help = "the size of minibatch")
    parser.add_argument("--clip-coef", type=float, default=0.1, help = "the surrogate clipping coefficient controlingt how much the policy distribution can be updated from the old policy distribution")
    parser.add_argument("--entro-coef", type=float, default=0.01, help = "the entropy coefficient controlling the exploration rate of the agent")
    parser.add_argument("--vf-coef", type=float, default=0.5, help = "the value function coefficient controlling the weight of value function loss in the total loss")
    parser.add_argument("--clip-vloss", type=bool, default=True, help = "the flag that whether the vaule loss is clipped")
    parser.add_argument("--anneal-lr", type=bool, default=True, help="flag of using annealing learning rate")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = args.num_env * args.horizon
    args.num_miniBatch = args.batch_size // args.size_miniBatch

    return args

def make_env(vis, env_name, seed):
    def _thunk():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # revcording the agent performance
        if vis: env = gym.wrappers.RecordVideo(env, f"videos")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _thunk

# orthogonal initialization of weight, constant initialization of bias
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

#PPO network
class Actor_Critic(nn.Module):
    def __init__(self, envs, action_type = 'discrete', std = 0.0):
        super(Actor_Critic, self).__init__()

        self.action_type = action_type

        self.shared_net = nn.Sequential(
<<<<<<< HEAD
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
=======
            layer_init(nn.Conv2d(num_inputs_layer, 32, 8, stride=4)),
>>>>>>> b2f8942043e6855b072111072c9e5a024991eae7
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        self.critic = layer_init(nn.Linear(512, 1), std=1)

<<<<<<< HEAD
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
=======
        self.actor = layer_init(nn.Linear(512, num_outputs), std=0.01)

        #self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        #self.apply(init_weights)
>>>>>>> b2f8942043e6855b072111072c9e5a024991eae7


    def forward(self, x):
        # normalize the input
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
        # normalize the input
        x = self.shared_net(x/255.0)
        value = self.critic(x)
        return value


def compute_gae(gamma, lam, next_value, next_mask, rewards, masks, values):

    next_value = next_value.reshape(1,-1)
    next_mask = next_mask.reshape(1, -1)

    #concatenate next_value and next_mask to values and masks
    cat_values = torch.cat((values, next_value))
    cat_masks = torch.cat((masks, next_mask))
    advantages = torch.zeros_like(rewards).to(device)
    lastgae = 0
    for t in reversed(range(args.horizon)):
        delta = rewards[t] + gamma * cat_values[t+1] * cat_masks[t+1] - cat_values[t]
        advantages[t] = lastgae = delta + gamma * lam * cat_masks[t+1] * lastgae               # generalized Advantage
    returns = advantages + values                                                               # Advantage + value = return
    return returns, advantages
  


def ppo_update(clip_coef, vf_coef, entro_coef, ppo_epochs, mini_batch_size, batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages, batch_values):


    batch_index = np.arange(args.batch_size)
    clipfracs = []
    for i in range(ppo_epochs):
        # train on mini_batch
        j = 0
        np.random.shuffle(batch_index) 
        for start in range(0, args.batch_size, args.size_miniBatch):

            end = start + args.size_miniBatch
            mini_batch_index = batch_index[start: end]

            mini_batch_state = batch_states[mini_batch_index]
            mini_batch_action = batch_actions[mini_batch_index]
            mini_batch_old_log_probs =  batch_log_probs[mini_batch_index]
            mini_batch_return = batch_returns[mini_batch_index]
            mini_batch_advantage = batch_advantages[mini_batch_index]
            mini_batch_value = batch_values[mini_batch_index]


            new_dist, new_value = model(mini_batch_state)

            # entropy of policy distribution: encourage explooration
            entropy = new_dist.entropy()     

            # log new_pi(a | s)        
            new_log_probs = new_dist.log_prob(mini_batch_action)       
            log_ratio = new_log_probs - mini_batch_old_log_probs

            # new_pi(a | s) / old_pi(a | s)
            ratio = log_ratio.exp()   

            #check if the ratio in first update of first epoch is alwayse 0
            #if i == 0 and j == 0:
            #  print(ratio)

            # calculate approximate Kl between old pi and new pi
            with torch.no_grad():
                approx_kl = ((ratio - 1)-log_ratio).mean()
                clipfracs +=  [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        

            # normalize the advantage
            if args.norm_adv:
                mini_batch_advantage = (mini_batch_advantage - mini_batch_advantage.mean())/ (mini_batch_advantage.std() + 1e-8)

            # surrogate objective
            surr1 = -mini_batch_advantage * ratio                        
            surr2 = -mini_batch_advantage * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            actor_loss = torch.max(surr1, surr2).mean()


            new_value = new_value.view(-1)
            # clipped value loss
            if args.clip_vloss:
                unclipped_critic_loss = (mini_batch_return - new_value) ** 2
                value_clipped = mini_batch_value + torch.clamp(new_value - mini_batch_value, -clip_coef, clip_coef)
                clipped_critic_loss = (mini_batch_return - value_clipped) ** 2
                critic_loss = 0.5 * (torch.max(unclipped_critic_loss, clipped_critic_loss).mean())
            else:
                critic_loss = 0.5*((mini_batch_return - new_value) ** 2).mean()      #MSE between true return and value function

            entropy_loss = entropy.mean()
            loss = vf_coef * critic_loss + actor_loss - entro_coef * entropy_loss

            j += 1

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()


        # in case the policy update too much
        if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

    return loss, actor_loss, critic_loss, entropy_loss, approx_kl, clipfracs



if __name__ == "__main__":
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    vis = False
    # seed the experiment
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    torch.manual_seed(args.seed_value)
    torch.backends.cudnn.deterministic = True


    # create parallel enviroments with different seeds
    envs = [make_env(vis, args.env_name, seed=args.seed_value+i) for i in range(args.num_env)]
    envs = gym.vector.SyncVectorEnv(envs)





    model = Actor_Critic(envs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # initialize tensors to store the trajectories
    states = torch.zeros((args.horizon, args.num_env) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.horizon, args.num_env) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((args.horizon, args.num_env)).to(device)
    rewards = torch.zeros((args.horizon, args.num_env)).to(device)
    dones = torch.zeros((args.horizon, args.num_env)).to(device)
    values = torch.zeros((args.horizon, args.num_env)).to(device)

    # initial wandb track
    print(args.wandb_track)
    if args.wandb_track:
        wandb.init(
            project = "PPO_Atari",
            config=vars(args),
            name = f"{args.exp_name}_{args.env_name}_{int(time.time())}",
            save_code=True
        )
        wandb.define_metric("global_step")
    
    print(device)
    max_step = args.max_step
    global_step = 0


    max_updates = max_step // args.batch_size
    num_updates = 0

    # initialize the next_value and next_done for learning a long horizon
    next_done = torch.zeros(args.num_env).to(device)                  
    next_state = torch.FloatTensor(envs.reset()).to(device)
    
     #main loop
    while global_step < max_step:

        entropy    = 0

        # set annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - num_updates/max_updates
            lr = frac * args.lr
            optimizer.param_groups[0]["lr"] = lr

        # generate num_envs trajectories with num_steps of old policy
        for step in range(args.horizon):

            global_step += 1 * args.num_env

            states[step] = next_state   
            dones[step] = next_done

            with torch.no_grad():
                dist, value = model(next_state)
                # flatten the value from (num_env,1) to (num_env)
                values[step] = value.flatten()     
                
                # sample a torch action from policy pi(a|s)
                action = dist.sample()                         # action: (num_env)x1 (continuous) (num_env) (discrete)
                actions[step] = action
                log_prob = dist.log_prob(action)               # log_prob: (num_env)x1 (continuous) (num_env) (discrete)
                log_probs[step] = log_prob
        
            # interact to the envs
            next_state, reward, next_done, info = envs.step(action.cpu().numpy())   #send action to cpu and convert it to numpy

            rewards[step] = torch.FloatTensor(reward).to(device).view(-1)       
            next_state = torch.FloatTensor(next_state).to(device)
            next_done = torch.FloatTensor(next_done).to(device)                        


            if "episode" in info:
                for env in info['episode']:
                    if env is not None:
                        print(f"global_step={global_step}, episodic_return={env['r']}")
                        if args.wandb_track:
                                 wandb.define_metric("Episodic_reward", step_metric="Global_step")
                                 wandb.define_metric("Episodic_lenth", step_metric="Global_step")
                                 wandb.log(
                                    {"Episodic_reward": env['r'],
                                    "Episodic_lenth": env['l'],
                                    "Global_step": global_step}
                                 )
                        break

        # get the mask for each state
        masks = 1.0 - dones
        with torch.no_grad():
            next_value = model.get_value(next_state)
            next_mask = 1.0 - next_done
            # computing generalized advantage estimation
            returns, advantages = compute_gae(args.gamma, args.lam, next_value, next_mask, rewards, masks, values)


        # flatten the batch
        batch_states = states.reshape((-1,) + envs.single_observation_space.shape)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_log_probs = log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_values = values.reshape(-1)
        batch_returns = returns.reshape(-1)

        # update the network
        loss, actor_loss, critic_loss, entropy, approx_kl, clipfracs= ppo_update(args.clip_coef, args.vf_coef, args.entro_coef, args.num_epoch, args.size_miniBatch, batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages, batch_values)
        num_updates += 1

        if args.wandb_track:
            wandb.define_metric("Total_loss", step_metric="Global_step")
            wandb.define_metric("Actor_loss", step_metric="Global_step")
            wandb.define_metric("Critic_loss", step_metric="Global_step")
            wandb.define_metric("Learning_rate", step_metric="Global_step")
            wandb.define_metric("Entropy", step_metric="Global_step")
            wandb.define_metric("Approx_KL", step_metric="Global_step")
            wandb.define_metric("clipfracs", step_metric="Global_step")
            wandb.log(
            {
              "Total_loss": loss,
              "Actor_loss": actor_loss,
              "Critic_loss": critic_loss,
              "Learning_rate": lr,
              "Entropy": entropy,
              "Approx_KL": approx_kl,
              "Global_step": global_step,
              "clipfracs": np.mean(clipfracs)
            }
        )
    envs.close()
    if args.wandb_track:
        wandb.finish()