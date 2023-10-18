import argparse
import os
import random
import time

import gym 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="SAC", help="the name of the experiment")
    parser.add_argument("--seed-value", type=int, default=1, help =" the seed of the experiment")
    parser.add_argument("--wandb-track", type=bool, default=False, help = "the flag of tracking the experiment on WandB")
    parser.add_argument("--env-name", type=str, default="Hopper-v2")
    parser.add_argument("--env-num", type=int, default=1, help="the number of environments")
    parser.add_argument("--proj-name", type=str, default="SAC", help="the project name in the wandb")
    parser.add_argument("--capture-video", type=bool, default=False, help="the flag of capturing video")

    parser.add_argument("--max-step", type=int, default=1000000, help = "the maximum step of the experiment")
    parser.add_argument("--explore-step", type=int, default=5000, help="the time step that used to expolre using random policy")
    parser.add_argument("--q-lr", type=float, default=1e-3, help="the learning rate of the Q function network")
    parser.add_argument("--policy-lr", type=float, default=3e-4, help="the learning rate of the policy network")
    parser.add_argument("--buf-capacity", type=int, default=1000000,  help="the capacity of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size used to train the networks")
    parser.add_argument("--alpha", type=float, default=0.2, help="the entropy regularization coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discounted factor")
    parser.add_argument("--tau", type=float, default=0.005, help="target networks updating coefficient")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency to update the policy (delayed)")
    parser.add_argument("--target-frequency", type=int, default=1, help="the frequency to update the target Q networks from the Q networks")

    args = parser.parse_args()
    return args



def make_env(env_name, seed, capture_video, run_name):
    def thunk():
        
        
        if capture_video:
            env = gym.make(env_name, render_mode="rgb_array")
            env.wrappers.RecordVideo(env, f"video/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

class replayBuffer():
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.capacity = capacity
        self.obs_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r_buffer = np.zeros(capacity, dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.current_size = 0

    def store(self, obs, act, r, next_obs, done):
        
        # flattent the observation 
        # doesn't support parallel envs
        obs.flatten()
        next_obs.flatten()

        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.r_buffer[self.ptr] = r
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size=32):
        idx = np.random.randint(0, self.current_size, size=batch_size)
        batch = dict(
            obs = self.obs_buffer[idx],
            act = self.act_buffer[idx],
            r = self.r_buffer[idx],
            next_obs = self.next_obs_buffer[idx],
            done = self.done_buffer[idx]
        )   
        
        return {key: torch.as_tensor(value, dtype=torch.float32).to(device) for key, value in batch.items()}



class softQNet(nn.Module):
    def __init__(self, env):
        super().__init__()

        input_dims = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)

        self.net = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self, x, a):
        x = torch.cat([x,a], 1)
        q = self.net(x)
        return q

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        input_dims = np.prod(env.single_observation_space.shape)
        output_dims = np.prod(env.single_action_space.shape)

        self.net = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(256, output_dims)
        self.fc_log_std = nn.Linear(256, output_dims)

    def forward(self, x):
        x = self.net(x)
        mu = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)   # squashes the std into a range of reasonable values for a std

        return mu, log_std
    
    def squash(self, a, log_prob, mu):
        a = torch.tanh(a)
        log_prob -= torch.log((1 - a.pow(2))+ 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = torch.tanh(mu)

        return a, log_prob, mu
    
    def get_action(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mu, std)

        # reparamatrition trick 
        a = dist.rsample()          # a = mean + std * eps, eps~N(0,1)
        log_prob = dist.log_prob(a)
        
        # squash the gaussian
        a, log_prob, mu = self.squash(a, log_prob, mu)

        return a, log_prob, mu
        

def sac_update(rep_buffer, actor, Q_net1, Q_net2, Q_target1, Q_target2, alpha):
    batch = rep_buffer.sample(args.batch_size)
    masks = (1 - batch["done"].flatten())
    # Q networks updating
    with torch.no_grad():
        netx_actions, next_log_probs, _ = actor.get_action(batch["next_obs"])
        target_q1 = Q_target1(batch["next_obs"], netx_actions)
        target_q2 = Q_target2(batch["next_obs"], netx_actions)
        min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        target_q = batch["r"].flatten() + masks * args.gamma * (min_target_q).view(-1) 

    q1 = Q_net1(batch["obs"], batch["act"]).view(-1)
    q2 = Q_net2(batch["obs"], batch["act"]).view(-1)
    q1_loss = F.mse_loss(q1, target_q)
    q2_loss = F.mse_loss(q2, target_q)
    q_loss = q1_loss + q2_loss

    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()
    
    # policy network updating (delayed)
    if global_step % args.policy_frequency == 0:
        # update the policy multiple times to compensate for the delay
        for _ in range(args.policy_frequency):
            actions, log_probs, _ = actor.get_action(batch["obs"])
            q1_policy = Q_net1(batch["obs"], actions)
            q2_policy = Q_net2(batch["obs"], actions)
            min_q_policy = torch.min(q1_policy, q2_policy)
            policy_loss = ((alpha * log_probs) - min_q_policy).mean()

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

    
    
    # record the trainning data for visulization purpose    
    if global_step % 100 ==0:
        if args.wandb_track:
            wandb.define_metric("losses/Q1_value", step_metric="Global_step")
            wandb.define_metric("losses/Q2_value", step_metric="Global_step")
            wandb.define_metric("losses/Q1_loss", step_metric="Global_step")
            wandb.define_metric("losses/Q2_loss", step_metric="Global_step")
            wandb.define_metric("losses/Q_loss", step_metric="Global_step")
            wandb.define_metric("losses/policy_loss", step_metric="Global_step")
            wandb.log(
            {
            "losses/Q1_value": q1.mean().item(),
            "losses/Q2_value": q2.mean().item(),
            "losses/Q1_loss": q1_loss.item(),
            "losses/Q2_loss": q2_loss.item(),
            "losses/Q_loss": q_loss.item() / 2.0,
            "losses/policy_lossL": policy_loss.item(),
            "Global_step": global_step
            }
        )
    


if __name__=="__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.env_name}_{args.seed_value}_{int(time.time())}"

    if args.wandb_track:
        import wandb

        wandb.init(
            project=args.proj_name,
            config=vars(args),
            name=run_name,
            save_code=True
        )

    # set the seed
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    torch.manual_seed(args.seed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create environment (DOES NOT support paralle envs)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_name, args.seed_value, args.capture_video, run_name)]) 
    max_action = float(envs.single_action_space.high[0])

    # initialize the network:
    # 1 policy network
    # 2 Q function networks
    # 2 target Q function networks
    actor = Actor(envs).to(device)
    Q_net1 = softQNet(envs).to(device)
    Q_net2 = softQNet(envs).to(device)
    Q_target1 = softQNet(envs).to(device)
    Q_target2 = softQNet(envs).to(device)
    Q_target1.load_state_dict(Q_net1.state_dict())
    Q_target2.load_state_dict(Q_net2.state_dict())

    policy_optim = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optim = optim.Adam(list(Q_net1.parameters()) + list(Q_net2.parameters()), lr=args.q_lr)

    # initialize the replay buffer
    rep_buffer = replayBuffer(envs.single_observation_space.shape[0], envs.single_action_space.shape[0], capacity=args.buf_capacity, device=device)

    # start training
    obs = envs.reset()

    episode_num = 0

    for global_step in range(args.max_step):
        # using random policy to explore the env before the the learning start
        # otherwise take actins from policy net
        if global_step < args.explore_step:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
        
        next_obs, rewards, dones, infos = envs.step(actions)


        if "episode" in infos:
            for info in infos["episode"]:
                print(f"episode_num={episode_num}, global_step={global_step}, episodic_return={info['r']}")
                episode_num += 1
                if args.wandb_track:
                    wandb.define_metric("Episodic_reward", step_metric="Global step")
                    wandb.define_metric("Episodic_length", step_metric="Global step")
                    wandb.log(
                        {"Episodic_reward": info['r'],
                        "Episodic_length": info['l'],
                        "Global step": global_step}
                    )
                break


        
        #handle the terminal observation
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["episode"][0]["terminal_observation"]

        # store data into replay buffer
        rep_buffer.store(obs, actions,rewards, real_next_obs, dones)

        obs = next_obs

        # update the networks after explorating phase
        if global_step > args.explore_step:
            sac_update(rep_buffer, actor, Q_net1, Q_net2, Q_target1, Q_target2, args.alpha)
        
        # update the target Q networks through exponentially moving average
        if global_step % args.target_frequency == 0:
            for theta, target_theta in zip(Q_net1.parameters(), Q_target1.parameters()):
                target_theta.data.copy_(args.tau * theta.data + (1 - args.tau)*target_theta.data)
            for theta, target_theta in zip(Q_net2.parameters(), Q_target2.parameters()):
                target_theta.data.copy_(args.tau * theta.data + (1 - args.tau)*target_theta.data)
        
    envs.close()
    if args.wandb_track:
        wandb.finish()


        


        
        






