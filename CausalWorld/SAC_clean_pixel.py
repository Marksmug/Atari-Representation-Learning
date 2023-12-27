import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld

from utils import GrayFrame
from utils import ResizeFrame
from utils import FrameStack
from utils import count_parameters



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="SAC",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--task-name", type=str, default="pushing")
    parser.add_argument("--skip-frame", type=int, default=3, help="control the runing frequency of the low level controller: 250Hz for skip frame of 1, 1Hz for skip frame of 250")
    parser.add_argument("--maximum-episode-length", type=int, default=600, help="the episode length of the task")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="pushing",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.95,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.001,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=1e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=1e-3,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(task_name, skip_frame, seed, maximum_episode_length, capture_video, run_name):
    def thunk():
        task = generate_task(task_generator_id=task_name,dense_reward_weights=np.array([2500, 2500, 0]),
                          fractional_reward_weight=100)
        if capture_video and idx == 0:
            env = CausalWorld(task=task, skip_frame=skip_frame, seed=seed, max_episode_length = maximum_episode_length, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = CausalWorld(task=task, skip_frame=skip_frame, seed=seed, max_episode_length = maximum_episode_length, observation_mode = 'pixel', camera_indicies = [0])
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # change: adding obs preprocessing and seed the env
        #env = GrayFrame(env, max_episode_length=maximum_episode_length)
        env = ResizeFrame(env, max_episode_length=maximum_episode_length, width=84, height=84)
        env = FrameStack(env, k=3, single_goal_obs=True)
        env.seed(seed)
        env.action_space.seed(seed)
        return env

    return thunk

# Replay buffer
class replayBuffer():
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.device = device
        self.capacity = capacity
        self.obs_buffer = np.zeros((tuple([capacity]) + obs_dim), dtype=np.float32)
        self.next_obs_buffer = np.zeros((tuple([capacity]) + obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r_buffer = np.zeros(capacity, dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.current_size = 0

    def store(self, obs, act, r, next_obs, done):
        
        # flattent the observation 
        # doesn't support parallel envs
        #obs.flatten()
        #next_obs.flatten()

        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.r_buffer[self.ptr] = r
        self.next_obs_buffer[self.ptr] = next_obs
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size=32):
        idx = np.random.randint(0, self.current_size, size=batch_size)
        # the reward is rescaled
        batch = dict(
            obs = self.obs_buffer[idx],
            act = self.act_buffer[idx],
            r = self.r_buffer[idx],
            next_obs = self.next_obs_buffer[idx],
            done = self.done_buffer[idx]
        )   
        
        return {key: torch.as_tensor(value, dtype=torch.float32).to(self.device) for key, value in batch.items()}
    

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        obs_shape = env.single_observation_space.shape
        act_dim = np.prod(env.single_action_space.shape)
        assert len(obs_shape)==3, "Input Dimension does not match"
        assert act_dim == 9, "Output Dimension does not match"

        self.CNN = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64 * 7 * 7, 64)
        )
        self.fc1 = nn.Linear(64 + act_dim, 256)
        #self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        # no hidden layer
        x = F.relu(self.CNN(x))
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        obs_shape = env.single_observation_space.shape
        act_dim = np.prod(env.single_action_space.shape)
        assert len(obs_shape)==3, "Input Dimension does not match"
        assert act_dim == 9, "Output Dimension does not match"

        self.CNN = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64 * 7 * 7, 64)
        )

        self.fc1 = nn.Linear(64 , 256)
        #self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        # no hidder layer
        x = F.relu(self.CNN(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        entropy = normal.entropy()
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, entropy


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parse_args()
    run_name = f"{args.task_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.task_name, args.skip_frame, args.seed, args.maximum_episode_length, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    count_parameters(actor)
    count_parameters(qf1)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    
    envs.single_observation_space.dtype = np.float32
    #rb = ReplayBuffer(
    #    args.buffer_size,
    #    envs.single_observation_space,
    #    envs.single_action_space,
    #    device,
    #    handle_timeout_termination=False,
    #)

    # initialize the replay buffer
    rb = replayBuffer(envs.single_observation_space.shape, envs.single_action_space.shape[0], capacity=args.buffer_size, device=device)
    start_time = time.time()
    episode_num = 0
    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    print(obs.shape)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            start = time.time()
            
                
            actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            end = time.time()
            policy_forward_time = end - start
            #print('policy network forward time per step: ', policy_forward_time)
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            print(f"episode_num={episode_num}, global_step={global_step}, Episodic_reward={infos['episode'][0]['r']}, fractional_success={infos['fractional_success'][0]}")
            episode_num += 1
            writer.add_scalar("charts/episodic_return", infos["episode"][0]["r"], global_step)
            writer.add_scalar("charts/episodic_length", infos["episode"][0]["l"], global_step)
            writer.add_scalar("charts/fractional_success", infos["fractional_success"][0], global_step)
            

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, termi in enumerate(terminations):
            if termi:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.store(obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, _= actor.get_action(data["next_obs"])
                qf1_next_target = qf1_target(data["next_obs"], next_state_actions)
                qf2_next_target = qf2_target(data["next_obs"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data["r"].flatten() + (1 - data["done"].flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data["obs"], data["act"]).view(-1)
            qf2_a_values = qf2(data["obs"], data["act"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _, entropy = actor.get_action(data["obs"])
                    qf1_pi = qf1(data["obs"], pi)
                    qf2_pi = qf2(data["obs"], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ , _= actor.get_action(data["obs"])
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                #print("entropy: ", entropy.mean().item())
            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/entropy", entropy.mean().item(), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
