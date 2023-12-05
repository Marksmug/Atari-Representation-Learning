

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

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld

from utils import replayBuffer
from utils import DtypeChange
from utils import GrayFrame
from utils import ResizeFrame
from utils import FrameStack
from utils import count_parameters


def parse_args():
    parser = argparse.ArgumentParser()

    # environment hyperparameters
    parser.add_argument("--exp-name", type=str, default="SAC_VAE", help="the name of the experiment")
    parser.add_argument("--seed-value", type=int, default=1, help =" the seed of the experiment")
    parser.add_argument("--wandb-track", type=bool, default=False, help = "the flag of tracking the experiment on WandB")
    parser.add_argument("--task-name", type=str, default="pushing")
    parser.add_argument("--skip-frame", type=int, default=3, help="control the runing frequency of the low level controller: 250Hz for skip frame of 1, 1Hz for skip frame of 250")
    parser.add_argument("--maximum-episode-length", type=int, default=600, help="the episode length of the task")
    parser.add_argument("--env-num", type=int, default=1, help="the number of environments")
    parser.add_argument("--proj-name", type=str, default="SAC", help="the project name in the wandb")
    parser.add_argument("--capture-video", type=bool, default=False, help="the flag of capturing video")
    parser.add_argument("--observe-mode", type=str, default="pixel", help="the observation mode of the env, either 'structured' or 'pixel'")

    # VAE hyperparameters
    parser.add_argument("--beta", type=float, default=1, help="beta value of VAE")
    parser.add_argument("--latent-dims", type=int, default=100, help="the latent dimensions that VAE learns")
    parser.add_argument("--VAE-lr", type=float, default=1e-4, help="the learning rate of VAE")
    parser.add_argument("--num-pre-epoches", type=int, default=100, help="the number of pretrain epoches")

    # SAC hyperparameters
    parser.add_argument("--max-step", type=int, default=10000000, help = "the maximum step of the experiment")
    parser.add_argument("--explore-step", type=int, default=2000, help="the time step that used to expolre using random policy")
    parser.add_argument("--q-lr", type=float, default=1e-4, help="the learning rate of the Q function network")
    parser.add_argument("--policy-lr", type=float, default=1e-4, help="the learning rate of the policy network")
    parser.add_argument("--buf-capacity", type=int, default=1000000,  help="the capacity of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="the batch size used to train the networks")
    parser.add_argument("--auto-alpha", type=bool, default=True, help="the flag of using auto tuning alpha")
    parser.add_argument("--alpha", type=float, default=1e-3, help="the entropy regularization coefficient")
    parser.add_argument("--gamma", type=float, default=0.95, help="the discounted factor")
    parser.add_argument("--tau", type=float, default=0.001, help="target networks updating coefficient")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency to update the policy (delayed)")
    parser.add_argument("--target-frequency", type=int, default=1, help="the frequency to update the target Q networks from the Q networks")


    args = parser.parse_args()
    return args



def make_env(task_name, skip_frame, seed, maximum_episode_length, observation_mode, capture_video, run_name):
    def thunk():
        
        task = generate_task(task_generator_id=task_name,dense_reward_weights=np.array([2500, 2500, 0]),
                      fractional_reward_weight=100)
        
        #piking task
        #task = generate_task(task_generator_id=task_name,
        #                  dense_reward_weights=np.array(
        #                      [250, 0, 125, 0, 750, 0, 0, 0.005]),
        #                  fractional_reward_weight=1,
        #                 tool_block_mass=0.02)
        if capture_video:
            env = CausalWorld(task=task, skip_frame=skip_frame, seed=seed, max_episode_length = maximum_episode_length, observation_mode = observation_mode, render_mode="rgb_array")
            # record the video every 250 episode
            env = gym.wrappers.RecordVideo(env, f"video/{run_name}", episode_trigger = lambda x: x % 250 == 0)
        else:
            env = CausalWorld(task=task, skip_frame=skip_frame, seed=seed,  max_episode_length = maximum_episode_length, observation_mode = observation_mode, camera_indicies = [0])
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = DtypeChange(env)                                                  # change the dtype to uint8
        env = GrayFrame(env, max_episode_length=maximum_episode_length)
        env = ResizeFrame(env, width=84, height=84)
        env = FrameStack(env, k=3, single_goal_obs=True)
        

        env.seed(seed)
        # sample random action in exploration phase
        env.action_space.seed(seed)
        #env.observation_space.seed(seed)
        return env
    return thunk


# weight initialization borrowed from 
# https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/sac_ae.py#L33-L45
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

# TO DO: write a function to calculate the output dim of Encoder (before the linear layer)    
class Encoder(nn.Module):
    def __init__(self, input_channel, latent_dims):
        super().__init__()

        self.enCNN = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride = 2),
            nn.Flatten(),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * 3 * 3, latent_dims)
        self.fc_logvar = nn.Linear(256 * 3 * 3, latent_dims)

        self.apply(weight_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        x = x/255.0
        x = self.enCNN(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, output_channel, latent_dims):
        super().__init__()

        self.fc_input = nn.Linear(latent_dims, 256 * 3 * 3)
        self.deCNN = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channel, 6, stride = 2),
        )

        self.apply(weight_init)

    def forward(self, z):
        z = self.fc_input(z)
        z = z.view(-1, 256, 3, 3)
        x = self.deCNN(z)
        return x
    
def loss_VAE(x_hat, x, mu, logvar, beta):
    # preprocessing image to [0, 1]
    x = x/255
    recon_loss = ((x - x_hat)**2).sum()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp()).sum()
    #print("kl_loss: ", kl_loss)
    #print("recon_loss: ", recon_loss)
    loss = recon_loss + (beta * kl_loss)
    
    return loss
def Pretrain_VAE(train_obs, encoder, decoder, optimizer, obs_layer, epoch = 10, batch_size = 256, beta = 1):
    print("Pretrainning VAE start...")
    for i in range(epoch):
        print("Epoch: ", i + 1)
        np.random.shuffle(train_obs)
        for start in range(0, len(train_obs), batch_size):
            end = start + batch_size

            batch = train_obs[start:end]          #with shape batchsize x 84 x 84 x 3

            # normalize the batch
            batch = torch.FloatTensor(batch).to(device)
          
            batch = batch.view(-1,obs_layer, batch.shape[-2],batch.shape[-1])           #with shape batchsize x 3 x 84 x 84
            

            z, mu, logvar = encoder(batch)
            batch_hat = decoder(z)
            
            optimizer.zero_grad()
            
            loss = loss_VAE(batch_hat, batch, mu, logvar, beta)
            loss.backward()
            
            optimizer.step()
        print("Current loss is ", loss.item()/batch_size)
    print("Pretraining finished")


def update_VAE(obs, encoder, decoder, VAE_optimizer, beta=1):
    z, mu, logvar = encoder(obs)
    obs_hat         = decoder(z)

    VAE_optimizer.zero_grad()

    loss = loss_VAE(obs_hat, obs, mu, logvar, beta)
    loss.backward()

    VAE_optimizer.step()
    
    if global_step %100 == 0 and args.wandb_track: 
        wandb.define_metric("losses/VAE loss", step_metric = "Global_step")
        wandb.log({"losses/VAE loss": loss.item()/len(obs),
                   "Global_step": global_step})
        



    

class softQNet(nn.Module):
    def __init__(self, env, encoder, latent_dims):
        super().__init__()

        self.encoder = encoder
        self.latent_dims = latent_dims
        # obs_shape = (x, 84, 84)

        act_dim = np.prod(env.single_action_space.shape)

        

        self.q1_net = nn.Sequential(
            nn.Linear(self.latent_dims + act_dim, 256),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(self.latent_dims + act_dim, 256),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # apply the weight init
        self.apply(weight_init)


    def forward(self, x, a, detach=True):
        # detach encoder allows to stop gradient propagation to the encoder
        x, _, _ = self.encoder(x)
        if detach:
            x.detach()
        assert x.shape[1] == self.latent_dims, "Encoder output dimention wrong"
        x = torch.cat([x,a], 1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

LOG_STD_MAX = 2
LOG_STD_MIN = -5





class Actor(nn.Module):
    def __init__(self, env, encoder, latent_dims):
        super().__init__()
        self.encoder = encoder
        self.latent_dims = latent_dims

        output_dims = np.prod(env.single_action_space.shape)


        self.actor_net = nn.Sequential(
            nn.Linear(self.latent_dims, 256),
        #    nn.ReLU(),
        #    nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.fc_mean = nn.Linear(256, output_dims)
        self.fc_log_std = nn.Linear(256, output_dims)


        # apply the weight init
        self.apply(weight_init)


    def forward(self, x, detach=True):
        # stop gradient propagation to the encoder
        x, _, _ = self.encoder(x)
        if  detach:
            x.detach()
        assert x.shape[1] == self.latent_dims, "Encoder output dimention wrong"
        x = self.actor_net(x)
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
        entropy = dist.entropy()
        # reparamatrition trick 
        a = dist.rsample()          # a = mean + std * eps, eps~N(0,1)
        log_prob = dist.log_prob(a)
        
        # squash the gaussian
        a, log_prob, mu = self.squash(a, log_prob, mu)

        return a, log_prob, mu, entropy
        

def sac_update(global_step, batch, actor, Q_net, Q_target, alpha, policy_optim, q_optim):

    policy_bp_time = 0
    q_bp_time = 0

    
    masks = (1 - batch["done"].flatten())


    # Q networks updating
    start = time.time()
    with torch.no_grad():
        netx_actions, next_log_probs, _, _ = actor.get_action(batch["next_obs"])
        target_q1, target_q2 = Q_target(batch["next_obs"], netx_actions)
        
        min_target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
        target_q = batch["r"].flatten() + masks * args.gamma * (min_target_q).view(-1)

    q1, q2 = Q_net(batch["obs"], batch["act"])
    q1_loss = F.mse_loss(q1.view(-1), target_q)
    q2_loss = F.mse_loss(q2.view(-1), target_q)
    q_loss = q1_loss + q2_loss

    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()

    end = time.time()
    q_bp_time = end - start
    #print('q network bp time per batch: ', q_bp_time)
    
    # policy network updating (delayed)
    if global_step % args.policy_frequency == 0:
        for _ in range(1):
            start = time.time()
            # update the policy multiple times to compensate for the delay
            #for _ in range(args.policy_frequency):
            actions, log_probs, _, entropy = actor.get_action(batch["obs"])
            q1_policy, q2_policy = Q_net(batch["obs"], actions)
            min_q_policy = torch.min(q1_policy, q2_policy).view(-1)
            policy_loss = ((alpha * log_probs) - min_q_policy).mean()

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()
            end = time.time()
            policy_bp_time = end - start
            #print('policy network bp time per batch: ', policy_bp_time)

            if args.auto_alpha:
                with torch.no_grad():
                    _, log_probs, _, _ = actor.get_action(batch["obs"])
                alpha_loss = (-log_alpha.exp()*(log_probs+ target_entropy)).mean()

                a_optim.zero_grad()
                alpha_loss.backward()
                a_optim.step()
                alpha = log_alpha.exp().item()

    
    # update the target Q networks through exponentially moving average
    if global_step % args.target_frequency == 0:
        for theta, target_theta in zip(Q_net.q1_net.parameters(), Q_target.q1_net.parameters()):
            target_theta.data.copy_(args.tau * theta.data + (1 - args.tau)*target_theta.data)
        for theta, target_theta in zip(Q_net.q2_net.parameters(), Q_target.q2_net.parameters()):
            target_theta.data.copy_(args.tau * theta.data + (1 - args.tau)*target_theta.data)
        for theta, target_theta in zip(Q_net.encoder.parameters(), Q_target.encoder.parameters()):
            target_theta.data.copy_(args.tau * theta.data + (1 - args.tau)*target_theta.data)
        


    # record the trainning data for visulization purpose    
    if global_step % 100 ==0:
        #print("entropy: ", entropy.mean().item())
        if args.wandb_track:
            wandb.define_metric("losses/Q1_value", step_metric="Global_step")
            wandb.define_metric("losses/Q2_value", step_metric="Global_step")
            wandb.define_metric("losses/Q1_loss", step_metric="Global_step")
            wandb.define_metric("losses/Q2_loss", step_metric="Global_step")
            wandb.define_metric("losses/Q_loss", step_metric="Global_step")
            wandb.define_metric("losses/policy_loss", step_metric="Global_step")
            wandb.define_metric("losses/entropy", step_metric="Global_step")
            wandb.log(
            {
            "losses/Q1_value": q1.mean().item(),
            "losses/Q2_value": q2.mean().item(),
            "losses/Q1_loss": q1_loss.item(),
            "losses/Q2_loss": q2_loss.item(),
            "losses/Q_loss": q_loss.item() / 2.0,
            "losses/policy_loss": policy_loss.item(),
            "losses/entropy": entropy.mean().item(),
            "Global_step": global_step
            }
            
        )
            if args.auto_alpha:
                wandb.define_metric("alpha", step_metric="Global_step")
                wandb.define_metric("losses/alpha_loss", step_metric="Global_step")
                wandb.log(
                    {
                    "alpha": alpha,
                    "losses/alpha_loss": alpha_loss.item()
                    }
                )
    
    
    

    return alpha
    


if __name__=="__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.task_name}_{args.seed_value}_{int(time.time())}"

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
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    # create environment (DOES NOT support paralle envs)
    envs = gym.vector.SyncVectorEnv([make_env(args.task_name, args.skip_frame, args.seed_value, args.maximum_episode_length, args.observe_mode, args.capture_video, run_name)]) 
    max_action = float(envs.single_action_space.high[0])


    # initialize the network:   # 1 policy network # 2 Q function networks # 2 target Q function networks
    obs_layer = envs.single_observation_space.shape[0]
    encoder = Encoder(obs_layer, args.latent_dims).to(device)
    decoder = Decoder(obs_layer, args.latent_dims).to(device)

    actor = Actor(envs, encoder, args.latent_dims).to(device)
    Q_net = softQNet(envs, encoder, args.latent_dims).to(device)

    Q_target = softQNet(envs, encoder, args.latent_dims).to(device)
    Q_target.load_state_dict(Q_net.state_dict())
 

    count_parameters(actor)
    count_parameters(Q_net)

    policy_optim = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    q_optim = optim.Adam(list(Q_net.parameters()), lr=args.q_lr)
    VAE_optimizor = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = args.VAE_lr)


    # auto entropy tuning
    if args.auto_alpha:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optim = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # initialize the replay buffer
    rep_buffer = replayBuffer(envs.single_observation_space.dtype, envs.single_observation_space.shape, envs.single_action_space.dtype, envs.single_action_space.shape[0], capacity=args.buf_capacity, device=device)

    # start training
    obs = envs.reset()

    # counting time
    policy_forward_time = 0
    env_step_time = 0


    episode_num = 0
    episode_time = time.time()
    for global_step in range(args.max_step):
        # using random policy to explore the env before the the learning start
        # otherwise take actins from policy net
        if global_step < args.explore_step:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                start = time.time()
                actions, _, _ , _= actor.get_action(torch.FloatTensor(obs).to(device))
                end = time.time()
                policy_forward_time = end - start
                #print('policy network forward time per step: ', policy_forward_time)
                
            actions = actions.detach().cpu().numpy()
        
        start = time.time()
        next_obs, rewards, dones, infos = envs.step(actions)
        end = time.time()
        env_step_time = end - start
        #print('env step time: ', env_step_time)


        if "episode" in infos:
            episode_time = time.time() - episode_time 
            print(f"episode_num={episode_num}, global_step={global_step}, Episodic_reward={infos['episode'][0]['r']}, fractional_success={infos['fractional_success'][0]},runing_time={episode_time}")
            episode_num += 1
            episode_time = time.time()
            if args.wandb_track:
                wandb.define_metric("Fractional_success", step_metric="Global step")
                wandb.define_metric("Episodic_reward", step_metric="Global step")
                wandb.define_metric("Episodic_length", step_metric="Global step")
                wandb.log(
                    {"Fractional_success": infos['fractional_success'][0],
                     "Episodic_return": infos['episode'][0]['r'],
                    "Episodic_length": infos['episode'][0]['l'],
                    "Global step": global_step}
                )



        
        #handle the terminal observation
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        # store data into replay buffer
        rep_buffer.store(obs, actions,rewards, real_next_obs, dones)

        obs = next_obs

        if global_step == args.explore_step - 1:

            # only takes explored samples to pretrain the VAE
            train_obs = rep_buffer.obs_buffer[0: args.explore_step-1].copy()
            Pretrain_VAE(train_obs, encoder, decoder, VAE_optimizor, obs_layer, args.num_pre_epoches, args.batch_size, args.beta)

        # update the networks after explorating phase
        if global_step > args.explore_step:

            batch = rep_buffer.sample(args.batch_size)
            train_obs   = batch['obs']
            # update SAC agent
            start = time.time()
            alpha = sac_update(global_step, batch, actor, Q_net, Q_target, alpha, policy_optim, q_optim)
            end = time.time()

            update_time = end - start

            # update VAE
            update_VAE(train_obs, encoder, decoder, VAE_optimizor, args.beta)
            #print(f"network updating teime per step: {update_time}")
        
        
        
        
    envs.close()
    if args.wandb_track:
        wandb.finish()


        


        
        






