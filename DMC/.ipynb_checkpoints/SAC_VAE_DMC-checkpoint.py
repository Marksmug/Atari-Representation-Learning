

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

import dmc2gym

from utils import replayBuffer
from utils import DtypeChange
from utils import GrayFrame
from utils import ResizeFrame
from utils import FrameStack
from utils import count_parameters


def parse_args():
    parser = argparse.ArgumentParser()

    # environment hyperparameters
    parser.add_argument("--exp-name", type=str, default="SAC_pixel_DMC", help="the name of the experiment")
    parser.add_argument("--seed-value", type=int, default=1, help =" the seed of the experiment")
    parser.add_argument("--wandb-track", type=bool, default=False, help = "the flag of tracking the experiment on WandB")
    parser.add_argument("--domain-name", type=str, default="finger")
    parser.add_argument("--task-name", type=str, default="spin")
    parser.add_argument("--skip-frame", type=int, default=2, help="control the runing frequency of the low level controller: 250Hz for skip frame of 1, 1Hz for skip frame of 250")
    parser.add_argument("--maximum-episode-length", type=int, default=600, help="the episode length of the task")
    parser.add_argument("--env-num", type=int, default=1, help="the number of environments")
    parser.add_argument("--proj-name", type=str, default="SAC", help="the project name in the wandb")
    parser.add_argument("--capture-video", type=bool, default=False, help="the flag of capturing video")
    parser.add_argument("--observe-mode", type=str, default="pixel", help="the observation mode of the env, either 'structured' or 'pixel'")
    parser.add_argument("--save-frequency", type=int, default=10000, help="the frequency to save the models loacally")

    # VAE hyperparameters
    parser.add_argument("--beta", type=float, default=1, help="beta value of VAE")
    parser.add_argument("--latent-dims", type=int, default=100, help="the latent dimensions that VAE learns")
    parser.add_argument("--VAE-lr", type=float, default=1e-4, help="the learning rate of VAE")
    parser.add_argument("--num-pre-epoches", type=int, default=100, help="the number of pretrain epoches")
    parser.add_argument("--batch-size-vae", type=int, default=128, help="the batch size used to train the networks")

    # SAC hyperparameters
    parser.add_argument("--max-step", type=int, default=10000000, help = "the maximum step of the experiment")
    parser.add_argument("--explore-step", type=int, default=2000, help="the time step that used to expolre using random policy")
    parser.add_argument("--q-lr", type=float, default=1e-4, help="the learning rate of the Q function network")
    parser.add_argument("--policy-lr", type=float, default=1e-4, help="the learning rate of the policy network")
    parser.add_argument("--buf-capacity", type=int, default=1000000,  help="the capacity of the replay buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="the batch size used to train the networks")
    parser.add_argument("--auto-alpha", type=bool, default=True, help="the flag of using auto tuning alpha")
    parser.add_argument("--alpha", type=float, default=1e-3, help="the entropy regularization coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discounted factor")
    parser.add_argument("--tau", type=float, default=0.01, help="target networks updating coefficient")
    parser.add_argument("--policy-frequency", type=int, default=2, help="the frequency to update the policy (delayed)")
    parser.add_argument("--target-frequency", type=int, default=2, help="the frequency to update the target Q networks from the Q networks")


    args = parser.parse_args()
    return args



def make_env(domain_name, task_name, skip_frame, seed,  observation_mode, capture_video, run_name):
    def thunk():

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            visualize_reward=False,
            from_pixels=(observation_mode),
            height=84,
            width=84,
            frame_skip=skip_frame
        )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FrameStack(env, 3)
        

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
    def __init__(self, obs_shape, latent_dims):
        super().__init__()
        
        self.input_channel = obs_shape[0]
        self.latent_dims = latent_dims

        self.enCNN = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, 3, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride = 1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.inference_mode():
            self.fc_dims = self.enCNN(torch.zeros(1, *obs_shape)).shape[1]

        self.fc_mu = nn.Linear(self.fc_dims, self.latent_dims)
        self.fc_logvar = nn.Linear(self.fc_dims, self.latent_dims)
        

        self.apply(weight_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        x = self.enCNN(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, obs_shape, latent_dims, fc_dims):
        super().__init__()
        self.output_channel = obs_shape[0]
        self.fc_input = nn.Linear(latent_dims, fc_dims)
        self.deCNN = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_channel, 3, stride = 2, output_padding=1),
        )

        self.apply(weight_init)

    def forward(self, z):
        z = self.fc_input(z)
        z = z.view(-1, 32, 39, 39)
        x = self.deCNN(z)
        assert x.shape == (x.shape[0], self.output_channel, 84, 84), f'reconstructed image dimension wrong, {x.shape}'
        return x
    
def loss_VAE(x_hat, x, mu, logvar, beta):

    recon_loss = ((x - x_hat)**2).sum()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()).sum()
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
            batch = torch.FloatTensor(batch).to(device)/255.0
            
          
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

    if global_step % 100 == 0:
        print("Current loss is ", loss.item() / args.batch_size_vae)

    if args.wandb_track and global_step % 100 == 0: 
        
        wandb.define_metric("losses/VAE loss", step_metric = "Global_step")
        wandb.log({"losses/VAE loss": loss.item()/len(obs),
                   "Global_step": global_step})
        



    

class softQNet(nn.Module):
    def __init__(self, env, latent_dims):
        super().__init__()

        self.latent_dims = latent_dims
        # obs_shape = (x, 84, 84)

        self.act_dim = np.prod(env.single_action_space.shape)
        

        self.q1_net = nn.Sequential(
            nn.Linear(self.latent_dims + self.act_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(self.latent_dims + self.act_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        

        # apply the weight init
        self.apply(weight_init)


    def forward(self, x, a):
        assert x.shape[1] == self.latent_dims, "Encoder output dimention wrong"
        x = torch.cat([x,a], 1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2

LOG_STD_MAX = 2
LOG_STD_MIN = -10





class Actor(nn.Module):
    def __init__(self, env, latent_dims):
        super().__init__()
 
        self.latent_dims = latent_dims

        self.action_dims = np.prod(env.single_action_space.shape)


        self.actor_net = nn.Sequential(
            nn.Linear(self.latent_dims, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        
        self.fc_mean = nn.Linear(1024, self.action_dims)
        self.fc_log_std = nn.Linear(1024, self.action_dims)

        self.output = dict()
        # apply the weight init
        self.apply(weight_init)


    def forward(self, x):

        assert x.shape[1] == self.latent_dims, "Encoder output dimension wrong"
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
        assert mu.shape == (len(x), self.action_dims), f"mu is of shape {mu.shape}, here is the mu\n:{mu}"
        std = log_std.exp()
        assert std.shape == (len(x), self.action_dims), f"mu is of shape {std.shape}, here is the mu\n:{std}"
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

    assert batch['obs'].shape == (args.batch_size, args.latent_dims), 'the input vector dimension of obs is wrong' 
    assert batch['next_obs'].shape == (args.batch_size, args.latent_dims), 'the input vector dimension of netx_obs is wrong' 
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
    envs = gym.vector.SyncVectorEnv([make_env(args.domain_name, args.task_name, args.skip_frame, args.seed_value, args.observe_mode, args.capture_video, run_name)]) 


    # initialize the network:   # 1 policy network # 2 Q function networks # 2 target Q function networks
    obs_shape = envs.single_observation_space.shape
    encoder = Encoder(obs_shape, args.latent_dims).to(device)
    decoder = Decoder(obs_shape, args.latent_dims, encoder.fc_dims).to(device)
    #encoder.load_state_dict(torch.load('Models/encoder.pth'))
    #decoder.load_state_dict(torch.load('Models/decoder.pth'))

    actor = Actor(envs, args.latent_dims).to(device)
    Q_net = softQNet(envs, args.latent_dims).to(device)

    Q_target = softQNet(envs, args.latent_dims).to(device)
    Q_target.load_state_dict(Q_net.state_dict())
 
    count_parameters(encoder)
    count_parameters(decoder)
    count_parameters(actor)
    count_parameters(Q_net)

    if args.wandb_track:
        wandb.watch(encoder, log='gradients')
        wandb.watch(actor, log='gradients')
        wandb.watch(Q_net, log='gradients')

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

    # start train
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
                z,_,_ = encoder(torch.FloatTensor(obs/255.0).to(device))
                actions, _, _ , _= actor.get_action(z)
                end = time.time()
                policy_forward_time = end - start
                if args.wandb_track and global_step % 100 == 0:
                    wandb.define_metric("z", step_metric="Global_step")
                    wandb.log({
                        "z": wandb.Histogram(z.detach().cpu().numpy()),
                        "Global_step": global_step
                    })
                #print('policy network forward time per step: ', policy_forward_time)
                
            actions = actions.detach().cpu().numpy()
        
        start = time.time()
        next_obs, rewards, dones, infos = envs.step(actions)
        end = time.time()
        env_step_time = end - start
        #print('env step time: ', env_step_time)


        if "episode" in infos:
            episode_time = time.time() - episode_time 
            print(f"episode_num={episode_num}, global_step={global_step}, Episodic_reward={infos['episode'][0]['r']}, runing_time={episode_time}")
            episode_num += 1
            episode_time = time.time()
            if args.wandb_track:
                wandb.define_metric("Episodic_reward", step_metric="Global step")
                wandb.define_metric("Episodic_length", step_metric="Global step")
                wandb.log(
                    {
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

        if global_step == args.explore_step :

            # only takes explored samples to pretrain the VAE
            train_obs = rep_buffer.obs_buffer[0: args.explore_step-1].copy()
            Pretrain_VAE(train_obs, encoder, decoder, VAE_optimizor, obs_shape[0], args.num_pre_epoches, args.batch_size, args.beta)

        # update the networks after explorating phase
        if global_step > args.explore_step:

            batch = rep_buffer.sample(args.batch_size)
            train_obs   = rep_buffer.sample(args.batch_size_vae)['obs'].clone().detach()
            train_obs = train_obs/255.0
            assert len(train_obs) == args.batch_size_vae
            # project the image obs to latent space
            obs_vectors, _, _ = encoder(batch['obs']/255.0)
            next_obs_vectors, _, _ = encoder(batch['next_obs']/255.0)
            batch['obs'] = obs_vectors.detach()
            batch['next_obs'] = next_obs_vectors.detach()
            
            # update SAC agent
            start = time.time()
            alpha = sac_update(global_step, batch, actor, Q_net, Q_target, alpha, policy_optim, q_optim)
            end = time.time()

            update_time = end - start

            # update VAE episodicly
            # if "episode" in infos and False:
            #     print("Updating Encoder")
            #     train_obs = rep_buffer.obs_buffer[rep_buffer.current_size - args.maximum_episode_length: rep_buffer.current_size].copy()
            #     for i in range(1):
            #         np.random.shuffle(train_obs)
            #         for start in range(0, len(train_obs), args.batch_size):
                        
            #             end = start + args.batch_size

            #             batch = train_obs[start:end]          #with shape batchsize x 84 x 84 x 3

            #             # normalize the batch
            #             batch = torch.FloatTensor(batch).to(device)
            
          
            #             batch = batch.view(-1,obs_layer, batch.shape[-2],batch.shape[-1])           #with shape batchsize x 3 x 84 x 84
            #             update_VAE(batch, encoder, decoder, VAE_optimizor, args.beta)

            # update VAE every step
            update_VAE(train_obs, encoder, decoder, VAE_optimizor, args.beta)
        
        if global_step % args.save_frequency == 0:
            torch.save(actor.state_dict(), 'Models/pixel_actor.pth')
            torch.save(Q_net.state_dict(), 'Models/pixel_Q.pth')
            torch.save(encoder.state_dict(), 'Models/encoder.pth')
            torch.save(decoder.state_dict(), 'Models/decoder.pth')
            #print(f"network updating teime per step: {update_time}")
        
        
        
        
    envs.close()
    if args.wandb_track:
        wandb.finish()


        


        
        






