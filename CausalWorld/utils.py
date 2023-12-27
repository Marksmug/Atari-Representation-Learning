import cv2
import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sys import getsizeof
import torch


class replayBuffer():
    def __init__(self, obs_dtype, obs_dim, act_dtype, act_dim, capacity, device):
        
        self.device = device
        self.capacity = capacity
        self.obs_dtype = obs_dtype
        self.act_dtype = act_dtype
        self.obs_buffer = np.zeros((tuple([capacity]) + obs_dim), dtype=self.obs_dtype)
        self.next_obs_buffer = np.zeros((tuple([capacity]) + obs_dim), dtype=self.obs_dtype)
        self.act_buffer = np.zeros((capacity, act_dim), dtype=self.act_dtype)
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
        
        return {key: torch.FloatTensor(value).to(self.device) for key, value in batch.items()}


# adapted from https://github.com/denisyarats/pytorch_sac_ae/blob/master/utils.py
class FrameStack(gym.Wrapper):
    
    """
    stack multiple frames together 
    as done in the Nature paper and later work.
    allow for single goal observation, i.e,
    get a single frame of goal obs for each view, 
    because the consecutive frames of goal obs does not provide
    dynamic information to the agent, i.e, goal object doesn't move
    
    

    :param env: Environment to wrap
    :param k: frames that stack together
    :param single_goal_obs: flag that control how many 
     frames of goal obs to stack
    """
    def __init__(self, env, k, single_goal_obs = True):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self.single_goal_obs = single_goal_obs
        self._frames = deque([], maxlen=k)
        self.shp = env.observation_space.shape
        self.num_view = int(self.shp[0]/2)
        # default channel is 1 (for Grayscale obs)
        self.num_channel = 1                   
        
        # reshape of the obs if it's RGB and 
        # change the number of channel 
        if len(self.shp) == 4:
            self.shp = (self.shp[0] * self.shp[-1], self.shp[1], self.shp[2])
            self.num_channel = 3
            
        if self.single_goal_obs:
            shape = ((self.num_view * self.num_channel * (k + 1),) + self.shp[1:])
        else:
            shape = ((self.shp[0] * k,) + self.shp[1:])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=shape,
            dtype=env.observation_space.dtype
        )
        

    def reset(self):
        obs = self.env.reset()
        # reshape the initial obs
        obs = obs.reshape(self.shp)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
         # reshape the obs
        obs = obs.reshape(self.shp)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        
        frames_target = []
        frames_goal = []
        # extract the target obs and goal obs to seperate list 
        for i in range(self._k):
            for j in range(self.shp[0]):
                if j < self.shp[0]/2:
                    frames_target.append(self._frames[i][j])
                else:
                    frames_goal.append(self._frames[i][j])
        # merge the two target obs list and goal obs list
        # each element in the list is multiple views of the object 
        last_frames = []
        if self.single_goal_obs:
            # only merge one frame from different view of goal
            for i in range(int(len(frames_goal)/self._k)):
                last_frames.append(frames_goal[i])
        else:
            last_frames = frames_goal
        return np.concatenate((frames_target,last_frames), axis=0)
    

# adapted from stable_baselines3/common/atari_wrappers.py

class ResizeFrame(gym.ObservationWrapper):
    """
    warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        assert isinstance(env.observation_space, gym.spaces.Box), f"Expected Box space, got {env.observation_space}"
        #self._max_episode_steps = max_episode_length
        if len(env.observation_space.shape) == 4:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(env.observation_space.shape[0], self.height, self.width, env.observation_space.shape[-1]),
                dtype=env.observation_space.dtype,  # type: ignore[arg-type]
            )
        if len(env.observation_space.shape) == 3:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(env.observation_space.shape[0], self.height, self.width),
                dtype=env.observation_space.dtype,  # type: ignore[arg-type]
            )
        

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        
        new_frame = np.empty(self.observation_space.shape, dtype = self.observation_space.dtype)
        for i in range(len(frame)):
            new_frame[i] = cv2.resize(frame[i], (self.width, self.height), interpolation=cv2.INTER_AREA)
        return new_frame
    

# adapted from stable_baselines3/common/atari_wrappers.py

class GrayFrame(gym.ObservationWrapper):
    """
    Convert frames to gray scale (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, max_episode_length) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), f"Expected Box space, got {env.observation_space}"
        
        self._max_episode_steps = max_episode_length
        if len(env.observation_space.shape) == 3:
            return
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape= env.observation_space.shape[:3],
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )
        

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        
        if self.observation_space.dtype != 'uint8':
            frame = np.float32(frame)
        new_frame = np.empty(self.observation_space.shape, dtype = self.observation_space.dtype)
        for i in range(len(frame)):
            new_frame[i] = cv2.cvtColor(frame[i], cv2.COLOR_RGB2GRAY) 


        return new_frame    


class DtypeChange(gym.ObservationWrapper):
    """
    Convert data type to uint8 (default)

    :param env: Environment to wrap
    :param type: data type

    """

    def __init__(self, env: gym.Env, dtype = 'uint8') -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box), f"Expected Box space, got {env.observation_space}"
        
        if len(env.observation_space.shape) == 3:
            return
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape= env.observation_space.shape,
            dtype=dtype,  # type: ignore[arg-type]
        )
        

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        assert cv2 is not None, "OpenCV is not installed, you can do `pip install opencv-python`"
        
        frame = frame * 255
        frame = frame.astype(np.uint8)

        return frame

# counting the parameters of the network
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")