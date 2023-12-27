import dmc2gym

env = dmc2gym.make(
                 domain_name='finger',
                 task_name='spin',
                 seed=1,
                 visualize_reward=False,
                 from_pixels=('pixel'),
                 height=84,
                 width=84,
                 frame_skip=2
             )

done = False
print(env.observation_space)
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)


# from dm_control import suite
# from dm_control.suite.wrappers import pixels
# import numpy as np
# env = suite.load('finger' , 'spin' )
# env = pixels.Wrapper(env)
# spec = env.action_spec()
# time_step = env.reset()
# total_reward = 0.0
# for _ in range(1000) :
#     action = np.random.uniform(spec.minimum , spec.maximum , spec.shape )
#     time_step = env.step( action )
#     print(action)
#     total_reward += time_step.reward

