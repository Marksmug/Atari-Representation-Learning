from dm_control import suite
env = suite.load('cartpole', 'swingup')

print(env.action_spec())