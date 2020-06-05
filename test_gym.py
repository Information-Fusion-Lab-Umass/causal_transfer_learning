import gym
env = gym.make('SourceMaze-v0')
env.reset()
env.render()
for i in range(1000):
    state, reward, done, info = env.step(env.action_space.sample())
    if done:
    	env.render()    
env.close()
