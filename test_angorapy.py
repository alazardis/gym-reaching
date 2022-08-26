import time
import gym
import gym_panda_reach
env = gym.make('panda-reach-v0')
env.reset()
env.reward_type = "sparse" #default is "dense"
for _ in range(100):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample()) # take a random action
    time.sleep(100)
env.close()