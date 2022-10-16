import gym
import gym_panda_reach
env = gym.make('panda-reach-v0')
env.reset()
env.reward_type = "sparse" #default is "dense"
for _ in range(10000):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample()) # take a random action
env.close()