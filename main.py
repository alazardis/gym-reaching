import gym
import gym_panda_reach

env = gym.make('panda-reach-v0', disable_env_checker=True, autoreset=True)
env.reset()
env.reward_type = "dense" #default is "dense"
for _ in range(100):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample()) # take a random action
env.close()