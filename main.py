from angorapy.common.wrappers import make_env
from angorapy.models import get_model_builder
from angorapy.agent.ppo_agent import PPOAgent
from datetime import datetime
from gym_panda_reach.envs import panda_env

import gym_panda_reach
import pandas as pd

env = make_env('panda-reach-v0', disable_env_checker=True, autoreset=True)
model_builder = get_model_builder("simple", "ffn")
agent = PPOAgent(model_builder, env)
agent.drill(100, 10, 200)

now = datetime.now()
current_time = now.strftime("%H:%M")

data = {'Time Elapsed': [current_time],
        'Grasp Quality': [panda_env.PandaEnv.compute_reward().grasp_quality]}

df = pd.DataFrame(data)

print(df)



