from angorapy.common.wrappers import make_env
from angorapy.models import get_model_builder
from angorapy.agent.ppo_agent import PPOAgent

import gym_panda_reach

env = make_env('panda-reach-v0', disable_env_checker=True, autoreset=True)
model_builder = get_model_builder("simple", "ffn")
agent = PPOAgent(model_builder, env)
agent.drill(100, 10, 200)