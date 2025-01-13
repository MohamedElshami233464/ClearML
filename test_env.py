from gymnasium.utils.env_checker import check_env
from ot2_gym_wrapper import OT2Env

env = OT2Env(render=False, max_steps=500)
check_env(env)
print("Environment check passed!")
