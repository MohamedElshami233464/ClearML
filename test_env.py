from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

# Initialize the environment
env = OT2Env(render=False, max_steps=500)

# Check if the environment adheres to Gymnasium's API
check_env(env)

print("Environment check passed!")
