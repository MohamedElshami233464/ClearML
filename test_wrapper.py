import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

env = OT2Env(render=False, max_steps=100)
obs, _ = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:
        break
env.close()
