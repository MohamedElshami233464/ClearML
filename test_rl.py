"""
test_rl.py

This script tests a trained RL agent in the OT-2 environment and logs its performance.
The goal is to ensure the agent meets accuracy requirements.

Author: [Mohamed Elshami]
"""

import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env  # Ensure this is your Task 10 Gym Wrapper

# Parse arguments
parser = argparse.ArgumentParser(description="Test RL agent for OT-2 control")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
args = parser.parse_args()

# Initialize the environment
env = OT2Env(render=True, max_steps=500)  # Enable rendering for visualization

# Load the trained model
model = PPO.load(args.model_path)

# Test the model
success_count = 0
for episode in range(args.episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            print(f"Episode {episode + 1} Reward: {episode_reward}")
            if info.get("is_success", False):
                success_count += 1

accuracy = success_count / args.episodes * 100
print(f"Testing completed. Accuracy: {accuracy}%")

# Clean up
env.close()
