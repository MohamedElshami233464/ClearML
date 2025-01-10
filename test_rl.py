"""
test_rl.py

This script tests a trained RL model for controlling the OT-2 pipette tip.
It evaluates the model's performance based on task-specific metrics.

Author: Mohamed Elshami
"""

import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env  # Task 10 Gym Wrapper


# Parse input arguments
parser = argparse.ArgumentParser(description="Test Trained RL Model for OT2")
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
args = parser.parse_args()

# Initialize the environment for testing
env = OT2Env(render=True, max_steps=500)  # Enable rendering for visual testing

# Load the trained model
model = PPO.load(args.model_path)
print(f"Loaded model from {args.model_path}")

# Initialize variables to track performance metrics
total_rewards = []
success_count = 0

# Run the test episodes
for episode in range(args.episodes):
    obs = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Get the action from the trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Take a step in the environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        # Render the environment
        env.render()

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}/{args.episodes}: Reward = {episode_reward}")

    # Check for success criteria if defined in your environment
    if info.get("is_success", False):
        success_count += 1

# Calculate and print summary metrics
avg_reward = sum(total_rewards) / len(total_rewards)
success_rate = (success_count / args.episodes) * 100
print(f"\nAverage Reward: {avg_reward}")
print(f"Success Rate: {success_rate}%")

# Clean up resources
env.close()
