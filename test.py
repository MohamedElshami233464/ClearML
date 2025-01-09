"""
test.py

This script evaluates a trained RL model on the OT2 environment.
- The model is tested over a specified number of episodes.
- Performance metrics (e.g., total rewards per episode) are logged and saved.

Author: [Mohamed Elshami]
"""

import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env

# Argument parsing for test configuration
parser = argparse.ArgumentParser(description="RL Model Testing for OT2 Environment")
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to evaluate')
args = parser.parse_args()

# Load the OT2 environment
env = OT2Env(render=True, max_steps=200)  # Enable rendering for visualization

# Load the trained model
print(f"Loading model from: {args.model_path}")
model = PPO.load(args.model_path)
print("Model loaded successfully.")

# Initialize metrics
total_rewards = []

# Run the evaluation
print("Starting evaluation...")
for episode in range(args.num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Log average performance
average_reward = sum(total_rewards) / args.num_episodes
print(f"Average Reward over {args.num_episodes} episodes: {average_reward}")

# Save results to a file
with open("results_test.txt", "w") as f:
    f.write("Episode Rewards:\n")
    for i, reward in enumerate(total_rewards, start=1):
        f.write(f"Episode {i}: {reward}\n")
    f.write(f"\nAverage Reward: {average_reward}\n")

print("Evaluation completed. Results saved to: results_test.txt")

# Clean up
env.close()