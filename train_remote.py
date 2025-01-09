"""
train_remote_test.py

This script demonstrates a minimal setup to run a reinforcement learning task remotely using ClearML.
It uses small hyperparameters for quick testing.

Author: [Mohamed Elshami]
"""

import os
import argparse
from clearml import Task
from stable_baselines3 import PPO
import gymnasium as gym

# Step 1: Initialize ClearML Task
task = Task.init(
    project_name="Mentor Group J/Group 2/Mohamed",  # Update with your group name
    task_name="Quick Test Experiment"
)

# Set the base Docker image
task.set_base_docker("deanis/2023y2b-rl:latest")

# Set the task to execute remotely
task.execute_remotely(queue_name="default")

# Step 2: Set up hyperparameters with argparse
parser = argparse.ArgumentParser(description="Quick Test: Train PPO on Pendulum-v1")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=128, help="Number of steps per update")
parser.add_argument("--total_timesteps", type=int, default=1000, help="Total timesteps for training")  # Small value for quick testing
args = parser.parse_args()

# Step 3: Set up the environment
env = gym.make("Pendulum-v1")

# Step 4: Define and train the model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    verbose=1
)

# Train the model
print("Starting quick test training remotely...")
model.learn(total_timesteps=args.total_timesteps)
print("Quick test training complete!")

# Save the final model
model.save("./models/quick_test_model")
print("Model saved as ./models/quick_test_model")
