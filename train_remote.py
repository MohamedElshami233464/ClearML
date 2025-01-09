"""
train_remote.py

This script demonstrates how to run a simple reinforcement learning task remotely using ClearML.

Author: [Mohamed Elshami]
"""

import os
import argparse
from clearml import Task
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gym

# Step 1: Initialize ClearML Task
task = Task.init(
    project_name="Mentor Group J/Group 2/Mohamed",  # Update with your group name
    task_name="Remote Test Experiment"
)

# Set the base Docker image
task.set_base_docker("deanis/2023y2b-rl:latest")

# Set the task to execute remotely
task.execute_remotely(queue_name="default")

# Step 2: Set up hyperparameters with argparse
parser = argparse.ArgumentParser(description="Train PPO on Pendulum-v1")
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per update")
parser.add_argument("--total_timesteps", type=int, default=10000, help="Total timesteps for training")
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

# Checkpoint callback to save models periodically
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./models/",
    name_prefix="rl_model"
)

# Train the model
print("Starting training remotely...")
model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
print("Training complete!")

# Save the final model
model.save("./models/final_model")
print("Model saved as ./models/final_model")
