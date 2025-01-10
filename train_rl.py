"""
train_rl.py

This script trains a Reinforcement Learning (RL) agent to control the OT-2 pipette tip using Stable Baselines 3 (PPO).
The goal is to achieve high accuracy in positioning within the working envelope while optimizing task performance.

Author: Mohamed Elshami
"""

import argparse
import os
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env  # Ensure this is your Task 10 Gym Wrapper
import wandb


# Parse hyperparameters
parser = argparse.ArgumentParser(description="RL Training for OT2 Environment using PPO")
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps for training')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for training')
args = parser.parse_args()


# Initialize WandB for experiment tracking
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"
run = wandb.init(project="RL_OT2_Control", name="Task11_RL_Training", sync_tensorboard=True)

# Initialize the Gym Environment
# Render should be set to False for training and True for visualization during testing
env = OT2Env(render=False, max_steps=500)  # Adjust max_steps based on task complexity

# Define the PPO model with specified hyperparameters
model = PPO(
    "MlpPolicy",  # Multi-Layer Perceptron policy for continuous control tasks
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    gamma=args.gamma,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",  # Log training metrics for visualization in TensorBoard
)

# Add a callback for WandB to track progress and save models periodically
wandb_callback = WandbCallback(
    model_save_freq=1000,  # Save model every 1000 steps
    model_save_path="models/",  # Directory to save intermediate and final models
    verbose=2,  # Print detailed logging
)

# Begin model training
print("Starting training...")
model.learn(total_timesteps=args.total_timesteps, callback=wandb_callback)
print("Training completed.")

# Save the final trained model
model_save_path = f"models/{run.id}/final_model"
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Clean up and close resources
env.close()
wandb.finish()
