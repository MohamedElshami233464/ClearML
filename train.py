"""
train.py

This script trains a Reinforcement Learning (RL) agent using PPO for controlling the OT2 environment.
- Runs remotely using ClearML.
- Integrated with WandB for logging.
- Avoids symlink issues on Windows.

Author: [Mohamed Elshami]
"""

import wandb
import argparse
import os
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env
from clearml import Task

# ClearML task initialization
task = Task.init(project_name="RL_OT2_Training", task_name="RL_PPO_Training_Remote")

# Logging setup: WandB integration
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"  # Replace with your WandB API key
os.environ["WANDB_DISABLE_SYMLINK"] = "true"  # Prevent WandB from using symlinks
run = wandb.init(project="RL_OT2_Control", name="PPO_OT2", sync_tensorboard=True)

# Argument parsing for hyperparameters
parser = argparse.ArgumentParser(description="RL Training for OT2 Environment using PPO")
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps for training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
args = parser.parse_args()

# Execute task remotely
task.execute_remotely(queue_name="default")

# Create the OT2 Environment
env = OT2Env(render=False, max_steps=200)

# Initialize the PPO model with provided hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    gamma=args.gamma,
    tensorboard_log=f"runs/{run.id}"
)

# Set up the WandB callback for saving models and logging metrics
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

# Train the model
print("Starting training...")
model.learn(
    total_timesteps=args.total_timesteps,
    callback=wandb_callback,
    progress_bar=True,
    tb_log_name=f"runs/{run.id}",
)
print("Training completed.")

# Save the final model
final_model_path = f"models/{run.id}/final_model.zip"
model.save(final_model_path)
print(f"Model saved to: {final_model_path}")

# Clean up
env.close()
wandb.finish()
