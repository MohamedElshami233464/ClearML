"""
train.py

This script trains a Reinforcement Learning (RL) agent using PPO for controlling the OT2 environment.
- The training process uses the Stable-Baselines3 library.
- All hyperparameters are logged, and the best model is saved for later evaluation.
- Fixes include bypassing symlink creation issues on Windows.

Author: [Mohamed Elshami]
"""

import wandb
import argparse
import os
import shutil
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env

# Logging setup: WandB integration
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"  # Replace with your WandB API key
os.environ["WANDB_DISABLE_SYMLINK"] = "true"  # Prevent WandB from using symlinks (Windows compatibility)
run = wandb.init(project="RL_OT2_Control", name="PPO_OT2", sync_tensorboard=True)

# Argument parsing for hyperparameters
parser = argparse.ArgumentParser(description="RL Training for OT2 Environment using PPO")
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps for training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
args = parser.parse_args()

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
def save_model_no_symlink(path):
    """Custom function to save model without using symlinks."""
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, "model.zip")
    shutil.copyfile("models/temp_model.zip", model_path)


class CustomWandbCallback(WandbCallback):
    def save_model(self):
        """Override save_model to avoid symlink errors."""
        self.model.save("models/temp_model.zip")  # Save the model temporarily
        save_model_no_symlink(self.model_save_path)  # Save it to WandB directory


wandb_callback = CustomWandbCallback(
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
