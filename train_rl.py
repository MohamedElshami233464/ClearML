"""
train_rl.py

This script trains an RL agent to control the OT-2 pipette tip using Stable Baselines 3 (PPO).
The goal is to achieve high accuracy in positioning within the working envelope.

Author: [Mohamed Elshami]
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
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for training')  # Add this line
args = parser.parse_args()


# Initialize WandB for tracking
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"
run = wandb.init(project="RL_OT2_Control", name="Task11_RL_Training", sync_tensorboard=True)

# Initialize the Gym Environment
env = OT2Env(render=False, max_steps=500)  # Adjust max_steps based on your environment

# Define the PPO model with hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    gamma=args.gamma,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",
)

# Add WandB callback
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path="models/",
    verbose=2,
)

# Training the model
print("Starting training...")
model.learn(total_timesteps=100000, callback=wandb_callback)

print("Training completed.")

# Save the final model
model.save(f"models/{run.id}/final_model")
print(f"Model saved at models/{run.id}/final_model")

# Clean up
env.close()
wandb.finish()
