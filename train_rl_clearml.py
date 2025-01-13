"""
train_rl_local.py

This script trains a Reinforcement Learning (RL) agent to control the OT-2 pipette tip using Stable Baselines 3 (PPO).
It runs locally and logs progress using TensorBoard and Weights and Biases (WandB).

Author: Mohamed Elshami
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env  # Ensure this is your Task 10 Gym Wrapper
import wandb

# Set WandB API key
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"

# Initialize WandB project
run = wandb.init(project="RL_OT2_Control", name="Local_RL_Training", sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Define hyperparameters
learning_rate = 0.0001
batch_size = 512
n_steps = 8192
gamma = 0.995
total_timesteps = 5_000_000
n_epochs = 10

# Initialize the Gym Environment
env = DummyVecEnv([lambda: OT2Env(render=False, max_steps=500)])  # Adjust `max_steps` as needed
check_env(env)  # Check if the environment adheres to Gym's API

# Define the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=learning_rate,
    batch_size=batch_size,
    n_steps=n_steps,
    gamma=gamma,
    n_epochs=n_epochs,
    tensorboard_log=f"runs/{run.id}",
)

# Set up WandB callback
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=save_path,
    verbose=2
)

# Combine callbacks
callbacks = CallbackList([wandb_callback])

# Train the model
print("Starting training...")
model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
print("Training completed.")

# Save the final model
final_model_path = f"{save_path}/final_model"
model.save(final_model_path)
wandb.save(final_model_path)
print(f"Model saved at {final_model_path}")

# Clean up resources
env.close()
wandb.finish()