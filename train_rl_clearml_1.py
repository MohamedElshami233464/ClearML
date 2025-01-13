"""
train_rl_local.py

This script trains a Reinforcement Learning (RL) agent to control the OT-2 pipette tip using Stable Baselines 3 (PPO).
It runs locally and logs progress using TensorBoard and Weights and Biases (WandB).

Author: Mohamed Elshami
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from gymnasium.utils.env_checker import check_env
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env  # Ensure this is your Task 10 Gym Wrapper
import wandb
import time

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
max_steps = 500  # Maximum steps per episode

# Initialize the raw environment
raw_env = OT2Env(render=False, max_steps=max_steps)

# Check the raw environment before wrapping it
check_env(raw_env)

# Wrap the environment after validation
env = DummyVecEnv([lambda: raw_env])


# Custom Logging Callback for WandB
class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log every 100 steps
        if self.n_calls % 100 == 0:
            wandb.log({
                "rollout/ep_len_mean": self.locals.get("ep_len_mean", None),
                "rollout/ep_rew_mean": self.locals.get("ep_rew_mean", None),
                "time/total_timesteps": self.num_timesteps,
                "time/elapsed_time": time.time() - self.start_time,
            })
        return True

    def _on_training_start(self):
        self.start_time = time.time()


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
callbacks = CallbackList([wandb_callback, CustomLoggingCallback()])

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
