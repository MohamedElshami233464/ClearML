"""
train_rl_clearml_2.py

This script trains a Reinforcement Learning (RL) agent to control the OT-2 pipette tip using Stable Baselines 3 (PPO).
It runs on the ClearML server and logs metrics to the Scalars section.

Author: Mohamed Elshami
"""

import os
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.utils.env_checker import check_env
from clearml import Task, Logger
from ot2_gym_wrapper import OT2Env

# Updated hyperparameters
learning_rate = 0.00005
batch_size = 1024
n_steps = 4096
gamma = 0.99
total_timesteps = 2_000_000
n_epochs = 5
max_steps = 500  # Maximum steps per episode

# Generate a unique experiment name based on hyperparameters and timestamp
experiment_name = f"RL_OT2_lr{learning_rate}_bs{batch_size}_ns{n_steps}_g{gamma}_epochs{n_epochs}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Initialize ClearML task with a specific task name
task = Task.init(project_name="RL_OT2_Project", task_name="RL_Training_Task11_2")
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

# Initialize the raw environment
raw_env = OT2Env(render=False, max_steps=max_steps)

# Check the raw environment before wrapping it
check_env(raw_env)

# Wrap the environment after validation
env = DummyVecEnv([lambda: raw_env])

# Custom Callback for ClearML logging
class ClearMLLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ClearMLLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # Log every 100 steps
            # Log scalars to ClearML
            Logger.current_logger().report_scalar(
                "Training", "Timesteps", iteration=self.num_timesteps, value=self.num_timesteps
            )
            ep_len_mean = self.locals.get("ep_info_buffer", None)
            if ep_len_mean is not None and len(ep_len_mean) > 0:
                ep_len = ep_len_mean.mean()
                Logger.current_logger().report_scalar(
                    "rollout", "ep_len_mean", iteration=self.num_timesteps, value=ep_len
                )
            ep_rew_mean = self.locals.get("rollout_buffer", None)
            if ep_rew_mean is not None and len(ep_rew_mean.rewards) > 0:
                rew_mean = sum(ep_rew_mean.rewards) / len(ep_rew_mean.rewards)
                Logger.current_logger().report_scalar(
                    "rollout", "ep_rew_mean", iteration=self.num_timesteps, value=rew_mean
                )
        return True

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
    tensorboard_log=f"runs/ClearML/{experiment_name}",
)

# Combine ClearML logging callback
clearml_callback = ClearMLLoggingCallback()

# Start training
print("Starting training...")
model.learn(total_timesteps=total_timesteps, callback=clearml_callback, progress_bar=True)
print("Training completed.")

# Save the final trained model
model_path = f"trained_model_{experiment_name}.zip"
model.save(model_path)
task.upload_artifact(name=f"final_model_{experiment_name}", artifact_object=model_path)
print(f"Model saved at {model_path}")

# Clean up resources
env.close()
