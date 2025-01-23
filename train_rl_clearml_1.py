import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from gymnasium.utils.env_checker import check_env
from clearml import Task, Logger
from ot2_gym_wrapper import OT2Env

# Initialize ClearML task
task = Task.init(project_name="RL_OT2_Project_1", task_name="Task11_RL_Training")
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

# Ensure GPU is used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define hyperparameters
hyperparams = {
    "learning_rate": 0.0001,
    "batch_size": 512,
    "n_steps": 8192,
    "gamma": 0.995,
    "n_epochs": 10,
    "max_steps": 500,  # Maximum steps per episode
    "total_timesteps": 5_000_000,
}

# Initialize the raw environment
raw_env = OT2Env(render=False, max_steps=hyperparams["max_steps"])

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
    device=device,  # Ensure model uses GPU
    **hyperparams,
    tensorboard_log=f"runs/ClearML",
)

# Combine ClearML logging callback
clearml_callback = ClearMLLoggingCallback()

# Start training
print("Starting training...")
model.learn(total_timesteps=hyperparams["total_timesteps"], callback=clearml_callback, progress_bar=True)
print("Training completed.")

# Save the final trained model
model_path = "trained_model.zip"
model.save(model_path)
task.upload_artifact(name="final_model", artifact_object=model_path)
print(f"Model saved at {model_path}")

# Clean up resources
env.close()