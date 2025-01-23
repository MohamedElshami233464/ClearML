import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env
import wandb


# Set WandB API key
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"

# Initialize WandB project
run = wandb.init(
    project="RL_OT2_Control",
    name="Optimized_RL_Training",
    sync_tensorboard=True
)
save_path = os.path.join("models", run.id)
os.makedirs(save_path, exist_ok=True)

# Define hyperparameters
hyperparameters = {
    "learning_rate": 1e-5,  # Reduced for finer updates
    "batch_size": 512,  # Larger batches for stability
    "n_steps": 2048,  # Smaller n_steps for frequent updates
    "gamma": 0.995,  # Slightly higher discount factor for long-term rewards
    "total_timesteps": 5_000_000,  # Increased for extended training
    "n_epochs": 20,  # More epochs for each batch
}


# Initialize the environment
raw_env = OT2Env(render=False, max_steps=500)
env = DummyVecEnv([lambda: raw_env])


# Define the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=hyperparameters["learning_rate"],
    batch_size=hyperparameters["batch_size"],
    n_steps=hyperparameters["n_steps"],
    gamma=hyperparameters["gamma"],
    n_epochs=hyperparameters["n_epochs"],
    tensorboard_log=f"runs/{run.id}",
)



# Set up WandB callback with symbolic link fix
class CustomWandbCallback(WandbCallback):
    def _on_step(self) -> bool:
        # Call the original WandbCallback _on_step
        result = super()._on_step()
        # Ensure files are saved without symbolic links
        try:
            wandb.save(self.path, base_path=self.model_save_path, policy="now")
        except Exception as e:
            print(f"Error saving model to WandB: {e}")
        return result

wandb_callback = CustomWandbCallback(
    model_save_freq=100000,
    model_save_path=save_path,
    verbose=2
)

# Combine callbacks
callbacks = CallbackList([wandb_callback])

# Train the model
print("Starting optimized training...")
try:
    model.learn(
        total_timesteps=hyperparameters["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )
    print("Optimized training completed.")
except Exception as e:
    print(f"Error during training: {e}")

# Save the final model
final_model_path = os.path.join(save_path, "final_model.zip")
model.save(final_model_path)
wandb.save(final_model_path, base_path=save_path, policy="now")
print(f"Final model saved at {final_model_path}")

# Clean up resources
env.close()
wandb.finish()
