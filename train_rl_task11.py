# Import required packages
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from ot2_gym_wrapper import OT2Env  # Import your custom environment
from clearml import Task
import wandb
import os
import argparse

# Set the WandB API key
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"

# Initialize ClearML Task
task = Task.init(project_name="Mentor Group J/Group 2/Mohamed",
                 task_name="Task11_training")

# Set Docker image for the task
task.set_base_docker("deanis/2023y2b-rl:latest")

# Ensure the task will run remotely
task.execute_remotely(queue_name="default")

# Initialize WandB
run = wandb.init(project="task11", sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Define hyperparameters using argparse
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=8192, help="Number of steps per update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy type for PPO")
parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping range for PPO")
parser.add_argument("--value_coefficient", type=float, default=0.5, help="Value function coefficient")
parser.add_argument("--total_timesteps", type=int, default=5000000, help="Total timesteps for training")
args = parser.parse_args()

# Initialize the environment
env = DummyVecEnv([lambda: OT2Env(render=False, max_steps=500)])  # Use DummyVecEnv for compatibility

# Check environment validity
check_env(env)

# Create the PPO model
model = PPO(
    policy=args.policy,
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    vf_coef=args.value_coefficient,
    tensorboard_log=f"runs/{run.id}"
)

# Add WandB callback for tracking
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=save_path,
    verbose=2
)

# Train the model
model.learn(
    total_timesteps=args.total_timesteps,
    callback=wandb_callback,
    progress_bar=True,
    tb_log_name=f"runs/{run.id}",
    reset_num_timesteps=False
)

# Save the final trained model
model.save(f"{save_path}/{args.total_timesteps}_final_model")
wandb.save(f"{save_path}/{args.total_timesteps}_final_model")

print(f"Model training completed and saved at: {save_path}")
