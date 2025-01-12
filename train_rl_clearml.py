import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from ot2_gym_wrapper import OT2Env  # Ensure this is your Task 10 Gym Wrapper
import wandb
from clearml import Task, Logger

# Initialize ClearML task
task = Task.init(project_name="RL_OT2_Project", task_name="Task11_RL_Training")
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

# Parse hyperparameters
parser = argparse.ArgumentParser(description="RL Training for OT2 Environment using PPO")
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--n_steps', type=int, default=8192, help='Number of steps per update')
parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor for rewards')
parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='Total timesteps for training')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for training')
args = parser.parse_args()

# Log hyperparameters to ClearML
task.connect(args.__dict__)

# Initialize WandB for experiment tracking
os.environ["WANDB_API_KEY"] = "53c5aad13580ec16ba2461389ae74b80dcbf8da7"
run = wandb.init(project="RL_OT2_Control", name="Task11_RL_Training", sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Initialize the Gym Environment
# Render should be set to False for training
env = DummyVecEnv([lambda: OT2Env(render=False, max_steps=500)])
check_env(env)

# Define a custom logging callback for ClearML
class ClearMLLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ClearMLLoggingCallback, self).__init__(verbose)
        self.logger = Logger.current_logger()

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # Log every 100 steps
            self.logger.report_scalar("Training", "Timesteps", iteration=self.num_timesteps, value=self.num_timesteps)
            ep_len_mean = self.locals.get("ep_len_mean", None)
            ep_rew_mean = self.locals.get("ep_rew_mean", None)
            if ep_len_mean is not None:
                self.logger.report_scalar("rollout", "ep_len_mean", self.num_timesteps, ep_len_mean)
            if ep_rew_mean is not None:
                self.logger.report_scalar("rollout", "ep_rew_mean", self.num_timesteps, ep_rew_mean)
        return True

# Create the PPO model with specified hyperparameters
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    gamma=args.gamma,
    n_epochs=args.n_epochs,
    tensorboard_log=f"runs/{run.id}",  # Log training metrics for visualization in TensorBoard
)

# Combine callbacks for WandB and ClearML logging
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=save_path,
    verbose=2
)
clearml_callback = ClearMLLoggingCallback()

callbacks = CallbackList([wandb_callback, clearml_callback])

# Start model training
print("Starting training...")
model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
print("Training completed.")

# Save the final trained model
final_model_path = f"{save_path}/final_model"
model.save(final_model_path)
wandb.save(final_model_path)
print(f"Model saved at {final_model_path}")

# Log model artifacts to ClearML
task.upload_artifact(name="final_model", artifact_object=final_model_path)

# Clean up resources
env.close()
wandb.finish()
