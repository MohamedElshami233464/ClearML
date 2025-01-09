import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env

# Initialize wandb
wandb.init(
    project="RL_Pipette_Control",
    name="PPO_Learning_Rate_Test",
    config={
        "learning_rate": 1e-4,  # Change this value for each run
        "total_timesteps": 50000,
        "gamma": 0.99,
    },
)

# Access wandb config
config = wandb.config

# Create the environment
env = OT2Env(render=False, max_steps=200)

# Initialize PPO agent with custom learning rate
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=config.learning_rate,
    gamma=config.gamma,  # Discount factor
)

# Train the model with WandbCallback for tracking
model.learn(total_timesteps=config.total_timesteps, callback=WandbCallback())

# Save the trained model
model.save(f"ppo_ot2_lr_{config.learning_rate}")

# Close the environment
env.close()

# Finish wandb run
wandb.finish()