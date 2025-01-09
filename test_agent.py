from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env

# Load the trained model
model = PPO.load("ppo_ot2")

# Create the environment
env = OT2Env(render=False, max_steps=200)

# Test the model
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode ended.")
        break

env.close()