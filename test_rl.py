import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env  # Ensure this matches your environment import

# Parse arguments
parser = argparse.ArgumentParser(description="Test Trained RL Model")
parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
parser.add_argument('--episodes', type=int, default=10, help="Number of episodes to test")
args = parser.parse_args()

# Load the environment
env = OT2Env(render=True, max_steps=500)  # Enable rendering for testing

# Load the model
model = PPO.load(args.model_path)
print(f"Loaded model from {args.model_path}")

# Run the test
for episode in range(args.episodes):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs  # Handle tuple return from reset

    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if isinstance(obs, tuple):
            obs, _ = obs  # Handle tuple return from step
        total_reward += reward
        step += 1

    print(f"Episode {episode + 1} completed in {step} steps with total reward: {total_reward}")

env.close()
