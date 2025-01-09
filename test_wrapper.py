import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

# Test the wrapper with Stable Baselines 3's check_env
env = OT2Env(render=False, max_steps=1000) 

# Check environment compatibility
check_env(env)

# Run the environment with random actions
print("Starting environment testing with random actions...")
num_episodes = 3  # Reduce the number of episodes

for episode in range(num_episodes):
    obs = env.reset()
    print(f"Episode {episode + 1} started.")
    done = False
    step = 0

    while not done:
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)

        # Log summary for every 50 steps
        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.2f}, Observation={obs}")

        if terminated or truncated:
            print(f"Episode {episode + 1} ended after {step + 1} steps.")
            break

        step += 1

env.close()