import numpy as np
from ot2_gym_wrapper import OT2Env


# Initialize the environment
env = OT2Env(render=False, max_steps=1000)

print("Testing random actions in the environment...")

# Test random actions
obs, _ = env.reset()
done = False
steps = 0
total_reward = 0
initial_distance = np.linalg.norm(obs[:3] - obs[3:])  # Initial distance to goal

print(f"\nInitial State: Pipette Position = {obs[:3]}, Goal Position = {obs[3:]}")
print(f"Initial Distance to Goal: {initial_distance:.6f} m\n")

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    current_distance = np.linalg.norm(obs[:3] - obs[3:])  # Current distance to goal
    print(f"Step {steps}:")
    print(f"    Action = {action}")
    print(f"    Reward = {reward:.6f}")
    print(f"    Current Distance to Goal = {current_distance:.6f} m")
    print(f"    Terminated = {terminated}, Truncated = {truncated}\n")
    
    if terminated or truncated:
        break

print(f"\nFinal State: Pipette Position = {obs[:3]}, Goal Position = {obs[3:]}")
final_distance = np.linalg.norm(obs[:3] - obs[3:])
print(f"Final Distance to Goal: {final_distance:.6f} m")
print(f"Total Reward: {total_reward:.6f}")
print(f"Total Steps Taken: {steps}")

# Check if the goal was achieved
if final_distance < 0.001:
    print("Goal achieved within the precision requirement of 1 mm.")
elif final_distance < 0.01:
    print("Goal achieved within the less strict precision requirement of 10 mm.")
else:
    print("Goal not achieved within the required precision.")

# Close the environment
env.close()
