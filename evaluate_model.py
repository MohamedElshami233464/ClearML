import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env


# Load the trained model
model_path = r"c:\Year 2\Y2- Block B\ClearML\models\model6.zip"
model = PPO.load(model_path)

# Initialize the environment
env = OT2Env(render=False, max_steps=1000)

# Evaluation parameters
num_episodes = 10
final_distances = []
distance_reductions = []
total_rewards = []

print("Evaluating the RL controller model...")

# Test the model
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    episode_rewards = 0
    previous_distance = np.linalg.norm(obs[:3] - obs[3:])
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate rewards
        episode_rewards += reward

        # Calculate distance reduction
        current_distance = np.linalg.norm(obs[:3] - obs[3:])
        distance_reductions.append(previous_distance - current_distance)
        previous_distance = current_distance

        if terminated or truncated:
            final_distances.append(current_distance)
            break

    total_rewards.append(episode_rewards)
    print(f"Episode {episode + 1}: Final Distance = {current_distance:.6f} m | Total Reward = {episode_rewards}")

# Calculate metrics
average_final_distance = np.mean(final_distances)
average_reward = np.mean(total_rewards)
average_distance_reduction = np.mean(distance_reductions)

# Print results
print("\nEvaluation Results:")
print(f"Average Final Distance: {average_final_distance:.6f} m")
print(f"Average Total Reward: {average_reward:.2f}")
print(f"Average Distance Reduction per Step: {average_distance_reduction:.6f} m")

# Check accuracy requirements
if average_final_distance <= 0.01:
    print("Meets accuracy requirements for 8.8 C (≤ 10 mm).")
else:
    print("Does NOT meet accuracy requirements for 8.8 C (≤ 10 mm).")

if average_final_distance <= 0.001:
    print("Meets accuracy requirements for 8.8 D (≤ 1 mm).")
else:
    print("Does NOT meet accuracy requirements for 8.8 D (≤ 1 mm).")

# Close the environment
env.close()
