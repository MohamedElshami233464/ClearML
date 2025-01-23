from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from ot2_gym_wrapper import OT2Env  # Custom environment wrapper
from clearml import Task  # Import ClearML's Task

# Paths to your trained models
model_paths = [
    r"c:\Year 2\Y2- Block B\ClearML\models\model1.zip",
    r"c:\Year 2\Y2- Block B\ClearML\models\model2.zip",
]

def evaluate_model_fixed_goal(model_path, env, fixed_goal, num_episodes=10):
    model = PPO.load(model_path)  # Load the model
    accuracies, steps_taken = [], []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        env.goal_position = fixed_goal  # Override the goal position

        done, truncated = False, False
        step_count = 0

        while not done and not truncated:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

            # Calculate distance to the fixed goal
            distance_to_goal = np.linalg.norm(obs[:3] - fixed_goal)

            if done or truncated:
                accuracies.append(distance_to_goal)
                steps_taken.append(step_count)

    # Compute average and standard deviation of results
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_steps = np.mean(steps_taken)

    return avg_accuracy, std_accuracy, avg_steps

def main():
    """
    Main function to evaluate all models with a fixed goal position.
    """
    fixed_goal_position = np.array([0.1, 0.1, 0.2], dtype=np.float32)  # Define fixed goal

    # Create the environment
    env = OT2Env(render=True, max_steps=1000)

    # Dictionary to store results
    results = {}

    # Evaluate each model
    for model_path in model_paths:
        try:
            print(f"Evaluating model: {model_path} with fixed goal {fixed_goal_position}")
            avg_accuracy, std_accuracy, avg_steps = evaluate_model_fixed_goal(
                model_path, env, fixed_goal_position
            )
            results[model_path] = {
                "avg_accuracy": avg_accuracy,
                "std_accuracy": std_accuracy,
                "avg_steps": avg_steps,
            }
            print(
                f"Model: {model_path} | Avg Accuracy: {avg_accuracy:.4f} m | "
                f"Std Dev: {std_accuracy:.4f} m | Avg Steps: {avg_steps:.2f}"
            )
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")

    # Handle empty results
    if not results:
        print("No results were obtained. Please check the model paths and environment setup.")
        return

    # Identify the best model based on accuracy
    best_model = min(results, key=lambda x: results[x]["avg_accuracy"])
    print(f"\nBest Model: {best_model}")
    print(
        f"Performance: Avg Accuracy = {results[best_model]['avg_accuracy']:.4f} m, "
        f"Std Dev = {results[best_model]['std_accuracy']:.4f} m, "
        f"Avg Steps = {results[best_model]['avg_steps']:.2f}"
    )

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
