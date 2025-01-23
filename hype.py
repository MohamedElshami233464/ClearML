from stable_baselines3 import PPO  # Replace with your algorithm (e.g., A2C, DQN, etc.)
import json

# Load the model
model_path = "c:\Year 2\Y2- Block B\ClearML\models\model6.zip"  # Replace with the path to your saved model
model = PPO.load(model_path)

# Extract hyperparameters
hyperparameters = model.get_parameters()

# Convert hyperparameters to JSON-friendly format
def serialize_hyperparameters(hyperparams):
    serialized_params = {}
    for key, value in hyperparams.items():
        if isinstance(value, dict):
            serialized_params[key] = serialize_hyperparameters(value)  # Recursively process nested dictionaries
        elif hasattr(value, "numpy"):  # Convert tensors to lists
            serialized_params[key] = value.numpy().tolist()
        else:
            serialized_params[key] = value
    return serialized_params

serialized_hyperparameters = serialize_hyperparameters(hyperparameters)

# Save or print the hyperparameters
print("Hyperparameters:")
print(json.dumps(serialized_hyperparameters, indent=4))

# Additional useful info
print("\nAdditional Information:")
print(f"Learning Rate: {model.learning_rate}")
print(f"Batch Size: {model.batch_size}")
print(f"Gamma (Discount Factor): {model.gamma}")
print(f"Number of Steps per Update: {model.n_steps}")
print(f"Clip Range: {model.clip_range}")
print(f"Entropy Coefficient: {model.ent_coef}")
print(f"Value Function Coefficient: {model.vf_coef}")
print(f"Max Gradient Norm: {model.max_grad_norm}")
