import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation  # Assuming this is the simulation environment provided

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action space (motor velocities for x, y, z axes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space (pipette position + goal position)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None
        self.previous_distance_to_goal = None  # Track distance changes for enhanced reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation environment
        self.sim.reset(num_agents=1)
        robot_id = self.sim.robotIds[0]

        # Get initial pipette position
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        # Set a random goal position within the working envelope
        self.goal_position = np.random.uniform(low=-0.2, high=0.2, size=(3,)).astype(np.float32)

        # Combine pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)
        self.steps = 0
        self.previous_distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        info = {}  # Additional information
        print(f"Initial Pipette Position: {pipette_position}, Goal Position: {self.goal_position}")
        return observation, info

    def step(self, action):
        # Append a placeholder action value for compatibility with the simulation
        full_action = np.append(action, [0])
        self.sim.run([full_action])

        robot_id = self.sim.robotIds[0]
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        # Compute distance to the goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        # Refined reward function
        reward = -distance_to_goal  # Base reward for minimizing distance
        if distance_to_goal < 0.001:
            reward += 100  # Large bonus for high precision
        elif distance_to_goal < 0.01:
            reward += 20
        elif distance_to_goal < 0.05:
            reward += 10

        # Reward for reducing the distance compared to the previous step
        if self.previous_distance_to_goal is not None:
            distance_reduction = self.previous_distance_to_goal - distance_to_goal
            reward += distance_reduction * 50  # Aggressive scaling for faster improvement

        # Penalty for large actions to encourage smoother movements
        action_penalty = -0.01 * np.linalg.norm(action)
        reward += action_penalty

        # Time penalty for taking too many steps
        if self.steps > 500:
            reward -= 0.1 * (self.steps - 500)

        # Update the previous distance
        self.previous_distance_to_goal = distance_to_goal

        # Check if the episode is terminated or truncated
        terminated = distance_to_goal < 0.001  # Goal condition
        truncated = self.steps >= self.max_steps  # Max steps condition

        # Combine pipette position and goal position as the observation
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)
        self.steps += 1

        info = {
            "distance_to_goal": distance_to_goal,
            "steps": self.steps,
            "terminated": terminated,
            "truncated": truncated,
        }

        print(f"Step: {self.steps}, Distance to Goal: {distance_to_goal:.6f}, "
              f"Reward: {reward:.6f}, Terminated: {terminated}, Truncated: {truncated}")

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.render:
            self.sim.render()

    def close(self):
        self.sim.close()