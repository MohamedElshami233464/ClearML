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
        self.sim = Simulation(num_agents=1)

        # Define action space (motor velocities for x, y, z axes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space (pipette position + goal position)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None

    def reset(self, seed=None, options=None):
        # Set a random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the simulation environment
        self.sim.reset(num_agents=1)

        # Log available robot IDs
        print(f"Robot IDs in simulation: {self.sim.robotIds}")

        # Dynamically retrieve the first available robot ID
        robot_id = self.sim.robotIds[0]

        # Use the dynamically retrieved robot ID
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        # Set a random goal position within the working envelope
        self.goal_position = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)

        # Combine pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Reset step count
        self.steps = 0

        # Return observation and an empty info dictionary
        return observation, {}




    def step(self, action):
        # Append a drop action to the 3D action vector (assuming no dropping in this task)
        full_action = np.append(action, [0])

        # Apply the action to the simulation
        self.sim.run([full_action])

        # Get the current pipette position
        robot_id = self.sim.robotIds[0]
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)

        # Define the reward as the negative distance to the goal
        reward = -distance_to_goal  # Encourages the agent to minimize the distance

        # Convert reward to a float to ensure compatibility
        reward = float(reward)

        # Check if the goal is reached
        goal_reached = distance_to_goal < 0.05  # Threshold for reaching the goal
        terminated = bool(goal_reached)

        # Check if the episode should be truncated due to maximum steps
        truncated = bool(self.steps >= self.max_steps)

        # Increment the step counter
        self.steps += 1

        # Combine pipette position and goal position into the observation
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Return the observation, reward, and flags
        return observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        if self.render:
            self.sim.render()  # Assuming the simulation supports rendering

    def close(self):
        self.sim.close()
