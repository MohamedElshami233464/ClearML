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
        self.sim = Simulation(num_agents=1, render=False)

        # Define action space (motor velocities for x, y, z axes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space (pipette position + goal position)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.steps = 0
        self.goal_position = None

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
        self.goal_position = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)

        # Combine pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)
        self.steps = 0

        info = {}  # Additional information (can be empty)
        return observation, info






    def step(self, action):
        full_action = np.append(action, [0])
        self.sim.run([full_action])
        robot_id = self.sim.robotIds[0]
        pipette_position = np.array(self.sim.get_pipette_position(robotId=robot_id), dtype=np.float32)

        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = -distance_to_goal  # Reward must be a float
        terminated = distance_to_goal < 0.05  # Goal condition
        truncated = self.steps >= self.max_steps  # Max steps condition

        self.steps += 1
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)
        info = {}  # Additional info

        return observation, reward, terminated, truncated, info



    def render(self, mode='human'):
        if self.render:
            self.sim.render()

    def close(self):
        self.sim.close()
