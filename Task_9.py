from sim_class import Simulation
import pybullet as p

def main():
    # Task 1: Basic Simulation Initialization
    print("Starting basic simulation...")
    sim = Simulation(num_agents=1, render=False)  # Use DIRECT mode for initial setup

    # Example action: Move the robot
    velocity_x, velocity_y, velocity_z = 0.1, 0.1, 0.1
    drop_command = 0
    actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

    # Run the simulation and get the state
    state = sim.run(actions, num_steps=100)
    print("Initial State:", state)

    # Disconnect from this simulation session
    p.disconnect()

    # Task 2: Working Envelope Determination
    print("Starting working envelope determination...")
    sim = Simulation(num_agents=1, render=True)  # Enable GUI for visualization

    # Define corners of the working envelope
    corners = []

    # Move to each corner and log coordinates
    corner_positions = [
        (0.5, 0.5, 0.5),   # Corner 1
        (0.5, 0.5, -0.5),  # Corner 2
        (0.5, -0.5, 0.5),  # Corner 3
        (0.5, -0.5, -0.5), # Corner 4
        (-0.5, 0.5, 0.5),  # Corner 5
        (-0.5, 0.5, -0.5), # Corner 6
        (-0.5, -0.5, 0.5), # Corner 7
        (-0.5, -0.5, -0.5) # Corner 8
    ]

    for i, (vx, vy, vz) in enumerate(corner_positions, start=1):
        actions = [[vx, vy, vz, 0]]
        state = sim.run(actions, num_steps=200)
        position = state['robotId_1']['joint_states']['joint_0']['position']
        corners.append(position)
        print(f"Corner {i} Position: {position}")

    # Print all recorded corners
    print("\nWorking Envelope Corners:")
    for i, corner in enumerate(corners, start=1):
        print(f"Corner {i}: {corner}")

    # Disconnect from this simulation session
    p.disconnect()

if __name__ == "__main__":
    main()
