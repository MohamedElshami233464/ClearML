import time
from pid_controller import PIDController
from sim_class import Simulation  # Assuming your simulation class from Task 11

# Initialize simulation
sim = Simulation(num_agents=1, render=True)  # Enable rendering for visual confirmation

# PID gains (start with rough guesses and tune later)
pid_x = PIDController(kp=1.0, ki=0.1, kd=0.01, output_limits=(-1, 1))
pid_y = PIDController(kp=1.0, ki=0.1, kd=0.01, output_limits=(-1, 1))
pid_z = PIDController(kp=1.0, ki=0.1, kd=0.01, output_limits=(-1, 1))

# Set the target position (within the working envelope)
target_position = [0.2, -0.3, 0.1]  # Example: X, Y, Z in meters
pid_x.setpoint, pid_y.setpoint, pid_z.setpoint = target_position

# Run the PID control loop
print(f"Moving pipette to target position: {target_position}")
for _ in range(1000):  # 1000 iterations or until convergence
    # Get the current pipette position
    pipette_position = sim.get_pipette_position(robotId=sim.robotIds[0])
    current_x, current_y, current_z = pipette_position

    # Compute control signals
    control_x = pid_x.compute(current_x)
    control_y = pid_y.compute(current_y)
    control_z = pid_z.compute(current_z)

    # Apply actions to the simulation
    actions = [[control_x, control_y, control_z, 0]]  # Last value is drop action, set to 0
    sim.run(actions)

    # Check accuracy
    error_x = abs(pid_x.setpoint - current_x)
    error_y = abs(pid_y.setpoint - current_y)
    error_z = abs(pid_z.setpoint - current_z)
    if error_x < 0.001 and error_y < 0.001 and error_z < 0.001:
        print(f"Target position reached with accuracy of 1 mm: {pipette_position}")
        break

    # Slow down the loop for debugging
    time.sleep(0.01)

# Cleanup
sim.close()
