from sim_class import Simulation

# Create the simulation object
sim = Simulation(num_agents=1, render=False)  # Adjust `num_agents` if needed

# Inspect joints for the first robot
sim.create_robots(1)  # This will trigger `log_joint_info` and print joint details