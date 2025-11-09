import pybullet as p
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (reset_joint_states, enable_force_torque_sensor, force_torque_sensing)

# --- (load_robot, load_box, create_world functions are the same) ---
def load_robot(urdf_dir):
    """
    Load UR5 robot with end-effector and reset joint states.
    """
    robot = p.loadURDF(os.path.join(urdf_dir, 'ur', 'ur5-ee.urdf'),
                       useFixedBase=True, basePosition=[0, 0, 0.5],
                       flags=p.URDF_USE_SELF_COLLISION) # <-- THIS IS REQUIRED FOR COLLISIONS
                       
    control_joint_indices = [0, 1, 2, 3, 4, 5]
    tool_index = 6
    reset_joint_states(robot, control_joint_indices, [0.0, -1.57, 1.57, -1.57, -1.57, -1.57])
    for _ in range(100):
        p.stepSimulation()
    return robot, control_joint_indices, tool_index

def load_box(position, orientation=(0,0,0,1), mass=1., dimensions=(1.,1.,1.), color=(0.5,0.5,0.5,1)):
    """
    Utility to load a box with collision and visual shape.
    """
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array(dimensions)/2.)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=np.array(dimensions)/2., rgbaColor=color)
    box = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape,
                             baseVisualShapeIndex=visual_shape, basePosition=position,
                             baseOrientation=(0,0,0,1)) 
    return box

def create_world(asset_dir):
    """
    Setup the simulation environment with plane, table, and a visual marker.
    """
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.81) # Ensure gravity is on

    p.loadURDF(os.path.join(asset_dir, 'plane', 'plane.urdf'))
    
    # This box surface is at Z=0.5 (base at 0.25, dimensions 0.5)
    load_box(position=np.array([0.6, 0., 0.25]), dimensions=(0.7, 1, 0.5), mass=0)

if __name__ == "__main__":
    asset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
    p.connect(p.GUI)
    time_step = 1./240
    p.setTimeStep(time_step)

    # Create environment and robot
    create_world(asset_dir)
    robot, controllable_joints, tool = load_robot(asset_dir)
    enable_force_torque_sensor(robot, tool)

    # --- Simulation variables ---
    iteration = 0
    
    # --- IMPEDANCE CONTROL (POSITION-BASED) ---
    K_z = 20000.0 # Virtual stiffness
    
    # --- Desired Position (base) ---
    penetration = 0.001 # 1mm
    surface_z = 0.5
    
    # --- *** FIX 1: Move center to a safer spot *** ---
    base_target_position = np.array([0.45, 0.0, surface_z - penetration]) 
    target_position = base_target_position.copy() # This one will be updated
    target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
    
    # --- *** FIX 2: Use a smaller radius *** ---
    interaction_type = 'dynamic'
    r = 0.02  # Radius of the circle (2 cm)
    w = 0.005 # Angular velocity (scaled)
    
    # Reference force for the plot
    Fz_desired_plot_value = 20.0
    
    # --- Data logging for plotting ---
    force_history = []
    time_history = []
    x_history = []
    y_history = []

    print("Starting simulation for Part 4 (Impedance Dynamic). Press Ctrl+C to stop.")

    try:
        while True:
            # Measure current force (for plotting)
            F = force_torque_sensing(robot, tool)
            Fz_current = F[2]
            
            # --- Update Target Position ---
            if interaction_type == "dynamic":
                target_position[:2] = np.array([
                    base_target_position[0] - r * np.sin(w * iteration * time_step * 100), 
                    base_target_position[1] + r * np.cos(w * iteration * time_step * 100),
                ])
            
            # --- We re-calculate the IK solution every step. ---
            joint_positions = p.calculateInverseKinematics(robot, tool, 
                                                           target_position, 
                                                           target_orientation)

            # --- IMPEDANCE CONTROL IMPLEMENTATION ---
            p.setJointMotorControlArray(robot, controllable_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=joint_positions)
            
            # --- Log data ---
            force_history.append(Fz_current)
            time_history.append(iteration * time_step)
            # Get the actual X,Y position for the trajectory plot
            x_actual = np.asarray(p.getLinkState(robot, tool)[0])
            x_history.append(x_actual[0])
            y_history.append(x_actual[1])

            
            p.stepSimulation()
            iteration += 1

            if iteration % 100 == 0:
                print(f"Step {iteration}, Fz_measured={Fz_current:.2f}, Target_Z={target_position[2]:.4f}, Actual_Z={x_actual[2]:.4f}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


    # --- Plotting ---
    print("Generating plot...")
    
    # Plot 1: Force vs. Time
    plt.figure()
    plt.plot(time_history, force_history, label="Measured Force (Fz)")
    plt.axhline(y=Fz_desired_plot_value, color='r', linestyle='--', label="Target Force (20N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Part 4: Impedance Control (Dynamic)") # <-- Updated title
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Trajectory (X vs Y)
    plt.figure()
    plt.plot(x_history, y_history, label="End-Effector Path")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Part 4: End-Effector Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Make X and Y scales the same
    
    plt.show() # Show both plots