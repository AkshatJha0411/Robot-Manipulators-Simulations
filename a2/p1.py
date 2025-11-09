import pybullet as p
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt  # For plotting results
from utils import (reset_joint_states, enable_force_torque_sensor, force_torque_sensing)

def load_robot(urdf_dir):
    """
    Load UR5 robot with end-effector and reset joint states.
    """
    robot = p.loadURDF(os.path.join(urdf_dir, 'ur', 'ur5-ee.urdf'),
                       useFixedBase=True, basePosition=[0, 0, 0.5],
                       flags=p.URDF_USE_SELF_COLLISION)
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

    # --- THIS IS THE FIXED LINE ---
    p.loadURDF(os.path.join(asset_dir, 'plane', 'plane.urdf')) # <-- Was 'plane.urf'
    # ---
    
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
    # Set to 'static' for Part 1
    interaction_type = 'static'
    
    # Start position: X=0.4, Y=0.1, Z=0.55 (5cm above block surface)
    fixed_target_position = np.array([0.4, 0.1, 0.55]) 
    
    dynamic_target_position = fixed_target_position.copy()
    fixed_orientation = p.getQuaternionFromEuler([np.pi, 0, 0]) # Pointing downwards
    
    # --- Controller parameters ---
    Fz_desired = 20.0 # Desired contact force in Newtons
    
    # Reduced gain to 0.001 to stop vibration.
    K_d_force = 0.001 # (Units: m/s per Newton) 
    
    # --- Data logging for plotting ---
    force_history = []
    time_history = []

    print("Starting simulation. Press Ctrl+C to stop.")

    try:
        while True:
            # Measure current force
            F = force_torque_sensing(robot, tool)
            Fz_current = F[2] # We only care about the Z-axis force

            # Get end-effector state (for reference / optional use)
            x = np.asarray(p.getLinkState(robot, tool)[0])
            
            # -------------------------------------------------
            # --- ADMITTANCE CONTROL IMPLEMENTATION ---
            
            # 1. Calculate the force error
            force_error = Fz_desired - Fz_current
            
            # 2. Apply Admittance Law (Virtual Damper)
            # Map force error to a desired velocity
            v_z_desired = K_d_force * force_error
            
            # Calculate the position change for this one time step
            delta_z = v_z_desired * time_step
            
            # 3. Update the target Z-position (with correct logic)
            # We SUBTRACT delta_z to move down
            dynamic_target_position[2] -= delta_z
            
            # -------------------------------------------------

            # This block updates the X and Y positions based on the interaction_type
            if interaction_type == 'static':
                # For static, X and Y stay fixed
                dynamic_target_position[:2] = np.array([fixed_target_position[0], fixed_target_position[1]])

            elif interaction_type == "dynamic":
                # For dynamic, X and Y move in a circle (for Part 2)
                # (Circular motion parameters are not defined here, but this is for Part 2)
                pass 

            # 4. Calculate Inverse Kinematics (IK)
            # This converts our desired end-effector position (x,y,z) into joint angles
            joint_positions = p.calculateInverseKinematics(robot, tool, 
                                                           dynamic_target_position, 
                                                           fixed_orientation)
            
            # 5. Apply joint position control
            # Send the calculated joint angles to the robot's motors
            p.setJointMotorControlArray(robot, controllable_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=joint_positions)

            # --- Log data ---
            force_history.append(Fz_current)
            time_history.append(iteration * time_step)
            
            p.stepSimulation()
            iteration += 1

            if iteration % 100 == 0:
                print(f"Step {iteration}, Fz_measured={Fz_current:.2f}, F_error={force_error:.2f}, Target_Z={dynamic_target_position[2]:.4f}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


    # --- Plotting ---
    print("Generating plot...")
    plt.figure()
    plt.plot(time_history, force_history, label="Measured Force (Fz)")
    plt.axhline(y=Fz_desired, color='r', linestyle='--', label="Desired Force (Fz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Part 1: Admittance Control (Static)")
    plt.legend()
    plt.grid(True)
    plt.show()