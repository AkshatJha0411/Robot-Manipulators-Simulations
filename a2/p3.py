import pybullet as pb
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (reset_joint_states, enable_force_torque_sensor, force_torque_sensing)


def spawn_manipulator(model_directory):
    """
    Load the UR5 robot arm with its tool and initialize joint angles.
    """
    arm_id = pb.loadURDF(
        os.path.join(model_directory, 'ur', 'ur5-ee.urdf'),
        useFixedBase=True,
        basePosition=[0, 0, 0.5],
        flags=pb.URDF_USE_SELF_COLLISION
    )

    joint_ids = [0, 1, 2, 3, 4, 5]
    ee_index = 6

    reset_joint_states(arm_id, joint_ids, [0.0, -1.57, 1.57, -1.57, -1.57, -1.57])

    for _ in range(100):
        pb.stepSimulation()

    return arm_id, joint_ids, ee_index


def add_box(base_pos, base_orn=(0, 0, 0, 1), mass_val=1.0, edges=(1., 1., 1.), rgba=(0.5, 0.5, 0.5, 1)):
    """
    Create a box rigid body for environment elements.
    """
    col_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=np.array(edges) / 2.)
    vis_shape = pb.createVisualShape(pb.GEOM_BOX, halfExtents=np.array(edges) / 2., rgbaColor=rgba)

    box_id = pb.createMultiBody(
        baseMass=mass_val,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=base_pos,
        baseOrientation=base_orn
    )
    return box_id


def init_scene(asset_directory):
    """
    Setup plane, gravity and support surface.
    """
    pb.resetSimulation()
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setRealTimeSimulation(0)
    pb.setGravity(0, 0, -9.81)

    pb.loadURDF(os.path.join(asset_directory, 'plane', 'plane.urdf'))

    # Surface positioned to match contact height
    add_box(base_pos=np.array([0.6, 0., 0.25]), edges=(0.7, 1, 0.5), mass_val=0)


if __name__ == "__main__":

    asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
    pb.connect(pb.GUI)

    sim_dt = 1. / 240
    pb.setTimeStep(sim_dt)

    init_scene(asset_root)
    arm_id, joint_ids, ee_index = spawn_manipulator(asset_root)
    enable_force_torque_sensor(arm_id, ee_index)

    sim_step = 0

    # Impedance parameters
    stiffness_z = 20000.0
    press_depth = 0.001
    table_height = 0.5

    pos_cmd = np.array([0.5, 0.0, table_height - press_depth])
    orn_cmd = pb.getQuaternionFromEuler([np.pi, 0, 0])

    desired_fz_line = 20.0

    fz_log = []
    t_log = []

    print("Running simulation (Position-Based Impedance). Ctrl+C to terminate.")

    try:
        while True:
            wrench = force_torque_sensing(arm_id, ee_index)
            fz_now = wrench[2]

            ik_solution = pb.calculateInverseKinematics(arm_id, ee_index, pos_cmd, orn_cmd)

            pb.setJointMotorControlArray(
                arm_id,
                joint_ids,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=ik_solution
            )

            fz_log.append(fz_now)
            t_log.append(sim_step * sim_dt)

            pb.stepSimulation()
            sim_step += 1

            if sim_step % 100 == 0:
                ee_xyz = np.asarray(pb.getLinkState(arm_id, ee_index)[0])
                print(f"step={sim_step}, Fz={fz_now:.2f}, target_z={pos_cmd[2]:.4f}, current_z={ee_xyz[2]:.4f}")

    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")

    print("Plotting results...")
    plt.figure()
    plt.plot(t_log, fz_log, label="Measured Fz")
    plt.axhline(y=desired_fz_line, linestyle='--', label="Reference 20N")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Static Position-Based Impedance Response")
    plt.legend()
    plt.grid(True)
    plt.show()
