import pybullet as pb
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (reset_joint_states, enable_force_torque_sensor, force_torque_sensing)

def spawn_manipulator(urdf_path):
    """
    Load UR5 arm and initialize its joint configuration.
    """
    arm_id = pb.loadURDF(
        os.path.join(urdf_path, 'ur', 'ur5-ee.urdf'),
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


def add_box(pos, orn=(0,0,0,1), mass_val=1., dims=(1.,1.,1.), rgba=(0.5,0.5,0.5,1)):
    """
    Create a rigid box with given geometry and orientation.
    """
    col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=np.array(dims)/2.)
    vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=np.array(dims)/2., rgbaColor=rgba)

    box_id = pb.createMultiBody(
        baseMass=mass_val,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn
    )
    return box_id


def init_scene(asset_folder):
    """
    Initialize plane, gravity and a tilted platform.
    """
    pb.resetSimulation()
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setRealTimeSimulation(0)
    pb.setGravity(0, 0, -9.81)

    pb.loadURDF(os.path.join(asset_folder, 'plane', 'plane.urdf'))

    tilt = 0.2
    incline_quat = pb.getQuaternionFromEuler([0, tilt, 0])

    add_box(pos=np.array([0.6, 0., 0.25]), dims=(0.7, 1, 0.5), mass_val=0, orn=incline_quat)


if __name__ == "__main__":
    asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
    pb.connect(pb.GUI)

    sim_dt = 1./240
    pb.setTimeStep(sim_dt)

    init_scene(asset_root)
    arm_id, joint_ids, ee_index = spawn_manipulator(asset_root)
    enable_force_torque_sensor(arm_id, ee_index)

    sim_step = 0
    motion_mode = 'dynamic'

    base_cmd_pos = np.array([0.4, 0.1, 0.55])
    pos_cmd = base_cmd_pos.copy()
    orn_cmd = pb.getQuaternionFromEuler([np.pi, 0, 0])

    desired_fz = 20.0
    adm_gain = 0.001

    circle_r = 0.05
    angular_rate = 0.005

    fz_log = []
    t_log = []
    traj_x = []
    traj_y = []
    traj_z = []

    print("Running Part 6 (Admittance Control on Inclined Surface). Ctrl+C to stop...")

    try:
        while True:
            wrench = force_torque_sensing(arm_id, ee_index)
            fz_now = wrench[2]

            ee_xyz = np.asarray(pb.getLinkState(arm_id, ee_index)[0])

            force_err = desired_fz - fz_now
            v_z_cmd = adm_gain * force_err
            pos_cmd[2] -= v_z_cmd * sim_dt

            if motion_mode == "dynamic":
                pos_cmd[:2] = np.array([
                    base_cmd_pos[0] - circle_r * np.sin(angular_rate * sim_step * sim_dt * 100),
                    base_cmd_pos[1] + circle_r * np.cos(angular_rate * sim_step * sim_dt * 100),
                ])

            ik_sol = pb.calculateInverseKinematics(arm_id, ee_index, pos_cmd, orn_cmd)

            pb.setJointMotorControlArray(
                arm_id, joint_ids,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=ik_sol
            )

            fz_log.append(fz_now)
            t_log.append(sim_step * sim_dt)
            traj_x.append(ee_xyz[0])
            traj_y.append(ee_xyz[1])
            traj_z.append(ee_xyz[2])

            pb.stepSimulation()
            sim_step += 1

            if sim_step % 120 == 0:
                print(f"[step {sim_step}] Fz={fz_now:.2f} err={force_err:.2f} z_target={pos_cmd[2]:.4f}")

    except KeyboardInterrupt:
        print("\nSimulation ended.")


    print("Generating Plots...")

    plt.figure()
    plt.plot(t_log, fz_log, label="Measured Fz")
    plt.axhline(y=desired_fz, color='r', linestyle='--', label="Desired Force")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Part 6: Admittance Control Force Response")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(traj_x, traj_z, label="EE Path (X-Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Part 6: Adaptation to Sloped Surface")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.show()
