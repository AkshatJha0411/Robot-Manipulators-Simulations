import pybullet as pb
import pybullet_data
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import (reset_joint_states, enable_force_torque_sensor, force_torque_sensing)


def add_box(pos, ori=(0,0,0,1), mass=1., dim=(1.,1.,1.), col=(0.5,0.5,0.5,1)):
    c_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=np.array(dim)/2.)
    v_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=np.array(dim)/2., rgbaColor=col)
    b_id = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c_id,
                              baseVisualShapeIndex=v_id,
                              basePosition=pos, baseOrientation=ori)
    return b_id


def setup_scene(asset_dir):
    pb.resetSimulation()
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setRealTimeSimulation(0)

    pb.loadURDF(os.path.join(asset_dir, 'plane', 'plane.urdf'))

    add_box(pos=np.array([0.6, 0., 0.25]),
            dim=(0.7, 1., 0.5),
            mass=0)


def spawn_manipulator(asset_dir):
    arm_id = pb.loadURDF(os.path.join(asset_dir, 'ur', 'ur5-ee.urdf'),
                         useFixedBase=True,
                         basePosition=[0,0,0.5],
                         flags=pb.URDF_USE_SELF_COLLISION)

    joint_ids = [0,1,2,3,4,5]
    ee_link = 6

    reset_joint_states(arm_id, joint_ids, [0., -1.57, 1.57, -1.57, -1.57, -1.57])

    for _ in range(100):
        pb.stepSimulation()

    return arm_id, joint_ids, ee_link


if __name__ == "__main__":
    asset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
    pb.connect(pb.GUI)

    sim_dt = 1./240
    pb.setTimeStep(sim_dt)

    setup_scene(asset_dir)
    arm_id, joint_ids, ee_link = spawn_manipulator(asset_dir)
    enable_force_torque_sensor(arm_id, ee_link)

    step_id = 0
    motion_mode = "dynamic"  # static / dynamic

    pos_ref = np.array([0.4, 0.1, 0.55])
    pos_cmd = pos_ref.copy()
    quat_ref = pb.getQuaternionFromEuler([np.pi, 0, 0])

    fz_target = 20.0
    adm_gain = 0.001

    circ_r = 0.05
    circ_w = 0.005

    force_log = []
    time_log = []
    x_log = []
    y_log = []

    print("simulation running. press ctrl+c to exit.")

    try:
        while True:
            ft = force_torque_sensing(arm_id, ee_link)
            fz_now = ft[2]

            ee_pos = np.asarray(pb.getLinkState(arm_id, ee_link)[0])

            f_err = fz_target - fz_now
            vz_cmd = adm_gain * f_err
            pos_cmd[2] -= vz_cmd * sim_dt

            if motion_mode == "static":
                pos_cmd[:2] = pos_ref[:2]
            else:
                t_scaled = step_id * sim_dt * 100
                pos_cmd[:2] = np.array([
                    pos_ref[0] - circ_r * np.sin(circ_w * t_scaled),
                    pos_ref[1] + circ_r * np.cos(circ_w * t_scaled)
                ])

            q_des = pb.calculateInverseKinematics(arm_id, ee_link, pos_cmd, quat_ref)

            pb.setJointMotorControlArray(arm_id, joint_ids,
                                         controlMode=pb.POSITION_CONTROL,
                                         targetPositions=q_des)

            force_log.append(fz_now)
            time_log.append(step_id * sim_dt)
            x_log.append(ee_pos[0])
            y_log.append(ee_pos[1])

            pb.stepSimulation()
            step_id += 1

            if step_id % 100 == 0:
                print(f"step {step_id}, fz={fz_now:.2f}, err={f_err:.2f}, z={pos_cmd[2]:.4f}")

    except KeyboardInterrupt:
        print("simulation stopped.")


    plt.figure()
    plt.plot(time_log, force_log, label="Fz")
    plt.axhline(y=fz_target, color='r', linestyle='--', label="Fz target")
    plt.xlabel("time (s)")
    plt.ylabel("force (N)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(x_log, y_log, label="path")
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()
