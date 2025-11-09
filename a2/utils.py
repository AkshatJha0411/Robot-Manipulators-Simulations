import numpy as np
import pybullet as p

def reset_joint_states(body_id, joint_ids, positions):
    """Reset joint states of the robot."""
    for i, joint_id in enumerate(joint_ids):
        p.resetJointState(body_id, jointIndex=joint_id, targetValue=positions[i])

def get_observation(robot, tool, target_position, target_orientation, Fz_desired):
    """Compute observation vector: pose error, force error, tool velocity."""
    x = np.asarray(p.getLinkState(robot, tool)[0])
    dx = np.asarray(p.getLinkState(robot, tool, computeLinkVelocity=1)[6])
    o = np.asarray(p.getLinkState(robot, tool)[1])
    do = np.asarray(p.getLinkState(robot, tool, computeLinkVelocity=1)[7])

    # tool velocity
    tool_velocity = np.concatenate((dx, do))

    # position error
    position_err = target_position - x
    orientation_err = target_orientation - o
    pose_error = np.concatenate((position_err, orientation_err))

    # force error
    F = np.asarray(p.getJointState(robot, 6)[2])   # [fx, fy, fz, mx, my, mz]
    Fz_current = F[2]
    Fz_error = Fz_current - Fz_desired
    force_error = np.array([0, 0, Fz_error, 0, 0, 0])

    return np.concatenate((pose_error, force_error, tool_velocity)).ravel(), Fz_current, Fz_error

def get_jacobian(robot, tool, q):
    if isinstance(q, np.ndarray):
        q = q.ravel().tolist()
    local_position = p.getLinkState(robot, tool)[2]
    dq = [0]*len(q)
    lin_jac, ang_jac = p.calculateJacobian(robot, tool, localPosition=local_position,
                                           objPositions=list(q), objVelocities=dq, objAccelerations=dq)
    return np.vstack((lin_jac, ang_jac))

def get_damped_least_squares_inverse(jacobian, damping_factor=0.01):
    J, k = jacobian, damping_factor
    return J.T.dot(np.linalg.inv(J.dot(J.T) + k**2 * np.identity(J.shape[0])))

def quaternion_error(quat_des, quat_cur):
    diff = quat_cur[-1] * quat_des[:3] - quat_des[-1] * quat_cur[:3] \
           - skew_matrix(quat_des[:3]).dot(quat_cur[:3])
    return diff

def skew_matrix(vector):
    x, y, z = np.array(vector).flatten()
    return np.array([[0., -z, y],
                     [z, 0., -x],
                     [-y, x, 0.]])

def map_action_values(output_lower_limit, output_upper_limit, action_value,
                      input_lower_limit=-1.0, input_upper_limit=1.0):
    return output_lower_limit + ((output_upper_limit - output_lower_limit) /
                                (input_upper_limit - input_lower_limit)) * (action_value - input_lower_limit)

def enable_force_torque_sensor(body_id, joint_ids):
    if isinstance(joint_ids, int):
        p.enableJointForceTorqueSensor(body_id, joint_ids, 1)
    else:
        for joint_id in joint_ids:
            p.enableJointForceTorqueSensor(body_id, joint_id, 1)

def force_torque_sensing(body_id, joint_ids):
    if isinstance(joint_ids, int):
        return np.asarray(p.getJointState(body_id, joint_ids)[2])
    return np.asarray([state[2] for state in p.getJointStates(body_id, joint_ids)])