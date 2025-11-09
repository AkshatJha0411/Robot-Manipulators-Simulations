import pybullet as p
import pybullet_data
import time
import numpy as np
import random
from utils import *
import argparse
import math
import os

# -------------------------------
# Global paths + constants
# -------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(current_dir, "assets")
ur5_joint_indices = list(range(0,6))
END_EFFECTOR_LINK_INDEX = 7

# -------------------------------
# Node class (from A1)
# -------------------------------
class Node:
    def __init__(self, config, parent=None, cost=0.0):
        self.config = np.array(config)
        self.parent = parent
        self.cost = cost
        self.children = []

# -------------------------------
# Path reconstruction (from A1)
# -------------------------------
def reconstruct_path(node):
    path = []
    current = node
    while current is not None:
        path.append(list(current.config))
        current = current.parent
    path.reverse()
    return path

def reconstruct_birrt_path(node_a, node_b):
    path_a = reconstruct_path(node_a)
    path_b = []
    current = node_b
    while current is not None:
        path_b.append(list(current.config))
        current = current.parent
    path_b.reverse()
    if np.allclose(path_a[-1], path_b[0]):
      return path_a + path_b[1:]
    else:
      return path_a + path_b

# -------------------------------
# VISUALIZATION (complete)
# -------------------------------
def execute_path_with_trail(body_id, link_index, joints, path, step_sleep, color_rgb, linewidth=2.0, lifetime=0):
    if not path:
        print("Warning: execute_path_with_trail received an empty path.")
        return
    last_pos = None
    for config in path:
        set_joint_positions(body_id, joints, config)
        link_state = p.getLinkState(body_id, link_index)
        current_pos = link_state[0]
        if last_pos is not None:
            p.addUserDebugLine(last_pos, current_pos, color_rgb, lineWidth=linewidth, lifeTime=lifetime)
        last_pos = current_pos
        p.stepSimulation()
        time.sleep(step_sleep)

def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id

# -------------------------------
# CLI args
# -------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--rrt', action='store_true', default=False)
    parser.add_argument('--rrt_task', action='store_true', default=False)
    parser.add_argument('--birrt_task', action='store_true', default=False)
    args = parser.parse_args()
    return args

# -------------------------------
# Planners (A1 logic copied clean)
# -------------------------------
def rrt_planner(start_config, goal_config, collision_fn, max_iterations=5000, step_size=0.5, goal_bias=0.1):
    start_node = Node(start_config)
    tree = [start_node]
    goal_config_arr = np.array(goal_config)
    for _ in range(max_iterations):
        if random.random() < goal_bias:
            q_rand = goal_config_arr
        else:
            q_rand = np.random.uniform(-np.pi, np.pi, 6)

        nearest_node = min(tree, key=lambda node: np.linalg.norm(node.config - q_rand))
        direction = q_rand - nearest_node.config
        dist = np.linalg.norm(direction)

        q_new_config = nearest_node.config + (direction/dist)*min(step_size, dist) if dist > 0 else nearest_node.config

        if not collision_fn(q_new_config):
            new_node = Node(q_new_config, parent=nearest_node)
            tree.append(new_node)

            if np.linalg.norm(new_node.config - goal_config_arr) < 0.5:
                print("Goal reached!")
                return reconstruct_path(new_node)

    print("Failed to find path.")
    return []

def rrt():
    results = []
    goals = [start_conf, intermediate_conf, end_conf]
    current_start = init_conf
    for i, goal in enumerate(goals):
        print(f"Planning Path {i+1} (RRT Joint)...")
        path = rrt_planner(current_start, goal, collision_fn)
        results.append(path)
        current_start = path[-1] if path else goal
    return results

def rrt_task_space_planner(start_config, goal_pos, collision_fn, max_iterations=5000, step_size=0.5, goal_bias=0.1):
    goal_quat = p.getQuaternionFromEuler([0,0,0])
    goal_config_arr = np.array(ik_conf(goal_pos, goal_quat))
    start_node = Node(start_config)
    tree = [start_node]
    for _ in range(max_iterations):
        if random.random() < goal_bias:
            q_rand = goal_config_arr
        else:
            x = random.uniform(0.1, 0.8)
            y = random.uniform(-0.5, 0.5)
            z = random.uniform(0.1, 0.7)
            q_rand = np.array(ik_conf([x,y,z], goal_quat))

        nearest_node = min(tree, key=lambda node: np.linalg.norm(node.config - q_rand))
        direction = q_rand - nearest_node.config
        dist = np.linalg.norm(direction)
        q_new = nearest_node.config + (direction/dist)*min(step_size, dist) if dist > 0 else nearest_node.config

        if not collision_fn(q_new):
            new_node = Node(q_new, parent=nearest_node)
            tree.append(new_node)

            if np.linalg.norm(new_node.config - goal_config_arr) < 0.5:
                print("Goal reached!")
                return reconstruct_path(new_node)

    print("Failed to find path.")
    return []

def rrt_task_space():
    results = []
    targets = [start, intermediate, end]
    current_start = init_conf
    quat = p.getQuaternionFromEuler([0,0,0])
    for i, T in enumerate(targets):
        print(f"Planning Path {i+1} (RRT Task)...")
        path = rrt_task_space_planner(current_start, T, collision_fn)
        results.append(path)
        current_start = path[-1] if path else ik_conf(T, quat)
    return results

def birrt_planner(start_config, goal_config, collision_fn, max_iterations=5000, step_size=0.5):
    start_node = Node(start_config)
    tree_a = [start_node]
    goal_node = Node(goal_config)
    tree_b = [goal_node]
    active_tree = tree_a
    goal_threshold = 0.5

    for _ in range(max_iterations):
        other_tree = tree_b if active_tree is tree_a else tree_a
        q_rand = np.random.uniform(-np.pi, np.pi, 6)

        nearest_node = min(active_tree, key=lambda node: np.linalg.norm(node.config - q_rand))
        direction = q_rand - nearest_node.config
        dist = np.linalg.norm(direction)
        q_new = nearest_node.config + (direction/dist)*min(step_size, dist) if dist>0 else nearest_node.config

        if not collision_fn(q_new):
            new_node = Node(q_new, parent=nearest_node)
            active_tree.append(new_node)

            nearest_other = min(other_tree, key=lambda node: np.linalg.norm(node.config - new_node.config))
            if np.linalg.norm(new_node.config - nearest_other.config) < goal_threshold:
                print("BiRRT Connected!")
                if active_tree is tree_a:
                    return reconstruct_birrt_path(new_node, nearest_other)
                else:
                    return reconstruct_birrt_path(nearest_other, new_node)

        active_tree = other_tree

    print("BiRRT Failed.")
    return []

def birrt():
    return [birrt_planner(init_conf, start_conf, collision_fn),
            birrt_planner(start_conf, intermediate_conf, collision_fn),
            birrt_planner(intermediate_conf, end_conf, collision_fn)]

def birrt_task_space():
    results = []
    quat = p.getQuaternionFromEuler([0,0,0])
    targets = [start, intermediate, end]
    current = init_conf
    for T in targets:
        goal = ik_conf(T, quat)
        path = birrt_planner(current, goal, collision_fn)
        results.append(path)
        current = goal
    return results

# -------------------------------
# IK helper
# -------------------------------
def ik_conf(target_pos, target_orn):
    q = p.calculateInverseKinematics(ur5_robo, 7, target_pos, target_orn)
    return list(q[:6])

# -------------------------------
# Pipeline (unchanged)
# -------------------------------
if __name__ == "__main__":
    args = get_args()

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane = p.loadURDF("plane.urdf")
    flags = p.URDF_USE_INERTIA_FROM_FILE

    robo_base = p.loadURDF(os.path.join(assets_dir, "ur5_base.urdf"), [0,0,0.18], p.getQuaternionFromEuler([0,0,0]), flags=flags, useFixedBase=True)
    ur5_robo = p.loadURDF(os.path.join(assets_dir, "ur5.urdf"), [0,0,0.38], p.getQuaternionFromEuler([0,0,0]), flags=flags, useFixedBase=True)

    table_position = [0.45, 0.0, 0.0]
    table_id = p.loadURDF(os.path.join(assets_dir, "table", "table.urdf"), table_position, p.getQuaternionFromEuler([0,0,1.5708]), flags=flags, useFixedBase=True)

    cube_pos= [[0.65,0,0.35],[0.4,-0.2,0.35],[0.4,0.2,0.35]]
    cube_id_1 = p.loadURDF(os.path.join(assets_dir,"cube_and_square","cube_small_yellow.urdf"), cube_pos[0], p.getQuaternionFromEuler([0,0,0]), flags=flags)
    cube_id_2 = p.loadURDF(os.path.join(assets_dir,"cube_and_square","cube_small_yellow.urdf"), cube_pos[1], p.getQuaternionFromEuler([0,0,0]), flags=flags)
    cube_id_3 = p.loadURDF(os.path.join(assets_dir,"cube_and_square","cube_small_yellow.urdf"), cube_pos[2], p.getQuaternionFromEuler([0,0,0]), flags=flags)

    obstacles = [plane, table_id, cube_id_1, cube_id_2, cube_id_3]
    collision_fn = get_collision_fn(ur5_robo, ur5_joint_indices, obstacles=obstacles,
                                    attachments=[], self_collisions=False, disabled_collisions=set())

    init_conf = [0,-math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
    set_joint_positions(ur5_robo, ur5_joint_indices, init_conf)

    start = [0.3,-0.375,0.45]
    intermediate = [0.45,0.0,0.45]
    end = [0.6,0.375,0.55]
    quat = p.getQuaternionFromEuler([0,0,0])

    start_conf = ik_conf(start, quat)
    intermediate_conf = ik_conf(intermediate, quat)
    end_conf = ik_conf(end, quat)

    draw_sphere_marker(start,0.02,[0,0,1,0.5])
    draw_sphere_marker(intermediate,0.02,[0,1,0,0.5])
    draw_sphere_marker(end,0.02,[1,0,0,0.5])

    if args.birrt:
        path1,path2,path3 = birrt()
    elif args.birrt_task:
        path1,path2,path3 = birrt_task_space()
    elif args.rrt_task:
        path1,path2,path3 = rrt_task_space()
    else:
        path1,path2,path3 = rrt()

    timeStep = 0.5

    while True:
        execute_path_with_trail(ur5_robo, 7, ur5_joint_indices, path1, timeStep, [0,0,1])
        execute_path_with_trail(ur5_robo, 7, ur5_joint_indices, path2, timeStep, [0,1,0])
        execute_path_with_trail(ur5_robo, 7, ur5_joint_indices, path3, timeStep, [1,0,0])
        print("finished!")
        time.sleep(6)
