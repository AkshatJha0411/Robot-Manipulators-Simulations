# AR523 - Robot Manipulators
## Motion Planning and Force Control (UR5 in PyBullet)

**Author:** Akshat Jha (B23336)  
**Platform:** PyBullet + Python  
**Robot:** UR5 Manipulator 

A UR5 is a 6-DoF collaborative industrial robotic arm made by Universal Robots, commonly used for flexible automation and research applications.

This repository contains implementations for two assignments completed (till now) as part of the AR523 coursework.  
Both assignments were carried out in the same simulation environment using the UR5 robot model.

The task statements and instructions are located in:
- `a1/Assignment1.pdf`
- `a2/Assignment2.pdf`

---

## Folder Structure

```

ASSIGNMENTS
│
├── a1
│   ├── assets/              # URDF + scene models
│   ├── utils.py             # Helper functions
│   ├── main.py              # Motion planning pipeline
│   ├── env.png              # Environment snapshot
│   └── Assignment1.pdf      # Problem statement
│
└── a2
├── assets/              # Surface, plane, and models
├── utils.py             # Shared helper functions
├── p1.py                # Admittance Control (Static)
├── p2.py                # Admittance Control (Dynamic)
├── p3.py                # Impedance Control (Static)
├── p4.py                # Impedance Control (Dynamic)
├── p6.py                # Control on varying/tilted surface
├── extra.py             # Additional variation testing
├── env.png              # Simulation environment snapshot
└── Assignment2.pdf      # Problem statement

````

---

## Assignment 1 — Sampling-Based Motion Planning (UR5)

### Goal
Plan collision-free manipulator motion between multiple target configurations.

### Implemented
| Method | Configuration Space | Description |
|-------|--------------------|-------------|
| RRT | Joint Space | Random sampling tree expansion to reach goals. |
| BiRRT | Joint Space | Two-tree approach for improved convergence. |
| RRT (Task-Space) | Cartesian EE Space | Sampling in EE-space + IK projection. |
| BiRRT (Task-Space) | Cartesian EE Space | Bi-directional version in task-space. |

### Run
```bash
cd a1
python main.py --rrt
python main.py --birrt
python main.py --rrt_task
python main.py --birrt_task
````

---

## Assignment 2 — Force-Based Interaction Control (UR5)

### Objective

Perform controlled interaction with a surface and study force response.

### Implemented Control Schemes

| Part | Control                    | Interaction Mode | File    |
| ---- | -------------------------- | ---------------- | ------- |
| 1    | Admittance                 | Static Contact   | `p1.py` |
| 2    | Admittance                 | Circular Motion  | `p2.py` |
| 3    | Impedance (Position-based) | Static Contact   | `p3.py` |
| 4    | Impedance (Position-based) | Circular Motion  | `p4.py` |

### Additional Surface Interaction

| File       | Description                                             |
| ---------- | ------------------------------------------------------- |
| `p6.py`    | Control behavior on a tilted / uneven / varying surface |
| `extra.py` | Alternative test path + robustness evaluation           |

---

## Setup

```bash
pip install pybullet numpy matplotlib
```

Make sure that the `assets/` folder inside both `a1/` and `a2/` is kept intact.

---

## Running Assignment 2

```bash
cd a2
python p1.py   # Admittance Static
python p2.py   # Admittance Dynamic
python p3.py   # Impedance Static
python p4.py   # Impedance Dynamic
python p6.py   # Interaction over varying surface
```

---

## Summary

* **Assignment 1** focuses on **path planning** using RRT-based sampling algorithms in both joint and task space.
* **Assignment 2** focuses on **force interaction control**, comparing admittance and impedance behaviors under both static and dynamic motion.
* A **non-flat surface interaction** setup is also included to observe stability and responsiveness under varying geometry.
