"""
Helper utilities for Assignment 2, Part 2: Franka robot arm path planning.

Provides:
  - load_scene(): Load the MuJoCo model with obstacle
  - get_joint_limits(): Return (lower, upper) joint limit arrays
  - check_collision(model, data, q): Check if a configuration is in collision
  - check_edge(model, data, q_a, q_b, n_checks=20): Check if edge is collision-free
  - set_config(model, data, q): Set robot to a configuration
  - animate_path(model, data, path): Visualize the planned path
  - load_problems(): Load start/goal configurations

Usage:
    python3 franka_utils.py 1

Loads problem 1, shows start/goal configs, and tests the collision checker.
"""

import numpy as np
import mujoco
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(SCRIPT_DIR, "franka_scene.xml")
PROBLEMS_PATH = os.path.join(SCRIPT_DIR, "franka_problems.npz")
N_JOINTS = 7


def load_scene(scene_path=SCENE_PATH):
    """Load the MuJoCo model and create data."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    return model, data


def get_joint_limits(model):
    """
    Return joint limits as (lower, upper) arrays of shape (7,).
    These define the configuration space bounds for sampling.
    """
    lower = model.jnt_range[:N_JOINTS, 0].copy()
    upper = model.jnt_range[:N_JOINTS, 1].copy()
    return lower, upper


def set_config(model, data, q):
    """
    Set the robot to configuration q and run forward kinematics.

    Parameters:
        model: MjModel
        data: MjData
        q: array of shape (7,) with joint angles
    """
    data.qpos[:N_JOINTS] = q
    mujoco.mj_forward(model, data)


def check_collision(model, data, q):
    """
    Check if configuration q is in collision.

    Returns True if any contacts are detected, False if collision-free.
    Ignores contacts with the floor (link0 resting on ground is not a collision).
    """
    set_config(model, data, q)
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        # Skip floor contacts with the base link (the robot is bolted down)
        pair = {geom1_name, geom2_name}
        if "floor" in pair and "link0_geom" in pair:
            continue
        return True
    return False


def check_edge(model, data, q_a, q_b, n_checks=20):
    """
    Check if the straight-line path from q_a to q_b in joint space
    is collision-free by sampling n_checks configurations along it.

    Returns True if collision-free, False if any sample is in collision.
    """
    for t in np.linspace(0, 1, n_checks):
        q = (1 - t) * q_a + t * q_b
        if check_collision(model, data, q):
            return False
    return True


def load_problems(problems_path=PROBLEMS_PATH):
    """
    Load planning problems. Returns a list of (start, goal) tuples,
    each containing 7-element joint angle arrays.
    """
    data = np.load(problems_path)
    problems = []
    i = 0
    while f"start_{i}" in data:
        problems.append((data[f"start_{i}"], data[f"goal_{i}"]))
        i += 1
    return problems


def animate_path(model, data, path, dt=0.05):
    """
    Animate the robot following a joint-space path using the MuJoCo viewer.

    Parameters:
        model: MjModel
        data: MjData
        path: array of shape (K, 7) with joint-space waypoints
        dt: time between waypoints (seconds)
    """
    import mujoco.viewer
    import time

    viewer = mujoco.viewer.launch_passive(model, data,show_left_ui=False, show_right_ui=False,)
    try:
        for q in path:
            if not viewer.is_running():
                break
            set_config(model, data, q)
            viewer.sync()
            time.sleep(dt)
        # Hold the final configuration
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a Franka planning problem and test collision checking"
    )
    parser.add_argument("num", type=int, help="Problem number (1-indexed)")
    args = parser.parse_args()

    model, data = load_scene()
    problems = load_problems()

    idx = args.num - 1
    if idx < 0 or idx >= len(problems):
        print(f"Problem {args.num} not found. Available: 1-{len(problems)}")
        return

    q_start, q_goal = problems[idx]
    lower, upper = get_joint_limits(model)

    print(f"Problem {args.num}:")
    print(f"  Start: {np.array2string(q_start, precision=3)}")
    print(f"  Goal:  {np.array2string(q_goal, precision=3)}")
    print(f"  Joint limits (lower): {np.array2string(lower, precision=3)}")
    print(f"  Joint limits (upper): {np.array2string(upper, precision=3)}")

    start_collision = check_collision(model, data, q_start)
    goal_collision = check_collision(model, data, q_goal)
    print(f"  Start in collision: {start_collision}")
    print(f"  Goal in collision:  {goal_collision}")

    # Show start, then goal
    print("\nShowing start configuration...")
    import mujoco.viewer
    import time

    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        set_config(model, data, q_start)
        viewer.sync()
        print("  (showing start config for 3 seconds, then goal)")
        time.sleep(3)
        set_config(model, data, q_goal)
        viewer.sync()
        print("  (showing goal config -- close viewer to exit)")
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
