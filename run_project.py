import numpy as np
import matplotlib.pyplot as plt
import time

from pose_estimation import LidarPoseEstimator
from controllers import (
    dynamics,
    rollout,
    wrap_angle,
    ILQRController,
    LQRController,
)


def create_reference_trajectory(T=180, dt=0.05):
    """
    Creates a nonlinear curved trajectory.

    This gives iLQR a meaningful advantage over plain LQR.
    """

    X_ref = []
    U_ref = []

    radius = 2.0
    speed = 0.6
    omega = speed / radius

    for k in range(T + 1):
        t = k * dt

        angle = omega * t

        px = radius * np.cos(angle)
        py = radius * np.sin(angle)
        theta = wrap_angle(angle + np.pi / 2)
        v = speed

        X_ref.append([px, py, theta, v])

    for k in range(T):
        a = 0.0
        turn_rate = omega
        U_ref.append([a, turn_rate])

    return np.array(X_ref), np.array(U_ref)


def simulate_controller(
    name,
    controller_type,
    X_ref,
    U_ref,
    use_estimated_pose,
    dt,
):
    """
    Runs one full experiment.

    controller_type:
        "lqr" or "ilqr"

    use_estimated_pose:
        False = controller receives true pose
        True  = controller receives LiDAR-estimated pose
    """

    lidar = LidarPoseEstimator(
        position_noise_std=0.05,
        heading_noise_std=0.03,
        drift_std=0.0015,
        seed=7,
    )

    x0 = X_ref[0].copy()
    x0[0] -= 0.3
    x0[1] += 0.2
    x0[2] += 0.15

    start_time = time.time()

    if controller_type == "ilqr":
        ilqr = ILQRController(dt=dt)
        X_nominal, U_nominal, K_list = ilqr.optimize(x0, X_ref, U_ref)
    elif controller_type == "lqr":
        lqr = LQRController(dt=dt)
        K_list = lqr.compute_gains(X_ref, U_ref)
        X_nominal = X_ref.copy()
        U_nominal = U_ref.copy()
    else:
        raise ValueError("controller_type must be 'lqr' or 'ilqr'")

    compute_time = time.time() - start_time

    true_states = [x0.copy()]
    estimated_states = []
    controls = []

    x_true = x0.copy()

    for k in range(len(U_ref)):
        if use_estimated_pose:
            x_feedback = lidar.estimate_pose(x_true)
        else:
            x_feedback = x_true.copy()

        estimated_states.append(x_feedback.copy())

        if controller_type == "ilqr":
            dx = x_feedback - X_nominal[k]
            dx[2] = wrap_angle(dx[2])
            u = U_nominal[k] + K_list[k] @ dx
        else:
            u = lqr.control(
                x_feedback=x_feedback,
                x_ref=X_ref[k],
                u_ref=U_ref[k],
                K=K_list[k],
            )

        controls.append(u.copy())

        x_true = dynamics(x_true, u, dt)
        true_states.append(x_true.copy())

    true_states = np.array(true_states)
    estimated_states = np.array(estimated_states)
    controls = np.array(controls)

    tracking_error = np.linalg.norm(true_states[:, :2] - X_ref[:, :2], axis=1)

    if use_estimated_pose:
        pose_error = np.linalg.norm(
            estimated_states[:, :2] - true_states[:-1, :2],
            axis=1,
        )
    else:
        pose_error = np.zeros(len(U_ref))

    control_effort = np.sum(np.linalg.norm(controls, axis=1) ** 2)

    return {
        "name": name,
        "true_states": true_states,
        "estimated_states": estimated_states,
        "controls": controls,
        "tracking_error": tracking_error,
        "pose_error": pose_error,
        "control_effort": control_effort,
        "compute_time": compute_time,
        "final_error": tracking_error[-1],
        "mean_error": np.mean(tracking_error),
    }


def print_results(results):
    print("\n================ RESULTS ================")

    for r in results:
        print(f"\n{r['name']}")
        print(f"  Mean tracking error: {r['mean_error']:.4f} m")
        print(f"  Final tracking error: {r['final_error']:.4f} m")
        print(f"  Control effort: {r['control_effort']:.4f}")
        print(f"  Computation time: {r['compute_time']:.4f} sec")

        if len(r["pose_error"]) > 0:
            print(f"  Mean pose estimation error: {np.mean(r['pose_error']):.4f} m")


def plot_trajectories(X_ref, results):
    plt.figure(figsize=(8, 8))

    plt.plot(X_ref[:, 0], X_ref[:, 1], "k--", linewidth=2, label="Desired trajectory")

    for r in results:
        X = r["true_states"]
        plt.plot(X[:, 0], X[:, 1], linewidth=1.8, label=r["name"])

    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Desired vs Actual Trajectories")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)


def plot_tracking_error(results):
    plt.figure(figsize=(10, 5))

    for r in results:
        plt.plot(r["tracking_error"], label=r["name"])

    plt.xlabel("Timestep")
    plt.ylabel("Tracking error")
    plt.title("Tracking Error Over Time")
    plt.legend()
    plt.grid(True)


def plot_pose_error(results):
    plt.figure(figsize=(10, 5))

    for r in results:
        if "Estimated" in r["name"]:
            plt.plot(r["pose_error"], label=r["name"])

    plt.xlabel("Timestep")
    plt.ylabel("Pose estimation error")
    plt.title("LiDAR-Based Pose Estimation Error")
    plt.legend()
    plt.grid(True)


def plot_control_effort(results):
    names = [r["name"] for r in results]
    efforts = [r["control_effort"] for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(names, efforts)
    plt.ylabel("Total control effort")
    plt.title("Control Effort Comparison")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y")


def main():
    dt = 0.05

    X_ref, U_ref = create_reference_trajectory(T=180, dt=dt)

    experiments = [
        ("LQR with Ground Truth Pose", "lqr", False),
        ("LQR with Estimated Pose", "lqr", True),
        ("iLQR with Ground Truth Pose", "ilqr", False),
        ("iLQR with Estimated Pose", "ilqr", True),
    ]

    results = []

    for name, controller_type, use_estimated_pose in experiments:
        print(f"Running experiment: {name}")

        result = simulate_controller(
            name=name,
            controller_type=controller_type,
            X_ref=X_ref,
            U_ref=U_ref,
            use_estimated_pose=use_estimated_pose,
            dt=dt,
        )

        results.append(result)

    print_results(results)

    plot_trajectories(X_ref, results)
    plot_tracking_error(results)
    plot_pose_error(results)
    plot_control_effort(results)

    plt.show()


if __name__ == "__main__":
    main()
