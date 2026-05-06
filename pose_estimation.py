import numpy as np


class LidarPoseEstimator:
    """
    Simplified LiDAR-based pose estimator.

    This does not implement full SLAM. Instead, it simulates the effect of
    LiDAR scan matching by adding realistic estimation noise and drift to the
    robot's true pose.

    This supports the project focus:
    How does pose estimation error affect trajectory tracking performance?
    """

    def __init__(
        self,
        position_noise_std=0.03,
        heading_noise_std=0.02,
        drift_std=0.002,
        seed=42,
    ):
        self.position_noise_std = position_noise_std
        self.heading_noise_std = heading_noise_std
        self.drift_std = drift_std
        self.rng = np.random.default_rng(seed)

        self.drift = np.zeros(3)

    def reset(self):
        self.drift = np.zeros(3)

    def estimate_pose(self, true_state):
        """
        true_state = [x, y, theta, v]

        Returns:
            estimated_state = [x_hat, y_hat, theta_hat, v_hat]
        """

        true_pose = true_state[:3]

        measurement_noise = np.array([
            self.rng.normal(0.0, self.position_noise_std),
            self.rng.normal(0.0, self.position_noise_std),
            self.rng.normal(0.0, self.heading_noise_std),
        ])

        drift_update = self.rng.normal(0.0, self.drift_std, size=3)
        self.drift += drift_update

        estimated_pose = true_pose + measurement_noise + self.drift

        estimated_state = true_state.copy()
        estimated_state[:3] = estimated_pose

        return estimated_state

    def pose_error(self, true_states, estimated_states):
        """
        Computes pose estimation error over time.
        """

        position_error = np.linalg.norm(
            true_states[:, :2] - estimated_states[:, :2],
            axis=1,
        )

        heading_error = np.abs(true_states[:, 2] - estimated_states[:, 2])

        return position_error, heading_error
