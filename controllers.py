import numpy as np
from scipy.linalg import solve_discrete_are


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def dynamics(x, u, dt):
    """
    Nonlinear unicycle-style robot dynamics.

    State:
        x = [px, py, theta, v]

    Control:
        u = [a, omega]

    where:
        px, py = position
        theta = heading
        v = speed
        a = acceleration
        omega = turn rate
    """

    px, py, theta, v = x
    a, omega = u

    next_x = np.zeros_like(x)

    next_x[0] = px + dt * v * np.cos(theta)
    next_x[1] = py + dt * v * np.sin(theta)
    next_x[2] = wrap_angle(theta + dt * omega)
    next_x[3] = v + dt * a

    return next_x


def finite_difference_jacobians(x, u, dt, eps=1e-5):
    """
    Numerically computes A and B matrices for local linearization.

    A = df/dx
    B = df/du
    """

    n = len(x)
    m = len(u)

    A = np.zeros((n, n))
    B = np.zeros((n, m))

    fx = dynamics(x, u, dt)

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (dynamics(x + dx, u, dt) - fx) / eps

    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        B[:, j] = (dynamics(x, u + du, dt) - fx) / eps

    return A, B


def trajectory_cost(X, U, X_ref, Q, R, Qf):
    cost = 0.0

    for k in range(len(U)):
        dx = X[k] - X_ref[k]
        dx[2] = wrap_angle(dx[2])
        cost += dx.T @ Q @ dx
        cost += U[k].T @ R @ U[k]

    dx_final = X[-1] - X_ref[-1]
    dx_final[2] = wrap_angle(dx_final[2])
    cost += dx_final.T @ Qf @ dx_final

    return cost


def rollout(x0, U, dt):
    X = [x0.copy()]

    x = x0.copy()

    for u in U:
        x = dynamics(x, u, dt)
        X.append(x.copy())

    return np.array(X)


class ILQRController:
    """
    iLQR trajectory optimizer for nonlinear trajectory tracking.
    """

    def __init__(
        self,
        dt,
        Q=None,
        R=None,
        Qf=None,
        max_iters=50,
        regularization=1e-6,
    ):
        self.dt = dt

        self.Q = Q if Q is not None else np.diag([20.0, 20.0, 5.0, 1.0])
        self.R = R if R is not None else np.diag([0.1, 0.1])
        self.Qf = Qf if Qf is not None else np.diag([50.0, 50.0, 10.0, 5.0])

        self.max_iters = max_iters
        self.regularization = regularization

    def optimize(self, x0, X_ref, U_init):
        """
        Returns:
            X_nominal
            U_nominal
            K_feedback
        """

        U = U_init.copy()
        N = len(U)

        X = rollout(x0, U, self.dt)

        for iteration in range(self.max_iters):
            old_cost = trajectory_cost(X, U, X_ref, self.Q, self.R, self.Qf)

            A_list = []
            B_list = []

            for k in range(N):
                A, B = finite_difference_jacobians(X[k], U[k], self.dt)
                A_list.append(A)
                B_list.append(B)

            Vx = self.Qf @ (X[-1] - X_ref[-1])
            Vx[2] = wrap_angle(Vx[2])

            Vxx = self.Qf.copy()

            k_list = []
            K_list = []

            diverged = False

            for k in reversed(range(N)):
                A = A_list[k]
                B = B_list[k]

                dx = X[k] - X_ref[k]
                dx[2] = wrap_angle(dx[2])

                lx = self.Q @ dx
                lu = self.R @ U[k]
                lxx = self.Q
                luu = self.R
                lux = np.zeros((2, 4))

                Qx = lx + A.T @ Vx
                Qu = lu + B.T @ Vx
                Qxx = lxx + A.T @ Vxx @ A
                Quu = luu + B.T @ Vxx @ B
                Qux = lux + B.T @ Vxx @ A

                Quu_reg = Quu + self.regularization * np.eye(Quu.shape[0])

                try:
                    Quu_inv = np.linalg.inv(Quu_reg)
                except np.linalg.LinAlgError:
                    diverged = True
                    break

                k_ff = -Quu_inv @ Qu
                K_fb = -Quu_inv @ Qux

                Vx = Qx + K_fb.T @ Quu @ k_ff + K_fb.T @ Qu + Qux.T @ k_ff
                Vxx = Qxx + K_fb.T @ Quu @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb

                k_list.insert(0, k_ff)
                K_list.insert(0, K_fb)

            if diverged:
                break

            improved = False

            for alpha in [1.0, 0.5, 0.25, 0.1, 0.05]:
                X_new = [x0.copy()]
                U_new = []

                x = x0.copy()

                for k in range(N):
                    dx = x - X[k]
                    dx[2] = wrap_angle(dx[2])

                    u = U[k] + alpha * k_list[k] + K_list[k] @ dx
                    x = dynamics(x, u, self.dt)

                    U_new.append(u.copy())
                    X_new.append(x.copy())

                X_new = np.array(X_new)
                U_new = np.array(U_new)

                new_cost = trajectory_cost(X_new, U_new, X_ref, self.Q, self.R, self.Qf)

                if new_cost < old_cost:
                    X = X_new
                    U = U_new
                    improved = True
                    break

            if not improved:
                break

        return X, U, np.array(K_list)


class LQRController:
    """
    Time-varying LQR baseline.

    This controller linearizes the nonlinear system along the desired
    trajectory and computes feedback gains.
    """

    def __init__(self, dt, Q=None, R=None):
        self.dt = dt

        self.Q = Q if Q is not None else np.diag([20.0, 20.0, 5.0, 1.0])
        self.R = R if R is not None else np.diag([0.1, 0.1])

    def compute_gains(self, X_ref, U_ref):
        N = len(U_ref)

        K_list = []

        P = self.Q.copy()

        for k in reversed(range(N)):
            A, B = finite_difference_jacobians(X_ref[k], U_ref[k], self.dt)

            S = self.R + B.T @ P @ B
            K = np.linalg.inv(S) @ B.T @ P @ A

            P = self.Q + A.T @ P @ (A - B @ K)

            K_list.insert(0, K)

        return np.array(K_list)

    def control(self, x_feedback, x_ref, u_ref, K):
        dx = x_feedback - x_ref
        dx[2] = wrap_angle(dx[2])

        u = u_ref - K @ dx

        return u
