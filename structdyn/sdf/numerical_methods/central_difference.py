import numpy as np


class CentralDifference:
    """Central difference Scheme for Solving Structural Dynamics Numerically"""

    def __init__(
        self, sdf, dt, u_init=None, u_dot_init=None, p_init=None, non_linear=False
    ):
        self.dt = dt
        self.non_linear = non_linear

        self.k, m, ji, w_n = sdf.k, sdf.m, sdf.ji, sdf.w_n
        c = 2 * m * w_n * ji  # Damping constant

        self.u_init = u_init if u_init is not None else 0
        self.u_dot_init = u_dot_init if u_dot_init is not None else 0
        p_init = 0 if p_init is None else p_init

        u_dot2_init = (p_init - c * self.u_dot_init - self.k * self.u_init) / m
        self.u_minus1 = self.u_init - dt * self.u_dot_init + (u_dot2_init * dt**2) / 2

        self.k_bar = m / dt**2 + c / (2 * dt)
        self.a = m / dt**2 - c / (2 * dt)
        self.b = self.k - 2 * m / dt**2
        self.b_bar = 2 * m / dt**2

    def compute_solution(self, time_steps, load_values, fs=None):
        """Computes the displacement, velocity and resisting force of the SDF system."""
        n_steps = len(time_steps)
        u, u_dot, p_bar, fs_val = (
            np.zeros(n_steps),
            np.zeros(n_steps),
            np.zeros(n_steps),
            np.zeros(n_steps),
        )

        u[0], u_dot[0], fs_last = self.u_init, self.u_dot_init, 0

        for i in range(n_steps - 1):
            u_prev = self.u_minus1 if i == 0 else u[i - 1]

            if self.non_linear:
                fs_val[i] = fs(u[i], u_prev, fs_last)
                p_bar[i] = (
                    load_values[i] - self.a * u_prev + self.b_bar * u[i] - fs_val[i]
                )
                fs_last = fs_val[i]
            else:
                p_bar[i] = load_values[i] - self.a * u_prev - self.b * u[i]
                fs_val[i] = self.k * u[i]

            u[i + 1] = p_bar[i] / self.k_bar
            u_dot[i] = (u[i + 1] - u_prev) / (2 * self.dt)

        fs_val[-1] = fs(u[-1], u[-2], fs_last) if self.non_linear else self.k * u[-1]
        return u, u_dot, fs_val
