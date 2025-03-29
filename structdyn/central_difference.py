import numpy as np


class CentralDifference:
    """Interpolation Scheme for Solving Structural Dynamics Numerically"""

    def __init__(
        self, sdf, dt, u_init=None, u_dot_init=None, p_init=None, non_linear=False
    ):
        self.dt = dt
        self.non_linear = non_linear

        k, m, ji, w_n, w_d = sdf.k, sdf.m, sdf.ji, sdf.w_n, sdf.w_d
        c = 2 * m * w_n * ji  # damping constant
        self.u_init = u_init if u_init is not None else 0
        self.u_dot_init = u_dot_init if u_dot_init is not None else 0
        p_init = 0 if p_init is None else p_init

        u_dot2_init = (p_init - c * self.u_dot_init - k * self.u_init) / m

        self.u_minus1 = (
            self.u_init - self.dt * self.u_dot_init + u_dot2_init * (dt**2) / 2
        )
        self.k_bar = m / dt**2 + c / (2 * dt)
        self.a = m / dt**2 - c / (2 * dt)
        self.b = k - 2 * m / dt**2
        self.b_bar = 2 * m / dt**2

    def compute_solution(self, time_steps, load_values, fs=None):
        """Computes the displacement and velocity response of the SDF system."""
        n_steps = len(time_steps)

        u = np.zeros(n_steps)
        u_dot, p_bar = np.zeros(n_steps), np.zeros(n_steps)
        u[0] = self.u_init
        u_dot[0] = self.u_dot_init
        u_last = 0
        fs_last = 0
        for i in range(n_steps - 1):
            if self.non_linear:
                if fs.__name__ == "get_fs":
                    p_bar[i] = (
                        load_values[i]
                        - self.a * (self.u_minus1 if i == 0 else u[i - 1])
                        + self.b_bar * u[i]
                        - fs(u[i])
                    )
                else:
                    fs_val = fs(u[i], u_last, fs_last)
                    p_bar[i] = (
                        load_values[i]
                        - self.a * (self.u_minus1 if i == 0 else u[i - 1])
                        + self.b_bar * u[i]
                        - fs_val
                    )
                    fs_last = fs_val
            else:
                p_bar[i] = (
                    load_values[i]
                    - self.a * (self.u_minus1 if i == 0 else u[i - 1])
                    - self.b * u[i]
                )
            u[i + 1] = p_bar[i] / self.k_bar
            u_dot[i] = (u[i + 1] - u[i - 1]) / (2 * self.dt)

            u_last = u[i]
        return u, u_dot
