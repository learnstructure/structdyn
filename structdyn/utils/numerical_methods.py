import numpy as np


class Interpolation:
    """Interpolation Scheme for Solving Structural Dynamics Numerically"""

    def __init__(self, sdf, dt):
        self.dt = dt

        k, m, ji, w_n, w_d = sdf.k, sdf.m, sdf.ji, sdf.w_n, sdf.w_d
        self.k = sdf.k
        e_jwnt = np.exp(-ji * w_n * dt)
        sin_wdt = np.sin(w_d * dt)
        cos_wdt = np.cos(w_d * dt)

        self.A = e_jwnt * (sin_wdt * ji / np.sqrt(1 - ji**2) + cos_wdt)
        self.B = e_jwnt * sin_wdt / w_d

        k1 = 2 * ji / (w_n * dt)
        k2 = 1 - 2 * ji**2
        k3 = np.sqrt(1 - ji**2)
        wdt = w_d * dt
        wnt = w_n * dt

        self.C = (
            k1
            + e_jwnt * ((k2 / wdt - ji / k3) * sin_wdt - (1 + 2 * ji / wnt) * cos_wdt)
        ) / k
        self.D = (1 - k1 + e_jwnt * (-k2 / wdt * sin_wdt + 2 * ji / wnt * cos_wdt)) / k

        self.A_ = -e_jwnt * w_n / k3 * sin_wdt
        self.B_ = e_jwnt * (cos_wdt - ji * sin_wdt / k3)
        self.C_ = (
            -1 / dt + e_jwnt * ((w_n / k3 + ji / (dt * k3)) * sin_wdt + cos_wdt / dt)
        ) / k
        self.D_ = (1 - e_jwnt * (ji * sin_wdt / k3 + cos_wdt)) / (k * dt)

    def compute_solution(self, time_steps, load_values, u_init=None, u_dot_init=None):
        """Computes the displacement, velocity and resisting force of the SDF system."""
        n_steps = len(time_steps)
        u, u_dot = np.zeros(n_steps), np.zeros(n_steps)

        u[0] = u_init if u_init is not None else 0
        u_dot[0] = u_dot_init if u_dot_init is not None else 0

        for i in range(n_steps - 1):
            u[i + 1] = (
                self.A * u[i]
                + self.B * u_dot[i]
                + self.C * load_values[i]
                + self.D * load_values[i + 1]
            )
            u_dot[i + 1] = (
                self.A_ * u[i]
                + self.B_ * u_dot[i]
                + self.C_ * load_values[i]
                + self.D_ * load_values[i + 1]
            )
        fs_val = self.k * u
        return u, u_dot, fs_val


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
