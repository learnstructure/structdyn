import numpy as np
import pandas as pd


class CentralDifference:
    """
    Central Difference Method for linear and nonlinear SDOF systems
    (Chopra Table 5.3.1 and Section 5.6)
    """

    def __init__(self, sdf, dt, u0=0.0, v0=0.0):
        self.sdf = sdf
        self.dt = dt

        self.m = sdf.m
        self.k = sdf.k
        self.c = sdf.c

        self.u0 = u0
        self.v0 = v0

        # Central difference constants
        self.k_bar = self.m / dt**2 + self.c / (2 * dt)
        self.a = self.m / dt**2 - self.c / (2 * dt)
        self.b = self.k - 2 * self.m / dt**2
        self.b_bar = 2 * self.m / dt**2

    def compute_solution(self, time, p):
        n = len(time)
        dt = self.dt

        u = np.zeros(n)
        v = np.zeros(n)
        acc = np.zeros(n)
        fs_hist = np.zeros(n)

        # Initial conditions
        u[0] = self.u0
        v[0] = self.v0

        acc[0] = (p[0] - self.c * v[0] - self.k * u[0]) / self.m

        # u_{-1}
        u_minus1 = u[0] - dt * v[0] + 0.5 * dt**2 * acc[0]

        for i in range(n - 1):
            u_prev = u_minus1 if i == 0 else u[i - 1]

            if self.sdf.fd == "linear":
                fs_i = self.k * u[i]
                p_hat = p[i] - self.a * u_prev - self.b * u[i]
            else:
                fs_i, _, _ = self.sdf.fd.trial_response(u[i])
                self.sdf.fd.commit_state(u[i])
                p_hat = p[i] - self.a * u_prev + self.b_bar * u[i] - fs_i

            fs_hist[i] = fs_i

            u[i + 1] = p_hat / self.k_bar
            v[i] = (u[i + 1] - u_prev) / (2 * dt)
            acc[i] = (u[i + 1] - 2 * u[i] + u_prev) / dt**2

            u_minus1 = u[i]

        # Last step
        fs_hist[-1] = (
            self.k * u[-1]
            if self.sdf.fd == "linear"
            else self.sdf.fd.trial_response(u[-1])[0]
        )

        v[-1] = (u[-1] - u[-2]) / dt
        acc[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dt**2

        return pd.DataFrame(
            {
                "time": time,
                "displacement": u,
                "velocity": v,
                "acceleration": acc,
                "resisting_force": fs_hist,
            }
        )
