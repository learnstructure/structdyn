import numpy as np
import pandas as pd


class CentralDifference:
    """
    Implements the Central Difference Method for solving the equation of motion for an SDF system.

    This method is an explicit time-stepping algorithm suitable for both linear and
    nonlinear Single Degree of Freedom (SDF) systems. It is conditionally stable
    and requires the time step `dt` to be smaller than a critical value (dt_crit = T_n / pi).
    """

    def __init__(self, sdf, dt, u0=0.0, v0=0.0):
        """
        Initializes the Central Difference solver.

        Parameters
        ----------
        sdf : SDF
            The Single Degree of Freedom system to be analyzed.
        dt : float
            The time step for the numerical integration.
        u0 : float, optional
            Initial displacement at time t=0, by default 0.0.
        v0 : float, optional
            Initial velocity at time t=0, by default 0.0.
        """
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
        """
        Executes the time-stepping solution.

        This method iterates through the time vector, calculating the displacement,
        velocity, and acceleration of the system at each step.

        Parameters
        ----------
        time : array-like
            An array representing the time vector of the analysis.
        p : array-like
            An array representing the external force applied at each time step.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the full time history of the response, including:
            - 'time': Time points.
            - 'displacement': Displacement at each time point.
            - 'velocity': Velocity at each time point.
            - 'acceleration': Acceleration at each time point.
            - 'resisting_force': Internal resisting force at each time point.
        """
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
