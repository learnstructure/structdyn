import numpy as np
import pandas as pd


class Interpolation:
    """
    Solves the equation of motion for a linear SDOF system using an exact
    integration method assuming linear interpolation of the excitation force.

    This method is based on the recurrence formulas derived from the exact solution
    of the differential equation for a given time step. It is unconditionally stable
    and highly accurate for linear systems.

    Reference: Chopra, A. K. (2020). Dynamics of Structures: Theory and Applications
    to Earthquake Engineering. Pearson Education.
    (See Eq. 5.2.5a-b and Table 5.2.1 for ji < 1)
    """

    def __init__(self, sdf, dt):
        """
        Initializes the Interpolation solver and pre-computes recurrence coefficients.

        Parameters
        ----------
        sdf : SDF
            The Single Degree of Freedom system to be analyzed. The system must be
            linear and underdamped (0 <= ji < 1).
        dt : float
            The time step for the numerical integration.
        """
        self.dt = dt
        self.sdf = sdf
        # System properties
        self.m = sdf.m
        self.k = sdf.k
        self.zeta = sdf.ji
        self.wn = sdf.w_n
        self.wd = sdf.w_d

        z = self.zeta
        wn = self.wn
        wd = self.wd
        k = self.k
        dt = self.dt

        # Common factors
        exp_term = np.exp(-z * wn * dt)
        sin_term = np.sin(wd * dt)
        cos_term = np.cos(wd * dt)
        sqrt_term = np.sqrt(1 - z**2)

        # --- Displacement recurrence coefficients (Eq. 5.2.5a) ---
        self.A = exp_term * (z / sqrt_term * sin_term + cos_term)
        self.B = exp_term * (sin_term / wd)

        k1 = 2 * z / (wn * dt)
        k2 = 1 - 2 * z**2

        self.C = (
            k1
            + exp_term
            * (
                (k2 / (wd * dt) - z / sqrt_term) * sin_term
                - (1 + 2 * z / (wn * dt)) * cos_term
            )
        ) / k

        self.D = (
            1
            - k1
            + exp_term * (-k2 / (wd * dt) * sin_term + 2 * z / (wn * dt) * cos_term)
        ) / k

        # --- Velocity recurrence coefficients (Eq. 5.2.5b) ---
        self.A_v = -exp_term * (wn / sqrt_term) * sin_term
        self.B_v = exp_term * (cos_term - z / sqrt_term * sin_term)

        self.C_v = (
            -1 / dt
            + exp_term
            * ((wn / sqrt_term + z / (dt * sqrt_term)) * sin_term + cos_term / dt)
        ) / k

        self.D_v = (1 - exp_term * (z / sqrt_term * sin_term + cos_term)) / (k * dt)

    def compute_solution(self, time, p, u0=0.0, v0=0.0):
        """
        Performs the time-stepping solution using the pre-computed coefficients.

        Parameters
        ----------
        time : array-like
            An array representing the time vector of the analysis.
        p : array-like
            An array representing the external force applied at each time step.
        u0 : float, optional
            Initial displacement at time t=0, by default 0.0.
        v0 : float, optional
            Initial velocity at time t=0, by default 0.0.

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

        u = np.zeros(n)
        v = np.zeros(n)
        # acc = np.zeros(n)

        u[0] = u0
        v[0] = v0

        for i in range(n - 1):
            u[i + 1] = self.A * u[i] + self.B * v[i] + self.C * p[i] + self.D * p[i + 1]

            v[i + 1] = (
                self.A_v * u[i]
                + self.B_v * v[i]
                + self.C_v * p[i]
                + self.D_v * p[i + 1]
            )
        acc = (p - self.sdf.c * v - self.k * u) / self.m
        fs = self.k * u

        return pd.DataFrame(
            {
                "time": time,
                "displacement": u,
                "velocity": v,
                "acceleration": acc,
                "resisting_force": fs,
            }
        )
