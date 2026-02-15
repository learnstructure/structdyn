import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve


class CentralDifferenceMDF:

    def __init__(self, mdf, dt, u0=None, v0=None):

        self.mdf = mdf
        self.dt = dt

        self.M = mdf.M
        self.C = mdf.C
        self.K = mdf.K

        self.ndof = mdf.ndof

        self.u0 = np.zeros(self.ndof) if u0 is None else np.asarray(u0, dtype=float)
        self.v0 = np.zeros(self.ndof) if v0 is None else np.asarray(v0, dtype=float)

        # Central difference matrices
        self.K_bar = self.M / dt**2 + self.C / (2 * dt)
        self.a = self.M / dt**2 - self.C / (2 * dt)
        self.b = self.K - 2 * self.M / dt**2

        # Factorize once (efficient + stable)
        self.lu, self.piv = lu_factor(self.K_bar)

    def compute_solution(self, time, P):
        """
        Parameters
        ----------
        time : array_like (nt,)
        P : array_like (nt, ndof)
            External force history
        """

        time = np.asarray(time, dtype=float)
        P = np.asarray(P, dtype=float)

        nt = len(time)
        dt = self.dt
        ndof = self.ndof

        if P.shape != (nt, ndof):
            raise ValueError("P must have shape (nt, ndof)")

        u = np.zeros((nt, ndof))
        v = np.zeros((nt, ndof))
        acc = np.zeros((nt, ndof))

        # Initial conditions
        u[0, :] = self.u0
        v[0, :] = self.v0

        # Initial acceleration
        acc[0, :] = np.linalg.solve(self.M, P[0] - self.C @ v[0] - self.K @ u[0])

        # Compute u_{-1}
        u_minus1 = u[0] - dt * v[0] + 0.5 * dt**2 * acc[0]

        # Time stepping
        for i in range(nt - 1):

            u_prev = u_minus1 if i == 0 else u[i - 1]

            # Effective force
            P_hat = P[i] - self.a @ u_prev - self.b @ u[i]

            # Solve for next displacement
            # u[i + 1] = np.linalg.solve(self.K_bar, P_hat)
            u[i + 1] = lu_solve((self.lu, self.piv), P_hat)

            # Velocity and acceleration (central difference)
            v[i] = (u[i + 1] - u_prev) / (2 * dt)
            acc[i] = (u[i + 1] - 2 * u[i] + u_prev) / dt**2

        # Last step velocity & acceleration
        v[-1] = (u[-1] - u[-2]) / dt
        acc[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dt**2

        # Build DataFrame
        data = {"time": time}

        for i in range(ndof):
            data[f"u{i+1}"] = u[:, i]
        for i in range(ndof):
            data[f"v{i+1}"] = v[:, i]
        for i in range(ndof):
            data[f"a{i+1}"] = acc[:, i]

        return pd.DataFrame(data)
