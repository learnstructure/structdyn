import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve


def get_newmark_parameters(acc_type="average"):
    if acc_type == "average":  # Constant average acceleration
        beta, gamma = 1 / 4, 1 / 2
    elif acc_type == "linear":  # Linear acceleration
        beta, gamma = 1 / 6, 1 / 2
    else:
        raise ValueError("acc_type must be 'average' or 'linear'")
    return beta, gamma


class NewmarkBetaMDF:

    def __init__(
        self,
        mdf,
        dt,
        u0=None,
        v0=None,
        acc_type="average",
    ):

        self.mdf = mdf
        self.dt = dt
        self.beta, self.gamma = get_newmark_parameters(acc_type)

        self.M = mdf.M
        self.C = mdf.C
        self.K = mdf.K
        self.ndof = mdf.ndof

        self.u0 = np.zeros(self.ndof) if u0 is None else np.asarray(u0, dtype=float)
        self.v0 = np.zeros(self.ndof) if v0 is None else np.asarray(v0, dtype=float)

        self._compute_newmark_constants()

    # ------------------------------------------------------------
    # Precompute Newmark matrices (Chopra Table 16.2.2)
    # ------------------------------------------------------------
    def _compute_newmark_constants(self):

        dt = self.dt
        beta = self.beta
        gamma = self.gamma

        self.a1 = self.M / (beta * dt**2) + gamma * self.C / (beta * dt)
        self.a2 = self.M / (beta * dt) + self.C * (gamma / beta - 1)
        self.a3 = (1 / (2 * beta) - 1) * self.M + dt * self.C * (gamma / (2 * beta) - 1)

        # Effective stiffness
        self.K_hat = self.K + self.a1

        # 🔥 Factorize once (efficient & stable)
        self.lu, self.piv = lu_factor(self.K_hat)

    # ------------------------------------------------------------
    # Compute response
    # ------------------------------------------------------------
    def compute_solution(self, time, P):

        time = np.asarray(time, dtype=float)
        P = np.asarray(P, dtype=float)

        nt = len(time)
        ndof = self.ndof

        if P.shape != (nt, ndof):
            raise ValueError("P must have shape (nt, ndof)")

        u = np.zeros((nt, ndof))
        v = np.zeros((nt, ndof))
        acc = np.zeros((nt, ndof))

        # Initial conditions
        u[0] = self.u0
        v[0] = self.v0

        # Initial acceleration from equilibrium
        acc[0] = np.linalg.solve(self.M, P[0] - self.C @ v[0] - self.K @ u[0])

        # Time stepping
        for i in range(nt - 1):

            # Effective load (Chopra Eq. 2.1)
            P_hat = P[i + 1] + self.a1 @ u[i] + self.a2 @ v[i] + self.a3 @ acc[i]

            # Solve for displacement (Eq. 2.2)
            u[i + 1] = lu_solve((self.lu, self.piv), P_hat)

            # Velocity update (Eq. 2.3)
            v[i + 1] = (
                self.gamma / (self.beta * self.dt) * (u[i + 1] - u[i])
                + (1 - self.gamma / self.beta) * v[i]
                + self.dt * (1 - self.gamma / (2 * self.beta)) * acc[i]
            )

            # Acceleration update (Eq. 2.4)
            acc[i + 1] = (
                (u[i + 1] - u[i]) / (self.beta * self.dt**2)
                - v[i] / (self.beta * self.dt)
                - (1 / (2 * self.beta) - 1) * acc[i]
            )

        # Build DataFrame
        data = {"time": time}

        for i in range(ndof):
            data[f"u{i+1}"] = u[:, i]
        for i in range(ndof):
            data[f"v{i+1}"] = v[:, i]
        for i in range(ndof):
            data[f"a{i+1}"] = acc[:, i]

        return pd.DataFrame(data)
