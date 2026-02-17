import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve


def get_newmark_parameters(acc_type="linear"):
    if acc_type == "average":  # Constant average acceleration
        beta, gamma = 1 / 4, 1 / 2
    elif acc_type == "linear":  # Linear acceleration
        beta, gamma = 1 / 6, 1 / 2
    else:
        raise ValueError("acc_type must be 'average' or 'linear'")
    return beta, gamma


class NewmarkBetaMDF:
    """
    Solves the equation of motion for a linear MDOF system using the Newmark-beta method.

    This class implements the implicit, unconditionally stable Newmark-beta time integration
    algorithm. The integration can be performed either in the physical coordinates of the system
    or in modal coordinates.
    """

    def __init__(
        self,
        mdf,
        dt,
        u0=None,
        v0=None,
        acc_type="linear",
        use_modal=False,
        n_modes=None,
    ):
        """
        Initializes the NewmarkBetaMDF solver.

        Parameters
        ----------
        mdf : MDF
            An instance of the MDF class, representing the system to be analyzed.
        dt : float
            The time step for the integration.
        u0 : array-like, optional
            The initial displacement vector of shape (ndof,). Defaults to a zero vector.
        v0 : array-like, optional
            The initial velocity vector of shape (ndof,). Defaults to a zero vector.
        acc_type : str, optional
            The assumed acceleration variation ('average' or 'linear'). Defaults to "linear".
        use_modal : bool, optional
            If True, integration is performed in modal coordinates. Requires pre-computed mode shapes.
            Defaults to False.
        n_modes : int, optional
            The number of modes to use for modal integration. Active only if `use_modal` is True.
            Defaults to all available modes.
        """

        self.mdf = mdf
        self.dt = dt
        self.beta, self.gamma = get_newmark_parameters(acc_type)

        self.M = mdf.M
        self.C = mdf.C
        self.K = mdf.K
        self.ndof = mdf.ndof

        self.u0 = np.zeros(self.ndof) if u0 is None else np.asarray(u0, dtype=float)
        self.v0 = np.zeros(self.ndof) if v0 is None else np.asarray(v0, dtype=float)

        self.use_modal = use_modal
        self.n_modes = n_modes

        # Precompute physical Newmark constants (used when use_modal=False)
        self._compute_newmark_constants()

    # ------------------------------------------------------------
    # Precompute Newmark matrices (physical coordinates)
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

        # Factorize once (efficient & stable)
        self.lu, self.piv = lu_factor(self.K_hat)

    # ------------------------------------------------------------
    # Compute response
    # ------------------------------------------------------------
    def compute_solution(self, time, P):
        """
        Integrates the equations of motion over the given time and force history.

        Parameters
        ----------
        time : array-like
            An array of time points of shape (nt,).
        P : array-like
            The external force history as an array of shape (nt, ndof).

        Returns
        -------
        pd.DataFrame
            A DataFrame with the response history, including columns for 'time',
            displacements ('u1', 'u2', ...), velocities ('v1', 'v2', ...), and
            accelerations ('a1', 'a2', ...).

        Raises
        ------
        ValueError
            If `P` has an incorrect shape or if `use_modal` is True and the number
            of requested modes exceeds the available modes.
        """
        time = np.asarray(time, dtype=float)
        P = np.asarray(P, dtype=float)
        nt = len(time)
        dt = self.dt
        ndof = self.ndof
        beta = self.beta
        gamma = self.gamma

        if P.shape != (nt, ndof):
            raise ValueError("P must have shape (nt, ndof)")

        if self.use_modal:
            # --------------------------------------------------------
            # Integration in modal coordinates (Chopra Table 16.2.2)
            # --------------------------------------------------------
            n_modes = self.n_modes
            if self.mdf.modal.phi is None:
                self.modal_analysis(n_modes=n_modes)

            phi = self.mdf.modal.phi  # shape (ndof, nmodes_total)
            nmodes_total = phi.shape[1]

            if n_modes is None:
                n_modes = nmodes_total
            elif n_modes > nmodes_total:
                raise ValueError(
                    f"Requested {n_modes} modes, but only {nmodes_total} available"
                )
            phi = phi[:, :n_modes]  # truncate

            # Reduced matrices (modal)
            M_r = phi.T @ self.M @ phi  # (n_modes, n_modes)
            C_r = phi.T @ self.C @ phi
            K_r = phi.T @ self.K @ phi

            # Initial modal displacements and velocities
            # q0 = inv(M_r) * (phi.T @ M @ u0)   (Chopra step 1.1)
            q0 = np.linalg.solve(M_r, phi.T @ self.M @ self.u0)
            qdot0 = np.linalg.solve(M_r, phi.T @ self.M @ self.v0)

            # Initial modal acceleration from equilibrium (Chopra step 1.3)
            P0_modal = phi.T @ P[0]
            qddot0 = np.linalg.solve(M_r, P0_modal - C_r @ qdot0 - K_r @ q0)

            # Precompute Newmark constants for the reduced system
            a1_r = M_r / (beta * dt**2) + gamma * C_r / (beta * dt)
            a2_r = M_r / (beta * dt) + C_r * (gamma / beta - 1)
            a3_r = (1 / (2 * beta) - 1) * M_r + dt * C_r * (gamma / (2 * beta) - 1)
            K_hat_r = K_r + a1_r

            # Factorize reduced effective stiffness
            lu_r, piv_r = lu_factor(K_hat_r)

            # Allocate modal arrays
            q = np.zeros((nt, n_modes))
            qdot = np.zeros((nt, n_modes))
            qddot = np.zeros((nt, n_modes))

            q[0] = q0
            qdot[0] = qdot0
            qddot[0] = qddot0

            # Time stepping in modal coordinates
            for i in range(nt - 1):
                # Modal force at next time step
                P_next_modal = phi.T @ P[i + 1]

                # Effective load (Chopra Eq. 2.1)
                P_hat_r = P_next_modal + a1_r @ q[i] + a2_r @ qdot[i] + a3_r @ qddot[i]

                # Solve for next modal displacement (Eq. 2.2)
                q_next = lu_solve((lu_r, piv_r), P_hat_r)
                q[i + 1] = q_next

                # Velocity update (Eq. 2.3)
                qdot[i + 1] = (
                    gamma / (beta * dt) * (q_next - q[i])
                    + (1 - gamma / beta) * qdot[i]
                    + dt * (1 - gamma / (2 * beta)) * qddot[i]
                )

                # Acceleration update (Eq. 2.4)
                qddot[i + 1] = (
                    (q_next - q[i]) / (beta * dt**2)
                    - qdot[i] / (beta * dt)
                    - (1 / (2 * beta) - 1) * qddot[i]
                )

            # Transform back to physical coordinates
            u = q @ phi.T  # (nt, ndof)
            v = qdot @ phi.T
            a = qddot @ phi.T

        else:
            # --------------------------------------------------------
            # Original physical‑space integration
            # --------------------------------------------------------
            u = np.zeros((nt, ndof))
            v = np.zeros((nt, ndof))
            a = np.zeros((nt, ndof))

            # Initial conditions
            u[0] = self.u0
            v[0] = self.v0
            a[0] = np.linalg.solve(self.M, P[0] - self.C @ v[0] - self.K @ u[0])

            # Time stepping
            for i in range(nt - 1):
                # Effective load (Chopra Eq. 2.1)
                P_hat = P[i + 1] + self.a1 @ u[i] + self.a2 @ v[i] + self.a3 @ a[i]

                # Solve for displacement (Eq. 2.2)
                u[i + 1] = lu_solve((self.lu, self.piv), P_hat)

                # Velocity update (Eq. 2.3)
                v[i + 1] = (
                    self.gamma / (self.beta * self.dt) * (u[i + 1] - u[i])
                    + (1 - self.gamma / self.beta) * v[i]
                    + self.dt * (1 - self.gamma / (2 * self.beta)) * a[i]
                )

                # Acceleration update (Eq. 2.4)
                a[i + 1] = (
                    (u[i + 1] - u[i]) / (self.beta * self.dt**2)
                    - v[i] / (self.beta * self.dt)
                    - (1 / (2 * self.beta) - 1) * a[i]
                )

        # Build DataFrame
        data = {"time": time}
        for i in range(ndof):
            data[f"u{i+1}"] = u[:, i]
        for i in range(ndof):
            data[f"v{i+1}"] = v[:, i]
        for i in range(ndof):
            data[f"a{i+1}"] = a[:, i]

        return pd.DataFrame(data)
