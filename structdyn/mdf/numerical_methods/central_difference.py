import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve


class CentralDifferenceMDF:
    """
    Solves the equation of motion for a linear MDOF system using the Central Difference method.

    This class implements the explicit, conditionally stable Central Difference time integration
    algorithm. The integration can be performed either in the physical coordinates of the system
    or in modal coordinates, which can be more efficient for systems where the response is
    dominated by a few modes.
    """

    def __init__(self, mdf, dt, u0=None, v0=None, use_modal=False, n_modes=None):
        """
        Initializes the CentralDifferenceMDF solver.

        Parameters
        ----------
        mdf : MDF
            An instance of the MDF class, representing the system to be analyzed.
            It must have .M, .C, .K attributes and a .modal attribute with .phi if using modal coordinates.
        dt : float
            The time step for the integration. The method is conditionally stable, and the time step
            must be smaller than a critical value (dt_crit = T_n / pi, where T_n is the smallest natural period).
        u0 : array-like, optional
            The initial displacement vector of shape (ndof,). If None, it is assumed to be zero.
            The default is None.
        v0 : array-like, optional
            The initial velocity vector of shape (ndof,). If None, it is assumed to be zero.
            The default is None.
        use_modal : bool, optional
            If True, the integration is performed in modal coordinates. This requires the `mdf` object
            to have its mode shapes computed. The default is False.
        n_modes : int, optional
            The number of modes to use for modal integration. If None, all available modes are used.
            This parameter is only active when `use_modal` is True. The default is None.
        """
        self.mdf = mdf
        self.dt = dt

        self.M = mdf.M
        self.C = mdf.C
        self.K = mdf.K
        self.ndof = mdf.ndof

        self.u0 = np.zeros(self.ndof) if u0 is None else np.asarray(u0, dtype=float)
        self.v0 = np.zeros(self.ndof) if v0 is None else np.asarray(v0, dtype=float)

        self.use_modal = use_modal
        self.n_modes = n_modes

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
            A DataFrame containing the response history. The columns include:
            - 'time': The time points.
            - 'u1', 'u2', ...: Displacement for each degree of freedom.
            - 'v1', 'v2', ...: Velocity for each degree of freedom.
            - 'a1', 'a2', ...: Acceleration for each degree of freedom.

        Raises
        ------
        ValueError
            If the shape of the force array `P` is not compatible with the time vector and the number of DOFs.
            If `use_modal` is True and the requested number of modes exceeds the available modes.
        """
        time = np.asarray(time, dtype=float)
        P = np.asarray(P, dtype=float)
        nt = len(time)
        dt = self.dt
        ndof = self.ndof

        if P.shape != (nt, ndof):
            raise ValueError("P must have shape (nt, ndof)")

        if self.use_modal:
            # ------------------------------------------------------------
            # Integration in modal coordinates
            # ------------------------------------------------------------
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
            q0 = np.linalg.solve(M_r, phi.T @ self.M @ self.u0)
            qdot0 = np.linalg.solve(M_r, phi.T @ self.M @ self.v0)

            # Initial modal acceleration
            P0_modal = phi.T @ P[0]
            qddot0 = np.linalg.solve(M_r, P0_modal - C_r @ qdot0 - K_r @ q0)

            # Compute q_{-1}
            q_minus1 = q0 - dt * qdot0 + 0.5 * dt**2 * qddot0

            # Effective matrices for the reduced system
            K_bar_r = M_r / dt**2 + C_r / (2 * dt)
            a_r = M_r / dt**2 - C_r / (2 * dt)
            b_r = K_r - 2 * M_r / dt**2

            # Factorize effective stiffness (constant)
            lu_r, piv_r = lu_factor(K_bar_r)

            # Arrays to store modal results
            q = np.zeros((nt, n_modes))
            qdot = np.zeros((nt, n_modes))
            qddot = np.zeros((nt, n_modes))

            q[0] = q0
            # qdot[0] and qddot[0] will be computed later from the recurrence,
            # but we keep the initial values for reference if needed.
            # The loop below computes them at the same time as u.

            # Time stepping in modal coordinates
            for i in range(nt - 1):
                # Current and previous modal displacements
                if i == 0:
                    q_prev = q_minus1
                else:
                    q_prev = q[i - 1]

                # Modal force at current time
                P_i_modal = phi.T @ P[i]

                # Effective load
                P_hat_r = P_i_modal - a_r @ q_prev - b_r @ q[i]

                # Solve for next displacement
                q_next = lu_solve((lu_r, piv_r), P_hat_r)
                q[i + 1] = q_next

                # Compute velocity and acceleration at current time (central difference)
                qdot[i] = (q_next - q_prev) / (2 * dt)
                qddot[i] = (q_next - 2 * q[i] + q_prev) / dt**2

            # Handle last time step (backward differences)
            if nt >= 2:
                qdot[-1] = (q[-1] - q[-2]) / dt
            if nt >= 3:
                qddot[-1] = (q[-1] - 2 * q[-2] + q[-3]) / dt**2
            else:
                qddot[-1] = qddot[-2]  # fallback

            # Transform back to physical coordinates
            u = q @ phi.T  # (nt, ndof)
            v = qdot @ phi.T
            a = qddot @ phi.T

        else:
            # ------------------------------------------------------------
            # Original physical‑space integration
            # ------------------------------------------------------------
            # Pre‑compute matrices used in the recurrence
            K_bar = self.M / dt**2 + self.C / (2 * dt)
            self.a = self.M / dt**2 - self.C / (2 * dt)
            self.b = self.K - 2 * self.M / dt**2

            # Factorize effective stiffness (constant)
            lu, piv = lu_factor(K_bar)

            # Initial acceleration
            a0 = np.linalg.solve(self.M, P[0] - self.C @ self.v0 - self.K @ self.u0)

            # Compute u_{-1}
            u_minus1 = self.u0 - dt * self.v0 + 0.5 * dt**2 * a0

            u = np.zeros((nt, ndof))
            v = np.zeros((nt, ndof))
            a = np.zeros((nt, ndof))

            u[0] = self.u0
            v[0] = self.v0
            a[0] = a0

            # Time stepping
            for i in range(nt - 1):
                u_prev = u_minus1 if i == 0 else u[i - 1]

                P_hat = P[i] - self.a @ u_prev - self.b @ u[i]
                u_next = lu_solve((lu, piv), P_hat)
                u[i + 1] = u_next

                v[i] = (u_next - u_prev) / (2 * dt)
                a[i] = (u_next - 2 * u[i] + u_prev) / dt**2

            # Last step
            if nt >= 2:
                v[-1] = (u[-1] - u[-2]) / dt
            if nt >= 3:
                a[-1] = (u[-1] - 2 * u[-2] + u[-3]) / dt**2
            else:
                a[-1] = a[-2]

        # Build DataFrame
        data = {"time": time}
        for i in range(ndof):
            data[f"u{i+1}"] = u[:, i]
        for i in range(ndof):
            data[f"v{i+1}"] = v[:, i]
        for i in range(ndof):
            data[f"a{i+1}"] = a[:, i]

        return pd.DataFrame(data)
