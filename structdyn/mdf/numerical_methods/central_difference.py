import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve


class CentralDifferenceMDF:
    """
    Central difference time integrator for linear MDOF systems.
    Can operate in physical coordinates or in modal coordinates (using pre‑computed mode shapes).
    """

    def __init__(self, mdf, dt, u0=None, v0=None, use_modal=False, n_modes=None):
        """
        Parameters
        ----------
        mdf : object
            Must have attributes .M, .C, .K (dense matrices) and .ndof.
            If modal data is present, it should be in .modal.omega and .modal.phi.
        dt : float
            Time step.
        u0, v0 : array_like, optional
            Initial displacement and velocity. If None, zero vectors are used.
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
        Integrate the equations of motion.

        Parameters
        ----------
        time : array_like (nt,)
            Discrete time instants.
        P : array_like (nt, ndof)
            External force history at each time instant.
        use_modal : bool, optional
            If True, perform integration in modal coordinates using pre‑computed mode shapes.
            Requires that self.mdf.modal.phi exists.
        n_modes : int, optional
            Number of modes to retain (only if use_modal=True). If None, all available modes are used.

        Returns
        -------
        pd.DataFrame
            Columns: 'time', u1..uN, v1..vN, a1..aN.
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
