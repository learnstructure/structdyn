import numpy as np
import pandas as pd


class NewmarkBetaNonLinear:
    """
    Solves the equations of motion for a non-linear MDOF system using the Newmark-Beta method.

    This class implements an implicit, step-by-step time integration algorithm.
    For non-linear systems, an iterative Newton-Raphson procedure is used
    within each time step to satisfy equilibrium.

    Attributes
    ----------
    system : object
        A system object that represents the structure. It must provide the
        mass matrix (M), damping matrix (C), number of DOFs (ndof), and
        methods to compute the resisting force and tangent stiffness.
    dt : float
        The time step for the integration.
    beta : float
        The Newmark-Beta parameter 'beta'. Defaults to 1/4 (average acceleration).
    gamma : float
        The Newmark-Beta parameter 'gamma'. Defaults to 1/2 (average acceleration).
    """

    def __init__(self, system, dt, beta=1 / 4, gamma=1 / 2, tol=1e-6, max_iter=50):

        self.system = system
        self.dt = dt
        self.beta = beta
        self.gamma = gamma
        # Pre‑compute coefficient matrices (these depend only on M, C, dt)
        self._compute_coeff_matrices()
        self.tol = tol
        self.max_iter = max_iter

    def _compute_coeff_matrices(self):
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        M = self.system.M
        C = self.system.C
        self.A1 = (1 / (beta * dt**2)) * M + (gamma / (beta * dt)) * C
        self.A2 = (1 / (beta * dt)) * M + (gamma / beta - 1) * C
        self.A3 = (1 / (2 * beta) - 1) * M + dt * (gamma / (2 * beta) - 1) * C

    def compute_solution(self, time, p, tol=None, max_iter=None):
        """
        Performs the step-by-step non-linear time history analysis.

        This method iterates through each time step, using a Newton-Raphson
        scheme to solve for the displacements that satisfy the dynamic
        equilibrium equation.

        Parameters
        ----------
        time : np.ndarray
            A 1D array of time points for the analysis.
        p : np.ndarray
            An array of external forces, `p(t)`, with shape `(len(time), ndof)`.
        tol : float, optional
            The convergence tolerance for the norm of the residual force vector
            in the Newton-Raphson iteration. The default is 1e-6.
        max_iter : int, optional
            The maximum number of iterations allowed per time step for the
            Newton-Raphson solver. The default is 20.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the full time history of the response.
            Columns include 'time', displacements ('u1', 'u2', ...),
            velocities ('v1', 'v2', ...), accelerations ('a1', 'a2', ...),
            and internal resisting forces ('fs1', 'fs2', ...).
        """
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        n = len(time)
        dt = self.dt
        ndof = self.system.ndof

        # Initial conditions
        u = np.zeros((n, ndof))
        v = np.zeros((n, ndof))
        a = np.zeros((n, ndof))
        fs_hist = np.zeros((n, ndof))

        u[0] = self.system.u0 if hasattr(self.system, "u0") else np.zeros(ndof)
        v[0] = self.system.v0 if hasattr(self.system, "v0") else np.zeros(ndof)

        # Initial state determination
        Fs0, Kt0 = self.system.assemble_resisting_force_and_tangent(u[0], v[0], dt)
        # Commit initial state (if nonlinear)
        if hasattr(self.system, "commit_elements"):
            self.system.commit_elements(u[0])
        a[0] = np.linalg.solve(self.system.M, p[0] - self.system.C @ v[0] - Fs0)
        fs_hist[0] = Fs0

        # Time stepping
        for i in range(n - 1):
            u_trial = u[i].copy()
            # We need Fs_trial and Kt_trial; start with values from previous step
            Fs_trial, Kt_trial = self.system.assemble_resisting_force_and_tangent(
                u_trial, v[i], dt
            )

            p_hat = p[i + 1] + self.A1 @ u[i] + self.A2 @ v[i] + self.system.M @ a[i]

            converged = False
            for it in range(max_iter):
                # Newmark kinematics
                a_trial = (
                    (u_trial - u[i]) / (self.beta * dt**2)
                    - v[i] / (self.beta * dt)
                    - (1 / (2 * self.beta) - 1) * a[i]
                )

                v_trial = v[i] + dt * ((1 - self.gamma) * a[i] + self.gamma * a_trial)

                # True residual
                R_hat = (
                    p[i + 1]
                    - Fs_trial
                    - self.system.C @ v_trial
                    - self.system.M @ a_trial
                )
                # R_hat = p_hat - Fs_trial - self.A1 @ u_trial

                if np.linalg.norm(R_hat) < tol:
                    converged = True
                    break

                K_hat = Kt_trial + self.A1
                du = np.linalg.solve(K_hat, R_hat)
                u_trial += du

                # Compute trial velocity (needed for rate‑dependent materials)
                v_trial = (
                    (self.gamma / (self.beta * dt)) * (u_trial - u[i])
                    + (1 - self.gamma / self.beta) * v[i]
                    + dt * (1 - self.gamma / (2 * self.beta)) * a[i]
                )

                # Update element forces and tangents
                Fs_trial, Kt_trial = self.system.assemble_resisting_force_and_tangent(
                    u_trial, v_trial, dt
                )

            if not converged:
                raise RuntimeError(f"No convergence at step {i+1}")

            # Commit element states (if nonlinear)
            if hasattr(self.system, "commit_elements"):
                self.system.commit_elements(u_trial)

            u_next = u_trial
            v_next = (
                (self.gamma / (self.beta * dt)) * (u_next - u[i])
                + (1 - self.gamma / self.beta) * v[i]
                + dt * (1 - self.gamma / (2 * self.beta)) * a[i]
            )
            a_next = (
                (u_next - u[i]) / (self.beta * dt**2)
                - v[i] / (self.beta * dt)
                - (1 / (2 * self.beta) - 1) * a[i]
            )

            u[i + 1] = u_next
            v[i + 1] = v_next
            a[i + 1] = a_next
            fs_hist[i + 1] = Fs_trial

        # Build DataFrame
        data = {"time": time}
        for i in range(ndof):
            data[f"u{i+1}"] = u[:, i]
        for i in range(ndof):
            data[f"v{i+1}"] = v[:, i]
        for i in range(ndof):
            data[f"a{i+1}"] = a[:, i]
        for i in range(ndof):
            data[f"fs{i+1}"] = fs_hist[:, i]

        return pd.DataFrame(data)
