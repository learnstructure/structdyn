import numpy as np
import pandas as pd


class NewmarkBetaNonLinear:
    def __init__(self, system, dt, beta=1 / 4, gamma=1 / 2):
        self.system = system
        self.dt = dt
        self.beta = beta
        self.gamma = gamma
        # Pre‑compute coefficient matrices (these depend only on M, C, dt)
        self._compute_coeff_matrices()

    def _compute_coeff_matrices(self):
        dt = self.dt
        beta = self.beta
        gamma = self.gamma
        M = self.system.M
        C = self.system.C
        self.A1 = (1 / (beta * dt**2)) * M + (gamma / (beta * dt)) * C
        self.A2 = (1 / (beta * dt)) * M + (gamma / beta - 1) * C
        self.A3 = (1 / (2 * beta) - 1) * M + dt * (gamma / (2 * beta) - 1) * C

    def compute_solution(self, time, p, tol=1e-6, max_iter=20):
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

            p_hat = p[i + 1] + self.A1 @ u[i] + self.A2 @ v[i] + self.A3 @ a[i]

            converged = False
            for it in range(max_iter):
                R_hat = p_hat - Fs_trial - self.A1 @ u_trial
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
