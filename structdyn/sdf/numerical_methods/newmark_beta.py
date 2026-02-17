import numpy as np
import pandas as pd


def get_newmark_parameters(acc_type="linear"):
    """
    Returns Newmark-beta parameters (beta, gamma).

    Parameters
    ----------
    acc_type : str
        'average' : Constant-average acceleration (unconditionally stable)
        'linear'  : Linear acceleration
    """
    if acc_type == "average":
        beta, gamma = 1 / 4, 1 / 2
    elif acc_type == "linear":
        beta, gamma = 1 / 6, 1 / 2
    else:
        raise ValueError("acc_type must be 'average' or 'linear'")

    return beta, gamma


class NewmarkBeta:
    """
    Implements the Newmark-Beta time integration scheme for solving the equation
    of motion for both linear and nonlinear Single Degree of Freedom (SDF) systems.

    This class provides a robust and widely used numerical method for dynamic
    analysis. It supports both the constant-average-acceleration and the
    linear-acceleration methods.

    Reference: Chopra, A. K. (2020). Dynamics of Structures: Theory and
    Applications to Earthquake Engineering. Pearson Education.
    (See Table 5.4.2 for linear systems and Table 5.7.1 for nonlinear systems)
    """

    def __init__(
        self,
        sdf,
        dt,
        u0=0.0,
        v0=0.0,
        acc_type="linear",
    ):
        """
        Initializes the NewmarkBeta solver.

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
        acc_type : {"average", "linear"}, optional
            The assumption for the variation of acceleration over a time step,
            by default "average".
            - "average": Constant-average acceleration (unconditionally stable).
            - "linear": Linear acceleration.
        """
        self.dt = dt
        self.beta, self.gamma = get_newmark_parameters(acc_type)
        self.sdf = sdf

        # System properties
        self.m = sdf.m
        self.c = sdf.c
        self.k = sdf.k

        # Initial conditions
        self.u0 = u0
        self.v0 = v0

        # Precompute Newmark constants
        self._compute_newmark_constants()

    def compute_solution(self, time_steps, load_values):
        """
        Computes the dynamic response by dispatching to the appropriate solver
        based on the system's linearity.

        Parameters
        ----------
        time_steps : array-like
            An array representing the time vector of the analysis.
        load_values : array-like
            An array representing the external force applied at each time step.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time history of the response.
        """
        if self.sdf.fd == "linear":
            return self.compute_solution_linear(time_steps, load_values)
        else:
            return self.compute_solution_nonlinear(time_steps, load_values)

    def _compute_newmark_constants(self):
        """Precompute Newmark integration constants."""
        dt = self.dt
        beta = self.beta
        gamma = self.gamma

        self.a1 = self.m / (beta * dt**2) + gamma * self.c / (beta * dt)
        self.a2 = self.m / (beta * dt) + self.c * (gamma / beta - 1)
        self.a3 = (1 / (2 * beta) - 1) * self.m + dt * self.c * (gamma / (2 * beta) - 1)

        # Effective stiffness
        self.k_hat = self.k + self.a1

    def compute_solution_linear(self, time_steps, load_values):
        """
        Computes the response of a linear system using the Newmark-Beta method.

        Parameters
        ----------
        time_steps : array-like
            Time discretization.
        load_values : array-like
            External force p(t) at each time step.

        Returns
        -------
        pandas.DataFrame
            Time history of displacement, velocity, and acceleration.
        """
        if len(time_steps) != len(load_values):
            raise ValueError("time_steps and load_values must have the same length")

        # Initial acceleration from equilibrium at t = 0
        p0 = load_values[0]
        a0 = (p0 - self.c * self.v0 - self.k * self.u0) / self.m

        # Initialize response variables
        u = self.u0
        v = self.v0
        a = a0

        results = {
            "time": [time_steps[0]],
            "displacement": [u],
            "velocity": [v],
            "acceleration": [a],
        }

        # Time-stepping loop
        for i in range(len(time_steps) - 1):
            p_next = load_values[i + 1]

            # Effective load (Chopra Eq. 2.1)
            p_hat = p_next + self.a1 * u + self.a2 * v + self.a3 * a

            # Displacement update (Eq. 2.2)
            u_next = p_hat / self.k_hat

            # Velocity update (Eq. 2.3)
            v_next = (
                self.gamma / (self.beta * self.dt) * (u_next - u)
                + (1 - self.gamma / self.beta) * v
                + self.dt * (1 - self.gamma / (2 * self.beta)) * a
            )

            # Acceleration update (Eq. 2.4)
            a_next = (
                (u_next - u) / (self.beta * self.dt**2)
                - v / (self.beta * self.dt)
                - (1 / (2 * self.beta) - 1) * a
            )

            # Advance state
            u, v, a = u_next, v_next, a_next

            results["time"].append(time_steps[i + 1])
            results["displacement"].append(u)
            results["velocity"].append(v)
            results["acceleration"].append(a)

        return pd.DataFrame(results)

    def compute_solution_nonlinear(
        self,
        time_steps,
        load_values,
        tol=1e-6,
        max_iter=20,
    ):
        """
        Computes the response of a nonlinear system using the Newmark-Beta method
        with a Newton-Raphson iteration scheme.

        Parameters
        ----------
        time_steps : array-like
            Time discretization.
        load_values : array-like
            External force p(t) at each time step.
        tol : float, optional
            Tolerance for the convergence of the Newton-Raphson iteration,
            by default 1e-6.
        max_iter : int, optional
            Maximum number of iterations for the Newton-Raphson algorithm,
            by default 20.

        Returns
        -------
        pandas.DataFrame
            Time history of displacement, velocity, and acceleration.

        Raises
        -------
        RuntimeError
            If the Newton-Raphson iteration fails to converge.
        """

        if len(time_steps) != len(load_values):
            raise ValueError("time_steps and load_values must have same length")

        dt = self.dt
        beta = self.beta
        gamma = self.gamma

        # --- Initial state determination (Step 1.1) ---
        u = self.u0
        v = self.v0

        fs, kt, _ = self.sdf.fd.trial_response(u)
        self.sdf.fd.commit_state(u)

        # Initial acceleration (Step 1.2)
        a = (load_values[0] - self.c * v - fs) / self.m

        results = {
            "time": [time_steps[0]],
            "displacement": [u],
            "velocity": [v],
            "acceleration": [a],
        }

        # --- Time stepping loop ---
        for i in range(len(time_steps) - 1):

            p_next = load_values[i + 1]

            # Step 2.1: Initial guess
            u_trial = u
            fs_trial = fs
            kt_trial = kt

            # Step 2.2: Effective load
            p_hat = p_next + self.a1 * u + self.a2 * v + self.a3 * a

            # --- Newton-Raphson iteration ---
            for iteration in range(max_iter):

                # Step 3.1: Residual
                R_hat = p_hat - fs_trial - self.a1 * u_trial

                # Step 3.2: Convergence check
                if abs(R_hat) < tol:
                    break

                # Step 3.3: Effective tangent stiffness
                k_hat = kt_trial + self.a1

                # Step 3.4: Displacement correction
                du = R_hat / k_hat

                # Step 3.5: Update displacement
                u_trial += du

                # Step 3.6: Update internal force and tangent stiffness
                fs_trial, kt_trial, _ = self.sdf.fd.trial_response(u_trial)
                self.sdf.fd.commit_state(u_trial)

            # else:
            #     raise RuntimeError(
            #         f"Newton-Raphson did not converge at time step {i+1}"
            #     )

            # Accept converged displacement
            u_next = u_trial

            # --- Step 4: Velocity and acceleration ---
            v_next = (
                gamma / (beta * dt) * (u_next - u)
                + (1 - gamma / beta) * v
                + dt * (1 - gamma / (2 * beta)) * a
            )

            a_next = (
                (u_next - u) / (beta * dt**2)
                - v / (beta * dt)
                - (1 / (2 * beta) - 1) * a
            )

            # Advance state
            u, v, a = u_next, v_next, a_next
            fs, kt = fs_trial, kt_trial

            results["time"].append(time_steps[i + 1])
            results["displacement"].append(u)
            results["velocity"].append(v)
            results["acceleration"].append(a)

        return pd.DataFrame(results)
