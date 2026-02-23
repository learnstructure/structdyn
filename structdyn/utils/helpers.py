import matplotlib.pyplot as plt
import numpy as np
from importlib import resources
import pandas as pd


class ElasticPerfectlyPlastic:
    """
    Represents an elastic-perfectly plastic force-deformation model.
    The model tracks the plastic deformation to correctly represent the unloading
    and reloading paths.
    """

    def __init__(self, uy=0.02, fy=36000):
        """
        Initializes the elastic-perfectly plastic material model.

        Parameters
        ----------
        uy : float, optional
            Yield deformation, the displacement at which the material yields,
            by default 0.02.
        fy : float, optional
            Yield force, the force at which the material starts to yield,
            by default 36000.
        """
        self.uy = uy
        self.fy = fy
        self.k0 = fy / uy

        # Committed (history) state
        self.u_p = 0.0

    def trial_response(self, u, v=None, dt=None):
        """
        Calculate the trial force and tangent stiffness for a given displacement.

        This method computes the resisting force and stiffness based on a trial
        displacement `u` without updating the model's history.

        Parameters
        ----------
        u : float
            The trial displacement.

        Returns
        -------
        fs_trial : float
            The trial resisting force.
        kt_trial : float
            The trial tangent stiffness.
        is_yielding : bool
            True if the trial state is in the plastic region, False otherwise.
        """
        fs_trial = self.k0 * (u - self.u_p)

        if abs(fs_trial) <= self.fy:
            return fs_trial, self.k0, False
        else:
            return self.fy * np.sign(fs_trial), 0.0, True

    def commit_state(self, u):
        """
        Update the plastic deformation history after a solution has converged.

        Parameters
        ----------
        u : float
            The converged displacement for the current step.
        """
        fs = self.k0 * (u - self.u_p)

        if abs(fs) > self.fy:
            self.u_p = u - (self.fy / self.k0) * np.sign(fs)


class BoucWen:
    """
    Bouc‑Wen smooth hysteretic model.

    Parameters
    ----------
    k0 : float
        Initial stiffness.
    alpha : float
        Ratio of post‑yield to initial stiffness (0 ≤ α ≤ 1).
    A : float, optional
        Hysteretic amplitude control (default = 1.0).
    beta : float, optional
        Hysteretic shape parameter (default = 0.5).
    gamma : float, optional
        Hysteretic shape parameter (default = 0.5).
    n : float, optional
        Exponent controlling the sharpness of the transition (n ≥ 1, default = 1).
    """

    def __init__(self, k0, alpha, A=1.0, beta=0.5, gamma=0.5, n=1):
        self.k0 = k0
        self.alpha = alpha
        self.A = A
        self.beta = beta
        self.gamma = gamma
        self.n = n

        # Internal state
        self.z = 0.0  # committed hysteretic displacement
        self.z_trial = 0.0  # trial value (used in predict/commit)

    def trial_response(self, u, v, dt):
        """
        Compute trial force, tangent stiffness and a boolean flag.

        Parameters
        ----------
        u : float
            Current trial displacement.
        v : float
            Current trial velocity (du/dt).
        dt : float
            Time step (must be positive).

        Returns
        -------
        fs_trial : float
            Trial restoring force.
        kt_trial : float
            Trial tangent stiffness (consistent with the direction of v).
        flag : bool
            Always False (included for interface compatibility).
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        z = self.z

        # ---- Bouc‑Wen evolution equation ----
        # dz/dt
        zdot = (
            self.A * v
            - self.beta * abs(v) * (abs(z) ** (self.n - 1)) * z
            - self.gamma * v * (abs(z) ** self.n)
        )

        # Explicit Euler integration
        z_trial = z + zdot * dt
        self.z_trial = z_trial

        # Restoring force
        fs_trial = self.k0 * (self.alpha * u + (1 - self.alpha) * z_trial)

        # ---- Consistent tangent stiffness ----
        # dz/du = A - β·sign(v)·|z|ⁿ⁻¹·z - γ·|z|ⁿ
        # (using the current committed z and the direction of v)
        if abs(v) > 1e-12:
            sign_v = np.sign(v)
        else:
            sign_v = 0.0  # velocity zero – direction undefined; use elastic part only

        dz_du = (
            self.A
            - self.beta * sign_v * (abs(z) ** (self.n - 1)) * z
            - self.gamma * (abs(z) ** self.n)
        )

        # Total tangent stiffness
        kt_trial = self.k0 * (self.alpha + (1 - self.alpha) * dz_du)

        # The boolean flag is not used in the Bouc‑Wen model, return False
        return fs_trial, kt_trial, False

    def commit_state(self, u):
        """
        Commit the trial hysteretic displacement as the new state.
        (The argument u is not needed but kept for interface consistency.)
        """
        self.z = self.z_trial

    def reset(self):
        """Reset the internal state to zero."""
        self.z = 0.0
        self.z_trial = 0.0


def elcentro_chopra(header=0):
    """
    Loads the El Centro ground motion data from Chopra's book.

    This function reads a CSV file containing the ground motion record for the
    1940 El Centro earthquake, as provided in Chopra's "Dynamics of Structures".

    Parameters
    ----------
    header : int, list of int, None, default 0
        Row number(s) to use as the column names, and the start of the data.
        Passed directly to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the ground motion data.
    """
    with resources.open_text(
        "structdyn.ground_motions.data", "elcentro_chopra.csv"
    ) as f:
        return pd.read_csv(f, header=header)


def plot_displacement(time_steps, displacement, text=None):
    """
    Plots the displacement time history.

    Parameters
    ----------
    time_steps : array-like
        The time vector.
    displacement : array-like
        The displacement time history.
    text : str, optional
        The title of the plot, by default None.
    """
    plt.plot(time_steps, displacement, marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.title(text)
    plt.show()
