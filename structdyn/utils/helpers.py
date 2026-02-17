import matplotlib.pyplot as plt
import numpy as np
from importlib import resources
import pandas as pd


class ElasticPerfectlyPlastic:
    """
    Represents an elastic-perfectly plastic force-deformation model.

    This class models a material that behaves elastically until it reaches a
    specified yield force, after which it deforms plastically with no increase
    in force. It is commonly used in nonlinear structural analysis to represent
    the behavior of ductile components.

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

    def trial_response(self, u):
        """
        Calculate the trial force and tangent stiffness for a given displacement.

        This method computes the resisting force and stiffness based on a trial
        displacement `u` without updating the model's history. This is a key
        step in iterative solution strategies like the Newton-Raphson method.

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

        Once an iterative solution for a time step is complete and a final
        displacement `u` is accepted, this method updates the accumulated
        plastic deformation `u_p`.

        Parameters
        ----------
        u : float
            The converged displacement for the current step.
        """
        fs = self.k0 * (u - self.u_p)

        if abs(fs) > self.fy:
            self.u_p = u - (self.fy / self.k0) * np.sign(fs)


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
