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
        Trial force and tangent stiffness
        (does NOT modify history)
        """
        fs_trial = self.k0 * (u - self.u_p)

        if abs(fs_trial) <= self.fy:
            return fs_trial, self.k0, False
        else:
            return self.fy * np.sign(fs_trial), 0.0, True

    def commit_state(self, u):
        """
        Update history AFTER convergence
        """
        fs = self.k0 * (u - self.u_p)

        if abs(fs) > self.fy:
            self.u_p = u - (self.fy / self.k0) * np.sign(fs)


# def fs_elastoplastic(uy=0.02, fy=36000):
#     """Get resisting force for given u.
#     uy is Yield deformation and fy is yield force"""

#     def get_fs_elastoplastic(
#         u,
#         u_last,
#         fs_last=0,
#     ):
#         fs = fs_last + fy / uy * (u - u_last)
#         fs = fs if abs(fs) < fy else fy if fs > fy else -fy
#         return fs

#     return get_fs_elastoplastic


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
    plt.plot(time_steps, displacement, marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.title(text)
    plt.show()
