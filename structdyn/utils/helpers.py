import matplotlib.pyplot as plt
import numpy as np
from importlib import resources
import pandas as pd


class ElasticPerfectlyPlastic:
    def __init__(self, uy=0.02, fy=36000):
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


def fs_elastoplastic(uy=0.02, fy=36000):
    """Get resisting force for given u.
    uy is Yield deformation and fy is yield force"""

    def get_fs_elastoplastic(
        u,
        u_last,
        fs_last=0,
    ):
        fs = fs_last + fy / uy * (u - u_last)
        fs = fs if abs(fs) < fy else fy if fs > fy else -fy
        return fs

    return get_fs_elastoplastic


def elcentro_chopra(header=None):
    """Get data from El Centro earthquake as in Chopra's book 5th edn"""
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
