import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


def elcentro():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    file_path = ROOT_DIR / "elcentro_mod.csv"
    el_centro = np.genfromtxt(file_path, delimiter=",")
    time_steps = el_centro[:, 0]
    acc_values = el_centro[:, 2]
    return time_steps, acc_values


def plot_displacement(time_steps, displacement, text=None):
    plt.plot(time_steps, displacement, marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.title(text)
    plt.show()
