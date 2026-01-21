import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files


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
    """Get data from El Centro earthquake"""
    # Read CSV file using importlib.resources for package data
    try:
        # Try using importlib.resources (modern approach)
        csv_file = files("structdyn").joinpath("ground_motions/elcentro_mod.csv")
        with csv_file.open('r') as f:
            el_centro = np.genfromtxt(f, delimiter=",")
    except (AttributeError, FileNotFoundError, TypeError):
        # Fallback to Path-based approach for local development
        csv_path = Path(__file__).parent.parent / "ground_motions" / "elcentro_mod.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find elcentro_mod.csv at {csv_path}")
        el_centro = np.genfromtxt(csv_path, delimiter=",")

    # Extract time steps and acceleration values
    time_steps = el_centro[:, 0]
    acc_values = el_centro[:, 2]

    return time_steps, acc_values


def plot_displacement(time_steps, displacement, text=None):
    plt.plot(time_steps, displacement, marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.title(text)
    plt.show()
