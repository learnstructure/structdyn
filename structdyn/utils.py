import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
import io


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
    url = "https://raw.githubusercontent.com/learnstructure/structdyn/main/elcentro_mod.csv"

    # Download the file
    response = requests.get(url)
    if response.status_code != 200:
        raise FileNotFoundError("Could not download elcentro_mod.csv from GitHub.")

    # Read CSV data using numpy
    el_centro = np.genfromtxt(io.StringIO(response.text), delimiter=",")

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
