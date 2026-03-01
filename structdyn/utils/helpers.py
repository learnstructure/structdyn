import matplotlib.pyplot as plt
import numpy as np
from importlib import resources
import pandas as pd


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
