import numpy as np


def _shear_building_logic(masses, stiffnesses):
    """
    Creates the mass and stiffness matrices for a shear building model.

    A shear building is a structural model where the floors are assumed to be
    rigid and the columns are assumed to be inextensible. The lateral stiffness
    of each story is represented by a single spring.

    Parameters
    ----------
    masses : array-like
        A list or array of the masses of each floor.
    stiffnesses : array-like
        A list or array of the lateral stiffnesses of each story.

    Returns
    -------
    M : ndarray
        The mass matrix of the shear building.
    K : ndarray
        The stiffness matrix of the shear building.

    Raises
    ------
    ValueError
        If the length of `stiffnesses` does not match the length of `masses`.
    """
    masses = np.asarray(masses, dtype=float)
    stiffnesses = np.asarray(stiffnesses, dtype=float)
    n = len(masses)
    if len(stiffnesses) != n:
        raise ValueError("stiffnesses must match masses length.")
    M = np.diag(masses)
    K = np.zeros((n, n))
    for i in range(n):
        if i < n - 1:
            K[i, i] = stiffnesses[i] + stiffnesses[i + 1]
            K[i, i + 1] = -stiffnesses[i + 1]
            K[i + 1, i] = -stiffnesses[i + 1]
        else:
            K[i, i] = stiffnesses[i]
    return M, K
