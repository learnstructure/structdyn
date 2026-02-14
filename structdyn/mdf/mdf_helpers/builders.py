import numpy as np


def _shear_building_logic(masses, stiffnesses):
    """
    Create MDF system for a shear building model.
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
