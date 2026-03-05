import numpy as np
import warnings


def check_stability_newmark(dt, tn_min, beta, gamma):
    """
    Checks the stability condition for the Newmark-Beta method.

    Parameters
    ----------
    dt : float
        The time step of the integration.
    tn_min : float
        The smallest natural period of the system.
    beta : float
        The Newmark beta parameter.
    gamma : float
        The Newmark gamma parameter.
    """
    # Check if the method is unconditionally stable
    if gamma == 2 * beta:
        return

    # For conditionally stable methods, check the stability criterion
    if tn_min > 0:
        stability_limit = 1 / (np.pi * np.sqrt(2 * (gamma - 2 * beta)))
        dt_tn_ratio = dt / tn_min

        if dt_tn_ratio > stability_limit:
            warnings.warn(
                f"Newmark-Beta method may be unstable. The ratio of the time step to the "
                f"smallest natural period (dt/Tn_min) is {dt_tn_ratio:.4f}, which exceeds the "
                f"stability limit of {stability_limit:.4f}. Consider decreasing the time step (dt) "
                f"or using an unconditionally stable method (e.g., acc_type='average').",
                UserWarning,
            )


def check_stability_central_difference(dt, tn_min):
    """
    Checks the stability condition for the Central Difference method.

    Parameters
    ----------
    dt : float
        The time step of the integration.
    tn_min : float
        The smallest natural period of the system.
    """
    if tn_min > 0:
        stability_limit = 1 / np.pi
        dt_tn_ratio = dt / tn_min

        if dt_tn_ratio > stability_limit:
            warnings.warn(
                f"Central Difference method may be unstable. The time step to natural "
                f"period ratio (dt/Tn) of {dt_tn_ratio:.4f} exceeds the stability limit of "
                f"{stability_limit:.4f}. Consider decreasing the time step (dt).",
                UserWarning,
            )
