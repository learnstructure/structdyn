import numpy as np


def free_vibration(sdf, t, u_0=0.1, u_dot_0=0):
    k, m = sdf.k, sdf.m
    w_n, w_d = sdf.w_n, sdf.w_d
    ji = sdf.ji

    u = np.exp(-ji * w_n * t) * (
        u_0 * np.cos(w_d * t) + (u_dot_0 + ji * w_n * u_0) / w_d * np.sin(w_d * t)
    )
    type = "Undamped free vibration" if ji == 0 else "Damped free vibration"

    return u, type
