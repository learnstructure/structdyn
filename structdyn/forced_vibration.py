import numpy as np


def harmonic_vibration(sdf, t, type="sin", p0=10000, w_ratio=0.5):
    k, m = sdf.k, sdf.m
    w_n, w_d = sdf.w_n, sdf.w_d
    ji = sdf.ji

    w = w_n * w_ratio
    k1 = (1 - w_ratio**2) ** 2 + (2 * ji * w_ratio) ** 2
    k2 = 2 * ji * w_ratio
    k3 = 1 - w_ratio**2

    if type == "sin":
        C = p0 * k3 / (k * k1)
        D = -p0 * k2 / (k * k1)
        text = "Steady-state Response: Sine Loading"
    else:
        C = p0 * k2 / (k * k1)
        D = p0 * k3 / (k * k1)
        text = "Steady-state Response: Cosine Loading"

    u = C * np.sin(w * t) + D * np.cos(w * t)
    return u, text
