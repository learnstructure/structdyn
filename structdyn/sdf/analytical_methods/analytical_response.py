import numpy as np
import pandas as pd


class AnalyticalResponse:
    """
    Analytical solutions for an existing SDF system.

    Supports:
    - Free vibration (underdamped)
    - Harmonic excitation (sine & cosine)
    """

    def __init__(self, sdf):
        self.sdf = sdf

        self.m = sdf.m
        self.k = sdf.k
        self.ji = sdf.ji
        self.c = sdf.c
        self.w_n = sdf.w_n
        self.t_n = sdf.t_n
        if not (0 <= self.ji < 1):
            raise ValueError(
                "Analytical solution currently supports underdamped systems only (0 ≤ ζ < 1)."
            )

        self.w_d = self.w_n * np.sqrt(1 - self.ji**2)

    # ---------------------------------------------------------
    # 1️⃣ Free Vibration (Underdamped)
    # ---------------------------------------------------------
    def free_vibration(self, u0, v0=0, time=None):
        if time is None:
            time = np.arange(0, 10 * self.t_n, self.t_n / 100)
        t = np.asarray(time)

        exp_term = np.exp(-self.ji * self.w_n * t)

        u = exp_term * (
            u0 * np.cos(self.w_d * t)
            + (v0 + self.ji * self.w_n * u0) / self.w_d * np.sin(self.w_d * t)
        )

        # Velocity (exact derivative is long — numerical is acceptable)
        v = np.gradient(u, t)

        # Acceleration from equation of motion
        a = (-self.c * v - self.k * u) / self.m

        return pd.DataFrame(
            {"time": t, "displacement": u, "velocity": v, "acceleration": a}
        )

    # ---------------------------------------------------------
    # 2️⃣ Harmonic Forcing (General Form)
    #     p(t) = p0 * sin(ωt)  OR  p0 * cos(ωt)
    # ---------------------------------------------------------
    def harmonic_response(self, p0, w, u0=0.0, v0=0.0, time=None, excitation="sine"):
        """
        excitation: "sine" or "cosine"
        """
        if time is None:
            time = np.arange(0, 10 * self.t_n, self.t_n / 100)
        t = np.asarray(time)

        r = w / self.w_n
        denom = (1 - r**2) ** 2 + (2 * self.ji * r) ** 2

        if excitation == "sine":

            C = (p0 / self.k) * (1 - r**2) / denom
            D = (p0 / self.k) * (-2 * self.ji * r) / denom

            A = u0 - D
            B = (v0 + self.ji * self.w_n * A - C * w) / self.w_d

            forcing_part = C * np.sin(w * t) + D * np.cos(w * t)

        elif excitation == "cosine":

            C = (p0 / self.k) * (2 * self.ji * r) / denom
            D = (p0 / self.k) * (1 - r**2) / denom

            A = u0 - D
            B = (v0 + self.ji * self.w_n * A + C * w) / self.w_d

            forcing_part = C * np.sin(w * t) + D * np.cos(w * t)

        else:
            raise ValueError("excitation must be 'sine' or 'cosine'")

        transient = np.exp(-self.ji * self.w_n * t) * (
            A * np.cos(self.w_d * t) + B * np.sin(self.w_d * t)
        )

        u = transient + forcing_part

        v = np.gradient(u, t)
        a = (
            -self.c * v
            - self.k * u
            + p0 * (np.sin(w * t) if excitation == "sine" else np.cos(w * t))
        ) / self.m

        return pd.DataFrame(
            {"time": t, "displacement": u, "velocity": v, "acceleration": a}
        )
