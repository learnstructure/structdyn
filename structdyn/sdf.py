import numpy as np


class SDF:
    """Single Degree of Freedom (SDF) System for Structural Dynamics"""

    def __init__(self, m, k, ji=0):
        self.m = m  # mass in kg
        self.k = k  # stiffness in N/m
        self.ji = ji  # damping ratio
        self.w_n = np.sqrt(self.k / self.m)  # natural frequency
        self.w_d = self.w_n * np.sqrt(1 - self.ji**2)  # damped natural frequency
        self.t_n = 2 * np.pi / self.w_n  # natural time period
