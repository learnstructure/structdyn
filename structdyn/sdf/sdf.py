import numpy as np
from structdyn.utils.helpers import ElasticPerfectlyPlastic


class SDF:
    """Single Degree of Freedom (SDF) System for Structural Dynamics"""

    def __init__(self, m, k, ji=0, fd="linear", **fd_params):
        """Initialize SDF system."""
        self.m = m  # mass in kg
        self.k = k  # stiffness in N/m
        self.ji = ji  # damping ratio
        if self.ji >= 1:
            raise ValueError("Damping ratio must be less than 1")
        self.w_n = np.sqrt(self.k / self.m)  # natural frequency
        self.w_d = self.w_n * np.sqrt(1 - self.ji**2)  # damped natural frequency
        self.t_n = 2 * np.pi / self.w_n  # natural time period
        self.c = 2 * self.m * self.w_n * self.ji  # damping constant
        if fd == "linear":
            self.fd = None
        elif fd == "elastoplastic":
            self.fd = ElasticPerfectlyPlastic(**fd_params)
        else:
            raise ValueError("fd must be 'linear' or 'elastoplastic'")

    def find_response(self, time_steps, load_values, method="newmark_beta", **kwargs):
        """Compute the response of the SDF system using the specified numerical method.
                Parameters
        ----------
        time_steps : array-like
            Time discretization
        load_values : array-like
            External force history
        method : str
            'newmark_beta', 'central_difference', or 'interpolation'
        kwargs :
            Additional parameters passed to the numerical scheme
        ."""
        dt = time_steps[1] - time_steps[0]
        fs = kwargs.pop("fs", None)
        if method == "newmark_beta":
            from structdyn.sdf.numerical_methods.newmark_beta import NewmarkBeta

            solver = NewmarkBeta(self, dt=dt, **kwargs)
        elif method == "central_difference":
            from structdyn.sdf.numerical_methods.central_difference import (
                CentralDifference,
            )

            solver = CentralDifference(self, dt=dt, **kwargs)
            return solver.compute_solution(time_steps, load_values, fs=fs)
        elif method == "interpolation":
            from structdyn.sdf.numerical_methods.interpolation import Interpolation

            solver = Interpolation(self, dt=dt, **kwargs)
        else:
            raise ValueError(
                "Invalid method. Choose 'newmark_beta', 'central_difference', or 'interpolation'."
            )
        return solver.compute_solution(time_steps, load_values)
