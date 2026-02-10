import numpy as np
from structdyn.utils.helpers import ElasticPerfectlyPlastic
from structdyn.ground_motions import GroundMotion


class SDF:
    """Single Degree of Freedom (SDF) System for Structural Dynamics"""

    def __init__(self, m, k, ji=0, fd="linear", **fd_params):
        """Initialize SDF system."""
        self.m = m  # mass in kg
        self.k = k  # stiffness in N/m
        self.ji = ji  # damping ratio
        if not (0 <= ji < 1):
            raise ValueError("Damping ratio must be between 0 and 1")
        self.w_n = np.sqrt(self.k / self.m)  # natural frequency
        self.w_d = self.w_n * np.sqrt(1 - self.ji**2)  # damped natural frequency
        self.t_n = 2 * np.pi / self.w_n  # natural time period
        self.c = 2 * self.m * self.w_n * self.ji  # damping constant
        if fd == "linear":
            self.fd = "linear"
        elif fd == "elastoplastic":
            self.fd = ElasticPerfectlyPlastic(**fd_params)
        else:
            raise ValueError("fd must be 'linear' or 'elastoplastic'")

    def find_response(self, time, load, method="newmark_beta", **kwargs):
        """Solve m u'' + c u' + f_s(u) = p(t)
                Parameters
        ----------
        time : array-like
            Time discretization
        load : array-like
            Generalized force history p(t)
            (for ground motion: p = -m * u_g_ddot)
        method : str
            'newmark_beta', 'central_difference', or 'interpolation'
        kwargs :
            Additional parameters passed to the numerical scheme
        ."""
        time = np.asarray(time)
        dt = time[1] - time[0]
        if not np.allclose(np.diff(time), dt):
            raise ValueError("Time vector must be uniformly spaced")
        # fs = kwargs.pop("fs", None)

        solver_class = self._get_solver_class(method)
        solver = solver_class(self, dt=dt, **kwargs)

        return solver.compute_solution(time, load)

    def find_response_ground_motion(self, gm, method="newmark_beta", **kwargs):
        """
        Solve base-excited system using ground motion.
        """
        if not isinstance(gm, GroundMotion):
            raise TypeError("gm must be a GroundMotion object")
        time = gm.time
        load = -self.m * gm.acc_g * 9.81

        return self.find_response(time, load, method=method, **kwargs)

    def _get_solver_class(self, method):
        if method == "newmark_beta":
            from structdyn.sdf.numerical_methods.newmark_beta import NewmarkBeta

            return NewmarkBeta

        if method == "central_difference":
            from structdyn.sdf.numerical_methods.central_difference import (
                CentralDifference,
            )

            return CentralDifference

        if method == "interpolation":
            from structdyn.sdf.numerical_methods.interpolation import Interpolation

            return Interpolation

        raise ValueError(
            f"Invalid method '{method}'. "
            "Choose 'newmark_beta', 'central_difference', or 'interpolation'."
        )
