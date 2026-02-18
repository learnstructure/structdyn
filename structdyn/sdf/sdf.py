import numpy as np
from structdyn.utils.helpers import ElasticPerfectlyPlastic
from structdyn.ground_motions import GroundMotion


class SDF:
    """Single Degree of Freedom (SDF) System for Structural Dynamics"""

    def __init__(self, m, k, ji=0, fd="linear", **fd_params):
        """
        Initializes an SDF system.

        Parameters
        ----------
        m : float
            Mass of the system in kilograms (kg).
        k : float
            Stiffness of the system in Newtons per meter (N/m).
        ji : float, optional
            Damping ratio (dimensionless), by default 0.
            Must be between 0 and 1.
        fd : str, optional
            Force-deformation model, by default "linear".
            Can be 'linear' or 'elastoplastic'.
        **fd_params : dict, optional
            Additional parameters for the force-deformation model.
            For 'elastoplastic', this would include 'uy' (yield displacement) and 'fy' (yield force).
        """
        self.m = m  # mass
        self.k = k  # stiffness
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
        """
        Computes the dynamic response of the SDF system.

        Solves the equation of motion: m u'' + c u' + f_s(u) = p(t).

        Parameters
        ----------
        time : array-like
            Array of time points.
        load : array-like
            Array of generalized force values at each time point.
        method : str, optional
            Numerical method for solving the equation of motion, by default "newmark_beta".
            Available methods: 'newmark_beta', 'central_difference', 'interpolation'.
        **kwargs : dict, optional
            Additional parameters for the numerical solver.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time history of the system's response
            (e.g., displacement, velocity, acceleration).
        """
        time = np.asarray(time)
        dt = time[1] - time[0]
        if not np.allclose(np.diff(time), dt):
            raise ValueError("Time vector must be uniformly spaced")

        solver_class = self._get_solver_class(method)
        solver = solver_class(self, dt=dt, **kwargs)

        return solver.compute_solution(time, load)

    def find_response_ground_motion(self, gm, method="newmark_beta", **kwargs):
        """
        Computes the response of the SDF system to ground motion.

        Solves the equation of motion for a base-excited system subjected to ground motion.

        Parameters
        ----------
        gm : GroundMotion
            A GroundMotion object representing the ground motion record.
        method : str, optional
            Numerical method to use, by default "newmark_beta".
        **kwargs : dict, optional
            Additional parameters for the numerical solver.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time history of the system's response.
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
