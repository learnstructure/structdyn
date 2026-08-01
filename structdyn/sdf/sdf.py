import numpy as np
from structdyn.utils.material_models import LinearElastic
from structdyn.ground_motions import GroundMotion
from structdyn.loads import LoadHistory
from structdyn.sdf.sdf_helpers.visualization import SDFVisualizer


class SDF:
    """Single Degree of Freedom (SDF) System for Structural Dynamics"""

    def __init__(self, m, k, ji=0, fd=None):
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
        fd : MaterialModel class, optional
            Force-deformation model, by default "LinearElastic".
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
        if fd is None or fd == "linear":
            self.fd = "linear"
        else:
            self.fd = fd
        self._plot = None  # Placeholder for the visualizer, will be set when accessed

    def find_response(self, load, method="newmark_beta", **kwargs):
        """
        Computes the dynamic response of the SDF system.

        Solves the equation of motion: m u'' + c u' + f_s(u) = p(t).

        Parameters
        ----------
        load : LoadHistory
            A :class:`~structdyn.loads.LoadHistory` bundling the time
            discretization and the force values to apply to the system.
        method : str, optional
            Numerical method for solving the equation of motion, by default
            "newmark_beta". Available methods: 'newmark_beta',
            'central_difference', 'interpolation'.
        **kwargs : dict, optional
            Additional parameters for the numerical solver.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the time history of the system's response
            (e.g., displacement, velocity, acceleration).

        Notes
        -----
        The previous ``find_response(time, load_values, ...)`` style — where
        the time vector and the force array were passed as separate
        arguments — has been removed. Build a :class:`~structdyn.loads.LoadHistory`
        instead and pass it as the single argument:

        >>> from structdyn.loads import LoadHistory
        >>> load = LoadHistory(time_steps=t, load_values=p)
        >>> sdf.find_response(load, method="newmark_beta")
        """
        if not isinstance(load, LoadHistory):
            raise TypeError(
                "find_response expects a LoadHistory as its first argument. "
                "Build one with structdyn.loads.LoadHistory(time_steps, "
                "load_values) and pass it directly. The legacy "
                "(time, load_values) call style has been removed."
            )

        time = load.time_steps
        dt = load.dt
        load_values = load.load_values

        solver_class = self._get_solver_class(method)
        solver = solver_class(self, dt=dt, **kwargs)

        return solver.compute_solution(time, load_values)

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
        # Effective force for an SDF system subject to ground acceleration:
        # p(t) = -m * a_g(t)
        p = -self.m * np.asarray(gm.acceleration, dtype=float)
        load = LoadHistory(np.asarray(gm.time, dtype=float), p)

        return self.find_response(load, method=method, **kwargs)

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

    @property
    def plot(self):
        """
        Provides access to visualization methods for the SDF system.

        Returns
        -------
        SDFVisualizer
            An instance of the SDFVisualizer class, which can be used to generate
            plots and animations of the system.
        """
        if self._plot is None:
            self._plot = SDFVisualizer(self)
        return self._plot
