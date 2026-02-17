import numpy as np
from .analytical_methods.modal_analysis import ModalAnalysis
from structdyn.ground_motions import GroundMotion


class MDF:
    """
    Linear Multi-Degree-of-Freedom (MDF) System

    Governing equation:
        M ü + C u̇ + K u = f(t)

    Parameters
    ----------
    M : (n, n) ndarray
        Mass matrix
    K : (n, n) ndarray
        Stiffness matrix
    C : (n, n) ndarray, optional
        Damping matrix (default: zero matrix)
    """

    def __init__(self, M, K, C=None):
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)

        if C is None:
            self.C = np.zeros_like(self.M)
        else:
            self.C = np.asarray(C, dtype=float)

        self.ndof = self.M.shape[0]

        self._validate()

        self.modal = ModalAnalysis(self)

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate(self):
        if self.M.shape != self.K.shape:
            raise ValueError("M and K must have the same dimensions.")

        if self.M.shape[0] != self.M.shape[1]:
            raise ValueError("M must be square.")

        if self.C.shape != self.M.shape:
            raise ValueError("C must have same dimensions as M.")

    def set_modal_damping(self, zeta, n_modes=None):
        omega, phi = self.modal.modal_analysis(n_modes=n_modes)

        zeta = np.asarray(zeta, dtype=float)

        n_modes = phi.shape[1]

        if len(zeta) != n_modes:
            raise ValueError("Length of zeta must equal number of modes used.")
        C = np.zeros_like(self.M)

        for i in range(n_modes):
            phi_i = phi[:, i].reshape(-1, 1)
            # Modal mass
            Mn = phi_i.T @ self.M @ phi_i
            coeff = 2 * zeta[i] * omega[i] / Mn
            # Modal contribution
            C += coeff * (self.M @ phi_i @ phi_i.T @ self.M)
        self.C = C
        return self.C

    # -------------------------------------------------
    # Shear Building Constructor
    # -------------------------------------------------
    @classmethod
    def from_shear_building(cls, masses, stiffnesses):
        """
        Create an MDOF shear building model.

        Parameters
        ----------
        masses : list or array
            Lumped masses at each floor from bottom to top
        stiffnesses : list or array
            Story stiffness values (length n) from bottom to top
        """
        from .mdf_helpers.builders import _shear_building_logic

        M, K = _shear_building_logic(masses, stiffnesses)
        return cls(M, K)

    def find_response(self, time, load, method="central_difference", **kwargs):
        from structdyn.mdf.numerical_methods.central_difference import (
            CentralDifferenceMDF,
        )
        from structdyn.mdf.numerical_methods.newmark_beta import NewmarkBetaMDF

        time = np.asarray(time)
        dt = time[1] - time[0]
        if not np.allclose(np.diff(time), dt):
            raise ValueError("Time vector must be uniformly spaced")

        if method == "newmark_beta":
            solver_class = NewmarkBetaMDF
        elif method == "central_difference":
            solver_class = CentralDifferenceMDF
        else:
            raise ValueError("method must be 'central_difference' or 'newmark_beta'")

        solver = solver_class(self, dt, **kwargs)
        return solver.compute_solution(time, load)

    def find_response_ground_motion(
        self, gm, inf_vec=None, method="central_difference", **kwargs
    ):
        if not isinstance(gm, GroundMotion):
            raise TypeError("gm must be a GroundMotion object")
        time = np.asarray(gm.time)
        ag = np.asarray(gm.acc_g) * 9.81  # convert to m/s²
        if inf_vec is None:
            inf_vec = np.ones(self.ndof)
        inf_vec = np.asarray(inf_vec)
        if inf_vec.shape != (self.ndof,):
            raise ValueError("inf_vec must have shape (ndof,)")

        # Compute effective inertia vector M r
        Mr = self.M @ inf_vec
        load = -ag[:, None] * Mr[None, :]
        return self.find_response(time, load, method=method, **kwargs)
