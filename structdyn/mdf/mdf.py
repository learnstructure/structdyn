import numpy as np
from scipy.linalg import eigh
from .analytical_methods.modal_analysis import ModalAnalysis


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

    # -------------------------------------------------
    # Rayleigh Damping
    # -------------------------------------------------

    # def set_rayleigh_damping(self, zeta1, zeta2, mode1=0, mode2=1):
    #     """
    #     Compute Rayleigh damping matrix:
    #         C = α M + β K

    #     Using two target damping ratios.
    #     """

    #     if self._omega is None:
    #         self.modal_analysis()

    #     w1 = self._omega[mode1]
    #     w2 = self._omega[mode2]

    #     A = np.array([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]])

    #     b = np.array([zeta1, zeta2])

    #     alpha, beta = np.linalg.solve(A, b)

    #     self.C = alpha * self.M + beta * self.K

    #     return alpha, beta

    # # -------------------------------------------------
    # # Participation Factors
    # # -------------------------------------------------

    # def participation_factors(self):
    #     """
    #     Compute modal participation factors Γ_r
    #     for base excitation (assuming influence vector = 1).
    #     """

    #     if self._phi is None:
    #         self.mass_normalize_modes()

    #     ones = np.ones(self.ndof)

    #     gamma = []

    #     for i in range(self.ndof):
    #         phi_r = self._phi[:, i]
    #         num = phi_r.T @ self.M @ ones
    #         den = phi_r.T @ self.M @ phi_r
    #         gamma.append(num / den)

    #     return np.array(gamma)

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
