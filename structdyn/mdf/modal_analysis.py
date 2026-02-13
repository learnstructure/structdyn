from scipy.linalg import eigh
import numpy as np


class ModalAnalysis:
    def __init__(self, mdf):
        self.mdf = mdf
        self.omega = None
        self.phi = None

    def modal_analysis(self):
        """
        Solves: K φ = λ M φ

        Returns
        -------
        omega : ndarray
            Natural circular frequencies (rad/s)
        phi : ndarray
            Mode shape matrix (columns are modes)
        """

        eigenvals, eigenvecs = eigh(self.mdf.K, self.mdf.M)

        # Remove small negative numerical values
        eigenvals = np.real(eigenvals)
        eigenvals[eigenvals < 0] = 0.0

        omega = np.sqrt(eigenvals)

        self.omega, self.phi = omega, eigenvecs
        return omega, eigenvecs

    # def modal_mass(self):
    #     modal_mass = []
    #     for i in range(self.mdf.ndof):
    #         modal_mass.append(self.phi[:, i].T @ self.mdf.M @ self.phi[:, i])
    #     self.Mn = modal_mass

    def normalize_modes(self, dof=-1):
        if self.phi is None:
            self.modal_analysis()
        phi = self.phi / self.phi[dof]
        self.phi = phi
        return phi

    def mass_normalize_modes(self):
        """
        Mass-normalize mode shapes such that:
            φᵀ M φ = I
        """
        if self.phi is None:
            self.modal_analysis()
        phi = self.phi.copy()
        for i in range(self.mdf.ndof):
            Mn = self.phi[:, i].T @ self.mdf.M @ self.phi[:, i]
            phi[:, i] /= np.sqrt(Mn)
        self.phi = phi
        return phi

    def modal_coordinates(self, u):
        qn = []
        for i in range(self.mdf.ndof):
            Mn = self.phi[:, i].T @ self.mdf.M @ self.phi[:, i]
            qn.append(((self.phi[:, i].T @ self.mdf.M @ u) / Mn).item())
        self.qn = qn
        return qn
