from scipy.linalg import eigh
import numpy as np
from structdyn.sdf.sdf import SDF
from structdyn.sdf.analytical_methods.analytical_response import AnalyticalResponse


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

    def get_Mn_Cn_Kn(self):
        self.normalize_modes()
        Mn_full = self.phi.T @ self.mdf.M @ self.phi
        Cn_full = self.phi.T @ self.mdf.C @ self.phi
        Kn_full = self.phi.T @ self.mdf.K @ self.phi
        self.Mn_full, self.Cn_full, self.Kn_full = Mn_full, Cn_full, Kn_full

    def free_vibration_response(self, u0, v0, time=None):
        u0 = np.asarray(u0, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        self.get_Mn_Cn_Kn()
        if time is None:
            time = np.arange(0, 10, 0.01)
        time = np.asarray(time)
        nt = len(time)
        ndof = self.mdf.ndof
        u = np.zeros((ndof, nt))
        for i in range(ndof):
            phi_n = self.phi[:, i]
            qn0 = phi_n.T @ self.mdf.M @ u0 / self.Mn_full[i, i]
            qn0_dot = phi_n.T @ self.mdf.M @ v0 / self.Mn_full[i, i]
            sdf = SDF(self.Mn_full[i, i], self.Kn_full[i, i], self.Cn_full[i, i])
            analytical = AnalyticalResponse(sdf)

            qn_t = (analytical.free_vibration(qn0, qn0_dot, time))[
                ["displacement"]
            ].values
            u += np.outer(phi_n, qn_t)
        return u
