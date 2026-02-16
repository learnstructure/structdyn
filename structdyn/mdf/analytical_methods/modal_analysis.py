from scipy.linalg import eigh
import numpy as np
import pandas as pd
from structdyn.sdf.sdf import SDF
from structdyn.sdf.analytical_methods.analytical_response import AnalyticalResponse


class ModalAnalysis:
    def __init__(self, mdf):
        self.mdf = mdf
        self.omega = None
        self.phi = None

    def modal_analysis(self, n_modes=None, dof_normalize=None):
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

        # Sort modes by ascending frequency
        idx = np.argsort(omega)
        omega = omega[idx]
        eigenvecs = eigenvecs[:, idx]

        if n_modes is not None:
            if n_modes < 1 or n_modes > self.mdf.ndof:
                raise ValueError("n_modes must be between 1 and ndof")

            omega = omega[:n_modes]
            eigenvecs = eigenvecs[:, :n_modes]

        self.omega, self.phi = omega, eigenvecs
        if dof_normalize is not None:
            self.normalize_modes(dof=dof_normalize)

        return self.omega, self.phi

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
        n_modes = phi.shape[1]
        for i in range(n_modes):
            Mn = self.phi[:, i].T @ self.mdf.M @ self.phi[:, i]
            phi[:, i] /= np.sqrt(Mn)
        self.phi = phi
        return phi

    def modal_coordinates(self, u):
        if self.phi is None:
            self.modal_analysis()
        qn = []
        n_modes = self.phi.shape[1]
        for i in range(n_modes):
            phi_i = self.phi[:, i]
            Mn = phi_i.T @ self.mdf.M @ phi_i
            qn.append((phi_i.T @ self.mdf.M @ u) / Mn)
        self.qn = np.array(qn)
        return self.qn

    def get_Mn_Cn_Kn(self):
        if self.phi is None:
            self.modal_analysis()
        # self.normalize_modes()
        Mn_full = self.phi.T @ self.mdf.M @ self.phi
        Cn_full = self.phi.T @ self.mdf.C @ self.phi
        Kn_full = self.phi.T @ self.mdf.K @ self.phi
        self.Mn_full, self.Cn_full, self.Kn_full = Mn_full, Cn_full, Kn_full

    def free_vibration_response(self, u0, v0, time=None, n_modes=None):
        if self.phi is None or (n_modes is not None and self.phi.shape[1] != n_modes):
            self.modal_analysis(n_modes=n_modes)

        u0 = np.asarray(u0, dtype=float)
        v0 = np.asarray(v0, dtype=float)

        if time is None:
            time = np.arange(0, 10, 0.01)
        time = np.asarray(time, dtype=float)

        nt = len(time)
        ndof = self.mdf.ndof
        n_modes = self.phi.shape[1]
        u = np.zeros((ndof, nt))

        for i in range(n_modes):
            phi_n = self.phi[:, i]
            Mn = phi_n.T @ self.mdf.M @ phi_n
            Cn = phi_n.T @ self.mdf.C @ phi_n
            Kn = phi_n.T @ self.mdf.K @ phi_n
            qn0 = phi_n.T @ self.mdf.M @ u0 / Mn
            qn0_dot = phi_n.T @ self.mdf.M @ v0 / Mn
            sdf = SDF(Mn, Kn, Cn)
            analytical = AnalyticalResponse(sdf)

            qn_t = (analytical.free_vibration(qn0, qn0_dot, time))[
                ["displacement"]
            ].values.flatten()  # shape (nt,)

            u += np.outer(phi_n, qn_t)

        u = u.T  # transpose to (nt, ndof)
        # build dataframe
        columns = ["time"] + [f"u{i+1}" for i in range(ndof)]
        data = np.column_stack((time, u))

        df = pd.DataFrame(data, columns=columns)

        return df
