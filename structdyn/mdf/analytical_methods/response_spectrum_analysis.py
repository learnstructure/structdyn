import numpy as np


class ResponseSpectrumAnalysis:
    def __init__(self, mdf, ji=0.05, n_modes=None, gm=None, Sd=None):
        self.mdf = mdf
        self.ji = ji
        if n_modes is None:
            self.n_modes = mdf.ndof
        else:
            self.n_modes = n_modes
        omega, phi = self.mdf.modal.modal_analysis(n_modes=self.n_modes)
        self.omega = omega
        self.phi = phi
        self.lambda_ = self.compute_modal_participation_factors()
        self.gm = gm
        self.Sd = Sd
        T = 2 * np.pi / self.omega
        self.Sd = self.compute_spectral_displacement(T)
        self.u = None
        self.f_eq = None

    def compute_spectral_displacement(self, T):
        if self.Sd is not None:
            return self.Sd
        elif self.gm is not None:
            from structdyn.sdf.response_spectrum import ResponseSpectrum

            rs = ResponseSpectrum(T, self.ji, self.gm)
            spectra = rs.compute()
            sorted_spectra = spectra.sort_values(by="T", ascending=False)
            return sorted_spectra["Sd"].values
        else:
            print(
                "Warning: No ground motion or spectral displacement provided. Provide later while calculating modal displacements."
            )

    def modal_diplacements(self, Sd=None):
        if Sd is not None:
            self.Sd = Sd
        u = np.zeros((self.mdf.ndof, self.n_modes))
        for i in range(self.n_modes):
            u[:, i] = self.lambda_[i] * self.phi[:, i] * self.Sd[i]
        self.u = u
        return u

    def modal_equivalent_forces(self):
        F = np.zeros((self.mdf.ndof, self.n_modes))
        for i in range(self.n_modes):
            F[:, i] = (
                self.lambda_[i]
                * self.mdf.M[i, i]
                * self.phi[:, i]
                * self.Sd[i]
                * self.omega[i] ** 2
            )
        self.f_eq = F
        return F

    def modal_base_shear(self):
        if self.f_eq is None:
            self.modal_equivalent_forces()
        base_shear = np.sum(self.f_eq, axis=0)
        return base_shear.reshape(1, -1)

    def compute_modal_participation_factors(self):
        lambda_list = []
        for i in range(self.n_modes):
            Mn = self.phi[:, i].T @ self.mdf.M @ self.phi[:, i]
            lambda_ = (self.phi[:, i].T @ self.mdf.M @ np.ones(self.mdf.ndof)) / Mn
            lambda_list.append(lambda_)
        return np.array(lambda_list)

    def combine_modal_responses(self, responses, method="SRSS"):
        if method == "SRSS":
            combined_response = np.sqrt(np.sum(responses**2, axis=1))
        else:
            raise ValueError("Method not supported")
        return combined_response

    def plot_response_spectrum(self):
        pass
