import numpy as np


class ResponseSpectrumAnalysis:
    """
    Performs Response Spectrum Analysis (RSA) for a given MDOF system.

    This class first computes the modal properties (natural frequencies, mode
    shapes, participation factors) of the structure. It then uses a design
    response spectrum to determine the peak response (displacements, forces)
    for each mode.

    Attributes
    ----------
    mdf : MDOF_System
        The multi-degree-of-freedom system object to be analyzed.
    ji : float
        The modal damping ratio (e.g., 0.05 for 5%).
    n_modes : int
        The number of modes to consider in the analysis.
    gm : GroundMotion, optional
            A GroundMotion object used to generate the response spectrum internally.
            Provide this OR `Sd`. The default is None.
    Sd : array-like, optional
            An array of pre-computed spectral displacement values corresponding to
            the system's modal periods. The order must match the periods from
            lowest to highest. Provide this OR `gm`. The default is None.
    """

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
        """
        Calculates the peak modal displacements for each mode.

        Each column in the returned array represents the peak displacement
        vector for the corresponding mode.

        Parameters
        ----------
        Sd : array-like, optional
            Can be provided here if not given during initialization.
            The default is None.

        Returns
        -------
        np.ndarray
            An array of shape (ndof, n_modes) containing the peak modal
            displacements.
        """
        if Sd is not None:
            self.Sd = Sd
        u = np.zeros((self.mdf.ndof, self.n_modes))
        for i in range(self.n_modes):
            u[:, i] = self.lambda_[i] * self.phi[:, i] * self.Sd[i]
        self.u = u
        return u

    def modal_equivalent_forces(self):
        """
        Calculates the peak equivalent static forces for each mode.

        These forces, when applied statically to the structure, would produce
        the same peak modal displacements.

        Returns
        -------
        np.ndarray
            An array of shape (ndof, n_modes) containing the peak modal
            equivalent static forces.
        """
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
        """
        Calculates the base shear for each individual mode.

        Returns
        -------
        np.ndarray
            A 1D array of shape (n_modes,) where each element is the base
            shear for the corresponding mode.
        """
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
        """
        Combines peak modal responses to estimate the total response.

        Parameters
        ----------
        responses : np.ndarray
            An array where each column represents the peak response of a mode.
            For example, the output from `modal_displacements`.
        method : str, optional
            The modal combination rule to use. Currently supported: 'SRSS'.
            The default is "SRSS".

        Returns
        -------
        np.ndarray
            A 1D array of the combined response, with length `ndof`.
        """
        if method == "SRSS":
            combined_response = np.sqrt(np.sum(responses**2, axis=1))
        else:
            raise ValueError("Method not supported")
        return combined_response

    def plot_response_spectrum(self):
        pass
