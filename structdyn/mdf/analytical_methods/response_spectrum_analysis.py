"""
Response Spectrum Analysis module.

Performs linear dynamic Response Spectrum Analysis (RSA) for Multi-Degree-of-Freedom
(MDF) structural systems.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ResponseSpectrumAnalysis:
    """
    Performs Response Spectrum Analysis (RSA) for a given MDOF system.

    This class computes the modal properties (natural frequencies, mode
    shapes, modal participation factors, modal effective mass, and modal mass
    participation ratios) of the structure. It then evaluates peak modal responses
    (displacements, base shear, element forces) from a response spectrum, and combines
    them using modal combination rules (SRSS, CQC, ABS).

    Parameters
    ----------
    mdf : MDF
        The multi-degree-of-freedom system object to be analyzed.
    ji : float or array-like, optional
        Modal damping ratio(s) (e.g., 0.05 for 5%). Default is 0.05.
    n_modes : int, optional
        Number of modes to consider in the analysis. If None, considers all DOFs.
    gm : GroundMotion or str, optional
        A GroundMotion object, built-in event name (e.g. 'elcentro', 'northridge',
        'loma_prieta', 'san_fernando', 'elcentro_chopra'), or composite event name
        specifying component explicitly (e.g. 'northridge:hor2').
    gm_component : str, optional
        Component to load when `gm` is a string ('hor1', 'hor2', 'up').
        If None, explicitly matched to the excitation direction:
        - direction='x' -> 'hor1' (horizontal 1 / X)
        - direction='y' -> 'hor2' (horizontal 2 / Y)
        - direction='z' -> 'up' (vertical / Z)
    Sd : array-like, optional
        Pre-computed spectral displacement values corresponding to modal periods.
    Sa : array-like, optional
        Pre-computed spectral acceleration values corresponding to modal periods.
    spectrum_curve : callable or tuple or dict, optional
        Continuous response spectrum curve Sa(T):
        - callable: f(T) -> Sa
        - tuple: (periods, sa_values)
        - dict: {"T": [...], "Sa": [...]}
    params : Any, optional
        SeismicParameters object from struct_load providing design response spectrum.
    direction : str, optional
        Ground motion excitation direction: 'x', 'y', or 'z'. Default is 'x'.
    inf_vec : array-like, optional
        Custom influence vector representing displacements under unit base translation.
        If omitted, derived automatically from `direction` and `mdf.structure` if available,
        or defaulted to all ones (for 1D shear buildings).
    scale_factor : float, optional
        Multiplier applied to the spectral values (e.g. gravity acceleration g = 9.80665 m/s²
        or 386.09 in/s² if Sa is specified in g). Default is 1.0.
    """

    def __init__(
        self,
        mdf: Any,
        ji: Optional[Union[float, List[float], np.ndarray]] = None,
        n_modes: Optional[int] = None,
        gm: Any = None,
        gm_component: Optional[str] = None,
        Sd: Optional[Union[List[float], np.ndarray]] = None,
        Sa: Optional[Union[List[float], np.ndarray]] = None,
        spectrum_curve: Optional[Union[Callable[[float], float], Tuple[Any, Any], Dict[str, Any]]] = None,
        params: Optional[Any] = None,
        direction: str = "x",
        inf_vec: Optional[Union[List[float], np.ndarray]] = None,
        scale_factor: float = 1.0,
    ):
        self.mdf = mdf
        self.scale_factor = float(scale_factor)
        self.direction = str(direction).lower()
        self.params = params
        self.spectrum_curve = spectrum_curve
        self.gm_component = gm_component

        if isinstance(gm, str):
            from structdyn.ground_motions.ground_motion import GroundMotion

            comp = gm_component
            if comp is None:
                if ":" in gm:
                    gm_name, comp = gm.split(":", 1)
                elif "/" in gm:
                    gm_name, comp = gm.split("/", 1)
                else:
                    gm_name = gm
                    direction_comp_map = {"x": "hor1", "y": "hor2", "z": "up"}
                    comp = direction_comp_map.get(self.direction, "hor1")
            else:
                gm_name = gm

            self.gm = GroundMotion.from_event(
                gm_name.strip(), component=comp.strip(), scale_factor=self.scale_factor
            )
        else:
            self.gm = gm

        if n_modes is None:
            self.n_modes = mdf.ndof
        else:
            self.n_modes = min(n_modes, mdf.ndof)

        if ji is None:
            ji = getattr(params, "damping_ratio", 0.05) if params is not None else 0.05

        if isinstance(ji, (list, tuple, np.ndarray)):
            self.ji = np.asarray(ji, dtype=float)
            if len(self.ji) < self.n_modes:
                self.ji = np.pad(self.ji, (0, self.n_modes - len(self.ji)), constant_values=self.ji[-1])
        else:
            self.ji = np.full(self.n_modes, float(ji), dtype=float)

        omega, phi = self.mdf.modal.modal_analysis(n_modes=self.n_modes)
        self.omega = omega
        self.phi = phi
        self.n_modes = len(omega)
        self.T = 2.0 * np.pi / self.omega

        if len(self.ji) != self.n_modes:
            if len(self.ji) > self.n_modes:
                self.ji = self.ji[: self.n_modes]
            else:
                self.ji = np.pad(self.ji, (0, self.n_modes - len(self.ji)), constant_values=self.ji[-1])

        self.inf_vec = self._resolve_influence_vector(self.direction, inf_vec)
        self.lambda_ = self.compute_modal_participation_factors(self.inf_vec)
        self.gamma = self.lambda_
        self.effective_mass = self.compute_effective_mass()
        self.mass_participation_ratio = self.compute_mass_participation_ratio()

        self.Sd = np.asarray(Sd, dtype=float) if Sd is not None else None
        self.Sa = np.asarray(Sa, dtype=float) if Sa is not None else None
        self.Sd = self.compute_spectral_displacement(self.T)

        self.u = None
        self.f_eq = None

    def _resolve_influence_vector(
        self,
        direction: str,
        inf_vec: Optional[Union[List[float], np.ndarray]],
    ) -> np.ndarray:
        if inf_vec is not None:
            vec = np.asarray(inf_vec, dtype=float)
            if vec.shape[0] != self.mdf.ndof:
                raise ValueError(
                    f"inf_vec length ({len(vec)}) must match mdf.ndof ({self.mdf.ndof})"
                )
            return vec

        struct = getattr(self.mdf, "structure", None)
        if struct is not None and hasattr(struct, "nodes") and hasattr(struct, "free_dofs"):
            first_node = next(iter(struct.nodes.values()))
            num_node_dofs = len(first_node.dofs)

            if num_node_dofs == 3:  # 2D frame: [ux, uy, rz]
                if direction == "x":
                    target_offset = 0
                elif direction == "y":
                    target_offset = 1
                elif direction == "z":
                    raise ValueError("2D structure does not have translational degrees of freedom along Z.")
                else:
                    raise ValueError(f"Unknown direction '{direction}'. Expected 'x' or 'y'.")
            elif num_node_dofs == 6:  # 3D frame: [ux, uy, uz, rx, ry, rz]
                if direction == "x":
                    target_offset = 0
                elif direction == "y":
                    target_offset = 1
                elif direction == "z":
                    target_offset = 2
                else:
                    raise ValueError(f"Unknown direction '{direction}'. Expected 'x', 'y', or 'z'.")
            else:
                return np.ones(self.mdf.ndof, dtype=float)

            r_global = np.zeros(struct.neq, dtype=float)
            for node in struct.nodes.values():
                r_global[node.dofs[target_offset]] = 1.0

            return r_global[struct.free_dofs]

        # 1D shear building or generic MDF fallback
        return np.ones(self.mdf.ndof, dtype=float)

    @staticmethod
    def _evaluate_spectrum_curve(
        T: np.ndarray,
        curve: Union[Callable[[float], float], Tuple[Any, Any], Dict[str, Any]],
    ) -> np.ndarray:
        """
        Evaluates a user-supplied continuous response spectrum curve Sa(T).
        """
        if callable(curve):
            return np.array([float(curve(float(t))) for t in T], dtype=float)
        elif isinstance(curve, (tuple, list)) and len(curve) == 2:
            t_pts = np.asarray(curve[0], dtype=float)
            sa_pts = np.asarray(curve[1], dtype=float)
            return np.interp(T, t_pts, sa_pts)
        elif isinstance(curve, dict):
            t_key = "T" if "T" in curve else "period" if "period" in curve else "periods"
            sa_key = "Sa" if "Sa" in curve else "sa" if "sa" in curve else "acc"
            t_pts = np.asarray(curve[t_key], dtype=float)
            sa_pts = np.asarray(curve[sa_key], dtype=float)
            return np.interp(T, t_pts, sa_pts)
        else:
            raise ValueError(
                f"Unsupported spectrum_curve format: {type(curve)}. "
                "Expected callable f(T), tuple (T_array, Sa_array), or dict {'T': ..., 'Sa': ...}."
            )

    def compute_spectral_displacement(self, T: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes or assigns spectral displacements Sd for each modal period.
        """
        if self.Sd is not None:
            sd_arr = np.asarray(self.Sd, dtype=float) * self.scale_factor
            if self.Sa is None:
                self.Sa = sd_arr * (self.omega ** 2)
            return sd_arr

        elif self.Sa is not None:
            sa_arr = np.asarray(self.Sa, dtype=float) * self.scale_factor
            self.Sa = sa_arr
            return sa_arr / (self.omega ** 2)

        elif self.params is not None:
            from struct_load import evaluate_design_spectral_acceleration

            sa_vals = evaluate_design_spectral_acceleration(self.params, T)
            sa_arr = np.asarray(sa_vals, dtype=float) * self.scale_factor
            self.Sa = sa_arr
            return sa_arr / (self.omega ** 2)

        elif self.spectrum_curve is not None:
            sa_arr = self._evaluate_spectrum_curve(T, self.spectrum_curve) * self.scale_factor
            self.Sa = sa_arr
            return sa_arr / (self.omega ** 2)

        elif self.gm is not None:
            from structdyn.sdf.response_spectrum import ResponseSpectrum

            damp = float(self.ji[0])
            rs = ResponseSpectrum(T, damp, self.gm)
            spectra = rs.compute()
            sd_arr = np.interp(T, spectra["T"].values, spectra["Sd"].values)
            # If gm had scale_factor == 1.0 (raw g values), convert to physical units with self.scale_factor
            if self.scale_factor != 1.0 and getattr(self.gm, "scale_factor", 1.0) == 1.0:
                sd_arr = sd_arr * self.scale_factor
            self.Sa = sd_arr * (self.omega ** 2)
            return sd_arr

        else:
            return None

    def compute_modal_participation_factors(
        self, inf_vec: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculates modal participation factors Gamma_n = (phi_n^T * M * inf_vec) / (phi_n^T * M * phi_n).
        """
        if inf_vec is None:
            inf_vec = self.inf_vec
        lambda_list = []
        for i in range(self.n_modes):
            phi_i = self.phi[:, i]
            Mn = phi_i.T @ self.mdf.M @ phi_i
            Ln = phi_i.T @ self.mdf.M @ inf_vec
            lambda_list.append(Ln / Mn)
        return np.array(lambda_list)

    def compute_effective_mass(self) -> np.ndarray:
        """
        Calculates modal effective mass M_n* for each mode.
        M_n* = (phi_n^T * M * inf_vec)^2 / (phi_n^T * M * phi_n) = Gamma_n^2 * M_n
        """
        eff_mass = []
        for i in range(self.n_modes):
            phi_i = self.phi[:, i]
            Mn = phi_i.T @ self.mdf.M @ phi_i
            eff_mass.append((self.lambda_[i] ** 2) * Mn)
        return np.array(eff_mass)

    def compute_mass_participation_ratio(self) -> np.ndarray:
        """
        Calculates modal mass participation ratio (MMPR) for each mode.
        MMPR_n = M_n* / M_total, where M_total = inf_vec^T * M * inf_vec
        """
        total_mass = self.inf_vec.T @ self.mdf.M @ self.inf_vec
        if abs(total_mass) < 1e-12:
            return np.zeros(self.n_modes)
        return self.effective_mass / total_mass

    def modal_displacements(self, Sd: Optional[Union[List[float], np.ndarray]] = None) -> np.ndarray:
        """
        Calculates peak modal displacements for each mode.

        Parameters
        ----------
        Sd : array-like, optional
            Spectral displacement array if not provided during initialization.

        Returns
        -------
        np.ndarray
            Array of shape (ndof, n_modes) containing peak modal displacements.
        """
        if Sd is not None:
            self.Sd = np.asarray(Sd, dtype=float)
        if self.Sd is None:
            raise ValueError(
                "Spectral displacement (Sd) is not defined. Provide ground motion, Sd, Sa, or spectrum curve."
            )
        u = np.zeros((self.mdf.ndof, self.n_modes))
        for i in range(self.n_modes):
            u[:, i] = self.lambda_[i] * self.phi[:, i] * self.Sd[i]
        self.u = u
        return u


    def modal_equivalent_forces(self) -> np.ndarray:
        """
        Calculates peak equivalent static lateral forces for each mode.

        F[:, i] = Gamma_i * (M @ phi[:, i]) * omega_i^2 * Sd[i]

        Returns
        -------
        np.ndarray
            Array of shape (ndof, n_modes) containing peak modal equivalent static forces.
        """
        if self.u is None:
            self.modal_displacements()
        F = np.zeros((self.mdf.ndof, self.n_modes))
        for i in range(self.n_modes):
            F[:, i] = (
                self.lambda_[i]
                * (self.mdf.M @ self.phi[:, i])
                * (self.omega[i] ** 2)
                * self.Sd[i]
            )
        self.f_eq = F
        return F

    def modal_base_shear(self) -> np.ndarray:
        """
        Calculates base shear for each individual mode.

        Returns
        -------
        np.ndarray
            Array of shape (1, n_modes) where each entry is the modal base shear.
        """
        if self.f_eq is None:
            self.modal_equivalent_forces()
        base_shear = self.inf_vec.T @ self.f_eq
        return base_shear.reshape(1, -1)

    def compute_cqc_matrix(self) -> np.ndarray:
        """
        Computes the Der Kiureghian cross-correlation matrix rho (n_modes x n_modes).
        """
        n = self.n_modes
        rho = np.ones((n, n), dtype=float)
        zeta = np.broadcast_to(self.ji, (n,))
        for i in range(n):
            for j in range(n):
                if i == j:
                    rho[i, j] = 1.0
                else:
                    r = self.omega[j] / self.omega[i]
                    zi = zeta[i]
                    zj = zeta[j]
                    num = 8.0 * np.sqrt(zi * zj) * (zi + r * zj) * (r ** 1.5)
                    den = (1.0 - r**2) ** 2 + 4.0 * zi * zj * r * (1.0 + r**2) + 4.0 * (zi**2 + zj**2) * (r**2)
                    rho[i, j] = num / den
        return rho

    def combine_modal_responses(
        self,
        responses: np.ndarray,
        method: str = "SRSS",
    ) -> np.ndarray:
        """
        Combines peak modal responses to estimate the total response.

        Parameters
        ----------
        responses : np.ndarray
            Array of modal responses. Can be 1D of shape (n_modes,) or 2D of shape (ndof, n_modes).
        method : str, optional
            Modal combination rule: 'SRSS', 'CQC', or 'ABS'. Default is 'SRSS'.

        Returns
        -------
        np.ndarray
            Combined response vector or scalar.
        """
        resp = np.asarray(responses, dtype=float)
        method_upper = method.upper()

        if method_upper == "SRSS":
            if resp.ndim == 1:
                return np.sqrt(np.sum(resp**2))
            return np.sqrt(np.sum(resp**2, axis=-1))

        elif method_upper == "CQC":
            rho = self.compute_cqc_matrix()
            if resp.ndim == 1:
                val = resp @ rho @ resp
                return np.sqrt(max(0.0, float(val)))
            elif resp.ndim == 2:
                vals = np.einsum("ki,ij,kj->k", resp, rho, resp)
                return np.sqrt(np.maximum(0.0, vals))
            else:
                raise ValueError("Responses must be a 1D or 2D array.")

        elif method_upper == "ABS":
            if resp.ndim == 1:
                return np.sum(np.abs(resp))
            return np.sum(np.abs(resp), axis=-1)

        else:
            raise ValueError(
                f"Method '{method}' not supported. Supported methods are 'SRSS', 'CQC', 'ABS'."
            )

    def compute_modal_element_forces(self, method: str = "CQC") -> Dict[str, Any]:
        """
        Calculates local internal forces for each element across all modes and
        combines them using the specified combination rule.

        Returns
        -------
        dict
            {
                "modal": {eid: (n_local_forces, n_modes) array},
                "combined": {eid: (n_local_forces,) array}
            }
        """
        struct = getattr(self.mdf, "structure", None)
        if struct is None or not hasattr(struct, "elements"):
            raise ValueError("No underlying finite element structure found on MDF.")

        if self.u is None:
            self.modal_displacements()

        modal_elem_forces: Dict[Any, np.ndarray] = {}
        combined_elem_forces: Dict[Any, np.ndarray] = {}

        for eid, el in struct.elements.items():
            dofs = el.node_i.dofs + el.node_j.dofs
            n_dofs = len(dofs)
            elem_modal_forces = np.zeros((n_dofs, self.n_modes), dtype=float)

            T = el.transformation_matrix()
            k_local = el.local_stiffness()

            for m in range(self.n_modes):
                d_m = np.zeros(struct.neq, dtype=float)
                d_m[struct.free_dofs] = self.u[:, m]
                u_elem_g = d_m[dofs]
                u_elem_local = T @ u_elem_g
                elem_modal_forces[:, m] = k_local @ u_elem_local

            modal_elem_forces[eid] = elem_modal_forces
            combined_elem_forces[eid] = self.combine_modal_responses(
                elem_modal_forces, method=method
            )

        return {"modal": modal_elem_forces, "combined": combined_elem_forces}

    def plot_response_spectrum(self):
        """Placeholder for plotting response spectrum."""
        pass
