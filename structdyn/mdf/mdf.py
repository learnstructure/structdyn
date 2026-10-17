import numpy as np
from .analytical_methods.modal_analysis import ModalAnalysis
from structdyn.ground_motions import GroundMotion
from structdyn.loads import LoadHistory
from structdyn.mdf.mdf_helpers.visualization import ShearBuildingVisualizer


class MDF:
    """
    Represents a linear Multi-Degree-of-Freedom (MDF) system.

    This class defines a structural system with multiple degrees of freedom governed by the second-order linear differential equation:

    M ü + C u̇ + K u = f(t)

    where:
    - M is the mass matrix
    - C is the damping matrix
    - K is the stiffness matrix
    - u is the displacement vector
    - f(t) is the external force vector

    Parameters
    ----------
    M : (n, n) array-like
        The mass matrix of the system.
    K : (n, n) array-like
        The stiffness matrix of the system.
    C : (n, n) array-like, optional
        The damping matrix. If not provided, it is initialized as a zero matrix.
    """

    def __init__(self, M, K, C=None, elements=None):
        """
        Initializes the MDF system with mass, stiffness, and optional damping matrices.
        """
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)

        if C is None:
            self.C = np.zeros_like(self.M)
        else:
            self.C = np.asarray(C, dtype=float)

        self.elements = elements  # list of Element objects (None means linear)

        self.ndof = self.M.shape[0]

        self._validate()

        self.modal = ModalAnalysis(self)
        self._visualizer = None  # Will be initialized when needed

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
        """
        Sets the damping matrix C based on modal damping ratios (Rayleigh damping).

        This method constructs a classical damping matrix C using the natural frequencies
        and mode shapes of the undamped system.

        Parameters
        ----------
        zeta : float, int, or array-like
            Modal damping ratio(s) for the modes to be included. If a scalar is provided,
            it is applied equally across all computed modes.
        n_modes : int, optional
            The number of modes to use for constructing the damping matrix.
            If None, all available modes are used. Defaults to None.

        Returns
        -------
        C : ndarray
            The resulting (n, n) damping matrix.
        """
        omega, phi = self.modal.modal_analysis(n_modes=n_modes)

        num_modes = phi.shape[1]
        if isinstance(zeta, (int, float)):
            zeta = np.full(num_modes, float(zeta))
        else:
            zeta = np.asarray(zeta, dtype=float)
            if len(zeta) != num_modes:
                # If uniform zeta array was passed with more modes than available, truncate cleanly
                if len(zeta) > num_modes and np.allclose(zeta, zeta[0]):
                    zeta = zeta[:num_modes]
                else:
                    raise ValueError(
                        f"Length of zeta ({len(zeta)}) must equal number of modes used ({num_modes})."
                    )

        C = np.zeros_like(self.M)

        for i in range(num_modes):
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
        Creates an MDF system representing a shear building model.

        Parameters
        ----------
        masses : list or array
            A list of lumped masses at each floor, starting from the bottom floor.
        stiffnesses : list or array
            A list of story stiffnesses, starting from the bottom story.
            The length must be equal to the number of masses.

        Returns
        -------
        MDF
            A new MDF instance representing the shear building.
        """
        from .mdf_helpers.builders import _shear_building_logic

        M, K = _shear_building_logic(masses, stiffnesses)
        instance = cls(M, K)
        instance.masses = masses
        instance.stiffnesses = stiffnesses
        return instance

    @classmethod
    def from_fem2d(
        cls,
        structure,
        zeta=None,
        n_modes=None,
        lumped=None,
        include_rotational_inertia=False,
    ):
        """
        Creates an MDF system directly from a fem2d Structure object (linear or non-linear).

        Parameters
        ----------
        structure : fem2d.Structure
            The fem2d structure.
        zeta : float or list of float, optional
            Modal damping ratio(s).
        n_modes : int, optional
            Number of modes for damping matrix calculation.
        lumped : bool, optional
            Whether to use lumped mass formulation.
        include_rotational_inertia : bool, optional
            Whether to include rotational mass inertia when lumped=True. Defaults to False.
        """
        if hasattr(structure, "structure"):
            structure = structure.structure

        structure.number_dofs()
        structure.apply_boundary_conditions()
        if hasattr(structure, "get_reduced_matrices"):
            import inspect
            sig = inspect.signature(structure.get_reduced_matrices)
            if "lumped" in sig.parameters and lumped is not None:
                K_ff, M_ff = structure.get_reduced_matrices(
                    lumped=lumped,
                    include_rotational_inertia=include_rotational_inertia,
                )
            else:
                K_ff, M_ff = structure.get_reduced_matrices()
        else:
            K_ff, M_ff = structure.get_reduced_matrices()

        instance = cls(M=M_ff, K=K_ff, elements=structure)
        instance.structure = structure

        if zeta is not None:
            if isinstance(zeta, (int, float)):
                n = len(structure.free_dofs)
                if n_modes is not None:
                    n = min(n, n_modes)
                zeta = [float(zeta)] * n
            instance.set_modal_damping(zeta, n_modes=n_modes)

        return instance

    @classmethod
    def from_fem3d(
        cls,
        structure,
        zeta=None,
        n_modes=None,
        lumped=None,
        include_rotational_inertia=False,
    ):
        """
        Creates an MDF system directly from a fem3d Structure or SimpleFrame object.

        Parameters
        ----------
        structure : fem3d.Structure or fem3d.SimpleFrame
            The fem3d structure or simple frame model.
        zeta : float or list of float, optional
            Modal damping ratio(s).
        n_modes : int, optional
            Number of modes for damping matrix calculation.
        lumped : bool, optional
            Whether to use lumped mass formulation.
        include_rotational_inertia : bool, optional
            Whether to include rotational mass inertia when lumped=True. Defaults to False.
        """
        if hasattr(structure, "structure"):
            structure = structure.structure

        structure.number_dofs()
        structure.apply_boundary_conditions()
        if hasattr(structure, "get_reduced_matrices"):
            import inspect
            sig = inspect.signature(structure.get_reduced_matrices)
            if "lumped" in sig.parameters and lumped is not None:
                K_ff, M_ff = structure.get_reduced_matrices(
                    lumped=lumped,
                    include_rotational_inertia=include_rotational_inertia,
                )
            else:
                K_ff, M_ff = structure.get_reduced_matrices()
        else:
            K_ff, M_ff = structure.get_reduced_matrices()

        instance = cls(M=M_ff, K=K_ff, elements=structure)
        instance.structure = structure

        if zeta is not None:
            if isinstance(zeta, (int, float)):
                n = len(structure.free_dofs)
                if n_modes is not None:
                    n = min(n, n_modes)
                zeta = [float(zeta)] * n
            instance.set_modal_damping(zeta, n_modes=n_modes)

        return instance

    @classmethod
    def from_fem(
        cls,
        structure,
        zeta=None,
        n_modes=None,
        lumped=None,
        include_rotational_inertia=False,
    ):
        """
        Creates an MDF system directly from either a fem2d or fem3d Structure or SimpleFrame.
        """
        if hasattr(structure, "structure"):
            structure = structure.structure

        structure.number_dofs()
        structure.apply_boundary_conditions()
        if hasattr(structure, "get_reduced_matrices"):
            import inspect
            sig = inspect.signature(structure.get_reduced_matrices)
            if "lumped" in sig.parameters and lumped is not None:
                K_ff, M_ff = structure.get_reduced_matrices(
                    lumped=lumped,
                    include_rotational_inertia=include_rotational_inertia,
                )
            else:
                K_ff, M_ff = structure.get_reduced_matrices()
        else:
            K_ff, M_ff = structure.get_reduced_matrices()

        instance = cls(M=M_ff, K=K_ff, elements=structure)
        instance.structure = structure

        if zeta is not None:
            if isinstance(zeta, (int, float)):
                n = len(structure.free_dofs)
                if n_modes is not None:
                    n = min(n, n_modes)
                zeta = [float(zeta)] * n
            instance.set_modal_damping(zeta, n_modes=n_modes)

        return instance

    def response_spectrum_analysis(self, **kwargs):
        """
        Performs Response Spectrum Analysis on the MDF system.

        Returns
        -------
        ResponseSpectrumAnalysis
            The configured ResponseSpectrumAnalysis instance.
        """
        from .analytical_methods.response_spectrum_analysis import ResponseSpectrumAnalysis
        return ResponseSpectrumAnalysis(self, **kwargs)

    def find_response(
        self, load, method=None, elements=None, **kwargs
    ):
        """
        Computes the dynamic response of the system to an external force.

        Parameters
        ----------
        load : LoadHistory
            A :class:`~structdyn.loads.LoadHistory` describing the force
            excitation. The load can be specified in two ways:

            - **Sparse form** — ``load_values`` is 1D and ``dof`` identifies
              which DOFs receive the force. The full ``(n_t, ndof)`` matrix
              is assembled automatically.
            - **Full form** — ``load_values`` is a 2D ``(n_t, ndof)`` matrix
              giving the force at every DOF at every time step.
        method : str, optional
            The numerical integration method to use.
            Options are 'central_difference' or 'newmark_beta'.
            The default is "central_difference".
        elements : list, optional
            Elements for non-linear analysis (unchanged).
        **kwargs :
            Additional keyword arguments to be passed to the numerical solver.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the displacement, velocity, and
            acceleration response history.

        Notes
        -----
        The previous ``find_response(time, load_2d, ...)`` style — where the
        time vector and force matrix were passed as separate arguments — has
        been removed. Build a :class:`~structdyn.loads.LoadHistory` and pass
        it as the single argument. For sparse excitation:

        >>> load = LoadHistory(time_steps=t, load_values=p, dof=[0])
        >>> mdf.find_response(load, method="newmark_beta")

        For full excitation (legacy behaviour):

        >>> load = LoadHistory(time_steps=t, load_values=F_2d)  # F_2d: (n_t, ndof)
        >>> mdf.find_response(load, method="newmark_beta")
        """
        if not isinstance(load, LoadHistory):
            raise TypeError(
                "find_response expects a LoadHistory as its first argument. "
                "Build one with structdyn.loads.LoadHistory(time_steps, "
                "load_values) — optionally with dof=[...]. The legacy "
                "(time, load_matrix) call style has been removed."
            )

        time = load.time_steps
        dt = load.dt
        # Expand the load into the full (n_t, ndof) force matrix
        load_matrix = load.expand(ndof=self.ndof)

        from structdyn.mdf.numerical_methods.central_difference import (
            CentralDifferenceMDF,
        )
        from structdyn.mdf.numerical_methods.newmark_beta import NewmarkBetaMDF

        if elements is not None:
            self.elements = elements

        is_fem_structure = False
        if hasattr(self, "structure") and self.structure is not None:
            is_fem_structure = True
        elif hasattr(self.elements, "free_dofs") and hasattr(self.elements, "elements"):
            is_fem_structure = True

        if method is None:
            method = "newmark_beta" if (self.elements is not None or is_fem_structure) else "central_difference"
        elif method == "central_difference" and (self.elements is not None or is_fem_structure):
            raise NotImplementedError(
                "Central difference nonlinear/FEM dynamic solver is not implemented. Please use method='newmark_beta'."
            )

        # Determine solver class based on method and nonlinearity
        if method == "newmark_beta":
            if self.elements is not None or is_fem_structure:
                from structdyn.mdf.numerical_methods.newmark_beta_non_linear import (
                    NewmarkBetaNonLinear,
                )

                solver_class = NewmarkBetaNonLinear
            else:
                solver_class = NewmarkBetaMDF
        elif method == "central_difference":
            solver_class = CentralDifferenceMDF
        else:
            raise ValueError("method must be 'central_difference' or 'newmark_beta'")

        solver = solver_class(self, dt, **kwargs)
        return solver.compute_solution(time, load_matrix)

    def find_response_ground_motion(
        self, gm, inf_vec=None, method=None, **kwargs
    ):
        """
        Computes the dynamic response of the system to ground motion.

        Parameters
        ----------
        gm : GroundMotion
            A GroundMotion object containing the ground acceleration history.
        inf_vec : array-like, optional
            The influence vector, which relates the ground motion to the degrees of freedom.
            If None, it is assumed to be a vector of ones (all DOFs are equally affected).
            The default is None.
        method : str, optional
            The numerical integration method to use.
            Options are 'central_difference' or 'newmark_beta'.
            The default is "central_difference".
        **kwargs :
            Additional keyword arguments to be passed to the numerical solver.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the displacement, velocity, and acceleration response history.
        """
        if not isinstance(gm, GroundMotion):
            raise TypeError("gm must be a GroundMotion object")
        time = np.asarray(gm.time)
        ag = np.asarray(gm.acceleration, dtype=float)
        if inf_vec is None:
            if hasattr(self, "structure") and self.structure is not None:
                struct = self.structure
                # Determine DOF stride per node: 6 for 3D, 3 for 2D
                dofs_per_node = 3
                if any(len(getattr(n, "dofs", [])) == 6 for n in getattr(struct, "nodes", {}).values()):
                    dofs_per_node = 6

                # Direction can be passed via kwargs or default to primary translation
                direction = str(kwargs.pop("direction", "x")).lower()
                if direction in ("x", "0"):
                    target_sub_dof = 0
                elif direction in ("y", "1"):
                    target_sub_dof = 1
                elif direction in ("z", "2"):
                    target_sub_dof = 2 if dofs_per_node == 6 else 1
                else:
                    target_sub_dof = 0

                inf_vec = np.zeros(self.ndof)
                for i, dof_idx in enumerate(struct.free_dofs):
                    if dof_idx % dofs_per_node == target_sub_dof:
                        inf_vec[i] = 1.0
            else:
                inf_vec = np.ones(self.ndof)
        inf_vec = np.asarray(inf_vec)
        if inf_vec.shape != (self.ndof,):
            raise ValueError("inf_vec must have shape (ndof,)")

        # Compute effective inertia vector M r
        Mr = self.M @ inf_vec
        # Build a full (n_t, ndof) LoadHistory from the ground motion
        # and the effective inertia vector. The 2D form is used so all
        # DOFs receive the correct share of the inertial force.
        full_load = -ag[:, None] * Mr[None, :]
        load = LoadHistory(time, full_load)
        return self.find_response(load, method=method, **kwargs)

    def assemble_resisting_force_and_tangent(self, u, v, dt):
        """
        Assembles the global resisting force and tangent stiffness matrix.

        This method is called by a non-linear solver at each iteration within
        a time step. It iterates through all elements defined in `self.elements`,
        gets their trial force and stiffness, and assembles them into the
        global resisting force vector `Fs` and tangent stiffness matrix `Kt`.

        Parameters
        ----------
        u : np.ndarray
            The trial displacement vector for the current iteration.
        v : np.ndarray
            The trial velocity vector for the current iteration.
        dt : float
            The time step size.

        Returns
        -------
        Fs : np.ndarray
            The global internal resisting force vector.
        Kt : np.ndarray
            The global tangent stiffness matrix.
        """
        struct = None
        if hasattr(self, "structure") and self.structure is not None:
            struct = self.structure
        elif hasattr(self.elements, "free_dofs") and hasattr(self.elements, "elements"):
            struct = self.elements

        if struct is not None:
            d = np.zeros(struct.neq)
            d[struct.free_dofs] = u
            f_int = np.zeros(struct.neq)
            Kt = np.zeros((struct.neq, struct.neq))
            for el in struct.elements.values():
                if hasattr(el, "update_state"):
                    el.update_state(d)
                dofs = el.node_i.dofs + el.node_j.dofs
                f_int[dofs] += el.get_internal_forces()
                Kt[np.ix_(dofs, dofs)] += el.get_tangent_stiffness()
            return f_int[struct.free_dofs], Kt[np.ix_(struct.free_dofs, struct.free_dofs)]

        Fs = np.zeros(self.ndof)
        Kt = np.zeros((self.ndof, self.ndof))
        if self.elements is not None:
            for elem in self.elements:
                fe, ke = elem.get_force_and_stiffness(u, v, dt)
                dofs = elem.dofs
                if len(dofs) == 1:
                    Fs[dofs[0]] += fe
                    Kt[dofs[0], dofs[0]] += ke
                elif len(dofs) == 2:
                    i, j = dofs
                    Fs[i] -= fe
                    Fs[j] += fe
                    Kt[i, i] += ke
                    Kt[i, j] -= ke
                    Kt[j, i] -= ke
                    Kt[j, j] += ke
                else:
                    raise ValueError("Element must have 1 or 2 DOFs")
        else:
            Fs = self.K @ u
            Kt = self.K
        return Fs, Kt

    def commit_elements(self, u):
        """
        Commits the state of all non-linear elements.

        This method is called by a non-linear solver at the end of a converged
        time step. It iterates through all elements and calls their `commit`
        method, passing the final converged displacement vector `u`. This allows
        each element to update its internal history variables.

        Parameters
        ----------
        u : np.ndarray
            The converged displacement vector for the time step.
        """
        struct = None
        if hasattr(self, "structure") and self.structure is not None:
            struct = self.structure
        elif hasattr(self.elements, "free_dofs") and hasattr(self.elements, "elements"):
            struct = self.elements

        if struct is not None:
            d = np.zeros(struct.neq)
            d[struct.free_dofs] = u
            for el in struct.elements.values():
                if hasattr(el, "update_state"):
                    el.update_state(d)
            if hasattr(struct, "commit_state"):
                struct.commit_state()
        elif self.elements is not None:
            for elem in self.elements:
                if hasattr(elem, "commit"):
                    elem.commit(u)

    @property
    def plot(self):
        """Provides access to plotting methods for the shear building."""
        if self._visualizer is None:
            self._visualizer = ShearBuildingVisualizer(self)
        return self._visualizer
