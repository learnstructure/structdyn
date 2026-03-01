import numpy as np
from .analytical_methods.modal_analysis import ModalAnalysis
from structdyn.ground_motions import GroundMotion


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
        zeta : array-like
            An array or list of modal damping ratios for the modes to be included.
        n_modes : int, optional
            The number of modes to use for constructing the damping matrix.
            If None, all modes are used. The default is None.

        Returns
        -------
        C : ndarray
            The resulting (n, n) damping matrix.
        """
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
        return cls(M, K)

    def find_response(
        self, time, load, method="central_difference", elements=None, **kwargs
    ):
        """
        Computes the dynamic response of the system to an external force.

        Parameters
        ----------
        time : array-like
            A uniformly spaced time vector.
        load : (nt, ndof) ndarray
            The external force history, where `nt` is the number of time steps
            and `ndof` is the number of degrees of freedom.
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
        from structdyn.mdf.numerical_methods.central_difference import (
            CentralDifferenceMDF,
        )
        from structdyn.mdf.numerical_methods.newmark_beta import NewmarkBetaMDF

        time = np.asarray(time)
        dt = time[1] - time[0]
        if not np.allclose(np.diff(time), dt):
            raise ValueError("Time vector must be uniformly spaced")

        if elements is not None:
            self.elements = elements

        # Determine solver class based on method and nonlinearity
        if method == "newmark_beta":
            if self.elements is not None:
                from structdyn.mdf.numerical_methods.newmark_beta_non_linear import (
                    NewmarkBetaNonLinear,
                )

                solver_class = NewmarkBetaNonLinear
            else:
                solver_class = NewmarkBetaMDF
        elif method == "central_difference":
            if self.elements is not None:
                raise NotImplementedError(
                    "Central difference nonlinear solver not yet implemented"
                )
            else:
                solver_class = CentralDifferenceMDF
        else:
            raise ValueError("method must be 'central_difference' or 'newmark_beta'")

        solver = solver_class(self, dt, **kwargs)
        return solver.compute_solution(time, load)

    def find_response_ground_motion(
        self, gm, inf_vec=None, method="central_difference", **kwargs
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
        Fs = np.zeros(self.ndof)
        Kt = np.zeros((self.ndof, self.ndof))
        for elem in self.elements:
            fe, ke = elem.get_force_and_stiffness(u, v, dt)
            dofs = elem.dofs
            if len(dofs) == 1:
                # Base story
                Fs[dofs[0]] += fe
                Kt[dofs[0], dofs[0]] += ke
            elif len(dofs) == 2:
                # Interior story
                i, j = dofs
                Fs[i] -= fe
                Fs[j] += fe
                Kt[i, i] += ke
                Kt[i, j] -= ke
                Kt[j, i] -= ke
                Kt[j, j] += ke
            else:
                raise ValueError("Element must have 1 or 2 DOFs")
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
        if self.elements is not None:
            for elem in self.elements:
                elem.commit(u)
