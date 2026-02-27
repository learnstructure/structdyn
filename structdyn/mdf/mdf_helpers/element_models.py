from abc import ABC, abstractmethod
import numpy as np


class Element(ABC):
    """Base class for a finite element with a material model."""

    def __init__(self, material_model, dof_indices):
        """
        Parameters
        ----------
        material_model : MaterialModel
            An instance of a class derived from MaterialModel (e.g., BoucWen).
        dof_indices : list of int
            Global degrees of freedom associated with this element.
        """
        self.material = material_model
        self.dofs = dof_indices  # list of global DOF indices

    @abstractmethod
    def compute_deformation(self, u_global, v_global):
        """
        Given global displacement and velocity vectors, return the
        local deformation (strain, drift, etc.) and its velocity.
        """
        pass

    def get_force_and_stiffness(self, u_global, v_global, dt):
        """
        Compute the element resisting force and tangent stiffness.
        Returns:
            fe : float or 1D array (force in local sense, or global contribution)
            ke : float or 2D array (element tangent stiffness in local coords)
        For a 1‑D element (e.g., story shear spring), fe is a scalar (force)
        and ke is a scalar (tangent stiffness). For a 2‑D or 3‑D element,
        you would return a vector/matrix, but we start simple.
        """
        u_local, v_local = self.compute_deformation(u_global, v_global)
        fs, kt, _ = self.material.trial_response(u_local, v_local, dt)
        return fs, kt

    def commit(self, u_global):
        """Commit the material state after convergence."""
        u_local, _ = self.compute_deformation(u_global, np.zeros_like(u_global))
        self.material.commit_state(u_local)


class ShearStoryElement(Element):
    """
    Element representing a story in a shear‑frame building.
    If one DOF is given, it is a base story (ground to floor).
    If two DOFs are given, it is an interior story between two floors.
    """

    def compute_deformation(self, u_global, v_global):
        if len(self.dofs) == 1:
            # Base story: drift = displacement of the only DOF
            i = self.dofs[0]
            u_local = u_global[i]
            v_local = v_global[i]
        elif len(self.dofs) == 2:
            # Interior story: drift = u_upper - u_lower
            i, j = self.dofs
            u_local = u_global[j] - u_global[i]
            v_local = v_global[j] - v_global[i]
        else:
            raise ValueError("ShearStoryElement must have 1 or 2 DOFs")
        return u_local, v_local
