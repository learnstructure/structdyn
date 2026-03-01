from abc import ABC, abstractmethod
import numpy as np


class MaterialModel(ABC):
    """
    Abstract base class for all material models.

    This class defines the interface that all material models must implement.
    It ensures that any material can be seamlessly integrated into a non-linear
    analysis element.

    Methods
    -------
    trial_response(u, v, dt)
        Computes the trial resisting force and tangent stiffness for a given
        displacement and velocity.
    commit_state(u)
        Updates the internal history variables of the material after a
        time step has converged.
    get_state(u, dt)
        A high-level convenience method for single-step updates where velocity
        is computed automatically.
    reset()
        Resets the material to its initial, undeformed state.
    """

    def __init__(self):
        self._u_prev = 0.0  # previous displacement (for automatic velocity)

    @abstractmethod
    def trial_response(self, u, v, dt):
        """
        Compute trial force and tangent stiffness.
        Returns (fs_trial, kt_trial, flag).
        """
        pass

    @abstractmethod
    def commit_state(self, u):
        """
        Update internal state after convergence.
        """
        pass

    def get_state(self, u, dt):
        """
        High‑level method: compute restoring force for a single step.
        Velocity is estimated from the stored previous displacement.
        The state is committed automatically.
        """
        v = (u - self._u_prev) / dt
        fs, kt, flag = self.trial_response(u, v, dt)
        self.commit_state(u)
        self._u_prev = u
        return fs, kt, flag

    def reset(self):
        """Reset internal state and previous displacement."""
        self._u_prev = 0.0


class LinearElastic(MaterialModel):
    """
    Linear elastic material model.

    Parameters
    ----------
    stiffness : float
        Elastic stiffness (force per unit displacement).
    """

    def __init__(self, stiffness):
        super().__init__()
        self.k = stiffness

    def trial_response(self, u, v, dt):
        """
        Return force = k*u, constant stiffness, and flag=False.
        """
        return self.k * u, self.k, False

    def commit_state(self, u):
        """
        No internal state to update.
        """
        pass

    def reset(self):
        """Reset to initial state (nothing to do except base)."""
        super().reset()


class ElasticPerfectlyPlastic(MaterialModel):
    """
    Elastic-Perfectly Plastic hysteretic material model.

    This is a non-linear model characterized by an initial linear elastic
    response followed by a plateau where the force remains constant upon
    further deformation.

    Parameters
    ----------
    uy : float
        Yield displacement.
    fy : float
        Yield force.
    """

    def __init__(self, uy=0.02, fy=36000):
        super().__init__()  # important to initialise base class
        self.uy = uy
        self.fy = fy
        self.k0 = fy / uy
        self.u_p = 0.0

    def trial_response(self, u, v, dt):
        fs_trial = self.k0 * (u - self.u_p)
        if abs(fs_trial) <= self.fy:
            return fs_trial, self.k0, False
        else:
            return self.fy * np.sign(fs_trial), 0.0, True

    def commit_state(self, u):
        fs = self.k0 * (u - self.u_p)
        if abs(fs) > self.fy:
            self.u_p = u - (self.fy / self.k0) * np.sign(fs)


class Bilinear(MaterialModel):
    """
    Bilinear hysteretic model with kinematic hardening.

    Parameters
    ----------
    uy : float
        Yield displacement.
    fy : float
        Yield force.
    alpha : float, optional
        Hardening ratio (post‑yield stiffness / initial stiffness), by default 0.0.
        For alpha = 0 the model reduces to elastic‑perfectly plastic.
    """

    def __init__(self, uy, fy, alpha=0.0):
        super().__init__()
        self.uy = uy
        self.fy = fy
        self.k0 = fy / uy  # initial stiffness
        self.alpha = alpha
        self.H = alpha * self.k0  # hardening modulus

        # State variables
        self.u_p = 0.0  # committed plastic displacement
        self.u_p_trial = 0.0  # trial plastic displacement
        self.was_plastic = False  # flag for commit

    def trial_response(self, u, v, dt):
        """
        Compute trial force and tangent stiffness.
        """
        # Trial elastic force
        fs_trial = self.k0 * (u - self.u_p)
        # Back stress (kinematic hardening)
        beta = self.H * self.u_p
        # Relative stress
        eta = fs_trial - beta

        if abs(eta) <= self.fy:
            # Elastic step
            self.was_plastic = False
            return fs_trial, self.k0, False
        else:
            # Plastic step
            sign_eta = np.sign(eta)
            # Plastic multiplier (change in plastic displacement)
            delta_u_p = (abs(eta) - self.fy) / (self.k0 + self.H)
            self.u_p_trial = self.u_p + delta_u_p * sign_eta
            # Consistent force
            fs = self.k0 * (u - self.u_p_trial)
            # Consistent tangent (elastic‑plastic modulus)
            kt = (self.k0 * self.H) / (self.k0 + self.H)
            self.was_plastic = True
            return fs, kt, True

    def commit_state(self, u):
        """Update plastic displacement after convergence."""
        if self.was_plastic:
            self.u_p = self.u_p_trial
        self.was_plastic = False  # reset flag

    def reset(self):
        """Reset to virgin state."""
        self.u_p = 0.0
        self.u_p_trial = 0.0
        self.was_plastic = False
        super().reset()  # also resets _u_prev from base class


class BoucWen(MaterialModel):
    """
    Bouc-Wen hysteretic model.

    This is a highly versatile model that can represent a wide range of smooth
    hysteretic behaviors. The restoring force is given by:
    fs = alpha*k0*u + (1-alpha)*k0*z
    where z is the hysteretic displacement governed by a differential equation.

    Parameters
    ----------
    k0 : float
        Initial stiffness.
    alpha : float
        Post-yield to pre-yield stiffness ratio.
    A : float, optional
        Controls the scale of the hysteretic component. Default is 1.0.
    beta : float, optional
        Controls the shape of the hysteresis loop. Default is 0.5.
    gamma : float, optional
        Controls the shape of the hysteresis loop. Default is 0.5.
    n : int, optional
        Controls the sharpness of the transition from elastic to plastic. Default is 1.
    """

    def __init__(self, k0, alpha, A=1.0, beta=0.5, gamma=0.5, n=1):
        super().__init__()
        self.k0 = k0
        self.alpha = alpha
        self.A = A
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.z = 0.0
        self.z_trial = 0.0

    def trial_response(self, u, v, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        z = self.z
        zdot = (
            self.A * v
            - self.beta * abs(v) * (abs(z) ** (self.n - 1)) * z
            - self.gamma * v * (abs(z) ** self.n)
        )
        self.z_trial = z + zdot * dt

        fs_trial = self.k0 * (self.alpha * u + (1 - self.alpha) * self.z_trial)

        # Tangent stiffness
        if abs(v) > 1e-12:
            sign_v = np.sign(v)
        else:
            sign_v = 0.0
        dz_du = (
            self.A
            - self.beta * sign_v * (abs(z) ** (self.n - 1)) * z
            - self.gamma * (abs(z) ** self.n)
        )
        kt_trial = self.k0 * (self.alpha + (1 - self.alpha) * dz_du)

        return fs_trial, kt_trial, False

    def commit_state(self, u):
        self.z = self.z_trial
