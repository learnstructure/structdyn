"""
Failures that used to be hidden now raise an error that names the cause: a singular stiffness matrix of the massless
DOFs (a mechanism), a negative eigenvalue beyond round-off (an unstable model), and an unrecognized ground motion
direction. Newmark integration defaults to constant average acceleration.
"""

import numpy as np
import pytest

from structdyn import MDF
from structdyn.mdf.numerical_methods.newmark_beta import NewmarkBetaMDF
from structdyn.sdf.numerical_methods.newmark_beta import NewmarkBeta
from structdyn.sdf.sdf import SDF


def test_mechanism_among_massless_dofs_is_reported():
    """DOF 3 has no mass and no stiffness: the static condensation has no solution."""
    M = np.diag([1.0, 1.0, 0.0])
    K = np.array([[2.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="mechanism"):
        MDF(M, K).modal.modal_analysis()


def test_unstable_model_is_reported():
    """A stiffness matrix with a negative eigenvalue (not positive semi-definite) is an unstable model."""
    M = np.eye(2)
    K = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    with pytest.raises(ValueError, match="negative eigenvalue"):
        MDF(M, K).modal.modal_analysis()


def test_rigid_body_mode_round_off_is_kept_at_zero():
    """A free two-mass system has a zero eigenvalue; its round-off is set to zero, not reported."""
    M = np.eye(2)
    K = np.array([[1.0, -1.0], [-1.0, 1.0]])
    omega, _ = MDF(M, K).modal.modal_analysis()
    assert omega[0] == pytest.approx(0.0, abs=1e-6) and omega[1] == pytest.approx(np.sqrt(2.0))


def test_newmark_defaults_to_average_acceleration():
    assert (NewmarkBeta(SDF(1.0, 100.0), dt=0.01).beta, NewmarkBeta(SDF(1.0, 100.0), dt=0.01).gamma) == (0.25, 0.5)
    mdf = MDF(np.eye(2), np.array([[2.0, -1.0], [-1.0, 1.0]]))
    solver = NewmarkBetaMDF(mdf, dt=0.01)
    assert (solver.beta, solver.gamma) == (0.25, 0.5)
