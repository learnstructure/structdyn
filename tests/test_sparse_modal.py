"""
Sparse systems in MDF: the dense/sparse switch (DENSE_DOF_LIMIT), the shift-invert eigensolver (eigsh), and the
operations refused on sparse systems.

Closed form: a fixed-free chain of n equal masses m and springs k has
    omega_j = 2 sqrt(k/m) sin((2j - 1) pi / (2 (2n + 1))),  j = 1 ... n
(e.g. Chopra, Dynamics of Structures, the uniform shear building).
"""

import math

import numpy as np
import pytest
import scipy.sparse as sp

import structdyn.mdf.mdf as mdf_module
from structdyn.mdf.mdf import DENSE_DOF_LIMIT, MDF


def chain(n, k=1000.0, m=2.0, mass_pattern=None, grounded=True):
    """Sparse K and M of a chain of n masses; spring 1 ties DOF 0 to the ground unless ``grounded`` is False."""
    main = np.full(n, 2.0 * k)
    main[-1] = k
    if not grounded:
        main[0] = k
    K = sp.diags([main, np.full(n - 1, -k), np.full(n - 1, -k)], [0, -1, 1], format="csr")
    masses = np.full(n, m) if mass_pattern is None else np.asarray(mass_pattern, float)
    return K, sp.diags(masses, format="csr")


def chain_omegas(n, k=1000.0, m=2.0, n_modes=6):
    j = np.arange(1, n_modes + 1)
    return 2.0 * math.sqrt(k / m) * np.sin((2 * j - 1) * math.pi / (2 * (2 * n + 1)))


@pytest.mark.parametrize("n, sparse_expected", [(DENSE_DOF_LIMIT, False), (DENSE_DOF_LIMIT + 1, True)])
def test_switch_on_both_sides_of_the_limit_matches_the_closed_form(n, sparse_expected):
    K, M = chain(n)
    system = MDF(M, K)
    assert system.is_sparse is sparse_expected
    assert sp.issparse(system.K) is sparse_expected
    omega, phi = system.modal.modal_analysis(n_modes=6)
    assert np.allclose(omega, chain_omegas(n), rtol=1e-9)
    # mass-normalized modes in both paths
    mn = phi.T @ (system.M @ phi)
    assert np.allclose(mn, np.eye(6), atol=1e-8)


def test_dense_input_is_never_made_sparse():
    K, M = chain(DENSE_DOF_LIMIT + 1)
    system = MDF(M.toarray(), K.toarray())
    assert not system.is_sparse and isinstance(system.K, np.ndarray)


def test_sparse_and_dense_paths_agree_with_massless_dofs(monkeypatch):
    # Every other mass is zero (like the rotations of a frame with lumped mass): the dense path condenses them
    # statically; the shift-invert path needs no condensation.
    n = 60
    K, M = chain(n, mass_pattern=[2.0 if i % 2 else 0.0 for i in range(n)])
    dense = MDF(M, K)
    w_dense, phi_dense = dense.modal.modal_analysis(n_modes=5)
    monkeypatch.setattr(mdf_module, "DENSE_DOF_LIMIT", 10)
    sparse = MDF(M, K)
    assert sparse.is_sparse
    w_sparse, phi_sparse = sparse.modal.modal_analysis(n_modes=5)
    assert np.allclose(w_sparse, w_dense, rtol=1e-9)
    for i in range(5):  # same shapes up to sign
        s = np.sign(phi_dense[:, i] @ phi_sparse[:, i])
        assert np.allclose(s * phi_sparse[:, i], phi_dense[:, i], atol=1e-8 * np.abs(phi_dense[:, i]).max())


def test_singular_stiffness_gives_a_clear_error():
    K, M = chain(DENSE_DOF_LIMIT + 1, grounded=False)  # free-free chain: a rigid-body mode
    with pytest.raises(ValueError, match="stiffness matrix is singular"):
        MDF(M, K).modal.modal_analysis(n_modes=3)


def test_sparse_system_needs_n_modes():
    K, M = chain(DENSE_DOF_LIMIT + 1)
    with pytest.raises(ValueError, match="needs n_modes"):
        MDF(M, K).modal.modal_analysis()


def test_damping_matrix_and_time_history_are_refused_on_sparse_systems():
    K, M = chain(DENSE_DOF_LIMIT + 1)
    system = MDF(M, K)
    with pytest.raises(ValueError, match="needs dense matrices"):
        system.set_modal_damping(0.05, n_modes=3)
    with pytest.raises(ValueError, match="needs dense matrices"):
        system.find_response_ground_motion(gm=None)
