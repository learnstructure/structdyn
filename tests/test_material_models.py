"""
Tests of the hysteretic material models against their defining force-deformation relations.
"""

import numpy as np
import pytest

from structdyn.utils.material_models import Bilinear, ElasticPerfectlyPlastic


def _path(model, displacements):
    """Force and tangent along a displacement path, committing the state after each point."""
    forces, tangents = [], []
    for u in displacements:
        fs, kt, _ = model.trial_response(u, 0.0, 0.01)
        model.commit_state(u)
        forces.append(fs)
        tangents.append(kt)
    return np.array(forces), np.array(tangents)


@pytest.mark.parametrize("alpha", [0.0, 0.02, 0.05, 0.2])
def test_bilinear_post_yield_stiffness_is_alpha_k0(alpha):
    """Monotonic loading follows fy + alpha * k0 * (u - uy) beyond yield, with tangent alpha * k0."""
    uy, fy = 0.01, 100.0
    k0 = fy / uy
    model = Bilinear(uy=uy, fy=fy, alpha=alpha)
    u = np.linspace(0.0, 4 * uy, 41)
    forces, tangents = _path(model, u)

    expected = np.where(u <= uy, k0 * u, fy + alpha * k0 * (u - uy))
    assert forces == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert tangents[u > uy] == pytest.approx(alpha * k0, rel=1e-12)


def test_bilinear_unloads_elastically_and_reverses_after_an_elastic_range_of_2fy():
    """Kinematic hardening: unloading with k0, reverse yielding when the force has dropped by 2 fy."""
    uy, fy, alpha = 0.01, 100.0, 0.05
    k0 = fy / uy
    model = Bilinear(uy=uy, fy=fy, alpha=alpha)
    u_max = 3 * uy
    f_max = fy + alpha * k0 * (u_max - uy)
    _path(model, np.linspace(0.0, u_max, 31))

    # Elastic unloading over a force drop of just under 2 fy
    u_rev = u_max - 1.99 * fy / k0
    f_rev, kt_rev = _path(model, [u_rev])
    assert f_rev[0] == pytest.approx(f_max - 1.99 * fy, rel=1e-12)
    assert kt_rev[0] == pytest.approx(k0, rel=1e-12)

    # Beyond a drop of 2 fy, the model yields in reverse with tangent alpha * k0
    u_yield = u_max - 2 * fy / k0
    u_beyond = u_yield - uy
    f_beyond, kt_beyond = _path(model, [u_beyond])
    assert f_beyond[0] == pytest.approx(f_max - 2 * fy - alpha * k0 * uy, rel=1e-12)
    assert kt_beyond[0] == pytest.approx(alpha * k0, rel=1e-12)


def test_bilinear_with_zero_alpha_matches_elastic_perfectly_plastic():
    uy, fy = 0.02, 36000.0
    u = 0.05 * np.sin(np.linspace(0.0, 4 * np.pi, 81))
    f_bilinear, _ = _path(Bilinear(uy=uy, fy=fy, alpha=0.0), u)
    f_epp, _ = _path(ElasticPerfectlyPlastic(uy=uy, fy=fy), u)
    assert f_bilinear == pytest.approx(f_epp, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
def test_bilinear_rejects_alpha_outside_zero_to_one(alpha):
    with pytest.raises(ValueError, match="alpha"):
        Bilinear(uy=0.01, fy=100.0, alpha=alpha)
