"""
Tests for the LoadHistory class — the dynamic load definition used by
SDF and MDF solvers.
"""

import numpy as np
import pytest

from structdyn import LoadHistory


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_basic_1d_construction():
    """A 1D load is valid for SDF use."""
    t = np.linspace(0, 1.0, 11)
    p = np.sin(2 * np.pi * t)
    lh = LoadHistory(t, p)

    assert lh.ndof == 1
    assert lh.dof is None
    assert lh.dt == pytest.approx(0.1)
    assert len(lh) == 11
    np.testing.assert_allclose(lh.time_steps, t)
    np.testing.assert_allclose(lh.load_values, p)


def test_full_2d_construction():
    """A 2D (n_t, ndof) load is valid for MDF use."""
    t = np.linspace(0, 1.0, 11)
    F = np.zeros((11, 3))
    F[:, 1] = 1.0
    lh = LoadHistory(t, F)

    assert lh.ndof == 3
    assert lh.dof is None
    assert len(lh) == 11


def test_sparse_construction_with_dof():
    """A 1D load with `dof` is sparse MDF load."""
    t = np.linspace(0, 1.0, 11)
    p = np.arange(11, dtype=float)
    lh = LoadHistory(t, p, dof=[0])

    assert lh.ndof is None
    assert lh.dof is not None
    np.testing.assert_array_equal(lh.dof, [0])


def test_dof_must_be_1d_int_array():
    t = np.linspace(0, 1.0, 11)
    p = np.zeros(11)
    with pytest.raises(ValueError, match="1D sequence"):
        LoadHistory(t, p, dof=np.array([[0, 1]]))
    with pytest.raises(ValueError, match="non-empty"):
        LoadHistory(t, p, dof=[])
    with pytest.raises(ValueError, match="non-negative"):
        LoadHistory(t, p, dof=[-1])
    with pytest.raises(ValueError, match="unique"):
        LoadHistory(t, p, dof=[1, 1])


def test_dof_forbidden_with_2d_load():
    t = np.linspace(0, 1.0, 11)
    F = np.zeros((11, 3))
    with pytest.raises(ValueError, match="dof"):
        LoadHistory(t, F, dof=[0])


def test_length_mismatch_raises():
    t = np.linspace(0, 1.0, 11)
    p = np.zeros(10)
    with pytest.raises(ValueError, match="length"):
        LoadHistory(t, p)


def test_non_uniform_time_raises():
    t = np.array([0.0, 0.1, 0.3])  # non-uniform
    p = np.zeros(3)
    with pytest.raises(ValueError, match="uniformly spaced"):
        LoadHistory(t, p)


def test_too_few_time_points_raises():
    t = np.array([0.0])
    p = np.array([0.0])
    with pytest.raises(ValueError, match="at least two"):
        LoadHistory(t, p)


def test_load_values_ndim_validation():
    t = np.linspace(0, 1.0, 11)
    bad = np.zeros((11, 3, 2))
    with pytest.raises(ValueError, match="1D.*2D"):
        LoadHistory(t, bad)


def test_time_steps_must_be_1d():
    bad = np.zeros((5, 5))
    p = np.zeros(5)
    with pytest.raises(ValueError, match="1D"):
        LoadHistory(bad, p)


def test_sparse_dof_1d_flag_is_normalized():
    """`dof` should be coerced to int ndarray."""
    t = np.linspace(0, 1.0, 11)
    p = np.zeros(11)
    lh = LoadHistory(t, p, dof=(2, 5))
    assert lh.dof.dtype.kind == "i"
    np.testing.assert_array_equal(lh.dof, [2, 5])


# ---------------------------------------------------------------------------
# expand() — placement into the full (n_t, ndof) matrix
# ---------------------------------------------------------------------------


def test_expand_sparse_single_dof():
    t = np.linspace(0, 1.0, 5)
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lh = LoadHistory(t, p, dof=[2])

    F = lh.expand(ndof=4)
    assert F.shape == (5, 4)
    expected = np.zeros((5, 4))
    expected[:, 2] = p
    np.testing.assert_allclose(F, expected)


def test_expand_sparse_multiple_dofs_share_value():
    t = np.linspace(0, 1.0, 5)
    p = np.arange(5, dtype=float)
    lh = LoadHistory(t, p, dof=[0, 2])

    F = lh.expand(ndof=3)
    assert F.shape == (5, 3)
    np.testing.assert_allclose(F[:, 0], p)
    np.testing.assert_allclose(F[:, 1], 0)
    np.testing.assert_allclose(F[:, 2], p)


def test_expand_full_2d_returns_same_array():
    t = np.linspace(0, 1.0, 5)
    F = np.arange(20, dtype=float).reshape(5, 4)
    lh = LoadHistory(t, F)
    F2 = lh.expand(ndof=4)
    np.testing.assert_allclose(F2, F)


def test_expand_full_2d_wrong_ndof_raises():
    t = np.linspace(0, 1.0, 5)
    F = np.zeros((5, 4))
    lh = LoadHistory(t, F)
    with pytest.raises(ValueError, match="ndof"):
        lh.expand(ndof=3)


def test_expand_sparse_dof_out_of_range_raises():
    t = np.linspace(0, 1.0, 5)
    p = np.zeros(5)
    lh = LoadHistory(t, p, dof=[3])
    with pytest.raises(ValueError, match="out of range"):
        lh.expand(ndof=3)


def test_expand_1d_without_dof_raises():
    t = np.linspace(0, 1.0, 5)
    p = np.zeros(5)
    lh = LoadHistory(t, p)  # 1D, no dof
    with pytest.raises(ValueError, match="dof"):
        lh.expand(ndof=3)


# ---------------------------------------------------------------------------
# from_constant
# ---------------------------------------------------------------------------


def test_from_constant_sdf():
    t = np.linspace(0, 1.0, 11)
    lh = LoadHistory.from_constant(t, p0=10.0)
    assert lh.ndof == 1
    np.testing.assert_allclose(lh.load_values, 10.0)


def test_from_constant_mdf_with_dof():
    t = np.linspace(0, 1.0, 11)
    lh = LoadHistory.from_constant(t, p0=10.0, dof=[2])
    F = lh.expand(ndof=4)
    np.testing.assert_allclose(F[:, 2], 10.0)
    np.testing.assert_allclose(F[:, [0, 1, 3]], 0.0)


def test_from_constant_mdf_full_vector():
    t = np.linspace(0, 1.0, 11)
    lh = LoadHistory.from_constant(t, p0=[1.0, 2.0, 3.0])
    assert lh.ndof == 3
    np.testing.assert_allclose(lh.load_values, np.tile([1.0, 2.0, 3.0], (11, 1)))


def test_from_constant_scalar_and_dof_must_not_combine_with_vector():
    t = np.linspace(0, 1.0, 11)
    with pytest.raises(ValueError):
        LoadHistory.from_constant(t, p0=[1.0, 2.0], dof=[0])


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_scalar_multiplication():
    t = np.linspace(0, 1.0, 5)
    p = np.arange(5, dtype=float)
    lh = LoadHistory(t, p)
    lh2 = lh * 3.0
    np.testing.assert_allclose(lh2.load_values, 3 * p)


def test_addition_same_dof():
    t = np.linspace(0, 1.0, 5)
    a = LoadHistory(t, np.ones(5), dof=[0])
    b = LoadHistory(t, 2 * np.ones(5), dof=[0])
    c = a + b
    np.testing.assert_allclose(c.load_values, 3 * np.ones(5))
    np.testing.assert_array_equal(c.dof, [0])


def test_addition_different_dof_raises():
    t = np.linspace(0, 1.0, 5)
    a = LoadHistory(t, np.ones(5), dof=[0])
    b = LoadHistory(t, np.ones(5), dof=[1])
    with pytest.raises(ValueError, match="different DOFs"):
        a + b


def test_addition_different_time_grids_raises():
    t1 = np.linspace(0, 1.0, 5)
    t2 = np.linspace(0, 1.05, 5)  # same length, different values
    a = LoadHistory(t1, np.ones(5), dof=[0])
    b = LoadHistory(t2, np.ones(5), dof=[0])
    with pytest.raises(ValueError, match="time grids"):
        a + b


def test_addition_full_2d():
    t = np.linspace(0, 1.0, 5)
    F1 = np.zeros((5, 3))
    F1[:, 0] = 1.0
    F2 = np.zeros((5, 3))
    F2[:, 1] = 2.0
    a = LoadHistory(t, F1)
    b = LoadHistory(t, F2)
    c = a + b
    np.testing.assert_allclose(c.load_values[:, 0], 1.0)
    np.testing.assert_allclose(c.load_values[:, 1], 2.0)
    np.testing.assert_allclose(c.load_values[:, 2], 0.0)


# ---------------------------------------------------------------------------
# from_ground_motion must NOT exist (was removed by design)
# ---------------------------------------------------------------------------


def test_from_ground_motion_attribute_removed():
    """`from_ground_motion` was removed; ground motion handling lives in
    SDF/MDF classes via the influence-vector convention."""
    assert not hasattr(LoadHistory, "from_ground_motion")


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr_contains_key_info():
    t = np.linspace(0, 1.0, 11)
    lh = LoadHistory(t, np.ones(11), dof=[0, 2])
    s = repr(lh)
    assert "LoadHistory" in s
    assert "dof" in s
    assert "dt" in s
