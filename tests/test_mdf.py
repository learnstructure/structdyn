"""
Test suite for the MDF (Multi Degree of Freedom) class.
Examples are taken from Chopra, A. K., "Dynamics of Structures", 5th Edition.
"""

import numpy as np
import pytest
from structdyn.mdf.mdf import MDF
from structdyn.ground_motions.ground_motion import GroundMotion


def test_example_10_4_modal_analysis():
    """
    Example 10.4: 2‑story shear building.
    Natural frequencies should be [0.70710678, 1.41421356] rad/s.
    """
    masses = [2, 1]
    stiffness = [2, 1]
    mdf = MDF.from_shear_building(masses, stiffness)

    omega, phi = mdf.modal.modal_analysis(dof_normalize=-1)

    expected = np.array([0.70710678, 1.41421356])
    assert omega == pytest.approx(expected, rel=1e-5)


def test_example_10_9_free_vibration():
    """
    Example 10.9: Free vibration of a 2‑story shear building with initial
    displacements [0.5, 1]. The maximum absolute displacement of the first floor
    should be 0.5.
    """
    masses = [2, 1]
    stiffness = [2, 1]
    mdf = MDF.from_shear_building(masses, stiffness)

    mdf.modal.modal_analysis(dof_normalize=-1)  # needed to compute modes

    u0 = [0.5, 1]
    v0 = [0, 0]
    time = np.arange(0, 20, 0.01)

    response = mdf.modal.free_vibration_response(u0, v0, time)
    max_u1 = response["u1"].abs().max()

    expected = 0.5
    assert max_u1 == pytest.approx(expected, abs=1e-10)


def test_example_11_4_modal_damping():
    """
    Example 11.4: 3‑story shear building with modal damping ratios 5% each.
    The last element of the damping matrix should be 287983.44117400126.
    """
    masses = [2e5, 2e5, 1e5]
    stiffness = [1e8, 1e8, 1e8]
    mdf = MDF.from_shear_building(masses, stiffness)

    mdf.set_modal_damping(zeta=[0.05, 0.05, 0.05])

    expected_last = 287983.44117400126
    assert mdf.C.flatten()[-1] == pytest.approx(expected_last, rel=1e-5)


def test_example_16_1_ground_motion_modal():
    """
    Example 16.1: 5‑story shear building subjected to a half‑cycle sine pulse.
    Response computed with Newmark's method using modal superposition (2 modes).
    The displacement of the top floor (u5) at t = 2.0 s (index 20) should be 0.05888401249961952.
    """
    # Building definition
    masses = np.ones(5) * 0.2591e5
    stiffness = np.ones(5) * 100e5
    mdf = MDF.from_shear_building(masses, stiffness)
    mdf.set_modal_damping(zeta=np.ones(5) * 0.05)

    # Ground motion: half‑cycle sine pulse
    time = np.arange(0, 2.01, 0.1)
    acc = (1.93 / 9.81) * np.sin(2 * np.pi * time)
    acc[time > 1.0] = 0
    gm = GroundMotion.from_arrays(acc, 0.1)

    # Influence vector (all DOFs excited equally)
    inf_vec = np.ones(5)

    # Response analysis
    response = mdf.find_response_ground_motion(
        gm, inf_vec, method="newmark_beta", use_modal=True, n_modes=2
    )

    expected = 0.05888401249961952
    assert response["u5"][20] == pytest.approx(expected, rel=1e-5)
