"""
Test suite for the SDF (Single Degree of Freedom) class.
Examples are taken from Chopra, A. K., "Dynamics of Structures", 5th Edition.
"""

import numpy as np
import pytest
from structdyn import SDF
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.sdf.response_spectrum import ResponseSpectrum
from structdyn.utils.helpers import elcentro_chopra


def test_sdf_initialization():
    """Test the basic initialization of an SDF object."""
    sdf = SDF(m=1000, k=10000, ji=0.05)
    assert sdf.m == 1000
    assert sdf.k == 10000
    assert sdf.ji == 0.05
    assert sdf.w_n == pytest.approx(np.sqrt(10))
    assert sdf.t_n == pytest.approx(2 * np.pi / np.sqrt(10))
    assert sdf.c == pytest.approx(2 * 1000 * np.sqrt(10) * 0.05)


def test_example_5_1_interpolation():
    """
    Example 5.1: Linear SDF, interpolation method.
    Displacement at t = 1.0 s (index 10) should be -0.034534260954800985.
    """
    dt = 0.1
    time_steps = np.arange(0, 1.01, dt)
    load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
    load_values[time_steps >= 0.6] = 0

    sdf = SDF(45594, 18 * 10**5, 0.05)
    responses = sdf.find_response(time_steps, load_values, method="interpolation")

    expected = -0.034534260954800985
    assert responses["displacement"][10] == pytest.approx(expected, rel=1e-5)


def test_example_5_2_central_difference():
    """
    Example 5.2: Linear SDF, central difference method.
    Displacement at t = 1.0 s (index 10) should be -0.03812718853606678.
    """
    dt = 0.1
    time_steps = np.arange(0, 1.01, dt)
    load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
    load_values[time_steps >= 0.6] = 0

    sdf = SDF(45594, 18 * 10**5, 0.05)
    responses = sdf.find_response(time_steps, load_values, method="central_difference")

    expected = -0.03812718853606678
    assert responses["displacement"][10] == pytest.approx(expected, rel=1e-5)


def test_example_5_4_newmark_linear():
    """
    Example 5.4: Linear SDF, Newmark's method with linear acceleration (γ=1/2, β=1/6).
    Displacement at t = 1.0 s (index 10) should be -0.03391195470077432.
    """
    dt = 0.1
    time_steps = np.arange(0, 1.01, dt)
    load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
    load_values[time_steps >= 0.6] = 0

    sdf = SDF(45594, 18 * 10**5, 0.05)
    responses = sdf.find_response(
        time_steps, load_values, method="newmark_beta", acc_type="linear"
    )

    expected = -0.03391195470077432
    assert responses["displacement"][10] == pytest.approx(expected, rel=1e-5)


def test_example_5_5_newmark_average_nonlinear():
    """
    Example 5.5: Elastoplastic SDF, Newmark's method with constant average acceleration (γ=1/2, β=1/4).
    Displacement at t = 1.0 s (index 10) should be 0.03606328101158249.
    """
    dt = 0.1
    time_steps = np.arange(0, 1.01, dt)
    load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
    load_values[time_steps >= 0.6] = 0

    sdf = SDF(45594, 18 * 10**5, 0.05, fd="elastoplastic", uy=0.02, fy=36000)
    responses = sdf.find_response(
        time_steps, load_values, method="newmark_beta", acc_type="average"
    )

    expected = 0.03606328101158249
    assert responses["displacement"][10] == pytest.approx(expected, rel=1e-5)


def test_section_6_4_elcentro_interpolation():
    """
    Section 6.4 (& Figure 6.4.1a): SDF (Tn=0.5 s, ζ=2%) subjected to El Centro ground motion,
    using interpolation method. The maximum absolute displacement should be 0.0679400697200714.
    """
    elc = elcentro_chopra()
    gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

    m = 1.0
    ji = 0.02
    t_n = 0.5
    w_n = 2 * np.pi / t_n
    k = w_n**2 * m
    sdf = SDF(m, k, ji)

    results = sdf.find_response_ground_motion(gm, method="interpolation")
    max_disp = results["displacement"].abs().max()

    expected = 0.0679400697200714
    assert max_disp == pytest.approx(expected, rel=1e-5)


def test_section_6_4_response_spectrum():
    """
    Section 6.4 (& Figure 6.6.2): Response spectrum for El Centro ground motion,
    damping ratio 2%. The spectral displacement at period = 2.0 s (index 20 of periods array)
    should be 0.1896749378231744.
    """
    elc = elcentro_chopra()
    gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

    periods = np.arange(0, 5.01, 0.1)  # 0.0 to 5.0 inclusive
    rs = ResponseSpectrum(periods, 0.02, gm)
    spectra = rs.compute()

    # Index 20 corresponds to period = 2.0 s (since step = 0.1)
    expected = 0.1896749378231744
    assert spectra["Sd"][20] == pytest.approx(expected, rel=1e-5)
