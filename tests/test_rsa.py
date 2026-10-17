"""
Tests for enhanced Response Spectrum Analysis (RSA).
"""

import numpy as np
import pytest
from structdyn.mdf.mdf import MDF
from structdyn.mdf.analytical_methods.response_spectrum_analysis import (
    ResponseSpectrumAnalysis,
)


def test_rsa_chopra_ex_13_8_2_exact():
    """
    Chopra 5th ed, Example 13.8.2.
    5-story shear building with El Centro spectrum.
    """
    masses = np.ones(5) * 45000  # kg
    stiffness = np.ones(5) * 54.82e5  # N/m
    mdf = MDF.from_shear_building(masses, stiffness)

    Sd_chopra = [0.13649, 0.06557, 0.03823, 0.02227, 0.01657]
    rsa = ResponseSpectrumAnalysis(mdf, ji=0.05, Sd=Sd_chopra)

    # Displacements
    u = rsa.modal_displacements()
    assert u.shape == (5, 5)

    # Base shear
    vb = rsa.modal_base_shear()
    combined_vb = rsa.combine_modal_responses(vb, method="SRSS")

    # Chopra result is 291.2 kN = 291222 N
    assert combined_vb[0] == pytest.approx(291221.9, rel=1e-3)


def test_rsa_continuous_spectrum_curve():
    """Test user-supplied continuous response spectrum curves (callable, tuple, dict)."""
    masses = [10000, 10000]
    stiffness = [1e6, 1e6]
    mdf = MDF.from_shear_building(masses, stiffness)

    # 1. Callable: constant acceleration Sa = 0.3g
    g = 9.80665
    rsa_callable = ResponseSpectrumAnalysis(
        mdf, ji=0.05, spectrum_curve=lambda t: 0.3 * g
    )
    assert np.allclose(rsa_callable.Sa, 0.3 * g)
    assert np.allclose(rsa_callable.Sd, (0.3 * g) / (rsa_callable.omega ** 2))

    # 2. Tuple (T, Sa)
    T_pts = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    Sa_pts = np.array([0.2, 0.5, 0.5, 0.25, 0.125]) * g
    rsa_tuple = ResponseSpectrumAnalysis(
        mdf, ji=0.05, spectrum_curve=(T_pts, Sa_pts)
    )
    expected_sa = np.interp(rsa_tuple.T, T_pts, Sa_pts)
    assert np.allclose(rsa_tuple.Sa, expected_sa)

    # 3. Dict {"T": ..., "Sa": ...}
    rsa_dict = ResponseSpectrumAnalysis(
        mdf, ji=0.05, spectrum_curve={"T": T_pts, "Sa": Sa_pts}
    )
    assert np.allclose(rsa_dict.Sa, expected_sa)


def test_rsa_asce7_spectrum_parameters():
    """Test ASCE 7 parameterized response spectrum (SDS, SD1, TL)."""
    masses = [20000, 20000, 20000]
    stiffness = [2e6, 2e6, 2e6]
    mdf = MDF.from_shear_building(masses, stiffness)

    from struct_load import SeismicParameters

    sds = 1.0
    sd1 = 0.6
    tl = 8.0
    params = SeismicParameters(sds=sds, sd1=sd1, tl=tl, site_class="D", damping_ratio=0.05)
    rsa_asce7 = ResponseSpectrumAnalysis(
        mdf, params=params, scale_factor=9.80665
    )

    t0 = 0.2 * sd1 / sds
    ts = sd1 / sds
    for t_val, sa_val in zip(rsa_asce7.T, rsa_asce7.Sa):
        if t_val < t0:
            expected = sds * (0.4 + 0.6 * (t_val / t0)) * 9.80665
        elif t_val <= ts:
            expected = sds * 9.80665
        else:
            expected = (sd1 / t_val) * 9.80665
        assert sa_val == pytest.approx(expected, rel=1e-5)


def test_rsa_cqc_combination_and_properties():
    """Verify CQC correlation matrix properties and compare against SRSS."""
    masses = [15000, 15000, 15000]
    stiffness = [1.5e6, 1.5e6, 1.5e6]
    mdf = MDF.from_shear_building(masses, stiffness)

    rsa = ResponseSpectrumAnalysis(mdf, ji=0.05, Sa=[1.0, 1.0, 1.0])
    rho = rsa.compute_cqc_matrix()

    # Diagonal must be 1.0
    assert np.allclose(np.diag(rho), 1.0)
    # Symmetric
    assert np.allclose(rho, rho.T)
    # Off-diagonals must be strictly between 0 and 1
    assert np.all((rho >= 0.0) & (rho <= 1.0))

    vb = rsa.modal_base_shear()
    srss_vb = rsa.combine_modal_responses(vb, method="SRSS")
    cqc_vb = rsa.combine_modal_responses(vb, method="CQC")

    # For well-spaced modes, CQC is very close to SRSS
    assert cqc_vb[0] == pytest.approx(srss_vb[0], rel=0.05)


def test_rsa_mass_participation():
    """Verify that sum of modal effective masses equals total mass for a shear building."""
    masses = [10000, 12000, 15000]
    stiffness = [2e6, 2e6, 2e6]
    mdf = MDF.from_shear_building(masses, stiffness)

    rsa = ResponseSpectrumAnalysis(mdf, ji=0.05, Sa=[1.0, 1.0, 1.0])

    total_mass = sum(masses)
    sum_eff_mass = np.sum(rsa.effective_mass)
    sum_mmpr = np.sum(rsa.mass_participation_ratio)

    assert sum_eff_mass == pytest.approx(total_mass, rel=1e-5)
    assert sum_mmpr == pytest.approx(1.0, rel=1e-5)
