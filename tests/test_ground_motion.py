"""
Unit tests for GroundMotion database access, discovery, spectrum generation, and RSA integration.
"""

import numpy as np
import pytest
from structdyn import GroundMotion, MDF, ResponseSpectrumAnalysis


def test_ground_motion_available_events():
    """Verify available_events discovers all built-in PEER events and Chopra CSV."""
    events = GroundMotion.available_events()
    assert isinstance(events, dict)
    assert "elcentro_chopra" in events
    assert "imperialValley_elCentro_1940" in events
    assert "lomaPrieta_corralitos_1989" in events
    assert "northridge_sylmar_1994" in events
    assert "sanFernando_pacoidaDam_1971" in events

    assert events["elcentro_chopra"] == ["csv"]
    for peer_event in [
        "imperialValley_elCentro_1940",
        "lomaPrieta_corralitos_1989",
        "northridge_sylmar_1994",
        "sanFernando_pacoidaDam_1971",
    ]:
        comps = events[peer_event]
        assert "hor1" in comps
        assert "hor2" in comps
        assert "up" in comps


def test_ground_motion_from_elcentro_chopra():
    """Verify GroundMotion.from_elcentro_chopra loads Chopra's record directly."""
    gm = GroundMotion.from_elcentro_chopra()
    assert gm.name == "elcentro_chopra"
    assert gm.dt == pytest.approx(0.02)
    assert len(gm.acc_g) == 1560
    assert gm.duration == pytest.approx(1559 * 0.02)
    assert gm.pga_g == pytest.approx(0.3188, rel=1e-3)
    assert gm.pga == pytest.approx(0.3188 * 9.81, rel=1e-3)


def test_ground_motion_from_event_peer_records():
    """Verify loading PEER records via aliases and canonical names."""
    # Northridge
    gm_nr = GroundMotion.from_event("northridge", component="hor1")
    assert "northridge" in gm_nr.name.lower()
    assert gm_nr.dt > 0
    assert len(gm_nr.acc_g) > 100
    assert gm_nr.pga_g > 0.05

    # Loma Prieta
    gm_lp = GroundMotion.from_event("loma_prieta", component="hor1")
    assert "lomaprieta" in gm_lp.name.lower()
    assert gm_lp.dt > 0

    # San Fernando
    gm_sf = GroundMotion.from_event("san_fernando", component="hor1")
    assert "sanfernando" in gm_sf.name.lower()
    assert gm_sf.dt > 0

    # Imperial Valley / El Centro
    gm_elc = GroundMotion.from_event("elcentro", component="hor1")
    assert "imperialvalley" in gm_elc.name.lower()

    # Chopra via from_event
    gm_ch = GroundMotion.from_event("chopra")
    assert gm_ch.name == "elcentro_chopra"
    assert len(gm_ch.acc_g) == 1560


def test_ground_motion_component_resolution():
    """Verify component aliases ('hor1', 'h1', '1', 'hor2', 'h2', '2', 'up', 'v')."""
    gm_h1 = GroundMotion.from_event("northridge", component="h1")
    gm_h2 = GroundMotion.from_event("northridge", component="h2")
    gm_v = GroundMotion.from_event("northridge", component="v")

    assert gm_h1.pga_g != gm_h2.pga_g
    assert gm_v.pga_g > 0


def test_ground_motion_response_spectrum():
    """Verify GroundMotion.response_spectrum computes elastic response spectrum DataFrame."""
    gm = GroundMotion.from_event("northridge")
    periods = np.array([0.1, 0.5, 1.0, 2.0])
    df = gm.response_spectrum(periods=periods, damping_ratio=0.05)

    assert "T" in df.columns
    assert "Sd" in df.columns
    assert "pSv" in df.columns
    assert "pSa" in df.columns
    assert "pSa (g)" in df.columns
    assert len(df) == len(periods)
    assert np.all(df["Sd"] > 0)
    assert np.all(df["pSa"] > 0)


def test_ground_motion_to_core():
    """Verify GroundMotion.to_core returns a valid struct_core.GroundMotion model."""
    gm = GroundMotion.from_elcentro_chopra()
    core_gm = gm.to_core(id="ElCentro_Chopra")

    assert core_gm.id == "ElCentro_Chopra"
    assert core_gm.dt == pytest.approx(0.02)
    assert len(core_gm.acc) == 1560


def test_rsa_with_string_ground_motion_name():
    """Verify ResponseSpectrumAnalysis accepts a string event name directly."""
    masses = np.ones(3) * 20000
    stiffness = np.ones(3) * 2e6
    mdf = MDF.from_shear_building(masses, stiffness)

    # 1. Using "elcentro" string (direction="x" -> hor1)
    rsa_elc = ResponseSpectrumAnalysis(mdf, ji=0.05, gm="elcentro", direction="x")
    assert rsa_elc.gm.component == "hor1"
    vb_elc = rsa_elc.modal_base_shear()
    assert vb_elc.shape == (1, 3)
    comb_vb_elc = rsa_elc.combine_modal_responses(vb_elc, method="SRSS")
    assert float(comb_vb_elc[0]) > 0

    # 2. Using "northridge" string with explicit direction="y" -> hor2
    rsa_nr_y = ResponseSpectrumAnalysis(mdf, ji=0.05, gm="northridge", direction="y")
    assert rsa_nr_y.gm.component == "hor2"
    vb_nr_y = rsa_nr_y.modal_base_shear()
    assert vb_nr_y.shape == (1, 3)

    # 3. Using composite syntax "northridge:hor2"
    rsa_comp_str = ResponseSpectrumAnalysis(mdf, ji=0.05, gm="northridge:hor2")
    assert rsa_comp_str.gm.component == "hor2"

    # 4. Using explicit gm_component parameter
    rsa_comp_param = ResponseSpectrumAnalysis(
        mdf, ji=0.05, gm="northridge", gm_component="hor2"
    )
    assert rsa_comp_param.gm.component == "hor2"


def test_ground_motion_invalid_event_raises():
    """Verify requesting a non-existent event raises a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Available events"):
        GroundMotion.from_event("non_existent_earthquake_xyz")
