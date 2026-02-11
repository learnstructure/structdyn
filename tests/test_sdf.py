import numpy as np
import pytest
from structdyn.sdf.sdf import SDF


def test_sdf_initialization():
    """Test the initialization of the SDF class."""
    sdf = SDF(m=1000, k=10000, ji=0.05)
    assert sdf.m == 1000
    assert sdf.k == 10000
    assert sdf.ji == 0.05
    assert sdf.w_n == pytest.approx(np.sqrt(10))
    assert sdf.t_n == pytest.approx(2 * np.pi / np.sqrt(10))
    assert sdf.c == pytest.approx(2 * 1000 * np.sqrt(10) * 0.05)


def test_find_response():
    """Test the find_response method of the SDF class."""
    sdf = SDF(m=45594, k=18e5, ji=0.05)
    time = np.arange(0, 1.01, 0.1)
    load = 50 * np.sin(np.pi * time / 0.6) * 1000
    load[time >= 0.6] = 0
    results = sdf.find_response(time, load, method="newmark_beta")
    assert "displacement" in results.columns
    assert "velocity" in results.columns
    assert "acceleration" in results.columns
    assert len(results) == len(time)
