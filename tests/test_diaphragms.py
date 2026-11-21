"""Verification tests for Milestone 2 rigid floor diaphragms in structdyn."""

import numpy as np
import pytest
from scipy.linalg import eigh

from fem3d import BeamElement, ElasticMaterial, Node, PointLoad, Section, Structure
from structdyn.ground_motions import GroundMotion
from structdyn.mdf import mdf as mdf_module
from structdyn.mdf.mdf import MDF
from structdyn.mdf.analytical_methods.response_spectrum_analysis import ResponseSpectrumAnalysis


def _eccentric_mdf_structure():
    E = 2.5e7
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))
    H = 4.0
    Lx, Ly = 6.0, 5.0
    specs = [
        (0.0, 0.0, 3.0e-3, 8.0e-4, 24.0, 10.0),
        (Lx, 0.0, 1.2e-3, 4.0e-4, 14.0, 5.0),
        (Lx, Ly, 0.9e-3, 3.0e-4, 10.0, 4.0),
        (0.0, Ly, 2.0e-3, 6.0e-4, 20.0, 8.0),
    ]
    s = Structure(units="kN, m")
    mat = ElasticMaterial(E=E, nu=nu, rho=0.0)
    top_nodes = []
    for idx, (x, y, I_b, J_t, m_val, iz_val) in enumerate(specs, start=1):
        bn = Node(idx, x, y, 0.0)
        bn.set_support(True, True, True, True, True, True)
        tn = Node(10 + idx, x, y, H)
        tn.set_support(uz_fixed=True, rx_fixed=True, ry_fixed=True)
        tn.set_mass(mass=[m_val, m_val, m_val], inertia=[0.0, 0.0, iz_val])
        s.add_node(bn)
        s.add_node(tn)
        sec = Section(A=0.25, I2=I_b, I3=I_b, J=J_t, material=mat)
        s.add_element(BeamElement(f"C{idx}", bn, tn, mat, sec, include_shear_deformation=False))
        top_nodes.append(tn)
    # Add an in-plane beam between top nodes 11 and 12 (axial force must vanish in RSA)
    bm_sec = Section(A=0.18, I2=1.0e-3, I3=2.0e-3, J=2.0e-4, material=mat)
    s.add_element(BeamElement("B_TOP", top_nodes[0], top_nodes[1], mat, bm_sec, include_shear_deformation=False))
    s.add_diaphragm("D1", [n.id for n in top_nodes], axis="Z")
    return s, top_nodes


def _multistory_diaphragm_structure():
    s = Structure(units="kN, m")
    mat = ElasticMaterial(E=2.5e7, nu=0.2, rho=2.4)
    col = Section.from_rectangle(0.4, 0.4, mat)
    bm = Section.from_rectangle(0.3, 0.5, mat)
    nodes = {}
    for k in range(4):
        z = 3.5 * k
        for idx, (x, y) in enumerate([(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)], start=1):
            nid = 10 * k + idx
            nd = Node(nid, x, y, z)
            if k == 0:
                nd.set_support(True, True, True, True, True, True)
            else:
                nd.set_mass(mass=8.0 if idx == 1 else 4.0)
            s.add_node(nd)
            nodes[nid] = nd
    for k in range(3):
        for idx in range(1, 5):
            s.add_element(BeamElement(f"C{k}_{idx}", nodes[10 * k + idx], nodes[10 * (k + 1) + idx], mat, col))
        b = 10 * (k + 1)
        s.add_element(BeamElement(f"BX1_{k}", nodes[b + 1], nodes[b + 2], mat, bm))
        s.add_element(BeamElement(f"BX2_{k}", nodes[b + 4], nodes[b + 3], mat, bm))
        s.add_element(BeamElement(f"BY1_{k}", nodes[b + 1], nodes[b + 4], mat, bm))
        s.add_element(BeamElement(f"BY2_{k}", nodes[b + 2], nodes[b + 3], mat, bm))
        s.add_diaphragm(f"D{k + 1}", [b + 1, b + 2, b + 3, b + 4], axis="Z")
    return s


def test_mdf_from_fem3d_scalar_zeta_and_rsa_beam_axial_force():
    s, _ = _eccentric_mdf_structure()
    mdf = MDF.from_fem3d(s, zeta=0.05, lumped=True)
    assert mdf.ndof == s.n_reduced_dofs == 3
    assert mdf.C.shape == (3, 3)

    rsa = ResponseSpectrumAnalysis(mdf, ji=0.05, Sa=[1.0, 0.8, 0.5], direction="x")
    assert rsa.mass_participation_ratio.sum() == pytest.approx(1.0, rel=1e-12)
    forces = rsa.compute_modal_element_forces(method="CQC")
    # In-plane beam inside rigid diaphragm carries zero modal and combined axial force
    assert np.max(np.abs(forces["modal"]["B_TOP"][0, :])) < 1e-10
    assert abs(forces["combined"]["B_TOP"][0]) < 1e-10
    assert abs(forces["combined"]["B_TOP"][6]) < 1e-10


def test_sparse_and_dense_modal_analysis_agree_on_diaphragm_model(monkeypatch):
    s_dense = _multistory_diaphragm_structure()
    mdf_dense = MDF.from_fem3d(s_dense, lumped=True)
    assert mdf_dense.is_sparse is False
    omega_dense, _ = mdf_dense.modal.modal_analysis(n_modes=6)

    s_sparse = _multistory_diaphragm_structure()
    monkeypatch.setattr(mdf_module, "DENSE_DOF_LIMIT", 1)
    mdf_sparse = MDF.from_fem3d(s_sparse, zeta=0.05, lumped=True)
    assert mdf_sparse.is_sparse is True
    assert len(mdf_sparse.modal_damping_ratios) == mdf_sparse.ndof
    omega_sparse, phi_sparse = mdf_sparse.modal.modal_analysis(n_modes=6)

    np.testing.assert_allclose(omega_sparse, omega_dense[:6], rtol=1e-10)
    full = s_sparse.expand_displacements(phi_sparse)
    assert full.shape == (s_sparse.neq, 6)


def test_find_response_ground_motion_diaphragm_time_history():
    s, top_nodes = _eccentric_mdf_structure()
    s.add_load(PointLoad(top_nodes[0], fx=0.0))  # ensure valid load assembly
    mdf = MDF.from_fem3d(s, zeta=0.0, lumped=True)

    # Sinusoidal ground acceleration x_g''(t) = a0 * sin(w_f * t)
    dt = 0.0005
    time = np.arange(0.0, 0.4 + dt / 2.0, dt)
    w_f = 15.0
    a0 = 2.0
    ag = a0 * np.sin(w_f * time)
    gm = GroundMotion(acc_g=ag, dt=dt, scale_factor=1.0)

    df = mdf.find_response_ground_motion(gm, direction="x", method="newmark_beta")
    q_cols = [f"u{i + 1}" for i in range(mdf.ndof)]
    q_hist = df[q_cols].to_numpy()  # (nt, 3)

    # Analytical undamped modal superposition for sin(w_f * t)
    omega, phi = mdf.modal.modal_analysis(n_modes=3)
    inf_vec = s.influence_vector("x")
    gamma = np.array([(phi[:, m] @ mdf.M @ inf_vec) / (phi[:, m] @ mdf.M @ phi[:, m]) for m in range(3)])
    q_exact = np.zeros_like(q_hist)
    for m in range(3):
        wn = omega[m]
        # Solution to D'' + wn^2 D = -gamma_m * a0 * sin(w_f * t), D(0)=D'(0)=0
        D_t = (-gamma[m] * a0 / (wn**2 - w_f**2)) * (np.sin(w_f * time) - (w_f / wn) * np.sin(wn * time))
        q_exact += D_t[:, None] * phi[:, m][None, :]

    # Newmark average-acceleration matches exact continuous response within small time-step error
    np.testing.assert_allclose(q_hist, q_exact, rtol=2e-3, atol=1e-6)

    # Expand to full nodal displacements at every time step and verify rigid-body kinematics
    u_full = s.expand_displacements(q_hist.T).T  # (nt, neq)
    n0 = top_nodes[0]
    u0 = u_full[:, n0.dofs]
    for tn in top_nodes[1:]:
        ui = u_full[:, tn.dofs]
        np.testing.assert_allclose(ui[:, 0], u0[:, 0] - (tn.y - n0.y) * u0[:, 5], atol=1e-13)
        np.testing.assert_allclose(ui[:, 1], u0[:, 1] + (tn.x - n0.x) * u0[:, 5], atol=1e-13)
        np.testing.assert_allclose(ui[:, 5], u0[:, 5], atol=1e-13)
