"""Verify structdyn against closed-form solutions, exact ODE solutions, a textbook result, and OpenSeesPy.

Run from the repository root:

    python paper/verification/verify.py

Writes, next to this script:
    results.csv     one row per check: structdyn value, reference value, relative difference
    results.md      the same as a Markdown table
    histories.npz   response histories used by plot_figure.py

References:
    closed form   exact eigenvalues of small systems
    exact ODE     scipy.integrate.solve_ivp (DOP853, rtol = 1e-12) of the equation of motion under the same
                  piecewise-linear excitation; the interpolation method is exact for such excitation
    textbook      Chopra, Dynamics of Structures, 5th ed., Example 13.8.2
    OpenSeesPy    the same integration scheme and time step in OpenSees (zeroLength springs, Rayleigh damping
                  with the initial stiffness, Newton-Raphson iteration for the nonlinear cases); structdyn iterates
                  to its default residual-force tolerance of 1e-6 N, OpenSees to a displacement increment of 1e-12 m
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from structdyn import MDF, SDF, Bilinear, ElasticPerfectlyPlastic, GroundMotion, LoadHistory
from structdyn.mdf.analytical_methods.response_spectrum_analysis import ResponseSpectrumAnalysis
from structdyn.mdf.mdf_helpers.element_models import ShearStoryElement


def _import_opensees():
    """Import OpenSeesPy; on Windows, add the Intel Fortran runtime of the intel-fortran-rt package to the DLL path."""
    if sys.platform == "win32":
        lib = Path(sys.prefix) / "Library" / "bin"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
    import openseespy.opensees as ops

    return ops


ops = _import_opensees()

HERE = Path(__file__).resolve().parent
G = 9.81  # m/s^2, conversion of the records from g, used identically in structdyn and OpenSees
OPS_TOL = 1e-12  # m, OpenSees displacement-increment tolerance

rows: list[dict] = []
histories: dict[str, np.ndarray] = {}


def record(case, reference, quantity, value, ref_value, note=""):
    rows.append({"case": case, "reference": reference, "quantity": quantity, "structdyn": value,
                 "reference_value": ref_value, "rel_diff": abs(value - ref_value) / abs(ref_value), "note": note})


def history_diff(u, u_ref):
    """Largest difference over the history, relative to the peak of the reference."""
    return float(np.max(np.abs(np.asarray(u) - np.asarray(u_ref))) / np.max(np.abs(u_ref)))


def record_history(case, reference, u, u_ref, note=""):
    peak, peak_ref = float(np.max(np.abs(u))), float(np.max(np.abs(u_ref)))
    rows.append({"case": case, "reference": reference, "quantity": "peak displacement (m)", "structdyn": peak,
                 "reference_value": peak_ref, "rel_diff": abs(peak - peak_ref) / peak_ref,
                 "max_hist_diff": history_diff(u, u_ref), "note": note})


# ----------------------------------------------------------------------------------------------------------------
# Excitations
# ----------------------------------------------------------------------------------------------------------------
# Chopra Examples 5.1-5.5: m = 45,594 kg, k = 1.8e6 N/m (Tn = 1.0 s), zeta = 5 %, half-sine pulse of 50 kN, dt = 0.1 s
M5, K5, ZETA5, DT5 = 45594.0, 18e5, 0.05, 0.1
T5 = np.arange(0, 1.01, DT5)
P5 = 50e3 * np.sin(np.pi * T5 / 0.6)
P5[T5 >= 0.6] = 0.0

ELC = GroundMotion.from_elcentro_chopra(scale_factor=G)  # El Centro 1940 S00E, Chopra Appendix 6, dt = 0.02 s
AG_G = np.asarray(ELC.acc_g, dtype=float)
DT_ELC = float(ELC.dt)


# ----------------------------------------------------------------------------------------------------------------
# OpenSeesPy models
# ----------------------------------------------------------------------------------------------------------------
def ops_sdf(m, k, c, dt, n_steps, integrator, material=("Elastic",), force=None, ag_g=None):
    """SDF system as a zeroLength spring; returns displacement and spring force histories."""
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    ops.node(1, 0.0)
    ops.node(2, 0.0)
    ops.fix(1, 1)
    ops.mass(2, m)
    ops.uniaxialMaterial(material[0], 1, *material[1:])
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)
    ops.rayleigh(c / m, 0.0, 0.0, 0.0)  # C = (c/m) M = c
    if force is not None:
        ops.timeSeries("Path", 1, "-dt", dt, "-values", *force)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 1.0)
    else:
        ops.timeSeries("Path", 1, "-dt", dt, "-values", *ag_g, "-factor", G)
        ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    if integrator[0] == "CentralDifference":
        ops.algorithm("Linear")
    else:
        ops.test("NormDispIncr", OPS_TOL, 100)
        ops.algorithm("Newton")
    ops.integrator(*integrator)
    ops.analysis("Transient")
    u, fs = [0.0], [0.0]
    for i in range(n_steps):
        if ops.analyze(1, dt) != 0:
            raise RuntimeError(f"OpenSees did not converge at step {i + 1}")
        u.append(ops.nodeDisp(2, 1))
        fs.append(ops.eleResponse(1, "force")[1])
    return np.array(u), np.array(fs)


def ops_shear_building(masses, k0, a0, a1, ag_g, dt, fy=None, alpha=None):
    """Shear building with elastic (fy None) or Steel01 story springs; returns floor displacements and the periods.

    zeroLength elements take part in Rayleigh damping only with -doRayleigh 1.
    """
    n = len(masses)
    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)
    for j in range(n + 1):
        ops.node(j, 0.0)
    ops.fix(0, 1)
    for j in range(1, n + 1):
        ops.mass(j, masses[j - 1])
        if fy is None:
            ops.uniaxialMaterial("Elastic", j, k0[j - 1])
        else:
            ops.uniaxialMaterial("Steel01", j, fy[j - 1], k0[j - 1], alpha)
        ops.element("zeroLength", j, j - 1, j, "-mat", j, "-dir", 1, "-doRayleigh", 1)
    periods = 2 * np.pi / np.sqrt(np.array(ops.eigen("-fullGenLapack", n)))
    ops.rayleigh(a0, 0.0, a1, 0.0)  # C = a0 M + a1 K_initial
    ops.timeSeries("Path", 1, "-dt", dt, "-values", *ag_g, "-factor", G)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", OPS_TOL, 100)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")
    u = [np.zeros(n)]
    for i in range(len(ag_g) - 1):
        if ops.analyze(1, dt) != 0:
            raise RuntimeError(f"OpenSees did not converge at step {i + 1}")
        u.append(np.array([ops.nodeDisp(j, 1) for j in range(1, n + 1)]))
    return np.array(u), periods


# ----------------------------------------------------------------------------------------------------------------
# Exact solution for piecewise-linear excitation
# ----------------------------------------------------------------------------------------------------------------
def exact_sdf(m, k, c, t, p):
    """Solve m u'' + c u' + k u = p(t), p linear between samples, segment by segment with DOP853."""
    y = np.zeros(2)
    u = [0.0]
    for i in range(len(t) - 1):
        t0, t1, p0, p1 = t[i], t[i + 1], p[i], p[i + 1]

        def rhs(tt, yy, t0=t0, t1=t1, p0=p0, p1=p1):
            pt = p0 + (p1 - p0) * (tt - t0) / (t1 - t0)
            return [yy[1], (pt - c * yy[1] - k * yy[0]) / m]

        sol = solve_ivp(rhs, (t0, t1), y, method="DOP853", rtol=1e-12, atol=1e-15)
        y = sol.y[:, -1]
        u.append(y[0])
    return np.array(u)


def main() -> None:
    c5 = 2 * ZETA5 * np.sqrt(K5 * M5)

    # 1. Closed form: Chopra Example 10.4, m1 = 2m, m2 = m, k1 = 2k, k2 = k (m = k = 1)
    omega, _ = MDF.from_shear_building([2, 1], [2, 1]).modal.modal_analysis()
    record("Ex. 10.4: 2-story shear building", "closed form", "omega_1 (rad/s)", float(omega[0]), np.sqrt(0.5))
    record("Ex. 10.4: 2-story shear building", "closed form", "omega_2 (rad/s)", float(omega[1]), np.sqrt(2.0))

    # 2. Closed form: cantilever with a massless rotation, condensed exactly (omega = sqrt(3EI / mL^3))
    E, I, L, m = 30000.0, 100.0, 120.0, 2.5
    K = np.array([[12 * E * I / L**3, -6 * E * I / L**2], [-6 * E * I / L**2, 4 * E * I / L]])
    omega, _ = MDF(M=np.diag([m, 0.0]), K=K).modal.modal_analysis()
    record("Cantilever, massless rotation", "closed form", "omega_1 (rad/s)", float(omega[0]),
           np.sqrt(3 * E * I / (m * L**3)))

    # 3. Exact ODE: interpolation method (exact for piecewise-linear excitation)
    u_sd = SDF(M5, K5, ZETA5).find_response(LoadHistory(T5, P5), method="interpolation")["displacement"].to_numpy()
    record_history("Ex. 5.1: linear SDF, half-sine pulse, interpolation", "exact ODE", u_sd,
                   exact_sdf(M5, K5, c5, T5, P5))

    tn, zeta = 0.5, 0.02  # Chopra Section 6.4, Figure 6.4.1
    k = (2 * np.pi / tn) ** 2
    t_elc = np.arange(len(AG_G)) * DT_ELC
    u_sd = SDF(1.0, k, zeta).find_response_ground_motion(ELC, method="interpolation")["displacement"].to_numpy()
    record_history("Sec. 6.4: SDF (Tn = 0.5 s, zeta = 2 %), El Centro, interpolation", "exact ODE", u_sd,
                   exact_sdf(1.0, k, 2 * zeta * np.sqrt(k), t_elc, -AG_G * G))

    # 4. OpenSeesPy, same scheme and time step: Examples 5.2, 5.4, 5.5
    n5 = len(T5) - 1
    u_sd = SDF(M5, K5, ZETA5).find_response(LoadHistory(T5, P5), method="central_difference")["displacement"]
    u_op, _ = ops_sdf(M5, K5, c5, DT5, n5, ("CentralDifference",), ("Elastic", K5), force=P5)
    record_history("Ex. 5.2: linear SDF, central difference", "OpenSeesPy", u_sd.to_numpy(), u_op)

    u_sd = SDF(M5, K5, ZETA5).find_response(LoadHistory(T5, P5), method="newmark_beta", acc_type="linear")
    u_op, _ = ops_sdf(M5, K5, c5, DT5, n5, ("Newmark", 0.5, 1 / 6), ("Elastic", K5), force=P5)
    record_history("Ex. 5.4: linear SDF, Newmark linear acceleration", "OpenSeesPy",
                   u_sd["displacement"].to_numpy(), u_op)

    epp = ElasticPerfectlyPlastic(uy=0.02, fy=36000)
    u_sd = SDF(M5, K5, ZETA5, fd=epp).find_response(LoadHistory(T5, P5), method="newmark_beta", acc_type="average")
    u_op, _ = ops_sdf(M5, K5, c5, DT5, n5, ("Newmark", 0.5, 0.25), ("ElasticPP", K5, 0.02), force=P5)
    record_history("Ex. 5.5: elastoplastic SDF, Newmark average acceleration", "OpenSeesPy",
                   u_sd["displacement"].to_numpy(), u_op)

    # 5. OpenSeesPy: bilinear SDF under El Centro (Tn = 0.5 s, zeta = 5 %, fy = 25 % of the elastic peak force)
    tn, zeta, alpha = 0.5, 0.05, 0.05
    k = (2 * np.pi / tn) ** 2
    c = 2 * zeta * np.sqrt(k)
    u_lin = SDF(1.0, k, zeta).find_response_ground_motion(ELC, method="newmark_beta", acc_type="average")
    u_lin = u_lin["displacement"].to_numpy()
    u_op, _ = ops_sdf(1.0, k, c, DT_ELC, len(AG_G) - 1, ("Newmark", 0.5, 0.25), ("Elastic", k), ag_g=AG_G)
    record_history("Linear SDF (Tn = 0.5 s, zeta = 5 %), El Centro, Newmark average acceleration", "OpenSeesPy",
                   u_lin, u_op)
    fy = 0.25 * k * np.max(np.abs(u_lin))
    mat = Bilinear(uy=fy / k, fy=fy, alpha=alpha)
    res = SDF(1.0, k, zeta, fd=mat).find_response_ground_motion(ELC, method="newmark_beta", acc_type="average")
    u_sd = res["displacement"].to_numpy()
    replay = Bilinear(uy=fy / k, fy=fy, alpha=alpha)  # spring force at the converged displacements
    fs_sd = []
    for ui in u_sd:
        fs_sd.append(replay.trial_response(ui, 0.0, DT_ELC)[0])
        replay.commit_state(ui)
    u_op, fs_op = ops_sdf(1.0, k, c, DT_ELC, len(AG_G) - 1, ("Newmark", 0.5, 0.25), ("Steel01", fy, k, alpha),
                          ag_g=AG_G)
    record_history("Bilinear SDF (alpha = 5 %, fy = 0.25 f0), El Centro, Newmark average acceleration",
                   "OpenSeesPy", u_sd, u_op,
                   note=f"ductility {np.max(np.abs(u_sd)) / (fy / k):.1f}")
    histories.update(sdf_t=t_elc, sdf_u_sd=u_sd, sdf_fs_sd=np.array(fs_sd), sdf_u_op=u_op, sdf_fs_op=fs_op,
                     sdf_uy=fy / k, sdf_fy=fy)

    # 6. OpenSeesPy: Example 13.8.2 building with bilinear stories under El Centro
    n = 5
    masses, k0 = np.full(n, 45000.0), np.full(n, 54.82e5)
    linear = MDF.from_shear_building(masses, k0)
    omega, _ = linear.modal.modal_analysis()
    zeta = 0.05
    a0 = 2 * zeta * omega[0] * omega[1] / (omega[0] + omega[1])
    a1 = 2 * zeta / (omega[0] + omega[1])
    C = a0 * linear.M + a1 * linear.K
    cols = [f"u{j}" for j in range(1, n + 1)]
    u_lin = MDF(linear.M, linear.K, C).find_response_ground_motion(ELC, np.ones(n), method="newmark_beta",
                                                                   acc_type="average")
    u_op, periods_op = ops_shear_building(masses, k0, a0, a1, AG_G, DT_ELC)
    for i in range(n):
        record("Ex. 13.8.2 building: periods", "OpenSeesPy", f"T_{i + 1} (s)", float(2 * np.pi / omega[i]),
               float(periods_op[i]))
    record_history("Ex. 13.8.2 building, linear, El Centro, Newmark average acceleration: roof", "OpenSeesPy",
                   u_lin[cols].to_numpy()[:, -1], u_op[:, -1])
    rows[-1]["max_hist_diff_all_floors"] = history_diff(u_lin[cols].to_numpy(), u_op)
    drift = np.diff(np.column_stack([np.zeros(len(u_lin)), u_lin[cols].to_numpy()]),
                    axis=1)
    fy = 0.5 * k0 * np.max(np.abs(drift), axis=0)  # each story yields at half its elastic peak shear
    alpha = 0.05
    elements = [ShearStoryElement(Bilinear(uy=fy[0] / k0[0], fy=fy[0], alpha=alpha), dof_indices=[0])]
    elements += [ShearStoryElement(Bilinear(uy=fy[j] / k0[j], fy=fy[j], alpha=alpha), dof_indices=[j - 1, j])
                 for j in range(1, n)]
    res = MDF(linear.M, linear.K, C).find_response_ground_motion(ELC, np.ones(n), method="newmark_beta",
                                                                 elements=elements)
    u_sd = res[cols].to_numpy()
    u_op, _ = ops_shear_building(masses, k0, a0, a1, AG_G, DT_ELC, fy=fy, alpha=alpha)
    record_history("Ex. 13.8.2 building, bilinear stories (alpha = 5 %), El Centro: roof", "OpenSeesPy",
                   u_sd[:, -1], u_op[:, -1], note=f"peak story ductility {np.max(np.abs(np.diff(np.column_stack([np.zeros(len(u_sd)), u_sd]), axis=1)) / (fy / k0)):.1f}")
    rows[-1]["max_hist_diff_all_floors"] = history_diff(u_sd, u_op)
    histories.update(mdf_t=t_elc, mdf_u_sd=u_sd, mdf_u_op=u_op)

    # 7. Textbook: Example 16.4, five-story building with bilinear stories under a half-cycle sine pulse
    t16 = np.arange(0, 2.01, 0.1)
    ag16 = 1.93 * np.sin(2 * np.pi * t16)
    ag16[t16 > 1.0] = 0.0
    building = MDF.from_shear_building(np.full(n, 0.2591e5), np.full(n, 100e5))
    building.set_modal_damping(zeta=np.full(n, 0.05))
    elements = [ShearStoryElement(Bilinear(uy=0.0125, fy=125e3, alpha=0.05), dof_indices=[0])]
    elements += [ShearStoryElement(Bilinear(uy=0.0125, fy=125e3, alpha=0.05), dof_indices=[j - 1, j])
                 for j in range(1, n)]
    res = building.find_response_ground_motion(GroundMotion.from_arrays(ag16, 0.1), np.ones(n),
                                               method="newmark_beta", elements=elements)
    record("Ex. 16.4: bilinear 5-story building, half-cycle sine pulse", "textbook (printed)",
           "u_5 at t = 2.0 s (cm)", float(res["u5"].iloc[20]) * 100, 4.0387, note="book value printed to 0.0001 cm")

    # 8. Textbook: Example 13.8.2, SRSS base shear from the printed spectral displacements
    rsa = ResponseSpectrumAnalysis(linear, ji=0.05, Sd=[0.13649, 0.06557, 0.03823, 0.02227, 0.01657])
    vb = float(rsa.combine_modal_responses(rsa.modal_base_shear(), method="SRSS")[0])
    record("Ex. 13.8.2: response spectrum analysis", "textbook (printed)", "SRSS base shear (kN)", vb / 1e3, 291.2,
           note="book value printed to 0.1 kN")

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False)
    np.savez_compressed(HERE / "histories.npz", **histories)
    write_markdown(df)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 60):
        print(df.drop(columns=["note"]))


def write_markdown(df: pd.DataFrame) -> None:
    lines = ["| Case | Reference | Quantity | structdyn | Reference | Relative difference | History difference |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in df.itertuples():
        hist = getattr(r, "max_hist_diff", np.nan)
        hist = "–" if pd.isna(hist) else f"{hist:.1e}"
        lines.append(f"| {r.case} | {r.reference} | {r.quantity} | {r.structdyn:.6g} | {r.reference_value:.6g} | "
                     f"{r.rel_diff:.1e} | {hist} |")
    lines.append("")
    lines.append("History difference: largest difference over the response history, relative to the peak of the "
                 "reference.")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
