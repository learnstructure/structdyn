# Example 13.8.2; Chopra A. K., Dynamics of structure, 5th edn
# Enhanced to showcase:
# 1. RSA from ground motion (El Centro)
# 2. RSA from specified spectral displacements (Chopra values)
# 3. RSA from continuous response spectrum curve (Sa vs T)
# 4. RSA from ASCE 7 parameters (SDS, SD1, TL)
# 5. Modal combinations: SRSS vs CQC
import numpy as np
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.utils.helpers import elcentro_chopra
from structdyn.mdf.mdf import MDF
from structdyn.mdf.analytical_methods.response_spectrum_analysis import (
    ResponseSpectrumAnalysis,
)

# -----------------------------------------------------------------------------
# 1. Define MDF system: 5-story shear building (Chopra Example 13.8.2)
# -----------------------------------------------------------------------------
masses = np.ones(5) * 45000  # mass of each floor in kg
stiffness = np.ones(5) * 54.82e5  # stiffness of each floor in N/m
mdf = MDF.from_shear_building(masses, stiffness)

print("=" * 60)
print("RESPONSE SPECTRUM ANALYSIS (Chopra Example 13.8.2)")
print("=" * 60)

# -----------------------------------------------------------------------------
# Method A: RSA from ground motion
# -----------------------------------------------------------------------------
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)
rsa_gm = ResponseSpectrumAnalysis(mdf, ji=0.05, gm=gm)

vb_gm = rsa_gm.modal_base_shear()
vb_srss_gm = rsa_gm.combine_modal_responses(vb_gm, method="SRSS")
vb_cqc_gm = rsa_gm.combine_modal_responses(vb_gm, method="CQC")

print("\n--- Method A: From Ground Motion (El Centro) ---")
print("Modal Periods (s):", np.round(rsa_gm.T, 4))
print("Modal Participation Factors (Gamma):", np.round(rsa_gm.gamma, 4))
print("Mass Participation Ratio (%):", np.round(rsa_gm.mass_participation_ratio * 100, 2))
print("Combined Base Shear (SRSS):", float(vb_srss_gm[0]), "N")
print("Combined Base Shear (CQC): ", float(vb_cqc_gm[0]), "N")

# -----------------------------------------------------------------------------
# Method B: RSA from discrete Chopra spectral displacements
# -----------------------------------------------------------------------------
Sd_chopra = [0.13649, 0.06557, 0.03823, 0.02227, 0.01657]
rsa_sd = ResponseSpectrumAnalysis(mdf, ji=0.05, Sd=Sd_chopra)
vb_sd = rsa_sd.modal_base_shear()
vb_srss_sd = rsa_sd.combine_modal_responses(vb_sd, method="SRSS")
vb_cqc_sd = rsa_sd.combine_modal_responses(vb_sd, method="CQC")

print("\n--- Method B: From Discrete Chopra Sd Values ---")
print("Modal Displacements u (m):\n", np.round(rsa_sd.modal_displacements(), 5))
print("Modal Equivalent Forces (N):\n", np.round(rsa_sd.modal_equivalent_forces(), 1))
print("Modal Base Shear (N):", np.round(vb_sd[0], 1))
print("Combined Base Shear (SRSS):", float(vb_srss_sd[0]), "N (Expected: 291222 N)")
print("Combined Base Shear (CQC): ", float(vb_cqc_sd[0]), "N")

# -----------------------------------------------------------------------------
# Method C: RSA from continuous response spectrum curve (Sa vs T)
# -----------------------------------------------------------------------------
T_curve = np.linspace(0.0, 4.0, 200)
# Example design curve: peak 0.6g plateau between 0.1s and 0.5s, then 0.3/T
Sa_curve = np.piecewise(
    T_curve,
    [T_curve < 0.1, (T_curve >= 0.1) & (T_curve <= 0.5), T_curve > 0.5],
    [lambda t: 0.2 + 4.0 * t, 0.6, lambda t: 0.3 / t]
)
rsa_curve = ResponseSpectrumAnalysis(
    mdf, ji=0.05, spectrum_curve=(T_curve, Sa_curve), scale_factor=9.80665
)
vb_curve = rsa_curve.modal_base_shear()
vb_cqc_curve = rsa_curve.combine_modal_responses(vb_curve, method="CQC")

print("\n--- Method C: From User Continuous Spectrum Curve ---")
print("Interpolated Sa at modal periods (m/s²):", np.round(rsa_curve.Sa, 3))
print("Combined Base Shear (CQC):", float(vb_cqc_curve[0]), "N")

# -----------------------------------------------------------------------------
# Method D: RSA from ASCE 7 parameters (SDS=1.0, SD1=0.6, TL=8.0)
# -----------------------------------------------------------------------------
rsa_asce = ResponseSpectrumAnalysis(
    mdf, ji=0.05, sds=1.0, sd1=0.6, tl=8.0, scale_factor=9.80665
)
vb_asce = rsa_asce.modal_base_shear()
vb_cqc_asce = rsa_asce.combine_modal_responses(vb_asce, method="CQC")

print("\n--- Method D: From ASCE 7 Parameters (SDS=1.0g, SD1=0.6g) ---")
print("ASCE 7 Sa at modal periods (m/s²):", np.round(rsa_asce.Sa, 3))
print("Combined Base Shear (CQC):", float(vb_cqc_asce[0]), "N")
print("=" * 60)
