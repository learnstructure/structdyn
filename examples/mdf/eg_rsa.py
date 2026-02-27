# Example 13.8.2; Chopra A. K., Dynamics of structure, 5th edn
import numpy as np
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.utils.helpers import elcentro_chopra
from structdyn.mdf.mdf import MDF
from structdyn.mdf.analytical_methods.response_spectrum_analysis import (
    ResponseSpectrumAnalysis,
)

# Define el centro ground motion from Chopra's book- Appendix 6
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

# Define mdf system: 5-story shear building
masses = np.ones(5) * 45000  # mass of each floor in kg
stiffness = np.ones(5) * 54.82e5  # stiffness of each floor in N/m
mdf = MDF.from_shear_building(masses, stiffness)

# Perform response spectrum analysis
# rsa = ResponseSpectrumAnalysis(mdf, ji=0.05, gm=gm)   #Use this method to compute spectral displacements from the ground motion
rsa = ResponseSpectrumAnalysis(
    mdf, ji=0.05, Sd=[0.13649, 0.06557, 0.03823, 0.02227, 0.01657]
)  # Using specified spectral displacements
# print("Natural Frequencies (rad/s):", rsa.omega)
# print("Mode shapes:", rsa.phi)
# print("Modal Participation Factors:", rsa.lambda_)
# print("Spectral Displacements (Sd):", rsa.Sd)

modal_u = rsa.modal_diplacements()
modal_f_eq = rsa.modal_equivalent_forces()
modal_base_shear = rsa.modal_base_shear()
combined_base_shear = rsa.combine_modal_responses(modal_base_shear, method="SRSS")

# Print results
print("Modal Displacements (u):\n", modal_u)
print("Modal Equivalent Forces:\n", modal_f_eq)
print("Modal Base Shear:\n", modal_base_shear)

print("Combined base shear (SRSS):\n", combined_base_shear)  # Result = 291221.90123453
