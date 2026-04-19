# HW4 - Earthquake resistant design of structures
# units in kips, in, s

import numpy as np
from structdyn import MDF, ResponseSpectrumAnalysis

# Define mdf system: 4-story shear building
g = 386.09
E = 6000
I = 20 * 20**3 / 12
L = 10 * 12
k = 2 * 12 * E * I / L**3  # 2 columns on each floor. 12EI/L^3 due to rigid beam
masses = [450 / g] * 4
stiffness = [k] * 4
mdf = MDF.from_shear_building(masses, stiffness)

# Modal analysis
omega, phi = mdf.modal.modal_analysis()

# print("Natural Frequencies (rad/s):", omega)
print("Natural time periods:", 2 * np.pi / omega)
print("Mode shapes:\n", phi)

mdf.plot.mode_shape([1, 2])

Sa = (
    np.array([1.17, 1.17, 0.950719, 0.861516]) * 386.1
)  # spectral acceleartions taken from design spectrum
Sd = Sa / omega**2  # spectral displacements

rsa = ResponseSpectrumAnalysis(mdf, Sd=Sd)

modal_f_eq = rsa.modal_equivalent_forces()
modal_base_shear = rsa.modal_base_shear()
combined_base_shear = rsa.combine_modal_responses(modal_base_shear, method="SRSS")

print("Modal lateral forces at story level:\n", modal_f_eq)
print("Modal base shear:\n", modal_base_shear)
print("Combined base shear:\n", combined_base_shear)  # 1890.08206545

print("Combined base shear after dividing by R/Ie:", combined_base_shear / (8 * 1))
