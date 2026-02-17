# Example 10.4; Chopra A. K., Dynamics of structure, 5th edn
import numpy as np
from structdyn.mdf.mdf import MDF

# Define mdf system: 2-story shear building
masses = [2, 1]
stiffness = [2, 1]
mdf = MDF.from_shear_building(masses, stiffness)
# print(mdf.M, mdf.K)

# Modal analysis
omega, phi = mdf.modal.modal_analysis(
    dof_normalize=-1
)  # -1 means: mode shape normalized based on roof dof

print("Natural Frequencies (rad/s):", omega)  # omegas = [0.70710678 1.41421356]
print("Mode shapes:", phi)

# u = np.array([1, 1])
# qn = mdf.modal.modal_coordinates(u)
# print("Modal coordinates:", qn)
