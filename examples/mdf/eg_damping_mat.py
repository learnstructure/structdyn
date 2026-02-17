# Example 11.4; Chopra A. K., Dynamics of structure, 5th edn
from structdyn.mdf.mdf import MDF

# Define MDF system
masses = [2e5, 2e5, 1e5]
stiffness = [1e8, 1e8, 1e8]
mdf = MDF.from_shear_building(masses, stiffness)

# Create damping matrix based on modal damping ratios; zeta
mdf.set_modal_damping(zeta=[0.05, 0.05, 0.05])

# Damping matrix
print(mdf.C)
print(mdf.C.flatten()[-1])  # result = 287983.44117400126
