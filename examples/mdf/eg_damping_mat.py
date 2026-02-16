import numpy as np
from structdyn.mdf.mdf import MDF

# Define MDF system
masses = [2e5, 2e5, 1e5]
stiffness = [1e8, 1e8, 1e8]
mdf = MDF.from_shear_building(masses, stiffness)

omega, phi = mdf.modal.modal_analysis()
print(omega)
print(phi)
print("-" * 50)
mdf.set_modal_damping(zeta=[0.05, 0.05, 0.05])
print(mdf.C)
