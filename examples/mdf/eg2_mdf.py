# Example 10.9; Chopra A. K., Dynamics of structure, 5th edn
import numpy as np
from structdyn.mdf.mdf import MDF
import matplotlib.pyplot as plt

# Define mdf mdf: 2-story shear building
masses = [2, 1]
stiffness = [2, 1]
mdf = MDF.from_shear_building(masses, stiffness)

# Modal analysis
omega, phi = mdf.modal.modal_analysis(
    dof_normalize=-1
)  # -1 means: mode shape normalized based on roof dof

# Initial conditions
u0 = [0.5, 1]
v0 = [0, 0]
time = np.arange(0, 20, 0.01)

# Run analysis
response = mdf.modal.free_vibration_response(u0, v0, time)
print(response["u1"].abs().max())  # result = 0.49999999999999994

response.plot(x="time", y="u1")
plt.show()
