# Example 16.1; Chopra A. K., Dynamics of structure, 5th edn
import numpy as np
from structdyn.mdf.mdf import MDF
import matplotlib.pyplot as plt
from structdyn.ground_motions.ground_motion import GroundMotion

# Define MDF system
masses = np.ones(5) * 0.2591 * 1e5
stiffness = np.ones(5) * 100 * 1e5
mdf = MDF.from_shear_building(masses, stiffness)
mdf.set_modal_damping(zeta=np.ones(5) * 0.05)  # set damping matrix

# Define ground motion as given in the same example
time = np.arange(0, 2.01, 0.1)
acc = (1.93 / 9.81) * np.sin(2 * np.pi * time)
acc[time > 1.0] = 0
gm = GroundMotion.from_arrays(acc, 0.1)

# Influence vector
inf_vec = np.ones(5)

# Run analysis
response = mdf.find_response_ground_motion(
    gm, inf_vec, method="newmark_beta", use_modal=True, n_modes=2
)
# Note we can get more accurate results by considering more modes. Also if we remove use_modal, then direct integration of coupled equations would give more accurate results.
print(response)
print(response["u5"][20])  # result = 0.05888401249961952

# Plot response
# res.plot(x="time", y="u5")
# plt.show()
