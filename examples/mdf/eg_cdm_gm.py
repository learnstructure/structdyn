import numpy as np
from structdyn.mdf.mdf import MDF
import matplotlib.pyplot as plt
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.utils.helpers import elcentro_chopra

# Define MDF system
masses = np.ones(5) * 0.2591 * 1e5
stiffness = np.ones(5) * 100 * 1e5
mdf = MDF.from_shear_building(masses, stiffness)
mdf.set_modal_damping(zeta=np.ones(5) * 0.05)

omega, phi = mdf.modal.modal_analysis()
# print(omega)
# print(phi)

# Define ground motion
time = np.arange(0, 2.01, 0.01)
acc = (1.93 / 9.81) * np.sin(2 * np.pi * time)
# plt.plot(time, acc)
# plt.show()
gm = GroundMotion.from_arrays(acc, 0.01)

inf_vec = np.ones(5)
res = mdf.find_response_ground_motion(gm, inf_vec)
print(res)
res.plot(x="time", y="u5")
plt.show()
