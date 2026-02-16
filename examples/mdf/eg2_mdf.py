import numpy as np
from structdyn.mdf.mdf import MDF
import matplotlib.pyplot as plt

##Example 10.1
# M = np.array([[2 / 6, 1 / 6], [1 / 6, 2 / 6]])
# K = np.array([[1, 0], [0, 2]])
# system = MDF(M, K)

# Example 10.4: 2-story shear building
masses = [2, 1]
stiffness = [2, 1]

system = MDF.from_shear_building(masses, stiffness)
omega, phi = system.modal.modal_analysis()
phi = system.modal.normalize_modes()

u0 = [0.5, 1]
v0 = [0, 0]
time = np.arange(0, 20, 0.01)

res = system.modal.free_vibration_response(u0, v0, time)
print(res)
print(res.shape)
# print("Natural Frequencies (rad/s):", omega)
# print("Mode shapes:", phi)
res.plot(x="time", y="u2")
plt.show()
