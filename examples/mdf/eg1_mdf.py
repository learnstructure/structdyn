import numpy as np
from structdyn.mdf.mdf import MDF

##Example 10.1
# M = np.array([[2 / 6, 1 / 6], [1 / 6, 2 / 6]])
# K = np.array([[1, 0], [0, 2]])
# system = MDF(M, K)

# Example 10.4: 2-story shear building
masses = [2, 1]
stiffness = [2, 1]

system = MDF.from_shear_building(masses, stiffness)
# print(system.M, system.K)

omega, phi = system.modal.modal_analysis()
phi = system.modal.normalize_modes()
u = np.array([1, 1])
qn = system.modal.modal_coordinates(u)
print("Natural Frequencies (rad/s):", omega)
print("Mode shapes:", phi)
print("Modal coordinates:", qn)
# print(system.M)
# print(system.modal.Mn)


# alpha, beta = system.set_rayleigh_damping(0.05, 0.05)
# gamma = system.participation_factors()
# print("Participation Factors:", gamma)
