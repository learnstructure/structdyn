import numpy as np
from structdyn.mdf.mdf import MDF
from structdyn.loads import LoadHistory
import matplotlib.pyplot as plt

# Define MDF system
masses = [2 * 45594, 45594]
stiffness = [2 * 18 * 10**5, 18 * 10**5]
mdf = MDF.from_shear_building(masses, stiffness)

# Define external load on the first DOF only (DOF 1)
dt = 0.01
time = np.arange(0, 1.01, dt)
p = 50 * np.sin(np.pi * time / 0.6) * 1000
p[time >= 0.6] = 0
load = LoadHistory(time, p, dof=[0])

res = mdf.find_response(load, method="newmark_beta")
print(res)
print(res['a2'].iloc[-1])  # result is -0.05480023537616319
