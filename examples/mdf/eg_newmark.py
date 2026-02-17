import numpy as np
from structdyn.mdf.mdf import MDF
import matplotlib.pyplot as plt

# Define MDF system
masses = [2 * 45594, 45594]
stiffness = [2 * 18 * 10**5, 18 * 10**5]
mdf = MDF.from_shear_building(masses, stiffness)

# Define external load
dt = 0.01
time = np.arange(0, 1.01, dt)
load = 50 * np.sin(np.pi * time / 0.6) * 1000
load[time >= 0.6] = 0
load = np.column_stack((load, np.zeros_like(load)))

res = mdf.find_response(time, load, method="newmark_beta")
print(res)
