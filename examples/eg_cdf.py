from structdyn import SDF
from structdyn import Interpolation, CentralDifference, NewmarkBeta
from structdyn import fs_elastoplastic
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# print(time_steps)

# Create SDF object
# sdf = SDF(45594, 18 * 10**5, 0.05)

sdf = SDF(45594, 18 * 10**5, 0.05, fd="elastoplastic", uy=0.02, fy=36000)

results = sdf.find_response(
    time_steps,
    load_values,
    method="central_difference",
)
print(results)
