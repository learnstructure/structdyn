from structdyn import SDF
from structdyn import Interpolation, CentralDifference
from structdyn import fs_elastoplastic
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0
# Create SDF object
sdf = SDF(45594, 18 * 10**5, 0.05)
displacement_1, velocity_1, fs1 = sdf.find_response(
    time_steps, load_values, method="interpolation"
)

# Print results
print("Displacement using Interpolation:\n", displacement_1)


# Plot results
plt.plot(
    time_steps, displacement_1 * 100, label="Interpolation method, Linear", marker="."
)

plt.xlabel("Time (s)")
plt.ylabel("Displacement (cm)")
plt.legend()
plt.show()

plt.plot(displacement_1, fs1, label="Force", marker=".")
plt.xlabel("Displacement")
plt.ylabel("Force (N)")
plt.legend()
plt.show()
