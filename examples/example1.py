from structdyn import SDF
from structdyn import Interpolation, CentralDifference
from structdyn import fs_elastoplastic, fs_hysteresis
import numpy as np
import matplotlib.pyplot as plt
from structdyn import SDF
from structdyn import Interpolation, CentralDifference
from structdyn import fs_elastoplastic, fs_hysteresis
import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0
# Create SDF object
sdf = SDF(45594, 18 * 10**5, 0.05)

solver_interpolation = Interpolation(sdf, dt)
solver_central_difference = CentralDifference(sdf, dt)
solver_non_linear = CentralDifference(sdf, dt, non_linear=True)

# Compute solutions
displacement_1, velocity_1 = solver_interpolation.compute_solution(
    time_steps, load_values
)
displacement_2, velocity_2 = solver_central_difference.compute_solution(
    time_steps, load_values
)
displacement_3, velocity_3 = solver_non_linear.compute_solution(
    time_steps, load_values, fs_elastoplastic()
)
displacement_4, velocity_4 = solver_non_linear.compute_solution(
    time_steps, load_values, fs_hysteresis()
)

# Print results
print("Displacement using Interpolation:\n", displacement_1)
print("Displacement using Central Difference:\n", displacement_2)
print("Displacement using Central Difference (elastoplastic):\n", displacement_3)
print("Displacement using Central Difference (hysteresis):\n", displacement_4)

# Plot results
plt.plot(
    time_steps, displacement_1 * 100, label="Interpolation method, Linear", marker="."
)
plt.plot(
    time_steps, displacement_2 * 100, label="Central Difference, Linear", marker="."
)
plt.plot(
    time_steps,
    displacement_3 * 100,
    label="Central Difference, Non-linear Elastoplastic",
    marker=".",
)
plt.plot(
    time_steps,
    displacement_4 * 100,
    label="Central Difference, Non-linear Hysteresis",
    marker=".",
)
plt.xlabel("Time (s)")
plt.ylabel("Displacement (cm)")
plt.legend()
plt.show()
