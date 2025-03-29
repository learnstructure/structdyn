from structdyn import SDF
from structdyn import Interpolation, CentralDifference
from structdyn import fs_elastoplastic, fs_hysteresis
import numpy as np

dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

sdf = SDF(45594, 18 * 10**5, 0.05)
solver = Interpolation(sdf, dt)
displacement_1, velocity_1 = solver.compute_solution(time_steps, load_values)
print("Displacement using Interpolation:\n", displacement_1)

solver = CentralDifference(sdf, dt)
displacement_2, velocity_2 = solver.compute_solution(time_steps, load_values)
print("Displacement using Central Difference:\n", displacement_2)

solver_non_linear = CentralDifference(sdf, dt, non_linear=True)
displacement_3, velocity_3 = solver_non_linear.compute_solution(
    time_steps, load_values, fs_elastoplastic()
)
print("Displacement using Central Difference (elastoplastic):\n", displacement_3)

displacement_4, velocity_4 = solver_non_linear.compute_solution(
    time_steps, load_values, fs_hysteresis()
)
print("Displacement using Central Difference (hysteresis):\n", displacement_4)
