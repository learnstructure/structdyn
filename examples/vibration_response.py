from structdyn import SDF
from structdyn import free_vibration, harmonic_vibration
import numpy as np
from structdyn import plot_displacement

sdf = SDF(m=45594, k=18 * 10**5, ji=0.1)

time_steps = np.arange(0, 5, 0.05)
u, type = free_vibration(sdf, time_steps)

plot_displacement(time_steps, u, type)

u, type = harmonic_vibration(sdf, time_steps, type="cosine")
plot_displacement(time_steps, u, type)
