import numpy as np
from structdyn import GroundMotion, elcentro_chopra

# from structdyn.ground_motions.ground_motion import GroundMotion
# from structdyn.utils.helpers import elcentro_chopra

# Records are in g. scale_factor converts them to the units of the analysis (9.81 for m/s^2,
# 386.09 for in/s^2); without it, they stay in g.

# Method 1: directly from Chopra's book- Appendix 6
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02, scale_factor=9.81)
print(gm.to_dataframe())

# Method 2: from numpy arrays
dt = 0.01
time = np.arange(0, 2.01, dt)
acc = (1.93 / 9.81) * np.sin(2 * np.pi * time)
gm = GroundMotion.from_arrays(acc, dt, scale_factor=9.81)
print(gm.to_dataframe())

# Method 3: from event (in-built PEER database)
gm = GroundMotion.from_event("northridge_sylmar_1994", component="hor1", scale_factor=9.81)
print(gm.to_dataframe())

# Method 4: from .at2 file (PEER database format)
# file_path = "path to your file"
# gm = GroundMotion.from_at2(file_path, scale_factor=9.81)
