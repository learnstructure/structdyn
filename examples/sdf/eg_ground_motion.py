import numpy as np
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.utils.helpers import elcentro_chopra
from pathlib import Path

# Method 1: directly from Chopra's book- Appendix 6
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)
print(gm.to_dataframe())

# Method 2: from numpy arrays
dt = 0.01
time = np.arange(0, 2.01, dt)
acc = (1.93 / 9.81) * np.sin(2 * np.pi * time)
gm = GroundMotion.from_arrays(acc, dt)
print(gm.to_dataframe())

# Method 3: from event (in-built PEER database)
gm = GroundMotion.from_event("northridge_sylmar_1994", component="hor1")
print(gm.to_dataframe())

# Method 4: from .at2 file (PEER database format)
# file_path = "path to your file"
# gm = GroundMotion.from_at2(file_path)
