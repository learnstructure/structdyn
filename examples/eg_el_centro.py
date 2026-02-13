from structdyn import SDF
import numpy as np
import pandas as pd
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.utils.helpers import elcentro_chopra

# gm = GroundMotion.from_event("imperialValley_elCentro_1940", component="hor1")

elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

m = 1
ji = 0.02
t_n = 0.5
w_n = 2 * np.pi / t_n
k = w_n**2 * m

sdf = SDF(m, k, ji)

results = sdf.find_response_ground_motion(gm, method="interpolation")
print(results["displacement"].abs().max())
