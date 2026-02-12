from structdyn import SDF
from structdyn import Interpolation, CentralDifference, NewmarkBeta
from structdyn import fs_elastoplastic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from structdyn.ground_motions.ground_motion import GroundMotion

# gm = GroundMotion.from_event("imperialValley_elCentro_1940", component="hor1")
p = Path(__file__).resolve().parent.parent
elc_path = p / "structdyn" / "ground_motions" / "data" / "elcentro_chopra.csv"
elc = pd.read_csv(elc_path, header=None)

gm = GroundMotion.from_arrays(elc[1], 0.02)

m = 1
ji = 0.02
t_n = 0.5
w_n = 2 * np.pi / t_n
k = w_n**2 * m

sdf = SDF(m, k, ji)

results = sdf.find_response_ground_motion(gm, method="interpolation")
print(results["displacement"].abs().max())
