from structdyn import SDF
from structdyn import Interpolation, CentralDifference, NewmarkBeta
from structdyn import fs_elastoplastic
import numpy as np
import matplotlib.pyplot as plt

from structdyn.ground_motions.ground_motion import GroundMotion

gm = GroundMotion.from_event("el_centro_1940", component="RSN6_IMPVALL.I_I-ELC180")
gm = gm.to_dataframe()

sdf = SDF(45594, 18 * 10**5, 0.05)

results = sdf.find_response(
    gm["time"],
    gm["acc_g"],
    method="central_difference",
)
print(results)
