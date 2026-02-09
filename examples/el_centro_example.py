from structdyn import SDF
from structdyn import Interpolation, CentralDifference, NewmarkBeta
from structdyn import fs_elastoplastic
import numpy as np
import matplotlib.pyplot as plt

from structdyn.ground_motions.ground_motion import GroundMotion

####old method####
# from structdyn.ground_motions.gm_helper import load_event, select_component
# el_centro = load_event("el_centro_1940")
# df_gm, dt = select_component(el_centro, component="rsn6_impvall.i_i-elc-up")
# print(df_gm)


gm = GroundMotion.from_event("el_centro_1940", component="rsn6_impvall.i_i-elc-up")

# print(gm.to_dataframe())
gm = gm.to_dataframe()

sdf = SDF(45594, 18 * 10**5, 0.05)

results = sdf.find_response(
    gm["time"],
    gm["acc_g"],
    method="central_difference",
)
print(results)
