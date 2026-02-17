# Section 6.4 (& Figure 6.4.1a); Chopra A. K., Dynamics of structure, 5th edn
from structdyn import SDF
import numpy as np
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.utils.helpers import elcentro_chopra

# Define el centro ground motion from Chopra's book- Appendix 6
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

# Create SDF object
m = 1
ji = 0.02
t_n = 0.5
w_n = 2 * np.pi / t_n
k = w_n**2 * m
sdf = SDF(m, k, ji)

# Analysis
results = sdf.find_response_ground_motion(gm, method="interpolation")
print(results["displacement"].abs().max())  # result is 0.0679400697200714
