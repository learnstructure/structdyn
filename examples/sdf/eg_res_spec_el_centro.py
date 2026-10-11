# Section 6.4 (& Figure 6.6.2); Chopra A. K., Dynamics of structure, 5th edn
import numpy as np
from structdyn import GroundMotion, ResponseSpectrum, elcentro_chopra

# Define el centro ground motion from Chopra's book- Appendix 6
# elc = elcentro_chopra()
# gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)   #method 1
# gm = GroundMotion.from_elcentro_chopra()          #method 2
gm = GroundMotion.from_event("elcentro_chopra")    #method 3

# Define the period range of interest
periods = np.arange(0, 5.01, 0.1)

# Create response spectrum object
rs = ResponseSpectrum(periods, 0.02, gm)

# Analysis
spectra = rs.compute()
print(spectra["Sd"][20])  # result is 0.1896749378231744
