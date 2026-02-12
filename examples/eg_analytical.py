from structdyn import SDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from structdyn.sdf.analytical_methods.analytical_response import AnalyticalResponse

sdf = SDF(m=1.0, k=100.0, ji=0.05)

analytical = AnalyticalResponse(sdf)

# Free vibration
df_free = analytical.free_vibration(u0=0.01, v0=0.0)
# print(df_free)
df_free.plot(x="time", y="displacement")
plt.show()

# Harmonic sine forcing
df_harm = analytical.harmonic_response(p0=10.0, w=5.0, excitation="sine")
df_harm.plot(x="time", y="displacement")
plt.show()
