from structdyn import SDF
from structdyn import Interpolation, CentralDifference, NewmarkBeta
from structdyn.utils.helpers import ElasticPerfectlyPlastic
import numpy as np
import matplotlib.pyplot as plt

fd = ElasticPerfectlyPlastic(uy=0.02, fy=36000)

u = np.array(
    [
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.03,
        0.02,
        0.01,
        0.0,
        -0.01,
        -0.02,
        -0.03,
        -0.04,
        -0.03,
        -0.02,
        -0.01,
        0.0,
    ]
)
fs = np.zeros_like(u)
for i in range(1, len(u)):
    fs[i], kt, yielded = fd.trial_response(u[i])
    print(f"u: {u[i]:.3f}, fs: {fs[i]:.1f}, kt: {kt:.1f}, yielded: {yielded}")
    fd.commit_state(u[i])

plt.plot(u, fs, marker="o")
plt.show()
