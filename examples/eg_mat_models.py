import numpy as np
import matplotlib.pyplot as plt
from structdyn.utils.material_models import (
    LinearElastic,
    ElasticPerfectlyPlastic,
    Bilinear,
    BoucWen,
    RambergOsgood,
    Takeda,
)

# Models to test (uncomment one at a time)
# model = LinearElastic(stiffness=36000 / 0.02)
# model = ElasticPerfectlyPlastic(uy=0.02, fy=36000)
# model = Bilinear(uy=0.02, fy=36000, alpha=0.001)
# model = BoucWen(k0=36000 / 0.02, alpha=0.0, A=0.02, beta=0.5, gamma=0.5, n=1)
# model = RambergOsgood(36000 / 0.02, 36000, n=10, alpha=1)
model = Takeda(K0=36000 / 0.02, Fy=36000, alpha=0.01, beta=0.3)

# Time and displacement history (increasing amplitude to see multiple loops)
t = np.linspace(0, 40, 5000)
dt = t[1] - t[0]

# Amplitude sweep from 0.2 to 2.0
amp = 0.2 + 1.8 * np.minimum(t / 30, 1.0)  # ramp up to 30 s, then constant
u = amp * np.sin(2 * np.pi * 0.5 * t)

fs = np.zeros_like(u)
for i, ui in enumerate(u):
    fs[i], _, _ = model.get_state(ui, dt)

plt.plot(u, fs)
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.title("Hysteresis with amplitude sweep")
plt.grid(True)
plt.show()
