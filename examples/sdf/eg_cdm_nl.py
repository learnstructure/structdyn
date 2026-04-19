# Added non-linearity to Example 5.2; Chopra A. K., Dynamics of structure, 5th edn
from structdyn import SDF
import numpy as np
import matplotlib.pyplot as plt

# from structdyn.utils.material_models import (
#     LinearElastic,
#     ElasticPerfectlyPlastic,
#     Bilinear,
#     BoucWen,
# )
from structdyn.utils import ElasticPerfectlyPlastic

# Define external load
dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# Create SDF object with non-linear material model
# material_model = LinearElastic(stiffness=36000 / 0.02)  # for linear analysis
material_model = ElasticPerfectlyPlastic(uy=0.02, fy=36000)
# material_model = Bilinear(uy=0.02, fy=36000)
# material_model = BoucWen(k0=36000 / 0.02, alpha=0.0, A=0.02, beta=0.5, gamma=0.5, n=1)
sdf = SDF(45594, 18 * 10**5, 0.05, fd=material_model)  # for non-linear analysis

# Analysis
responses = sdf.find_response(
    time_steps,
    load_values,
    method="central_difference",
)
print(responses)
print(responses["displacement"][10])  # result is 0.0407254127878451
