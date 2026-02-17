# Example 5.4; Chopra A. K., Dynamics of structure, 5th edn
from structdyn import SDF
import numpy as np

# Define external load
dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# Create SDF object
sdf = SDF(45594, 18 * 10**5, 0.05)

# Analysis
responses = sdf.find_response(
    time_steps, load_values, method="newmark_beta", acc_type="linear"
)
print(responses)
print(responses["displacement"][10])  # result is -0.03391195470077432
