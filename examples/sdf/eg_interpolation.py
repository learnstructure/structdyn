# Example 5.1; Chopra A. K., Dynamics of structure, 5th edn
from structdyn import SDF
import numpy as np

# Define external load
dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# Create SDF object
sdf = SDF(45594, 18 * 10**5, 0.05)

# Run analysis
responses = sdf.find_response(
    time_steps,
    load_values,
    method="interpolation",
)
print(responses)
print(responses["displacement"][10])  # result is -0.034534260954800985
