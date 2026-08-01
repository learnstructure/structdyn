# Example 5.2; Chopra A. K., Dynamics of structure, 5th edn
from structdyn import SDF
from structdyn.loads import LoadHistory
import numpy as np
import matplotlib.pyplot as plt

# Define external load
dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# Build the load history
load = LoadHistory(time_steps, load_values)

# Create SDF object
sdf = SDF(45594, 18 * 10**5, 0.05)

# Analysis
responses = sdf.find_response(
    load,
    method="central_difference",
)
print(responses)
print(responses["displacement"][10])  # result is -0.03812718853606678
