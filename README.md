# structdyn: Structural Dynamics Solver

`structdyn` is a Python library for solving single-degree-of-freedom (SDF) dynamic problems using numerical methods.

## Installation
### Clone the repository:
git clone https://github.com/learnstructure/structdyn.git 

cd structdyn

python -m examples.example1 #to see an example quickly

### or install without cloning the repository:
pip install git+https://github.com/learnstructure/structdyn.git

## Custom Usage
```python
import numpy as np
from structdyn import SDF, Interpolation, CentralDifference, fs_elastoplastic, fs_hysteresis

#Define load
dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# Define system properties
sdf = SDF(m=45594, k=18e5, ji=0.05)

# Solve using Central Difference Method
solver = Interpolation(sdf, dt)
displacement, velocity= solver.compute_solution(time_steps, load_values)
print("Displacement using Interpolation:\n", displacement)


