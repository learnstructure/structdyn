# structdyn: Structural Dynamics Solver

`structdyn` is a Python library for solving single-degree-of-freedom (SDF) dynamic problems using numerical methods.

## Installation
Clone the repository:
git clone https://github.com/learnstructure/structdyn.git cd structdyn


## Usage
```python
import numpy as np
from structdyn import SDF, CentralDifference, fs

# Define system properties
sdf = SDF(m=45594, k=18e5, ji=0.05)
time_steps = np.arange(0, 1.01, 0.1)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000

# Solve using Central Difference Method
solver = CentralDifference(sdf, dt=0.1, non_linear=True)
displacement = solver.compute_solution(time_steps, load_values, fs)


