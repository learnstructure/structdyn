# Structural Dynamics Library (`structdyn`)

A Python library for structural dynamics analysis.

[![Documentation Status](https://readthedocs.org/projects/structdyn/badge/?version=latest)](https://structdyn.readthedocs.io/en/latest/)

## Features

*   **Single-Degree-of-Freedom (SDF) Systems:**
    *   Define linear and nonlinear SDF systems.
    *   Analyze free and forced vibrations.
    *   Calculate responses using analytical and numerical methods.
*   **Multi-Degree-of-Freedom (MDF) Systems:**
    *   Define MDF systems with custom mass and stiffness matrices.
    *   Perform modal analysis to determine natural frequencies and mode shapes.
    *   Analyze the dynamic response using modal superposition.
*   **Ground Motion Analysis:**
    *   Load and scale ground motion records.
    *   Generate response spectra.
    *   Analyze the response of structures to earthquake excitations.
*   **Numerical Methods:**
    *   A suite of numerical solvers, including:
        *   Central Difference Method
        *   Newmark-Beta Method
        *   Linear Interpolation Method
*   **Material Models:**
    *   Elastic-perfectly plastic material model for nonlinear analysis.


## Installation

Install the package using pip:

```bash
pip install git+https://github.com/learnstructure/structdyn.git
```

## Usage

Here's a quick example of how to use `structdyn` to solve a simple SDF system:

```python
import numpy as np
from structdyn.sdf.sdf import SDF

# 1. Define the structure
m = 45594  # mass (kg)
k = 18e5  # stiffness (N/m)
ji = 0.05  # damping ratio
sdf = SDF(m, k, ji)

# 2. Define the dynamic loading
dt = 0.1
time = np.arange(0, 10.01, dt)
load = np.zeros_like(time)
load[time <= 2] = 1000 * np.sin(np.pi * time[time <= 2])

# 3. Solve the equation of motion
# Available methods: 'newmark_beta', 'central_difference', 'interpolation'
results = sdf.find_response(time, load, method="newmark_beta")

# 4. Print the results
print(results)
```

## Examples

To run the examples provided in the `examples` directory, clone the repository and run the desired example file:

```bash
git clone https://github.com/learnstructure/structdyn.git
cd structdyn
pip install -e .
python -m examples.sdf.eg_newmark
```

### Analytical Methods

`structdyn` can also solve for the response of an SDF system analytically for certain cases.

#### Free Vibration

```python
from structdyn import SDF
from structdyn.sdf.analytical_methods.analytical_response import AnalyticalResponse

sdf = SDF(m=1.0, k=100.0, ji=0.05)

analytical = AnalyticalResponse(sdf)

# Free vibration response
df_free = analytical.free_vibration(u0=0.01, v0=0.0)
print(df_free)
```

#### Harmonic Forcing

```python
from structdyn import SDF
from structdyn.sdf.analytical_methods.analytical_response import AnalyticalResponse

sdf = SDF(m=1.0, k=100.0, ji=0.05)

analytical = AnalyticalResponse(sdf)

# Harmonic sine forcing response
df_harm = analytical.harmonic_response(p0=10.0, w=5.0, excitation="sine")
print(df_harm)
```

### Ground Motion Analysis

`structdyn` can be used to analyze the response of a structure to ground motion. The library includes several ground motion records, which can be loaded as follows:

```python
from structdyn.sdf.sdf import SDF
from structdyn.ground_motions.ground_motion import GroundMotion

# Load the El Centro ground motion record
gm = GroundMotion.from_event("el_centro_1940", component="RSN6_IMPVALL.I_I-ELC180")

# Define an SDF system
sdf = SDF(45594, 18 * 10**5, 0.05)

# Solve for the response of the SDF system to the ground motion
results = sdf.find_response_ground_motion(gm, method='central_difference')

# Print the results
print(results)
```

### Nonlinear Analysis

`structdyn` supports nonlinear analysis using the `fd` parameter of the `SDF` class. Currently, the only available nonlinear model is the elastic-perfectly plastic model.

```python
import numpy as np
from structdyn.sdf.sdf import SDF

# Define an elastic-perfectly plastic SDF system
sdf_epp = SDF(m=45594, k=18e5, ji=0.05, fd="elastoplastic", f_y=200000)

# Define the dynamic loading
dt = 0.1
time = np.arange(0, 10.01, dt)
load = np.zeros_like(time)
load[time <= 2] = 1000 * np.sin(np.pi * time[time <= 2])


# Solve for the response of the nonlinear system
results_epp = sdf_epp.find_response(time, load)

# Print the results
print(results_epp)
```

### Response Spectrum Analysis

`structdyn` can be used to compute the response spectrum of a ground motion.

```python
import numpy as np
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.sdf.response_spectrum import ResponseSpectrum

# Load the El Centro ground motion record
gm = GroundMotion.from_event("el_centro_1940", component="RSN6_IMPVALL.I_I-ELC180")

# Define the periods for which to compute the response spectrum
periods = np.arange(0, 5.01, 0.1)

# Create a ResponseSpectrum object
rs = ResponseSpectrum(periods, 0.02, gm)

# Compute the response spectrum
results = rs.compute()

# Print the results
print(results)
```

## Running Tests

To run the tests, you will need to install `pytest`. You can then run the tests from the root directory of the project:

```bash
pip install pytest
pytest
```

## Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and add tests.
4.  Run the tests to ensure that everything is working correctly.
5.  Submit a pull request.