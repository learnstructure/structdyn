# Structural Dynamics Library (`structdyn`)
An open-source Python library for structural dynamics analysis

**structdyn** is a Python package for performing structural dynamics and earthquake engineering analysis of Single-Degree-of-Freedom (SDF) and Multi-Degree-of-Freedom (MDF) systems. It provides a suite of tools for analyzing both linear and non-linear behavior, making it a versatile library for students, faculties and researchers.

See the documentation here: [![Documentation Status](https://readthedocs.org/projects/structdyn/badge/?version=latest)](https://structdyn.readthedocs.io/en/latest/)


## Features

**Single-Degree-of-Freedom (SDF) Systems:**

*   **Analytical Solutions:** Compute the response of SDF systems to harmonic loads.
*   **Numerical Integration:** Solve the equation of motion using robust numerical methods, including Newmark-Beta and Central Difference.
*   **Response Spectra:** Generate earthquake response spectra for displacement, velocity, and acceleration.

**Multi-Degree-of-Freedom (MDF) Systems:**

*   **Shear Building Models:** Quickly create MDF systems for typical shear buildings.
*   **Modal Analysis:** Compute natural frequencies, mode shapes, and modal participation factors.
*   **Response History Analysis:** Perform linear and non-linear time history analysis using direct integration.
*   **Response Spectrum Analysis (RSA):** Estimate peak responses using modal combination methods (SRSS).

**Visualization:**

*   **Animated Responses:** Animate the dynamic response of both SDF and MDF systems to visualize structural behavior over time.
*   **Mode Shapes:** Plot the mode shapes of MDF systems to understand their vibrational characteristics.

**Non-Linear Analysis:**

*   **Material Models:** A library of hysteretic material models, including elastic perfectly plastic, bilinear, Bouc-Wen & Ramberg-Osgood model.
*   **Element Formulations:** Define non-linear behavior at the element level for MDF systems.
*   **Robust Solvers:** Utilizes iterative Newton-Raphson schemes within the Newmark-Beta solver for accurate non-linear solutions.

**Ground Motion Tools:**

*   **Record Processing:** Easily load, process, and manipulate earthquake ground motion records.
*   **Standard Formats:** Includes helper functions for common ground motion data, such as the El Centro record from Chopra's book.


## Installation

The recommended way to install `structdyn` is from the Python Package Index (PyPI) using `pip`:

```bash
pip install structdyn
```

This will install the latest stable, officially released version.

**Developer Installation**

If you want to install the very latest development version directly from the source, you can install it from the GitHub repository:

```bash
pip install git+https://github.com/learnstructure/structdyn.git
```

## Quick Example: Response Spectrum Analysis (MDF system)

Here's an example of how to perform a Response Spectrum Analysis for a 5-story shear building, based on **Example 13.8.2 from Chopra, "Dynamics of Structures", 5th Ed**.

```python
import numpy as np
from structdyn.mdf.mdf import MDF
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.mdf.analytical_methods.response_spectrum_analysis import ResponseSpectrumAnalysis
from structdyn.utils.helpers import elcentro_chopra

# 1. Define the ground motion (El Centro, from Chopra's book)
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

# 2. Define the 5-story shear building
masses = np.ones(5) * 45000      # kg
stiffness = np.ones(5) * 54.82e5 # N/m
mdf = MDF.from_shear_building(masses, stiffness)

# 3. Perform Response Spectrum Analysis
rsa = ResponseSpectrumAnalysis(mdf, ji=0.05, gm=gm)

# 4. Calculate modal base shear and combine using SRSS
modal_base_shear = rsa.modal_base_shear()
combined_base_shear = rsa.combine_modal_responses(modal_base_shear, method="SRSS")

# 5. Print the result
print(f"Combined base shear (SRSS): {combined_base_shear[0]:.2f} N")
# Expected result from Chopra, Example 13.8.2: 291221.90 N
```

<details>
<summary>Click for more examples (SDF Systems, Ground Motion, etc.)</summary>

### Harmonic Forcing

```python
from structdyn import SDF
from structdyn.sdf.analytical_methods.analytical_response import AnalyticalResponse

sdf = SDF(m=1.0, k=100.0, ji=0.05)

analytical = AnalyticalResponse(sdf)

# Harmonic sine forcing response
df_harm = analytical.harmonic_response(p0=10.0, w=5.0, excitation="sine")
print(df_harm)
```

### SDF Ground Motion Response

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

### Response Spectrum Generation (SDF)

`structdyn` can be used to compute the response spectrum of a ground motion.

```python
# Section 6.4 (& Figure 6.6.2); Chopra A. K., Dynamics of structure, 5th edn
import numpy as np
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.sdf.response_spectrum import ResponseSpectrum
from structdyn.utils.helpers import elcentro_chopra

# Define el centro ground motion from Chopra's book- Appendix 6
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

# Define the period range of interest
periods = np.arange(0, 5.01, 0.1)

# Create response spectrum object
rs = ResponseSpectrum(periods, 0.02, gm)

# Analysis
spectra = rs.compute()
print(spectra["Sd"][20])  # result is 0.1896749378231744
```

### Non-linear analysis SDF system
```python
# Example 5.5; Chopra A. K., Dynamics of structure, 5th edn
from structdyn import SDF
import numpy as np
from structdyn.utils.material_models import ElasticPerfectlyPlastic

# Define external load
dt = 0.1
time_steps = np.arange(0, 1.01, dt)
load_values = 50 * np.sin(np.pi * time_steps / 0.6) * 1000
load_values[time_steps >= 0.6] = 0

# Create SDF object
material_model = ElasticPerfectlyPlastic(uy=0.02, fy=36000)
sdf = SDF(45594, 18 * 10**5, 0.05, fd=material_model)

# Analysis
responses = sdf.find_response(
    time_steps, load_values, method="newmark_beta", acc_type="average"
)

print(responses)
print(responses["displacement"][10])  # result is 0.03606328101158249

```

</details>


## Validated Examples

For more detailed examples, including animations and advanced use cases, please see the `examples` directory in the repository.

[**Browse the examples folder**](./examples)

## Citing `structdyn`

If you use `structdyn` in your research or work, please cite it as follows:

> Mandal, A. (2026). StructDyn: An open-source Python library for structural dynamics analysis (Version 0.7.4) [Computer software]. https://doi.org/10.5281/zenodo.18676816

Here is the citation in BibTeX format: 

```bibtex
@software{Mandal_structdyn_2026,
  author = {Mandal, Abinash},
  title = {{StructDyn: An open-source Python library for structural dynamics analysis}},
  version = {0.7.4},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18676816},
  url = {https://doi.org/10.5281/zenodo.18676816}
}
```

## Running Tests

To run the tests, you will need to install `pytest`. You can then run the tests from the root directory of the project:

```bash
pip install pytest
pytest
```

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or bug fix, and submit a pull request.