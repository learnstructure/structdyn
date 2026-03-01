StructDyn documentation
=======================

StructDyn: An open-source Python library for structural dynamics analysis.

**structdyn** is a Python package for performing structural dynamics analysis of Single-Degree-of-Freedom (SDF) and Multi-Degree-of-Freedom (MDF) systems. It provides tools for analyzing both linear and non-linear behavior, making it a versatile library for researchers, students, and engineers.

Features
--------

**Single-Degree-of-Freedom (SDF) Systems:**

*   **Analytical Solutions:** Compute the response of SDF systems to harmonic loads.
*   **Numerical Integration:** Solve the equation of motion using a variety of robust numerical methods, including Linear Interpolation,Newmark-Beta and Central Difference.
*   **Response Spectra:** Generate earthquake response spectra for displacement, velocity, and acceleration.

**Multi-Degree-of-Freedom (MDF) Systems:**

*   **Shear Building Models:** Quickly create MDF systems for typical shear buildings.
*   **Modal Analysis:** Compute natural frequencies, mode shapes, and modal participation factors.
*   **Response History Analysis:** Perform linear and non-linear time history analysis using direct integration.
*   **Response Spectrum Analysis (RSA):** Estimate peak responses using modal combination methods (SRSS).

**Non-Linear Analysis:**

*   **Material Models:** A library of hysteretic material models, including Bilinear, and Bouc-Wen.
*   **Element Formulations:** Define non-linear behavior at the element level.
*   **Robust Solvers:** Utilize iterative Newton-Raphson schemes within the Newmark-Beta solver for accurate non-linear solutions.

**Ground Motion Tools:**

*   **Record Processing:** Easily load, process, and manipulate earthquake ground motion records.
*   **Standard Formats:** Includes helper functions for common ground motion data, such as the El Centro record.

**Coming Soon:**

*   More advanced material models (e.g., Ramberg-Osgood).
*   More modal combination rules (e.g., CQC).

Installation
------------

.. code-block:: bash

   pip install structdyn


Quick Example
-------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from structdyn.ground_motions.ground_motion import GroundMotion
   from structdyn.sdf.response_spectrum import ResponseSpectrum
   from structdyn.utils.helpers import elcentro_chopra

   elc = elcentro_chopra()
   gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

   periods = np.arange(0, 5.01, 0.1)
   rs = ResponseSpectrum(periods, 0.02, gm)

   results = rs.compute()
   print(results)


API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
