---
title: 'structdyn: An open-source Python library for structural dynamics analysis'
authors:
  - name: Abinash Mandal
    orcid: 0009-0000-4234-0861
    affiliation: 1
affiliations:
 - name: University of Nevada, Reno
   index: 1
date: 25 February 2026
bibliography: paper.bib
tags:
  - Python
  - structural dynamics
  - earthquake engineering
  - SDOF
  - MDOF
  - numerical integration
---

# Summary

StructDyn is an open‑source Python library for performing classical structural dynamics analysis. It provides a simple, accessible, and transparent framework for modeling and analyzing the dynamic response of Single‑Degree‑of‑Freedom (SDF) and Multi‑Degree‑of‑Freedom (MDF) systems. The library features a comprehensive suite of numerical integration methods (central difference, Newmark‑Beta, and linear interpolation), tools for modal analysis, and utilities for handling earthquake ground motion data, including direct support for the PEER NGA‑West2 format. Its design prioritizes clarity and ease of use, with implementations that closely follow the algorithms presented in foundational textbooks such as Chopra’s *Dynamics of Structures* [@Chopra2020]. The library is thoroughly tested against examples from the literature, making it an ideal tool for teaching, self‑learning, and research in structural and earthquake engineering.

# Statement of Need

The study of structural dynamics is fundamental to civil, mechanical, and aerospace engineering. While powerful commercial and open‑source software packages exist (e.g., OpenSees, SAP2000, ANSYS), they are often complex, proprietary, or require significant expertise to modify and extend. Students, educators, and researchers frequently need to implement textbook algorithms themselves to understand their inner workings. This creates a barrier to entry and hinders rapid prototyping of new ideas.

StructDyn addresses this gap by providing a **transparent**, **well‑documented**, and **easy‑to‑use** Python implementation of the of the core methods in structural dynamics. Its modular, object-oriented design allows users to easily inspect the source code, modify existing components, or extend its functionality with new numerical methods or material models. This makes it an ideal pedagogical tool for teaching the principles of structural dynamics and a convenient workbench for researchers to test new ideas on simple, well-defined systems before moving to more complex models. By focusing on clarity and ease of use, StructDyn lowers the barrier to computational structural dynamics, enabling a wider audience to explore and apply these critical engineering principles.

# Functionality and Design

StructDyn is organized into several core modules that reflect its key capabilities:

*   **Single-Degree-of-Freedom (SDF) Systems:** The `sdf` module allows users to define an `SDFSystem` by specifying mass, stiffness, and damping. It supports both analytical solutions for classical cases (e.g., free vibration, harmonic loading) and numerical solutions for arbitrary dynamic loading using interpolation, central difference, or Newmark‑Beta methods.
*   **Multi-Degree-of-Freedom (MDF) Systems:** The `mdf` module enables the creation of an `MDFSystem` from mass and stiffness matrices. It includes a `ModalAnalysis` class to compute natural frequencies and mode shapes. Damping can be built from modal damping ratios.
*   **Numerical Solvers:** A suite of well‑established methods is provided: `central difference` (explicit), `Newmark‑Beta` (with average or linear acceleration), and a `linear interpolation` method for SDF systems. These solvers are implemented step‑by‑step following the formulations in Chopra’s textbook [@Chopra2020]. For large MDF systems, modal superposition can be used to reduce computational cost.
*   **Ground Motion Handling:** The `ground_motions` module provides tools to load, scale, and process earthquake records. It includes support for the common PEER NGA-West2 format, a widely used database in earthquake engineering [@Ancheta2014]. This allows for straightforward analysis of structural response to seismic events.

## Example Usage

The library's object-oriented design leads to an intuitive workflow. The following example demonstrates how to develop a response spectrum due to earthquake ground motion.

```python
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

This focus on simplicity and transparency makes structdyn a valuable asset for both educational and research purposes.

# Acknowledgements

This work relies on the foundational numerical libraries that underpin the scientific Python ecosystem, namely NumPy [@numpy], SciPy [@scipy], Pandas [@pandas], and Matplotlib [@matplotlib]. Acknowledgment is also made to the Pacific Earthquake Engineering Research (PEER) Center for making the NGA-West2 ground motion database available. Additionally, the author acknowledges the use of Large Language Models, including OpenAI's ChatGPT, DeepSeek, and Google's Gemini, for assistance with code cleaning, documentation, and manuscript editing. No specific funding was received for this work.

# References
