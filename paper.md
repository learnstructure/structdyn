---
title: 'StructDyn: A Python library for structural dynamics analysis'
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
  - SDOF, MDOF
  - numerical integration
---

# Summary

StructDyn is an open-source Python library for performing classical structural dynamics analysis. It provides a simple, accessible, and transparent framework for modeling and analyzing the dynamic response of Single-Degree-of-Freedom (SDF) and Multi-Degree-of-Freedom (MDF) systems. The library features a range of numerical integration methods, tools for modal analysis, and utilities for handling earthquake ground motion data. Its design prioritizes clarity and ease of use, making it an ideal tool for teaching, self-learning, and preliminary research in the field of structural and earthquake engineering.

# Statement of Need

The study of structural dynamics is fundamental to civil, mechanical, and aerospace engineering. While numerous commercial software packages exist for complex structural analysis, there is a distinct need for a tool that is simple, transparent, and geared towards education and preliminary research. Many students, educators, and researchers in structural dynamics rely on implementing algorithms directly from textbooks to understand their nuances.

StructDyn addresses this need by providing a direct, easy-to-use Python implementation of the methods presented in foundational textbooks like Chopra's *Dynamics of Structures* [@Chopra2020]. Its modular, object-oriented design allows users to easily inspect the source code, modify existing components, or extend its functionality with new numerical methods or material models. This makes it an ideal pedagogical tool for teaching the principles of structural dynamics and a convenient workbench for researchers to test new ideas on simple, well-defined systems before moving to more complex models. By focusing on clarity and ease of use, StructDyn lowers the barrier to computational structural dynamics, enabling a wider audience to explore and apply these critical engineering principles.

# Functionality and Design

StructDyn is organized into several core modules that reflect its key capabilities:

*   **Single-Degree-of-Freedom (SDF) Systems:** The `sdf` module allows users to define an `SDFSystem` by specifying mass, stiffness, and damping. It supports both analytical solutions for classical cases (e.g., free vibration, harmonic loading) and numerical solutions for arbitrary dynamic loading.
*   **Multi-Degree-of-Freedom (MDF) Systems:** The `mdf` module enables the creation of an `MDFSystem` from mass and stiffness matrices. It includes a `ModalAnalysis` class to compute natural frequencies and mode shapes, which are essential for understanding the system's inherent dynamic properties.
*   **Numerical Solvers:** A suite of well-established numerical integration methods are provided, including the Central Difference, Newmark-Beta, and Linear Interpolation methods. These solvers are implemented in a clear, step-by-step manner that closely follows textbook formulations.
*   **Ground Motion Handling:** The `ground_motions` module provides tools to load, scale, and process earthquake records. It includes support for the common PEER NGA-West2 format, a widely used database in earthquake engineering [@Ancheta2014]. This allows for straightforward analysis of structural response to seismic events.

## Example Usage

The library's object-oriented design leads to an intuitive workflow. The following example demonstrates how to define a simple MDF system and find its natural periods.

```python
import numpy as np
from structdyn.mdf.mdf_system import MDFSystem
from structdyn.mdf.modal_analysis import ModalAnalysis

# Define a 2-DOF system
mass = np.array([[1.0, 0.0], [0.0, 1.0]])
stiffness = np.array([[200.0, -100.0], [-100.0, 100.0]])

# Create the MDF system and perform modal analysis
system = MDFSystem(mass, stiffness)
modes = ModalAnalysis(system)

# Print the natural periods
print(f"Natural Periods (s): {modes.T}")
```

This focus on simplicity and transparency makes StructDyn a valuable asset for both educational and research purposes.

# Acknowledgements

This work relies on the foundational numerical libraries that underpin the scientific Python ecosystem, namely NumPy [@numpy], SciPy [@scipy], Pandas [@pandas], and Matplotlib [@matplotlib]. Acknowledgment is also made to the Pacific Earthquake Engineering Research (PEER) Center for making the NGA-West2 ground motion database available. Additionally, the author acknowledges the use of Large Language Models, including OpenAI's ChatGPT, DeepSeek, and Google's Gemini, for assistance with code cleaning, documentation, and manuscript editing.

# References
