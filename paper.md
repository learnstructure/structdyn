---
title: 'StructDyn: An open-source Python library for structural dynamics analysis'
authors:
  - name: Abinash Mandal
    orcid: 0009-0000-4234-0861
    affiliation: 1
affiliations:
 - name: University of Nevada, Reno
   index: 1
date: February 2026
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
StructDyn is an open‑source Python library for performing structural dynamics analysis using numerical methods. It provides a simple, accessible, and transparent framework for modeling and analyzing the dynamic response of Single‑Degree‑of‑Freedom (SDF) and Multi‑Degree‑of‑Freedom (MDF) systems, both linear and non-linear. The library features a comprehensive suite of numerical integration methods (central difference, Newmark‑Beta, and linear interpolation), tools for modal analysis, and utilities for handling earthquake ground motion data, including direct support for the PEER NGA‑West2 format. Its design prioritizes clarity and ease of use, with implementations that closely follow the algorithms presented in foundational textbooks such as Chopra’s *Dynamics of Structures* [@Chopra2020]. The library is thoroughly tested against examples from the book, making it an ideal tool for teaching, self‑learning, and research in structural and earthquake engineering. The modularity of the library allows to integrate finite element method libraries to perform analysis on general structures with known mass and stiffness matrix.

# Statement of Need

The study of structural dynamics is fundamental to civil, mechanical, and aerospace engineering. While powerful commercial and open‑source software packages exist (e.g., OpenSees, SAP2000, ANSYS), they are often complex, proprietary, or require significant expertise to modify and extend. Students, educators, and researchers frequently need to implement textbook algorithms themselves to understand their inner workings. This creates a barrier to entry and hinders rapid prototyping of new ideas.

StructDyn addresses this gap by providing a **transparent**, **well‑documented**, and **easy‑to‑use** Python implementation of the core methods in structural dynamics. Its modular, object‑oriented design allows users to easily inspect the source code, modify existing components, or extend its functionality with new numerical methods or material models. This makes it an ideal pedagogical tool for teaching the principles of structural dynamics and a convenient workbench for researchers to test new ideas on simple, well‑defined systems before moving to more complex models. By focusing on clarity and ease of use, StructDyn lowers the barrier to computational structural dynamics, enabling a wider audience to explore and apply these critical engineering principles.


# State of the Field

The field of structural analysis is dominated by powerful, feature-rich finite element analysis (FEA) software such as SAP2000, ETABS, and the open-source framework OpenSees. While these tools are industry standards for complex, large-scale modeling, `structdyn` occupies a distinct and complementary niche. It is not intended to replace these comprehensive FEA packages, but rather to provide a more accessible, scriptable environment for fundamental structural dynamics analysis, preliminary design, and education. While OpenSees is unparalleled for detailed finite element analysis of complex structures, `structdyn` prioritizes simplicity and transparency, making it ideal for teaching fundamental concepts and rapidly prototyping new algorithms. Its Python-based nature allows for easy integration into larger computational workflows, which is often cumbersome with compiled, standalone software.


# Software Design

`structdyn` is designed with an intuitive object-oriented architecture. The primary components are the `SDF` and `MDF` classes, which encapsulate the properties and methods for single and multi-degree-of-freedom systems, respectively.

- **Modularity:** The library is organized into logical modules. For instance, `MDF` systems can be easily created from shear building properties. Modal analysis is handled by a dedicated `modal` submodule. Numerical solvers, ground motion tools, and visualization are all implemented as distinct, well-defined components.
- **Dependencies:** `structdyn` is built upon the standard scientific Python stack, leveraging `numpy` [@numpy] for numerical operations, `scipy` [@scipy] for core scientific computing algorithms (including its equation solvers), and `pandas` for data management. Plots and animations are generated using `matplotlib` [@matplotlib].
- **Extensibility:** The modular design makes the library easy to extend. For example, users can define custom non-linear material models or implement new numerical integration schemes.
- **Visualization:** A key design choice was to tightly integrate visualization with the analysis objects. The `plot` attribute on `MDF` objects provides direct access to visualization methods for mode shapes and animated responses, making the interpretation of results seamless.

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
print(spectra["Sd"][20])  # result is 0.189675
```

This focus on simplicity and transparency makes structdyn a valuable asset for both educational and research purposes.

# Research Impact

`structdyn` is a valuable tool for both education and research in structural engineering.

- **Educational Tool:** Its simple API and visualization capabilities make it an excellent tool for teaching and learning the fundamental concepts of structural dynamics. Students can easily explore the effects of changing mass, stiffness, or damping on a structure's response.
- **Research Platform:** For researchers, `structdyn` provides a platform for conducting parametric studies, validating new analytical models, and performing preliminary analyses that can inform more detailed FEA modeling. Because it is open-source and scriptable, it promotes reproducible and transparent research workflows.
- **Preliminary Design:** Practicing engineers can use `structdyn` to quickly evaluate the dynamic performance of different design alternatives in the early stages of a project.

# AI usage disclosure

The author acknowledges the use of Large Language Models, including OpenAI's ChatGPT, DeepSeek, and Google's Gemini, for assistance with code cleaning, documentation, and manuscript editing. All AI-assisted content was reviewed, edited, and validated by the author for accuracy and clarity.

# Acknowledgements

This work relies on the foundational numerical libraries that underpin the scientific Python ecosystem, namely NumPy [@numpy], SciPy [@scipy], Pandas [@pandas], and Matplotlib [@matplotlib]. Acknowledgment is also made to the Pacific Earthquake Engineering Research (PEER) Center for making the NGA-West2 ground motion database available. No specific funding was received for this work.

# References
