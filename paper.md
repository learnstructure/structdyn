---
title: 'StructDyn: A Python package for structural dynamics analysis'
authors:
  - name: Your Name
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Your Affiliation
   index: 1
date: 17 July 2024
bibliography: paper.bib
tags:
  - Python
  - structural dynamics
  - earthquake engineering
  - SDOF
  - numerical integration
---

# Summary

StructDyn is a lightweight and user-friendly Python package for the dynamic analysis of structures. The library is designed to provide clear and accessible implementations of fundamental concepts in structural dynamics, focusing primarily on the analysis of Single-Degree-of-Freedom (SDF) systems. It offers tools for solving the equation of motion through various numerical integration schemes, including the Newmark-Beta method and an exact integration method for linearly interpolated excitations. StructDyn supports both linear-elastic and nonlinear force-deformation behaviors, with a built-in model for elastic-perfectly plastic materials. The package also includes utilities for reading and scaling ground motion acceleration records, making it a comprehensive tool for introductory earthquake engineering applications.

# Statement of Need

The analysis of how structures respond to dynamic loads is a cornerstone of civil and earthquake engineering. While sophisticated commercial software packages and advanced open-source frameworks like OpenSees exist for large-scale, complex structural analysis, there is a distinct need for a tool that is simple, transparent, and geared towards education and preliminary research. Many students, educators, and researchers in structural dynamics rely on implementing algorithms directly from textbooks to understand their nuances.

StructDyn addresses this need by providing a direct, easy-to-use Python implementation of the methods presented in foundational textbooks like Chopra's *Dynamics of Structures* [@Chopra2020]. Its modular, object-oriented design allows users to easily inspect the source code, modify existing components, or extend its functionality with new numerical methods or material models. This makes it an ideal pedagogical tool for teaching the principles of structural dynamics and a convenient workbench for researchers to test new ideas on simple, well-defined systems before moving to more complex models. By focusing on clarity and ease of use, StructDyn lowers the barrier to computational structural dynamics, enabling a wider audience to explore and apply these critical engineering principles.

# References
