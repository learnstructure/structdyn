# from structdyn.utils.material_models import BoucWen
# from structdyn.mdf.mdf_helpers.element_models import ShearStoryElement

# # Material models for each story
# story1_material = BoucWen(k0=1000, alpha=0.02, A=1.0, beta=0.5, gamma=0.5, n=1)
# story2_material = BoucWen(k0=800, alpha=0.02, A=1.0, beta=0.5, gamma=0.5, n=1)

# # Elements: story between ground (DOF 0) and first floor (DOF 1), etc.
# elements = [
#     ShearStoryElement(story1_material, dof_indices=[0, 1]),
#     ShearStoryElement(story2_material, dof_indices=[1, 2]),
# ]

# print("Elements created with Bouc-Wen material models:")
# for i, element in enumerate(elements):
#     print(f"Element {i+1}: {element}")

import numpy as np
import matplotlib.pyplot as plt
from structdyn.mdf import MDF
from structdyn.utils.material_models import BoucWen
from structdyn.mdf.mdf_helpers.element_models import ShearStoryElement

# Masses of the two floors
masses = [1.0, 1.0]
ndof = len(masses)

# Placeholder stiffness matrix (not used when elements are present)
K_placeholder = np.zeros((ndof, ndof))

# Create the MDF system
system = MDF(np.diag(masses), K_placeholder)

# Material models
story1_material = BoucWen(k0=1000, alpha=0.02, A=1.0, beta=0.5, gamma=0.5, n=1)
story2_material = BoucWen(k0=800, alpha=0.02, A=1.0, beta=0.5, gamma=0.5, n=1)

# Elements
elements = [
    ShearStoryElement(story1_material, dof_indices=[0]),  # base story
    ShearStoryElement(story2_material, dof_indices=[0, 1]),  # interior story
]

# Load definition (example: sinusoidal force on top floor)
dt = 0.01
t = np.arange(0, 10, dt)
p = np.zeros((len(t), ndof))
p[:, 1] = 10 * np.sin(2 * np.pi * 1.0 * t)

# Run analysis
response = system.find_response(
    time=t,
    load=p,
    method="newmark_beta",
    elements=elements,  # tol=1e-6, max_iter=20
)

print(response)
# Plot top floor displacement
response.plot(
    x="time",
    y="u1",
    kind="line",
)
plt.show()
