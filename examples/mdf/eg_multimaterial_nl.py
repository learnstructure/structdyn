# Just for demonstrating how to use different material models in an MDF analysis. The results are not meant to be meaningful.
import numpy as np
import matplotlib.pyplot as plt
from structdyn.mdf import MDF
from structdyn.utils.material_models import (
    BoucWen,
    Bilinear,
    ElasticPerfectlyPlastic,
    RambergOsgood,
    LinearElastic,
)
from structdyn.mdf.mdf_helpers.element_models import ShearStoryElement
from structdyn.ground_motions.ground_motion import GroundMotion

# Define ground motion as given in the same example
dt = 0.1
time = np.arange(0, 2.01, dt)
acc = (1.93 / 9.81) * np.sin(2 * np.pi * time)
acc[time > 1.0] = 0
gm = GroundMotion.from_arrays(acc, dt)
# Influence vector
inf_vec = np.ones(5)

# Define MDF system
masses = np.ones(5) * 0.2591 * 1e5
stiffness = np.ones(5) * 100 * 1e5
system = MDF.from_shear_building(masses, stiffness)

# system = MDF(np.diag(masses), np.diag(stiffness))
system.set_modal_damping(zeta=[0.05, 0.05, 0.05, 0.05, 0.05])  # set damping matrix

# Material models
story1_material = BoucWen(k0=125e3 / 0.0125)  # base story
story2_material = ElasticPerfectlyPlastic(uy=0.0125, fy=125e3)  # interior story
story3_material = Bilinear(uy=0.0125, fy=125e3, alpha=0.05)  # interior story
story4_material = RambergOsgood(125e3 / 0.0125, 125e3, n=10, alpha=1)  # interior story
story5_material = LinearElastic(stiffness=125e3 / 0.0125)  # top story

# Elements
elements = [
    ShearStoryElement(story1_material, dof_indices=[0]),  # base story
    ShearStoryElement(story2_material, dof_indices=[0, 1]),  # interior story
    ShearStoryElement(story3_material, dof_indices=[1, 2]),  # interior story
    ShearStoryElement(story4_material, dof_indices=[2, 3]),  # interior story
    ShearStoryElement(story5_material, dof_indices=[3, 4]),  # top story
]

# Run analysis
response = system.find_response_ground_motion(
    gm, inf_vec, method="newmark_beta", elements=elements, tol=1e-6, max_iter=20
)


print(response.iloc[:, :6])  # print displacements of all 5 DOFs
print(response.iloc[20, 5])  # result is 0.03998638459760659

# Plot top floor displacement
# response.plot(
#     x="time",
#     y="u1",
#     kind="line",
# )
# plt.show()
