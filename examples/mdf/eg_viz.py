import numpy as np
from structdyn.mdf import MDF
from structdyn.ground_motions import GroundMotion
from structdyn.utils.helpers import elcentro_chopra

# 1. Create a shear building model
masses = [1000, 1000, 1000, 1000]  # Mass at each story (kg)
stiffnesses = [2e6, 2e6, 2e6, 2e6]  # Stiffness of each story (N/m)
shear_building = MDF.from_shear_building(masses, stiffnesses)

# 2. Plot the undeformed structure
# This will open a matplotlib window showing the building layout.
shear_building.plot.structure()

# # 3. Perform modal analysis and plot a mode shape
# # This calculates the modes and then plots the second mode shape.
shear_building.modal.modal_analysis()
shear_building.plot.mode_shape(mode_number=3)

# # 4. Simulate a dynamic response
# # For example, let's apply a simple sinusoidal ground motion.
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], 0.02)

response = shear_building.find_response_ground_motion(gm)

# # 5. Animate the response
# # This will create and display an animation of the building's movement.
# # The 'anim' object is returned if you need to save or further process it.
# anim = shear_building.plot.animate_response(response)
anim = shear_building.plot.animate_response(
    response,
    scale_factor=20.0,  # Increased scale factor for clarity
    ground_motion=(gm.time, gm.acc_g),
    # save_path="building_response.mp4",  # <-- to save animation, you need to have ffmpeg installed and uncomment this line
)
