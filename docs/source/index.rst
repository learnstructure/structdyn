structdyn documentation
=======================

A Python library for structural dynamics analysis.

Features
--------

- Single Degree of Freedom (SDF) systems
- Ground motion handling
- Response spectrum computation
- Analytical response solutions

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
