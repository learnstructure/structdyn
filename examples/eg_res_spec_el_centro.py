import numpy as np
import pandas as pd
from pathlib import Path
from structdyn.ground_motions.ground_motion import GroundMotion
from structdyn.sdf.response_spectrum import ResponseSpectrum

root_path = Path(__file__).resolve().parent.parent
elc_path = root_path / "structdyn" / "ground_motions" / "data" / "elcentro_chopra.csv"
elc = pd.read_csv(elc_path, header=None)

gm = GroundMotion.from_arrays(elc[1], 0.02)

periods = np.arange(0, 5.01, 0.1)
rs = ResponseSpectrum(periods, 0.02, gm)

results = rs.compute()
print(results)
