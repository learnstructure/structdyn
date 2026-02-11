import numpy as np
import pandas as pd
from structdyn.sdf.sdf import SDF


class ResponseSpectrum:
    """
    Linear elastic displacement response spectrum generator.
    """

    def __init__(self, periods, damping_ratio, ground_motion, method="interpolation"):
        self.periods = np.asarray(periods, dtype=float)
        self.ji = damping_ratio
        self.gm = ground_motion
        self.method = method

        if not (0 <= damping_ratio < 1):
            raise ValueError("Damping ratio must be between 0 and 1")

        if np.any(self.periods < 0):
            raise ValueError("Periods must be positive")

    def compute(self):
        periods = self.periods.copy()
        include_zero = np.any(periods == 0)
        # Remove zero from numerical loop
        periods_nonzero = periods[periods > 0]

        Sd_list = []

        for T in periods_nonzero:

            # Natural frequency
            w_n = 2 * np.pi / T

            m = 1.0  # Use unit mass
            k = m * w_n**2

            sdf = SDF(m, k, ji=self.ji)

            results = sdf.find_response_ground_motion(self.gm, method=self.method)

            u = results["displacement"]

            Sd = np.max(np.abs(u))
            Sd_list.append(Sd)

        Sd = np.array(Sd_list)

        # Compute pseudo spectral values
        w_n_array = 2 * np.pi / periods_nonzero
        pSv = w_n_array * Sd
        pSa = (w_n_array**2) * Sd / 9.81  # convert to g

        df = pd.DataFrame({"T": periods_nonzero, "Sd": Sd, "pSv": pSv, "pSa (g)": pSa})

        # Handle T = 0 case explicitly
        if include_zero:
            PGA = np.max(np.abs(self.gm.acc_g))  # already in g if acc_g is in g

            zero_row = pd.DataFrame(
                {"T": [0.0], "Sd": [0.0], "pSv": [0.0], "pSa (g)": [PGA]}
            )

            df = pd.concat([zero_row, df], ignore_index=True)

        return df.sort_values("T").reset_index(drop=True)
