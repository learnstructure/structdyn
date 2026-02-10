from pathlib import Path
import numpy as np
import pandas as pd
import re
from importlib import resources


class GroundMotion:
    """Ground motion acceleration time history. (Acceleration in unit of g)"""

    def __init__(self, acc_g, dt, name=None, component=None):
        self.acc_g = np.asarray(acc_g)
        self.dt = float(dt)
        self.time = np.arange(len(acc_g)) * dt

        self.name = name
        self.component = component

    # ---------- Constructors ----------

    @classmethod
    def from_at2(cls, file_path):
        file_path = Path(file_path)
        acc, dt = cls._read_at2(file_path)
        return cls(acc, dt, name=file_path.stem)

    @classmethod
    def from_event(cls, event_name, component, base_dir=None):
        if base_dir is None:
            with resources.as_file(
                resources.files("structdyn.ground_motions") / "data"
            ) as data_dir:
                base_dir = Path(data_dir)

        event_dir = base_dir / event_name
        if not event_dir.exists():
            raise FileNotFoundError(f"Event '{event_name}' not found")

        files = list(event_dir.glob("*.AT2"))
        if not files:
            raise FileNotFoundError("No AT2 files found")

        selected = cls._select_component(files, component)
        acc, dt = cls._read_at2(selected)

        return cls(acc, dt, name=event_name, component=component)

    @classmethod
    def from_arrays(cls, acc_g, dt, name="user_motion"):
        return cls(acc_g, dt, name=name)

    # ---------- Utilities ----------

    @staticmethod
    def _read_at2(file_path):
        acc = []
        dt = None

        with open(file_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "NPTS" in line and "DT" in line:
                dt = float(re.search(r"DT=\s*([0-9.]+)", line).group(1))
                data_start = i + 1
                break

        for line in lines[data_start:]:
            acc.extend(float(x) for x in line.split())

        return np.array(acc), dt

    @staticmethod
    def _select_component(files, component):
        component = component.lower()
        for f in files:
            if component in f.stem.lower():
                return f
        raise ValueError(f"Component '{component}' not found")

    # ---------- Operations ----------

    def scale(self, factor):
        self.acc_g *= factor
        return self

    def scale_to_pga(self, target_pga_g):
        current = np.max(np.abs(self.acc_g))
        self.acc_g *= target_pga_g / current
        return self

    def to_dataframe(self):
        return pd.DataFrame({"time": self.time, "acc_g": self.acc_g})
