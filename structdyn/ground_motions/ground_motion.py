from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import re
from importlib import resources


class GroundMotion:
    """Represents a ground motion acceleration time history.

    This class provides a container for ground motion data, typically acceleration
    time histories. It includes methods for reading data from common file formats
    (like AT2), scaling the motion, and accessing its properties.

    Attributes
    ----------
    acc_g : numpy.ndarray
        The acceleration time history in the input units (commonly g).
    scale_factor : float
        Conversion factor applied to ``acc_g`` when the physical acceleration is
        needed for response analysis.
    dt : float
        The time step of the acceleration data.
    time : numpy.ndarray
        The time vector corresponding to the acceleration data.
    name : str, optional
        A name for the ground motion event.
    component : str, optional
        The component of the ground motion (e.g., 'h1', 'h2', 'v').
    """

    def __init__(self, acc_g, dt, name=None, component=None, scale_factor=9.81):
        """Initializes the GroundMotion object.

        Parameters
        ----------
        acc_g : array-like
            Acceleration time history in the input units (commonly g).
        dt : float
            Time step of the acceleration data.
        name : str, optional
            Name of the ground motion event, by default None.
        component : str, optional
            Component of the ground motion, by default None.
        scale_factor : float, optional
            Conversion factor applied to ``acc_g`` when computing the physical
            acceleration used in response analysis. The default is 9.81.
        """
        self.acc_g = np.asarray(acc_g, dtype=float)
        self.dt = float(dt)
        self.time = np.arange(len(self.acc_g)) * dt

        self.name = name
        self.component = component
        self.scale_factor = float(scale_factor)

    # ---------- Properties ----------

    @property
    def acceleration(self):
        """Return the acceleration history in physical units (acc_g * scale_factor)."""
        return self.acc_g * self.scale_factor

    @property
    def pga_g(self) -> float:
        """Peak Ground Acceleration (PGA) in units of g."""
        return float(np.max(np.abs(self.acc_g))) if len(self.acc_g) > 0 else 0.0

    @property
    def pga(self) -> float:
        """Peak Ground Acceleration (PGA) in physical acceleration units."""
        return float(np.max(np.abs(self.acceleration))) if len(self.acc_g) > 0 else 0.0

    @property
    def duration(self) -> float:
        """Total record duration in seconds."""
        return float(self.time[-1]) if len(self.time) > 0 else 0.0

    # ---------- Utilities ----------

    @staticmethod
    def _data_dir(base_dir=None) -> Path:
        """Resolves the root directory for built-in ground motions."""
        if base_dir is not None:
            return Path(base_dir)
        with resources.as_file(
            resources.files("structdyn.ground_motions") / "data"
        ) as d:
            return Path(d)

    @classmethod
    def available_events(cls, base_dir=None) -> dict[str, list[str]]:
        """Lists all available built-in ground motion events and their components.

        Parameters
        ----------
        base_dir : str or pathlib.Path, optional
            Custom base directory to scan. If None, scans the built-in database.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping event names to list of available components.
        """
        data_dir = cls._data_dir(base_dir)
        events: dict[str, list[str]] = {}

        # 1. Chopra El Centro CSV
        if (data_dir / "elcentro_chopra.csv").exists():
            events["elcentro_chopra"] = ["csv"]

        # 2. Subdirectories containing AT2 files
        for item in sorted(data_dir.iterdir()):
            if item.is_dir():
                at2_files = list(item.glob("*.AT2"))
                if at2_files:
                    comps: list[str] = []
                    for f in at2_files:
                        stem = f.stem.lower()
                        if "hor1" in stem:
                            comps.append("hor1")
                        elif "hor2" in stem:
                            comps.append("hor2")
                        elif "up" in stem:
                            comps.append("up")
                        else:
                            comps.append(f.stem)
                    events[item.name] = sorted(comps)

        return events

    # ---------- Constructors ----------

    @classmethod
    def from_at2(cls, file_path, scale_factor=9.81):
        """Creates a GroundMotion object from a PEER NGA (.AT2) file.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the .AT2 file.
        scale_factor : float, optional
            Conversion factor to physical acceleration (default: 9.81).

        Returns
        -------
        GroundMotion
            A new GroundMotion instance.
        """
        file_path = Path(file_path)
        acc, dt = cls._read_at2(file_path)
        return cls(acc, dt, name=file_path.stem, scale_factor=scale_factor)

    @classmethod
    def from_elcentro_chopra(cls, scale_factor=9.81):
        """Creates a GroundMotion object from Chopra's El Centro 1940 CSV record.

        Parameters
        ----------
        scale_factor : float, optional
            Conversion factor applied to acc_g (default: 9.81).

        Returns
        -------
        GroundMotion
            GroundMotion instance for the 1940 El Centro record (dt=0.02 s).
        """
        data_dir = cls._data_dir()
        csv_file = data_dir / "elcentro_chopra.csv"
        df = pd.read_csv(csv_file)
        dt = float(df["time"].iloc[1] - df["time"].iloc[0])
        return cls(
            acc_g=df["acc (g)"].to_numpy(),
            dt=dt,
            name="elcentro_chopra",
            component="hor1",
            scale_factor=scale_factor,
        )

    @classmethod
    def from_event(cls, event_name, component="hor1", base_dir=None, scale_factor=9.81):
        """Loads a ground motion from the built-in event database.

        Parameters
        ----------
        event_name : str
            The name or alias of the earthquake event. Supported built-in events:
            - 'elcentro_chopra' (or 'chopra')
            - 'imperialValley_elCentro_1940' (or 'elcentro', 'imperial', 'imperial_valley')
            - 'northridge_sylmar_1994' (or 'northridge', 'sylmar')
            - 'lomaPrieta_corralitos_1989' (or 'loma_prieta', 'lomaprieta')
            - 'sanFernando_pacoidaDam_1971' (or 'san_fernando', 'sanfernando')
        component : str, optional
            The specific component to load ('hor1'/'h1', 'hor2'/'h2', 'up'/'v').
            Defaults to 'hor1'.
        base_dir : str or pathlib.Path, optional
            Custom base directory of ground motion data.
        scale_factor : float, optional
            Conversion factor applied to acc_g. Defaults to 9.81.

        Returns
        -------
        GroundMotion
            A new GroundMotion instance.

        Raises
        -------
        FileNotFoundError
            If the specified event or component is not found.
        """
        # Support composite syntax like 'northridge:hor2' or 'northridge/hor2'
        if ":" in str(event_name):
            event_name, comp_override = str(event_name).split(":", 1)
            component = comp_override.strip()
        elif "/" in str(event_name):
            event_name, comp_override = str(event_name).split("/", 1)
            component = comp_override.strip()

        norm_name = str(event_name).strip().lower().replace("-", "_").replace(" ", "_")

        # 1. Check for Chopra CSV
        if norm_name in ("elcentro_chopra", "chopra", "el_centro_chopra"):
            return cls.from_elcentro_chopra(scale_factor=scale_factor)

        data_dir = cls._data_dir(base_dir)

        # 2. Check for exact folder match
        target_dir = None
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.lower() == norm_name:
                target_dir = item
                break

        # 3. Canonical aliases mapping
        if target_dir is None:
            aliases = {
                "elcentro": "imperialvalley_elcentro_1940",
                "el_centro": "imperialvalley_elcentro_1940",
                "el_centro_1940": "imperialvalley_elcentro_1940",
                "imperial": "imperialvalley_elcentro_1940",
                "imperial_valley": "imperialvalley_elcentro_1940",
                "northridge": "northridge_sylmar_1994",
                "sylmar": "northridge_sylmar_1994",
                "loma_prieta": "lomaprieta_corralitos_1989",
                "lomaprieta": "lomaprieta_corralitos_1989",
                "corralitos": "lomaprieta_corralitos_1989",
                "san_fernando": "sanfernando_pacoidadam_1971",
                "sanfernando": "sanfernando_pacoidadam_1971",
                "pacoima": "sanfernando_pacoidadam_1971",
            }
            mapped = aliases.get(norm_name)
            for item in data_dir.iterdir():
                if item.is_dir():
                    clean_item_name = item.name.lower().replace("-", "_")
                    if mapped and mapped in clean_item_name:
                        target_dir = item
                        break
                    elif norm_name in clean_item_name:
                        target_dir = item
                        break

        if target_dir is None or not target_dir.exists():
            avail = list(cls.available_events(base_dir).keys())
            raise FileNotFoundError(
                f"Event '{event_name}' not found. Available events: {avail}"
            )

        files = list(target_dir.glob("*.AT2"))
        if not files:
            raise FileNotFoundError(f"No AT2 files found in '{target_dir}'")

        selected, canonical_comp = cls._select_component(files, component)
        acc, dt = cls._read_at2(selected)

        return cls(
            acc,
            dt,
            name=target_dir.name,
            component=canonical_comp,
            scale_factor=scale_factor,
        )

    @classmethod
    def from_arrays(cls, acc_g, dt, name="user_motion", scale_factor=9.81):
        """Creates a GroundMotion object directly from arrays.

        Parameters
        ----------
        acc_g : array-like
            Acceleration time history in units of g.
        dt : float
            Time step of the acceleration data.
        name : str, optional
            A name for the motion, by default "user_motion".
        scale_factor : float, optional
            Conversion factor applied to acc_g. Default is 9.81.

        Returns
        -------
        GroundMotion
            A new GroundMotion instance.
        """
        return cls(acc_g, dt, name=name, scale_factor=scale_factor)

    # ---------- Internal Helpers ----------

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
        comp = str(component).strip().lower()
        comp_map = {
            "h1": "hor1",
            "hor1": "hor1",
            "1": "hor1",
            "x": "hor1",
            "h2": "hor2",
            "hor2": "hor2",
            "2": "hor2",
            "y": "hor2",
            "v": "up",
            "vert": "up",
            "vertical": "up",
            "up": "up",
            "z": "up",
        }
        target_token = comp_map.get(comp, comp)
        for f in files:
            stem = f.stem.lower()
            if target_token in stem:
                return f, target_token
        # Fallback to direct substring match
        for f in files:
            if comp in f.stem.lower():
                return f, comp
        available = [f.stem for f in files]
        raise ValueError(
            f"Component '{component}' not found in {available}. "
            f"Available canonical components are 'hor1' (or 'h1', 'x'), 'hor2' (or 'h2', 'y'), 'up' (or 'v', 'z')."
        )

    # ---------- Operations & Analysis ----------

    def scale(self, factor):
        """Scale the ground motion by a multiplier."""
        self.acc_g *= factor
        return self

    def scale_to_pga(self, target_pga_g):
        """Scale the ground motion so peak ground acceleration equals target_pga_g."""
        current = np.max(np.abs(self.acc_g))
        if current > 0:
            self.acc_g *= target_pga_g / current
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Returns time history as a pandas DataFrame with ['time', 'acc_g', 'acc']."""
        return pd.DataFrame({
            "time": self.time,
            "acc_g": self.acc_g,
            "acc": self.acceleration,
        })

    def response_spectrum(
        self,
        periods=None,
        damping_ratio: float = 0.05,
        method: str = "interpolation",
    ) -> pd.DataFrame:
        """Computes the linear elastic response spectrum for this ground motion.

        Parameters
        ----------
        periods : array-like, optional
            Periods (in seconds) to evaluate. If None, defaults to 100 points
            from 0.02 to 4.0 s.
        damping_ratio : float, optional
            Modal damping ratio (e.g. 0.05 for 5%). Defaults to 0.05.
        method : str, optional
            Integration method ('interpolation', 'newmark_beta', 'central_difference').
            Defaults to 'interpolation'.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ['T', 'Sd', 'pSv', 'pSa', 'pSa (g)'].
        """
        from structdyn.sdf.response_spectrum import ResponseSpectrum

        if periods is None:
            periods = np.linspace(0.02, 4.0, 100)
        rs = ResponseSpectrum(
            periods=periods,
            damping_ratio=damping_ratio,
            ground_motion=self,
            method=method,
        )
        return rs.compute()

    def to_core(self, id: Optional[str] = None) -> Any:
        """Converts to a struct_core.GroundMotion schema model.

        Parameters
        ----------
        id : str, optional
            Identifier for the schema model. If None, defaults to self.name or 'gm'.

        Returns
        -------
        struct_core.GroundMotion
            Schema instance populated with this record.
        """
        from struct_core import GroundMotion as CoreGroundMotion

        return CoreGroundMotion(
            id=id or self.name or "gm",
            dt=float(self.dt),
            acc=self.acc_g.tolist(),
            name=self.name,
        )

