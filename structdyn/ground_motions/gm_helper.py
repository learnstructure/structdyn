import re
import pandas as pd
from pathlib import Path

GM_DIR = Path(__file__).resolve().parent


def read_at2(file_path, g_to_ms2=True):
    acc = []
    dt = None
    npts = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse header
    for i, line in enumerate(lines):
        if "NPTS" in line and "DT" in line:
            npts = int(re.search(r"NPTS=\s*(\d+)", line).group(1))
            dt = float(re.search(r"DT=\s*([0-9.]+)", line).group(1))
            data_start = i + 1
            break

    # Read data
    for line in lines[data_start:]:
        acc.extend([float(x) for x in line.split()])

    acc = acc[:npts]

    if g_to_ms2:
        acc = [a * 9.81 for a in acc]

    df = pd.DataFrame({"time": [i * dt for i in range(len(acc))], "acc": acc})

    return df, dt


def load_event(event_name):
    event_path = GM_DIR / event_name

    if not event_path.exists():
        raise FileNotFoundError(f"Event '{event_name}' not found in {GM_DIR}")

    records = {}

    for file in event_path.glob("*.AT2"):
        key = file.stem.lower()
        df, dt = read_at2(file)
        records[key] = {"data": df, "dt": dt, "file": file.name}

    return records


def select_component(event_dict, component="ns"):
    for key in event_dict:
        if component in key:
            return event_dict[key]["data"], event_dict[key]["dt"]
    raise ValueError(f"Component {component} not found")


if __name__ == "__main__":
    el_centro = load_event("el_centro_1940")
    # print(el_centro.keys())
    df_gm, dt = select_component(el_centro, component="rsn6_impvall.i_i-elc-up")
    # ug_ddot = df_gm["acc"].values
    print(df_gm)

# run the file as a module
# python -m structdyn.ground_motions.gm_helper from project root
