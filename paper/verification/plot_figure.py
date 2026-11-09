"""Plot the comparison of structdyn with OpenSeesPy from histories.npz (written by verify.py).

Run from the repository root after verify.py:

    python paper/verification/plot_figure.py

Writes paper/figure_verification.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figure_verification.png"

# Categorical slots 1 and 2 of the reference palette, validated on white (CVD dE 24.7, normal dE 33.6, >= 3:1)
STRUCTDYN = "#2a78d6"
OPENSEES = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#55554f"
GRID = "#e4e4e0"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelcolor=INK, width=0.6, length=3)


def main() -> None:
    h = np.load(HERE / "histories.npz")
    uy, fy = float(h["sdf_uy"]), float(h["sdf_fy"])
    assert np.allclose(h["sdf_fs_sd"], h["sdf_fs_op"], rtol=0, atol=1e-9 * fy), "spring forces differ"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    # (a) Bilinear SDF: hysteresis loop
    ax = axes[0]
    every = 12
    ax.plot(h["sdf_u_sd"] / uy, h["sdf_fs_sd"] / fy, color=STRUCTDYN, linewidth=1.2, label="structdyn")
    ax.plot(h["sdf_u_op"][::every] / uy, h["sdf_fs_op"][::every] / fy, linestyle="none", marker="o",
            markersize=3.2, markerfacecolor="none", markeredgecolor=OPENSEES, markeredgewidth=0.7,
            label="OpenSeesPy")
    ax.set_xlabel("Displacement $u/u_y$")
    ax.set_ylabel("Spring force $f_s/f_y$")
    ax.set_title("(a) Bilinear SDF system", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    style(ax)

    # (b) Bilinear shear building: roof displacement
    ax = axes[1]
    t = h["mdf_t"]
    ax.plot(t, h["mdf_u_sd"][:, -1] * 100, color=STRUCTDYN, linewidth=1.0, label="structdyn")
    ax.plot(t[::every], h["mdf_u_op"][::every, -1] * 100, linestyle="none", marker="o", markersize=3.2,
            markerfacecolor="none", markeredgecolor=OPENSEES, markeredgewidth=0.7, label="OpenSeesPy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Roof displacement (cm)")
    ax.set_title("(b) Bilinear 5-story building", loc="left")
    ax.set_xlim(0, t[-1])
    ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    style(ax)

    # (c) Difference between the programs, relative to the peak response
    ax = axes[2]
    d_sdf = np.abs(h["sdf_u_sd"] - h["sdf_u_op"]) / np.max(np.abs(h["sdf_u_op"]))
    d_mdf = np.max(np.abs(h["mdf_u_sd"] - h["mdf_u_op"]), axis=1) / np.max(np.abs(h["mdf_u_op"]))
    floor = 1e-17  # zero differences are drawn at the bottom of the axis
    ax.semilogy(h["sdf_t"], np.maximum(d_sdf, floor), color=INK, linewidth=0.6, label="SDF system (a)")
    ax.semilogy(t, np.maximum(d_mdf, floor), color=INK_2, linewidth=0.6, linestyle=(0, (3, 1.5)),
                label="building, all floors (b)")
    ax.set_ylim(1e-17, 1e-9)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Difference / peak response")
    ax.set_title("(c) structdyn − OpenSeesPy", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    style(ax)

    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
