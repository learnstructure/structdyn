# Verification of `structdyn`

These scripts check `structdyn` against references that do not depend on it, and produce the table and the figure
of the JOSS paper.

| Reference | Checks |
| --- | --- |
| Closed form | Natural frequencies of Chopra's Example 10.4 and of a cantilever whose rotation is massless (static condensation). |
| Exact ODE solution | The interpolation method, which is exact for piecewise-linear excitation, against `scipy.integrate.solve_ivp` (DOP853, `rtol = 1e-12`) under the same excitation: Example 5.1 and the El Centro response of Section 6.4. |
| OpenSeesPy | The same integration scheme and time step in OpenSees: central difference (Example 5.2); Newmark linear acceleration (Example 5.4); Newmark average acceleration with an elastoplastic spring (Example 5.5); a linear and a bilinear SDF system under El Centro (`Bilinear` against `Steel01`); and the five-story building of Example 13.8.2, linear and with bilinear stories, under El Centro, including its periods. |
| Textbook | The SRSS base shear of Chopra's Example 13.8.2 from the printed spectral displacements (291.2 kN), and the top-floor displacement of the bilinear building of Example 16.4 at 2.0 s (4.0387 cm). |

Examples and sections refer to Chopra, *Dynamics of Structures*, 5th edition. The El Centro record is the S00E
component of Chopra's Appendix 6, bundled with `structdyn`.

## Run

From the repository root:

```bash
pip install -e .
pip install -r paper/verification/requirements.txt
python paper/verification/verify.py       # writes results.csv, results.md, histories.npz
python paper/verification/plot_figure.py  # writes paper/figure_verification.png
```

The committed results were produced with Python 3.12.12, NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1,
Matplotlib 3.11.2, and OpenSeesPy 3.8.0.0 on Windows 11.

**OpenSeesPy on Windows.** The Windows build of OpenSeesPy 3.8.0.0 is linked against Python 3.12 and the Intel
Fortran runtime. Use a Python 3.12 environment; `requirements.txt` installs the runtime (`intel-fortran-rt`), and
`verify.py` adds its folder to the DLL search path. On Linux and macOS, `pip install openseespy` is enough.

## Results

Every check agrees to round-off except the two textbook checks, which agree within the rounding of the printed
values: 0.02 % for Example 13.8.2 and 0.03 % for Example 16.4 (`results.md`). The two programs are compared with the same model:

- springs as `zeroLength` elements with `-doRayleigh 1`, without which OpenSees ignores the stiffness-proportional
  part of the Rayleigh damping of these elements;
- Rayleigh damping with the initial stiffness (`C = a0 M + a1 K0`), passed to `structdyn` as the matrix `C`;
- the Newmark variant named explicitly in `structdyn` (`acc_type`), whose default is linear acceleration;
- Newton–Raphson iteration to 1e-6 N on the residual force in `structdyn` and to 1e-12 m on the displacement
  increment in OpenSees.

This comparison found that `Bilinear` had a post-yield stiffness of `alpha / (1 + alpha) * k0` instead of the
documented `alpha * k0`; see the changelog. With the correction, Example 16.4 moved from 1.0 % to 0.03 % of the
printed result.
