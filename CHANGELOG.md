# Changelog

All notable changes to `structdyn` are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). Releases are published on
[PyPI](https://pypi.org/project/structdyn/) and archived on
[Zenodo](https://doi.org/10.5281/zenodo.18676816).

## [Unreleased]

### Added
- Continuous integration: the test suite runs on Linux, Windows, and macOS with Python 3.11–3.13 for every push to `main` and every pull request.
- This changelog, a support and governance section in `CONTRIBUTING.md`, and project links in `pyproject.toml`.

### Changed
- The workflow that builds the JOSS paper draft moved to `.github/workflows/`, where GitHub runs it, and runs only when the paper changes.
- `GroundMotion.to_core` and response spectrum analysis with `params` raise an `ImportError` that names the missing optional package (`struct_core` or `struct_load`) and how to install it.
- The tests of these two integrations are skipped when the optional packages are not installed; a separate CI job installs them from GitHub and runs the full suite.

### Fixed
- `Bilinear`: the post-yield stiffness is now `alpha * k0`, as documented. The kinematic hardening modulus was set to `alpha * k0`, which gave a post-yield stiffness of `alpha / (1 + alpha) * k0` (0.0476 `k0` instead of 0.05 `k0` for `alpha = 0.05`). Nonlinear responses computed with `Bilinear` and `alpha > 0` change; the top-floor displacement of the Example 16.4 test at 2.0 s changes by 1.0 %. Found by comparing with OpenSees `Steel01`. `alpha` outside `0 <= alpha < 1` now raises a `ValueError`.

## [0.8.0] - 2026-11-03

The package uploaded to PyPI as 0.8.0 on 2026-09-07 was built before the changes marked (\*).
They are part of the GitHub release v0.8.0 and its Zenodo archive, and reach PyPI with the next release.

### Added
- `MDF.from_fem2d`, `MDF.from_fem3d`, and `MDF.from_fem`: build an MDF system from the stiffness and mass matrices of a `fem2d` or `fem3d` frame model, so that frame models can be analysed with the modal, response spectrum, and response-history methods of `structdyn` (\* for `fem3d`).
- Exact static (Guyan) condensation of massless degrees of freedom in modal analysis (\*).
- `LoadHistory`: force histories applied to selected degrees of freedom of an MDF system.
- `GroundMotion` scale factor for unit conversion and record scaling.
- `GroundMotion.from_csv`, `GroundMotion.from_file`, `GroundMotion.from_elcentro_chopra`, `GroundMotion.available_events`, and the `pga`, `pga_g`, and `duration` properties (\*).
- Response spectrum analysis with the ASCE 7 design spectrum (`SDS`, `SD1`, `TL`), CQC, SRSS, and ABS modal combination, effective modal masses, and mass participation ratios (\*).

### Changed
- **Breaking**: `MDF.find_response` takes a `LoadHistory` instead of separate time and load arrays.
- **Breaking** (\*): the default `scale_factor` of `GroundMotion` is now 1.0, so accelerations stay in the units of the record unless a conversion factor is given. It was 9.81, which silently converted records in g to m/s².
- Invalid or unsupported inputs now raise an error instead of falling back silently: the central difference method for nonlinear and frame models, and a Ramberg–Osgood iteration that does not converge (\*).

## [0.7.6] - 2026-07-25

### Fixed
- The MDF numerical methods are exported from `structdyn.mdf`, so `MDF.find_response` no longer raises a missing-attribute error ([#2](https://github.com/learnstructure/structdyn/issues/2), reported by @Saharsha-Jamarkattel).

## [0.7.5] - 2026-04-19

### Changed
- The main classes (`SDF`, `MDF`, `GroundMotion`, `ResponseSpectrum`, and the material models) can be imported directly from `structdyn`.

## [0.7.4] - 2026-03-07

### Added
- Visualization of SDF and MDF systems: the structure, mode shapes, and animated responses.
- Takeda hysteretic model.
- Stability checks of the time step for the central difference and Newmark methods.

## [0.7.2] - 2026-03-02

### Added
- Ramberg–Osgood material model, and MDF systems whose stories use different material models.

### Changed
- Nonlinear MDF analysis follows the Newmark procedure with Newton–Raphson iteration of Chopra, *Dynamics of Structures*.

## [0.7.0] - 2026-03-01

### Added
- Material model interface with linear elastic, elastic–perfectly plastic, bilinear, and Bouc–Wen models for nonlinear SDF analysis.
- Nonlinear response-history analysis of MDF shear buildings.
- Response spectrum analysis of MDF systems.

## [0.5.1] - 2026-02-17

First release on PyPI; same code as 0.5.0 with updated citation metadata.

## [0.5.0] - 2026-02-17

### Added
- SDF systems: analytical response to harmonic loads; interpolation, central difference, and Newmark time stepping; elastic–perfectly plastic response.
- Ground motions: PEER NGA-West2 `.AT2` reader, bundled records, and the El Centro record of Chopra's *Dynamics of Structures*.
- Elastic response spectra.
- MDF systems: shear-building builder, modal analysis, free vibration, damping matrices from modal damping ratios, modal superposition, and central difference and Newmark time stepping.
- Test suite reproducing worked examples of Chopra's *Dynamics of Structures*, and documentation on Read the Docs.

[Unreleased]: https://github.com/learnstructure/structdyn/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/learnstructure/structdyn/compare/v0.7.4...v0.8.0
[0.7.4]: https://github.com/learnstructure/structdyn/compare/v0.5.1...v0.7.4
[0.5.1]: https://github.com/learnstructure/structdyn/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/learnstructure/structdyn/releases/tag/v0.5.0
