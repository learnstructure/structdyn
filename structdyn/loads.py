"""
Load history definitions for dynamic structural analysis.

This module provides the `LoadHistory` class, which bundles a time vector
together with the corresponding force values applied to a structure during
a dynamic analysis. It is the dynamic counterpart of a static load case
and is intended for use with both SDF and MDF systems.

Typical usage
-------------
>>> import numpy as np
>>> from structdyn.loads import LoadHistory
>>> t = np.linspace(0, 1.0, 101)
>>> p = np.sin(2 * np.pi * t)
>>> load = LoadHistory(time_steps=t, load_values=p)
>>> sdf.find_response(load, method="newmark_beta")

For MDF systems, the load can be specified on a subset of DOFs, and the
full (n_t, ndof) force matrix is assembled automatically:

>>> load = LoadHistory(t, p, dof=[0])     # act only on DOF 0
>>> F = load.expand(ndof=3)              # shape (n_t, 3)
>>> mdf.find_response(load, method="newmark_beta")
"""

from __future__ import annotations

import numpy as np


class LoadHistory:
    """
    A dynamic load history: a time vector paired with the force values applied
    at each time step.

    This class is the dynamic-load counterpart of a static load case. It keeps
    the time discretization and the load values together so that they cannot be
    accidentally mismatched, and performs the validation (length, uniform time
    spacing, dimensionality) once at construction time instead of in every
    numerical solver.

    For MDF systems, the load may be specified only on a subset of degrees of
    freedom via the ``dof`` argument. The full ``(n_t, ndof)`` force matrix is
    then assembled on demand by :meth:`expand`.

    Parameters
    ----------
    time_steps : array-like
        1D array of time points (uniformly spaced). Units are arbitrary, but
        should be consistent with the chosen unit system.
    load_values : array-like
        Force values at each time step. Either:

        - a 1D array of shape ``(n,)`` — typical for SDF, or for MDF when the
          load acts only on a subset of DOFs (use ``dof`` to specify which);
        - a 2D array of shape ``(n, ndof)`` — full force matrix for an MDF
          system. ``dof`` is then ignored.
    dof : int or sequence of int, optional
        Indices of the degrees of freedom on which the load acts. Required
        when ``load_values`` is 1D and the load is intended for an MDF system.
        When ``load_values`` is 2D, ``dof`` must be ``None`` (the column count
        already specifies the DOFs).

    Attributes
    ----------
    time_steps : numpy.ndarray
        Time vector as a 1D float array.
    load_values : numpy.ndarray
        Force values as a 1D (SDF or sparse MDF) or 2D (full MDF) float array.
    ndof : int or None
        Number of DOFs the load resolves to. ``1`` for an SDF-style 1D load
        (no ``dof`` specified), ``None`` for a sparse MDF load (use
        :meth:`expand` to place it into a full matrix), or an integer for a
        full 2D load.
    dof : numpy.ndarray or None
        Indices of the DOFs the load acts on, or ``None`` if not specified.
    dt : float
        Uniform time step.
    """

    def __init__(self, time_steps, load_values, dof=None):
        time_steps = np.asarray(time_steps, dtype=float)
        load_values = np.asarray(load_values, dtype=float)

        if time_steps.ndim != 1:
            raise ValueError("time_steps must be a 1D array")

        if load_values.ndim == 1:
            self.dof = self._normalize_dof(dof)
            # 1D load:
            #   - with dof specified  -> sparse MDF (ndof resolved via expand)
            #   - without dof          -> SDF or scalar 1D MDF (ndof = 1)
            self.ndof = None if self.dof is not None else 1
        elif load_values.ndim == 2:
            if dof is not None:
                raise ValueError(
                    "When load_values is a 2D (n_t, ndof) matrix, "
                    "'dof' must be None: the columns already specify the DOFs."
                )
            self.dof = None
            self.ndof = load_values.shape[1]
        else:
            raise ValueError(
                "load_values must be 1D (SDF) or 2D (MDF), "
                f"got ndim={load_values.ndim}"
            )

        if time_steps.shape[0] != load_values.shape[0]:
            raise ValueError(
                "time_steps and load_values must have the same length "
                f"(got {time_steps.shape[0]} and {load_values.shape[0]})"
            )

        if time_steps.size < 2:
            raise ValueError("time_steps must contain at least two points")

        dt = np.diff(time_steps)
        if not np.allclose(dt, dt[0]):
            raise ValueError("Time vector must be uniformly spaced")

        if self.dof is not None and load_values.ndim == 1:
            # Sanity: dof not exceeding the load width
            if load_values.ndim == 1 and self.dof.size > 1:
                # 1D values with multiple DOFs means each column should be
                # a separate load — caller should pass a 2D array instead.
                if load_values.shape[0] == self.dof.size:
                    raise ValueError(
                        "load_values is 1D but its length matches len(dof); "
                        "pass a 2D (n_t, len(dof)) array instead."
                    )

        self.time_steps = time_steps
        self.load_values = load_values
        self.dt = float(dt[0])

    @staticmethod
    def _normalize_dof(dof):
        if dof is None:
            return None
        d = np.asarray(dof, dtype=int)
        if d.ndim != 1:
            raise ValueError("dof must be a 1D sequence of int")
        if d.size == 0:
            raise ValueError("dof must be non-empty")
        if d.size != len(np.unique(d)):
            raise ValueError("dof indices must be unique")
        if np.any(d < 0):
            raise ValueError("dof indices must be non-negative")
        return d

    # ------------------------------------------------------------------ #
    # Expansion into the full (n_t, ndof) force matrix
    # ------------------------------------------------------------------ #
    def expand(self, ndof):
        """
        Return the full ``(n_t, ndof)`` force matrix.

        For a 1D load with a ``dof`` specification, the values are placed at
        the indicated DOF columns and the rest are zero. For a 2D load,
        the array is returned unchanged (must already have ``ndof`` columns).

        Parameters
        ----------
        ndof : int
            Number of degrees of freedom of the target MDF system.

        Returns
        -------
        numpy.ndarray
            Force matrix of shape ``(n_t, ndof)``.
        """
        if self.load_values.ndim == 2:
            if self.load_values.shape[1] != ndof:
                raise ValueError(
                    f"LoadHistory has shape (n_t, {self.load_values.shape[1]}) "
                    f"but the MDF system has ndof={ndof}"
                )
            return self.load_values

        # 1D load — needs placement
        if self.dof is None:
            raise ValueError(
                "Cannot expand a 1D LoadHistory without a 'dof' specification. "
                "Construct it with LoadHistory(t, p, dof=[...]) or pass a "
                "2D (n_t, ndof) load_values matrix."
            )
        if np.any(self.dof >= ndof):
            raise ValueError(
                f"dof index {int(self.dof.max())} is out of range for "
                f"ndof={ndof}"
            )

        F = np.zeros((len(self), ndof), dtype=float)
        if self.dof.size == 1:
            F[:, self.dof[0]] = self.load_values
        else:
            # Many DOFs receive the same 1D time-history value
            F[:, self.dof] = self.load_values[:, None]
        return F

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_constant(cls, time_steps, p0, dof=None):
        """
        Build a LoadHistory whose load is constant in time.

        Parameters
        ----------
        time_steps : array-like
            1D time vector.
        p0 : float or array-like
            Constant force value. Scalar for SDF, length-``ndof`` array for MDF.
        dof : int or sequence of int, optional
            When ``p0`` is a scalar and the load is for an MDF system, the
            DOFs on which the constant load is applied.
        """
        time_steps = np.asarray(time_steps, dtype=float)
        p0 = np.asarray(p0, dtype=float)
        if p0.ndim == 0:
            return cls(time_steps, np.full_like(time_steps, float(p0)), dof=dof)
        if dof is not None:
            raise ValueError(
                "Pass either a scalar p0 (with dof=) or a length-ndof p0 "
                "array (without dof=). Not both."
            )
        load = np.tile(p0, (time_steps.shape[0], 1))
        return cls(time_steps, load)

    # ------------------------------------------------------------------ #
    # Sequence / array-like protocol
    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.time_steps.shape[0]

    def __repr__(self):
        if self.dof is not None:
            dof_str = f"dof={list(self.dof)}"
        else:
            dof_str = f"ndof={self.ndof}"
        return (
            f"LoadHistory(n={len(self)}, {dof_str}, "
            f"dt={self.dt:g}, t=[{self.time_steps[0]:g}, "
            f"..., {self.time_steps[-1]:g}])"
        )

    # ------------------------------------------------------------------ #
    # Arithmetic — combine load histories (e.g. self-weight + lateral)
    # ------------------------------------------------------------------ #
    def __add__(self, other):
        if not isinstance(other, LoadHistory):
            return NotImplemented
        if not np.allclose(self.time_steps, other.time_steps):
            raise ValueError("Cannot add LoadHistories on different time grids")
        # Both 1D with the same DOF index → 1D result
        if self.dof is not None and other.dof is not None:
            if not np.array_equal(self.dof, other.dof):
                raise ValueError(
                    "Cannot add LoadHistories on different DOFs "
                    f"({list(self.dof)} vs {list(other.dof)}); "
                    "combine them via expand() into a full load matrix."
                )
            return LoadHistory(
                self.time_steps, self.load_values + other.load_values, dof=self.dof
            )
        # Otherwise (one is 2D), require compatible ndof
        if self.ndof != other.ndof:
            raise ValueError(
                f"Cannot add LoadHistories with different ndof "
                f"({self.ndof} vs {other.ndof})"
            )
        if self.dof is None and other.dof is None:
            return LoadHistory(self.time_steps, self.load_values + other.load_values)
        raise ValueError(
            "Cannot add a sparse LoadHistory (with dof=) to a full one "
            "(without dof=); expand first."
        )

    def __mul__(self, scalar):
        try:
            s = float(scalar)
        except (TypeError, ValueError):
            return NotImplemented
        kwargs = {"dof": self.dof} if self.dof is not None else {}
        return LoadHistory(self.time_steps, self.load_values * s, **kwargs)


__all__ = ["LoadHistory"]
