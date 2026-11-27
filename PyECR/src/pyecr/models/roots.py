"""Characteristic roots for a finite surface-exchange boundary."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

__all__ = ["ASYMPTOTIC_BI_THRESHOLD", "solve_alpha_roots"]


ASYMPTOTIC_BI_THRESHOLD = 1.0e8
"""Bi above which roots use a continuous large-Bi approximation."""


def solve_alpha_roots(
    bi: float,
    n_terms: int,
    *,
    start_index: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return roots of ``alpha * tan(alpha) = Bi`` and their residuals.

    Root number ``n`` is bracketed strictly inside
    ``((n - 1) * pi, (n - 1/2) * pi)``.  For very large Bi, the root is
    evaluated with its first continuous correction below the tangent
    singularity.  Residuals are always evaluated directly as
    ``alpha * tan(alpha) - Bi`` in float64 arithmetic.

    Parameters
    ----------
    bi:
        Positive, finite Biot number.
    n_terms:
        Number of positive roots to return.
    start_index:
        Zero-based global index of the first returned root. The default full
        request remains unchanged; opt-in prefix expansion uses disjoint blocks.
    """

    _validate_inputs(bi, n_terms)
    if isinstance(start_index, bool) or not isinstance(start_index, Integral):
        raise TypeError("start_index must be a nonnegative integer")
    if start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    bi_float = float(bi)
    count = int(n_terms)

    if bi_float > ASYMPTOTIC_BI_THRESHOLD:
        roots = (_large_bi_roots(bi_float, count) if start_index == 0 else
                 _large_bi_roots(bi_float, count, start_index=start_index))
    else:
        roots = (_brentq_roots(bi_float, count) if start_index == 0 else
                 _brentq_roots(bi_float, count, start_index=start_index))

    residuals = roots * np.tan(roots) - bi_float
    return roots, residuals.astype(np.float64, copy=False)


def _validate_inputs(bi: float, n_terms: int) -> None:
    if isinstance(bi, bool) or not isinstance(bi, Real):
        raise ValueError("Bi must be positive and finite")
    if not isfinite(float(bi)) or bi <= 0:
        raise ValueError("Bi must be positive and finite")

    if isinstance(n_terms, bool) or not isinstance(n_terms, Integral):
        raise TypeError("n_terms must be a positive integer")
    if n_terms <= 0:
        raise ValueError("n_terms must be a positive integer")


def _brentq_roots(bi: float, n_terms: int, *, start_index: int = 0) -> NDArray[np.float64]:
    roots = np.empty(n_terms, dtype=np.float64)
    minimum_rtol = 4.0 * np.finfo(np.float64).eps

    def equation(alpha: float) -> float:
        return alpha * np.tan(alpha) - bi

    for index in range(start_index, start_index + n_terms):
        left_boundary = index * np.pi
        right_boundary = (index + 0.5) * np.pi
        left = np.nextafter(left_boundary, right_boundary)
        right = np.nextafter(right_boundary, left_boundary)

        roots[index - start_index] = brentq(
            equation,
            left,
            right,
            xtol=np.finfo(np.float64).tiny,
            rtol=minimum_rtol,
            maxiter=100,
        )

    return roots


def _large_bi_roots(bi: float, n_terms: int, *, start_index: int = 0) -> NDArray[np.float64]:
    indices = np.arange(start_index + 1, start_index + n_terms + 1, dtype=np.float64)
    singularities = (indices - 0.5) * np.pi

    roots = singularities - singularities / (bi + 1.0)
    largest_interior_values = np.nextafter(singularities, -np.inf)
    return np.minimum(roots, largest_interior_values).astype(np.float64, copy=False)
