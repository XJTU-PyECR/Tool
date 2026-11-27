"""Reviewable DCT range suggestions; these functions never run an inversion."""
from math import isfinite

import numpy as np


def suggest_tau_range(time_s):
    """Use the core solver's time-window heuristic, in processed seconds.

    The padding is a search suggestion, not an identifiable-time-scale interval.
    Absolute acquisition timestamps must already have been zeroed in preprocessing.
    """
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError('At least two finite processed times are required.')
    if np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError('Use nonnegative, ordered processed times in seconds.')
    positive = times[times > 0]
    if not positive.size:
        raise ValueError('A positive processed observation time is required.')
    lower, upper = float(positive.min()) / 10., float(positive.max()) * 10.
    if not 0 < lower < upper < np.inf:
        raise ValueError('The suggested time range exceeds numerical limits.')
    return lower, upper


def gcv_boundary_side(lambdas, selected):
    values = tuple(float(x) for x in lambdas)
    if len(values) < 3 or selected is None:
        return None
    if selected == values[0]:
        return 'lower'
    if selected == values[-1]:
        return 'upper'
    return None


def extend_gcv_grid(lambdas, selected):
    """Keep tested candidates and add four decades on the selected boundary.

    Limits prevent repeated explicit extensions from growing without a budget.
    They are numerical search limits, not physical confidence bounds.
    """
    values = tuple(float(x) for x in lambdas)
    if (len(values) < 3 or not all(isfinite(x) and x > 0 for x in values)
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError('GCV candidates must be increasing, positive and finite.')
    side = gcv_boundary_side(values, selected)
    if side is None:
        raise ValueError('The stored GCV optimum is not at a search boundary.')
    edge = values[0] if side == 'lower' else values[-1]
    extra = [edge * 10.**(i * (-1 if side == 'lower' else 1)) for i in range(1, 5)]
    merged = tuple(sorted(set(values).union(x for x in extra if 1e-24 <= x <= 1e12)))
    if len(merged) == len(values) or len(merged) > 101:
        raise ValueError('GCV expansion limit reached; review candidates manually.')
    return merged
