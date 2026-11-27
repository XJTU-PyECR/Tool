"""Piecewise-linear geometry in log10(tau), independent of inversion/selection."""
import numpy as np


def half_height_bounds(x, y, peak_index):
    """Zero-baseline FWHM within a supplied basin, with interpolated crossings.

    A boundary above half height clips the width; it is not a measured crossing.
    This differs from a width measured at half *prominence*.
    """
    half = .5*y[peak_index]
    li = ri = int(peak_index)
    while li > 0 and y[li-1] > half:
        li -= 1
    while ri < len(x)-1 and y[ri+1] > half:
        ri += 1
    lt = li == 0 and y[0] > half
    rt = ri == len(x)-1 and y[-1] > half
    left = float(x[li]) if li == 0 else float(np.interp(half, [y[li-1], y[li]], [x[li-1], x[li]]))
    right = float(x[ri]) if ri == len(x)-1 else float(np.interp(half, [y[ri+1], y[ri]], [x[ri+1], x[ri]]))
    return left, right, bool(lt or rt)
