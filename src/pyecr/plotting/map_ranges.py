"""Physical/log10 range parsing; no objective, optimizer or GUI dependency."""
from math import isfinite, log10
import numpy as np


def log_range(values, *, coordinate='Physical', nodes=401):
    if len(values)!=2:
        raise ValueError('Enter a minimum and maximum.')
    a,b=(float(v) for v in values)
    if not all(isfinite(v) for v in (a,b)):
        raise ValueError('Range values must be finite.')
    if coordinate=='Physical':
        if min(a,b)<=0:
            raise ValueError('D and k must be positive.')
        a,b=log10(a),log10(b)
    elif coordinate!='log10':
        raise ValueError('Unknown coordinate mode.')
    if a>=b:
        raise ValueError('Minimum must be smaller than maximum.')
    # Also rejects physical underflow/overflow and grids collapsed by float64.
    try:physical=(10.**a,10.**b)
    except OverflowError:raise ValueError('Range cannot be represented in physical units.') from None
    if min(physical)<=0 or not all(isfinite(v) for v in physical):
        raise ValueError('Range cannot be represented in physical units.')
    grid=np.linspace(a,b,nodes)
    if np.any(np.diff(grid)<=0) or np.any(np.diff(np.power(10.,grid))<=0):
        raise ValueError('Range is too narrow for the selected grid.')
    return (a,b)


def within_bounds(values,bounds,*,name='Range'):
    # No clipping or epsilon acceptance: scientific cores enforce these same
    # endpoints. Tolerance here could silently change an intended request.
    if values[0]<bounds[0] or values[1]>bounds[1]:
        raise ValueError(f'{name} must stay within [{10**bounds[0]:.8g}, {10**bounds[1]:.8g}] in physical units (model/label bounds).')
    return values


def auto_range(center,half,bounds):
    if not isfinite(half) or half<=0:
        raise ValueError('Auto half-width must be positive and finite.')
    values=(max(center-half,bounds[0]),min(center+half,bounds[1]))
    return log_range(values,coordinate='log10')

