"""Existing numerical-contact rule, shared without changing tolerances.

This is not ModelSelection's wider credibility proximity policy.
"""
from math import log10

VERSION = "numerical-bound-contact/1.0.0"


def bound_contact(estimate, bounds, *, is_log):
    if bounds is None:
        return None, None
    lower, upper = bounds
    if is_log and estimate > 0.0 and lower > 0.0 and upper > 0.0:
        coordinate = log10(estimate)
        lower_distance = abs(coordinate - log10(lower))
        upper_distance = abs(coordinate - log10(upper))
        tolerance = 1.0e-5
    else:
        scale = max(abs(lower), abs(upper), 1.0)
        tolerance = scale * 1.0e-8
        lower_distance = abs(estimate - lower)
        upper_distance = abs(estimate - upper)
    if lower_distance <= tolerance:
        return True, "lower"
    if upper_distance <= tolerance:
        return True, "upper"
    return False, None


def contact_reason(estimate, bounds, *, is_log):
    if bounds is None:
        return "Numerical contact unavailable: no configured parameter bound"
    lower, upper = bounds
    if is_log and min(estimate, lower, upper) > 0:
        return (f"Numerical contact: log10 distance to lower/upper "
                f"{abs(log10(estimate)-log10(lower)):.6g}/{abs(log10(estimate)-log10(upper)):.6g} decade; "
                "tolerance=1e-5 decade")
    return (f"Numerical contact: physical distance to lower/upper "
            f"{abs(estimate-lower):.6g}/{abs(estimate-upper):.6g}; "
            f"tolerance={max(abs(lower),abs(upper),1.0)*1e-8:.6g}")
