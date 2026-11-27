"""Signed candidate basins/FWHM; no peak-to-species or process-count assignment."""
import numpy as np
from scipy.signal import find_peaks

from pyecr.contracts import SignedPeakDiagnostic, SignedDctDiagnosticConfig
from pyecr.contracts.common import require
from pyecr.dct.core import CandidateTimescale
from pyecr.dct.peak_geometry import half_height_bounds


def extract_signed_peaks(tau_s, chi, config=SignedDctDiagnosticConfig()):
    tau, chi = np.asarray(tau_s, dtype=float), np.asarray(chi, dtype=float)
    require(tau.ndim == chi.ndim == 1 and len(tau) == len(chi) >= 3 and np.all(np.isfinite(chi))
            and np.all(np.isfinite(tau)) and np.all(tau > 0) and np.all(np.diff(tau) > 0), 'invalid signed peak arrays')
    x = np.log10(tau)
    maximum = float(np.max(np.abs(chi)))
    if maximum == 0:
        return (), ('NO_RESOLVABLE_PEAK',), 'no_resolvable_peak'
    result = []
    for direction, sign in (('positive', 1.), ('negative', -1.)):
        z = sign*chi
        # Padding makes endpoint maxima visible; they remain explicitly bounded
        # candidates, never treated as complete interior peaks.
        indices, properties = find_peaks(np.r_[0., np.maximum(z, 0.), 0.],
                                prominence=config.prominence_fraction*maximum,
                                plateau_size=(None, None))
        indices = indices-1
        for j, index in enumerate(indices):
            if not 0 <= index < len(x) or z[index] <= 0:
                continue
            left, right = int(index), int(index)
            while left > 0 and z[left-1] > 0:
                left -= 1
            while right < len(x)-1 and z[right+1] > 0:
                right += 1
            xl = float(x[left]) if left == 0 else float(np.interp(0., [z[left-1], z[left]], [x[left-1], x[left]]))
            xr = float(x[right]) if right == len(x)-1 else float(np.interp(0., [z[right+1], z[right]], [x[right+1], x[right]]))
            # Split multiple same-sign maxima at the intervening minimum. Do
            # not sum distinct lobes or discard a sign before the inversion.
            if j and indices[j-1] >= left:
                valley = int(indices[j-1]+np.argmin(z[indices[j-1]:index+1]))
                xl = float(x[valley])
            if j+1 < len(indices) and indices[j+1] <= right:
                valley = int(index+np.argmin(z[index:indices[j+1]+1]))
                xr = float(x[valley])
            inside = (x > xl) & (x < xr)
            bx = np.r_[xl, x[inside], xr]
            bz = np.interp(bx, x, z)
            # Exact integral of the piecewise-linear density on the basin.
            area = float(np.trapezoid(bz, bx))
            if area < config.minimum_absolute_area or xr <= xl:
                continue
            peak_ix = int(np.argmin(abs(bx-x[index])))
            # Insert peak sample if it is an endpoint/flat-top representative.
            assert bx[peak_ix] == x[index]
            hl, hr, clipped = half_height_bounds(bx, bz, peak_ix)
            boundary = index <= 1 or index >= len(x)-2
            # A complete FWHM does not imply a complete area. A material tail
            # at an outer tau edge is separately reported, using the declared
            # prominence fraction as a relative-height heuristic (not a CI).
            revised = config.algorithm_version != 'signed-diagnostics/1'
            tail = ((xl == x[0] and bz[0] >= config.prominence_fraction*z[index]) or
                    (xr == x[-1] and bz[-1] >= config.prominence_fraction*z[index]))
            flat = properties['plateau_sizes'][j] > 1
            warnings = tuple(k for condition, k in ((boundary, 'BOUNDARY_PEAK'), (clipped, 'WIDTH_TRUNCATED'),
                (revised and tail, 'BASIN_TRUNCATED'), (revised and flat, 'FLAT_TOP_PEAK')) if condition)
            result.append(SignedPeakDiagnostic(candidate=CandidateTimescale(float(tau[index]), sign*area, hr-hl, int(index)),
                direction=direction, absolute_area=area, absolute_height=float(z[index]), basin_log10_tau=(xl, xr),
                half_height_log10_tau=(hl, hr), boundary=bool(boundary), width_truncated=clipped, warnings=warnings))
    result.sort(key=lambda p: p.candidate.position_s)
    flags = set(w for p in result for w in p.warnings)
    significant_sign = np.sign(chi[np.abs(chi) > .01*maximum])
    sign_changes = int(np.count_nonzero(np.diff(significant_sign)))
    if sign_changes >= 4 or (config.algorithm_version == 'signed-diagnostics/1' and len(result) >= 6):
        flags.add('OSCILLATORY_SPECTRUM')
    elif len(result) >= 6:
        flags.add('MANY_CANDIDATES')
    if not result:
        flags.add('NO_RESOLVABLE_PEAK')
        resolution = 'no_resolvable_peak'
    elif len(result) == 1:
        resolution = 'single_candidate_not_process_count'
    else:
        overlap = any(np.log10(b.candidate.position_s/a.candidate.position_s) < config.resolution_decades
                      or a.half_height_log10_tau[1] >= b.half_height_log10_tau[0]
                      for a, b in zip(result, result[1:]))
        resolution = 'unresolved_overlap' if overlap else 'separated_candidates_conditional'
        if overlap:
            flags.add('UNRESOLVED_NEIGHBOURING_CANDIDATES')
    if 'OSCILLATORY_SPECTRUM' in flags:
        resolution = 'oscillatory_candidates_not_resolved'
    elif result and all(p.boundary or p.width_truncated for p in result):
        resolution = 'boundary_limited_candidates'
    return tuple(result), tuple(sorted(flags)), resolution
