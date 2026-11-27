"""Auditable component roles and sign conventions, without fitting or inversion."""
import numpy as np
from pyecr.dct.core import _candidate_timescales


def component_readout(bundle, index):
    capture=bundle.component_capture
    physical=capture.components[index]
    key=physical.component_key
    role={'component_H':'H slot','component_O':'O slot'}.get(key,bundle.labels[index])
    delta=capture.signed_amplitudes[index]
    B=bundle.B_j[index]
    result=bundle.independent_results[index]
    peaks=()
    warnings=()
    if result is not None:
        peaks,warnings=_candidate_timescales(np.asarray(result.tau_s),np.asarray(result.log10_tau),
            np.asarray(result.chi),np.asarray(result.mass),prominence_fraction=.03,minimum_area=.05,
            resolution_limit_decades=.5)
    return dict(key=key,role=role,label=f'C{index+1} / {role}',delta_sigma=delta,B=B,
        factor=B/bundle.S_total,unit=physical.signal_unit,peaks=peaks,warnings=warnings,
        process_key=key.removeprefix('component_') if key in ('component_H','component_O') else key,
        explanation=(f'C{index+1} = parent {role}; conductivity change {delta:+.6g} {physical.signal_unit}. '
            f'Terminal-deviation weight B/S={B/bundle.S_total:+.5g} (B = -change). '
            'A fitted role, not independent chemical identification.'))
