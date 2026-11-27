"""Conditional peak/transport references, never a D/k estimator or assignment.

Lengths are half of the declared full dimensions. Rectangular mode rates add;
their masses multiply. A constant-tau locus assumes a fundamental mode and
illustrates non-uniqueness rather than estimating two coefficients from a peak.
"""
from itertools import product
from math import pi, prod, log10, isfinite
import numpy as np
from pyecr.models.roots import solve_alpha_roots


def fundamental(geometry, D, k=None):
    geometry.validate()
    half = np.asarray([L/2 for L,on in zip(geometry.full_dims_cm,geometry.active_dimensions) if on])
    if D is None:
        return 1/(k*np.sum(1/half)), 0., 1.
    if k is None:
        return 1/(D*np.sum((pi/(2*half))**2)), 1., 0.
    bi = k*half/D
    roots = np.asarray([solve_alpha_roots(float(b),1)[0][0] for b in bi])
    rates = D*(roots/half)**2
    # d ln(alpha^2) / d ln(Bi), using alpha*tan(alpha)=Bi.
    alpha_elasticity = 2*bi/(roots*(np.tan(roots)+roots/np.cos(roots)**2))
    sensitivity_k = float(np.clip(rates @ alpha_elasticity / np.sum(rates),0.,1.))
    return float(1/np.sum(rates)), 1-sensitivity_k, sensitivity_k


def constant_tau_locus(geometry, tau_s, bi_grid=None):
    geometry.validate()
    if not isfinite(tau_s) or tau_s <= 0:
        raise ValueError('Positive finite peak time required')
    half = np.asarray([L/2 for L,on in zip(geometry.full_dims_cm,geometry.active_dimensions) if on])
    reference = float(min(half))
    grid = np.geomspace(1e-4,1e4,121) if bi_grid is None else np.asarray(bi_grid,dtype=float)
    result = []
    for bi in grid:
        roots = np.asarray([solve_alpha_roots(float(bi*a/reference),1)[0][0] for a in half])
        D = float(1/(tau_s*np.sum((roots/half)**2)))
        k = float(D*bi/reference)
        result.append(dict(Bi_reference=float(bi),D_cm2_s=D,k_cm_s=k,log10_D=log10(D),log10_k=log10(k)))
    return result


def transport_guide(summary, peaks, *, per_axis=4, time_s=(), process_keys=None):
    """Numerical references are available even for a weak parent, labelled so."""
    if isinstance(per_axis,bool) or not isinstance(per_axis,int) or not 1 <= per_axis <= 8:
        raise ValueError('Modes per axis must be an integer in [1,8]')
    if process_keys is not None and not set(process_keys).issubset({p.process for p in summary.processes}):
        raise ValueError('Requested process is not a declared parent role')
    half = [(axis,L/2) for axis,L,on in zip('xyz',summary.geometry.full_dims_cm,summary.geometry.active_dimensions) if on]
    ideal = summary.parameter_evidence['boundary']['mode']=='ideal_step'
    result = dict(version='conditional-dct-transport-guide/1', parent_reliability=summary.parent_reliability,
        reliability_reasons=list(summary.reliability_reasons), model_id=summary.model_id,
        geometry=summary.geometry.to_dict(), processes=[], peak_comparisons=[], peak_support=[],
        cautions=['Conditional on the declared model and geometry. Nearest mode is a time-scale comparison, not species assignment or an independent validation.',
                  'One peak does not uniquely determine D and k. The locus assumes this peak is the fundamental material mode.',
                  'One physical process has several diffusion modes; Cartesian mode rates add in more than one active dimension.',
                  'Peak basin area and discrete modal mass have different meanings; no amplitude/species assignment is made.'])
    if summary.parent_reliability not in ('supported','not_applicable'):
        result['cautions'].insert(0,'Parent precision/reliability is weak or pending. Values below are exploratory references; original Apply/CI gates remain in force.')
    if not ideal:
        result['cautions'].insert(0,'Nonideal input boundary: material modal peak matching is unavailable without treating the input response. No material deconvolution is performed.')
    times=np.asarray(time_s,dtype=float)
    positive=times[times>0]
    if len(positive):
        for i,peak in enumerate(peaks):
            notes=[]
            if peak<positive.min():notes.append('Faster than the first positive observation; early response is weakly sampled')
            if peak>times.max():notes.append('Slower than the observation window; terminal/baseline and slow modes can trade off')
            result['peak_support'].append(dict(peak=i+1,tau_s=peak,
                samples_in_0_1_to_3_tau=int(np.count_nonzero((times>=.1*peak)&(times<=3*peak))),
                note='; '.join(notes) or 'Within sampled range; resolution unproven'))
    for p in summary.processes:
        if process_keys is not None and p.process not in process_keys:continue
        D,k = p.D_cm2_s,p.k_cm_s
        axes = [dict(axis=a.axis,Bi=a.value,regime=a.regime) for a in summary.axes if a.process==p.process]
        entry = dict(process=p.process,label=p.label,D_cm2_s=D,k_cm_s=k,axes=axes,modes=[],tau1_s=None,
                     tau_surface_s=None,tau_diffusion_s=None,sensitivity_D=None,sensitivity_k=None,advice=[])
        if k is not None: entry['tau_surface_s']=float(1/(k*sum(1/a for _,a in half)))
        if D is not None: entry['tau_diffusion_s']=float(1/(D*sum((pi/(2*a))**2 for _,a in half)))
        missing = (p.D_name is not None and D is None) or (p.k_name is not None and k is None)
        if missing:
            entry['advice'].append('A coefficient required by this model is missing or invalid; no missing D or k is inferred.')
        elif not ideal:
            entry['advice'].append('Compare Bi and model parameters only; observed DCT peaks also contain the input response.')
        else:
            tau,sD,sk=fundamental(summary.geometry,D,k)
            entry.update(tau1_s=tau,sensitivity_D=sD,sensitivity_k=sk)
            axis_modes=[]
            for _,a in half:
                if D is None:
                    axis_modes.append([(k/a,1.)])
                else:
                    bi=None if k is None else k*a/D
                    roots=np.asarray([(i+.5)*pi for i in range(per_axis)]) if bi is None else solve_alpha_roots(float(bi),per_axis)[0]
                    coefficients=2/roots**2 if bi is None else 2/(roots**2*(1+(roots/bi)**2+1/bi))
                    axis_modes.append(list(zip(D*(roots/a)**2,coefficients)))
            for indices in product(*(range(len(m)) for m in axis_modes)):
                rate=sum(m[i][0] for m,i in zip(axis_modes,indices))
                mass=prod(m[i][1] for m,i in zip(axis_modes,indices))
                entry['modes'].append(dict(indices=[i+1 for i in indices],tau_s=float(1/rate),mass=float(mass)))
            entry['modes'].sort(key=lambda m:m['tau_s'],reverse=True)
            if D is None:
                entry['advice'].append('Surface-only model: tau=V/(k*S_exposed). D and Bi are not identified by this model.')
            elif k is None:
                entry['advice'].append('Instant-surface model: tau follows D. Finite k and Bi cannot be inferred from this fit.')
            elif sk < .1:
                entry['advice'].append('Fundamental time is insensitive to k (<10% log sensitivity). A good curve fit can leave k poorly determined; use a thinner sample or additional geometry, and preserve early-time observations.')
            elif sD < .1:
                entry['advice'].append('Fundamental time is insensitive to D (<10% log sensitivity). A single dominant peak mostly constrains surface exchange; consider a thicker sample or additional geometry.')
            else:
                entry['advice'].append('Both D and k affect the fundamental time. Joint uncertainty and additional geometry/early-time information are still needed to separate them.')
        result['processes'].append(entry)
        for i,peak in enumerate(peaks):
            if not entry['modes']:continue
            nearest=min(entry['modes'],key=lambda m:abs(log10(peak/m['tau_s'])))
            near=sum(abs(log10(peak/m['tau_s']))<=.3 for m in entry['modes'])
            result['peak_comparisons'].append(dict(peak=i+1,tau_peak_s=peak,process=p.label,
                mode=nearest['indices'],tau_mode_s=nearest['tau_s'],ratio=peak/nearest['tau_s'],
                delta_decades=abs(log10(peak/nearest['tau_s'])),modes_within_0_3_decade=near))
    return result
