"""Conditional rectangular modal references, extracted from production kernels.

These discrete remaining-fraction masses are not DCT density heights. No peak
matching or inverse Bi calculation occurs here. A bounded Cartesian prefix is
explicitly truncated; rates add and coefficients multiply across active axes.
"""
from itertools import product
import numpy as np
from pyecr.contracts.common import require, integer, time_vector, canonical_sha256
from pyecr.models.factory import ModelFactory

VERSION = 'production-rectangular-modes/1'
MAX_FORWARD_TERMS_TIMES_ROWS = 8_000_000  # explicit review allocation budget; never crop observations


def modal_reference(summary, model, solver, boundary, time_s, *, per_axis=4):
    integer(per_axis, 'per_axis', minimum=1)
    require(per_axis <= 8, 'Modal prefix budget exceeds 8 per active axis')
    times = np.asarray(time_vector(time_s))
    require(model.model_id.value == summary.model_id and model.geometry == summary.geometry,
            'Modal model/transport geometry conflict')
    require(canonical_sha256(boundary.to_dict()) == canonical_sha256(summary.parameter_evidence['boundary']), 'Modal boundary/source conflict')
    solver.validate(); model.validate(); boundary.validate()
    result = dict(algorithm_version=VERSION, status='unsupported', reasons=[],
                  boundary=boundary.to_dict(), geometry=summary.geometry.to_dict(),
                  per_axis_budget=per_axis, time_s=times.tolist(), processes=[],
                  response_scope='observed_effective_response', deconvolved=False,
                  interpretation='Conditional material remaining-fraction modes, not peak assignments, DCT chi or fitted evidence.')
    if boundary.mode != 'ideal_step':
        result['reasons'] = ['Unsupported modal material comparison: nonideal boundary. Observed effective response; no deconvolution.']
        return result
    if any(p.D_cm2_s is None or (p.k_name is not None and p.k_cm_s is None) for p in summary.processes):
        result['reasons'] = ['Unsupported diffusion modal comparison: required D or k is not supplied by this source; no missing coefficient is inferred from peak ratios.']
        return result
    if len(times)*solver.max_terms > MAX_FORWARD_TERMS_TIMES_ROWS:
        result['reasons']=['Unsupported modal forward allocation: rows * production max_terms exceeds 8000000. Transport context remains readable; no observations were cropped.']
        return result
    kernel = ModelFactory().create(model, boundary=boundary).kernel
    warnings=[]
    for p in summary.processes:
        args = dict(diffusion_coefficient_cm2_s=p.D_cm2_s, term_tol=solver.relative_tolerance, max_terms=solver.max_terms)
        if p.k_name is not None: args['surface_exchange_coefficient_cm_s'] = p.k_cm_s
        full = kernel.evaluate(times, **args)
        # The production selector returns one term at exactly t=0. A positive
        # near-zero probe requests the bounded coefficient prefix without
        # altering production tolerances or replacing any observation.
        probe_time=float(np.finfo(float).tiny)
        probe = kernel.evaluate(np.array([probe_time]), **{**args, 'max_terms': min(per_axis, solver.max_terms)})
        axis_data = []
        for axis, out in zip('xyz', probe.axis_results, strict=True):
            if out is None: continue
            d = out.diagnostics
            roots = d.get('roots', d.get('alphas'))
            axis_data.append(dict(axis=axis, roots=np.asarray(roots).tolist(),
                coefficients=np.asarray(d['coefficients']).tolist(), rates_s_inv=np.asarray(d['rates_s_inv']).tolist(),
                Bi=d.get('bi'), n_terms_used=out.n_terms_used, coefficient_sum=out.coefficient_sum,
                coefficient_sum_error=d['coefficient_sum_error'], converged=bool(d['converged']),
                root_residual_max_abs=d.get('root_residual_max_abs'),
                asymptotic_roots_used=bool(d.get('asymptotic_roots_used', False))))
        modes = []
        for indices in product(*(range(len(a['rates_s_inv'])) for a in axis_data)):
            rate = sum(a['rates_s_inv'][i] for a,i in zip(axis_data,indices,strict=True))
            weight = float(np.prod([a['coefficients'][i] for a,i in zip(axis_data,indices,strict=True)]))
            modes.append(dict(indices=list(indices), rate_s_inv=rate, tau_s=1/rate,
                              log10_tau=float(np.log10(1/rate)), mass=weight))
        modes.sort(key=lambda m:m['rate_s_inv'])
        partial = np.exp(-np.outer(times,[m['rate_s_inv'] for m in modes])) @ np.array([m['mass'] for m in modes])
        result['processes'].append(dict(process=p.process, label=p.label, D_cm2_s=p.D_cm2_s, k_cm_s=p.k_cm_s,
            axes=axis_data, modes=modes, total_retained_mass=sum(m['mass'] for m in modes),
            production_remaining=np.asarray(full.F).tolist(), truncated_remaining=partial.tolist(),
            truncation_difference=(np.asarray(full.F)-partial).tolist(),
            production_axis_terms=[None if a is None else a.n_terms_used for a in full.axis_results],
            zero_time_note='Production F(0)=1 is imposed by the existing kernel. Truncated masses are NOT renormalized.',
            production_term_tol=solver.relative_tolerance, production_max_terms=solver.max_terms))
        result['processes'][-1]['prefix_probe_time_s']=probe_time
        result['processes'][-1]['production_axis_evidence']=[None if a is None else dict(
            n_terms_used=a.n_terms_used,converged=bool(a.diagnostics['converged']),
            coefficient_sum=a.coefficient_sum,coefficient_sum_error=float(a.diagnostics['coefficient_sum_error']),
            truncation_error=float(a.diagnostics['truncation_error']),
            asymptotic_roots_used=bool(a.diagnostics.get('asymptotic_roots_used',False))) for a in full.axis_results]
        if any(a is not None and not a.diagnostics['converged'] for a in full.axis_results):
            warnings.append(p.process+': production forward did not reach its declared series tolerance within max_terms; inspect recorded truncation evidence.')
    result['status']='available'
    result['reasons']=['Conditional ideal rectangular reference. Finite truncation and fixed physical inputs; no species or CI claim.',*warnings]
    return result
