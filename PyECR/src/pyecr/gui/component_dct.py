"""Prepare spectra, reconstructions and residuals for the component DCT views."""
from dataclasses import replace
import json
from math import sqrt,log10
from pyecr.gui.signed_dct import SignedDctDisplay

NOTICE = (
    'Both curves come from the same fit. Compare their summed reconstruction with the measured trace.\n'
    'The curves share a noise source. Evidence for two processes or for H/O assignments must come from the experiment.\n'
    'The spectra depend on λ, constraints, noise and the terminal reference. '
    'Component spectra use manual λ and have no calibrated confidence intervals. The original fit settings are preserved.'
)


def compact_component_metrics(bundle, view='Comparison', scale='Common signed scale'):
    def rmse(values):return sqrt(sum(v*v for v in values)/len(values)) if values else None
    def fmt(value):return 'N/A' if value is None else f'{value:.5g}'
    if view in {'component_1','component_2'}:
        r=bundle.independent_results[int(view[-1])-1]
        if r is None:return view+': unavailable'
        factor=abs(r.B/r.S_total) if scale=='Common signed scale' else 1.
        return f'{view} RMSE({"Y" if scale=="Common signed scale" else "shape"}) {fmt(abs(factor)*rmse(r.residuals))}'
    whole='Whole → data RMSE(Y) '+fmt(rmse(bundle.overall_result.residuals))
    if view!='Comparison':return whole
    reconstructed=bundle.component_time_reconstruction
    data=tuple(a-b for a,b in zip(bundle.parent_source.signal,reconstructed)) if reconstructed else ()
    return ('Components → data RMSE(Y) '+fmt(rmse(data))+'\nComponents → Fit RMSE(Y) '
            +fmt(rmse(bundle.component_time_residuals))+' · '+whole)


def display_component_job(job,view='Comparison',scale='Common signed scale'):
    if job is None:return SignedDctDisplay(summary='Select Run DCT to calculate component spectra.'),{}
    if job.bundle is None:
        return SignedDctDisplay(summary=f'{job.job_id} | {job.status}',details=job.failure or NOTICE),{}
    b=job.bundle;n=b.parent_source.normalization;overall=b.overall_result
    detail=[NOTICE,f'job={job.job_id} | {job.status}',*job.stale_reasons,
        f'Parent Fit/Diagnostics identity: {b.component_capture.identity.to_dict()}',
        f'profile={n.spec.profile_id}; terminal={n.spec.reference_kind}, sigma_infinity={n.reference_value:.12g}, S_total={b.S_total:.12g} {n.spec.signal_unit}',
        f'B={b.B_j}; B/S={b.common_scale_factors}; delta={b.delta:.17g}',
        'Y_fit = delta + sum((B_j/S_total) * r_j). Delta is retained, not silently subtracted.',
        f'Full observation count={len(b.parent_source.signal)}; display grid / zoom never enters error calculation.',
        f'labels={b.labels}; semantics={b.label_semantics}; reported_to_input={b.component_capture.reported_to_input}',
        *b.state.warnings]
    for i,(r,reason) in enumerate(zip(b.independent_results,b.unavailable_reasons)):
        detail.append(f'component_{i+1}: '+(reason if r is None else
            f'lambda={r.config.grid.regularization_lambda:.17g} Manual; RMSE(r)={sqrt(sum(v*v for v in r.residuals)/len(r.residuals)):.9g}; unit mass={sum(r.mass):.9g}; no independent-noise CI'))
    if b.component_time_residuals:
        detail.append(f'RMSE(Y_fit - delta - component DCT sum)={sqrt(sum(v*v for v in b.component_time_residuals)/len(b.component_time_residuals)):.9g}')
    source=b.parent_source.source_kind.value
    e=b.component_capture.eligibility
    if e is not None:
        detail.extend((f'Component eligibility: {e.policy_version}; can_run={e.can_run}; original reliable={e.original_reliable}; best_reliable={e.original_best_reliable}',
            (f'Original gate r1={e.residual_lag1}; |r1| limit={e.residual_limit}; original gate confidence={e.gate_confidence_level}'
             if e.residual_limit is not None else f'Captured ACF(1)={e.residual_lag1}; multi-lag/time-spacing evidence in parent Diagnostics; local C={e.gate_confidence_level}'),
            *e.original_gate_failures,'With correlated residuals, confidence intervals are uncalibrated. Check the parent fit diagnostics.'))
    detail.append('Parent observations: '+source+'; component curves remain fit_component.')
    detail.append('λ / peak stability view refers to the whole '+('experimental' if source=='experiment' else 'simulated-observation')+' inversion. Component Manual sensitivity is NOT_RUN; no component CI.')
    extras={}
    if view in {'component_1','component_2'}:
        ix=0 if view=='component_1' else 1;r=b.independent_results[ix]
        if r is None:
            return SignedDctDisplay(summary=f'{job.status} | {view} unavailable',details='\n'.join(detail)),{}
        factor=r.B/r.S_total if scale=='Common signed scale' else 1.
        result=SignedDctDisplay(x=r.log10_tau,chi=tuple(factor*v for v in r.chi),time=r.source.time_s,
            observed=tuple(factor*v for v in r.source.signal),reconstruction=tuple(factor*v for v in r.reconstruction),
            residual=tuple(factor*v for v in r.residuals),summary=f'{job.status} | {view} · {scale} · Manual λ={r.config.grid.regularization_lambda:.6g}',details='\n'.join(detail))
    else:
        reconstruction=b.component_time_reconstruction if view=='Comparison' else overall.reconstruction
        # An unavailable component must not make an old combined curve appear.
        residual=tuple(a-c for a,c in zip(b.parent_source.signal,reconstruction))
        result=SignedDctDisplay(x=overall.log10_tau,chi=overall.chi,time=overall.source.time_s,
            observed=b.parent_source.signal,reconstruction=reconstruction,residual=residual,
            summary=f'{job.status} | {view} · full observations={len(overall.source.signal)} · delta={b.delta:.9g}',details='\n'.join(detail))
        if view=='Comparison':
            for i,r in enumerate(b.independent_results):
                if r:extras[f'component_{i+1}']=(r.log10_tau,r.common_chi)
            if all(b.independent_results) and all(r.log10_tau==overall.log10_tau for r in b.independent_results):
                extras['sum']=(overall.log10_tau,tuple(sum(v) for v in zip(*(r.common_chi for r in b.independent_results))))
            extras.update(parent_fit=(overall.source.time_s,b.normalized_parent_fit),delta=(overall.source.time_s,(b.delta,)*len(overall.source.time_s)),
                overall=(overall.source.time_s,overall.reconstruction),model_error=(overall.source.time_s,b.component_time_residuals))
    scores=[(log10(c.result.config.grid.regularization_lambda),c.gcv_score) for c in b.overall_analysis.candidates
        if c.gcv_score is not None and c.result.config.grid.regularization_lambda>0]
    shifts=[(log10(p.candidate.position_s),p.max_position_shift_decades) for p in b.overall_analysis.peak_diagnostics if p.max_position_shift_decades is not None]
    result=replace(result,lambda_x=tuple(x for x,y in scores),lambda_score=tuple(y for x,y in scores),
        peak_x=tuple(x for x,y in shifts),peak_shift=tuple(y for x,y in shifts))
    return result,extras


def complete_component_evidence(job,draft):
    view,_=display_component_job(job)
    return ('Component DCT run record\n'+view.details+'\n\nCurrent draft (not applied):\n'+
        json.dumps(draft,ensure_ascii=False,indent=2,allow_nan=False)+'\n\nStored component job JSON (all arrays):\n'+
        ('null' if job is None else json.dumps(job.to_dict(),ensure_ascii=False,indent=2,allow_nan=False)))
