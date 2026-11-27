"""Read-only GUI/report presentation of frozen Balanced/Suppression jobs."""
from math import log10,sqrt
import json
from pyecr.gui.signed_dct import SignedDctDisplay,SIGN_LEGEND,DISPLAY_NOTICE
from pyecr.gui.dct_scan import LambdaScan,ScanCandidate


def convex_scan(analysis):
    if analysis is None:return LambdaScan()
    return LambdaScan(analysis.selected_result.selected_lambda if analysis.selected_result else None,
        tuple(ScanCandidate(c.result.selected_lambda,c.score,c.result.tau_s,c.result.chi,(),c.result.reconstruction,
            sqrt(sum(v*v for v in c.result.residuals)/len(c.result.residuals)) if c.result.residuals else None,c.failure or 'Stored convex spectrum') for c in sorted(analysis.candidates,key=lambda c:c.result.selected_lambda)),
        analysis.source.time_s,analysis.source.signal,True,
        analysis.options.mode+' experimental mode; frozen candidates only. '+analysis.score_name+' is not GCV or a confidence interval.',analysis.score_name,True)


def display_convex_job(job,*,physical=False):
    a=job.analysis
    lines=[f'job_id={job.job_id}; status={job.status}',*job.stale_reasons,*((job.failure,) if job.failure else ()),
        'Whole Signed mode: '+job.request.convex_options.mode,'Options: '+json.dumps(job.request.convex_options.to_dict(),ensure_ascii=False),
        'Frozen scientific source, normalization and algorithm are independent of current draft settings.',*job.warnings,SIGN_LEGEND,DISPLAY_NOTICE,
        'Open edges; exact absolute nodal-mass penalty; 0.25-decade cubic basis is a prior, not physical resolution.',
        'Sensitivity comparisons NOT_RUN. No automatic H/O, unit-mass normalization, nonlinear GCV or calibrated CI.']
    if a is None or a.selected_result is None:
        return SignedDctDisplay(summary=job.status+': no converged '+job.request.convex_options.mode+' spectrum',details='\n'.join(lines))
    r=a.selected_result;n=a.source.normalization
    for p in a.peak_diagnostics:
        lines.append(f'{p.direction}: tau={p.candidate.position_s:.17g}s; signed_area={p.signed_area:.17g}; absolute_area={p.absolute_area:.17g}; width={p.candidate.width_decades:.17g} decade; {p.warnings}; position/area stability not checked')
    lines.append(json.dumps(a.to_dict(),ensure_ascii=False,allow_nan=False))
    scored=sorted((c for c in a.candidates if c.score is not None and c.result.selected_lambda>0),key=lambda c:c.result.selected_lambda)
    return SignedDctDisplay(x=r.log10_tau,chi=r.physical_chi if physical else r.chi,time=a.source.time_s,
        observed=tuple(n.reference_value+n.scale_value*v for v in a.source.signal) if physical else a.source.signal,
        reconstruction=r.physical_reconstruction if physical else r.reconstruction,residual=r.physical_residuals if physical else r.residuals,
        lambda_x=tuple(log10(c.result.selected_lambda) for c in scored),lambda_score=tuple(c.score for c in scored),unit=r.physical_unit if physical else '1',
        summary=f'{a.options.mode} · experimental | lambda={r.selected_lambda:.6g} · {a.options.selection} | sigma_inf={n.reference_value:.9g}; common S={n.scale_value:.9g} {n.spec.signal_unit}; independent_shape=False',details='\n'.join(lines))


def review_convex(analysis,job):
    from pyecr.gui.components.dct_review import DctReview,number,peak_limits
    if analysis.selected_result is None:return DctReview(title='No converged convex spectrum',severity='warning',peak_empty=job.failure or 'All candidates failed; Details retain failures')
    r=analysis.selected_result;g=r.config.grid;auto=analysis.options.selection=='discrepancy'
    peaks=tuple((str(i),number(p.candidate.position_s),'+' if p.direction=='positive' else '−',number(p.signed_area),number(p.candidate.width_decades),peak_limits(p.warnings)) for i,p in enumerate(analysis.peak_diagnostics,1))
    chosen=next((c for c in analysis.candidates if c.result.selected_lambda==r.selected_lambda),None)
    return DctReview(title=analysis.options.mode+' · '+('Historical result' if job.stale_reasons else 'Saved experimental result'),severity='warning',
        scope='Whole Signed, common scale S; positive and negative contributions allowed. Mode changes edit the draft only.',
        settings=f'{len(r.source.signal)} observations · tau {g.tau_min_s:g}–{g.tau_max_s:g} s · {g.n_tau} nodes · lambda {r.selected_lambda:g}',
        resolution='Conditional peak candidates; basis spacing is not physical resolution or CI.',peaks=peaks,
        lambda_summary=('Known-noise auto' if auto else 'Manual')+f' lambda = {r.selected_lambda:g}',
        lambda_note='Discrepancy uses declared independent physical noise; it is not GCV.' if auto else 'Explicit Manual lambda retained; noise was not estimated or invented.',
        lambda_rows=tuple((number(c.result.selected_lambda),number(c.score),'Selected' if c.result.selected_lambda==r.selected_lambda else '',c.failure or 'Converged') for c in analysis.candidates),
        selected_score=(log10(r.selected_lambda),chosen.score) if auto and chosen and chosen.score is not None and r.selected_lambda>0 else (),
        lambda_plot=auto,stability_summary='NOT_RUN — no peak confidence or stability claim.',
        cautions=(*job.stale_reasons,'Experimental mode: weak, wide or slow true negative contributions may be reduced.','Suppression uses stronger absolute-mass shrinkage than Balanced.','No new default or automatic physical interpretation.'))
