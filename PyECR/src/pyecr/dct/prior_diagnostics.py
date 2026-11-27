"""Read-only spectrum diagnostics, conditional on the recorded source/prior.

No inverse call, peak smoothing, species assignment, CI or current publication.
"""
import numpy as np
from pyecr.contracts import SignedDctDiagnosticConfig
from pyecr.contracts.common import require
from pyecr.dct.signed_peaks import extract_signed_peaks
from pyecr.diagnostics.evaluation import residual_evidence


def diagnose_prior_spectrum(result,*,buffer_decades=.5,search_report=None,peak_policy=None):
    require(buffer_decades>0,'diagnostic edge width must be positive')
    if not result.state.ready:
        return dict(version='signed-prior-diagnostics/1',status='FAIL',reasons=result.state.blocking_reasons,
            mass_metrics=None,physical_reliability='NOT_ESTABLISHED')
    p=peak_policy or SignedDctDiagnosticConfig()
    tau=np.asarray(result.tau_s);x=np.asarray(result.log10_tau);chi=np.asarray(result.chi);q=np.asarray(result.quadrature_weights)
    t=np.asarray(result.source.time_s);weights=np.sqrt(result.config.observation_weights or np.ones(len(t)))
    pos=float(q@np.maximum(chi,0));neg=float(q@np.maximum(-chi,0));absolute=pos+neg;net=pos-neg
    edge=(x<x[0]+buffer_decades)|(x>x[-1]-buffer_decades)
    positive_times=t[t>0];first=float(positive_times[0]) if len(positive_times) else None
    counts=np.array([np.count_nonzero((t>=.1*ta)&(t<=3*ta)) for ta in tau])
    weak=(counts<3)|(tau>t[-1])|((tau<first) if first is not None else np.ones(len(tau),bool))
    peaks,flags,resolution=extract_signed_peaks(result.tau_s,result.chi,p)
    residual=residual_evidence(weights*np.asarray(result.residuals),result.source.time_s)
    boundary=None;selection_state='manual'
    if search_report is not None:
        selection_state=search_report.termination
        values=[trial.result.config.grid.regularization_lambda for trial in search_report.trials]
        selected=search_report.selected_lambda
        boundary=selected is not None and selected in {min(values),max(values)}
    near_zero=abs(net)<=64*np.finfo(float).eps*max(absolute,1.)
    weak_edge=float(np.sum(q[weak&edge]*np.abs(chi[weak&edge])))
    return dict(version='signed-prior-diagnostics/1',status='COMPUTED_CONDITIONAL',
        mass_metrics=dict(positive=pos,negative=neg,net=net,absolute=absolute,
            edge_absolute=float(np.sum(q[edge]*np.abs(chi[edge]))),weak_support_absolute=float(np.sum(q[weak]*np.abs(chi[weak]))),
            weak_support_negative=float(np.sum(q[weak]*np.maximum(-chi[weak],0))),
            cancellation_ratio=None if near_zero else absolute/abs(net),
            cancellation_fraction=1-abs(net)/absolute if absolute>0 else 0.,near_zero_net=bool(near_zero)),
        weak_support=dict(rule='heuristic v1: tau<first positive time, tau>last time, or <3 samples in 0.1tau..3tau',
            first_positive_time_s=first,last_time_s=float(t[-1]),counts=counts.tolist(),mask=weak.tolist()),
        boundary_unclosed=bool(weak_edge>=p.minimum_absolute_area or 'BASIN_TRUNCATED' in flags or 'BOUNDARY_PEAK' in flags),
        edge_width_decades=buffer_decades,selection_boundary=boundary,selection_state=selection_state,
        peaks=[peak.to_dict() for peak in peaks],peak_flags=flags,resolution_status=resolution,
        residual_dependence=residual.dependence,residual_explanation=residual.explanation,
        df_semantics=result.solve_diagnostics.df_semantics if result.solve_diagnostics.schema_version==3 else 'fixed_linear',
        statements=('Masses are conditional on terminal, S, kernel, tau range and prior; signed total is not normalized to one.',
            'Weak support is a sampling heuristic, not a calibrated confidence limit. Boundary tails are not forcibly removed.',
            'Frozen-UP resolution/df is conditional only. FWHM is not CI. Whole peaks have no automatic H/O labels.'),
        physical_reliability='NOT_ESTABLISHED')
