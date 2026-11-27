"""One frozen presentation shared by GUI and experimental Reports."""
from dataclasses import dataclass
from math import sqrt,log10
import json
from pyecr.project.prior_archive import SignedPriorArchive
from pyecr.gui.signed_dct import SignedDctDisplay,SIGN_LEGEND
from pyecr.gui.dct_scan import LambdaScan,ScanCandidate


def prior_scan(archive):
    report=archive.report;rows=[]
    for trial in report.trials:
        r=trial.result;ready=r.state.ready
        rows.append(ScanCandidate(r.config.grid.regularization_lambda,trial.score,
            r.tau_s if ready else (),r.chi if ready else (),(),r.reconstruction if ready else (),
            sqrt(sum(v*v for v in r.residuals)/len(r.residuals)) if ready else None,
            trial.failure or r.prior_evidence.status))
    label={'gcv':'GCV','discrepancy':'Discrepancy distance','numerical_discrepancy':'Numerical tolerance distance','blocked_cv':'Held-out physical MSE'}[report.policy.strategy]
    return LambdaScan(report.selected_lambda,tuple(rows),report.source.time_s,report.source.signal,True,
        'Archived experimental spectrum. Score: '+label+'. UP degrees of freedom assume fixed weights and cannot be used for nonlinear GCV.',label,False)


def prior_display(archive,*,physical=False):
    report=archive.report;r=report.selected_result;cfg=report.config
    details=[archive.to_dict()['scope'],f'algorithm={cfg.algorithm_version}; solver={cfg.solver}; prior={cfg.prior_spec.to_dict()}',
        f'selection={report.policy.to_dict()}; termination={report.termination}; budget used: evaluations={report.evaluations}, subproblems={report.subproblems}, seconds={report.seconds}',
        f'noise SD physical declaration={report.physical_noise_sd!r}',f'source={report.source.identity.to_dict()}',
        f'provenance={archive.provenance}',SIGN_LEGEND,'Whole Signed uses common S, independent_shape=False; no unit-mass renormalization.',
        'Trusted time support and boundary diagnostics are conditional heuristics; not CI or physical species identification.',
        'Negative mass diagnostic is a positive magnitude; signed negative contribution is minus that magnitude. Peak signed areas and m keep their negative sign.',
        json.dumps(archive.diagnostics,ensure_ascii=False,sort_keys=True)]
    for trial in report.trials:
        details.append(f'lambda={trial.result.config.grid.regularization_lambda:.17g}; score={trial.score}; '+
            f'status={trial.result.prior_evidence.status}; failure={trial.failure}; seconds={trial.seconds}; subproblems={trial.subproblems}')
    if r is None:return SignedDctDisplay(summary='Experimental FAIL: no selected spectrum',details='\n'.join(details))
    n=r.source.normalization;observed=tuple(n.reference_value+n.scale_value*y for y in r.source.signal) if physical else r.source.signal
    return SignedDctDisplay(x=r.log10_tau,chi=r.physical_chi if physical else r.chi,time=r.source.time_s,observed=observed,
        reconstruction=r.physical_reconstruction if physical else r.reconstruction,residual=r.physical_residuals if physical else r.residuals,
        lambda_x=tuple(log10(t.result.config.grid.regularization_lambda) for t in report.trials if t.score is not None),
        lambda_score=tuple(t.score for t in report.trials if t.score is not None),
        unit=r.physical_unit if physical else '1',summary=f'Archived experimental spectrum | {r.prior_evidence.status} | lambda={r.selected_lambda:.6g} | '+
            f'sigma_inf={n.reference_value:.9g}; common S={n.scale_value:.9g} {n.spec.signal_unit}; independent_shape=False',details='\n'.join(details))
