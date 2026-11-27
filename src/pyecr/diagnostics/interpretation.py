"""Read-only interpretation of captured fields. Never assesses gates or refits."""
from collections.abc import Mapping
from pyecr.fitting.reproducibility import fit_randomness, reproducibility_issue

VERSION = "diagnostic-interpretation/1.0.0"
BOUND_LEGEND = (
    "Numerical contact (bound_hit) tests tight solver-scale proximity: log10 tolerance "
    "1e-5 decade, or physical tolerance 1e-8 × max(|bounds|,1). "
    "Diagnostics v2 checks each boundary independently in optimizer coordinates and reports "
    "CI truncation separately; residual dependence does not fail every parameter. "
    "In legacy policy v1, PARAMETER_AT_BOUND is the broader credibility proximity gate (1% of the supplied "
    "range), not proof of actual bound contact. Diagnostics supplies physical bounds; "
    "independent Batch supplies log10 coordinate bounds. Different coordinates can give "
    "different proximity outcomes. Fit local checks are not the complete credibility gate."
)


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def reproducibility_text(fit):
    if fit is None:
        return "Reproducibility: unavailable; no Fit"
    if not hasattr(fit, 'params'):
        return 'Reproducibility: unavailable; no captured provenance'
    issue = reproducibility_issue(fit)
    if issue:
        return issue
    record = fit_randomness(fit)
    if not record['stochastic']:
        return "Reproducibility: stochastic=false; no random seed used"
    return f"Reproducibility: stochastic=true; actual seed={record['random_seed']} (captured)"


def diagnostic_summary(fit, diagnostics=None):
    if fit is None:
        return "No Fit result."
    lines = ["Optimizer: " + ("successful termination" if field(fit,'exit_status',0) > 0 else "success not reported")]
    if diagnostics is None:
        lines.append("Diagnostics: not run; adequacy and identifiability not assessed")
    else:
        advisor = field(diagnostics, 'reliability_advisor', {})
        lines.append(f"Observed fit adequacy: {field(advisor,'model_adequacy_status','unavailable')} | "
                     f"Parameter identifiability: {field(advisor,'identifiability_status','unavailable')}")
        evaluation = field(diagnostics, 'evaluation')
        if evaluation is not None:
            lines.extend(evaluation_summary(evaluation).splitlines())
        else:
            lines.append("Credibility gates: " + ("passed" if field(diagnostics,'reliable',False) else "warning; inspect Diagnostics"))
    lines.append(reproducibility_text(fit))
    if 'surface_k' in fit.diagnostics or 'dual_surface_k' in fit.diagnostics:
        lines.append("Assumptions: analytic surface control, ideal_step; small perturbation and independent effective processes")
    elif 'fixed_d' in fit.diagnostics:
        lines.append("Assumptions: fixed-D conditional k fit; CI does not propagate fixed-D source uncertainty")
    else:
        lines.append("Assumptions: inspect applied model and boundary; goodness of fit does not establish physical attribution")
    return '\n'.join(lines)


def evaluation_summary(evaluation):
    """One captured record serves GUI, clipboard and exported reports."""
    from pyecr.diagnostics.evaluation import interval_label
    r = field(evaluation, 'residual', {})
    usable = 'usable' if field(evaluation, 'point_estimates_usable', False) else 'not supported'
    return (f"{field(evaluation,'policy_version')}: point estimates {usable}; "
            f"local CI {interval_label(field(evaluation,'interval_status'))}\n"
            f"Residual dependence: {field(r,'dependence')}; n={field(r,'count')}; "
            f"lag-1={field(r,'lag1')}; spacing={field(r,'spacing')}\n"
            f"{field(r,'explanation','')}")
