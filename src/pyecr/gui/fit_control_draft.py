"""Prepare draft fit settings for the Analysis panel."""
from pyecr.contracts import ConfidencePolicy, FitControlSpec, TerminationPolicy, parse_confidence_percent
from pyecr.contracts.common import canonical_sha256, number, integer
from pyecr.fitting.solver_strategy import SolverBudget, SolverOptions

VERSION = 'fit-control-ui/1'
MONITORED_FIELDS = ('objective_change_tolerance','parameter_change_tolerance','patience',
                    'max_iterations','max_total_nfev','max_wall_seconds','max_trace_records','ftol','xtol','gtol')


def _field(value,name,*,whole=False):
    try:
        if isinstance(value,bool):raise ValueError
        parsed=int(str(value).strip()) if whole else float(value)
    except (ValueError,TypeError) as exc:raise ValueError(name+': enter a positive '+('integer' if whole else 'finite number')) from exc
    return integer(parsed,name,minimum=1) if whole else number(parsed,name,positive=True)


def parse_draft(values, config):
    """Return a validated intention. Unsupported intentions are never dispatched."""
    confidence = ConfidencePolicy(confidence_level=parse_confidence_percent(values.get('confidence_percent')))
    mode = values.get('mode')
    if mode not in ('Legacy','Monitored'): raise ValueError('Choose Legacy or Monitored')
    if mode == 'Legacy':
        # Reading the existing path, not setting any optimizer/kernel tolerance.
        termination = TerminationPolicy.legacy_for(config) if config is not None else TerminationPolicy()
    else:
        get = lambda name: values.get(name,'')
        wall = None if not str(get('max_wall_seconds')).strip() else _field(get('max_wall_seconds'),'max_wall_seconds')
        budget = SolverBudget(None if config is None else config.fit.max_nfev,
            max_iterations=_field(get('max_iterations'),'max_iterations',whole=True),
            max_total_nfev=_field(get('max_total_nfev'),'max_total_nfev',whole=True),
            max_wall_seconds=wall,max_trace_records=_field(get('max_trace_records'),'max_trace_records',whole=True))
        solver=SolverOptions.from_config(config) if config is not None else SolverOptions()
        termination = TerminationPolicy(mode='monitored',solver_options=solver if solver.strategy=='trf' else SolverOptions(),
            objective_change_tolerance=_field(get('objective_change_tolerance'),'objective_change_tolerance'),
            parameter_change_tolerance=_field(get('parameter_change_tolerance'),'parameter_change_tolerance'),
            patience=_field(get('patience'),'patience',whole=True),budget=budget,
            **{k:(None if k in ('ftol','xtol') and str(get(k)).strip().lower() in ('','none','disabled') else _field(get(k),k)) for k in ('ftol','xtol','gtol')})
    return FitControlSpec(output_ci=confidence,termination=termination)


def unavailable_reason(values, config):
    """Validate a draft without dispatch; capability rejection is explicit."""
    reasons=[]
    if values.get('mode')=='Monitored' and config is not None and SolverOptions.from_config(config).strategy!='trf':
        reasons.append('Monitored supports TRF only; change solver explicitly in Model settings.')
    try: controls=parse_draft(values,config)
    except (ValueError,TypeError) as exc:return None, ' '.join((*reasons,'Invalid draft: '+str(exc)))
    if controls.termination.mode=='monitored' and controls.termination.budget.max_trace_records<2:
        reasons.append('Monitored trace needs at least two records to retain initial and terminal evidence.')
    # R06 selected local intervals have a separate explicit Recompute action.
    # Run freezes its explicit execution policy; C never replaces the 95% gate CI.
    return controls,' '.join(reasons)


def draft_identity(values):
    return canonical_sha256(dict(version=VERSION,values=values))
