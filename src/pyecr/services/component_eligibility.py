"""Read-only U01 assessment. Reuses the original local-95% gate calculation.

Only a canonical autocorrelation gate and its proven aggregate consequences can
be warnings. Unknown, independently failing or inconsistent evidence blocks.
No forward/refit, alternate CI method or Diagnostics publication occurs here.
"""
from math import isfinite
from pyecr.contracts.eligibility import FeatureEligibility, EligibilityReason, AUTOCORRELATION
from pyecr.diagnostics.model_selection import ParameterBound, select_model
from pyecr.models.ids import ModelID
from pyecr.preprocessing.quality import evaluate_fit_quality
from pyecr.services.diagnostics_service import (DiagnosticsPresentation, _local_uncertainty,
    _parameter_rows, _reliability_advisor, _bi_presentation, _correlation_presentation,
    _jacobian_presentation, _multistart_presentation, _residual_summary)

SHORT_WARNING='Residual correlation warning. Component DCT is available.'


def evaluate_component_eligibility(dataset, config, fit, diag, identity):
    """Evaluate frozen, typed parent evidence; caller retains provenance checks."""
    if isinstance(diag, DiagnosticsPresentation) and diag.evaluation is not None:
        return _evaluate_v2(dataset, config, fit, diag, identity)
    reasons=[];rho=None
    def add(code,origin,message=None,*,warning=False,causes=()):
        reasons.append(EligibilityReason(code=code,origin=origin,message=message or code,
            disposition='warning' if warning else 'blocking',causes=causes))
    def result():
        blockers=tuple(dict.fromkeys(r.message for r in reasons if r.disposition=='blocking'))
        return FeatureEligibility(source=identity,can_run=not blockers,
            warnings=tuple(dict.fromkeys(r.message for r in reasons if r.disposition=='warning')),
            blocking_reasons=blockers,reasons=tuple(reasons),original_reliable=bool(getattr(diag,'reliable',False)),
            original_best_reliable=getattr(diag,'best_reliable',None),
            original_gate_failures=tuple(getattr(diag,'gate_failures',())),residual_lag1=rho)
    try:
        if not isinstance(diag,DiagnosticsPresentation):
            add('PARENT_DIAGNOSTICS_MISSING','diagnostics','PARENT_DIAGNOSTICS_MISSING: explicitly run current basic Diagnostics')
            return result()
        from .component_source import _mode_reasons
        for message in _mode_reasons(config):
            add(message.partition(':')[0],'component.support_matrix',message)
        if diag.model_id!=config.model_id.value or diag.time_s!=tuple(dataset.processed_time) or diag.residuals!=tuple(fit.residuals):
            add('PARENT_IDENTITY_MISMATCH','diagnostics','Diagnostics model/observation/residual identity conflict')
        for message in evaluate_fit_quality(dataset,config).blocking_reasons:
            add(message.partition(':')[0],'preprocessing.quality',message)
        if fit.exit_status<=0 or not diag.optimizer_success:
            add('PARENT_OPTIMIZER_INCOMPLETE','optimizer')
        uncertainty,why=_local_uncertainty(fit)
        if uncertainty is None:add('PARENT_UNCERTAINTY_UNAVAILABLE','local_uncertainty',f'PARENT_UNCERTAINTY_UNAVAILABLE: {why}')
        elif uncertainty.rank<len(fit.params):add('PARENT_JACOBIAN_RANK_DEFICIENT','local_uncertainty.rank')
        bounds={n:ParameterBound(*b,space='physical') for n,b in config.parameter_bounds.items() if n in fit.params}
        if config.model_id is ModelID.M8_DUAL_K:
            bounds={n:ParameterBound(*b,space='physical') for n,b in fit.diagnostics['dual_surface_k']['parameter_bounds'].items()}
        selection=select_model({config.model_id.value:fit},
            uncertainties=None if uncertainty is None else {config.model_id.value:uncertainty},
            parameter_bounds={config.model_id.value:bounds})
        core=selection.assessments[config.model_id.value]
        rho=float(core.residual_lag1_autocorrelation)
        if not isfinite(rho):rho=None
        expected_gates=list(core.gate_failures)
        if uncertainty is not None and uncertainty.rank<len(fit.params):
            expected_gates.append(f'JACOBIAN_RANK_DEFICIENT: rank={uncertainty.rank}, parameters={len(fit.params)}.')
        expected_gates=tuple(dict.fromkeys(expected_gates))
        gates_match=diag.gate_failures==expected_gates
        if not gates_match:add('PARENT_GATE_ORIGIN_UNRESOLVED','diagnostics.gate_failures','PARENT_GATE_ORIGIN_UNRESOLVED: Diagnostics and recomputed original gate differ')
        only_auto=(len(core.gate_failures)==1 and core.gate_failures[0].partition(':')[0]==AUTOCORRELATION
                   and rho is not None and abs(rho)>.30 and gates_match)
        for origin,messages in [('core.gate_failures',core.gate_failures),('diagnostics.gate_failures',diag.gate_failures)]:
            for message in messages:
                code=message.partition(':')[0]
                add(code,origin,message,warning=code==AUTOCORRELATION and gates_match and rho is not None and abs(rho)>.30)
        if diag.reliable!=core.reliable or diag.best_reliable!=selection.best_reliable:
            add('PARENT_FIT_UNRELIABLE','diagnostics.aggregate','PARENT_FIT_UNRELIABLE: reliability summary has no matching original gate origin')
        elif not diag.reliable or diag.best_reliable!=config.model_id.value:
            add('PARENT_FIT_UNRELIABLE','diagnostics.aggregate','PARENT_FIT_UNRELIABLE: current Diagnostics does not support this model',
                warning=only_auto,causes=tuple(m.partition(':')[0] for m in core.gate_failures))
        if not core.reliable:
            add('PARENT_CORE_GATE_REJECTED','core.aggregate',warning=only_auto,
                causes=tuple(m.partition(':')[0] for m in core.gate_failures))
        canonical=_parameter_rows(config,fit,uncertainty,core,why)
        expected_correlation=_correlation_presentation(fit,uncertainty,why)
        expected_jacobian=_jacobian_presentation(fit,uncertainty,why)
        expected_residuals=_residual_summary(fit.residuals)
        if (diag.parameter_uncertainty.rows!=diag.parameters or diag.correlation!=expected_correlation
                or diag.jacobian!=expected_jacobian or diag.residual_summary!=expected_residuals
                or diag.optimizer_success!=(fit.exit_status>0)):
            add('PARENT_EVIDENCE_ORIGIN_UNRESOLVED','diagnostics','PARENT_EVIDENCE_ORIGIN_UNRESOLVED: stored CI/rank/correlation/residual evidence differs from its parent')
        if len(diag.parameters)!=len(canonical) or len({r.name for r in diag.parameters})!=len(canonical):
            add('PARENT_PARAMETER_IDENTITY_MISMATCH','diagnostics.parameters')
        rows={p.name:p for p in diag.parameters}
        for expected in canonical:
            actual=rows.get(expected.name)
            origin='diagnostics.parameters.'+expected.name
            if actual!=expected:
                add('PARENT_PARAMETER_ORIGIN_UNRESOLVED',origin,f'PARENT_PARAMETER_ORIGIN_UNRESOLVED: {expected.name}: stored row differs from original 95% evidence')
            if not expected.reliable:
                # This EXACT canonical row has no CI/bound/rank failure. Its sole
                # reason was attached by _parameter_rows from core.reliable.
                derived=only_auto and actual==expected and expected.reliability_reason=='model credibility Gate failed'
                add('PARENT_PARAMETER_UNRELIABLE',origin,f'PARENT_PARAMETER_UNRELIABLE: {expected.name}: {expected.reliability_reason}',
                    warning=derived,causes=(AUTOCORRELATION,) if derived else ())
        # Independently recompute the cheap advisor (no sensitivity/forward/map).
        # Stored map-derived concerns are also retained; poor is never inferred
        # to mean autocorrelation from prose, because the current rule is independent.
        advisor=_reliability_advisor(fit,_bi_presentation(config,fit),canonical,
            expected_correlation,expected_jacobian,
            _multistart_presentation(fit.candidates),expected_residuals,
            error_map_result=None,dual_error_map_result=None)
        for origin,a in [('diagnostics.advisor',diag.reliability_advisor),('core.advisor',advisor)]:
            if a.identifiability_status not in {'supported','caution'}:
                add('PARENT_IDENTIFIABILITY',origin,f'PARENT_IDENTIFIABILITY: {a.identifiability_status}')
            if a.model_adequacy_status not in {'supported','caution'}:
                add('PARENT_MODEL_ADEQUACY',origin,f'PARENT_MODEL_ADEQUACY: {a.model_adequacy_status}')
    except (ValueError,TypeError,KeyError,AttributeError,RuntimeError) as exc:
        add('PARENT_ELIGIBILITY_UNRESOLVED','eligibility',f'PARENT_ELIGIBILITY_UNRESOLVED: {type(exc).__name__}: {exc}')
    return result()


def _evaluate_v2(dataset, config, fit, diag, identity):
    from dataclasses import fields
    from pyecr.contracts.eligibility import COMPONENT_POLICY_V2
    from .diagnostics_service import canonical_diagnostics
    from .component_source import _mode_reasons
    reasons = []
    def add(code, origin, message, warning=False):
        reasons.append(EligibilityReason(code=code, origin=origin, message=message,
            disposition='warning' if warning else 'blocking'))
    try:
        for message in (*_mode_reasons(config), *evaluate_fit_quality(dataset, config).blocking_reasons):
            add(message.partition(':')[0], 'component.support_matrix', message)
        expected = canonical_diagnostics(dataset, config, fit, diag)
        for field in fields(diag):
            if field.name != 'sensitivity' and getattr(diag, field.name) != getattr(expected, field.name):
                add('PARENT_EVIDENCE_ORIGIN_UNRESOLVED', 'diagnostics.'+field.name,
                    'PARENT_EVIDENCE_ORIGIN_UNRESOLVED: captured v2 '+field.name+' differs from source evidence')
        e = expected.evaluation
        for message in e.blocking_reasons:
            add(message.partition(':')[0], 'diagnostics.evaluation', message)
        for message in e.warnings:
            add(message.partition(':')[0], 'diagnostics.evaluation', message, True)
        for row in expected.parameters:
            if not row.reliable:
                add('PARENT_PARAMETER_UNRELIABLE', 'diagnostics.parameters.'+row.name,
                    f'PARENT_PARAMETER_UNRELIABLE: {row.name}: {row.reliability_reason}')
        if not e.point_estimates_usable or e.identifiability_status in {'weak','unavailable'}:
            add('PARENT_INFERENCE_UNSUPPORTED', 'diagnostics.evaluation',
                'PARENT_INFERENCE_UNSUPPORTED: point estimate or parameter precision does not support components')
        if e.interval_status not in {'conditional_iid', 'assumptions_unverified'}:
            add('PARENT_INTERVAL_UNAVAILABLE', 'diagnostics.evaluation',
                'PARENT_INTERVAL_UNAVAILABLE: '+e.interval_status)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        add('PARENT_ELIGIBILITY_UNRESOLVED', 'eligibility', 'PARENT_ELIGIBILITY_UNRESOLVED: '+str(exc))
    blockers = tuple(dict.fromkeys(r.message for r in reasons if r.disposition == 'blocking'))
    return FeatureEligibility(source=identity, can_run=not blockers,
        warnings=tuple(dict.fromkeys(r.message for r in reasons if r.disposition=='warning')),
        blocking_reasons=blockers, reasons=tuple(reasons), original_reliable=diag.reliable,
        original_best_reliable=diag.best_reliable, original_gate_failures=diag.gate_failures,
        residual_lag1=diag.evaluation.residual.lag1, policy_version=COMPONENT_POLICY_V2, residual_limit=None)


def eligibility_from_snapshot(snapshot,recipe):
    """Request/archive verification from saved original evidence, without jobs."""
    from pyecr.project.repository import _diagnostics_from_dict, DIAGNOSTICS_SCHEMA_VERSION
    from pyecr.contracts import SourceIdentity
    from .component_source import _mode_reasons
    from pyecr.contracts.common import require
    require(not _mode_reasons(snapshot.analysis_config),'; '.join(_mode_reasons(snapshot.analysis_config)))
    diag=_diagnostics_from_dict({'schema_version':DIAGNOSTICS_SCHEMA_VERSION,'kind':'DiagnosticsPresentation','payload':dict(snapshot.diagnostics_result)})
    identity=SourceIdentity.capture(snapshot.dataset,recipe,config=snapshot.analysis_config,
        fit=snapshot.fit_result,diagnostics=snapshot.diagnostics_result)
    return evaluate_component_eligibility(snapshot.dataset,snapshot.analysis_config,snapshot.fit_result,diag,identity)
