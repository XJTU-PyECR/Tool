"""Read-only, gated component observation capture. No inverse problem is run."""
from dataclasses import replace

import numpy as np

from pyecr.contracts import ComponentSourceCapture, DctSourceSnapshot, ResultState, SourceIdentity, SourceKind
from pyecr.contracts.common import require
from pyecr.diagnostics.model_selection import ParameterBound, select_model
from pyecr.models.ids import ModelID
from pyecr.plotting.decomposition import (DecompositionData, VERSION,
                                          build_observation_decomposition)
from pyecr.preprocessing.quality import evaluate_fit_quality
from pyecr.preprocessing.signed_normalization import cumulative_t0_s, raw_observations_sha256
from pyecr.report.snapshot import _diagnostics_mapping
from pyecr.services.diagnostics_service import DiagnosticsPresentation, _local_uncertainty
from pyecr.services.simulation_transfer import capture_simulation_parent,validate_simulation_capture
from pyecr.services.component_eligibility import evaluate_component_eligibility

SUPPORTED_MODELS = frozenset({ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK, ModelID.M8_DUAL_K})


def _revision(session):
    r = session.current_revision
    return (r.data_revision, r.config_revision, r.physical_input_revision)


def _identity(session):
    return SourceIdentity.capture(session.dataset, session.preprocessing.config,
        config=session.current_analysis_config, fit=session.fit_result,
        diagnostics=_diagnostics_mapping(session.diagnostics))


def _unavailable(reasons, *, status='error', identity=None, warnings=()):
    return ComponentSourceCapture(identity=identity,
        state=ResultState(status=status, blocking_reasons=tuple(dict.fromkeys(reasons)),
                          warnings=tuple(dict.fromkeys(warnings))))


def _parent_reasons(session):
    reasons = []
    if session.fit_result is not None and session.fit_generation < 1:
        reasons.append('PARENT_GENERATION_UNRECORDED: this historical result has no persisted Fit generation; '
                       'view the saved result or explicitly run a new Fit before capturing components')
    for name in ('dataset', 'preprocessing', 'current_analysis_config', 'fit_result'):
        if getattr(session, name, None) is None: reasons.append(f'PARENT_MISSING: {name}')
    if not isinstance(session.diagnostics, DiagnosticsPresentation):
        reasons.append('PARENT_DIAGNOSTICS_MISSING: explicitly run current basic Diagnostics')
    if reasons: return reasons
    if session.fit_stale or session.fit_revision != session.current_revision:
        reasons.append('PARENT_FIT_STALE: run Fit explicitly for the current inputs')
    if session.diagnostics_stale or session.diagnostics_fit_revision != session.fit_revision:
        reasons.append('PARENT_DIAGNOSTICS_STALE: run Diagnostics explicitly for the current Fit')
    if session.diagnostics_fit_generation != session.fit_generation:
        reasons.append('PARENT_DIAGNOSTICS_GENERATION_MISMATCH: current Fit needs explicit Diagnostics')
    if not isinstance(session.diagnostics_identity, SourceIdentity):
        reasons.append('PARENT_IDENTITY_UNRECORDED: run current Diagnostics explicitly; no legacy inference')
    else:
        try: session.diagnostics_identity.assert_matches(_identity(session))
        except (ValueError, TypeError, KeyError) as exc:
            reasons.append(f'PARENT_IDENTITY_MISMATCH: {exc}')
    return reasons


def _reliability_reasons(dataset, config, fit, diag):
    """Reuse the existing gate, including bounds in the reported M8 order.

    A cached or manually edited reliable=True alone is not evidence. This local
    check neither runs a Fit nor accepts/replaces Diagnostics in the session.
    """
    if diag.evaluation is not None:
        from dataclasses import fields
        from .diagnostics_service import canonical_diagnostics
        expected = canonical_diagnostics(dataset, config, fit, diag)
        reasons = list(evaluate_fit_quality(dataset, config).blocking_reasons)
        if any(getattr(diag, f.name) != getattr(expected, f.name)
               for f in fields(diag) if f.name != 'sensitivity'):
            reasons.append('PARENT_EVIDENCE_ORIGIN_UNRESOLVED: v2 evidence differs from source')
        reasons.extend(expected.gate_failures)
        if not expected.reliable:
            reasons.append('PARENT_INFERENCE_UNSUPPORTED: '+expected.evaluation.summary)
        return reasons
    reasons = list(evaluate_fit_quality(dataset, config).blocking_reasons)
    if fit.exit_status <= 0 or not diag.optimizer_success:
        reasons.append('PARENT_OPTIMIZER_INCOMPLETE')
    if not diag.reliable or diag.best_reliable != config.model_id.value:
        reasons.append('PARENT_FIT_UNRELIABLE: current Diagnostics does not support this model')
    reasons.extend(diag.gate_failures)
    require(diag.model_id == config.model_id.value, 'Diagnostics/model identity conflict')
    require(diag.time_s == tuple(dataset.processed_time) and diag.residuals == tuple(fit.residuals),
            'Diagnostics observation/residual identity conflict')
    rows = {p.name: p for p in diag.parameters}
    for name, value in fit.params.items():
        row = rows.get(name)
        if row is None or row.estimate != value:
            reasons.append(f'PARENT_PARAMETER_IDENTITY_MISMATCH: {name}')
        elif not row.reliable:
            reasons.append(f'PARENT_PARAMETER_UNRELIABLE: {name}: {row.reliability_reason}')
    uncertainty, why = _local_uncertainty(fit)
    if uncertainty is None:
        reasons.append(f'PARENT_UNCERTAINTY_UNAVAILABLE: {why}')
    elif uncertainty.rank < len(fit.params):
        reasons.append('PARENT_JACOBIAN_RANK_DEFICIENT')
    bounds = {name: ParameterBound(*pair, space='physical') for name, pair in
              config.parameter_bounds.items() if name in fit.params}
    if config.model_id is ModelID.M8_DUAL_K:
        bounds = {name: ParameterBound(*pair, space='physical') for name, pair in
                  fit.diagnostics['dual_surface_k']['parameter_bounds'].items()}
    assessment = select_model({config.model_id.value: fit},
        uncertainties=None if uncertainty is None else {config.model_id.value: uncertainty},
        parameter_bounds={config.model_id.value: bounds}).assessments[config.model_id.value]
    reasons.extend(assessment.gate_failures)
    if not assessment.reliable: reasons.append('PARENT_CORE_GATE_REJECTED')
    advisor = diag.reliability_advisor
    if advisor.identifiability_status in {'weak', 'unavailable'}:
        reasons.append(f'PARENT_IDENTIFIABILITY: {advisor.identifiability_status}')
    if advisor.model_adequacy_status in {'poor', 'unavailable'}:
        reasons.append(f'PARENT_MODEL_ADEQUACY: {advisor.model_adequacy_status}')
    return reasons


def _mode_reasons(config):
    if config.model_id not in SUPPORTED_MODELS:
        return ('PARENT_MODEL_UNSUPPORTED: expected M2_DUAL_HO_D, M3_DUAL_HO_DK or M8_DUAL_K',)
    if 'fixed_d' in config.model_specific or config.boundary.mode != 'ideal_step' or config.fit.loss != 'linear':
        return ('CONDITIONAL_MODE_NOT_VALIDATED: P08 supports standard ideal-step linear-loss joint fits; '
                'fixed-D and other conditional adapters require separate validation',)
    return ()


def capture_component_sources(session, *, decomposition=None) -> ComponentSourceCapture:
    """Explicit P08 entry point. An unavailable capture contains reasons, no curves.

    Only the full observation branch of the existing production decomposition is
    evaluated. Display curves, even when supplied in a cached view, are not read.
    No smoothing, cropping, normalization, refit, Diagnostics publication or DCT.
    """
    identity = None
    try:
        reasons = _parent_reasons(session)
        if reasons:
            return _unavailable(reasons, status='stale' if any('STALE' in r or 'MISMATCH' in r for r in reasons) else 'error')
        dataset, config, fit, diag = (session.dataset, session.current_analysis_config,
                                     session.fit_result, session.diagnostics)
        identity = _identity(session)
        revision, generation = _revision(session), session.fit_generation
        unsupported = _mode_reasons(config)
        if unsupported: return _unavailable(unsupported, status='unsupported', identity=identity)
        simulation_parent=capture_simulation_parent(dataset)
        observation = build_observation_decomposition(dataset, config, fit)
        require(observation['model_id'] == config.model_id.value,
                'selected/config model identity conflict; do not rename the selected Fit')
        require((observation['dataset_sha256'], observation['config_sha256'], observation['fit_result_sha256']) ==
                (identity.dataset_sha256, identity.config_sha256, identity.fit_sha256), 'decomposition source identity conflict')
        if decomposition is not None:
            require(isinstance(decomposition, DecompositionData), 'cached view must be DecompositionData')
            for name in ('model_id', 'observation', 'observed', 'residuals', 'original_raw_indices', 'y0',
                         'signed_amplitudes', 'signal_unit', 'signal_type', 'dataset_sha256', 'config_sha256', 'fit_result_sha256'):
                require(getattr(decomposition, name) == observation[name], f'cached observation identity conflict: {name}')
        eligibility=evaluate_component_eligibility(dataset,config,fit,diag,identity)
        reasons = list(eligibility.blocking_reasons)
        amplitudes = observation['signed_amplitudes']
        # Exact zero plus an explicit roundoff-only near-zero test, in source units.
        # This is not a statistical detection threshold or a new Fit gate.
        floor = 64*np.finfo(float).eps*max(abs(observation['y0']), *map(abs, amplitudes),
                                         float(np.max(np.abs(fit.y_fit))), np.finfo(float).tiny)
        keys = ('component_1', 'component_2') if config.model_id is ModelID.M8_DUAL_K else ('component_H', 'component_O')
        for key, amplitude in zip(keys, amplitudes):
            if abs(amplitude) <= floor:
                reasons.append(f'COMPONENT_ZERO_OR_ROUNDOFF_AMPLITUDE: {key}; |A|={abs(amplitude):.17g}, floor={floor:.17g}')
        if reasons: return _unavailable(reasons, identity=identity, warnings=diag.warnings)
        parent = replace(DctSourceSnapshot.capture_experiment(dataset, session.preprocessing.config,
            t0_s=cumulative_t0_s(dataset)), identity=identity)
        grid = observation['observation']
        components = tuple(DctSourceSnapshot(schema_version=2,eligibility=eligibility,source_kind=SourceKind.FIT_COMPONENT, identity=identity,
            time_s=parent.time_s, signal=signal, raw_indices=parent.raw_indices, raw_count=parent.raw_count,
            signal_type=parent.signal_type, signal_unit=parent.signal_unit,
            representation='physical' if parent.signal_type != 'normalized' else 'unqualified_normalized',
            state=ResultState(), parent_source_sha256=parent.content_sha256, component_key=key,
            component_initial_value=0., component_terminal_value=amplitude,
            parent_reliability=eligibility.parent_reliability, t0_s=parent.t0_s)
            for key, signal, amplitude in zip(keys, (grid.component_1, grid.component_2), amplitudes))
        m8 = config.model_id is ModelID.M8_DUAL_K
        mapping = tuple(int(i) for i in fit.diagnostics['dual_surface_k']['reported_to_input']) if m8 else (0, 1)
        warnings = tuple(dict.fromkeys((*diag.warnings,*eligibility.warnings,
            *((f'PARENT_IDENTIFIABILITY_CAUTION: {diag.reliability_advisor.identifiability_explanation}',)
              if diag.reliability_advisor.identifiability_status == 'caution' else ()),
            *((f'PARENT_MODEL_ADEQUACY_CAUTION: {diag.reliability_advisor.model_adequacy_explanation}',)
              if diag.reliability_advisor.model_adequacy_status == 'caution' else ()),
            'MODEL_CONDITIONAL_COMPONENTS: fitted parent roles; not independently measured or chemically assigned',
            'COMPONENT_DCT_NOT_RUN: unnormalized observation sources only',
            *(('SURFACE_CONTROLLED_ASSUMPTION: effective M8 modes; D and Bi are not estimated',) if m8 else ()))))
        if simulation_parent is not None:
            warnings=(*warnings,'SIMULATION_DERIVED_PARENT: estimated from explicitly sent simulated observations; given truth is separate; no independent experimental or component-noise evidence')
        result = ComponentSourceCapture(schema_version=3,eligibility=eligibility,
            extraction_version='component-observation-source/3-capability',
            simulation_parent=simulation_parent,state=ResultState(status='warning', warnings=warnings), identity=identity,
            parent_source=parent, components=components,
            labels=('Effective process 1', 'Effective process 2') if m8 else observation['labels'],
            label_semantics='effective_modes_not_species' if m8 else 'declared_H_O_slots_not_species_proof',
            baseline=observation['y0'], signed_amplitudes=amplitudes,
            directions=tuple('increase' if a > 0 else 'decrease' for a in amplitudes),
            parent_step_type=config.model.step_type, fit_total=grid.total,
            asymptotic_total=observation['y0']+sum(amplitudes), mask=tuple(bool(v) for v in dataset.mask),
            raw_sha256=raw_observations_sha256(dataset), reported_to_input=mapping,
            input_view_mode='zero' if decomposition is None else decomposition.options.mode,
            observation_parity_max_abs=observation['observation_parity_max_abs'],
            revision=revision, fit_generation=generation, decomposition_version=VERSION)
        late = component_source_current_reasons(result, session)
        return _unavailable(late, status='stale', identity=identity) if late else result
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        return _unavailable((f'COMPONENT_SOURCE_INVALID: {type(exc).__name__}: {exc}',), identity=identity)


def component_source_current_reasons(capture: ComponentSourceCapture, session) -> tuple[str, ...]:
    """Check before a future P09 run; an immutable historical capture is not current."""
    reasons = list(_parent_reasons(session))
    if not capture.state.ready: reasons.extend(capture.state.blocking_reasons)
    if reasons: return tuple(dict.fromkeys(reasons))
    try:
        capture.identity.assert_matches(_identity(session))
        validate_simulation_capture(capture,session.dataset)
        require(not _mode_reasons(session.current_analysis_config), '; '.join(_mode_reasons(session.current_analysis_config)))
        require(capture.revision == _revision(session) and capture.fit_generation == session.fit_generation,
                'component capture workflow revision/Fit generation conflict')
        require(capture.raw_sha256 == raw_observations_sha256(session.dataset), 'component raw identity conflict')
        # A loaded JSON's identity claims are not a proof of its curve contents.
        # Reuse the production observation extraction, never its display grid.
        observation = build_observation_decomposition(session.dataset, session.current_analysis_config, session.fit_result)
        grid = observation['observation']
        require(capture.fit_total == grid.total and capture.baseline == observation['y0']
                and capture.signed_amplitudes == observation['signed_amplitudes'], 'component parent numerical binding conflict')
        require(capture.parent_source.time_s == grid.time_s
                and capture.parent_source.signal == observation['observed']
                and capture.parent_source.raw_indices == observation['original_raw_indices']
                and capture.parent_source.raw_count == len(session.dataset.raw_time)
                and capture.mask == tuple(bool(v) for v in session.dataset.mask)
                and capture.parent_source.t0_s == cumulative_t0_s(session.dataset), 'component parent observation/origin binding conflict')
        require(tuple(c.signal for c in capture.components) == (grid.component_1, grid.component_2),
                'component numerical binding conflict')
        require(capture.parent_source.signal_unit == observation['signal_unit']
                and capture.parent_source.signal_type == observation['signal_type'], 'component parent unit binding conflict')
        mapping = tuple(session.fit_result.diagnostics['dual_surface_k']['reported_to_input']) if session.current_analysis_config.model_id is ModelID.M8_DUAL_K else (0, 1)
        require(capture.reported_to_input == mapping and capture.parent_step_type == session.current_analysis_config.model.step_type,
                'component label permutation/direction binding conflict')
        if capture.eligibility is None:
            # Old archives retain their original strict meaning, never inferred eligible.
            reasons.extend(_reliability_reasons(session.dataset,session.current_analysis_config,session.fit_result,session.diagnostics))
        else:
            current=evaluate_component_eligibility(session.dataset,session.current_analysis_config,session.fit_result,session.diagnostics,capture.identity)
            require(current==capture.eligibility,'component eligibility no longer matches original evidence')
            reasons.extend(current.blocking_reasons)
    except (ValueError, TypeError, KeyError) as exc:
        reasons.append(f'COMPONENT_SOURCE_STALE: {exc}')
    return tuple(reasons)
