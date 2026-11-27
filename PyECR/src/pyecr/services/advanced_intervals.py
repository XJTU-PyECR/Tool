"""R07 selected advanced output; existing samples/scans and old gate stay apart."""
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.stats import chi2

from pyecr.contracts import ConfidencePolicy, FitControlBinding, FitControlSpec, TerminationPolicy
from pyecr.contracts.common import canonical_sha256, exact_fields, integer, require, strict_json_loads
from pyecr.services.local_intervals import _current_source
from pyecr.uncertainty import UncertaintyMethod, UncertaintyResult
from pyecr.uncertainty.bootstrap import _percentile_intervals, _to_coordinate_samples
from pyecr.uncertainty.profile import _profile_interval

VERSION='advanced-selected-output/1'
MIN_TAIL=1.
SPARSE_TAIL=5.
ROUNDING=1e-12


def checked_C(C):return ConfidencePolicy(confidence_level=C).confidence_level


def observation_conditions(dataset,config,fit):
    """Registry units and full observation mapping; no scientific calculation."""
    from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
    unit=str(dataset.metadata.get('signal_unit','signal'))
    units={p.name:(unit if p.unit=='signal' else p.unit) for p in DEFAULT_MODEL_REGISTRY.get(config.model_id).parameters}
    require(set(fit.params)<=set(units),'Unsupported parent parameter/model identity')
    return dict(N=len(fit.y_fit),p_free=len(fit.params),raw_indices=np.flatnonzero(dataset.mask).tolist(),
        signal_unit=unit,time_unit='s',units={name:units[name] for name in fit.params},
        coordinate_units={name:'decade' if fit.param_space[name]=='log' else units[name] for name in fit.params})


def relevel_result(fit,result,C,*,minimum_success_fraction=None):
    """Only quantiles/threshold crossings; no refit or resampling callable.

    Low-level statistical operation. The GUI/controller additionally requires
    the R01 parent binding before treating this as current selected output.
    """
    C=checked_C(C)
    require(isinstance(result,UncertaintyResult),'Unsupported: no stored uncertainty method/result')
    require(tuple(result.parameter_names)==tuple(fit.params) and dict(result.estimates)==dict(fit.params)
            and dict(result.param_space)==dict(fit.param_space),'Saved evidence/parent parameters conflict')
    diagnostics={**result.diagnostics,'output_version':VERSION,'selected_confidence_level':C,
        'gate_confidence_level':.95,'gate_unchanged':True,'coverage_calibrated':False}
    # Method-specific old warnings are replaced by reasons for the NEW level.
    warnings=[w for w in result.warnings if not w.startswith(('BOOTSTRAP_TAIL_','PROFILE_INTERVAL_TRUNCATED:'))]
    intervals={}
    if result.method is UncertaintyMethod.RESIDUAL_BOOTSTRAP:
        coordinates=result.coordinate_samples;physical=result.physical_samples
        require(coordinates is not None and physical is not None,
                'Saved samples unavailable; explicitly Run bootstrap to obtain complete samples')
        require(len(coordinates)==len(physical)==result.n_successful,'Incomplete saved sample count')
        require(np.allclose(_to_coordinate_samples(fit,physical),coordinates,rtol=1e-12,atol=1e-14),'Saved physical/coordinate sample conflict')
        minimum=(result.diagnostics.get('minimum_success_fraction',.8) if minimum_success_fraction is None else minimum_success_fraction)
        require(0<minimum<=1,'invalid saved success policy')
        n=result.n_successful;fraction=n/result.n_requested if result.n_requested else 0.;tail=n*(1-C)/2
        status='completed';reason=''
        if n<2 or fraction+ROUNDING<minimum:
            status='unavailable';reason='LOW_BOOTSTRAP_SUCCESS_RATE: insufficient successful refits; no interval published'
        elif tail+ROUNDING<MIN_TAIL:
            status='unavailable';reason='BOOTSTRAP_TAIL_BUDGET: fewer than one expected sample per tail; explicitly choose a new budget/run'
        elif tail+ROUNDING<SPARSE_TAIL:
            warnings.append(f'BOOTSTRAP_TAIL_SPARSE: {tail:.6g} expected samples per tail; finite-budget percentile estimate is not calibrated coverage.')
        if not reason:intervals=_percentile_intervals(fit,coordinates,confidence_level=C)
        diagnostics.update(output_status=status,output_reason=reason,quantile_method='linear',minimum_success_fraction=minimum,
            expected_samples_per_tail=tail,minimum_expected_samples_per_tail=MIN_TAIL,sparse_tail_warning_below=SPARSE_TAIL,
            tail_probabilities=[(1-C)/2,(1+C)/2],sample_evidence_sha256=canonical_sha256(dict(names=result.parameter_names,
                physical=physical.tolist(),coordinates=coordinates.tolist())),recompute_action='saved_samples_quantiles_only')
        profiles={}
    elif result.method is UncertaintyMethod.PROFILE_LIKELIHOOD:
        require(bool(result.profiles),'Saved profile unavailable; explicitly Run a constrained scan')
        cutoff=float(chi2.ppf(C,df=1));profiles={};ranges={};support={}
        for name,old in result.profiles.items():
            require(old.space=='log' and name in fit.params,'Unsupported saved profile coordinate/parameter')
            trace=replace(old,cutoff=cutoff);profiles[name]=trace
            interval,truncated=_profile_interval(fit,name,trace,confidence_level=C)
            if interval is not None and not truncated:
                intervals[name]=interval;ranges[name]='bracketed'
            else:
                ranges[name]='truncated_saved_scan' if interval is not None else 'unavailable'
                if interval is not None:support[name]=dict(lower_scan_value=interval.lower,upper_scan_value=interval.upper,
                    meaning='Observed accepted scan support only; NOT finite CI endpoints')
                warnings.append(f'PROFILE_INTERVAL_TRUNCATED: {name}; saved scan does not bracket both crossings at C={C:.12g}; explicitly extend/run, no extrapolation.')
        reason='' if len(intervals)==len(profiles) else 'Saved profile does not bracket every requested interval; explicit new/expanded scan required'
        diagnostics.update(output_status='completed' if not reason else 'partial' if intervals else 'unavailable',output_reason=reason,
            likelihood_ratio_cutoff=cutoff,chi_square_degrees_of_freedom=1,range_status=ranges,observed_scan_support_not_CI=support,
            recompute_action='saved_scan_threshold_only',interpolation='Existing adjacent successful crossings; no extrapolation')
    else:raise ValueError('Unsupported saved uncertainty method')
    return replace(result,confidence_level=C,intervals=intervals,profiles=profiles,diagnostics=diagnostics,warnings=tuple(dict.fromkeys(warnings)))


def make_binding(source,config,C,method,options):
    policy=ConfidencePolicy(confidence_level=checked_C(C),method=method,
        method_options_sha256=canonical_sha256(dict(version=VERSION,options=options)))
    return FitControlBinding.capture(source,FitControlSpec(output_ci=policy,termination=TerminationPolicy.legacy_for(config)))


def bind_presentation(presentation,binding,generation):
    require(presentation.method.value==binding.controls.output_ci.method,'Advanced method/request conflict')
    if presentation.result is not None:binding.controls.output_ci.assert_result(presentation.result)
    if presentation.experimental_result is not None:
        require(presentation.experimental_result.confidence_level==binding.controls.output_ci.confidence_level,'Experimental actual C/request conflict')
    return replace(presentation,binding=binding,fit_generation=generation)


def assert_current(presentation,session):
    require(presentation.binding is not None,'Saved advanced result has no parent identity; explicitly Run for the current parent')
    presentation.binding.assert_ci_source(_current_source(session))
    require(presentation.fit_generation==session.fit_generation,'Advanced parent generation conflict; historical result only')


def recompute_saved(presentation,session,C):
    assert_current(presentation,session)
    require(presentation.result is not None or presentation.experimental_result is not None,'No complete saved method evidence; explicitly Run advanced analysis')
    if presentation.experimental_result is not None:
        from pyecr.services.m1_deep_uncertainty import recompute_experimental_samples
        experiment=recompute_experimental_samples(presentation.experimental_result,session.fit_result,C)
        updated=replace(presentation,experimental_result=experiment,available=experiment.status=='complete',
            reason=None if experiment.status=='complete' else 'Experimental saved samples cannot support this C',warnings=experiment.warnings)
        evidence=experiment.to_dict()
    else:
        result=relevel_result(session.fit_result,presentation.result,C)
        updated=replace(presentation,result=result,available=bool(result.intervals),
            reason=result.diagnostics.get('output_reason') or None,warnings=result.warnings)
        from pyecr.report.snapshot import _json_safe
        evidence=_json_safe(presentation.result.to_dict())
    binding=make_binding(_current_source(session),session.current_analysis_config,C,presentation.method.value,
        dict(action='explicit_saved_evidence_recompute',parent_evidence_sha256=canonical_sha256(evidence)))
    return bind_presentation(updated,binding,session.fit_generation)


def advanced_to_dict(presentation):
    from pyecr.report.snapshot import _json_safe
    value=dict(schema_version=1,version=VERSION,method=presentation.method.value,available=presentation.available,
        reason=presentation.reason,warnings=list(presentation.warnings),binding=None if presentation.binding is None else presentation.binding.to_dict(),
        fit_generation=presentation.fit_generation,result=None if presentation.result is None else _json_safe(presentation.result.to_dict()),
        experimental_result=None if presentation.experimental_result is None else presentation.experimental_result.to_dict())
    return {**value,'record_sha256':canonical_sha256(value)}


def advanced_from_dict(value):
    from pyecr.services.diagnostics_service import AdvancedDiagnosticsPresentation
    from pyecr.services.m1_deep_uncertainty import ExperimentalM1UncertaintyResult
    exact_fields(value,('schema_version','version','method','available','reason','warnings','binding','fit_generation','result','experimental_result','record_sha256'),'Advanced selected output')
    require(integer(value['schema_version'],'schema_version',minimum=1)==1 and value['version']==VERSION,'Unknown advanced output version')
    require(isinstance(value['available'],bool),'available must be boolean')
    require(value['reason'] is None or isinstance(value['reason'],str),'reason must be text or null')
    require(isinstance(value['warnings'],list) and all(isinstance(w,str) for w in value['warnings']),'warnings must be text list')
    if value['fit_generation'] is not None:integer(value['fit_generation'],'fit_generation',minimum=0)
    require(not (value['result'] is not None and value['experimental_result'] is not None),'Ambiguous advanced methods/evidence')
    require(canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'})==value['record_sha256'],'Advanced evidence hash conflict')
    binding=None if value['binding'] is None else FitControlBinding.from_dict(value['binding'])
    presentation=AdvancedDiagnosticsPresentation(method=UncertaintyMethod(value['method']),available=value['available'],reason=value['reason'],
        warnings=tuple(value['warnings']),result=None if value['result'] is None else UncertaintyResult.from_dict(value['result']),
        experimental_result=None if value['experimental_result'] is None else ExperimentalM1UncertaintyResult.from_dict(value['experimental_result']))
    if binding is not None:presentation=bind_presentation(presentation,binding,value['fit_generation'])
    return presentation


def save_advanced(presentation,path):
    path=Path(path);require(not path.exists(),'Advanced output file exists; choose a new path')
    value=advanced_to_dict(presentation);staging=path.with_name(path.name+'.'+uuid4().hex+'.staging')
    try:
        staging.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        require(advanced_to_dict(advanced_from_dict(strict_json_loads(staging.read_text(encoding='utf-8'))))==value,'Advanced staging roundtrip conflict')
        staging.rename(path)
    finally:
        if staging.exists():staging.unlink()
    return path
