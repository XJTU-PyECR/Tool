"""Explicit noise references and post-stop evidence; no optimizer or gate hooks."""
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from pyecr.contracts.common import SourceIdentity,require,number,canonical_sha256,strict_json_loads,exact_fields
from pyecr.contracts.noise import NoiseReference,NoiseComparison
from pyecr.fitting.iteration_observation import residual_metrics
from pyecr.preprocessing import PreprocessConfig,raw_observations_sha256
from pyecr.services.diagnostics_service import _residual_summary
from pyecr.units import validate_pair,factor_to_reference

VERSION='post-stop-noise-evidence/1'
GUARDS=dict(min_platform_n=8,lag1_abs=.3,drift_span_over_sample_sd=2.,half_SD_ratio=2.,min_n_for_half_SD=40)
CONDITIONS=('Descriptive comparison after stopping; never a stop condition or reliability gate.',
    'Ratio assumes independent, stationary, zero-mean, constant-SD errors and an adequate local model.',
    'Model discrepancy may contribute to residuals; no calibrated probability or confidence interval is calculated.')


def response_profile(dataset):
    kind=str(dataset.metadata.get('signal_type','unknown'))
    return kind if kind!='normalized' else 'normalized:'+str(dataset.metadata.get('normalization_mode','imported_unspecified'))


def structure_checks(times,values):
    """Read-only, predeclared heuristic guards, not significance tests."""
    y=np.asarray(values,dtype=float);t=np.asarray(times,dtype=float)
    require(y.ndim==1 and y.shape==t.shape and y.size>0 and np.all(np.isfinite(y)) and np.all(np.isfinite(t)),'Finite full arrays required for noise checks')
    summary=_residual_summary(y);n=len(y);sd=summary.standard_deviation
    issues=[];warnings=[];drift=half=None
    if n<GUARDS['min_platform_n']:warnings.append('too_few_points_for_structure_checks')
    else:
        if summary.lag1_autocorrelation is not None and abs(summary.lag1_autocorrelation)>=GUARDS['lag1_abs']:issues.append('correlated_structure: abs(lag1) >= 0.3')
        if np.ptp(t)>0 and sd>0:
            # A diagnostic slope, not subtraction/detrending of any data.
            x=(t-t.min())/np.ptp(t);x=x-x.mean()
            drift=abs(float(x@(y-y.mean())/(x@x)))/sd
            if drift>=GUARDS['drift_span_over_sample_sd']:issues.append('drift_structure: fitted span/sample SD >= 2')
        if n>=GUARDS['min_n_for_half_SD']:
            a=float(np.std(y[:n//2],ddof=1));b=float(np.std(y[n//2:],ddof=1))
            if min(a,b)>0:
                half=max(a,b)/min(a,b)
                if half>=GUARDS['half_SD_ratio']:issues.append('heteroscedastic_structure: half-window SD ratio >= 2')
            elif max(a,b)>0:issues.append('heteroscedastic_structure: one half has zero SD')
    if summary.lag1_autocorrelation is None:warnings.append('lag1_unassessed: '+str(summary.lag1_reason))
    return dict(lag1=summary.lag1_autocorrelation,drift_span_over_sd=drift,half_sd_ratio=half,issues=tuple(issues),warnings=tuple(warnings))


def manual_reference(dataset,recipe,*,sd,unit,assumptions='unknown',basis='User-declared external SD'):
    """Explicit user declaration on this exact current response representation."""
    kind=str(dataset.metadata.get('signal_type','unknown'));validate_pair(kind,unit)
    sd=number(sd,'Noise SD',positive=True);source=SourceIdentity.capture(dataset,recipe)
    return NoiseReference(source=source,origin=source,kind='manual_sd',sigma=sd,signal_type=kind,signal_unit=unit,
        response_profile=response_profile(dataset),assumptions=assumptions,basis=basis,
        data_origin=str(dataset.metadata.get('source_kind','user_source_unspecified')),method='external_declared_sd',raw_count=len(dataset.raw_signal),
        time_unit=str(dataset.metadata.get('time_unit','s')),input_sigma=sd,input_unit=unit,input_profile=response_profile(dataset),conditions=CONDITIONS)


def platform_reference(pipeline,which,*,assumptions='unknown',basis='Explicitly selected accepted platform statistics'):
    """Locate the last accepted recipe step, not preview/metadata alone.

    Returns the reference plus its immutable parent dataset/recipe for a later
    explicit single-normalization conversion. Never commits a recipe step.
    """
    require(which in ('initial','final'),'Select Initial or Final accepted platform')
    op='choose_'+which+'_stability_window'
    found=[i for i,s in enumerate(pipeline.config.steps) if s.operation==op]
    require(bool(found),'No accepted '+which+' platform; Preview candidates are not noise references')
    i=found[-1];recipe=PreprocessConfig(pipeline.config.steps[:i+1]);dataset=recipe.replay(pipeline.source)
    record=dataset.history[-1];offset=len(pipeline.source.history)
    require(canonical_sha256(record.to_dict())==canonical_sha256(pipeline.dataset.history[offset+i].to_dict()),
        'Accepted platform changed under later replay; select and accept its current interval explicitly')
    w=dataset.metadata['stability_windows'][which];indices=tuple(w['raw_indices']);raw=np.flatnonzero(dataset.mask)
    selected=np.isin(raw,indices);y=dataset.processed_signal[selected];t=dataset.processed_time[selected]
    require(tuple(raw[selected])==indices and len(y)==w['n_points']>=2,'Accepted platform row mapping is not current')
    sd=float(np.std(y,ddof=1));require(sd>0 and np.isfinite(sd),'Accepted platform SD is zero or unavailable; no invented noise floor')
    checks=structure_checks(t,y);issues=list(checks['issues'])
    if len(y)<GUARDS['min_platform_n']:issues.append('platform_sample_too_short: fewer than 8 accepted observations')
    origin=SourceIdentity.capture(dataset,recipe);current=pipeline.dataset
    # Explicitly selecting an older accepted window can bind it to the current
    # response if only other platform declarations followed. No signal/mask/
    # time/representation change is silently reinterpreted.
    suffix=pipeline.config.steps[i+1:]
    if all(s.operation in ('choose_initial_stability_window','choose_final_stability_window') for s in suffix) and (
            np.array_equal(current.processed_signal,dataset.processed_signal) and np.array_equal(current.mask,dataset.mask)
            and np.array_equal(current.processed_time,dataset.processed_time) and response_profile(current)==response_profile(dataset)):
        dataset=current;recipe=pipeline.config
    source=SourceIdentity.capture(dataset,recipe);unit=str(w['signal_unit'])
    ref=NoiseReference(source=source,origin=origin,kind='accepted_'+which+'_platform',sigma=sd,
        signal_type=str(w['signal_type']),signal_unit=unit,response_profile=response_profile(dataset),assumptions=assumptions,basis=basis,
        data_origin=str(dataset.metadata.get('source_kind','user_source_unspecified')),method='accepted_window_sample_sd_ddof1',
        raw_count=len(dataset.raw_signal),time_unit=str(w['time_unit']),raw_indices=indices,sample_times=tuple(map(float,t)),
        sample_values=tuple(map(float,y)),interval=(float(w['start_time']),float(w['end_time'])),sample_count=len(y),
        input_sigma=sd,input_unit=unit,input_profile=response_profile(dataset),issues=tuple(issues),
        conditions=(*CONDITIONS,'Sample SD uses ddof=1; application outside this accepted window assumes the same noise process.',*checks['warnings']))
    return ref,dataset,recipe


def convert_normalized_reference(reference,parent,parent_recipe,target,target_recipe):
    """Explicit SD-only conversion; verify one accepted production step on a copy."""
    require(reference.parent_reference_id is None,'Chained noise-scale conversions Unsupported; declare a new reference')
    require(reference.source==SourceIdentity.capture(parent,parent_recipe),'Noise parent dataset/recipe conflict')
    require(len(target_recipe.steps)==len(parent_recipe.steps)+1 and target_recipe.steps[:-1]==parent_recipe.steps,
        'Noise conversion requires exactly one accepted normalization after its parent; other recipe changes remain Stale')
    step=target_recipe.steps[-1]
    require(step.operation in ('normalize','normalize_signed_dct'),'Noise conversion requires an accepted affine normalization')
    require(raw_observations_sha256(parent)==raw_observations_sha256(target) and np.array_equal(parent.mask,target.mask),
        'Noise normalization raw/mask conflict')
    from pyecr.preprocessing.pipeline import _apply_processing_step
    # Existing production function, detached verification only, never Analysis.
    checked=_apply_processing_step(parent,step,recipe=parent_recipe)
    require(SourceIdentity.capture(checked,target_recipe)==SourceIdentity.capture(target,target_recipe),'Accepted normalization replay conflict')
    record=checked.history[-1]
    if step.operation=='normalize':
        scale=abs(float(record.parameters['final_value'])-float(record.parameters['initial_value']))
        version='existing-progress-remaining/1'
    else:
        from pyecr.contracts import SignedNormalizationResult
        result=SignedNormalizationResult.from_dict(record.parameters['normalization'])
        require(result.state.ready,'Signed normalization scale is unavailable')
        scale=result.scale_value;version='accepted-signed-terminal/1'
    unit=str(parent.metadata['signal_unit']);factor=factor_to_reference(reference.signal_unit)/factor_to_reference(unit)
    sigma=reference.sigma*factor/scale
    return replace(reference,source=SourceIdentity.capture(target,target_recipe),sigma=sigma,signal_type='normalized',signal_unit='1',
        response_profile=response_profile(target),parent_reference_id=reference.content_sha256,normalization_scale=scale,
        normalization_scale_unit=unit,normalization_version=version,
        conditions=(*reference.conditions,'Conditional on fixed accepted normalization endpoints/scale; their uncertainty is not propagated.'))


def comparison_from_arrays(*,dataset,recipe,residuals,p_free,reference=None,config=None,fit=None,
                           rank=None,loss='linear',weighting='unknown',execution_request_id=None,candidate_source=None):
    """Generic pure metric primitive; callers must supply qualified source evidence.

    Tests may use analytical residual arrays without making a fake FitResult.
    Such arrays remain stopped/native candidates unless a real Fit is provided.
    """
    r=np.asarray(residuals,dtype=float);require(r.shape==dataset.processed_signal.shape,'Noise comparison needs every valid observation')
    m=residual_metrics(r,p_free=p_free,unit=str(dataset.metadata.get('signal_unit','signal')))
    source=SourceIdentity.capture(dataset,recipe,config=config,fit=fit)
    if candidate_source is not None:
        require(fit is None and isinstance(candidate_source,SourceIdentity) and candidate_source.fit_id is None,'Invalid candidate parent source')
        require((source.dataset_sha256,source.recipe_sha256)==(candidate_source.dataset_sha256,candidate_source.recipe_sha256),'Candidate parent dataset/recipe conflict')
        source=candidate_source
    checks=structure_checks(dataset.processed_time,r);reasons=[];warnings=list(checks['warnings'])
    status='not_set' if reference is None else 'not_comparable';sigma=factor=None
    if reference is None:reasons.append('noise_reference_not_set')
    else:
        require(isinstance(reference,NoiseReference),'Expected immutable NoiseReference')
        if (reference.source.dataset_sha256,reference.source.recipe_sha256)!=(source.dataset_sha256,source.recipe_sha256):
            status='stale';reasons.append('noise_reference_source_stale: explicit re-declaration or accepted-scale conversion required')
        if reference.signal_type!=str(dataset.metadata.get('signal_type','unknown')) or reference.response_profile!=response_profile(dataset):
            reasons.append('noise_response_representation_mismatch')
        else:
            try:
                validate_pair(reference.signal_type,reference.signal_unit);validate_pair(reference.signal_type,m['residual_unit'])
                factor=factor_to_reference(reference.signal_unit)/factor_to_reference(m['residual_unit']);sigma=reference.sigma*factor
            except (ValueError,TypeError):reasons.append('noise_unit_scale_unknown_or_incompatible')
        if reference.assumptions!='iid_stationary_constant_sd':reasons.append('noise_assumptions_'+reference.assumptions)
        reasons.extend('reference_'+x for x in reference.issues)
    if m['df']<=0:reasons.append('RSE_unavailable: N <= p_free')
    if loss!='linear':reasons.append('robust_loss: raw RSE is descriptive, not an IID residual-variance estimate')
    if weighting!='known_unit_weights':reasons.append('weights_unknown_or_nonunit: no unweighted noise ratio')
    if rank is None:reasons.append('jacobian_rank_unavailable')
    elif rank<p_free:reasons.append('jacobian_rank_deficient')
    reasons.extend('residual_'+x for x in checks['issues'])
    ratio=None
    if reference is not None and not reasons:
        status='conditional';ratio=m['RSE']/sigma
    return NoiseComparison(source=source,reference=reference,residual_role='physical_fit_residual' if fit is not None else 'stopped_native_candidate',
        raw_indices=tuple(map(int,np.flatnonzero(dataset.mask))),raw_count=len(dataset.raw_signal),residuals=tuple(map(float,r)),time=tuple(map(float,dataset.processed_time)),
        unit=m['residual_unit'],response_profile=response_profile(dataset),p_free=p_free,raw_SSE=m['raw_SSE'],raw_RMSE=m['RMSE'],RSE=m['RSE'],
        sigma_noise=sigma,RSE_over_sigma=ratio,noise_unit_factor=factor,status=status,reasons=tuple(reasons),warnings=tuple(warnings),
        conditions=(*CONDITIONS,*(() if reference is None else tuple(c for c in reference.conditions if c not in CONDITIONS))),
        loss=loss,weighting=weighting,rank=rank,lag1=checks['lag1'],drift_span_over_sd=checks['drift_span_over_sd'],half_sd_ratio=checks['half_sd_ratio'],
        execution_request_id=execution_request_id,N=m['N'],df=m['df'])


def compare_current_fit(session,reference=None):
    """No forward/optimizer/Diagnostics invocation; stored full residual/J only."""
    require(session.dataset is not None and session.current_analysis_config is not None and session.fit_result is not None,'Noise metrics need a current parent Fit')
    require(not session.fit_stale and session.fit_revision==session.current_revision,'Noise parent Fit is Stale; retained history is not current')
    ds=session.dataset;fit=session.fit_result;config=session.current_analysis_config
    require(fit.y_fit.shape==fit.residuals.shape==ds.processed_signal.shape,'Parent Fit full-observation arrays conflict')
    # Existing paths report either residual sign. Check identity, do not change it.
    expected=ds.processed_signal-fit.y_fit
    require(np.allclose(fit.residuals,expected,rtol=1e-10,atol=1e-12) or np.allclose(-fit.residuals,expected,rtol=1e-10,atol=1e-12),'Parent residual/data identity conflict')
    p=len(fit.params);rank=None
    if fit.jacobian is not None:
        j=np.asarray(fit.jacobian)
        if j.shape==(len(expected),p) and np.all(np.isfinite(j)):rank=int(np.linalg.matrix_rank(j))
    known=config.model_id.value in {'M0_SINGLE_D','M1_SINGLE_DK','M2_DUAL_HO_D','M3_DUAL_HO_DK','M7_SINGLE_K','M8_DUAL_K'}
    weight_keys={'weights','observation_weights','residual_weights','weighting'}
    unknown=any(k in ds.metadata or k in config.model_specific or k in fit.diagnostics for k in weight_keys)
    return comparison_from_arrays(dataset=ds,recipe=session.preprocessing.config,residuals=fit.residuals,p_free=p,
        reference=reference,config=config,fit=fit,rank=rank,loss=config.fit.loss,weighting='known_unit_weights' if known and not unknown else 'unknown')


def compare_stopped_candidate(dataset,recipe,record,reference=None):
    from pyecr.fitting.monitored import read_monitored
    record=read_monitored(json.dumps(record,allow_nan=False))
    binding=record['binding'];source=SourceIdentity.from_dict(binding['source']);current=SourceIdentity.capture(dataset,recipe)
    require((source.dataset_sha256,source.recipe_sha256)==(current.dataset_sha256,current.recipe_sha256),'Stopped candidate source is Stale')
    require(bool(record['contexts']),'Stopped job has no complete native endpoint')
    c=record['contexts'][-1];r=c['monitored_execution']['endpoint_raw_residuals']
    require(r is not None,'Stopped job has no finite complete residual vector')
    return comparison_from_arrays(dataset=dataset,recipe=recipe,residuals=r,p_free=len(c['accepted_trace']['parameter_names']),reference=reference,
        rank=None,loss=c['accepted_trace']['loss'],weighting='known_unit_weights' if c['accepted_trace']['weights_sha256'] is None else 'nonunit',
        execution_request_id=binding['execution_request_id'],candidate_source=source)


def save_noise_evidence(payload,path):
    require(isinstance(payload,(NoiseReference,NoiseComparison)),'Expected noise reference or comparison')
    path=Path(path);require(not path.exists(),'Noise destination already exists; use a new file')
    value=dict(version=VERSION,payload=payload.to_dict());value['sha256']=canonical_sha256(value)
    stage=path.with_name(path.name+'.'+uuid4().hex+'.staging')
    try:
        stage.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        require(read_noise_evidence(stage.read_text(encoding='utf-8'))==payload,'Noise staging parity conflict');stage.rename(path)
    finally:
        if stage.exists():stage.unlink()
    return path


def read_noise_evidence(text):
    value=strict_json_loads(text);exact_fields(value,('version','payload','sha256'),'Noise sidecar')
    require(value['version']==VERSION,'Unknown noise evidence version')
    require(value['sha256']==canonical_sha256({k:v for k,v in value.items() if k!='sha256'}),'Noise sidecar hash conflict')
    from pyecr.contracts import read_contract
    result=read_contract(value['payload']);require(isinstance(result,(NoiseReference,NoiseComparison)),'Not a noise payload')
    return result
