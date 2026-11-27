"""Source-bound read-only transport summary; shared entry for UI/DCT/report."""
from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4
import numpy as np

from pyecr.contracts.common import (SourceIdentity,require,canonical_sha256,strict_json_loads,exact_fields)
from pyecr.contracts.transport import TransportSummary,TransportInterval
from pyecr.diagnostics.transport import transport_points,log_ratio_gradient,local_interval_bounds,VERSION
from pyecr.models.ids import ModelID

CONDITIONS=('Bi uses each active full dimension L: k*(L/2)/D; dimensions stored in cm.',
    'D/k is a characteristic half-length. 2D/k is the Bi=1 full thickness of a symmetric two-face slab.',
    '0.03/30 regimes are explanatory heuristics, never a new reliability gate.',
    'H/O names retain parent-model roles only; effective modes are not new chemical assignments.',
    'Geometry and boundary are fixed background; nonideal boundaries limit interpretation of observed timescales.')


def _fit_inputs(dataset,recipe,config,fit,diagnostics=None):
    from pyecr.services.local_intervals import _metadata
    from pyecr.fitting.reproducibility import config_hash
    from pyecr.report.snapshot import _diagnostics_mapping
    require(fit.y_fit.shape==fit.residuals.shape==dataset.processed_signal.shape,'Transport parent needs every aligned valid observation')
    delta=dataset.processed_signal-fit.y_fit
    require(np.allclose(fit.residuals,delta,rtol=1e-10,atol=1e-12) or np.allclose(-fit.residuals,delta,rtol=1e-10,atol=1e-12),'Transport residual/data conflict')
    require(np.isclose(float(fit.residuals@fit.residuals),fit.SSE,rtol=1e-10,atol=1e-12),'Transport parent SSE conflict')
    declared=fit.diagnostics.get('analysis_artifacts',{}).get('model_name',fit.diagnostics.get('model_id'))
    require(declared==config.model_id.value,'Transport parent model identity conflict')
    source=SourceIdentity.capture(dataset,recipe,config=config,fit=fit,
        diagnostics=None if diagnostics is None else _diagnostics_mapping(diagnostics))
    meta=_metadata(fit)
    for k,v in (('dataset_sha256',source.dataset_sha256),('config_sha256',source.config_sha256)):
        require(k not in meta or meta[k]==v,'Transport '+k+' conflict')
    original=fit.diagnostics.get('fitting_randomness',{}).get('effective_config')
    if original:
        from pyecr.config import AnalysisConfig
        require(config_hash(AnalysisConfig.from_dict(original))==source.config_sha256,'Transport original Fit/config conflict')
    fixed={**config.model.fixed_parameters,**meta.get('fixed_parameters',{})}
    require(not(set(fit.params)&set(fixed)),'Transport fixed/free conflict')
    if 'fixed_d' in config.model_specific:
        from pyecr.services.fixed_d import validate_fixed_diffusion
        validate_fixed_diffusion(config)
        require('fixed_d' in fit.diagnostics,'Conditional fixed-D parent evidence missing')
    params={**fixed,**fit.params};reasons=[]
    reliability='pending'
    if diagnostics is not None:
        from pyecr.services.component_source import _reliability_reasons
        reasons=_reliability_reasons(dataset,config,fit,diagnostics)
        reliability='unreliable' if reasons else 'supported'
    elif fit.exit_status<=0:
        reliability='unreliable';reasons=['Native optimizer did not succeed']
    else:reasons=['Current bound Diagnostics not available; reliability pending']
    return source,params,fixed,reliability,tuple(dict.fromkeys(reasons))


def summarize_fit(dataset,recipe,config,fit,*,diagnostics=None,local_record=None):
    """Detached current or explicitly historical source. No optimizer/forward/CI job."""
    source,params,fixed,reliability,reasons=_fit_inputs(dataset,recipe,config,fit,diagnostics)
    from pyecr.services.diagnostics_service import _bi_limits
    lo,hi=_bi_limits(config)
    labels={}
    if config.model_id is ModelID.M8_DUAL_K:
        labels={f'component_{i+1}':v for i,v in enumerate(fit.diagnostics.get('dual_surface_k',{}).get('component_labels',()))}
    processes,axes=transport_points(config.model_id,config.model.geometry,params,fixed=fixed,labels=labels,surface_limit=lo,diffusion_limit=hi)
    from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
    units={p.name:(str(dataset.metadata.get('signal_unit','signal')) if p.unit=='signal' else p.unit) for p in DEFAULT_MODEL_REGISTRY.get(config.model_id).parameters}
    parameter_evidence=dict(estimated_parameters=dict(fit.params),fixed_parameters=fixed,given_parameters={},parameter_units=units,
        coordinates=dict(fit.param_space),bounds=dict(config.parameter_bounds),fixed_D_sources=dict(config.model_specific.get('fixed_d',{}).get('sources',{})),
        boundary=config.boundary.to_dict(),source_label=str(dataset.metadata.get('source_path',dataset.metadata.get('dataset_name','in-memory source'))),
        native_status=fit.exit_status,N=len(dataset.processed_signal))
    conditional='fixed_d' in config.model_specific
    conditions=(*CONDITIONS,'Fixed parameters are background; their uncertainty is not propagated.',
        'Parameter origin: fitted estimates conditional on fixed D.' if conditional else 'Parameter origin: fitted estimates.',
        'Input boundary: '+config.boundary.mode,
        'Data origin: '+str(dataset.metadata.get('source_kind','user_source_unspecified')))
    summary=TransportSummary(source=source,source_kind='conditional_fixed_d' if conditional else 'fit_estimate',simulation_request_sha256=None,
        model_id=config.model_id.value,geometry=config.model.geometry,processes=processes,axes=axes,
        raw_indices=tuple(map(int,np.flatnonzero(dataset.mask))),raw_count=len(dataset.raw_signal),parent_reliability=reliability,
        reliability_reasons=reasons,conditions=conditions,surface_limit=lo,diffusion_limit=hi,parameter_evidence=parameter_evidence)
    return derive_local_intervals(summary,fit,local_record) if local_record is not None else summary


def summarize_session(session,*,local_record=None):
    from pyecr.services.local_intervals import _current_source
    _current_source(session)  # strict current revision; historical must use detached API
    diag=None if session.diagnostics_stale else session.diagnostics
    return summarize_fit(session.dataset,session.preprocessing.config,session.current_analysis_config,session.fit_result,
        diagnostics=diag,local_record=local_record)


def summarize_simulation(request):
    from pyecr.contracts import SimulationRequest
    require(isinstance(request,SimulationRequest),'Expected frozen SimulationRequest; no fake Fit')
    processes,axes=transport_points(request.model.model_id,request.model.geometry,request.given_parameters,given=True)
    from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
    units={p.name:(request.signal_unit if p.unit=='signal' else p.unit) for p in DEFAULT_MODEL_REGISTRY.get(request.model.model_id).parameters}
    parameter_evidence=dict(estimated_parameters={},fixed_parameters={},given_parameters=dict(request.given_parameters),parameter_units=units,
        coordinates={},bounds={},fixed_D_sources={},boundary=request.boundary.to_dict(),source_label=request.name,native_status=None,N=len(request.time_s))
    return TransportSummary(source=None,source_kind='simulation_given',simulation_request_sha256=request.content_sha256,
        model_id=request.model.model_id.value,geometry=request.model.geometry,processes=processes,axes=axes,
        raw_indices=tuple(range(len(request.time_s))),raw_count=len(request.time_s),parent_reliability='not_applicable',reliability_reasons=(),
        conditions=(*CONDITIONS,'Given simulation parameters, not fitted estimates or statistical reliability.',
                    'Parameter origin: '+request.parameter_origin,'Input boundary: '+request.boundary.mode),surface_limit=.03,diffusion_limit=30.,
        interval_reason='N/A: given parameters have no fitted joint uncertainty.',parameter_evidence=parameter_evidence)


def derive_local_intervals(summary,fit,record):
    """Conditional log-ratio delta intervals from already selected R06 evidence.

    ln(D/k) gradient in full parent coordinates preserves the D,k cross term;
    nuisance derivatives are zero. log in FitResult/R06 means log10, not ln.
    No marginal endpoint arithmetic and no confidence/gate recomputation.
    """
    from pyecr.services.local_intervals import LocalIntervalRecord
    require(isinstance(record,LocalIntervalRecord),'Unsupported derived interval method: selected bound R06 joint covariance required')
    require(summary.source is not None,'Given simulation is not a CI parent')
    record.binding.assert_ci_source(replace(summary.source,diagnostics_sha256=None))
    require(record.binding.source.fit_sha256==canonical_sha256(fit.to_dict()),'Derived interval/Fit identity conflict')
    # Revalidate saved endpoints/C/hash via the existing reader; no estimator run.
    record=LocalIntervalRecord.from_dict(record.to_dict());e=record.evidence
    evidence=dict(method='local_log_ratio_delta_student_t',parent_record=record.to_dict(),
        coordinate_definition='R06 log means log10; propagated coordinate is ln(D/k)',
        conditional='Fixed geometry, boundary and fixed parameters; no fixed-D uncertainty propagation.',
        confidence_level=e['confidence_level'],df=e['df'],gradients={},log_variances={},
        gate_policy_id=record.binding.gate_policy_id,gate_confidence_level=.95,coverage_calibrated=False)
    if summary.parent_reliability!='supported':
        return replace(summary,joint_evidence=evidence,interval_reason='N/A: parent reliability is not supported by the original 95% policy.')
    order=tuple(e['parameter_order']);p=len(order)
    reasons=[];intervals=[]
    qualified=e['status'] in ('completed','partial') and e['rank']==p and e['df']>0 and e['covariance'] is not None
    if not qualified:return replace(summary,joint_evidence=evidence,interval_reason='N/A: selected covariance is unavailable or rank/df unqualified.')
    cov=np.asarray(e['covariance'],dtype=float)
    require(cov.shape==(p,p) and np.all(np.isfinite(cov)),'Joint covariance shape/finite conflict')
    require(np.allclose(cov,cov.T,rtol=1e-12,atol=0),'Joint covariance is not symmetric')
    require(float(np.min(np.linalg.eigvalsh(cov)))>=0,'Joint covariance is not positive semidefinite')
    require(e['jacobian_sha256']==canonical_sha256(fit.jacobian.tolist()),'Joint covariance/Jacobian identity conflict')
    require(e['raw_indices']==summary.raw_indices or tuple(e['raw_indices'])==summary.raw_indices,'Joint covariance full-row conflict')
    for proc in summary.processes:
        if proc.half_length_cm is None:reasons.append(proc.process+': '+proc.reason);continue
        why=''
        for name,value,sign,unit in ((proc.D_name,proc.D_cm2_s,1.,'cm^2/s'),(proc.k_name,proc.k_cm_s,-1.,'cm/s')):
            if name in order:
                if name not in e['intervals']:why='N/A: nonregular/unavailable '+name;break
                require(e['units'][name]==unit,'Derived interval parameter-unit conflict')
            else:
                require(name in e['fixed_parameters'] and e['fixed_parameters'][name]==value,'Derived fixed-parameter identity conflict')
        if why:reasons.append(proc.process+': '+why);continue
        grad=np.asarray(log_ratio_gradient(order,e['coordinates'],{proc.D_name:proc.D_cm2_s,proc.k_name:proc.k_cm_s},proc.D_name,proc.k_name))
        variance=float(grad@cov@grad)
        require(np.isfinite(variance) and variance>=0,'Derived log-ratio variance invalid')
        evidence['gradients'][proc.process]=grad.tolist();evidence['log_variances'][proc.process]=variance
        values=[('half_length',None,proc.half_length_cm,'cm'),('slab_full_thickness',None,proc.slab_full_thickness_cm,'cm')]
        values.extend(('Bi',a.axis,a.value,'1') for a in summary.axes if a.process==proc.process and a.value is not None)
        for quantity,axis,value,unit in values:
            try:endpoints=local_interval_bounds(value,variance,e['confidence_level'],e['df'])
            except (ValueError,OverflowError):reasons.append(proc.process+': interval overflow/underflow');continue
            intervals.append(TransportInterval(process=proc.process,quantity=quantity,axis=axis,estimate=value,
                lower=float(endpoints[0]),upper=float(endpoints[1]),unit=unit,confidence_level=e['confidence_level']))
    return replace(summary,intervals=tuple(intervals),interval_reason='; '.join(reasons) if reasons else '' if intervals else 'N/A: no paired process.',joint_evidence=evidence)


def save_transport_summary(summary,path):
    require(isinstance(summary,TransportSummary),'Expected immutable transport summary')
    path=Path(path);require(not path.exists(),'Transport destination exists; choose a new file')
    value=dict(version=VERSION,payload=summary.to_dict(),point_id=summary.point_id,summary_id=summary.content_sha256)
    value['sha256']=canonical_sha256(value);stage=path.with_name(path.name+'.'+uuid4().hex+'.staging')
    try:
        stage.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        require(read_transport_summary(stage.read_text(encoding='utf-8'))==summary,'Transport staging roundtrip conflict');stage.rename(path)
    finally:
        if stage.exists():stage.unlink()
    return path


def read_transport_summary(text):
    value=strict_json_loads(text);exact_fields(value,('version','payload','point_id','summary_id','sha256'),'transport sidecar')
    require(value['version']==VERSION,'Unknown transport sidecar version')
    require(value['sha256']==canonical_sha256({k:v for k,v in value.items() if k!='sha256'}),'Transport sidecar hash conflict')
    result=TransportSummary.from_dict(value['payload'])
    require(result.point_id==value['point_id'] and result.content_sha256==value['summary_id'],'Transport summary identity conflict')
    return result
