"""Selected-output local CI, isolated from the unchanged local95 gate paths.

No optimizer, prediction callable, numeric Jacobian or FitResult construction.
The only covariance/Student-t computation is the existing local estimator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from pyecr.contracts import ConfidencePolicy, FitControlBinding, FitControlSpec, SourceIdentity, TerminationPolicy
from pyecr.contracts.common import (canonical_sha256, exact_fields, integer, number,
                                    require, strict_json_loads)
from pyecr.contracts.fit_controls import SIX_MODELS
from pyecr.diagnostics.boundary_contact import bound_contact
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.uncertainty.local import ConfidenceInterval, estimate_local_uncertainty, student_t

VERSION = 'selected-local-ci/1'


def _freeze(value):
    if isinstance(value, dict):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value, list):return tuple(_freeze(v) for v in value)
    return value


def _plain(value):
    if hasattr(value,'items'):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class LocalIntervalRecord:
    """Source-bound serializable evidence around existing ConfidenceInterval values.

    This is a sidecar, not a replacement local covariance or physical Fit type.
    Failed/nonregular rows have explicit reasons and null endpoints.
    """
    binding: FitControlBinding
    fit_generation: int
    evidence: object

    def __post_init__(self):
        require(isinstance(self.binding,FitControlBinding) and self.binding.ci_request_id is not None,'local CI requires a parent Fit binding')
        require(self.binding.controls.output_ci.method=='local_covariance','Unsupported CI method in local service')
        integer(self.fit_generation,'fit_generation')
        e=_plain(self.evidence)
        exact_fields(e,('version','method','confidence_level','status','reason','N','p_free','df','parameter_order','coordinates','units',
            'intervals','unavailable','fixed_parameters','raw_indices','jacobian_sha256','rank','condition_number','residual_variance',
            'covariance','conditions','warnings'),'local CI evidence')
        require(e['version']==VERSION and e['method']=='local_covariance','Unknown local CI version/method')
        require(number(e['confidence_level'],'confidence_level')==self.binding.controls.output_ci.confidence_level,'CI level conflicts with request')
        n=integer(e['N'],'N');p=integer(e['p_free'],'p_free')
        require(e['df']==n-p and len(e['parameter_order'])==p and len(set(e['parameter_order']))==p,'CI N/p/df/order conflict')
        require(set(e['coordinates'])==set(e['units'])==set(e['parameter_order']),'CI coordinate/unit/order conflict')
        require(all(s in {'log','linear'} for s in e['coordinates'].values()),'Unknown coordinate; log in FitResult means log10')
        require(all(isinstance(u,str) and u.strip() for u in e['units'].values()),'Missing physical parameter unit')
        require(set(e['intervals'])<=set(e['parameter_order']) and not (set(e['fixed_parameters'])&set(e['parameter_order'])),'Fixed/free interval conflict')
        require(e['status'] in {'completed','partial','unavailable','unsupported'},'Unknown local CI status')
        require(e['status'] not in {'unavailable','unsupported'} or (bool(e['reason']) and not e['intervals']),'Unavailable CI needs a reason and no endpoints')
        require(e['status']!='completed' or len(e['intervals'])==p,'Completed CI cannot hide unavailable free parameters')
        require(set(e['intervals'])|set(e['unavailable'])==set(e['parameter_order']) and not(set(e['intervals'])&set(e['unavailable'])),'Every free parameter needs CI or N/A reason')
        require(len(e['raw_indices'])==n and e['raw_indices']==sorted(set(e['raw_indices'])),'CI full row mapping conflict')
        for i in e['raw_indices']:integer(i,'raw index')
        require(e['conditions']['gate_policy_id']==self.binding.gate_policy_id and e['conditions']['gate_confidence_level']==.95
            and e['conditions']['model_id']==self.binding.source.model_id,'CI/gate/model policy conflict')
        for name,row in e['intervals'].items():
            exact_fields(row,ConfidenceInterval.__dataclass_fields__,'ConfidenceInterval')
            interval=ConfidenceInterval(**row);self.binding.controls.output_ci.assert_result(interval)
            require(interval.parameter==name and interval.space==e['coordinates'][name],'CI coordinate/parameter conflict')
            for k,v in row.items():
                if k not in {'parameter','space'}:number(v,'CI '+k)
            require(interval.lower<=interval.estimate<=interval.upper and interval.coordinate_lower<=interval.coordinate_estimate<=interval.coordinate_upper,'Invalid CI endpoint ordering')
            require(e['df']>0 and e['rank']==p,'Intervals require qualified rank and positive df')
            # Check the saved endpoints with the SAME Student-t primitive used
            # by local.py. A new C label cannot be pasted onto old endpoints.
            half=float(student_t.ppf(.5+e['confidence_level']/2,e['df']))*interval.coordinate_standard_error
            expected=np.array([interval.coordinate_estimate-half,interval.coordinate_estimate+half])
            require(np.allclose([interval.coordinate_lower,interval.coordinate_upper],expected,rtol=1e-12,atol=1e-14),'CI endpoints/Student-t confidence conflict')
            physical=10.**expected if interval.space=='log' else expected
            require(np.allclose([interval.lower,interval.upper],physical,rtol=1e-12,atol=0.),'CI physical coordinate recovery conflict')
        canonical_sha256(e)  # rejects nonfinite JSON or non-replayable objects
        object.__setattr__(self,'evidence',_freeze(e))

    def to_dict(self):
        value=dict(schema_version=1,binding=self.binding.to_dict(),fit_generation=self.fit_generation,evidence=_plain(self.evidence))
        return {**value,'record_sha256':canonical_sha256(value)}

    def to_json(self):return json.dumps(self.to_dict(),ensure_ascii=False,allow_nan=False,indent=2)

    @classmethod
    def from_dict(cls,value):
        exact_fields(value,('schema_version','binding','fit_generation','evidence','record_sha256'),'LocalIntervalRecord')
        require(integer(value['schema_version'],'schema_version',minimum=1)==1,'Unknown local interval schema version')
        require(canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'})==value['record_sha256'],'Local interval record hash conflict')
        return cls(FitControlBinding.from_dict(value['binding']),value['fit_generation'],value['evidence'])

    @classmethod
    def from_json(cls,text):return cls.from_dict(strict_json_loads(text))


def save_local_intervals(record,path):
    """Explicit new JSON sidecar only. Verify staging; never overwrite a file."""
    path=Path(path)
    require(isinstance(record,LocalIntervalRecord),'Expected source-bound local CI record')
    require(not path.exists(),'Interval destination already exists; choose a new file')
    staging=path.with_name(path.name+'.'+uuid4().hex+'.staging')
    try:
        staging.write_text(record.to_json()+'\n',encoding='utf-8')
        require(LocalIntervalRecord.from_json(staging.read_text(encoding='utf-8')).to_dict()==record.to_dict(),'Staging interval roundtrip failed')
        # Windows rename fails if another writer created the destination.
        staging.rename(path)
    finally:
        if staging.exists():staging.unlink()
    return path


def _metadata(fit):
    for name in ('fixed_d','surface_k','dual_surface_k'):
        if name in fit.diagnostics:return fit.diagnostics[name]
    return {}


def recompute_local_intervals(dataset,recipe,config,fit,binding,*,fit_generation=0):
    """All observations of this exact parent; no fallback finite differences.

    Both legacy residual signs are verified, never rewritten. Log means log10
    in FitResult; ln or unknown optimizer spaces are Unsupported.
    """
    current=SourceIdentity.capture(dataset,recipe,config=config,fit=fit)
    binding.assert_ci_source(current)
    require(binding.controls.output_ci.method=='local_covariance','Unsupported: local service does not compute advanced intervals')
    n=len(dataset.processed_signal);names=tuple(fit.params);p=len(names);meta=_metadata(fit)
    require(n==len(fit.y_fit) and n==int(np.count_nonzero(dataset.mask)),'Full observation/Fit row mismatch')
    diff=dataset.processed_signal-fit.y_fit
    require(np.allclose(diff,fit.residuals,rtol=1e-10,atol=1e-12) or np.allclose(-diff,fit.residuals,rtol=1e-10,atol=1e-12),'Parent residual/dataset identity conflict')
    require(np.isclose(np.dot(fit.residuals,fit.residuals),fit.SSE,rtol=1e-10,atol=1e-12),'Parent SSE/residual conflict')
    descriptor=DEFAULT_MODEL_REGISTRY.get(config.model_id)
    units={d.name:(str(dataset.metadata.get('signal_unit','signal')) if d.unit=='signal' else d.unit) for d in descriptor.parameters}
    require(set(names)<=set(units),'Parent parameter/model identity conflict')
    declared=fit.diagnostics.get('analysis_artifacts',{}).get('model_name',fit.diagnostics.get('model_id'))
    require(declared==config.model_id.value,'Selected Fit/model identity conflict; do not borrow another model intervals')
    original=fit.diagnostics.get('fitting_randomness',{}).get('effective_config')
    if original:
        from pyecr.config import AnalysisConfig
        from pyecr.fitting.reproducibility import config_hash
        require(config_hash(AnalysisConfig.from_dict(original))==current.config_sha256,'Parent original Fit/config identity conflict')
    for key,actual in (('dataset_sha256',current.dataset_sha256),('config_sha256',current.config_sha256)):
        require(key not in meta or meta[key]==actual,'Parent '+key+' conflict')
    fixed={**dict(config.model.fixed_parameters),**dict(meta.get('fixed_parameters',{}))}
    # M0/M1 normalized observation defaults can be fixed by the old fitter.
    for name in ('amplitude','baseline'):
        if name not in names and name not in fixed and name in units and fit.diagnostics.get('normalized_data'):
            fixed[name]=1. if name=='amplitude' else 0.
    require(not(set(names)&set(fixed)),'Parent fixed/free parameter conflict')
    threshold=config.model_specific.get('single_ecr',{}).get('condition_threshold',1e12)
    e=dict(version=VERSION,method='local_covariance',confidence_level=binding.controls.output_ci.confidence_level,
        status='unavailable',reason='',N=n,p_free=p,df=n-p,parameter_order=list(names),coordinates=dict(fit.param_space),units={k:units[k] for k in names},
        intervals={},unavailable={},fixed_parameters=fixed,raw_indices=np.flatnonzero(dataset.mask).tolist(),jacobian_sha256=None,
        rank=None,condition_number=None,residual_variance=None,covariance=None,
        conditions=dict(scope=meta.get('ci_scope','conditional_local_linearization'),fixed_inputs=meta.get('ci_conditions',{}),
            geometry=config.model.geometry.to_dict(),boundary=config.boundary.to_dict(),loss=config.fit.loss,
            covariance='s2 * pinv(J.T @ J)',variance='SSE/(N-p_free)',quantile='Student-t((1+C)/2, df)',
            coordinate_definition='FitResult log = log10; linear = physical',jacobian_source='stored complete parent Fit Jacobian only',
            coordinate_units={name:('decade' if fit.param_space[name]=='log' else units[name]) for name in names},
            condition_threshold=threshold,svd_rcond=None,weighting='unweighted linear least squares; iid constant-variance local approximation',
            gate_policy_id=binding.gate_policy_id,gate_confidence_level=.95,gate_unchanged=True,
            model_id=config.model_id.value,propagates_fixed_input_uncertainty=False),warnings=[])
    reason='';unsupported=False
    if config.model_id.value not in SIX_MODELS:reason='Unsupported physical model';unsupported=True
    elif config.fit.loss!='linear':reason='Unsupported: selected local covariance has not been validated for robust loss';unsupported=True
    elif config.boundary.mode not in descriptor.supported_boundary_modes:reason='Unsupported boundary/model combination';unsupported=True
    elif config.boundary.mode!='ideal_step' and config.model_id.value not in {'M0_SINGLE_D','M1_SINGLE_DK'}:reason='Unsupported conditional boundary path';unsupported=True
    elif any(v not in {'log','linear'} for v in fit.param_space.values()):reason='Unsupported optimizer coordinate (ln is not log10)';unsupported=True
    elif fit.jacobian is None:reason='N/A: parent Fit has no stored full Jacobian'
    elif n<=p or p==0:reason='N/A: local covariance requires N > p_free > 0'
    elif fit.exit_status<=0:reason='N/A: parent optimizer did not complete successfully'
    if 'fixed_d' in config.model_specific and 'fixed_d' not in fit.diagnostics:
        reason='Unsupported: conditional fixed-D parent adapter evidence is missing';unsupported=True
    if 'fixed_d' in fit.diagnostics:
        from pyecr.services.fixed_d import validate_fixed_diffusion
        validate_fixed_diffusion(config)
        if config.model_id.value not in {'M1_SINGLE_DK','M3_DUAL_HO_DK'} or config.boundary.mode!='ideal_step':
            reason='Unsupported fixed-D path';unsupported=True
    if not reason:
        # Keep the exact existing local estimator and its existing threshold.
        local=estimate_local_uncertainty(fit,confidence_level=binding.controls.output_ci.confidence_level,condition_threshold=threshold)
        binding.controls.output_ci.assert_result(local)
        e.update(jacobian_sha256=canonical_sha256(local.jacobian.tolist()),rank=local.rank,
            condition_number=local.condition_number if np.isfinite(local.condition_number) else None,
            residual_variance=local.residual_variance if np.isfinite(local.residual_variance) else None,
            covariance=[[float(v) if np.isfinite(v) else None for v in row] for row in local.covariance],warnings=list(local.warnings))
        if not local.reliable_separation:reason='N/A: '+('; '.join(local.warnings))
        bounds=meta.get('parameter_bounds',config.parameter_bounds)
        for name in names:
            why=reason
            if not why and meta.get('ci_availability',{}).get(name,meta.get('ci_available',True)) is False:
                why='N/A: '+str(meta.get('ci_unavailable_reason','Parent nonregular CI condition'))
            if not why and bound_contact(fit.params[name],bounds.get(name),is_log=fit.param_space[name]=='log')[0]:
                why='N/A: parent parameter touches a numerical bound'
            row=asdict(local.confidence_intervals[name])
            if not why and not all(np.isfinite(v) for k,v in row.items() if k not in {'parameter','space'}):why='N/A: local interval is nonfinite at this C'
            if why:e['unavailable'][name]=why
            else:e['intervals'][name]=row
    if reason:e['unavailable']={name:reason for name in names}
    e['reason']=reason or ('Some local intervals unavailable; inspect parameter reasons' if e['unavailable'] else '')
    e['status']='unsupported' if unsupported else 'completed' if len(e['intervals'])==p and p else 'partial' if e['intervals'] else 'unavailable'
    return LocalIntervalRecord(binding,fit_generation,e)


def _current_source(session):
    require(session.dataset is not None and session.current_analysis_config is not None and session.fit_result is not None,'N/A: Import, Apply and explicitly run Fit first')
    require(not session.fit_stale and session.fit_revision==session.current_revision,'N/A: parent Fit is Stale')
    return SourceIdentity.capture(session.dataset,session.preprocessing.config,config=session.current_analysis_config,fit=session.fit_result)


class LocalIntervalStore:
    """Freeze/compute/publish separately so late evidence cannot change parents."""
    def __init__(self):self.requests={};self.history=[];self.failures={};self.active=None;self.latest=None

    def fail(self,job,reason):
        request=self.requests[job]
        self.failures[job]=dict(binding=request[4].to_dict(),fit_generation=request[5],reason=str(reason))

    def begin(self,session,policy):
        require(isinstance(policy,ConfidencePolicy) and policy.method=='local_covariance','Unsupported CI policy')
        source=_current_source(session)
        require(source.model_id in SIX_MODELS,'Unsupported: selected local CI supports the six physical models only')
        binding=FitControlBinding.capture(source,FitControlSpec(output_ci=policy,
            termination=TerminationPolicy.legacy_for(session.current_analysis_config)))
        job=uuid4().hex
        self.requests[job]=(session.dataset,session.preprocessing.config,session.current_analysis_config,session.fit_result,binding,session.fit_generation)
        self.active=job
        return job

    def compute(self,job):
        ds,recipe,config,fit,binding,generation=self.requests[job]
        return recompute_local_intervals(ds,recipe,config,fit,binding,fit_generation=generation)

    def publish(self,job,record,session):
        request=self.requests[job]
        require(record.binding==request[4] and record.fit_generation==request[5],'Late interval record/request identity conflict')
        current=job==self.active
        try:
            record.binding.assert_ci_source(_current_source(session))
            current=current and record.fit_generation==session.fit_generation
        except (ValueError,TypeError):current=False
        self.history.append(dict(job_id=job,record=record,current_at_publication=current))
        if current:self.latest=record
        return current

    def current(self,session):
        if self.latest is None:return None
        try:self.latest.binding.assert_ci_source(_current_source(session))
        except (ValueError,TypeError):return None
        return self.latest if self.latest.fit_generation==session.fit_generation else None
