"""Immutable source-bound transport evidence; no Fit or DCT result substitute."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from pyecr.config import SampleGeometry,_freeze_json
from .common import (Contract,SourceIdentity,require,number,text,strings,rows,digest,canonical_sha256)


@dataclass(frozen=True,slots=True,kw_only=True)
class TransportProcess(Contract):
    KIND:ClassVar[str]='transport_process'
    process:str
    label:str
    D_name:str|None
    k_name:str|None
    D_cm2_s:float|None
    k_cm_s:float|None
    D_origin:str
    k_origin:str
    half_length_cm:float|None
    slab_full_thickness_cm:float|None
    reason:str

    def __post_init__(self):
        Contract.__post_init__(self)
        for k in ('process','label','D_origin','k_origin'):text(getattr(self,k),k)
        for k in ('D_name','k_name'):
            if getattr(self,k) is not None:text(getattr(self,k),k)
        for k in ('D_cm2_s','k_cm_s','half_length_cm','slab_full_thickness_cm'):
            if getattr(self,k) is not None:number(getattr(self,k),k,positive=True)
        ready=self.half_length_cm is not None
        require(ready==(self.slab_full_thickness_cm is not None),'Transport size availability conflict')
        require(not ready or (self.D_cm2_s is not None and self.k_cm_s is not None and not self.reason),'Transport sizes need finite D/k')
        require(ready or bool(self.reason),'Unavailable process needs a reason')


@dataclass(frozen=True,slots=True,kw_only=True)
class TransportAxis(Contract):
    KIND:ClassVar[str]='transport_axis'
    process:str
    axis:str
    full_dimension_cm:float
    value:float|None
    regime:str
    reason:str

    def __post_init__(self):
        Contract.__post_init__(self);text(self.process,'process')
        require(self.axis in 'xyz' and len(self.axis)==1,'Unknown transport axis')
        number(self.full_dimension_cm,'full_dimension_cm',positive=True)
        if self.value is not None:number(self.value,'Bi',positive=True)
        require(self.regime in ('surface_limited','mixed_control','diffusion_limited','unavailable'),'Unknown regime')
        require((self.value is None)==(self.regime=='unavailable'),'Bi/regime availability conflict')
        require(self.value is not None or bool(self.reason),'Unavailable axis needs a reason')


@dataclass(frozen=True,slots=True,kw_only=True)
class TransportInterval(Contract):
    KIND:ClassVar[str]='transport_interval'
    process:str
    quantity:str
    axis:str|None
    estimate:float
    lower:float
    upper:float
    unit:str
    confidence_level:float
    method:str='local_log_ratio_delta_student_t'

    def __post_init__(self):
        Contract.__post_init__(self);text(self.process,'process')
        require(self.quantity in ('half_length','slab_full_thickness','Bi'),'Unknown derived quantity')
        require((self.axis in ('x','y','z')) if self.quantity=='Bi' else self.axis is None,'Derived interval axis conflict')
        for k in ('estimate','lower','upper'):number(getattr(self,k),k,positive=True)
        require(self.lower<=self.estimate<=self.upper,'Derived interval endpoint order')
        require(0<number(self.confidence_level,'C')<1,'Derived C must be within (0,1)')
        require(self.unit==('1' if self.quantity=='Bi' else 'cm'),'Derived interval unit conflict')
        require(self.method=='local_log_ratio_delta_student_t','Unsupported derived CI method')


@dataclass(frozen=True,slots=True,kw_only=True)
class TransportSummary(Contract):
    KIND:ClassVar[str]='transport_summary'
    source:SourceIdentity|None
    source_kind:str
    simulation_request_sha256:str|None
    model_id:str
    geometry:SampleGeometry
    processes:tuple[TransportProcess,...]
    axes:tuple[TransportAxis,...]
    raw_indices:tuple[int,...]
    raw_count:int
    parent_reliability:str
    reliability_reasons:tuple[str,...]
    conditions:tuple[str,...]
    surface_limit:float
    diffusion_limit:float
    parameter_evidence:Mapping[str,object]
    intervals:tuple[TransportInterval,...]=()
    interval_reason:str='No selected qualified joint covariance record.'
    joint_evidence:Mapping[str,object]|None=None
    algorithm_version:str='transport-regime/1'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.algorithm_version=='transport-regime/1','Unknown transport algorithm version')
        require(self.source_kind in ('fit_estimate','conditional_fixed_d','simulation_given'),'Unknown transport source kind')
        require(self.parent_reliability in ('pending','supported','unreliable','not_applicable'),'Unknown transport parent reliability')
        if self.source_kind=='simulation_given':
            require(self.source is None and self.simulation_request_sha256 is not None and self.parent_reliability=='not_applicable','Given transport must not masquerade as Fit')
            digest(self.simulation_request_sha256,'simulation_request_sha256')
        else:
            require(isinstance(self.source,SourceIdentity) and self.source.fit_id is not None and self.source.model_id==self.model_id,'Transport requires actual matching parent Fit')
            require(self.simulation_request_sha256 is None and self.parent_reliability!='not_applicable','Fit/given source conflict')
        require(isinstance(self.geometry,SampleGeometry),'Transport needs complete cm geometry')
        self.geometry.validate()
        require(number(self.surface_limit,'surface_limit',positive=True)<number(self.diffusion_limit,'diffusion_limit',positive=True),'Invalid heuristic regime limits')
        for k,t in (('processes',TransportProcess),('axes',TransportAxis),('intervals',TransportInterval)):
            v=tuple(getattr(self,k));require(all(isinstance(x,t) for x in v),'Invalid transport '+k);object.__setattr__(self,k,v)
        for k in ('conditions','reliability_reasons'):object.__setattr__(self,k,strings(getattr(self,k),k))
        from .common import integer
        integer(self.raw_count,'raw_count',minimum=1);object.__setattr__(self,'raw_indices',rows(self.raw_indices,self.raw_count))
        require(len(self.processes)>0 and len({p.process for p in self.processes})==len(self.processes),'Transport process identity conflict')
        from pyecr.diagnostics.transport import DEFINITIONS
        require(self.model_id in {m.value for m in DEFINITIONS},'Unsupported transport summary model_id')
        from pyecr.models.ids import ModelID
        expected=DEFINITIONS[ModelID(self.model_id)]
        require(tuple((p.process,p.D_name,p.k_name) for p in self.processes)==tuple((p,d,k) for p,_,d,k in expected),'Transport model/process role conflict')
        require(isinstance(self.parameter_evidence,Mapping),'Missing transport parameter evidence')
        from .common import exact_fields
        exact_fields(self.parameter_evidence,('estimated_parameters','fixed_parameters','given_parameters','parameter_units',
            'coordinates','bounds','fixed_D_sources','boundary','source_label','native_status','N'),'transport parameter evidence')
        pe=self.parameter_evidence
        require(not (set(pe['estimated_parameters'])&set(pe['fixed_parameters'])),'Transport parameter origin conflict')
        require(not pe['estimated_parameters'] if self.source_kind=='simulation_given' else not pe['given_parameters'],'Given/estimated parameters cannot be mixed')
        values={**pe['given_parameters'],**pe['fixed_parameters'],**pe['estimated_parameters']}
        for p in self.processes:
            for name,value,unit in ((p.D_name,p.D_cm2_s,'cm^2/s'),(p.k_name,p.k_cm_s,'cm/s')):
                if value is not None:require(values.get(name)==value and pe['parameter_units'].get(name)==unit,'Transport parameter value/unit conflict')
        require(pe['N']==len(self.raw_indices),'Transport observation count conflict')
        object.__setattr__(self,'parameter_evidence',_freeze_json(pe,path='transport_parameter_evidence'))
        # One implementation also verifies serialized point/axis arithmetic.
        from pyecr.diagnostics.transport import point_values
        for p in self.processes:
            values=point_values(p.D_cm2_s,p.k_cm_s,self.geometry,self.surface_limit,self.diffusion_limit)
            require((p.half_length_cm,p.slab_full_thickness_cm)==values[:2],'Transport size/input conflict')
            expected=tuple((p.process,*v) for v in values[2])
            actual=tuple((a.process,a.axis,a.full_dimension_cm,a.value,a.regime) for a in self.axes if a.process==p.process)
            require(actual==expected,'Transport active-axis/value conflict')
        require(set(a.process for a in self.axes)==set(p.process for p in self.processes),'Unknown process in transport axes')
        if self.joint_evidence is not None:
            require(isinstance(self.joint_evidence,Mapping),'Joint evidence must be serializable')
            object.__setattr__(self,'joint_evidence',_freeze_json(self.joint_evidence,path='transport_joint_evidence'))
        require(not self.intervals or self.joint_evidence is not None,'Derived intervals require joint evidence')
        if self.parent_reliability!='supported':require(not self.intervals,'Unqualified parent cannot acquire derived intervals')
        if self.intervals:
            from dataclasses import replace
            import numpy as np
            from pyecr.services.local_intervals import LocalIntervalRecord
            from pyecr.diagnostics.transport import log_ratio_gradient,local_interval_bounds
            record=LocalIntervalRecord.from_dict(self.joint_evidence['parent_record']);e=record.evidence
            record.binding.assert_ci_source(replace(self.source,diagnostics_sha256=None))
            require(e['confidence_level']==self.joint_evidence['confidence_level'] and e['df']==self.joint_evidence['df'],'Derived confidence/df evidence conflict')
            require(e['rank']==len(e['parameter_order']) and e['df']>0,'Derived rank/df conflict')
            cov=np.asarray(e['covariance'],dtype=float);order=tuple(e['parameter_order'])
            require(cov.shape==(len(order),len(order)) and np.all(np.isfinite(cov)),'Derived covariance shape/finite conflict')
            require(len({(x.process,x.quantity,x.axis) for x in self.intervals})==len(self.intervals),'Duplicate derived interval')
            for row in self.intervals:
                proc=next((p for p in self.processes if p.process==row.process),None)
                require(proc is not None and proc.half_length_cm is not None,'Derived process conflict')
                grad=log_ratio_gradient(order,e['coordinates'],{proc.D_name:proc.D_cm2_s,proc.k_name:proc.k_cm_s},proc.D_name,proc.k_name)
                require(tuple(self.joint_evidence['gradients'][row.process])==grad,'Derived coordinate gradient conflict')
                variance=float(np.asarray(grad)@cov@np.asarray(grad))
                require(np.isclose(variance,self.joint_evidence['log_variances'][row.process],rtol=1e-12,atol=0),'Derived covariance/cross-term conflict')
                point=proc.half_length_cm if row.quantity=='half_length' else proc.slab_full_thickness_cm if row.quantity=='slab_full_thickness' else next((a.value for a in self.axes if a.process==row.process and a.axis==row.axis),None)
                require(point is not None,'Derived interval axis is inactive or unavailable')
                require(row.estimate==point and row.confidence_level==e['confidence_level'],'Derived point/confidence label conflict')
                bounds=local_interval_bounds(point,variance,row.confidence_level,e['df'])
                require(np.allclose((row.lower,row.upper),bounds,rtol=1e-12,atol=0),'Derived interval endpoints/evidence conflict')
        canonical_sha256(self.to_dict())

    @property
    def point_id(self):
        return canonical_sha256({k:v for k,v in self.to_dict().items() if k not in ('intervals','interval_reason','joint_evidence')})

    def assert_source(self,source):
        require(self.source is not None,'Given transport has no parent Fit source')
        self.source.assert_matches(source)
