"""Persistent accepted-iterate evidence, separate from transient FitProgressEvent.

Only validates recorded evidence. No callback, counters, stopping or fitting runs.
"""
from __future__ import annotations
from dataclasses import dataclass,fields
from math import isclose
from typing import ClassVar,get_type_hints
from .common import Contract, integer, number, require, strings, vector, text
from .fit_controls import FitControlBinding, CHANGE_VERSION

class _CounterContract(Contract):
    """v1 remains strict; v2 distinguishes native attempts from real evaluations.

    SciPy VectorFunction may reuse a cached residual at identical coordinates.
    Native nfev therefore need not be <= instrumented residual evaluations.
    All R01 rho, units, budgets, accepted-event and native-state rules remain.
    """
    def __post_init__(self):
        require(integer(self.schema_version,'schema_version',minimum=1) in {1,2},f'unsupported {self.KIND} schema_version')

    @classmethod
    def from_dict(cls,value):
        from .common import exact_fields,decode
        exact_fields(value,['contract_kind',*[f.name for f in fields(cls)]],cls.__name__)
        require(value['contract_kind']==cls.KIND,'Trace contract kind conflict')
        require(integer(value['schema_version'],'schema_version',minimum=1) in {1,2},'Unknown counter trace version')
        hints=get_type_hints(cls)
        return cls(**{f.name:decode(hints[f.name],value[f.name]) for f in fields(cls)})


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedIterate(_CounterContract):
    KIND: ClassVar[str] = 'accepted_iterate'
    job_id: str
    start_id: str
    stage_id: str
    sequence: int
    event: str  # initial / accepted / terminal; never residual/Jacobian trials
    accepted_iteration: int
    nfev: int
    njev: int | None
    total_nfev: int
    total_njev: int | None
    total_accepted_iterations: int
    residual_calls: int
    forward_calls: int
    diagnostic_calls: int
    elapsed_s: float
    F: float | None
    SSE: float | None
    z: tuple[float,...]
    rho_F: float | None
    rho_z: float | None
    finite: bool
    bounds: tuple[int,...]  # -1 lower, 0 interior, +1 upper; coordinate order
    optimality: float | None = None
    stop_reason: str = 'none'
    native_status: int | None = None
    native_success: bool | None = None
    native_message: str | None = None
    finalization: str = 'pending'
    observation_count: int = 0
    full_observation_count: int = 0

    def __post_init__(self):
        _CounterContract.__post_init__(self)
        for k in ('job_id','start_id','stage_id'):text(getattr(self,k),k)
        integer(self.sequence,'sequence',minimum=1)
        require(self.event in {'initial','accepted','terminal'},'only initial/accepted/terminal events belong in this trace')
        for k in ('accepted_iteration','total_accepted_iterations','nfev','total_nfev','residual_calls','forward_calls','diagnostic_calls','observation_count','full_observation_count'):
            integer(getattr(self,k),k)
        for k in ('njev','total_njev'):
            if getattr(self,k) is not None:integer(getattr(self,k),k)
        require(self.total_nfev>=self.nfev,'total_nfev below local nfev')
        require(self.total_accepted_iterations>=self.accepted_iteration,'total accepted iterations below local count')
        if self.schema_version==1:require(self.residual_calls>=self.total_nfev,'actual residual calls cannot be below total native nfev')
        require(self.njev is None or self.total_njev is None or self.total_njev>=self.njev,'total_njev below local njev')
        object.__setattr__(self,'elapsed_s',number(self.elapsed_s,'elapsed_s',nonnegative=True))
        require(isinstance(self.finite,bool),'finite must be boolean')
        object.__setattr__(self,'z',vector(self.z,'dimensionless coordinates z'))
        require(isinstance(self.bounds,(tuple,list)),'bounds must be an array')
        require(all(type(v) is int and v in {-1,0,1} for v in self.bounds),'invalid bound state')
        object.__setattr__(self,'bounds',tuple(self.bounds))
        for k in ('F','SSE','rho_F','rho_z','optimality'):
            if getattr(self,k) is not None:object.__setattr__(self,k,number(getattr(self,k),k,nonnegative=True))
        require(self.event!='initial' or self.accepted_iteration==0,'initial point is not an accepted step')
        require(self.event!='accepted' or self.accepted_iteration>=1,'accepted iteration must be positive')
        require(self.event=='terminal' or self.finite,'nonfinite evaluation must terminate, never become accepted')
        if self.finite:
            require(self.F is not None and self.SSE is not None and len(self.z)>0,'finite evidence requires F/SSE/z')
            require(len(self.bounds)==len(self.z),'bounds/coordinate length conflict')
        else:
            require(self.F is None and self.SSE is None and not self.z and not self.bounds and self.rho_F is None and self.rho_z is None,
                    'nonfinite failure uses null metrics, not JSON NaN or fabricated numbers')
        require((self.rho_F is None)==(self.rho_z is None),'relative changes must be jointly available')
        require(self.event!='initial' or self.rho_F is None,'initial cannot advance a stability window')
        require(self.stop_reason in {'none','native_gtol','native_ftol','native_xtol','native_ftol_xtol','plateau','stage_plateau',
                                    'stagnated','budget','cancelled','nonfinite','failed'},'unknown stop reason')
        require(self.event=='terminal' or (self.stop_reason=='none' and self.native_status is None and self.native_success is None
                                         and self.native_message is None and self.finalization=='pending'),
                'only terminal events carry native termination/finalization')
        if self.event=='terminal':require(self.stop_reason!='none','terminal reason required')
        require((self.native_status is None)==(self.native_success is None),'native status/success must be jointly available')
        if self.native_status is not None:
            require(type(self.native_status) is int and self.native_status in {-2,-1,0,1,2,3,4},'unknown native status')
            require(type(self.native_success) is bool and self.native_success==(self.native_status>0),'native status/success conflict')
            text(self.native_message,'native_message')
        else:require(self.native_message is None,'native message without native status')
        native_map={'native_gtol':1,'native_ftol':2,'native_xtol':3,'native_ftol_xtol':4}
        if self.stop_reason in native_map:require(self.native_status==native_map[self.stop_reason],'native stop reason/status conflict')
        if self.stop_reason in {'plateau','stage_plateau','stagnated'}:
            require(self.native_status==-2,'callback plateau retains -2, never a fabricated positive native status')
        require(self.finalization in {'pending','candidate','full_observations_verified','rejected'},'unknown finalization state')
        require(self.observation_count<=self.full_observation_count,'observation count exceeds frozen full grid')
        if self.finalization=='full_observations_verified':
            require(self.event=='terminal' and self.native_success is True and self.finite
                    and self.observation_count==self.full_observation_count>0,'full finalization requires a finite positive native full-grid endpoint')


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedTrace(_CounterContract):
    KIND: ClassVar[str] = 'accepted_trace'
    job_id: str
    binding: FitControlBinding
    parameter_names: tuple[str,...]
    coordinate_kinds: tuple[str,...]  # log10 / linear
    coordinate_references: tuple[float,...]
    coordinate_scales: tuple[float,...]
    coordinate_units: tuple[str,...]  # original linear units; log10 is dimensionless
    F_ref: float
    F_ref_source: str
    objective_unit: str
    residual_unit: str
    loss: str
    f_scale: float
    weights_sha256: str | None
    change_definition: str
    record_budget: int
    emitted_count: int
    dropped_records: int
    records: tuple[AcceptedIterate,...]
    state: str = 'running'
    warnings: tuple[str,...] = ()
    noise_status: str = 'not_evaluated'
    reliability_status: str = 'not_evaluated'

    def __post_init__(self):
        _CounterContract.__post_init__(self)
        text(self.job_id,'job_id');require(isinstance(self.binding,FitControlBinding),'frozen control binding required')
        for k in ('parameter_names','coordinate_kinds','coordinate_units','warnings'):
            object.__setattr__(self,k,strings(getattr(self,k),k))
        n=len(self.parameter_names);require(n>0 and len(set(self.parameter_names))==n,'parameter names must be unique')
        require(len(self.coordinate_kinds)==len(self.coordinate_units)==n and set(self.coordinate_kinds)<={'log10','linear'},'coordinate metadata conflict')
        require(all(kind!='log10' or unit=='1' for kind,unit in zip(self.coordinate_kinds,self.coordinate_units)),'log10 coordinate must be dimensionless')
        for k in ('coordinate_references','coordinate_scales'):
            object.__setattr__(self,k,vector(getattr(self,k),k,positive=k=='coordinate_scales'))
            require(len(getattr(self,k))==n,'coordinate scale/reference length conflict')
        object.__setattr__(self,'F_ref',number(self.F_ref,'F_ref',positive=True))
        for k in ('F_ref_source','objective_unit','residual_unit'):text(getattr(self,k),k)
        require(self.loss in {'linear','soft_l1','huber','cauchy','arctan'},'unknown loss')
        object.__setattr__(self,'f_scale',number(self.f_scale,'f_scale',positive=True))
        if self.weights_sha256 is not None:
            from .common import digest
            digest(self.weights_sha256,'weights_sha256')
        require(self.change_definition==CHANGE_VERSION==self.binding.controls.termination.change_definition,'relative-change version conflict')
        integer(self.record_budget,'record_budget',minimum=1)
        require(self.record_budget==self.binding.controls.termination.budget.max_trace_records,'trace budget/binding conflict')
        integer(self.emitted_count,'emitted_count');integer(self.dropped_records,'dropped_records')
        require(isinstance(self.records,(tuple,list)) and all(isinstance(r,AcceptedIterate) for r in self.records),'typed trace records required')
        object.__setattr__(self,'records',tuple(self.records))
        require(all(r.schema_version<=self.schema_version for r in self.records),'Counter record version exceeds trace version')
        require(len(self.records)<=self.record_budget and len(self.records)+self.dropped_records==self.emitted_count,'trace storage count/budget conflict')
        require(self.state in {'running','completed','cancelled','failed','stale','history'},'unknown trace state')
        require(self.noise_status=='not_evaluated' and self.reliability_status=='not_evaluated',
                'optimizer trace cannot claim noise compatibility or statistical reliability; use separate evidence')
        require(not self.dropped_records or self.warnings,'truncated evidence requires an explicit warning')
        require(all(r.job_id==self.job_id for r in self.records),'trace/job identity conflict')
        sequences=[r.sequence for r in self.records]
        require(all(a<b for a,b in zip(sequences,sequences[1:])) and (not sequences or sequences[-1]<=self.emitted_count),'trace sequence conflict')
        if not self.dropped_records:require(sequences==list(range(1,len(self.records)+1)),'unreported missing trace records')
        previous={};last_global=None
        for r in self.records:
            require(not r.finite or len(r.z)==n,'coordinate/trace length conflict')
            policy=self.binding.controls.termination
            if r.stage_id=='final_polish':
                require(policy.final_polish=='once_full_native','final polish was not explicitly enabled in the frozen request')
                require(r.observation_count==r.full_observation_count>0,'polish requires the full observation grid')
                require(r.nfev<=policy.polish_max_nfev or r.stop_reason=='budget','polish exceeded its explicit budget')
            if policy.stage_order:
                require(r.stage_id in policy.stage_order or (r.stage_id=='final_polish' and policy.final_polish=='once_full_native'),
                        'undeclared stage ID')
            if self.loss=='linear' and self.weights_sha256 is None and r.finite:
                require(isclose(r.F,r.SSE/2.,rel_tol=1e-12,abs_tol=1e-14),'linear unweighted F must equal raw SSE/2')
            if last_global is not None:
                require(all(getattr(r,k)>=getattr(last_global,k) for k in ('total_nfev','total_accepted_iterations','residual_calls','forward_calls','diagnostic_calls','elapsed_s')),
                        'whole-job counters cannot reset at a new start/stage/polish')
                require(r.total_njev is None or last_global.total_njev is None or r.total_njev>=last_global.total_njev,
                        'whole-job njev cannot reset')
                if not self.dropped_records:
                    require(r.total_accepted_iterations==last_global.total_accepted_iterations+(r.event=='accepted'),
                            'only accepted events advance total iteration count')
            key=(r.start_id,r.stage_id);prev=previous.get(key)
            if prev is not None:
                require(r.nfev>=prev.nfev and r.accepted_iteration>=prev.accepted_iteration,'local counters cannot decrease')
                require(r.njev is None or prev.njev is None or r.njev>=prev.njev,'local njev cannot decrease')
                if not self.dropped_records and r.event=='accepted':
                    require(r.accepted_iteration==prev.accepted_iteration+1 and prev.event!='terminal','accepted step ordering conflict')
                if not self.dropped_records and r.event=='terminal':require(r.accepted_iteration==prev.accepted_iteration,'terminal cannot add an accepted step')
                if not self.dropped_records and prev.finite and r.finite and r.event=='accepted':
                    require(r.z!=prev.z,'duplicate evaluation is not an accepted step')
                    require(r.rho_F is not None,'accepted change evidence missing')
                    if r.rho_F is not None:
                        phi,old_phi=r.F/self.F_ref,prev.F/self.F_ref
                        rf=abs(phi-old_phi)/max(abs(old_phi),1e-12)
                        rz=max(abs(a-b) for a,b in zip(r.z,prev.z))/(1+max(abs(v) for v in prev.z))
                        require(isclose(r.rho_F,rf,rel_tol=1e-10,abs_tol=1e-14) and isclose(r.rho_z,rz,rel_tol=1e-10,abs_tol=1e-14),
                                'relative changes conflict with frozen definition/scales')
            elif not self.dropped_records:
                require(r.event=='initial' or (r.event=='terminal' and r.accepted_iteration==0),
                        'each optimizer call needs an initial record before accepted iterations')
            previous[key]=r;last_global=r
        polish_initials=[r for r in self.records if r.stage_id=='final_polish' and r.event=='initial']
        require(len(polish_initials)<=1,'final polish is once per job, not once per start')
        if self.state in {'completed','cancelled','failed'}:
            require(self.records and self.records[-1].event=='terminal','finished trace requires retained terminal evidence')
            last=self.records[-1]
            if self.state=='completed':require(last.native_success is True and last.finalization=='full_observations_verified','completed is numerical/full-grid only, not a gate waiver')
            if self.state=='cancelled':require(last.stop_reason=='cancelled','cancelled trace reason conflict')
            if self.state=='failed':require(last.finalization!='full_observations_verified','failed trace cannot be finalized')
            budget=self.binding.controls.termination.budget
            caps={'nfev':budget.max_nfev,'accepted_iteration':budget.max_iterations,'total_nfev':budget.max_total_nfev,
                  'total_njev':budget.max_njev,'residual_calls':budget.max_residual_calls,'forward_calls':budget.max_forward_calls,
                  'diagnostic_calls':budget.max_diagnostic_calls,'elapsed_s':budget.max_wall_seconds}
            exceeded=any(cap is not None and any(getattr(r,k) is not None and getattr(r,k)>cap for r in self.records)
                         for k,cap in caps.items())
            accepted_total=max(r.total_accepted_iterations for r in self.records)
            exceeded|=budget.max_total_iterations is not None and accepted_total>budget.max_total_iterations
            require(not exceeded or (self.state=='failed' and last.stop_reason=='budget' and bool(self.warnings)),
                    'budget overshoot must remain a warned budget failure, not completed')
