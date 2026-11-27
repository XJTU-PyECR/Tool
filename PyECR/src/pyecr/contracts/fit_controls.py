"""R01 immutable control metadata. No optimizer, interval calculation or dispatch.

Physical AnalysisConfig/SourceIdentity/FitResult serialization stays unchanged.
The explicit reader is for an optional sidecar; project integration is R13.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import ClassVar

from pyecr.fitting.solver_strategy import SolverBudget, SolverOptions
from .common import (Contract, ContractError, IdentityConflictError, SourceIdentity,
                     canonical_sha256, exact_fields, integer, number, require, strings, text)


GATE_VERSION = 'legacy95-gates/1'
CHANGE_VERSION = 'normalized-F-inf-z/1'
SIX_MODELS = frozenset({'M0_SINGLE_D','M1_SINGLE_DK','M2_DUAL_HO_D','M3_DUAL_HO_DK','M7_SINGLE_K','M8_DUAL_K'})


def parse_confidence_percent(value) -> float:
    """UI numeric percent, never a guessed fraction: 0.95 means 0.95 percent."""
    if value is None or (isinstance(value,str) and not value.strip()):
        return .95
    if isinstance(value,str):
        try:value=float(value.strip())
        except ValueError as exc:raise ContractError('Confidence (%) requires a numeric percent without a percent suffix') from exc
    c=number(value,'Confidence (%)')/100.
    require(0 < c < 1,'Confidence (%) must be strictly between 0 and 100')
    return c


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfidencePolicy(Contract):
    KIND: ClassVar[str] = 'confidence_policy'
    confidence_level: float = .95
    method: str = 'local_covariance'
    method_options_sha256: str | None = None

    def __post_init__(self):
        Contract.__post_init__(self)
        c=number(self.confidence_level,'confidence_level')
        require(0<c<1,'confidence_level must be strictly between 0 and 1')
        object.__setattr__(self,'confidence_level',c)
        require(self.method in {'local_covariance','residual_bootstrap','profile_likelihood'},'unsupported CI method')
        if self.method_options_sha256 is not None:
            from .common import digest
            digest(self.method_options_sha256,'method_options_sha256')
        require(self.method=='local_covariance' or self.method_options_sha256 is not None,
                'advanced CI requires frozen method/options/sample or profile identity')

    def assert_result(self,result):
        """Check actual existing result/interval types; never relabel endpoints."""
        from pyecr.uncertainty.local import ConfidenceInterval, LocalUncertaintyResult
        from pyecr.uncertainty.base import UncertaintyInterval, UncertaintyResult
        require(isinstance(result,(ConfidenceInterval,LocalUncertaintyResult,UncertaintyInterval,UncertaintyResult)),
                'use an existing uncertainty result/interval, not a replacement payload')
        actual=number(result.confidence_level,'result confidence_level')
        if actual!=self.confidence_level:
            raise IdentityConflictError('CI level conflict: existing endpoints cannot be relabelled')
        if isinstance(result,(ConfidenceInterval,LocalUncertaintyResult)):
            require(self.method=='local_covariance','local result cannot stand in for another CI method')
            intervals=result.confidence_intervals.values() if isinstance(result,LocalUncertaintyResult) else ()
        elif isinstance(result,UncertaintyResult):
            require(result.method.value==self.method,'CI method conflict')
            intervals=result.intervals.values()
        else:
            raise ContractError('generic interval has no method identity; validate its containing UncertaintyResult')
        for interval in intervals:
            require(isinstance(interval,(ConfidenceInterval,UncertaintyInterval)),'invalid nested uncertainty interval')
            if number(interval.confidence_level,'interval confidence_level')!=self.confidence_level:
                raise IdentityConflictError('CI result/endpoint level conflict; metadata cannot relabel old intervals')


@dataclass(frozen=True, slots=True, kw_only=True)
class ReliabilityPolicy(Contract):
    """Reference the old rules; do not fork or reimplement the gate thresholds."""
    KIND: ClassVar[str] = 'reliability_policy'
    confidence_level: float = .95
    policy_version: str = GATE_VERSION
    bound_policy_version: str = 'credibility-bound-proximity/1.0.0'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(number(self.confidence_level,'gate confidence_level')==.95,'legacy gate requires 95% independently of output CI')
        require(self.policy_version==GATE_VERSION and self.bound_policy_version=='credibility-bound-proximity/1.0.0',
                'unknown reliability policy version; no fallback')


@dataclass(frozen=True, slots=True, kw_only=True)
class TerminationPolicy(Contract):
    KIND: ClassVar[str] = 'termination_policy'
    mode: str = 'legacy'
    solver_options: SolverOptions = field(default_factory=SolverOptions)
    ftol: float | None = 1e-8
    xtol: float | None = 1e-8
    gtol: float | None = 1e-8
    objective_change_tolerance: float | None = None
    parameter_change_tolerance: float | None = None
    patience: int | None = None
    change_definition: str = CHANGE_VERSION
    budget: SolverBudget = field(default_factory=lambda:SolverBudget(5000,max_trace_records=10000))
    stage_handoff: bool = False
    stage_order: tuple[str,...] = ()
    final_polish: str = 'disabled'
    polish_max_nfev: int = 0
    polish_tolerances: tuple[float,float,float] | None = None

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.mode in {'legacy','monitored'},'unknown termination mode')
        require(isinstance(self.solver_options,SolverOptions),'reuse SolverOptions')
        require(number(self.solver_options.condition_limit,'condition_limit',positive=True)>1,'condition_limit must exceed 1')
        require(isinstance(self.budget,SolverBudget),'reuse SolverBudget')
        require(self.change_definition==CHANGE_VERSION,'unknown relative-change definition')
        for k in ('ftol','xtol','gtol'):
            if getattr(self,k) is not None:
                object.__setattr__(self,k,number(getattr(self,k),k,positive=True))
                require(getattr(self,k)>2.220446049250313e-16,f'{k} must exceed machine epsilon or be disabled with None')
        require(any(getattr(self,k) is not None for k in ('ftol','xtol','gtol')),'all native tolerances disabled')
        if self.solver_options.strategy=='lm_transformed_experimental':
            require(all(getattr(self,k) is not None for k in ('ftol','xtol','gtol')),'LM requires all native tolerances')
        require(isinstance(self.stage_handoff,bool),'stage_handoff must be boolean')
        object.__setattr__(self,'stage_order',strings(self.stage_order,'stage_order'))
        require(len(set(self.stage_order))==len(self.stage_order),'stage IDs must be unique')
        require('final_polish' not in self.stage_order,'final_polish is a reserved stage ID')
        require(not self.stage_handoff or len(self.stage_order)>=2,'handoff requires a predefined stage order')
        require(self.final_polish in {'disabled','once_full_native'},'unknown final polish strategy')
        integer(self.polish_max_nfev,'polish_max_nfev')
        if self.final_polish=='disabled':
            require(self.polish_max_nfev==0 and self.polish_tolerances is None,'disabled polish cannot hide a polish request')
        else:
            integer(self.polish_max_nfev,'polish_max_nfev',minimum=1)
            require(isinstance(self.polish_tolerances,(tuple,list)) and len(self.polish_tolerances)==3,'polish requires explicit native tolerances')
            object.__setattr__(self,'polish_tolerances',tuple(number(v,'polish tolerance',positive=True) for v in self.polish_tolerances))
            require(all(v>2.220446049250313e-16 for v in self.polish_tolerances),'invalid polish tolerance')
            require(self.budget.max_total_nfev is not None and self.polish_max_nfev<=self.budget.max_total_nfev,
                    'polish must fit an explicit whole-job evaluation budget')
        if self.mode=='legacy':
            require(self.objective_change_tolerance is None and self.parameter_change_tolerance is None and self.patience is None,
                    'legacy does not enable monitored thresholds')
            require(not self.stage_handoff and self.final_polish=='disabled','legacy cannot silently enable handoff/polish')
        else:
            require(self.solver_options.strategy=='trf','monitored contract supports TRF only; execution is not yet implemented')
            require(self.gtol is not None,'monitored must retain native gtol')
            for k in ('objective_change_tolerance','parameter_change_tolerance'):
                object.__setattr__(self,k,number(getattr(self,k),k,positive=True))
            integer(self.patience,'patience',minimum=1)
            require(self.budget.max_iterations is not None and self.budget.max_total_nfev is not None,
                    'monitored requires explicit accepted-iteration and whole-job nfev budgets')
        require(self.budget.max_trace_records is not None,'trace storage budget is required')

    def to_dict(self):
        result=Contract.to_dict(self)
        result['solver_options']={f.name:getattr(self.solver_options,f.name) for f in fields(SolverOptions)}
        return result

    @classmethod
    def from_dict(cls,value):
        exact_fields(value,['contract_kind',*[f.name for f in fields(cls)]],cls.__name__)
        value=dict(value)
        opt=value['solver_options']
        exact_fields(opt,[f.name for f in fields(SolverOptions)],'SolverOptions')
        try:value['solver_options']=SolverOptions(**opt)
        except (TypeError,ValueError) as exc:raise ContractError(f'invalid SolverOptions: {exc}') from exc
        value['budget']=SolverBudget.from_dict(value['budget'])
        return super(TerminationPolicy,cls).from_dict(value)

    @classmethod
    def legacy_for(cls,config):
        """Record R00's actual old defaults, never pass them to an optimizer."""
        from pyecr.config import AnalysisConfig
        require(isinstance(config,AnalysisConfig),'actual AnalysisConfig required for legacy defaults')
        config.validate();mid=config.model_id.value
        require(mid in SIX_MODELS,'legacy adapter requires one of the six scientific models')
        requested=SolverOptions.from_config(config)
        fixed_d='fixed_d' in config.model_specific
        direct=mid in {'M7_SINGLE_K','M8_DUAL_K'} or (mid=='M3_DUAL_HO_DK' and fixed_d)
        require(not direct or requested.strategy=='trf','legacy direct backend supports TRF only; do not relabel an unsupported request')
        if fixed_d:
            require(mid in {'M1_SINGLE_DK','M3_DUAL_HO_DK'} and requested.strategy=='trf'
                    and config.boundary.mode=='ideal_step' and config.fit.loss=='linear',
                    'legacy fixed-D metadata requires the existing ideal-step/linear/TRF path')
        tol=config.solver.relative_tolerance if direct else 1e-8
        return cls(solver_options=SolverOptions() if direct else requested,ftol=tol,xtol=tol,gtol=tol,
                   budget=SolverBudget(config.fit.max_nfev,max_trace_records=10000))


@dataclass(frozen=True, slots=True, kw_only=True)
class FitControlSpec(Contract):
    KIND: ClassVar[str] = 'fit_control_spec'
    output_ci: ConfidencePolicy = field(default_factory=ConfidencePolicy)
    gate: ReliabilityPolicy = field(default_factory=ReliabilityPolicy)
    termination: TerminationPolicy = field(default_factory=TerminationPolicy)
    defaulted_legacy: bool = False

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.output_ci,ConfidencePolicy) and isinstance(self.gate,ReliabilityPolicy)
                and isinstance(self.termination,TerminationPolicy),'invalid nested fit controls')
        require(isinstance(self.defaulted_legacy,bool),'defaulted_legacy must be boolean')
        require(not self.defaulted_legacy or (self.output_ci==ConfidencePolicy() and self.termination.mode=='legacy'),
                'defaulted_legacy cannot claim a custom C or monitored mode')


def read_fit_controls(record,*,config,existing_result=None):
    """Only an ABSENT sidecar means legacy. Present null/partial/new-version fails.

    The enclosing old project/result reader remains responsible for its version.
    This function does not rewrite the record, FitResult, endpoints or config.
    """
    require(isinstance(record,Mapping),'enclosing record must be an object')
    if 'fit_controls' not in record:
        result=FitControlSpec(termination=TerminationPolicy.legacy_for(config),defaulted_legacy=True)
    else:result=FitControlSpec.from_dict(record['fit_controls'])
    if existing_result is not None:result.output_ci.assert_result(existing_result)
    return result


def _problem(source):
    return {k:getattr(source,k) for k in ('dataset_sha256','recipe_sha256','config_sha256','model_id')}


def _fit(source):
    return {**_problem(source),**{k:getattr(source,k) for k in ('fit_id','fit_sha256')}}


@dataclass(frozen=True, slots=True, kw_only=True)
class FitControlBinding(Contract):
    KIND: ClassVar[str] = 'fit_control_binding'
    source: SourceIdentity
    controls: FitControlSpec
    ci_request_id: str | None
    gate_policy_id: str
    execution_policy_id: str
    execution_request_id: str

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.source,SourceIdentity) and self.source.config_sha256 is not None,'binding requires existing config/source identity')
        require(self.source.model_id in SIX_MODELS,'fit control binding supports the six physical scientific models only')
        require(isinstance(self.controls,FitControlSpec),'invalid fit controls')
        expected=self.ids(self.source,self.controls)
        for k,v in expected.items():
            if getattr(self,k)!=v:raise IdentityConflictError(f'{k} conflicts with source or policy')

    @staticmethod
    def ids(source,controls):
        execution=controls.termination.content_sha256
        return dict(ci_request_id=None if source.fit_id is None else canonical_sha256({'source':_fit(source),'policy':controls.output_ci}),
                    gate_policy_id=controls.gate.content_sha256,execution_policy_id=execution,
                    execution_request_id=canonical_sha256({'source':_problem(source),'execution_policy_id':execution}))

    @classmethod
    def capture(cls,source,controls):
        require(isinstance(source,SourceIdentity) and isinstance(controls,FitControlSpec),'source and controls required')
        return cls(source=source,controls=controls,**cls.ids(source,controls))

    def assert_ci_source(self,source,policy=None):
        require(self.source.fit_id is not None,'CI request needs an actual parent Fit identity')
        if not isinstance(source,SourceIdentity) or _fit(self.source)!=_fit(source):
            raise IdentityConflictError('CI parent dataset/recipe/config/Fit conflict')
        if policy is not None and policy!=self.controls.output_ci:
            raise IdentityConflictError('CI request policy conflict; old endpoints retain their original C')

    def assert_execution_source(self,source):
        if not isinstance(source,SourceIdentity) or _problem(source)!=_problem(self.source):
            raise IdentityConflictError('execution source changed; late result belongs to its frozen history')

    def current_dependencies(self,source,*,accepted_ci_policy=None):
        """Pure classification after explicit actions; draft editing is not an action."""
        require(isinstance(source,SourceIdentity),'current source required')
        problem_same=_problem(source)==_problem(self.source)
        fit_same=problem_same and _fit(source)==_fit(self.source)
        ci_same=fit_same and (accepted_ci_policy is None or accepted_ci_policy==self.controls.output_ci)
        return dict(physical_fit='unavailable' if source.fit_id is None else 'current' if fit_same else 'stale',
                    output_ci='unavailable' if source.fit_id is None else 'current' if ci_same else 'stale',
                    gate='unavailable' if source.diagnostics_sha256 is None else 'current' if fit_same and self.source.diagnostics_sha256==source.diagnostics_sha256 else 'stale',
                    execution_source='current' if problem_same else 'stale',
                    running_request='frozen; draft policy edits never mutate or submit it')
