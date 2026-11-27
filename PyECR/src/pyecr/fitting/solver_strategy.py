"""P59A GUI-independent local solver portfolio and transient audit records.

No ECR equations are implemented here. The adapters capture the actual
production residual/parameterization. ContextVar scopes are nested/task-local.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
import ctypes
import inspect
import math
import time
from typing import Callable, Mapping, Protocol

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares
from pyecr.fitting.iteration_observation import observed_least_squares, diagnostic_evaluation, observation_active
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit, logit

VERSION = 'p59a-local-v1'
STRATEGIES = ('trf','dogbox','lm_transformed_experimental')


class SolverRejected(Exception):
    """Visible explicit rejection; not caught as a failed start/fallback."""


@dataclass(frozen=True)
class SolverOptions:
    strategy: str = 'trf'
    dogbox_policy: str = 'warn'
    condition_limit: float = 1e8

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError('unknown local solver strategy')
        if self.dogbox_policy not in ('warn','reject'):
            raise ValueError('dogbox_policy must be warn or reject')
        if not math.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError('condition_limit must be finite and >1')

    @classmethod
    def from_config(cls,config):
        raw=getattr(config,'model_specific',{}).get('local_solver',{})
        if not isinstance(raw,Mapping):raise ValueError('local_solver must be a mapping')
        if set(raw)-{'strategy','dogbox_policy','condition_limit'}:
            raise ValueError('unknown local_solver fields')
        result=cls(**raw)
        loss=getattr(getattr(config,'fit',None),'loss','linear')
        if result.strategy=='lm_transformed_experimental' and loss!='linear':
            raise SolverRejected('LM transformed experimental supports only linear loss')
        for section in ('single_ecr','dual_ho'):
            value=getattr(config,'model_specific',{}).get(section,{})
            if not isinstance(value,Mapping):raise ValueError(f'{section} must be a mapping')
            if result.strategy!='trf' and value.get('jacobian_mode','legacy')!='legacy':
                raise SolverRejected('portfolio cannot combine Dogbox/LM with experimental Jacobian')
        return result


@dataclass(frozen=True)
class SolverBudget:
    max_nfev: int | None
    definition: str = 'SciPy optimizer nfev cap; numeric-Jacobian/diagnostic calls reported separately'
    schema_version: int = 1
    max_iterations: int | None = None  # accepted iterations per optimizer call
    max_total_iterations: int | None = None
    max_total_nfev: int | None = None
    max_njev: int | None = None  # whole job, including all starts/stages/polish
    max_residual_calls: int | None = None
    max_forward_calls: int | None = None
    max_diagnostic_calls: int | None = None
    max_wall_seconds: float | None = None
    max_trace_records: int | None = None

    def __post_init__(self):
        from pyecr.contracts.common import integer, number, text, require
        require(integer(self.schema_version,'schema_version',minimum=1)==1,'unsupported solver budget version')
        object.__setattr__(self,'schema_version',int(self.schema_version))
        text(self.definition,'budget definition')
        for name in ('max_nfev','max_iterations','max_total_iterations','max_total_nfev','max_njev',
                     'max_residual_calls','max_forward_calls','max_diagnostic_calls','max_trace_records'):
            value=getattr(self,name)
            if value is not None:object.__setattr__(self,name,integer(value,name,minimum=1))
        if self.max_wall_seconds is not None:
            object.__setattr__(self,'max_wall_seconds',number(self.max_wall_seconds,'max_wall_seconds',positive=True))

    def to_dict(self):
        from dataclasses import fields
        return {'contract_kind':'solver_budget',**{f.name:getattr(self,f.name) for f in fields(self)}}

    @classmethod
    def from_dict(cls,value):
        from dataclasses import fields
        from pyecr.contracts.common import exact_fields, require
        exact_fields(value,['contract_kind',*[f.name for f in fields(cls)]],'SolverBudget')
        require(value['contract_kind']=='solver_budget','expected solver_budget')
        return cls(**{f.name:value[f.name] for f in fields(cls)})

    def require_legacy_runtime(self):
        """R01 fields are contracts, not promises that the old solver enforces them."""
        from dataclasses import fields
        if any(getattr(self,f.name) is not None for f in fields(self)
               if f.name not in {'max_nfev','definition','schema_version'}):
            raise SolverRejected('R01 extended budgets require an explicit future execution adapter; legacy solver does not enforce them')


@dataclass(frozen=True)
class SolverParameterization:
    coordinate_names: tuple[str,...]
    decode: Callable[[np.ndarray],dict[str,float]]
    fixed_parameters: Mapping[str,float]
    version: str
    fitted_parameter_count: int | None = None


@dataclass(frozen=True)
class SolverProblem:
    residual: Callable[[np.ndarray],np.ndarray]
    full_residual: Callable[[np.ndarray],np.ndarray]
    lower: np.ndarray
    upper: np.ndarray
    parameterization: SolverParameterization
    model_id: str
    seed: int | None
    full_data_scope: str = 'current_processed_fit_points'


@dataclass(frozen=True)
class SolverRun:
    payload: Mapping[str,object]

    def __post_init__(self):
        from types import MappingProxyType
        def freeze(v):
            if isinstance(v,Mapping):return MappingProxyType({k:freeze(w) for k,w in v.items()})
            if isinstance(v,(list,tuple)):return tuple(freeze(w) for w in v)
            return v
        object.__setattr__(self,'payload',freeze(self.payload))

    def to_dict(self):
        def thaw(v):
            if isinstance(v,Mapping):return {k:thaw(w) for k,w in v.items()}
            if isinstance(v,tuple):return [thaw(w) for w in v]
            return v
        return thaw(self.payload)


class SolverStrategy(Protocol):
    def solve(self,problem:SolverProblem,start:np.ndarray,budget:SolverBudget,**kwargs): ...


def finite(v):
    v=float(v)
    return v if math.isfinite(v) else None


def spectrum(j):
    j=np.atleast_2d(j)
    s=np.linalg.svd(j,compute_uv=False)
    tolerance=np.finfo(float).eps*max(j.shape)*s[0] if s.size else 0.
    rank=int(np.count_nonzero(s>tolerance))
    condition=finite(s[0]/s[-1]) if s.size and s[-1]>0 else None
    return dict(rank=rank,columns=int(j.shape[1]),condition=condition,
                singular_values=s.tolist(),coordinate_space='original optimizer coordinates',rank_tolerance=float(tolerance))


def memory_snapshot():
    """Process measurements; never interpreted as per-worker isolated peak."""
    try:
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        c=Counters();c.cb=ctypes.sizeof(c)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
        api=ctypes.WinDLL('psapi',use_last_error=True)
        api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
        return dict(RSS_bytes=int(c.WorkingSetSize),peak_RSS_process_bytes=int(c.PeakWorkingSetSize),reason=None,
                    scope='whole process; cumulative peak includes imports/other starts')
    except Exception as exc:
        return dict(RSS_bytes=None,peak_RSS_process_bytes=None,reason=f'{type(exc).__name__}: {exc}',scope='unavailable')


class LogisticBoxChart:
    """Explicit open-box experimental LM chart, retaining an exact start."""
    def __init__(self,lower,upper,start):
        self.lower=np.asarray(lower,dtype=float);self.upper=np.asarray(upper,dtype=float)
        self.anchor=np.asarray(start,dtype=float).copy()
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise SolverRejected('LM chart requires finite original bounds')
        if np.any(start<=self.lower) or np.any(start>=self.upper):
            raise SolverRejected('LM chart cannot encode an endpoint physical start')
        self.start=logit((start-self.lower)/(self.upper-self.lower))
        self.saturation_calls=0

    def decode(self,z):
        if np.iscomplexobj(z):raise SolverRejected('complex-step Unsupported')
        z=np.asarray(z,dtype=float)
        if np.array_equal(z,self.start):return self.anchor.copy()
        f=expit(z)
        q=(1.-f)*self.lower+f*self.upper
        repaired=np.clip(q,self.lower,self.upper)
        if np.any(abs(q-repaired)>64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(self.lower),abs(self.upper)))):
            raise SolverRejected('LM chart produced material bound violation')
        self.saturation_calls+=int(np.any((repaired==self.lower)|(repaired==self.upper)))
        return repaired


@dataclass
class SolverLedger:
    options: SolverOptions
    full_provider: Callable | None = None
    provenance: Mapping[str,object] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    def snapshots(self):
        result=[]
        for row in self.records:
            group=[r for r in self.records if r['model_id']==row['model_id'] and r.get('fitted_SSE') is not None and r['status']!='rejected']
            ranked=sorted(group,key=lambda r:(r['fitted_SSE'],r['start_index']))
            full=sorted(group,key=lambda r:(r['full_data_SSE'],r['start_index']))
            item=dict(row,candidate_rank=next((i+1 for i,r in enumerate(ranked) if r is row),None),
                      full_data_rank=next((i+1 for i,r in enumerate(full) if r is row),None))
            item['candidate_rank_reason']=None if item['candidate_rank'] is not None else 'No eligible finite candidate; rejection/cancellation/exception'
            result.append(SolverRun(item))
        return tuple(result)


_active:ContextVar[SolverLedger|None]=ContextVar('pyecr_local_solver',default=None)


@contextmanager
def solver_scope(options=None,*,full_provider=None,provenance=None):
    ledger=SolverLedger(options or SolverOptions(),full_provider,provenance or {})
    token=_active.set(ledger)
    try:yield ledger
    finally:_active.reset(token)


def current_solver_runs():
    ledger=_active.get()
    return ledger.snapshots() if ledger is not None else ()


def current_solver_options():
    ledger = _active.get()
    return ledger.options if ledger is not None else SolverOptions()


def _production_problem(fun,lower,upper,ledger):
    """Typed production adapters; forward/objective arithmetic is not copied."""
    bound=getattr(fun,'__self__',None)
    explicit = getattr(bound, 'solver_problem', None)
    if explicit is not None:
        if ledger.full_provider is not None:
            from dataclasses import replace
            full = ledger.full_provider(bound.evaluator, bound.parameterization)
            return replace(explicit,
                full_residual=lambda q: full(bound.project(q).coordinates),
                full_data_scope='frozen_full_observation_grid')
        return explicit
    if bound is not None and hasattr(bound,'context'):
        from pyecr.fitting.single import _reported_params
        c=bound.context;names=tuple(c.parameter_names)
        fixed=_reported_params(tuple(c.fixed_coordinates),np.array(list(c.fixed_coordinates.values()))) if c.fixed_coordinates else {}
        for name in ('amplitude','baseline'):
            if name not in names:fixed[name]=float(getattr(c.forward_context,name))
        decode=lambda q:{**fixed,**_reported_params(names,q)}
        par=SolverParameterization(names,decode,fixed,'single-log10-v1')
        kind='M1' if 'log10_k' in names or 'log10_k' in c.fixed_coordinates else 'M0'
        full=ledger.full_provider(bound,None) if ledger.full_provider else fun
        seed=ledger.provenance.get('seed')
    else:
        values=inspect.getclosurevars(fun).nonlocals
        p,e=values['parameterization'],values['evaluator']
        constrained=p.config.label_constraint.value=='D_H>=D_O'
        names=('log10_D_O','eta') if constrained else ('log10_D_H','log10_D_O')
        names+=('log10_A_H','log10_A_O')
        if p.is_m3:names+=('log10_k_H','log10_k_O')
        if p.config.fit_baseline:names+=('y0',)
        par=SolverParameterization(names,p.decode,{} if p.config.fit_baseline else {'y0':p.fixed_baseline},'dual-production-layout-v1')
        kind='M3' if p.is_m3 else 'M2'
        full=ledger.full_provider(e,p) if ledger.full_provider else fun
        seed=p.config.random_seed
    return SolverProblem(fun,full,np.asarray(lower),np.asarray(upper),par,kind,seed,
                         'frozen_full_observation_grid' if ledger.full_provider else 'current_processed_fit_points')


@dataclass(frozen=True)
class ScipySolverStrategy:
    options:SolverOptions

    def solve(self,problem,start,budget,**kwargs):
        budget.require_legacy_runtime()
        ledger=_active.get()
        record=dict(strategy=self.options.strategy,version=VERSION,model_id=problem.model_id,
            start_index=sum(r['model_id']==problem.model_id for r in ledger.records) if ledger else 0,
            coordinate_names=list(problem.parameterization.coordinate_names),coordinate_version=problem.parameterization.version,
            seed=problem.seed,budget=dict(max_nfev=budget.max_nfev,definition=budget.definition),
            initial_coordinates=np.asarray(start).tolist(),initial_physical=problem.parameterization.decode(start),
            lower=problem.lower.tolist(),upper=problem.upper.tolist(),fixed_parameters=dict(problem.parameterization.fixed_parameters),
            full_data_scope=problem.full_data_scope,status='exception',warnings=[],provenance=dict(ledger.provenance) if ledger else {})
        record.update(nfev=None,njev=None,njev_reason='optimizer did not complete',optimality=None,
            optimality_coordinate_space=None,bound_hits=None,fitted_SSE=None,full_data_SSE=None,
            optimizer_cost=None,solver_wall_s=None,final_physical=None,candidate_rank=None,full_data_rank=None,
            unavailable_reason='optimizer did not complete',
            candidate_rank_reason='Start ensemble incomplete; use final analysis ledger',
            bound_hits_coordinate_space='original optimizer coordinates; eta endpoints include label constraint')
        result=None
        begin=time.perf_counter();before=memory_snapshot();optimizer_calls=diagnostic_calls=0
        def counted(q):
            nonlocal optimizer_calls
            optimizer_calls+=1
            return problem.residual(q)
        def diagnostic(q):
            nonlocal diagnostic_calls
            diagnostic_calls+=1
            with diagnostic_evaluation():
                return problem.residual(q)
        def assess(j,stage):
            info=spectrum(j);record[stage+'_Jacobian']=info
            if info['rank']<info['columns'] or info['condition'] is None or info['condition']>self.options.condition_limit:
                text=f'Dogbox NOT RECOMMENDED: {stage} raw Jacobian rank={info["rank"]}/{info["columns"]}, condition={info["condition"]}; heuristic limit={self.options.condition_limit:g}'
                if self.options.strategy=='dogbox':
                    record['warnings'].append(text)
                    if self.options.dogbox_policy=='reject':raise SolverRejected(text+'; no fallback')
        try:
            method=self.options.strategy
            if method=='lm_transformed_experimental' and observation_active():
                raise SolverRejected('LM accepted-iterate observation Unsupported; no residual-call iteration emulation')
            if kwargs.get('jac')=='cs':raise SolverRejected('complex-step Unsupported; requires separate full-production compatibility validation')
            if method!='trf' and callable(kwargs.get('jac')):raise SolverRejected('custom Jacobian cannot be combined with portfolio candidate')
            if method=='lm_transformed_experimental' and kwargs.get('loss','linear')!='linear':
                raise SolverRejected('LM transformed experimental supports only linear loss')
            # Precheck and postcheck are separate diagnostic calls; they do not
            # change the max_nfev budget or any objective values.
            initial_j=np.atleast_2d(approx_derivative(diagnostic,start,method='2-point',bounds=(problem.lower,problem.upper)))
            assess(initial_j,'initial')
            record['initial_residual_count']=int(initial_j.shape[0])
            if method=='lm_transformed_experimental' and initial_j.shape[0]<initial_j.shape[1]:
                raise SolverRejected('LM requires residual count >= free coordinate count; no fallback')
            started=time.perf_counter()
            options=dict(kwargs,max_nfev=budget.max_nfev)
            options.setdefault('bounds',(problem.lower,problem.upper))
            chart=None
            if method=='lm_transformed_experimental':
                chart=LogisticBoxChart(problem.lower,problem.upper,np.asarray(start))
                options.pop('bounds',None);options.update(method='lm',jac='2-point',loss='linear',x_scale=1)
                result=scipy_least_squares(lambda z:counted(chart.decode(z)),chart.start,**options)
                record['native_final_coordinates']=result.x.tolist()
                record['optimality_coordinate_space']='unconstrained logistic chart'
                result.x=chart.decode(result.x)
                result.jac=approx_derivative(diagnostic,result.x,method='2-point',bounds=(problem.lower,problem.upper))
                result.fun=diagnostic(result.x)
                result.grad=result.jac.T@result.fun
                record['chart_saturation_calls']=chart.saturation_calls
                if chart.saturation_calls:record['warnings'].append('LM chart reached float64 box saturation; transformed derivative may lose information')
                record['warnings'].append('LM transformed path experimental; local CI uses full original coordinates, not calibrated coverage')
            else:
                options.update(method=method)
                result=observed_least_squares(counted,start,_observation_context={
                    'names':problem.parameterization.coordinate_names,
                    'kinds':tuple('log10' if n.startswith('log10_') else 'linear' for n in problem.parameterization.coordinate_names),
                    'stage_id':problem.model_id,
                    'fitted_parameter_count':problem.parameterization.fitted_parameter_count},**options)
                record['optimality_coordinate_space']='original optimizer coordinates'
            record['solver_wall_s']=time.perf_counter()-started
            raw_j=result.jac if kwargs.get('loss','linear')=='linear' else approx_derivative(diagnostic,result.x,method='2-point',bounds=(problem.lower,problem.upper))
            r=diagnostic(result.x)
            full_r=np.asarray(problem.full_residual(result.x))
            # This is one explicit full-data objective evaluation, separately
            # counted rather than hidden in optimizer nfev.
            diagnostic_calls+=1
            old_gradient=raw_j.T@r
            distances=np.minimum(result.x-problem.lower,problem.upper-result.x)
            hits=distances<=1e-7*np.maximum(1.,problem.upper-problem.lower)
            record.update(status='success' if result.success else 'nonconverged',nfev=int(result.nfev),
                njev=None if result.njev is None else int(result.njev),
                njev_reason='SciPy numerical LM does not expose njev' if result.njev is None else None,
                termination_reason=str(result.message),exit_status=int(result.status),optimality=finite(result.optimality),
                raw_old_gradient_inf=finite(np.max(abs(old_gradient))),
                bound_hits=dict(zip(problem.parameterization.coordinate_names,map(bool,hits),strict=True)),
                final_coordinates=result.x.tolist(),final_physical=problem.parameterization.decode(result.x),
                fitted_SSE=float(r@r),full_data_SSE=float(full_r@full_r),optimizer_cost=float(result.cost),
                full_data_points=int(full_r.size),fitted_points=int(r.size),unavailable_reason=None)
            assess(raw_j,'final')
            return result
        except BaseException as exc:
            record.update(status='rejected' if isinstance(exc,SolverRejected) else 'cancelled' if isinstance(exc,CancelledError) else 'exception',termination_reason=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record.update(actual_residual_calls=optimizer_calls,diagnostic_residual_calls=diagnostic_calls,
                wall_s=time.perf_counter()-begin,RSS_before=before,RSS_after=memory_snapshot())
            if ledger is not None:ledger.records.append(record)
            if result is not None:
                # Direct interface clients receive the same immutable statistics;
                # analysis ledger assigns ranks after all starts are known.
                result.solver_run=SolverRun(record)


def least_squares(fun,start,**kwargs):
    """Production call adapter; absent scope retains direct SciPy legacy path."""
    ledger=_active.get()
    if ledger is None:return scipy_least_squares(fun,start,**kwargs)
    lo,hi=kwargs['bounds']
    problem=_production_problem(fun,lo,hi,ledger)
    return ScipySolverStrategy(ledger.options).solve(problem,start,SolverBudget(kwargs.get('max_nfev')),**kwargs)
