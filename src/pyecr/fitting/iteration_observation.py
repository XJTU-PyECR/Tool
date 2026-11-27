"""Opt-in, read-only SciPy accepted-point observation; never a stop controller.

Uses existing R01 AcceptedTrace/AcceptedIterate. Extra metrics are a versioned
companion, not another FitResult or another forward model. No callback evaluates
the objective. Unobserved callers retain the exact native solver call.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import OrderedDict
from dataclasses import replace
from concurrent.futures import CancelledError
from pathlib import Path
from time import monotonic
from uuid import uuid4
from types import MappingProxyType
import inspect
import json

import numpy as np
import scipy
from scipy.optimize import least_squares as native_least_squares
from scipy.optimize._lsq.least_squares import construct_loss_function

VERSION='readonly-accepted-observation/1'
_observer=ContextVar('pyecr_readonly_observer',default=None)
_phase=ContextVar('pyecr_observation_phase',default='diagnostic')


def observation_active():return _observer.get() is not None


def monitored_execution_active():
    return getattr(_observer.get(), 'policy', None) is not None


def projection_execution_supported():
    """Profiled residual roundoff needs a native objective/step stop.

    Strict Monitored runs may deliberately disable both native stops. Keep
    their full joint coordinates rather than alter the declared stop policy.
    """
    policy = getattr(_observer.get(), 'policy', None)
    return policy is None or (policy.ftol is not None and policy.xtol is not None)


def capabilities():
    return dict(scipy_version=scipy.__version__,callback_keyword='callback' in inspect.signature(native_least_squares).parameters,
        methods={'trf':'formal callback','dogbox':'formal callback where production supports it','lm':'Unsupported: no accepted-iterate callback'},
        intermediate_fields=['x','fun','cost','nit','nfev'],intermediate_njev='not reported; null, never inferred from residual calls')


def residual_metrics(residuals,*,p_free,unit,weighted_SSE=None,F=None):
    r=np.asarray(residuals,dtype=float)
    if r.ndim!=1 or not r.size or not np.all(np.isfinite(r)):raise ValueError('Metrics need the complete finite residual vector')
    if type(p_free) is not int or p_free<0:raise ValueError('p_free must be a nonnegative integer')
    sse=float(r@r);n=len(r);df=n-p_free
    return dict(raw_SSE=sse,weighted_SSE=sse if weighted_SSE is None else float(weighted_SSE),F=F,
        N=n,p_free=p_free,df=df,RMSE=float(np.sqrt(sse/n)),RSE=None if df<=0 else float(np.sqrt(sse/df)),
        RSE_reason=None if df>0 else 'N <= p_free; residual standard error unavailable',
        residual_unit=unit,RMSE_unit=unit,RSE_unit=unit,SSE_unit=f'({unit})^2',
        RMSE_definition='sqrt(raw SSE/N); existing RMSE meaning retained',RSE_definition='sqrt(raw SSE/(N-p_free))')


@contextmanager
def evaluation_phase(phase):
    token=_phase.set(phase)
    try:yield
    finally:_phase.reset(token)


def count_observed_call(kind):
    obs=_observer.get()
    if obs is None:return
    phase=_phase.get()
    if hasattr(obs,'before_count'):obs.before_count(kind,phase)
    if kind=='forward':
        obs.counters['forward_calls']+=1
        if phase=='diagnostic':obs.counters['diagnostic_calls']+=1
    elif kind=='residual' and phase=='diagnostic':
        obs.counters['residual_calls']+=1
        # This is an unwrapped production residual. Its following forward hook
        # is counted as a diagnostic call; do not double count that evaluation.
    elif kind=='jacobian' and phase=='diagnostic':
        obs.counters['analytic_jacobian_calls']+=1;obs.counters['diagnostic_calls']+=1


def observe_direct_forward():
    # Joint residuals already emit the legacy forward hook. Initializers and
    # direct finalization calls do not; count those without altering old UI
    # telemetry or counting an optimizer residual twice.
    if _phase.get() not in {'optimizer_residual','explicit_diagnostic'}:count_observed_call('forward')


def observe_start(phase,index,total):
    obs=_observer.get()
    if obs is not None:obs.production_start=dict(phase=phase,index_1based=index,total_starts=total)


@contextmanager
def diagnostic_evaluation():
    obs=_observer.get()
    if obs is not None:
        if hasattr(obs,'before_diagnostic'):obs.before_diagnostic()
        obs.counters['diagnostic_calls']+=1;obs.counters['residual_calls']+=1
    with evaluation_phase('explicit_diagnostic'):yield


class ReadOnlyFitObserver:
    def __init__(self,*,recipe=None,job_id=None,record_budget=512,max_contexts=128,F_ref=None,
                 objective_tolerance=1e-7,parameter_tolerance=1e-6,patience=7):
        from pyecr.contracts.common import integer,number
        from pyecr.preprocessing import PreprocessConfig
        self.recipe_declared=recipe is not None
        self.recipe=recipe or PreprocessConfig();self.job_id=job_id or uuid4().hex
        self.record_budget=integer(record_budget,'record_budget',minimum=2)
        self.max_contexts=integer(max_contexts,'max_contexts',minimum=1)
        self.given_F_ref=None if F_ref is None else number(F_ref,'F_ref',positive=True)
        self._candidate_policy=MappingProxyType(dict(objective_tolerance=number(objective_tolerance,'objective_tolerance',positive=True),
            parameter_tolerance=number(parameter_tolerance,'parameter_tolerance',positive=True),patience=integer(patience,'patience',minimum=1),
            effect='candidate counts only; no stopping, stage handoff or polish'))
        self.contexts=[];self.warnings=[];self.dropped_contexts=0;self.binding=None;self.state='not_started'
        self.production_start=dict(phase='standalone native call',index_1based=None,total_starts=None)
        self.counters=dict(accepted_iterations=0,native_nfev=0,native_njev=0,residual_calls=0,forward_calls=0,
            diagnostic_calls=0,analytic_jacobian_calls=0,native_njev_complete=True,native_nfev_complete=True,accepted_iterations_complete=True)
        self.final_metrics=None;self.final_residuals=None;self.final_source=None;self._call_index=0

    @property
    def candidate_policy(self):return self._candidate_policy

    def initialize(self,dataset,config,*,job_id=None,screening=None,solver_options=None):
        from pyecr.contracts import FitControlBinding,FitControlSpec,SourceIdentity,TerminationPolicy
        from pyecr.fitting.solver_strategy import SolverRejected
        if self.state!='not_started':raise ValueError('Observer is single-use; create another for a new job')
        if dataset.history and not self.recipe_declared:raise ValueError('Processed source needs its explicit accepted recipe; no inferred empty recipe')
        if not capabilities()['callback_keyword']:raise SolverRejected('Unsupported: installed SciPy has no formal callback')
        if solver_options is not None and solver_options.strategy=='lm_transformed_experimental':
            raise SolverRejected('LM accepted-iterate observation Unsupported; no fun-call iteration emulation')
        if screening is not None and screening.enabled and not getattr(getattr(self,'production_execution',None),'screening_enabled',False):
            raise SolverRejected('P60 accepted observation Unsupported: subset objective cannot be labelled full-observation raw SSE under R01; ordinary P60 remains available without observer')
        self.dataset=dataset;self.config=config;self.unit=str(dataset.metadata.get('signal_unit','signal'))
        self.N=len(dataset.processed_signal);self.job_id=job_id or self.job_id;self.started=monotonic()
        policy=TerminationPolicy.legacy_for(config)
        if solver_options is not None:policy=replace(policy,solver_options=solver_options)
        # A job may contain stages with different per-call caps; exact call
        # tolerances/caps are frozen in each context, not enforced by observation.
        policy=replace(policy,budget=replace(policy.budget,max_nfev=None,max_trace_records=self.record_budget))
        self.binding=FitControlBinding.capture(SourceIdentity.capture(dataset,self.recipe,config=config),FitControlSpec(termination=policy))
        self.state='running'

    @contextmanager
    def scope(self):
        if self.binding is None:raise ValueError('Initialize observation with its actual source first')
        token=_observer.set(self)
        try:yield self
        except BaseException as exc:
            self.state='cancelled' if isinstance(exc,CancelledError) else 'failed';self.warnings.append(f'{type(exc).__name__}: {exc}');raise
        finally:_observer.reset(token)

    def finish_fit(self,fit):
        from pyecr.contracts import SourceIdentity
        self.final_metrics=residual_metrics(fit.residuals,p_free=len(fit.params),unit=self.unit)
        matching=[c for c in self.contexts if c.records and c.records[-1].event=='terminal' and c.last_fun is not None
            and (np.array_equal(c.last_fun,fit.residuals) or np.array_equal(-c.last_fun,fit.residuals))]
        costs={c.records[-1].F for c in matching}
        if len(costs)==1:
            selected=matching[-1]
            self.final_metrics.update(F=selected.records[-1].F,weighted_SSE=selected.metrics[-1]['weighted_SSE'],
                loss=selected.loss,f_scale=selected.f_scale,F_reason=None,
                selected_native_contexts=[c.id for c in matching],F_source='native cost at exact full-residual match (either documented residual sign)')
            for context in matching:
                last=context.records[-1]
                if fit.exit_status>0 and last.native_success is True and len(fit.residuals)==self.N:
                    context.records[-1]=replace(last,finalization='full_observations_verified')
                    context.state='completed'
        else:self.final_metrics.update(F=None,F_reason='No unique native cost with an exact complete selected-residual match; no inferred objective')
        self.final_metrics.update(native_status=fit.exit_status,numerically_converged=fit.exit_status>0,
            noise_status='not_evaluated',reliability_status='not_evaluated',finalization='unchanged production full-observation fitter')
        self.final_residuals=fit.residuals.tolist()
        self.final_source=SourceIdentity.capture(self.dataset,self.recipe,config=self.config,fit=fit).to_dict()
        self.state='completed'

    def snapshot(self):
        if self.binding is None:raise ValueError('Observation has not started')
        value=dict(version=VERSION,mode='readonly',capabilities=capabilities(),job_id=self.job_id,state=self.state,
            binding=self.binding.to_dict(),candidate_policy=dict(self.candidate_policy),counters=dict(self.counters),
            counter_scope='whole explicitly observed fit job, all production starts/stages and diagnostic hooks; never GUI display sampling',
            native_njev_reason=None if self.counters['native_njev_complete'] else 'One or more native calls did not report njev; reported subtotal only',
            native_nfev_reason=None if self.counters['native_nfev_complete'] else 'Exception/cancellation: native nfev subtotal uses only reported snapshots; not inferred from fun calls',
            observations=dict(N=self.N,raw_count=len(self.dataset.raw_signal),raw_indices=np.flatnonzero(self.dataset.mask).tolist(),
                processed_time_s=self.dataset.processed_time.tolist(),time_unit=str(self.dataset.metadata.get('time_unit','s')),signal_unit=self.unit),
            context_storage=dict(max_contexts=self.max_contexts,record_budget_per_context=self.record_budget,dropped_contexts=self.dropped_contexts),
            warnings=list(self.warnings),contexts=[c.snapshot() for c in self.contexts],
            final_metrics=self.final_metrics,final_residuals=self.final_residuals,final_source=self.final_source)
        from pyecr.contracts.common import canonical_sha256
        value['record_sha256']=canonical_sha256(value)
        return json.loads(json.dumps(value,allow_nan=False))

    def save(self,path):
        from pyecr.contracts.common import strict_json_loads,require
        path=Path(path);require(not path.exists(),'Observation file exists; use a new path')
        value=self.snapshot();stage=path.with_name(path.name+'.'+uuid4().hex+'.staging')
        try:
            stage.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
            loaded=read_observation(stage.read_text(encoding='utf-8'));require(loaded==value,'Observation staging parity conflict')
            stage.rename(path)
        finally:
            if stage.exists():stage.unlink()
        return path


def read_observation(text):
    from pyecr.contracts import AcceptedTrace,FitControlBinding,SourceIdentity
    from pyecr.contracts.common import strict_json_loads,require,exact_fields,canonical_sha256
    value=strict_json_loads(text)
    exact_fields(value,('version','mode','capabilities','job_id','state','binding','candidate_policy','counters','counter_scope',
        'native_njev_reason','native_nfev_reason','observations','context_storage','warnings','contexts','final_metrics','final_residuals','final_source','record_sha256'),'Read-only observation')
    require(value['version']==VERSION and value['mode']=='readonly','Unknown observation version/mode')
    require(value['record_sha256']==canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'}),'Observation hash conflict')
    binding=FitControlBinding.from_dict(value['binding'])
    source_rows=value['observations'];indices=source_rows['raw_indices']
    require(len(indices)==len(source_rows['processed_time_s'])==source_rows['N'] and indices==sorted(set(indices)), 'Observation full-row mapping conflict')
    require(all(type(i) is int and 0<=i<source_rows['raw_count'] for i in indices),'Observation raw index outside source')
    if value['final_source'] is not None:binding.assert_execution_source(SourceIdentity.from_dict(value['final_source']))
    if value['final_metrics'] is not None:
        m=value['final_metrics'];r=np.asarray(value['final_residuals'],dtype=float)
        require(len(r)==m['N'] and np.all(np.isfinite(r)),'Incomplete final residuals')
        require(np.isclose(float(r@r),m['raw_SSE'],rtol=1e-12,atol=1e-14),'Final raw SSE conflict')
    for context in value['contexts']:
        trace=AcceptedTrace.from_dict(context['accepted_trace'])
        require(trace.binding.source==binding.source and trace.job_id==value['job_id'],'Observation source/job conflict')
        rows=context['metrics'];require(len(rows)==len(trace.records),'Trace/metric storage conflict')
        for row,record in zip(rows,trace.records):
            require(row['sequence']==record.sequence and row['F']==record.F and row['raw_SSE']==record.SSE,'Trace/metric identity conflict')
            if record.finite:
                require(np.isclose(row['RMSE']**2*row['N'],record.SSE,rtol=1e-12,atol=1e-14),'RMSE conflict')
                require(row['RSE'] is None if row['df']<=0 else np.isclose(row['RSE']**2*row['df'],record.SSE,rtol=1e-12,atol=1e-14),'RSE conflict')
    return value


class _Call:
    def __init__(self,obs,start,options,context):
        self.obs=obs;obs._call_index+=1;self.id=f'call-{obs._call_index:04d}'
        self.stage=context.get('stage_id',obs.config.model_id.value);self.names=tuple(context['names'])
        self.kinds=tuple(context.get('kinds',('linear',)*len(self.names)))
        self.fitted_parameter_count = context.get('fitted_parameter_count') or len(self.names)
        self.units=tuple(context.get('units',tuple('1' if k=='log10' or n=='eta' else obs.unit for n,k in zip(self.names,self.kinds))))
        self.ref=np.array(start,dtype=float,copy=True);lo,hi=options.get('bounds',(-np.inf,np.inf))
        self.lo=np.broadcast_to(lo,self.ref.shape).copy();self.hi=np.broadcast_to(hi,self.ref.shape).copy()
        widths=self.hi-self.lo;self.scales=np.where(np.isfinite(widths)&(widths>0),widths,np.maximum(1.,abs(self.ref)))
        self.loss=options.get('loss','linear');self.f_scale=options.get('f_scale',1.)
        if self.loss not in {'linear','soft_l1','huber','cauchy','arctan'}:raise ValueError('Unsupported observer loss')
        self.weights=None if context.get('weights') is None else np.asarray(context['weights'],dtype=float).copy()
        if self.weights is not None and (self.weights.shape!=(obs.N,) or np.any(self.weights<=0) or not np.all(np.isfinite(self.weights))):
            raise ValueError('Observation weights need positive full-row values; zero-weight raw recovery Unsupported')
        self.options={k:options.get(k,1e-8) for k in ('ftol','xtol','gtol')}
        self.options.update(method=options.get('method','trf'),loss=self.loss,f_scale=self.f_scale,max_nfev=options.get('max_nfev'),
            production_start=dict(obs.production_start),
            max_nfev_effective=options.get('max_nfev') if options.get('max_nfev') is not None else 100*len(self.names),
            budget_scope='SciPy nfev per optimizer call, not iterations; job totals sum all starts/stages',
            scale_rule='finite optimizer-bound width; otherwise max(1,abs(start)); reference=start, frozen before native call')
        if context.get('fitted_parameter_count') is not None:
            self.options['fitted_parameter_count'] = self.fitted_parameter_count
        from pyecr.contracts import FitControlBinding
        actual=replace(obs.binding.controls.termination,solver_options=replace(obs.binding.controls.termination.solver_options,strategy=self.options['method']),
            ftol=self.options['ftol'],xtol=self.options['xtol'],gtol=self.options['gtol'],
            budget=replace(obs.binding.controls.termination.budget,max_nfev=self.options['max_nfev_effective']))
        self.binding=FitControlBinding.capture(obs.binding.source,replace(obs.binding.controls,termination=actual))
        self.records=[];self.metrics=[];self.emitted=0;self.dropped=0;self.accepted=0;self.nfev=0;self.njev=None
        self.callbacks=0;self.duplicates=0;self.previous=None;self.F_ref=obs.given_F_ref;self.native_nit=None
        self.initial_seen=False;self.observer_error=None
        self.objective_count=0;self.joint_count=0;self.state='running';self.warnings=[]
        self.base_nfev=obs.counters['native_nfev'];self.base_njev=obs.counters['native_njev'];self.last_fun=None
        self.residual_cache=OrderedDict();self.residual_cache_budget=max(16,4*len(self.names)+4)

    def remember_residual(self,q,r):
        key=np.asarray(q,dtype=float).tobytes();self.residual_cache[key]=np.array(r,dtype=float,copy=True)
        self.residual_cache.move_to_end(key)
        if len(self.residual_cache)>self.residual_cache_budget:self.residual_cache.popitem(last=False)

    def raw_native_residual(self,q):
        # SciPy1.18.1 dogbox supplies scaled f to the robust callback. Read the
        # exact native fun evaluation saved BEFORE its in-place loss scaling.
        key=np.asarray(q,dtype=float).tobytes()
        if key in self.residual_cache:return self.residual_cache[key]
        if self.previous is not None and np.array_equal(q,self.previous[2]):return self.last_fun
        raise ValueError('Native accepted point has no exact saved residual; observer cannot invent an evaluation')

    def emit(self,event,q,r,F,*,native=None,reason='none'):
        from pyecr.contracts import AcceptedIterate
        obs=self.obs;r=np.array(r,dtype=float,copy=True);q=np.array(q,dtype=float,copy=True)
        finite=bool(r.shape==(obs.N,) and np.all(np.isfinite(q)) and np.all(np.isfinite(r)) and np.isfinite(F))
        if r.shape!=(obs.N,) and event!='terminal':raise ValueError('Accepted observation requires every frozen eligible row; subset trace Unsupported')
        if not finite and event!='terminal':raise ValueError('Nonfinite accepted observation')
        if self.F_ref is None:self.F_ref=max(float(F),np.finfo(float).tiny) if finite else 1.
        raw=(r if self.weights is None else r/np.sqrt(self.weights)) if finite else np.array([])
        sse=float(raw@raw) if finite else None;wsse=float(r@r) if finite else None
        z=(q-self.ref)/self.scales if finite else np.array([])
        rhoF=rhoz=None
        objective_count,joint_count,accepted=self.objective_count,self.joint_count,self.accepted
        total_accepted=obs.counters['accepted_iterations']
        if event=='accepted':
            oldF,oldz,_=self.previous
            rhoF=abs(F/self.F_ref-oldF/self.F_ref)/max(abs(oldF/self.F_ref),1e-12)
            rhoz=float(np.max(abs(z-oldz))/(1+np.max(abs(oldz))))
            objective_count=objective_count+1 if rhoF<obs.candidate_policy['objective_tolerance'] else 0
            joint_count=joint_count+1 if rhoF<obs.candidate_policy['objective_tolerance'] and rhoz<obs.candidate_policy['parameter_tolerance'] else 0
            accepted+=1;total_accepted+=1
        bounds=tuple(map(int,np.where(q<=self.lo,-1,np.where(q>=self.hi,1,0)))) if finite else ()
        success=None if native is None else bool(native.success)
        rec=AcceptedIterate(schema_version=2 if self.base_nfev+self.nfev>obs.counters['residual_calls'] else 1,
            job_id=obs.job_id,start_id=self.id,stage_id=self.stage,sequence=self.emitted+1,event=event,
            accepted_iteration=accepted,nfev=self.nfev,njev=self.njev,total_nfev=self.base_nfev+self.nfev,
            total_njev=None if self.njev is None else self.base_njev+self.njev,total_accepted_iterations=total_accepted,
            residual_calls=obs.counters['residual_calls'],forward_calls=obs.counters['forward_calls'],diagnostic_calls=obs.counters['diagnostic_calls'],
            elapsed_s=max(0.,monotonic()-obs.started),F=float(F) if finite else None,SSE=sse,z=tuple(map(float,z)),rho_F=rhoF,rho_z=rhoz,finite=finite,bounds=bounds,
            optimality=None if native is None else float(native.optimality),
            native_status=None if native is None else int(native.status),native_success=success,native_message=None if native is None else str(native.message),
            stop_reason=reason,finalization='candidate' if event=='terminal' and success else 'rejected' if event=='terminal' else 'pending',
            observation_count=len(r),full_observation_count=obs.N)
        metric=residual_metrics(raw,p_free=self.fitted_parameter_count,unit=obs.unit,weighted_SSE=wsse,F=rec.F) if finite else dict(raw_SSE=None,F=None)
        metric.update(sequence=rec.sequence,objective_patience_count=objective_count,joint_patience_count=joint_count,
            patience_candidate=joint_count>=obs.candidate_policy['patience'],candidate_never_stops=True,
            native_nit=self.native_nit,
            native_active_mask=None if native is None or not hasattr(native,'active_mask') else list(map(int,native.active_mask)))
        self.emitted+=1  # Failed contract construction is not a stored trace record.
        self.objective_count,self.joint_count,self.accepted=objective_count,joint_count,accepted
        obs.counters['accepted_iterations']=total_accepted
        if len(self.records)>=obs.record_budget:
            self.records.pop();self.metrics.pop();self.dropped+=1
            self.warnings=['TRACE_RECORD_BUDGET: first records plus latest/terminal retained; omitted records not replayed as a continuous window']
        self.records.append(rec);self.metrics.append(metric)
        if finite and event in {'initial','accepted'}:self.previous=(float(F),z.copy(),q.copy())
        self.last_fun=r.copy()

    def callback(self,intermediate_result):
        self.callbacks+=1;self.nfev=int(intermediate_result.nfev)
        self.native_nit=int(intermediate_result.nit) if hasattr(intermediate_result,'nit') else None
        if self.observer_error is not None:return
        q=np.asarray(intermediate_result.x)
        if self.previous is None or np.array_equal(q,self.previous[2]):self.duplicates+=1;return
        try:self.emit('accepted',q,self.raw_native_residual(q),float(intermediate_result.cost))
        except Exception as exc:
            self.observer_error=f'{type(exc).__name__}: {exc}'
            self.obs.counters['accepted_iterations_complete']=False
            self.warnings.append('OBSERVER_ERROR: accepted count is a confirmed subtotal; native solver continues; '+self.observer_error)

    def snapshot(self):
        from pyecr.contracts import AcceptedTrace
        from pyecr.contracts.common import canonical_sha256
        obs=self.obs
        trace=AcceptedTrace(schema_version=max((r.schema_version for r in self.records),default=1),
            job_id=obs.job_id,binding=self.binding,parameter_names=self.names,coordinate_kinds=self.kinds,
            coordinate_references=tuple(map(float,self.ref)),coordinate_scales=tuple(map(float,self.scales)),coordinate_units=self.units,
            F_ref=self.F_ref or 1.,F_ref_source='explicit given' if obs.given_F_ref is not None else 'first native residual objective; positive machine-tiny floor, not a noise estimate',
            objective_unit=f'({obs.unit})^2',residual_unit=obs.unit,loss=self.loss,f_scale=self.f_scale,
            weights_sha256=None if self.weights is None else canonical_sha256(self.weights.tolist()),change_definition=obs.binding.controls.termination.change_definition,
            record_budget=obs.record_budget,emitted_count=self.emitted,dropped_records=self.dropped,records=tuple(self.records),state=self.state,warnings=tuple(self.warnings))
        return dict(accepted_trace=trace.to_dict(),metrics=list(self.metrics),native_options=dict(self.options),
            callback_calls=self.callbacks,duplicate_callbacks_ignored=self.duplicates,
            residual_source='exact-coordinate native fun cache before robust in-place scaling; no callback objective evaluation',
            residual_cache_budget=self.residual_cache_budget,
            observer_error=self.observer_error,
            optimizer_bounds=dict(lower=[float(v) if np.isfinite(v) else None for v in self.lo],upper=[float(v) if np.isfinite(v) else None for v in self.hi],
                definition='trace bounds are exact contact; terminal native_active_mask is reported separately'),
            finalization_scope='native optimizer candidate only; parent full-grid verification is separate')


def observed_least_squares(fun,start,*,_observation_context=None,**options):
    obs=_observer.get()
    if obs is None:return native_least_squares(fun,start,**options)
    if hasattr(obs,'solve_monitored'):
        return obs.solve_monitored(fun,start,_observation_context,options)
    if options.get('method','trf') not in {'trf','dogbox'}:raise ValueError('Unsupported observed solver; formal TRF/dogbox callback required')
    if options.get('jac')=='cs':raise ValueError('Unsupported complex-step observation; real-coordinate compatibility is not verified')
    if options.get('callback') is not None or options.get('workers') is not None:raise ValueError('Observer composition with external callback/parallel workers Unsupported')
    context=_observation_context
    if context is None:raise ValueError('Frozen coordinate names required for accepted observation')
    call=_Call(obs,start,options,context)
    if len(obs.contexts)>=obs.max_contexts:
        obs.contexts.pop();obs.dropped_contexts+=1;obs.warnings.append('CONTEXT_BUDGET: first contexts and latest context retained')
    obs.contexts.append(call)
    def counted(q,*args,**kwargs):
        obs.counters['residual_calls']+=1
        with evaluation_phase('optimizer_residual'):r=fun(q,*args,**kwargs)
        call.remember_residual(q,r)
        if not call.initial_seen:
            call.initial_seen=True
            call.nfev=1
            if np.all(np.isfinite(r)):
                lossfun=construct_loss_function(len(r),call.loss,call.f_scale)
                F=float(np.dot(r,r)/2) if lossfun is None else float(lossfun(np.asarray(r),cost_only=True))
                call.emit('initial',q,r,F)
        return r
    jac=options.get('jac')
    if callable(jac):
        def observed_jac(q,*args,**kwargs):
            obs.counters['analytic_jacobian_calls']+=1
            with evaluation_phase('optimizer_jacobian'):return jac(q,*args,**kwargs)
        options={**options,'jac':observed_jac}
    try:
        result=native_least_squares(counted,start,callback=call.callback,**options)
        call.nfev=int(result.nfev);call.njev=None if result.njev is None else int(result.njev)
        reason={1:'native_gtol',2:'native_ftol',3:'native_xtol',4:'native_ftol_xtol',0:'budget'}.get(int(result.status),'failed')
        call.emit('terminal',result.x,call.raw_native_residual(result.x),float(result.cost),native=result,reason=reason)
        # Native endpoint is deliberately only a candidate. The unchanged
        # production fitter performs its own full-array finalization afterwards.
        call.state='history'
        return result
    except BaseException as exc:
        obs.counters['native_nfev_complete']=False
        if call.previous is not None:
            call.emit('terminal',call.previous[2],call.last_fun,call.previous[0],reason='cancelled' if isinstance(exc,CancelledError) else 'failed')
        else:
            nonfinite=any(not np.all(np.isfinite(r)) for r in call.residual_cache.values())
            call.emit('terminal',call.ref,[],float('nan'),reason='cancelled' if isinstance(exc,CancelledError) else 'nonfinite' if nonfinite else 'failed')
        call.state='cancelled' if isinstance(exc,CancelledError) else 'failed'
        raise
    finally:
        obs.counters['native_nfev']+=call.nfev
        if call.njev is None:obs.counters['native_njev_complete']=False
        else:obs.counters['native_njev']+=call.njev
        # Probes are needed only while this native call runs. Retain just the
        # endpoint vector for exact parent matching, not O(starts*probes*N).
        call.residual_cache.clear()


@contextmanager
def observation_scope(observer,dataset,config,*,job_id=None,screening=None,solver_options=None):
    if observer is None:
        token=_observer.set(None)
        try:yield
        finally:_observer.reset(token)
        return
    if not isinstance(observer,ReadOnlyFitObserver):raise TypeError('ReadOnlyFitObserver required')
    observer.initialize(dataset,config,job_id=job_id,screening=screening,solver_options=solver_options)
    with observer.scope():yield
