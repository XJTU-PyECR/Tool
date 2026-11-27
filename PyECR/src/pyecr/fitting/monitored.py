"""Explicit TRF execution policy over R08's accepted-point collector.

No forward equations, fitting result substitutes, or reliability rules live here.
Unsuccessful native endpoints escape the old MultiStart candidate selectors.
"""
from contextlib import contextmanager
from concurrent.futures import CancelledError
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic
from types import MappingProxyType
from uuid import uuid4

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize._lsq.common import CL_scaling_vector
from scipy.optimize._lsq.least_squares import construct_loss_function

from .iteration_observation import (ReadOnlyFitObserver, _Call, _observer,
    evaluation_phase, native_least_squares, read_observation, VERSION as OBS_VERSION)

VERSION = 'monitored-trf/2'  # v2 retains full native candidate arrays; R01 trace stays v1


class MonitoredStop(Exception):
    """Retained stop candidate; deliberately not caught as a failed start."""
    def __init__(self, reason, *, native_status=None):
        self.reason=reason;self.native_status=native_status
        super().__init__('Monitored candidate retained: '+reason+'; no successful Fit published')


class MonitoredBudgetExceeded(MonitoredStop):
    pass


class MonitoredNumericalError(MonitoredStop):
    """Nonfinite execution is an error, not a stable stopped candidate."""


def default_monitored_policy(*,max_nfev=5000,max_iterations=1000,max_total_nfev=20000):
    from pyecr.contracts import TerminationPolicy
    from .solver_strategy import SolverBudget
    return TerminationPolicy(mode='monitored',ftol=None,xtol=None,gtol=1e-8,
        objective_change_tolerance=1e-7,parameter_change_tolerance=1e-6,patience=7,
        budget=SolverBudget(max_nfev,max_iterations=max_iterations,max_total_nfev=max_total_nfev,max_trace_records=10000))


class MonitoredFitObserver(ReadOnlyFitObserver):
    """One frozen job. Explicit service/API opt-in; Legacy remains untouched."""
    def __init__(self,policy,*,recipe=None,job_id=None,cancel_check=None,F_ref=None):
        from pyecr.contracts import TerminationPolicy
        if not isinstance(policy,TerminationPolicy) or policy.mode!='monitored':
            raise ValueError('A frozen monitored TerminationPolicy is required')
        if (policy.stage_handoff or policy.stage_order) and not (policy.stage_handoff and policy.stage_order in {('M2','M3'),('screening','refinement')}):
            raise ValueError('Unsupported: stage handoff execution is not verified; full-data refinement cannot be skipped')
        if policy.budget.max_njev is not None:
            raise ValueError('Unsupported: hard njev budget; SciPy callback does not report intermediate njev')
        super().__init__(recipe=recipe,job_id=job_id,F_ref=F_ref,record_budget=policy.budget.max_trace_records,
            objective_tolerance=policy.objective_change_tolerance,parameter_tolerance=policy.parameter_change_tolerance,patience=policy.patience)
        self._policy=policy;self.cancel_check=cancel_check;self.polish_used=False;self.outcome=None
        self._candidate_policy=MappingProxyType({**dict(self.candidate_policy),'effect':'explicit monitored AND stop; never statistical reliability'})

    @property
    def policy(self):return self._policy

    def initialize(self,dataset,config,**kwargs):
        from pyecr.contracts import FitControlBinding
        from .solver_strategy import SolverRejected
        requested=kwargs.get('solver_options')
        if self.policy.stage_handoff and getattr(self,'production_execution',None) is None:
            raise SolverRejected('Stage handoff requires the verified production service; standalone composition Unsupported')
        if requested is not None and requested!=self.policy.solver_options:
            raise SolverRejected('Monitored solver policy conflicts with configured solver; no automatic solver switch')
        super().initialize(dataset,config,**kwargs)
        self.binding=FitControlBinding.capture(self.binding.source,replace(self.binding.controls,termination=self.policy))
        self.outcome=dict(classification='running',qualified=False,reason='No completed native fit or reliability assessment',polish_used=False)

    @contextmanager
    def scope(self):
        token=_observer.set(self)
        try:yield self
        except BaseException as exc:
            self.state='cancelled' if isinstance(exc,CancelledError) else 'failed'
            self.outcome=dict(classification='cancelled' if isinstance(exc,CancelledError) else
                'stopped_candidate' if isinstance(exc,MonitoredStop) and not isinstance(exc,MonitoredNumericalError) else 'error',qualified=False,
                reason=str(exc),polish_used=self.polish_used)
            self.warnings.append(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if self.state=='running' and self.contexts:
                self.state='history'
                self.outcome=dict(classification='native_candidate',qualified=False,
                    reason='Native call returned; no production Fit finalization or gate evaluated',polish_used=self.polish_used)
            self.execution_elapsed_s=max(0.,monotonic()-self.started)
            _observer.reset(token)

    def check_time(self):
        if self.cancel_check:self.cancel_check()
        cap=self.policy.budget.max_wall_seconds
        if cap is not None and monotonic()-self.started>=cap:
            raise MonitoredBudgetExceeded('max_wall_seconds reached (cooperative boundary)')

    def check_counter(self,name,increment=1):
        self.check_time();cap=getattr(self.policy.budget,'max_'+name)
        if cap is not None and self.counters[name]+increment>cap:
            raise MonitoredBudgetExceeded('max_'+name+' reached')

    def before_count(self,kind,phase):
        self.check_time()
        if kind=='forward':
            self.check_counter('forward_calls')
            if phase=='diagnostic':self.check_counter('diagnostic_calls')
        elif kind=='residual' and phase=='diagnostic':self.check_counter('residual_calls')
        elif kind=='jacobian' and phase=='diagnostic':self.check_counter('diagnostic_calls')

    def before_diagnostic(self):
        self.check_counter('diagnostic_calls');self.check_counter('residual_calls')

    def _add_context(self,call):
        if len(self.contexts)>=self.max_contexts:
            self.contexts.pop();self.dropped_contexts+=1
            self.warnings.append('CONTEXT_BUDGET: first contexts and latest retained; execution counters remain whole-job')
        self.contexts.append(call)

    def solve_monitored(self,fun,start,context,options):
        if getattr(self,'production_execution',None) is not None:
            return self.production_execution.solve(fun,start,context,options)
        from .solver_strategy import SolverRejected
        if options.get('method','trf')!='trf':raise SolverRejected('Monitored Unsupported for solver other than TRF; no fallback')
        if context is None:raise SolverRejected('Monitored needs frozen production coordinate metadata')
        if options.get('callback') is not None or options.get('workers') is not None or options.get('jac')=='cs':
            raise SolverRejected('Monitored Unsupported: callback composition, workers or complex-step')
        if options.get('args') or options.get('kwargs') or options.get('jac_sparsity') is not None:
            raise SolverRejected('Monitored Unsupported: auxiliary args or sparse Jacobian composition')
        b=self.policy.budget;self.check_time()
        remaining=b.max_total_nfev-self.counters['native_nfev']
        if remaining<=0:raise MonitoredBudgetExceeded('max_total_nfev reached before new start/stage')
        caps=[remaining,options.get('max_nfev') or 100*len(start)]
        if b.max_nfev is not None:caps.append(b.max_nfev)
        actual={**options,'method':'trf','ftol':self.policy.ftol,'xtol':self.policy.xtol,'gtol':self.policy.gtol,'max_nfev':min(caps)}
        result,call=self._run(fun,start,context,actual,polish=False)
        if result.status==-2 and call.stop_reason in {'plateau','stagnated'} and self.policy.final_polish=='once_full_native' and not self.polish_used:
            self.check_time();remaining=b.max_total_nfev-self.counters['native_nfev']
            if remaining<=0:raise MonitoredBudgetExceeded('No total nfev budget remains for explicitly requested polish',native_status=-2)
            self.polish_used=True
            ftol,xtol,gtol=self.policy.polish_tolerances
            # Same complete residual callable, loss, bounds and units. No new
            # stage subset, normalization, model or observation objective.
            result,call=self._run(fun,result.x,{**context,'stage_id':'final_polish'},
                {**options,'method':'trf','ftol':ftol,'xtol':xtol,'gtol':gtol,
                 'max_nfev':min(remaining,self.policy.polish_max_nfev,b.max_nfev or remaining)},polish=True)
        if not result.success:
            raise MonitoredStop(call.stop_reason,native_status=int(result.status))
        return result  # the actual successful native object, never a status edit

    def _run(self,fun,start,context,options,*,polish):
        self.check_time();call=_MonitoredCall(self,start,options,context,polish=polish)
        self._add_context(call);jac=options.get('jac','2-point')
        def counted(q):
            self.check_counter('residual_calls');self.counters['residual_calls']+=1
            with evaluation_phase('optimizer_residual'):r=fun(q)
            call.remember_residual(q,r)
            if not np.all(np.isfinite(r)):raise MonitoredNumericalError('nonfinite residual evaluation')
            if not call.initial_seen:
                call.initial_seen=True;call.nfev=1
                lossfun=construct_loss_function(len(r),call.loss,call.f_scale)
                F=float(np.dot(r,r)/2) if lossfun is None else float(lossfun(np.asarray(r),cost_only=True))
                call.emit('initial',q,r,F)
            return r
        def diagnostic_residual(q):
            self.check_counter('diagnostic_calls');self.check_counter('residual_calls')
            self.counters['diagnostic_calls']+=1;self.counters['residual_calls']+=1
            with evaluation_phase('explicit_diagnostic'):r=fun(q)
            if not np.all(np.isfinite(r)):raise MonitoredNumericalError('nonfinite diagnostic residual')
            return r
        def diagnostic_jac(q,r):
            if callable(jac):
                self.check_counter('diagnostic_calls');self.counters['diagnostic_calls']+=1
                self.counters['analytic_jacobian_calls']+=1
                with evaluation_phase('explicit_diagnostic'):j=jac(q)
            else:
                j=approx_derivative(diagnostic_residual,q,method=jac,f0=r,bounds=(call.lo,call.hi),rel_step=options.get('diff_step'))
            j=np.atleast_2d(j)
            if not np.all(np.isfinite(j)):raise MonitoredNumericalError('nonfinite diagnostic Jacobian')
            return j
        call.gradient_jac=diagnostic_jac
        if callable(jac):
            def counted_jac(q):
                self.check_time();self.counters['analytic_jacobian_calls']+=1
                with evaluation_phase('optimizer_jacobian'):return jac(q)
            options={**options,'jac':counted_jac}
        result=None
        try:
            result=native_least_squares(counted,start,callback=call.callback,**options)
            call.nfev=int(result.nfev);call.njev=None if result.njev is None else int(result.njev)
            self.check_time()
            reason=call.stop_reason if result.status==-2 else {1:'native_gtol',2:'native_ftol',3:'native_xtol',4:'native_ftol_xtol',0:'budget'}.get(int(result.status),'failed')
            call.stop_reason=reason
            call.emit('terminal',result.x,call.raw_native_residual(result.x),float(result.cost),native=result,reason=reason)
            call.state='history'
            return result,call
        except BaseException as exc:
            call.warnings.append(f'NATIVE_EXCEPTION: {type(exc).__name__}: {exc}')
            if result is None:self.counters['native_nfev_complete']=False
            reason='cancelled' if isinstance(exc,CancelledError) else 'budget' if isinstance(exc,MonitoredBudgetExceeded) else 'nonfinite' if 'nonfinite' in str(exc).lower() else 'failed'
            call.stop_reason=reason
            if reason=='budget':call.warnings.append('Cooperative budget boundary reached; native subtotal may be incomplete; no successful finalization')
            if result is not None:
                call.emit('terminal',result.x,call.raw_native_residual(result.x),float(result.cost),native=result,reason=reason)
                call.records[-1]=replace(call.records[-1],finalization='rejected')
            elif call.previous is not None:call.emit('terminal',call.previous[2],call.last_fun,call.previous[0],reason=reason)
            else:call.emit('terminal',call.ref,[],float('nan'),reason=reason)
            call.state='cancelled' if isinstance(exc,CancelledError) else 'failed'
            if reason=='nonfinite' and not isinstance(exc,MonitoredNumericalError):
                raise MonitoredNumericalError('nonfinite execution: '+str(exc)) from exc
            raise
        finally:
            self.counters['native_nfev']+=call.nfev
            if call.njev is None:self.counters['native_njev_complete']=False
            else:self.counters['native_njev']+=call.njev
            call.residual_cache.clear()

    def finish_fit(self,fit):
        self.check_time()
        if fit.exit_status<=0:raise MonitoredStop('Production finalization is not a positive native result',native_status=fit.exit_status)
        if getattr(self,'production_execution',None) is not None:self.production_execution.finish(fit)
        super().finish_fit(fit)
        try:self.check_time()  # Full metadata/finalization cost is still in this job.
        except BaseException:
            self.final_source=self.final_metrics=self.final_residuals=None
            if getattr(self,'production_execution',None) is not None:self.production_execution.finalized=False
            raise
        # Trace cannot substitute for the existing reliability/Diagnostics gates.
        self.outcome=dict(classification={1:'native_stationarity',2:'native_ftol',3:'native_xtol',4:'native_ftol_xtol'}[fit.exit_status],qualified=False,
            reason='Actual native success; statistical qualification requires unchanged 95% Fit/Diagnostics gates',polish_used=self.polish_used)

    def snapshot(self):
        from pyecr.contracts.common import canonical_sha256
        value=super().snapshot();value.update(version=VERSION,mode='monitored',execution_outcome=self.outcome,
            budget_policy=self.policy.budget.to_dict(),polish_used=self.polish_used)
        if getattr(self,'production_execution',None) is not None:
            value.update(version='monitored-trf/3',production_execution=self.production_execution.snapshot())
        value['record_sha256']=canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'})
        return value

    def save(self,path):
        path=Path(path)
        if path.exists():raise ValueError('Trace file exists; use a new path')
        value=self.snapshot();stage=path.with_name(path.name+'.'+uuid4().hex+'.staging')
        try:
            stage.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            if read_monitored(stage.read_text(encoding='utf-8'))!=value:raise ValueError('Staging trace parity conflict')
            stage.rename(path)
        finally:
            if stage.exists():stage.unlink()
        return path


class _MonitoredCall(_Call):
    def __init__(self,*args,polish=False,**kwargs):
        super().__init__(*args,**kwargs);self.polish=polish;self.stop_reason='none';self.gradient_jac=None;self.endpoint_coordinates=None

    def emit(self,event,q,r,F,**kwargs):
        super().emit(event,q,r,F,**kwargs)
        self.endpoint_coordinates=np.asarray(q,dtype=float).copy() if self.records[-1].finite else None

    def callback(self,intermediate_result):
        # Intentionally does not inherit R08's observer-error suppression.
        # SciPy itself catches only StopIteration and returns untouched -2.
        self.obs.check_time();self.callbacks+=1;self.nfev=int(intermediate_result.nfev)
        self.native_nit=int(intermediate_result.nit)
        q=np.asarray(intermediate_result.x,dtype=float)
        if self.previous is None or np.array_equal(q,self.previous[2]):self.duplicates+=1;return
        r=self.raw_native_residual(q);self.emit('accepted',q,r,float(intermediate_result.cost))
        b=self.obs.policy.budget
        # Counters are advanced only by this accepted record. Diagnostic calls
        # below never advance the window, including finite differences.
        if (b.max_iterations is not None and self.accepted>=b.max_iterations) or (
                b.max_total_iterations is not None and self.obs.counters['accepted_iterations']>=b.max_total_iterations):
            self.stop_reason='budget';raise StopIteration
        j=self.gradient_jac(q,r)
        lossfun=construct_loss_function(len(r),self.loss,self.f_scale)
        g=j.T@(r if lossfun is None else lossfun(r)[1]*r)
        v,_=CL_scaling_vector(q,g,self.lo,self.hi)
        projected=float(np.linalg.norm(g*v,ord=np.inf))
        if not np.all(np.isfinite(g)) or not np.isfinite(projected):raise MonitoredNumericalError('nonfinite gradient')
        self.metrics[-1].update(diagnostic_gradient=g.tolist(),diagnostic_optimality=projected,
            gradient_source='explicit full-residual Jacobian and loss derivative; SciPy CL bound scaling; not native callback optimality',
            candidate_never_stops=False,polish=self.polish)
        if not self.polish and self.joint_count>=self.obs.policy.patience:
            self.stop_reason='stagnated' if projected>=self.obs.policy.gtol else 'plateau'
            raise StopIteration

    def snapshot(self):
        value=super().snapshot()
        value['monitored_execution']=dict(version=VERSION,polish=self.polish,stop_reason=self.stop_reason,
            termination_class='native_stationarity' if self.stop_reason=='native_gtol' else self.stop_reason,
            joint_window_completed=self.joint_count>=self.obs.policy.patience,
            policy_id=self.obs.binding.execution_policy_id,
            large_gradient_rule='diagnostic TRF CL-scaled infinity norm >= frozen gtol => stagnated',
            native_status_unchanged=True,
            endpoint_coordinates=None if self.endpoint_coordinates is None else self.endpoint_coordinates.tolist(),
            endpoint_raw_residuals=None if self.endpoint_coordinates is None else (self.last_fun if self.weights is None else self.last_fun/np.sqrt(self.weights)).tolist(),
            endpoint_weighted_residuals=None if self.endpoint_coordinates is None else self.last_fun.tolist(),
            observation_weights=None if self.weights is None else self.weights.tolist(),
            residual_role='native candidate residual (prediction minus observations in production; no FitResult substitution)',
            endpoint_unit=self.obs.unit)
        return value


def read_monitored(text):
    try:return _read_monitored(text)
    except (KeyError,TypeError,IndexError) as exc:
        raise ValueError('Invalid monitored trace: missing or malformed field '+str(exc)) from exc


def _read_monitored(text):
    from pyecr.contracts.common import strict_json_loads,canonical_sha256,require
    value=strict_json_loads(text)
    if value.get('version')=='monitored-trf/3':
        from .production_stop import read_production
        return read_production(value,_read_monitored)
    require(value.get('version')==VERSION and value.get('mode')=='monitored','Unknown monitored version/mode')
    require(value.get('record_sha256')==canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'}),'Monitored trace hash conflict')
    base=dict(value)
    for key in ('execution_outcome','budget_policy','polish_used'):require(key in base,'Missing monitored '+key);base.pop(key)
    base.update(version=OBS_VERSION,mode='readonly')
    base['record_sha256']=canonical_sha256({k:v for k,v in base.items() if k!='record_sha256'})
    parsed=read_observation(json.dumps(base,allow_nan=False))
    from pyecr.contracts import FitControlBinding
    binding=FitControlBinding.from_dict(parsed['binding'])
    require(binding.controls.termination.mode=='monitored','Monitored binding conflict')
    require(value['state'] in {'running','history','completed','cancelled','failed'},'Unknown monitored state')
    outcome=value['execution_outcome']
    require(outcome['classification'] in {'running','native_candidate','native_stationarity','native_ftol','native_xtol','native_ftol_xtol',
        'stopped_candidate','error','cancelled'} and outcome['qualified'] is False,'Trace cannot claim statistical qualification')
    require(isinstance(outcome['reason'],str) and bool(outcome['reason']),'Execution reason required')
    require(value['budget_policy']==binding.controls.termination.budget.to_dict(),'Budget policy conflict')
    require(type(value['polish_used']) is bool,'Invalid polish flag')
    require(outcome['polish_used']==value['polish_used'],'Outcome/polish conflict')
    polish_count=sum(c['monitored_execution']['polish'] for c in value['contexts'])
    require(polish_count<=1 and (polish_count==0 or value['polish_used']),'Polish identity conflict')
    for c in value['contexts']:
        info=c['monitored_execution'];require(info['version']==VERSION and info['policy_id']==binding.execution_policy_id,'Execution identity conflict')
        tr=c['accepted_trace'];terminal=tr['records'][-1];opts=c['native_options']
        if terminal['finite']:
            raw=np.asarray(info['endpoint_raw_residuals'],dtype=float);weighted=np.asarray(info['endpoint_weighted_residuals'],dtype=float)
            q=np.asarray(info['endpoint_coordinates'],dtype=float)
            require(raw.shape==weighted.shape==(terminal['full_observation_count'],) and np.all(np.isfinite(raw)) and np.all(np.isfinite(weighted)),'Incomplete candidate residual arrays')
            require(q.shape==(len(tr['parameter_names']),) and np.all(np.isfinite(q)),'Invalid candidate coordinates')
            require(np.isclose(float(raw@raw),terminal['SSE'],rtol=1e-12,atol=1e-14),'Candidate residual/SSE conflict')
            reconstructed=np.asarray(tr['coordinate_references'])+np.asarray(terminal['z'])*np.asarray(tr['coordinate_scales'])
            require(np.allclose(q,reconstructed,rtol=1e-12,atol=1e-14),'Candidate coordinate identity conflict')
            weights=info['observation_weights']
            require((weights is None)==(tr['weights_sha256'] is None),'Candidate weight identity conflict')
            if weights is None:require(np.array_equal(raw,weighted),'Unweighted candidate array conflict')
            else:
                w=np.asarray(weights,dtype=float)
                require(w.shape==raw.shape and np.all(np.isfinite(w)) and np.all(w>0),'Invalid candidate weights')
                require(canonical_sha256(weights)==tr['weights_sha256'] and np.allclose(raw*np.sqrt(w),weighted,rtol=1e-12,atol=1e-14),'Candidate weight/array conflict')
        else:require(all(info[k] is None for k in ('endpoint_coordinates','endpoint_raw_residuals','endpoint_weighted_residuals')),'Nonfinite terminal cannot claim finite endpoint arrays')
        policy=binding.controls.termination
        expected_tols=policy.polish_tolerances if info['polish'] else (policy.ftol,policy.xtol,policy.gtol)
        require(opts['method']=='trf' and tuple(opts[k] for k in ('ftol','xtol','gtol'))==tuple(expected_tols),'Actual native tolerance/method conflict')
        actual=replace(policy,ftol=opts['ftol'],xtol=opts['xtol'],gtol=opts['gtol'],
            budget=replace(policy.budget,max_nfev=opts['max_nfev_effective']))
        cb=FitControlBinding.from_dict(tr['binding'])
        require(cb.controls==replace(binding.controls,termination=actual),'Context execution binding conflict')
        require(type(info['polish']) is bool and info['native_status_unchanged'] is True,'Invalid execution flags')
        require(info['stop_reason']==terminal['stop_reason'],'Stop reason conflict')
        if terminal['native_status']==-2:require(terminal['native_success'] is False,'Callback status cannot be relabelled')
        require(info['joint_window_completed']==(c['metrics'][-1]['joint_patience_count']>=policy.patience),'Window completion conflict')
        if not tr['dropped_records']:
            joint=objective=0
            for rec,row in zip(tr['records'],c['metrics']):
                if rec['event']=='accepted':
                    objective=objective+1 if rec['rho_F']<policy.objective_change_tolerance else 0
                    joint=joint+1 if rec['rho_F']<policy.objective_change_tolerance and rec['rho_z']<policy.parameter_change_tolerance else 0
                require(row['joint_patience_count']==joint and row['objective_patience_count']==objective,'Accepted AND window conflict')
        lo=np.array([float('-inf') if x is None else x for x in c['optimizer_bounds']['lower']])
        hi=np.array([float('inf') if x is None else x for x in c['optimizer_bounds']['upper']])
        for rec,row in zip(tr['records'],c['metrics']):
            if 'diagnostic_gradient' in row:
                g=np.asarray(row['diagnostic_gradient'],dtype=float)
                require(g.shape==lo.shape and np.all(np.isfinite(g)),'Invalid diagnostic gradient')
                q=np.asarray(tr['coordinate_references'])+np.asarray(rec['z'])*np.asarray(tr['coordinate_scales'])
                v,_=CL_scaling_vector(q,g,lo,hi)
                require(np.isclose(row['diagnostic_optimality'],np.linalg.norm(g*v,ord=np.inf),rtol=1e-10,atol=1e-14),'Diagnostic optimality conflict')
    return value
