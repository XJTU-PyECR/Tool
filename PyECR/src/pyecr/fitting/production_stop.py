"""Explicit job-level ensemble selection and one full-objective native polish.

No model, residual arithmetic, candidate status edits, or reliability rules.
Production fitters retain their original candidates and finalize the separate
polished native endpoint through their existing full-data scientific paths.
"""
from dataclasses import replace
from contextlib import contextmanager
from time import monotonic
import json
import numpy as np
from .iteration_observation import _observer
from .solver_strategy import SolverRejected
from pyecr.contracts.common import canonical_sha256,require

VERSION='monitored-production/2'  # v2 declares each stage's actual observation rows

def execution():
    return getattr(_observer.get(),'production_execution',None)

def candidate_eligible(result):
    """A stopped candidate is eligible for review/polish, never publication."""
    if result.success and result.status>0:return True
    ex=execution()
    return bool(ex and result.status==-2 and any(e['result'] is result and e['call'].stop_reason in {'plateau','stagnated'} for e in ex.entries))

def polish_ensemble(stage,reason='Minimum actual native F then original start order within the selected production objective'):
    ex=execution()
    return None if ex is None else ex.polish(stage,reason)

def polish_joint(best,*,phase,decode,predict,jacobian,observed):
    output=polish_ensemble(phase)
    if output is None:return best
    result,index=output;params=decode(result.x);prediction=np.asarray(predict(params));r=np.asarray(observed)-prediction
    j=np.asarray(jacobian(params) if jacobian else result.jac)
    return replace(best,index=index,params=params,coordinates=np.array(result.x),prediction=prediction,
        residuals=r,sse=float(r@r),jacobian=j,status=int(result.status),nfev=int(result.nfev))

class ProductionExecution:
    def __init__(self,obs,*,screening_enabled=False):
        self.obs=obs;self.entries=[];self.selection=None;self.polish_record=None;self.finalized=False
        self.stage_contexts=[];self.handoffs=[];self.visited_stages=[]
        self.screening_enabled=screening_enabled;self.active_objective=None

    @contextmanager
    def objective_scope(self,stage,positions):
        previous=self.active_objective
        self.active_objective=(stage,tuple(int(i) for i in positions))
        try:yield
        finally:self.active_objective=previous

    def check_stage(self,stage):
        if not self.obs.policy.stage_handoff:return
        order=self.obs.policy.stage_order
        if not self.visited_stages or stage!=self.visited_stages[-1]:
            expected=order[len(self.visited_stages)] if len(self.visited_stages)<len(order) else None
            if stage!=expected:raise SolverRejected('Declared stage order conflict; no automatic stage/solver switch')
            self.visited_stages.append(stage)

    def record_stage(self,call,scope):
        current=dict(context_id=call.id,stage_id=call.stage,production_start=dict(call.options['production_start']),
            **scope,
            coordinate_names=list(call.names),coordinate_kinds=list(call.kinds),references=call.ref.tolist(),scales=call.scales.tolist(),
            F_ref=call.F_ref,loss=call.loss,f_scale=call.f_scale,initial_window_count=call.metrics[0]['joint_patience_count'],
            initial_accepted=call.records[0].accepted_iteration,global_before_native_nfev=call.base_nfev)
        if self.stage_contexts and self.stage_contexts[-1]['stage_id']!=current['stage_id']:
            previous=self.stage_contexts[-1]
            common=set(previous['coordinate_names']) & set(current['coordinate_names'])
            self.handoffs.append(dict(from_stage=previous['stage_id'],to_stage=current['stage_id'],
                previous_context=previous['context_id'],next_context=current['context_id'],
                coordinate_role_mapping={n:n for n in sorted(common)},new_coordinate_roles=[n for n in current['coordinate_names'] if n not in common],
                parameter_transfer='Unchanged production seeded starts in the next model coordinate codec; no copied optimizer coordinates or patience',
                global_native_nfev=call.base_nfev,initial_window_count=current['initial_window_count']))
        self.stage_contexts.append(current)

    def solve(self,fun,start,context,options):
        from .monitored import MonitoredStop,MonitoredBudgetExceeded
        obs=self.obs;b=obs.policy.budget;obs.check_time()
        if options.get('method','trf')!='trf' or context is None:
            raise SolverRejected('Monitored production requires explicit TRF coordinate metadata; no fallback')
        if options.get('callback') is not None or options.get('workers') is not None or options.get('jac')=='cs':
            raise SolverRejected('Monitored production rejects external callbacks, workers and complex-step')
        if options.get('args') or options.get('kwargs') or options.get('jac_sparsity') is not None:
            raise SolverRejected('Monitored production rejects auxiliary arguments and sparse Jacobian composition')
        positions=tuple(range(len(obs.dataset.processed_time)))
        context=dict(context)
        if self.active_objective is not None:
            stage,positions=self.active_objective
            context.update(model_stage=context.get('stage_id',obs.config.model_id.value),stage_id=stage)
        scope=dict(kind='full' if positions==tuple(range(len(obs.dataset.processed_time))) else 'subset',
            processed_positions=list(positions),raw_indices=[int(np.flatnonzero(obs.dataset.mask)[i]) for i in positions],
            processed_time_s=[float(obs.dataset.processed_time[i]) for i in positions],full_observation_count=len(positions),
            full_job_observation_count=len(obs.dataset.processed_time))
        remaining=b.max_total_nfev-obs.counters['native_nfev']
        if remaining<=0:raise MonitoredBudgetExceeded('No whole-job nfev remains before start/stage')
        caps=[remaining,options.get('max_nfev') or 100*len(start),b.max_nfev or remaining]
        actual={**options,'method':'trf','ftol':obs.policy.ftol,'xtol':obs.policy.xtol,'gtol':obs.policy.gtol,'max_nfev':min(caps)}
        self.check_stage(context.get('stage_id',obs.config.model_id.value))
        complete_N=obs.N;obs.N=len(positions)
        try:result,call=obs._run(fun,start,context,actual,polish=False)
        finally:obs.N=complete_N
        self.record_stage(call,scope)
        entry=dict(result=result,call=call,fun=fun,options=dict(options),context=dict(context),scope=scope,start_index=call.options['production_start']['index_1based'])
        self.entries.append(entry)
        # P60 intentionally ranks capped coarse endpoints. Native 0 stays 0;
        # these subset endpoints can never enter the final-polish pool.
        if self.screening_enabled and call.stage=='screening' and result.status==0:return result
        if result.status==0 or call.stop_reason=='budget':raise MonitoredBudgetExceeded('Native start budget exhausted',native_status=int(result.status))
        if not candidate_eligible(result):raise MonitoredStop(call.stop_reason,native_status=int(result.status))
        if result.status==-2 and call.stage=='M2' and obs.config.model_id.value=='M3_DUAL_HO_DK' and not obs.policy.stage_handoff:
            raise MonitoredStop('Explicit M2 to M3 stage handoff required for stopped intermediate candidates',native_status=-2)
        return result

    def polish(self,stage,reason):
        from .monitored import MonitoredStop,MonitoredBudgetExceeded
        obs=self.obs;require(not obs.polish_used,'Only one global production polish is allowed')
        if obs.policy.stage_handoff:
            require(tuple(self.visited_stages)==obs.policy.stage_order,'All declared production stages must run before global polish')
        candidates=[e for e in self.entries if e['context'].get('model_stage',e['call'].stage)==stage and e['scope']['kind']=='full' and e['call'].stage!='screening']
        require(candidates,'No actual native candidates for selected production stage')
        selected=min(enumerate(candidates),key=lambda row:(float(row[1]['result'].cost),row[0]))[1]
        result=selected['result'];call=selected['call'];b=obs.policy.budget
        rows=tuple(int(i) for i in np.flatnonzero(obs.dataset.mask))
        require(np.shape(result.fun)==(len(rows),),'Final polish requires the complete observation objective')
        self.selection=dict(stage_id=stage,source_stage_id=call.stage,context_id=call.id,reason=reason,
            rank_rule='minimum actual native cost F; stable original call order for ties',candidate_context_ids=[e['call'].id for e in candidates],
            source=obs.binding.source.to_dict(),raw_indices=list(rows),processed_time_s=obs.dataset.processed_time.tolist(),
            full_observation_count=len(rows),loss=call.loss,f_scale=call.f_scale,native_status=int(result.status),native_success=bool(result.success),
            F=float(result.cost),raw_SSE=call.records[-1].SSE,coordinates=result.x.tolist(),
            coordinate_names=list(call.names),coordinate_references=call.ref.tolist(),coordinate_scales=call.scales.tolist(),
            bounds=call.snapshot()['optimizer_bounds'],start_index_1based=selected['start_index'])
        self.selection['sha256']=canonical_sha256(self.selection)
        obs.check_time();remaining=b.max_total_nfev-obs.counters['native_nfev']
        if remaining<=0:raise MonitoredBudgetExceeded('No whole-job nfev remains for global polish')
        if b.max_total_iterations is not None and obs.counters['accepted_iterations']>=b.max_total_iterations:
            raise MonitoredBudgetExceeded('No whole-job accepted-iteration budget remains for global polish')
        before=dict(obs.counters);obs.polish_used=True
        ftol,xtol,gtol=obs.policy.polish_tolerances
        polished,pcall=obs._run(selected['fun'],np.array(result.x,copy=True),{**selected['context'],'stage_id':'final_polish'},
            {**selected['options'],'method':'trf','ftol':ftol,'xtol':xtol,'gtol':gtol,
             'max_nfev':min(remaining,b.max_nfev or remaining,obs.policy.polish_max_nfev)},polish=True)
        self.polish_record=dict(context_id=pcall.id,parent_context_id=call.id,raw_indices=list(rows),
            before=before,after=dict(obs.counters),native_status=int(polished.status),native_success=bool(polished.success),
            coordinates=polished.x.tolist(),F=float(polished.cost),endpoint_raw_residuals=pcall.snapshot()['monitored_execution']['endpoint_raw_residuals'])
        if not polished.success or polished.status<=0:raise MonitoredStop('global final polish '+pcall.stop_reason,native_status=int(polished.status))
        require(np.shape(polished.fun)==(len(rows),) and np.all(np.isfinite(polished.fun)) and np.all(np.isfinite(polished.jac)),
            'Polish has invalid full-data arrays')
        if self.screening_enabled:
            from .coarse_to_fine import current_screening
            ledger=current_screening();require(ledger is not None and len(ledger.stages)==1,'Verified P60 single-model ledger required')
            ledger.stages[0].update(status='completed',global_polish=dict(context_id=pcall.id,parent_context_id=call.id,
                native_status=int(polished.status),native_success=bool(polished.success),raw_indices=list(rows)))
        return polished,(selected['start_index'] or 1)-1

    def finish(self,fit):
        require(self.polish_record and self.polish_record['native_status']>0,'Production final polish did not succeed')
        require(fit.exit_status==self.polish_record['native_status'],'Production Fit must retain polish native terminal status')
        r=np.asarray(self.polish_record['endpoint_raw_residuals'])
        require(r.shape==fit.residuals.shape==(self.obs.N,) and np.all(np.isfinite(fit.y_fit)),'Full observation finalization required')
        require(np.allclose(r,-fit.residuals,rtol=2e-10,atol=2e-12) or np.allclose(r,fit.residuals,rtol=2e-10,atol=2e-12),
            'Finalized Fit/residual differs from selected global polish objective')
        self.finalized=True

    def snapshot(self):
        candidates=[]
        for e in self.entries:
            c=e['call'];r=e['result']
            candidates.append(dict(context_id=c.id,stage_id=c.stage,start_index_1based=e['start_index'],
                objective_scope=e['scope'],
                native_status=int(r.status),native_success=bool(r.success),native_nfev=int(r.nfev),native_njev=None if r.njev is None else int(r.njev),
                accepted=c.accepted,F=float(r.cost),coordinates=r.x.tolist(),raw_residuals=c.snapshot()['monitored_execution']['endpoint_raw_residuals']))
        value=dict(version=VERSION,candidates=candidates,selection=self.selection,polish=self.polish_record,
            declared_stage_order=list(self.obs.policy.stage_order),stage_contexts=self.stage_contexts,handoffs=self.handoffs,
            finalized=self.finalized,final_counters=dict(self.obs.counters),
            elapsed_seconds=getattr(self.obs,'execution_elapsed_s',None),
            auxiliary_linear_iterations='not nonlinear accepted iterations; forward hooks included in whole-job calls')
        return json.loads(json.dumps(value,allow_nan=False))

def configure_production(obs,config,screening):
    if obs.state!='not_started':raise ValueError('Observer is single-use; previous frozen execution is retained')
    if obs.policy.final_polish!='disabled' or obs.policy.stage_handoff or (screening is not None and screening.enabled):
        from pyecr.config import AnalysisConfig
        from pyecr.contracts.fit_controls import SIX_MODELS
        if not isinstance(config,AnalysisConfig) or config.model_id.value not in SIX_MODELS:
            raise SolverRejected('Production Monitored composition requires a supported six-model AnalysisConfig')
        if any(config.model_specific.get(section,{}).get('jacobian_mode','legacy')!='legacy' for section in ('single_ecr','dual_ho')):
            raise SolverRejected('Production polish with experimental hybrid Jacobian Unsupported until separately validated')
        if config.fit.loss!='linear' and not (config.model_id.value=='M0_SINGLE_D' and config.fit.loss=='soft_l1'):
            raise SolverRejected('Production polish loss/model combination Unsupported; verified robust scope is M0 soft_l1')
    if screening is not None and screening.enabled:
        if (not obs.policy.stage_handoff or obs.policy.stage_order!=('screening','refinement') or obs.policy.final_polish=='disabled'
                or config.model_id.value not in {'M0_SINGLE_D','M1_SINGLE_DK'} or 'fixed_d' in config.model_specific
                or config.fit.loss!='linear' or config.model.boundary_mode!='ideal_step'
                or config.model_specific.get('single_ecr',{}).get('jacobian_mode','legacy')!='legacy'):
            raise SolverRejected('P60 Monitored Unsupported combination: verified M0/M1 ideal-step linear numeric-Jacobian screening/refinement and one full polish required')
        obs.production_execution=ProductionExecution(obs,screening_enabled=True);return
    if obs.policy.stage_handoff:
        if obs.policy.final_polish=='disabled' or config.model_id.value!='M3_DUAL_HO_DK' or 'fixed_d' in config.model_specific or obs.policy.stage_order!=('M2','M3'):
            raise SolverRejected('Unsupported stage handoff combination; requires explicit M2/M3 production order and one final polish')
    if obs.policy.final_polish=='disabled':return
    obs.production_execution=ProductionExecution(obs)

def read_production(value,read_base):
    """Validate the new envelope and the unchanged R09 native/trace contract."""
    data=dict(value);production=data.pop('production_execution')
    data['version']='monitored-trf/2';data['record_sha256']=canonical_sha256({k:v for k,v in data.items() if k!='record_sha256'})
    read_base(json.dumps(data,allow_nan=False))
    require(value['record_sha256']==canonical_sha256({k:v for k,v in value.items() if k!='record_sha256'}),'Production trace hash conflict')
    require(production['version'] in {'monitored-production/1',VERSION},'Unknown production execution version')
    require(production['final_counters']==value['counters'],'Production whole-job counters conflict')
    for row in production.get('stage_contexts',()):
        require(row['initial_window_count']==row['initial_accepted']==0,'Stage/start window was carried across contexts')
        if production['version']=='monitored-production/1':
            require(row['raw_indices']==value['observations']['raw_indices'],'Legacy full-stage row scope conflict')
            continue
        positions=row['processed_positions'];original=value['observations']
        require(positions==sorted(set(positions)) and all(type(i) is int and 0<=i<original['N'] for i in positions),'Stage positions conflict')
        require(row['raw_indices']==[original['raw_indices'][i] for i in positions]
            and row['processed_time_s']==[original['processed_time_s'][i] for i in positions],'Stage observation scope conflict')
        require(row['kind'] in {'full','subset'} and row['full_job_observation_count']==original['N']
            and (row['kind']=='full')==(positions==list(range(original['N']))) and row['full_observation_count']==len(positions),'Stage objective extent conflict')
    for transition in production.get('handoffs',()):
        require(transition['initial_window_count']==0,'Stage handoff reused an accepted window')
    selection=production['selection'];polish=production['polish'];contexts={c['accepted_trace']['records'][0]['start_id']:c for c in value['contexts']}
    for e in production['candidates']:
        if e['context_id'] not in contexts:continue
        c=contexts[e['context_id']];last=c['accepted_trace']['records'][-1]
        require((e['native_status'],e['native_success'],e['F'])==(last['native_status'],last['native_success'],last['F']),'Original candidate terminal conflict')
        require(e['raw_residuals']==c['monitored_execution']['endpoint_raw_residuals'],'Original candidate arrays conflict')
    if selection:
        require(selection['sha256']==canonical_sha256({k:v for k,v in selection.items() if k!='sha256'}),'Frozen selection hash conflict')
        require(selection['raw_indices']==value['observations']['raw_indices'],'Polish observation row set conflict')
        chosen=next(e for e in production['candidates'] if e['context_id']==selection['context_id'])
        require(chosen['F']==selection['F'] and chosen['native_status']==selection['native_status'],'Selection/candidate conflict')
        require(chosen['coordinates']==selection['coordinates'] and selection['source']==value['binding']['source'],'Selection coordinates/source conflict')
        ranked=[next(e for e in production['candidates'] if e['context_id']==key) for key in selection['candidate_context_ids']]
        require(chosen['context_id']==min(enumerate(ranked),key=lambda r:(r[1]['F'],r[0]))[1]['context_id'],'Selection did not use frozen actual F rank')
    if polish:
        require(selection and polish['parent_context_id']==selection['context_id'],'Polish parent conflict')
        require(polish['raw_indices']==selection['raw_indices'],'Polish/full observation conflict')
        c=contexts[polish['context_id']];native=c['accepted_trace']['records'][-1];info=c['monitored_execution']
        require((polish['native_status'],polish['native_success'],polish['F'])==(native['native_status'],native['native_success'],native['F']),
            'Polish native terminal conflict')
        require(polish['coordinates']==info['endpoint_coordinates'] and polish['endpoint_raw_residuals']==info['endpoint_raw_residuals'],'Polish endpoint arrays conflict')
        require(polish['after']['native_nfev']==polish['before']['native_nfev']+native['nfev'],'Polish nfev conservation conflict')
        require(polish['after']['accepted_iterations']==polish['before']['accepted_iterations']+native['accepted_iteration'],'Polish accepted conservation conflict')
        require(sum(e['native_nfev'] for e in production['candidates'])+native['nfev']==value['counters']['native_nfev'],'Whole-job nfev conservation conflict')
        require(sum(e['accepted'] for e in production['candidates'])+native['accepted_iteration']==value['counters']['accepted_iterations'],'Whole-job accepted conservation conflict')
    for name in ('native_nfev','accepted_iterations','residual_calls','forward_calls','diagnostic_calls'):
        cap=value['budget_policy'].get({'native_nfev':'max_total_nfev','accepted_iterations':'max_total_iterations'}.get(name,'max_'+name))
        require(type(value['counters'][name]) is int and value['counters'][name]>=0 and (cap is None or value['counters'][name]<=cap),'Whole-job budget exceeded: '+name)
    if production['finalized'] or value['state']=='completed':
        require(polish and polish['native_success'] and polish['native_status']>0 and production['finalized'],'No positive finalized global polish')
    return value
