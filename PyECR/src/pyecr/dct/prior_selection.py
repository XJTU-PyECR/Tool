"""Budgeted, explicit prior search. No truth access, peak-count score or publish.

Adaptive fits cannot use frozen-R df as nonlinear GCV. Each CV training fold
recomputes its own normalization AND both UP initializations/weights.
"""
from dataclasses import dataclass,asdict,replace,fields
from math import log10,isfinite
from time import perf_counter
import numpy as np
from pyecr.contracts import SignedDctConfig,DctSourceSnapshot
from pyecr.contracts.common import require,integer,number
from pyecr.dct.signed import fit_signed_dct
from pyecr.dct.signed_linear import LinearCache
from pyecr.dct.signed_prior_kernel import count_subproblems
from pyecr.dct.linear_selection import _fold_sources
from pyecr.dct.execution import check_cancel,run_numeric


@dataclass(frozen=True,slots=True)
class PriorSearchPolicy:
    version: str='signed-prior-search/1'
    strategy: str='gcv'
    coarse_lambdas: tuple=tuple(float(x) for x in np.logspace(-10,-4,7))
    lambda_min: float=1e-14
    lambda_max: float=1.
    expansions: int=2
    refinement_rounds: int=1
    max_candidates: int=13
    max_evaluations: int=39
    max_subproblems: int=5000
    max_seconds: float=180.
    tie_rtol: float=1e-10
    tie_atol: float=1e-18
    tie_rule: str='largest_lambda_within_score_tie'
    folds: int=2
    purge_seconds: float=9.
    cache_bytes: int=128*1024**2
    numerical_sd_relative_S: float=1e-8
    normalization_policy: str='external_terminal_train_maxabs_per_fold'

    def __post_init__(self):
        require(self.version=='signed-prior-search/1' and self.strategy in {'gcv','discrepancy','numerical_discrepancy','blocked_cv'},'unknown prior selection strategy/version')
        values=tuple(number(v,'coarse lambda',positive=True) for v in self.coarse_lambdas)
        require(len(values)>=3 and all(a<b for a,b in zip(values,values[1:])),'lambda path must increase')
        object.__setattr__(self,'coarse_lambdas',values)
        for name in ('lambda_min','lambda_max','max_seconds','numerical_sd_relative_S'):number(getattr(self,name),name,positive=True)
        require(self.lambda_min<=values[0]<values[-1]<=self.lambda_max,'coarse lambda outside bounds')
        for name in ('max_candidates','max_evaluations'):integer(getattr(self,name),name,minimum=len(values))
        for name in ('expansions','refinement_rounds','cache_bytes'):integer(getattr(self,name),name)
        integer(self.max_subproblems,'max_subproblems',minimum=1);integer(self.folds,'folds',minimum=2)
        for name in ('tie_rtol','tie_atol','purge_seconds'):number(getattr(self,name),name,nonnegative=True)
        require(self.tie_rule=='largest_lambda_within_score_tie' and self.normalization_policy=='external_terminal_train_maxabs_per_fold','unknown selection tie/normalization policy')

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,value):
        require(set(value)=={f.name for f in fields(cls)},'unknown/missing prior selection fields')
        return cls(**value)


@dataclass(frozen=True,slots=True)
class PriorFoldEvidence:
    train_indices: tuple
    validation_indices: tuple
    training_terminal: float
    training_S: float
    result: object
    status: str
    weighted_physical_sse: float | None

    def to_dict(self):
        return dict(train_indices=self.train_indices,validation_indices=self.validation_indices,
            training_terminal=self.training_terminal,training_S=self.training_S,result=self.result.to_dict(),
            status=self.status,weighted_physical_sse=self.weighted_physical_sse)


@dataclass(frozen=True,slots=True)
class PriorTrial:
    result: object
    score: float | None
    failure: str | None
    seconds: float
    subproblems: int
    fold_evidence: tuple=()

    def to_dict(self):
        return dict(result=self.result.to_dict(),score=self.score,failure=self.failure,seconds=self.seconds,
            subproblems=self.subproblems,fold_evidence=[f.to_dict() for f in self.fold_evidence])


@dataclass(frozen=True,slots=True)
class PriorSearchReport:
    source: object
    config: object
    policy: PriorSearchPolicy
    trials: tuple
    selected_lambda: float | None
    tied_lambdas: tuple
    termination: str
    evaluations: int
    subproblems: int
    seconds: float
    cache: tuple
    physical_noise_sd: tuple=()

    @property
    def selected_result(self):
        return next((t.result for t in self.trials if t.result.config.grid.regularization_lambda==self.selected_lambda),None) if self.selected_lambda is not None else None

    def to_dict(self):
        return dict(version='signed-prior-search-report/1',source=self.source.to_dict(),config=self.config.to_dict(),policy=self.policy.to_dict(),
            trials=[t.to_dict() for t in self.trials],selected_lambda=self.selected_lambda,tied_lambdas=self.tied_lambdas,
            termination=self.termination,evaluations=self.evaluations,subproblems=self.subproblems,seconds=self.seconds,
            cache=dict(self.cache),physical_noise_sd=self.physical_noise_sd,
            semantics='Fixed-linear GCV only; adaptive df is conditional frozen-R, NOT nonlinear GCV. No CI, species assignment or current publication.')


def prior_score_ties(trials,policy):
    valid=[t for t in trials if t.failure is None and t.score is not None]
    if not valid:return (),None
    minimum=min(t.score for t in valid)
    tied=tuple(sorted(t.result.config.grid.regularization_lambda for t in valid if abs(t.score-minimum)<=policy.tie_atol+policy.tie_rtol*abs(minimum)))
    return tied,max(tied)


def search_signed_priors(source,config,policy=PriorSearchPolicy(),*,noise_sd_physical=(),cancel_check=None,progress_callback=None):
    require(isinstance(source,DctSourceSnapshot) and isinstance(config,SignedDctConfig) and config.schema_version==3,'explicit prior source/config required')
    source.require_usable();require(config.grid.lambda_mode=='manual','prior search starts from a manual versioned config')
    require(not config.prior_spec.adaptive or policy.strategy!='gcv','frozen UP df is not the full nonlinear GCV')
    require(not config.prior_spec.hard_zero_control,'hard-zero controls cannot be promoted by automatic search')
    check=cancel_check or check_cancel;progress=progress_callback or (lambda f,m:None);check()
    cache=LinearCache(policy.cache_bytes);start=perf_counter();trials=[];seen=set();evaluations=0;inner=0
    folds=_fold_sources(source,config,policy) if policy.strategy=='blocked_cv' else ()
    cost=1+len(folds);require(cost*len(policy.coarse_lambdas)<=policy.max_evaluations,'budget cannot cover the coarse grid and all folds')
    target=None;noise=()
    if policy.strategy=='discrepancy':
        sd=np.asarray(noise_sd_physical,dtype=float)
        require(sd.shape==(len(source.signal),) and np.all(np.isfinite(sd)) and np.all(sd>0),'trusted physical SD must cover every observation')
        noise=tuple(float(v) for v in sd)
        weights=np.asarray(config.observation_weights or np.ones(len(sd)))
        target=float(np.sum(weights*(sd/source.normalization.scale_value)**2))
    elif policy.strategy=='numerical_discrepancy':
        require(not len(noise_sd_physical),'numerical tolerance must not masquerade as measured noise')
        target=float(sum(config.observation_weights or np.ones(len(source.signal)))*policy.numerical_sd_relative_S**2)
    else:require(not len(noise_sd_physical),'unused noise metadata')
    if target is not None:require(np.isfinite(target) and target>0,'invalid discrepancy target')
    def run(snapshot,cfg):
        nonlocal evaluations,inner
        check()
        with count_subproblems() as counter:
            result=(fit_signed_dct(snapshot,cfg,linear_cache=cache) if cancel_check is None else
                run_numeric(False,check,fit_signed_dct,snapshot,cfg,linear_cache=cache))
        evaluations+=1;inner+=counter[0];check()
        return result,counter[0]
    def evaluate(lam):
        began=perf_counter();cfg=replace(config,grid=replace(config.grid,regularization_lambda=float(lam)))
        result,actual=run(source,cfg);failure=None;score=None;fold_records=[]
        if not result.state.ready:failure='; '.join(result.state.blocking_reasons)
        elif policy.strategy=='gcv':
            df=result.solve_diagnostics.effective_df;n=len(source.signal)
            if n-df<=64*np.finfo(float).eps*n:failure='GCV denominator saturated'
            else:score=float(n*result.solve_diagnostics.weighted_sse/(n-df)**2)
        elif policy.strategy in {'discrepancy','numerical_discrepancy'}:
            score=abs(float(np.log(max(result.solve_diagnostics.weighted_sse,np.finfo(float).tiny)/target)))
        else:
            total=0.;count=0
            for fs,fc,train,val,physical in folds:
                fr,ni=run(fs,replace(fc,grid=cfg.grid));actual+=ni
                record=PriorFoldEvidence(train_indices=tuple(int(i) for i in train),validation_indices=tuple(int(i) for i in val),
                    training_terminal=fs.normalization.reference_value,training_S=fs.normalization.scale_value,
                    result=fr,status='FAIL',weighted_physical_sse=None)
                if not fr.state.ready:failure='CV outer solve failed: '+'; '.join(fr.state.blocking_reasons)
                else:
                    with np.errstate(over='ignore',under='ignore'):
                        pred=np.exp(-np.asarray(source.time_s)[val,None]/np.asarray(fr.tau_s)[None,:])@np.asarray(fr.quadrature_mass)
                    pred=fs.normalization.reference_value+fs.normalization.scale_value*pred
                    weights=np.asarray(config.observation_weights or np.ones(len(source.signal)))[val]
                    sse=float(np.sum(weights*(pred-physical[val])**2))
                    if not np.isfinite(sse):failure='nonfinite held-out prediction'
                    else:total+=sse;count+=len(val);record=replace(record,status='PASS',weighted_physical_sse=sse)
                fold_records.append(record)
            if failure is None:score=total/count
        if score is not None and not isfinite(score):failure='nonfinite score';score=None
        if failure is not None:score=None
        trials.append(PriorTrial(result,score,failure,perf_counter()-began,actual,tuple(fold_records)))
        seen.add(float(lam));progress(min(1.,evaluations/policy.max_evaluations),f'Prior lambda {lam:g}: '+(failure or 'scored'));check()
    def add(values):
        for lam in sorted(set(values)-seen):
            if len(trials)>=policy.max_candidates or evaluations+cost>policy.max_evaluations:return 'evaluation_budget'
            # Reserve the declared worst case before starting a candidate,
            # including every fold and both initializations. No partial path
            # can silently overshoot the numerical subproblem budget.
            reserve=cost*(1+2*config.prior_spec.max_iterations if config.prior_spec.adaptive else 1)
            if inner+reserve>policy.max_subproblems:return 'inner_subproblem_budget'
            if perf_counter()-start>=policy.max_seconds:return 'wall_time_budget'
            evaluate(lam)
        return None
    termination=add(policy.coarse_lambdas);expanded=refined=0
    while termination is None:
        ties,best=prior_score_ties(trials,policy)
        if best is None:termination='all_candidates_failed';break
        ordered=sorted(seen);index=ordered.index(best)
        if index in (0,len(ordered)-1):
            if expanded>=policy.expansions:termination='boundary_expansion_budget';break
            direction=-1 if index==0 else 1
            proposed=[float(10**min(log10(policy.lambda_max),max(log10(policy.lambda_min),log10(best)+direction*d))) for d in (1.,2.)]
            expanded+=1
            if set(proposed)<=seen:termination='lambda_range_limit';break
        else:
            if refined>=policy.refinement_rounds:termination='refinement_budget';break
            proposed=[float(10**((log10(ordered[index-1])+log10(best))/2)),float(10**((log10(best)+log10(ordered[index+1]))/2))];refined+=1
        termination=add(proposed)
    ties,best=prior_score_ties(trials,policy);check()
    return PriorSearchReport(source,config,policy,tuple(trials),best,ties,termination,evaluations,inner,perf_counter()-start,
        tuple(cache.telemetry().items()),noise)


def read_prior_search(value):
    """Strict read-only replay, including every fold's freshly estimated weights."""
    from pyecr.contracts import read_contract
    import json
    names={'version','source','config','policy','trials','selected_lambda','tied_lambdas','termination','evaluations','subproblems',
        'seconds','cache','physical_noise_sd','semantics'}
    require(set(value)==names and value['version']=='signed-prior-search-report/1','unknown prior search report')
    source=read_contract(value['source']);config=read_contract(value['config']);policy=PriorSearchPolicy.from_dict(value['policy'])
    require(config.schema_version==3 and config.grid.lambda_mode=='manual','prior report config conflict')
    require(not config.prior_spec.adaptive or policy.strategy!='gcv','nonlinear frozen-weight GCV is forbidden')
    noise=np.asarray(value['physical_noise_sd'],dtype=float)
    require((noise.shape==(len(source.signal),) and np.all(np.isfinite(noise)) and np.all(noise>0))
        if policy.strategy=='discrepancy' else noise.size==0,'saved noise metadata conflicts with strategy/observations')
    require(len(value['trials'])<=policy.max_candidates,'candidate budget conflict')
    require(value['termination'] in {'evaluation_budget','inner_subproblem_budget','wall_time_budget','all_candidates_failed',
        'boundary_expansion_budget','lambda_range_limit','refinement_budget'},'unknown search termination')
    number(value['seconds'],'search seconds',nonnegative=True)
    expected_folds=_fold_sources(source,config,policy) if policy.strategy=='blocked_cv' else ()
    trials=[];seen=set()
    for item in value['trials']:
        require(set(item)=={'result','score','failure','seconds','subproblems','fold_evidence'},'unknown/missing prior trial fields')
        result=read_contract(item['result']);lam=result.config.grid.regularization_lambda
        require(result.source==source and result.config==replace(config,grid=replace(config.grid,regularization_lambda=lam)), 'prior candidate/source/config conflict')
        require(lam not in seen and policy.lambda_min<=lam<=policy.lambda_max,'duplicate/out-of-range lambda');seen.add(lam)
        score=item['score'];failure=item['failure']
        require((score is None)==(failure is not None),'score/failure conflict')
        if score is not None:number(score,'score',nonnegative=True);require(result.state.ready,'failed outer iteration cannot have a valid score')
        number(item['seconds'],'trial seconds',nonnegative=True);integer(item['subproblems'],'subproblem count')
        folds=[]
        require(len(item['fold_evidence']) in (0,len(expected_folds)),'fold count conflict')
        if result.state.ready:require(len(item['fold_evidence'])==len(expected_folds),'missing trained CV folds')
        for fold_index,f in enumerate(item['fold_evidence']):
            require(set(f)=={p.name for p in fields(PriorFoldEvidence)},'unknown/missing fold evidence fields')
            fr=read_contract(f['result']);train=tuple(f['train_indices']);val=tuple(f['validation_indices'])
            fs,fc,expected_train,expected_val,physical=expected_folds[fold_index]
            require(train==tuple(expected_train) and val==tuple(expected_val) and fr.source==fs,'saved CV split/source conflicts with training-only recipe')
            require(not set(train)&set(val) and len(train)>=2 and len(val)>0,'invalid held-out split')
            require(fr.config==replace(config,grid=replace(config.grid,regularization_lambda=lam),
                observation_weights=tuple(config.observation_weights[i] for i in train) if config.observation_weights else ()),'fold prior/config conflict')
            require(fr.source.normalization.reference_value==f['training_terminal'] and fr.source.normalization.scale_value==f['training_S'],'fold normalization conflict')
            require(f['status'] in {'PASS','FAIL'},'unknown fold state')
            if f['status']=='PASS':require(fr.state.ready,'successful fold lacks converged solve');number(f['weighted_physical_sse'],'held-out SSE',nonnegative=True)
            folds.append(PriorFoldEvidence(train,val,f['training_terminal'],f['training_S'],fr,f['status'],f['weighted_physical_sse']))
        if score is not None:
            if policy.strategy=='gcv':
                n=len(source.signal);df=result.solve_diagnostics.effective_df
                expected=n*result.solve_diagnostics.weighted_sse/(n-df)**2
            elif policy.strategy in {'discrepancy','numerical_discrepancy'}:
                weights=np.asarray(config.observation_weights or np.ones(len(source.signal)))
                target=(float(np.sum(weights*(np.asarray(value['physical_noise_sd'])/source.normalization.scale_value)**2))
                    if policy.strategy=='discrepancy' else float(np.sum(weights)*policy.numerical_sd_relative_S**2))
                require(target>0 and np.isfinite(target),'invalid saved discrepancy target')
                expected=abs(float(np.log(max(result.solve_diagnostics.weighted_sse,np.finfo(float).tiny)/target)))
            else:
                require(len(folds)==policy.folds and all(f.status=='PASS' for f in folds),'incomplete CV score')
                expected=sum(f.weighted_physical_sse for f in folds)/sum(len(f.validation_indices) for f in folds)
            require(np.isclose(score,expected,rtol=1e-10,atol=1e-14),'saved prior score/evidence conflict')
        trials.append(PriorTrial(result,score,failure,item['seconds'],item['subproblems'],tuple(folds)))
    ties,best=prior_score_ties(trials,policy)
    require(tuple(value['tied_lambdas'])==ties and value['selected_lambda']==best,'prior tie/selection conflict')
    require(value['evaluations']==len(trials)+sum(len(t.fold_evidence) for t in trials) and value['evaluations']<=policy.max_evaluations,'evaluation budget conflict')
    require(value['subproblems']==sum(t.subproblems for t in trials),'inner count conflict')
    require(value['subproblems']<=policy.max_subproblems,'inner subproblem budget exceeded')
    report=PriorSearchReport(source,config,policy,tuple(trials),best,ties,value['termination'],value['evaluations'],value['subproblems'],
        value['seconds'],tuple(value['cache'].items()),tuple(value['physical_noise_sd']))
    require(json.dumps(report.to_dict(),sort_keys=True,allow_nan=False)==json.dumps(value,sort_keys=True,allow_nan=False),'prior search wire conflict')
    return report
