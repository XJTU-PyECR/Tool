"""Explicit, replayable fixed-L2 search; never used by default GUI actions.

Manual and fixed-grid GCV remain in signed_analysis. This separate opt-in
entry freezes budgets/ties/noise assumptions and records every tried lambda.
It does not score peaks or publish current. It is not valid for adaptive UP.
"""
from dataclasses import asdict, dataclass, fields, replace
from math import isfinite, log10
from time import perf_counter

import numpy as np

from pyecr.contracts import (ContractError,DctSourceSnapshot,SignedDctConfig,SignedDctDiagnosticConfig,
    SignedNormalizationSpec,SourceIdentity)
from pyecr.contracts.common import require,integer,number
from pyecr.dct.signed_analysis import _candidate
from pyecr.dct.signed import fit_signed_dct
from pyecr.dct.signed_linear import LinearCache
from pyecr.dct.execution import check_cancel


@dataclass(frozen=True, slots=True)
class LinearSearchConfig:
    version: str = 'fixed-linear-search/1'
    strategy: str = 'adaptive_gcv'
    coarse_lambdas: tuple = tuple(float(x) for x in np.logspace(-8,0,9))
    lambda_min: float = 1e-14
    lambda_max: float = 1e4
    max_expansions: int = 3
    expansion_decades: float = 2.
    refinement_rounds: int = 2
    max_candidates: int = 33
    max_evaluations: int = 180
    tie_rtol: float = 1e-10
    tie_atol: float = 1e-18
    tie_rule: str = 'largest_lambda_in_score_tie'
    cache_bytes: int = 128*1024**2
    noise_model: str = 'unverified'
    folds: int = 4
    purge_seconds: float = 3.
    normalization_policy: str = 'external_terminal_train_maxabs_per_fold'

    def __post_init__(self):
        require(self.version == 'fixed-linear-search/1', 'unknown search version')
        require(self.strategy in {'adaptive_gcv','discrepancy','blocked_cv'}, 'unknown fixed-linear search strategy')
        values = tuple(number(x,'coarse lambda',positive=True) for x in self.coarse_lambdas)
        require(len(values)>=3 and all(a<b for a,b in zip(values,values[1:])), 'coarse lambdas must increase')
        object.__setattr__(self,'coarse_lambdas',values)
        for name in ('lambda_min','lambda_max','expansion_decades'):
            object.__setattr__(self,name,number(getattr(self,name),name,positive=True))
        require(self.lambda_min <= values[0] < values[-1] <= self.lambda_max, 'lambda bounds exclude initial grid')
        for name in ('max_expansions','refinement_rounds','cache_bytes'):
            integer(getattr(self,name),name)
        for name in ('max_candidates','max_evaluations'):
            integer(getattr(self,name),name,minimum=len(values))
        integer(self.folds,'folds',minimum=2)
        for name in ('tie_rtol','tie_atol','purge_seconds'):
            number(getattr(self,name),name,nonnegative=True)
        require(self.tie_rule == 'largest_lambda_in_score_tie', 'unknown tie rule')
        require(self.noise_model in {'unverified','independent_known_physical_sd','time_correlated'}, 'unknown noise declaration')
        if self.strategy == 'discrepancy':
            require(self.noise_model == 'independent_known_physical_sd', 'discrepancy requires independently supplied physical noise SD')
        if self.strategy == 'blocked_cv':
            require(self.normalization_policy == 'external_terminal_train_maxabs_per_fold', 'unknown fold normalization')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls,value):
        require(set(value)=={f.name for f in fields(cls)}, 'missing/unknown search configuration fields')
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    train_indices: tuple
    validation_indices: tuple
    source_sha256: str
    terminal: float
    scale: float
    weighted_physical_sse: float | None
    status: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SearchTrial:
    candidate: object
    score: float | None
    failure: str | None
    seconds: float
    folds: tuple = ()

    def to_dict(self):
        return dict(candidate=self.candidate.to_dict(),selection_score=self.score,failure=self.failure,
                    seconds=self.seconds,folds=[asdict(f) for f in self.folds])


@dataclass(frozen=True, slots=True)
class LinearSearchResult:
    source: DctSourceSnapshot
    base_config: SignedDctConfig
    search_config: LinearSearchConfig
    trials: tuple
    selected_lambda: float | None
    tie_lambdas: tuple
    termination: str
    evaluations_used: int
    cache_telemetry: tuple
    noise_sd_physical: tuple = ()

    @property
    def selected_result(self):
        if self.selected_lambda is None:return None
        return next((t.candidate.result for t in self.trials if t.candidate.result.selected_lambda==self.selected_lambda),None)

    def to_dict(self):
        return dict(version='fixed-linear-search-result/1',source=self.source.to_dict(),base_config=self.base_config.to_dict(),
            search_config=self.search_config.to_dict(),trials=[t.to_dict() for t in self.trials],selected_lambda=self.selected_lambda,
            tie_lambdas=self.tie_lambdas,termination=self.termination,evaluations_used=self.evaluations_used,
            cache_telemetry=dict(self.cache_telemetry),noise_sd_physical=self.noise_sd_physical,
            assumptions='Fixed W, kernel, terminal, time origin, mask and L2. GCV conditional on independent whitened errors; no CI or physical-reliability claim.')


def score_ties(trials, policy):
    valid=[t for t in trials if t.failure is None and t.score is not None and isfinite(t.score)]
    if not valid: return (),None
    minimum=min(t.score for t in valid)
    tied=tuple(sorted(t.candidate.result.selected_lambda for t in valid if abs(t.score-minimum)<=policy.tie_atol+policy.tie_rtol*abs(minimum)))
    return tied,max(tied)


def _fold_sources(source, config, policy):
    from pyecr.io.dataset import EcrDataset
    from pyecr.preprocessing import PreprocessConfig,normalize_signed_dct
    require(source.normalization.spec.reference_kind == 'external_known_terminal',
        'Blocked CV needs an independently declared terminal; an all-data accepted tail would leak held-out observations')
    t=np.asarray(source.time_s);norm=source.normalization
    physical=norm.reference_value+norm.scale_value*np.asarray(source.signal)
    prepared=[]
    for val in np.array_split(np.arange(len(t)),policy.folds):
        require(len(val)>0,'empty validation block')
        train=np.flatnonzero((t<t[val[0]]-policy.purge_seconds)|(t>t[val[-1]]+policy.purge_seconds))
        require(len(train)>=2,'purge gap leaves fewer than two training observations')
        ds=EcrDataset.from_raw(t[train],physical[train],signal_type=norm.spec.signal_type,signal_unit=norm.spec.signal_unit)
        recipe=PreprocessConfig()
        spec=SignedNormalizationSpec(identity=SourceIdentity.capture(ds,recipe),raw_count=len(train),raw_indices=tuple(range(len(train))),
            signal_type=norm.spec.signal_type,signal_unit=norm.spec.signal_unit,reference_kind='external_known_terminal',
            reference_value=norm.reference_value,reference_evidence='Blocked-CV training copy; independent frozen external terminal: '+norm.spec.reference_evidence)
        folded=normalize_signed_dct(ds,spec,recipe=recipe)
        require(folded.state.ready,'training-only normalization failed: '+'; '.join(folded.state.blocking_reasons))
        snapshot=DctSourceSnapshot.capture_experiment(ds,recipe,normalization=folded)
        w=config.observation_weights
        cfg=replace(config,observation_weights=tuple(w[i] for i in train) if w else ())
        prepared.append((snapshot,cfg,train,val,physical))
    return prepared


def search_fixed_linear(source, config, policy=LinearSearchConfig(), *, noise_sd_physical=(),
                        cancel_check=None, progress_callback=None):
    """Explicit search returns evidence; cancellation raises without a result.

    Budgets count actual numerical solves (full data plus each CV fold).
    The existing fixed-grid GCV and its exact-min rule are left unchanged.
    """
    require(isinstance(source,DctSourceSnapshot) and isinstance(config,SignedDctConfig),'typed source/config required')
    source.require_usable()
    require(source.representation=='signed_terminal' and source.normalization is not None,
            'fixed-linear search requires an explicit terminal-normalized whole source')
    require(config.schema_version==2 and config.grid.lambda_mode=='manual', 'search requires explicit v2 manual base config')
    require(isinstance(policy,LinearSearchConfig),'typed search policy required')
    check=cancel_check or check_cancel
    progress=progress_callback or (lambda f,m:None)
    check()
    cache=LinearCache(policy.cache_bytes)
    peak_policy=SignedDctDiagnosticConfig(lambda_factors=(),tau_ranges_s=(),grid_sizes=(),terminal_offsets_in_S=())
    folds=_fold_sources(source,config,policy) if policy.strategy=='blocked_cv' else ()
    cost=1+len(folds);require(len(policy.coarse_lambdas)*cost<=policy.max_evaluations,'budget cannot cover coarse grid and all folds')
    noise_target=None
    require(policy.strategy=='discrepancy' or len(noise_sd_physical)==0,
            'physical noise SD is only consumed by the explicit discrepancy strategy')
    if policy.strategy=='discrepancy':
        sd=np.asarray(noise_sd_physical,dtype=float)
        require(sd.shape==(len(source.signal),) and np.all(np.isfinite(sd)) and np.all(sd>0), 'noise SD must cover all observations and be strictly positive')
        w=np.asarray(config.observation_weights or np.ones(len(sd)))
        noise_target=float(np.sum(w*(sd/source.normalization.scale_value)**2))
        require(np.isfinite(noise_target) and noise_target>0,'invalid discrepancy noise energy')
    trials=[];seen=set();used=0
    def evaluate(lam):
        nonlocal used
        check();start=perf_counter()
        cfg=replace(config,grid=replace(config.grid,regularization_lambda=float(lam)))
        candidate=_candidate(source,cfg,peak_policy,check,linear_cache=cache);used+=1
        failure=None;score=None;fold_results=[]
        if not candidate.result.state.ready:
            failure='NUMERIC_FAILURE: '+'; '.join(candidate.result.state.blocking_reasons)
        elif policy.strategy=='adaptive_gcv':
            score=candidate.gcv_score;failure=candidate.failure
        elif policy.strategy=='discrepancy':
            sse=candidate.result.solve_diagnostics.weighted_sse
            score=abs(float(np.log(max(sse,np.finfo(float).tiny)/noise_target)))
        else:
            total=0.;count=0
            for fold_source,fold_cfg,train,val,physical in folds:
                check()
                r=fit_signed_dct(fold_source,replace(fold_cfg,grid=cfg.grid),linear_cache=cache);used+=1
                e=FoldEvidence(tuple(int(i) for i in train),tuple(int(i) for i in val),fold_source.content_sha256,
                    fold_source.normalization.reference_value,fold_source.normalization.scale_value,None,'FAIL')
                if not r.state.ready:
                    failure='CV_FOLD_FAILURE: '+'; '.join(r.state.blocking_reasons)
                    e=replace(e,reason=failure)
                else:
                    # Predictions use every held-out time; no held-out point enters terminal or S.
                    with np.errstate(over='ignore',under='ignore'):
                        pred=np.exp(-np.asarray(source.time_s)[val,None]/np.asarray(r.tau_s)[None,:])@np.asarray(r.quadrature_mass)
                    pred=fold_source.normalization.reference_value+fold_source.normalization.scale_value*pred
                    w=np.asarray(config.observation_weights or np.ones(len(source.signal)))[val]
                    sse=float(np.sum(w*(physical[val]-pred)**2))
                    if not np.isfinite(sse): failure='CV_FOLD_FAILURE: nonfinite held-out SSE'
                    else:
                        total+=sse;count+=len(val);e=replace(e,weighted_physical_sse=sse,status='PASS')
                fold_results.append(e);check()
            if failure is None: score=total/count
        if score is not None and not isfinite(score): failure='nonfinite selection score';score=None
        trials.append(SearchTrial(candidate,score,failure,perf_counter()-start,tuple(fold_results)))
        seen.add(float(lam));check();progress(used/policy.max_evaluations,f'Linear lambda {lam:g}: '+(failure or 'scored'))
    def add(values):
        for lam in sorted(set(values)-seen):
            if len(trials)>=policy.max_candidates or used+cost>policy.max_evaluations: return False
            evaluate(lam)
        return True
    add(policy.coarse_lambdas)
    termination='refinement_budget';expansions=0;refinements=0
    while True:
        check();ties,best=score_ties(trials,policy)
        if best is None: termination='all_candidates_failed';break
        values=sorted(seen);index=values.index(best);boundary=index in {0,len(values)-1}
        if boundary:
            if expansions>=policy.max_expansions: termination='boundary_expansion_budget';break
            direction=-1 if index==0 else 1
            proposed=[float(10**min(log10(policy.lambda_max),max(log10(policy.lambda_min),log10(best)+direction*d)))
                      for d in (policy.expansion_decades/2,policy.expansion_decades)]
            expansions+=1
            if set(proposed)<=seen: termination='lambda_range_limit';break
        else:
            if refinements>=policy.refinement_rounds: termination='refinement_budget';break
            def midpoint(a,b):
                product=a*b
                return float(np.sqrt(product)) if 0<product<float('inf') else float(10**((log10(a)+log10(b))/2))
            proposed=[midpoint(values[index-1],best),midpoint(best,values[index+1])]
            refinements+=1
        if not add(proposed): termination='evaluation_or_candidate_budget';break
    ties,best=score_ties(trials,policy)
    check();progress(1.,'Fixed-linear search complete: '+termination);check()
    return LinearSearchResult(source,config,policy,tuple(trials),best,ties,termination,used,tuple(cache.telemetry().items()),
        tuple(float(x) for x in noise_sd_physical))


def read_linear_search(value):
    """Strict read-only evidence restoration. Never reruns a solve or upgrades B0."""
    import json
    from pyecr.contracts import read_contract
    names={'version','source','base_config','search_config','trials','selected_lambda','tie_lambdas',
        'termination','evaluations_used','cache_telemetry','noise_sd_physical','assumptions'}
    require(set(value)==names and value['version']=='fixed-linear-search-result/1','unknown/malformed linear search result')
    source=read_contract(value['source']);cfg=read_contract(value['base_config'])
    policy=LinearSearchConfig.from_dict(value['search_config']);trials=[]
    require(cfg.schema_version==2 and cfg.grid.lambda_mode=='manual','saved search requires v2 manual base config')
    require(value['termination'] in {'refinement_budget','all_candidates_failed','boundary_expansion_budget',
        'lambda_range_limit','evaluation_or_candidate_budget'},'unknown search termination')
    seen=set()
    for trial in value['trials']:
        require(set(trial)=={'candidate','selection_score','failure','seconds','folds'},'unknown/missing search trial fields')
        candidate=read_contract(trial['candidate'])
        require(candidate.result.source==source,'search trial/source conflict')
        # Failed solves keep the requested lambda in config even without payload.
        expected=replace(cfg,grid=replace(cfg.grid,regularization_lambda=candidate.result.config.grid.regularization_lambda))
        require(candidate.result.config==expected,'search trial/config conflict')
        lam=candidate.result.config.grid.regularization_lambda
        require(policy.lambda_min<=lam<=policy.lambda_max and lam not in seen,'duplicate/out-of-range saved lambda')
        seen.add(lam)
        if candidate.result.state.ready:
            require(candidate.raw_effective_df==candidate.result.solve_diagnostics.effective_df,'candidate/solve trace conflict')
        score=trial['selection_score']
        if score is not None:number(score,'selection_score',nonnegative=True)
        require((trial['failure'] is None)==(score is not None),'score/failure conflict')
        number(trial['seconds'],'trial seconds',nonnegative=True)
        folds=[]
        for fold in trial['folds']:
            require(set(fold)=={f.name for f in fields(FoldEvidence)},'unknown/missing fold evidence fields')
            f=FoldEvidence(**{**fold,'train_indices':tuple(fold['train_indices']),'validation_indices':tuple(fold['validation_indices'])})
            require(f.status in {'PASS','FAIL'} and not set(f.train_indices)&set(f.validation_indices),'invalid fold split/status')
            require(f.scale>0 and isfinite(f.scale) and isfinite(f.terminal),'invalid training normalization evidence')
            folds.append(f)
        trials.append(SearchTrial(candidate,score,trial['failure'],trial['seconds'],tuple(folds)))
        if trial['failure'] is None and policy.strategy=='adaptive_gcv':
            require(score==candidate.gcv_score,'saved search/GCV score conflict')
    ties,best=score_ties(trials,policy)
    require(tuple(value['tie_lambdas'])==ties and value['selected_lambda']==best,'saved search tie/selection conflict')
    used=integer(value['evaluations_used'],'evaluations_used')
    require(used<=policy.max_evaluations and len(trials)<=policy.max_candidates,'saved search exceeds budget')
    require(used==len(trials)+sum(len(t.folds) for t in trials),'actual evaluation count conflict')
    noise=tuple(number(x,'physical noise SD',positive=True) for x in value['noise_sd_physical'])
    if policy.strategy=='discrepancy':
        require(len(noise)==len(source.signal),'saved noise SD/source length conflict')
        w=np.asarray(cfg.observation_weights or np.ones(len(noise)))
        target=float(np.sum(w*(np.asarray(noise)/source.normalization.scale_value)**2))
        for t in trials:
            if t.failure is None:
                expected=abs(float(np.log(max(t.candidate.result.solve_diagnostics.weighted_sse,np.finfo(float).tiny)/target)))
                require(np.isclose(t.score,expected,rtol=1e-12,atol=1e-14),'saved discrepancy score/noise conflict')
    else:require(not noise,'unused saved noise SD')
    if policy.strategy=='blocked_cv':
        for t in trials:
            if t.failure is None:
                require(len(t.folds)==policy.folds and all(f.status=='PASS' for f in t.folds),'incomplete CV score')
                count=sum(len(f.validation_indices) for f in t.folds)
                expected=sum(f.weighted_physical_sse for f in t.folds)/count
                require(np.isclose(t.score,expected,rtol=1e-12,atol=1e-14),'saved CV score/fold SSE conflict')
    report=LinearSearchResult(source,cfg,policy,tuple(trials),best,ties,value['termination'],used,
        tuple(value['cache_telemetry'].items()),noise)
    require(json.dumps(report.to_dict(),sort_keys=True,allow_nan=False)==json.dumps(value,sort_keys=True,allow_nan=False),
        'search wire/evidence conflict')
    return report
