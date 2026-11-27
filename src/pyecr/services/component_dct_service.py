"""Explicit component DCT jobs, with a real current P08 parent and P02 copy."""
from concurrent.futures import CancelledError
from dataclasses import dataclass,replace
from threading import RLock
from uuid import uuid4

import numpy as np

from pyecr.contracts import (ComponentDctBundle,ComponentDctConfig,ResultState,SignedDctDiagnosticConfig,
    ComponentSourceCapture,SignedDctConfig,SignedNormalizationSpec)
from pyecr.io.dataset import EcrDataset
from pyecr.preprocessing import PreprocessConfig
from pyecr.report.snapshot import AnalysisSnapshot
from pyecr.contracts.common import require,SourceIdentity
from pyecr.contracts.signed_analysis import default_diagnostic_config
from pyecr.dct.components import fit_component_spectrum
from pyecr.dct.signed_analysis import analyze_signed_dct
from pyecr.services.analysis import CancellationToken
from pyecr.services.component_source import capture_component_sources,component_source_current_reasons
from pyecr.services.dct_sources import prepare_signed_experiment
from pyecr.services.simulation_transfer import validate_simulation_capture


@dataclass(frozen=True,slots=True)
class ComponentDctRequest:
    dataset: EcrDataset
    recipe: PreprocessConfig
    capture: ComponentSourceCapture
    config: SignedDctConfig
    component_configs: tuple[ComponentDctConfig,ComponentDctConfig]
    normalization_spec: SignedNormalizationSpec | None
    diagnostic_config: SignedDctDiagnosticConfig
    parent_snapshot: AnalysisSnapshot

    def __post_init__(self):
        self.capture.require_usable()
        require(isinstance(self.parent_snapshot,AnalysisSnapshot),'actual immutable parent snapshot required')
        SourceIdentity.capture(self.dataset,self.recipe,config=self.parent_snapshot.analysis_config,
            fit=self.parent_snapshot.fit_result,diagnostics=self.parent_snapshot.diagnostics_result).assert_matches(self.capture.identity)
        validate_simulation_capture(self.capture,self.dataset)
        if self.capture.eligibility is not None:
            from pyecr.services.component_eligibility import eligibility_from_snapshot
            require(eligibility_from_snapshot(self.parent_snapshot,self.recipe)==self.capture.eligibility,
                    'Component request eligibility/original parent evidence conflict')
        if self.capture.simulation_parent is not None:
            evidence=self.parent_snapshot.scientific_evidence
            require(evidence and evidence['fit_generation']==self.capture.fit_generation,
                'Simulation component request requires persisted current Fit generation')
            SourceIdentity.from_dict(evidence['source']).assert_matches(self.capture.identity)
            require(evidence['gate_policy']['confidence_level']==.95,'Simulation cannot change reliability gate')

    def to_dict(self):
        return dict(schema_version=1,capture=self.capture.to_dict(),config=self.config.to_dict(),
            component_configs=[c.to_dict() for c in self.component_configs],recipe=self.recipe.to_dict(),
            normalization_spec=None if self.normalization_spec is None else self.normalization_spec.to_dict(),
            diagnostic_config=self.diagnostic_config.to_dict(),parent_snapshot=self.parent_snapshot.to_dict())


@dataclass(frozen=True,slots=True)
class ComponentDctJob:
    job_id: str
    request: ComponentDctRequest
    status: str = 'queued'
    progress: tuple[tuple[float,str],...] = ()
    bundle: ComponentDctBundle | None = None
    failure: str | None = None
    stale_reasons: tuple[str,...] = ()
    cancellation_requested: bool = False

    def to_dict(self):
        return dict(schema_version=1,job_id=self.job_id,request=self.request.to_dict(),status=self.status,
            progress=list(self.progress),bundle=None if self.bundle is None else self.bundle.to_dict(),
            failure=self.failure,stale_reasons=list(self.stale_reasons),cancellation_requested=self.cancellation_requested)


class ComponentDctService:
    def __init__(self, *, isolate_numeric=False):
        self.isolate_numeric = bool(isolate_numeric)
        self._jobs={};self._tokens={};self._lock=RLock();self._session=None;self._binding=None

    def bind_session(self,session):
        """Cheap immutable-owner check per UI frame; full content check on change."""
        binding=(id(session),session.current_revision,session.fit_generation,id(session.dataset),
            id(session.current_analysis_config),id(session.fit_result),id(session.diagnostics),
            id(session.preprocessing.config) if session.preprocessing else None,session.fit_stale,
            session.diagnostics_stale,session.diagnostics_identity,session.diagnostics_fit_generation)
        with self._lock:
            self._session=session
            if binding==self._binding:return
            self._binding=binding
            for key,j in tuple(self._jobs.items()):
                reasons=component_source_current_reasons(j.request.capture,session)
                if reasons:
                    self._jobs[key]=replace(j,stale_reasons=reasons,
                        status='stale' if j.status in {'completed','warning'} else j.status)

    def create_job(self,session,config,component_configs,*,normalization_spec=None,
                   diagnostic_config=None):
        self.bind_session(session)
        capture=capture_component_sources(session);capture.require_usable()
        require(len(component_configs)==2 and all(isinstance(c,ComponentDctConfig) for c in component_configs),
                'two explicit component Manual configs required')
        # Validate the physical-model-independent whole config without selecting a model.
        from pyecr.contracts import SignedDctConfig
        require(isinstance(config,SignedDctConfig),'whole signed config required')
        if diagnostic_config is None:
            diagnostic_config = default_diagnostic_config(config.grid)
        require(isinstance(diagnostic_config,SignedDctDiagnosticConfig),'signed diagnostic config required')
        with self._lock:
            require(not any(j.status in {'queued','running'} for j in self._jobs.values()),'component DCT job already running')
            key=uuid4().hex
            snapshot=AnalysisSnapshot.capture(session.dataset,session.current_analysis_config,session.fit_result,session.diagnostics)
            if capture.simulation_parent is not None:
                from pyecr.report.scientific_evidence import capture_scientific
                snapshot=replace(snapshot,schema_version=3,scientific_evidence=capture_scientific(session,snapshot))
            req=ComponentDctRequest(session.dataset,session.preprocessing.config,capture,config,
                tuple(component_configs),normalization_spec,diagnostic_config,
                snapshot)
            self._jobs[key]=ComponentDctJob(key,req);self._tokens[key]=CancellationToken()
        return key

    def get_job(self,key):
        with self._lock:return self._jobs[key]

    def restore_history(self,jobs):
        require(all(isinstance(j,ComponentDctJob) and j.status=='stale' for j in jobs),
                'only historical jobs may be restored')
        with self._lock:
            require(not self._tokens,'cannot replace history during a running job')
            self._jobs={j.job_id:j for j in jobs};self._binding=None

    @property
    def history(self):
        with self._lock:return tuple(self._jobs.values())

    @property
    def current(self):
        with self._lock:
            if self._jobs:
                j=next(reversed(self._jobs.values()))
                if j.status in {'completed','warning'} and not j.stale_reasons:return j
        return None

    def cancel(self,key):
        with self._lock:
            j=self._jobs[key]
            if j.status not in {'queued','running'}:return False
            self._tokens[key].cancel()
            self._jobs[key]=replace(j,cancellation_requested=True,status='cancelled' if j.status=='queued' else 'running')
            if j.status=='queued':self._tokens.pop(key,None)
        return True

    def fail_queued(self,key,exc):
        with self._lock:
            j=self._jobs[key]
            if j.status=='cancelled':return j
            require(j.status=='queued','only a queued component job can fail submission')
            self._jobs[key]=replace(j,status='error',failure=f'{type(exc).__name__}: {exc}')
            self._tokens.pop(key,None)

    def run(self,key,*,progress_callback=None):
        with self._lock:
            j=self._jobs[key]
            if j.status=='cancelled':return j
            require(j.status=='queued','component job may execute only once')
            self._jobs[key]=replace(j,status='running');token=self._tokens[key];req=j.request
        def progress(f,m):
            token.raise_if_cancelled()
            with self._lock:
                old=self._jobs[key]
                require(not old.progress or f>=old.progress[-1][0],'decreasing component progress')
                self._jobs[key]=replace(old,progress=(*old.progress,(float(f),str(m))))
            if progress_callback:progress_callback(f,m)
            token.raise_if_cancelled()
        try:
            progress(0.,'Checking frozen parent; preparing explicit P02 DCT copy')
            reasons=component_source_current_reasons(req.capture,self._session)
            if reasons:
                with self._lock:self._jobs[key]=replace(self._jobs[key],status='stale',stale_reasons=reasons)
                return self.get_job(key)
            prepared=prepare_signed_experiment(req.dataset,req.recipe,normalization_spec=req.normalization_spec,
                diagnostic_config=req.diagnostic_config)
            require(prepared.source is not None,'; '.join(prepared.normalization.state.blocking_reasons))
            # Attach actual P08 provenance, without changing normalization numbers.
            identity=req.capture.identity
            norm=replace(prepared.normalization,spec=replace(prepared.normalization.spec,identity=identity),output_identity=None)
            parent=replace(prepared.source,identity=identity,normalization=norm)
            variants=tuple(replace(v,identity=identity,normalization=replace(v.normalization,
                spec=replace(v.normalization.spec,identity=identity),output_identity=None)) for v in prepared.terminal_sources)
            from pyecr.dct.execution import run_numeric
            overall=run_numeric(self.isolate_numeric, token.raise_if_cancelled, analyze_signed_dct, parent,req.config,req.diagnostic_config,terminal_sources=variants,
                cancel_check=token.raise_if_cancelled,progress_callback=lambda f,m:progress(.05+.6*f,m))
            require(overall.selected_result is not None and overall.selected_result.state.ready,
                    '; '.join(overall.state.blocking_reasons) or 'whole signed analysis unavailable')
            parent_rmse=float(np.sqrt(np.mean((np.asarray(req.capture.parent_source.signal)-req.capture.fit_total)**2)))
            results=[];unavailable=[]
            for i,(source,cfg) in enumerate(zip(req.capture.components,req.component_configs)):
                progress(.7+i*.13,f'Component {i+1}: model-derived remaining shape, Manual lambda')
                try:
                    result=run_numeric(self.isolate_numeric, token.raise_if_cancelled, fit_component_spectrum, source,parent,cfg,parent_rmse=parent_rmse)
                    results.append(result);unavailable.append(None)
                except (ValueError,RuntimeError) as exc:
                    results.append(None);unavailable.append(f'{type(exc).__name__}: {exc}')
            warnings=(*req.capture.state.warnings,*norm.state.warnings,*overall.state.warnings,
                *(f'WHOLE_DCT_FLAG: {flag}' for flag in overall.flags),
                *((prepared.terminal_preparation_failure,) if prepared.terminal_preparation_failure else ()),
                'MODEL_DERIVED_COMPONENTS: not independent experiments/noise samples or proof of two processes; no chemical assignment or automatic initial-value changes',
                'SPECTRAL_DIFFERENCES_EXPECTED: different lambda, constraints, experimental noise and terminal reference; assess time reconstruction first',
                *tuple(r for r in unavailable if r))
            bundle=ComponentDctBundle(schema_version=2,parent_source=parent,parent_model_id=identity.model_id,
                parent_reliability=req.capture.components[0].parent_reliability,component_keys=tuple(c.component_key for c in req.capture.components),
                labels=req.capture.labels,label_semantics=req.capture.label_semantics,
                component_results=(None,None),unavailable_reasons=tuple(unavailable),B_j=req.capture.B_j,
                S_total=norm.scale_value,delta=(req.capture.asymptotic_total-norm.reference_value)/norm.scale_value,
                baseline=req.capture.baseline,state=ResultState(status='warning',warnings=tuple(dict.fromkeys(warnings))),
                overall_result=overall.selected_result,component_capture=req.capture,independent_results=tuple(results),overall_analysis=overall)
            progress(1.,'Checking current parent before publication')
            with self._lock:
                token.raise_if_cancelled()
                reasons=component_source_current_reasons(req.capture,self._session)
                old=self._jobs[key]
                self._jobs[key]=replace(old,bundle=bundle,status='stale' if reasons or old.stale_reasons else 'warning',
                    stale_reasons=reasons or old.stale_reasons)
        except CancelledError:
            with self._lock:self._jobs[key]=replace(self._jobs[key],status='cancelled',bundle=None)
        except Exception as exc:
            with self._lock:self._jobs[key]=replace(self._jobs[key],status='error',failure=f'{type(exc).__name__}: {exc}',bundle=None)
        finally:
            with self._lock:self._tokens.pop(key,None)
        return self.get_job(key)
