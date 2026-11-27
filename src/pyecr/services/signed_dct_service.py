"""Whole signed-DCT jobs independent of physical Fit and legacy NNLS services."""
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from threading import RLock
from uuid import uuid4

from pyecr.contracts import (SignedDctAnalysis, SignedDctConfig, SignedDctDiagnosticConfig,
                            SignedNormalizationSpec, SourceIdentity)
from pyecr.contracts.common import digest, integer, number, require
from pyecr.contracts.signed_analysis import default_diagnostic_config
from pyecr.dct.signed_analysis import analyze_signed_dct
from pyecr.io.dataset import EcrDataset
from pyecr.preprocessing import PreprocessConfig
from pyecr.preprocessing.signed_normalization import cumulative_t0_s, raw_observations_sha256
from pyecr.services.analysis import CancellationToken
from pyecr.services.dct_sources import PreparedSignedDctInput, prepare_signed_experiment


@dataclass(frozen=True, slots=True)
class DctPhysicalContext:
    """Optional evidence only: never an inversion prerequisite or Bi estimate."""
    boundary_mode: str = 'unspecified'
    model_id: str | None = None
    config_sha256: str | None = None
    geometry_cm: tuple[float, float, float] | None = None
    active_dimensions: tuple[bool, bool, bool] | None = None
    schema_version: int = 1

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version') == 1
                and self.boundary_mode in {'unspecified', 'ideal_step', 'exponential_flush', 'measured_history'},
                'unknown physical-context version or boundary declaration')
        require((self.model_id is None) == (self.config_sha256 is None), 'model/config context identity pair required')
        if self.model_id is not None:
            from pyecr.models.ids import ModelID
            require(self.model_id in {m.value for m in ModelID}, 'canonical scientific model_id required')
            digest(self.config_sha256, 'config_sha256')
        if self.geometry_cm is not None:
            values = tuple(number(v, 'complete dimension cm', positive=True) for v in self.geometry_cm)
            require(len(values) == 3, 'record all three complete specimen dimensions')
            object.__setattr__(self, 'geometry_cm', values)
        if self.active_dimensions is not None:
            values = tuple(self.active_dimensions)
            require(self.geometry_cm is not None and len(values) == 3 and all(isinstance(v, bool) for v in values),
                    'active axes require complete measured dimensions and three booleans')
            object.__setattr__(self, 'active_dimensions', values)

    @classmethod
    def capture(cls, config):
        if config is None:
            return cls()
        from pyecr.fitting.reproducibility import config_hash
        g = config.model.geometry
        return cls(boundary_mode=config.boundary.mode, model_id=config.model_id.value,
            config_sha256=config_hash(config), geometry_cm=g.full_dims_cm, active_dimensions=g.active_dimensions)

    @property
    def warnings(self):
        boundary = ('BOUNDARY_NOT_CONFIRMED_IDEAL: no ideal-step evidence'
                    if self.boundary_mode == 'unspecified' else
                    f'NONIDEAL_BOUNDARY: {self.boundary_mode}; no deconvolution'
                    if self.boundary_mode != 'ideal_step' else
                    'IDEAL_STEP_DECLARED: conditional on the supplied boundary declaration')
        return (boundary, 'OBSERVED_EFFECTIVE_RESPONSE: time-scale inversion of observed data; '
                'no material-response deconvolution, species assignment or automatic model/Bi suggestion')

    def to_dict(self):
        return dict(schema_version=self.schema_version, boundary_mode=self.boundary_mode,
            model_id=self.model_id, config_sha256=self.config_sha256,
            geometry_cm=None if self.geometry_cm is None else list(self.geometry_cm),
            active_dimensions=None if self.active_dimensions is None else list(self.active_dimensions),
            length_unit='cm',
            response_scope='observed_effective_response', deconvolved=False, bi=None)


@dataclass(frozen=True, slots=True)
class SignedDctRequest:
    dataset: EcrDataset
    recipe: PreprocessConfig
    identity: SourceIdentity
    config: SignedDctConfig
    diagnostic_config: SignedDctDiagnosticConfig
    normalization_spec: SignedNormalizationSpec | None
    physical_context: DctPhysicalContext
    schema_version: int = 1
    convex_options: object | None = None

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version') == 1, 'unknown signed request version')
        require(isinstance(self.config, SignedDctConfig) and isinstance(self.diagnostic_config, SignedDctDiagnosticConfig)
                and isinstance(self.physical_context, DctPhysicalContext), 'explicit signed request contracts required')
        SourceIdentity.capture(self.dataset, self.recipe).assert_matches(self.identity)
        if self.convex_options is not None:
            from pyecr.contracts.convex_dct import ConvexDctOptions
            require(isinstance(self.convex_options,ConvexDctOptions),'typed convex options required')
            require(self.config.schema_version==2 and self.config.solver=='reused_svd' and self.config.grid.lambda_mode=='manual','convex options require an explicit v2 static base config')


@dataclass(frozen=True, slots=True)
class SignedDctProgress:
    sequence: int
    fraction: float
    message: str


@dataclass(frozen=True, slots=True)
class SignedDctJob:
    job_id: str
    request: SignedDctRequest
    generation: int
    status: str = 'queued'
    progress: tuple[SignedDctProgress, ...] = ()
    prepared: PreparedSignedDctInput | None = None
    analysis: SignedDctAnalysis | None = None
    stale_reasons: tuple[str, ...] = ()
    cancellation_requested: bool = False
    failure: str | None = None
    exception_type: str | None = None
    schema_version: int = 1

    @property
    def warnings(self):
        return tuple(dict.fromkeys((*self.request.physical_context.warnings,
            *(self.prepared.normalization.state.warnings if self.prepared else ()),
            *((self.prepared.terminal_preparation_failure,) if self.prepared and self.prepared.terminal_preparation_failure else ()),
            *(self.analysis.state.warnings if self.analysis else ()))))

    def to_dict(self):
        """Versioned audit snapshot; project loading/persistence belongs to P10."""
        r = self.request
        ds = r.dataset
        value = dict(schema_version=self.schema_version, job_id=self.job_id, generation=self.generation,
            status=self.status, identity=r.identity.to_dict(), config=r.config.to_dict(),
            diagnostic_config=r.diagnostic_config.to_dict(), recipe=r.recipe.to_dict(),
            input_observations=dict(processed_time=ds.processed_time.tolist(),
                processed_signal=ds.processed_signal.tolist(), mask=ds.mask.tolist(),
                raw_count=len(ds.raw_time), time_unit=ds.metadata.get('time_unit'),
                signal_type=ds.metadata.get('signal_type'), signal_unit=ds.metadata.get('signal_unit'),
                t0_s=cumulative_t0_s(ds), raw_sha256=raw_observations_sha256(ds)),
            normalization_spec=None if r.normalization_spec is None else r.normalization_spec.to_dict(),
            physical_context=r.physical_context.to_dict(), warnings=list(self.warnings),
            progress=[dict(sequence=p.sequence, fraction=p.fraction, message=p.message) for p in self.progress],
            prepared=None if self.prepared is None else self.prepared.to_dict(),
            analysis=None if self.analysis is None else self.analysis.to_dict(),
            stale_reasons=list(self.stale_reasons), cancellation_requested=self.cancellation_requested,
            failure=self.failure, exception_type=self.exception_type)
        if r.convex_options is not None:value['convex_options']=r.convex_options.to_dict()
        return value


class WholeSignedDctService:
    """Thread-safe audit history with owner-bound current publication.

    The owner must bind input changes (including project replacement) before
    reading current results. GuiController does this on invalidation and polls.
    A worker only executes its frozen request; it can never rebind owner inputs.
    """
    def __init__(self, *, isolate_numeric=False):
        self.isolate_numeric = bool(isolate_numeric)
        self._lock = RLock()
        self._jobs = {}
        self._tokens = {}
        self._identity = None
        self._owner_token = None
        self._generation = 0
        self._current_id = None

    def _invalidate(self, reason):
        self._generation += 1
        self._current_id = None
        for key, old in tuple(self._jobs.items()):
            if reason not in old.stale_reasons:
                self._jobs[key] = replace(old, stale_reasons=(*old.stale_reasons, reason),
                    status='stale' if old.status in {'completed', 'warning'} else old.status)

    def bind_inputs(self, dataset, recipe=None, *, force=False, owner_token=None):
        """Owner action, not worker publication; force marks a new project context."""
        identity = None if dataset is None else SourceIdentity.capture(dataset, recipe)
        with self._lock:
            if force or identity != self._identity or (owner_token is not None and owner_token != self._owner_token):
                self._invalidate('Owner dataset/recipe or project context changed')
                self._identity = identity
            if owner_token is not None:
                self._owner_token = owner_token

    def create_job(self, dataset, recipe, config, *, normalization_spec=None,
                   diagnostic_config=None, physical_context=None,convex_options=None):
        require(isinstance(dataset, EcrDataset) and isinstance(recipe, PreprocessConfig), 'dataset/recipe required')
        if diagnostic_config is None and convex_options is not None:
            diagnostic_config=SignedDctDiagnosticConfig(lambda_factors=(),tau_ranges_s=(),grid_sizes=(),terminal_offsets_in_S=())
        if diagnostic_config is None and isinstance(config, SignedDctConfig):
            diagnostic_config = default_diagnostic_config(config.grid)
        require(isinstance(config, SignedDctConfig) and isinstance(diagnostic_config, SignedDctDiagnosticConfig),
                'signed config/diagnostic config required')
        require(normalization_spec is None or isinstance(normalization_spec, SignedNormalizationSpec),
                'explicit normalization spec required')
        context = physical_context or DctPhysicalContext()
        require(isinstance(context, DctPhysicalContext), 'physical context must remain separate evidence')
        identity = SourceIdentity.capture(dataset, recipe)
        request = SignedDctRequest(dataset, recipe, identity, config, diagnostic_config, normalization_spec, context,convex_options=convex_options)
        with self._lock:
            self.bind_inputs(dataset, recipe)
            self._invalidate('Superseded by an explicit signed-DCT request')
            key = uuid4().hex
            self._jobs[key] = SignedDctJob(key, request, self._generation)
            self._tokens[key] = CancellationToken()
            return key

    def get_job(self, job_id):
        with self._lock:
            return self._jobs[job_id]

    def restore_history(self, jobs):
        """Replace history with validated, non-executable project evidence."""
        require(all(isinstance(j, SignedDctJob) and j.status == 'stale' for j in jobs),
                'only historical jobs may be restored')
        with self._lock:
            require(not self._tokens, 'cannot replace history during a running job')
            self._jobs = {j.job_id: j for j in jobs}
            self._current_id = None

    @property
    def history(self):
        with self._lock:
            return tuple(self._jobs.values())

    @property
    def current(self):
        with self._lock:
            return None if self._current_id is None else self._jobs[self._current_id]

    def cancel(self, job_id):
        with self._lock:
            old = self._jobs[job_id]
            if old.status not in {'queued', 'running'}:
                return False
            requested = self._tokens[job_id].cancel()
            self._jobs[job_id] = replace(old, cancellation_requested=True,
                status='cancelled' if old.status == 'queued' else old.status,
                failure='Cancelled before execution' if old.status == 'queued' else old.failure)
            if old.status == 'queued':
                self._tokens.pop(job_id, None)
            return requested

    def fail_queued(self, job_id, exc):
        """Runner rejection before execution (e.g. closed runner)."""
        with self._lock:
            old = self._jobs[job_id]
            if old.status == 'cancelled':
                return old
            require(old.status == 'queued', 'only a queued job can fail submission')
            self._jobs[job_id] = replace(old, status='error', failure=str(exc), exception_type=type(exc).__name__)
            self._tokens.pop(job_id, None)

    def run(self, job_id, *, progress_callback=None):
        with self._lock:
            old = self._jobs[job_id]
            if old.status == 'cancelled':
                return old
            require(old.status == 'queued', 'signed-DCT job may execute only once')
            self._jobs[job_id] = replace(old, status='running')
            token = self._tokens[job_id]
            request = old.request

        def progress(fraction, message):
            token.raise_if_cancelled()
            with self._lock:
                old = self._jobs[job_id]
                require(0 <= fraction <= 1 and (not old.progress or fraction >= old.progress[-1].fraction),
                        'invalid or decreasing signed-DCT progress')
                event = SignedDctProgress(len(old.progress)+1, float(fraction), str(message))
                self._jobs[job_id] = replace(old, progress=(*old.progress, event))
            if progress_callback is not None:
                progress_callback(fraction, message)
            token.raise_if_cancelled()

        try:
            progress(0., 'Preparing P02 DCT copy from frozen processed observations')
            prepared = prepare_signed_experiment(request.dataset, request.recipe,
                normalization_spec=request.normalization_spec, diagnostic_config=request.diagnostic_config)
            token.raise_if_cancelled()
            with self._lock:
                self._jobs[job_id] = replace(self._jobs[job_id], prepared=prepared)
            if prepared.source is None:
                status = prepared.normalization.state.status
                failure = '; '.join(prepared.normalization.state.blocking_reasons)
                analysis = None
            else:
                from pyecr.dct.execution import run_numeric
                if request.convex_options is not None:
                    from pyecr.dct.convex_gui import analyze_convex_gui
                    analysis=run_numeric(self.isolate_numeric,token.raise_if_cancelled,analyze_convex_gui,prepared.source,request.config,request.convex_options,request.diagnostic_config,
                        cancel_check=token.raise_if_cancelled,progress_callback=lambda f,m:progress(.05+.9*f,m))
                else:
                    analysis = run_numeric(self.isolate_numeric, token.raise_if_cancelled, analyze_signed_dct, prepared.source, request.config, request.diagnostic_config,
                        terminal_sources=prepared.terminal_sources, cancel_check=token.raise_if_cancelled,
                        progress_callback=lambda f, m: progress(.05+.9*f, m))
                status = analysis.state.status
                failure = '; '.join(analysis.state.blocking_reasons) or None
            progress(1., 'Computation finished; checking owner source before publication')
            with self._lock:
                token.raise_if_cancelled()
                old = self._jobs[job_id]
                current = (old.generation == self._generation and not old.stale_reasons
                           and request.identity == self._identity)
                ready = status in {'completed', 'warning'}
                job = replace(old, analysis=analysis, failure=failure,
                    status='stale' if ready and not current else 'warning' if ready else status)
                self._jobs[job_id] = job
                if current and ready:
                    self._current_id = job_id
        except Exception as exc:
            with self._lock:
                old = self._jobs[job_id]
                cancelled = isinstance(exc, CancelledError) or token.is_cancelled
                job = replace(old, status='cancelled' if cancelled else 'error',
                    failure=str(exc) or type(exc).__name__, exception_type=type(exc).__name__,
                    cancellation_requested=old.cancellation_requested or cancelled)
                self._jobs[job_id] = job
        finally:
            with self._lock:
                self._tokens.pop(job_id, None)
        return job
