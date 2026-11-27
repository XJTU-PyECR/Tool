"""Service boundary between the desktop UI and fitting core."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from math import isfinite, log10
from numbers import Integral, Real
from threading import Event, Lock
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.config import AnalysisConfig, ModelConfig
from pyecr.diagnostics.boundary_contact import bound_contact as _bound_contact
from pyecr.fitting.base import FitCandidate, FitResult
from pyecr.fitting.coarse_to_fine import (
    CoarseToFineOptions, coarse_to_fine_scope, current_screening,
)
from pyecr.fitting.dual_ho import (
    DualHOFitConfig,
    DualHOFitResult,
    M3TriggerEvidence,
    fit_dual_ho,
)
from pyecr.fitting.single import fit_single_ecr
from pyecr.fitting.surface_k import fit_surface_k
from pyecr.fitting.dual_surface_k import fit_dual_surface_k
from pyecr.io.dataset import EcrDataset
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.component_cache import ComponentCacheDiagnostics, ComponentCacheScope
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.models.root_cache import RootCacheDiagnostics, RootCacheScope, ROOT_MODES
from pyecr.fitting.progress import FitProgressEvent, fit_telemetry_scope, fit_stage, capture_fit_cache
from pyecr.fitting.solver_strategy import SolverOptions, SolverRun, solver_scope, current_solver_runs, VERSION as SOLVER_VERSION, SolverRejected
from pyecr.uncertainty.local import (
    LocalUncertaintyResult,
    estimate_local_uncertainty,
)

__all__ = [
    "AnalysisBackend",
    "AnalysisBackendNotRegisteredError",
    "AnalysisBackendRegistry",
    "AnalysisService",
    "CancellationToken",
    "CoreFitOutput",
    "DualHOBackend",
    "DualHOAnalysisConfig",
    "FitParameterPresentation",
    "FitPresentation",
    "ProgressReporter",
    "SingleEcrBackend",
]


ProgressReporter = Callable[[float, str], None]


class AnalysisBackend(Protocol):
    """Callable adapter from a validated config to a core fit result."""

    def __call__(
        self,
        dataset: EcrDataset,
        config: object,
        *,
        cancel_token: CancellationToken,
        progress_callback: ProgressReporter,
    ) -> FitResult | CoreFitOutput | DualHOFitResult: ...


class SingleEcrBackend:
    """AnalysisService adapter for production M0/M1 inversion."""

    def __call__(
        self,
        dataset: EcrDataset,
        config: object,
        *,
        cancel_token: CancellationToken,
        progress_callback: ProgressReporter,
    ) -> CoreFitOutput:
        if not isinstance(config, AnalysisConfig):
            raise TypeError("SingleEcrBackend requires AnalysisConfig")
        if "fixed_d" in config.model_specific:
            from pyecr.fitting.fixed_d import fit_fixed_diffusion
            fit=fit_fixed_diffusion(dataset,config,cancel_check=cancel_token.raise_if_cancelled,progress_callback=progress_callback)
            return CoreFitOutput(fit_result=fit,model_name=config.model_id.value)
        cancel_token.raise_if_cancelled()

        def fit_progress(fraction: float, message: str) -> None:
            cancel_token.raise_if_cancelled()
            progress_callback(0.05 + 0.75 * fraction, message)

        result = fit_single_ecr(
            dataset,
            config,
            cancel_check=cancel_token.raise_if_cancelled,
            progress_callback=fit_progress,
        )
        cancel_token.raise_if_cancelled()
        progress_callback(0.82, "Estimating local M0/M1 uncertainty")
        uncertainty = estimate_local_uncertainty(result)
        return CoreFitOutput(
            fit_result=result,
            uncertainty=uncertainty,
            model_name=config.model.model_id.value,
        )


class CancellationToken:
    """Thread-safe cooperative cancellation signal for analysis backends."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        """Request cancellation and report whether this was the first request."""

        already_cancelled = self._event.is_set()
        self._event.set()
        return not already_cancelled

    def raise_if_cancelled(self) -> None:
        """Raise the standard future cancellation exception when requested."""

        if self._event.is_set():
            raise CancelledError("analysis cancelled by user")


@dataclass(frozen=True, slots=True)
class CoreFitOutput:
    """Optional core artifacts accompanying the required ``FitResult``."""

    fit_result: FitResult
    component_H: ArrayLike | None = None
    component_O: ArrayLike | None = None
    uncertainty: LocalUncertaintyResult | None = None
    warnings: tuple[str, ...] = ()
    model_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult")
        size = self.fit_result.y_fit.size
        component_h = _optional_component("component_H", self.component_H, size)
        component_o = _optional_component("component_O", self.component_O, size)
        if self.uncertainty is not None and not isinstance(
            self.uncertainty, LocalUncertaintyResult
        ):
            raise TypeError("uncertainty must be a LocalUncertaintyResult or None")
        warnings = tuple(self.warnings)
        if not all(isinstance(item, str) for item in warnings):
            raise TypeError("warnings must contain strings")
        model_name = None if self.model_name is None else str(self.model_name)
        object.__setattr__(self, "component_H", component_h)
        object.__setattr__(self, "component_O", component_o)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "model_name", model_name)

    def to_dict(self) -> dict[str, object]:
        """Return recoverable core output without relying on service memory."""

        return {
            "schema_version": 1,
            "fit_result": self.fit_result.to_dict(),
            "component_H": (
                None if self.component_H is None else self.component_H.tolist()
            ),
            "component_O": (
                None if self.component_O is None else self.component_O.tolist()
            ),
            "warnings": list(self.warnings),
            "model_name": self.model_name,
            "uncertainty_recoverable_from_fit_jacobian": (
                self.fit_result.jacobian is not None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CoreFitOutput:
        """Restore persisted fit and components; local uncertainty is recomputable."""

        if not isinstance(value, Mapping):
            raise TypeError("CoreFitOutput data must be a mapping")
        if value.get("schema_version") != 1:
            raise ValueError("unsupported CoreFitOutput schema_version")
        fit_data = value.get("fit_result")
        if not isinstance(fit_data, Mapping):
            raise TypeError("CoreFitOutput.fit_result must be a mapping")
        component_h = value.get("component_H")
        component_o = value.get("component_O")
        warnings = value.get("warnings", ())
        if not isinstance(warnings, (list, tuple)):
            raise TypeError("CoreFitOutput.warnings must be a sequence")
        model_name = value.get("model_name")
        if model_name is not None and not isinstance(model_name, str):
            raise TypeError("CoreFitOutput.model_name must be a string or None")
        return cls(
            fit_result=FitResult.from_dict(fit_data),
            component_H=component_h,  # type: ignore[arg-type]
            component_O=component_o,  # type: ignore[arg-type]
            warnings=tuple(warnings),  # type: ignore[arg-type]
            model_name=model_name,
        )


@dataclass(frozen=True, slots=True)
class FitParameterPresentation:
    """One parameter row displayed by the Fit page."""

    name: str
    estimate: float
    unit: str
    ci_lower: float | None
    ci_upper: float | None
    bound_hit: bool | None
    bound_side: str | None
    reliability_flag: bool
    reliability_reason: str


@dataclass(frozen=True, slots=True)
class FitPresentation:
    """Detached arrays, parameter rows, and reliability status for the UI."""

    time_s: tuple[float, ...]
    data: tuple[float, ...]
    fit: tuple[float, ...]
    component_H: tuple[float, ...]
    component_O: tuple[float, ...]
    residuals: tuple[float, ...]
    parameters: tuple[FitParameterPresentation, ...]
    model_name: str
    SSE: float
    RMSE: float
    R2: float
    exit_status: int
    nfev: int
    reliable: bool
    reliability_label: str
    warnings: tuple[str, ...]
    component_1: tuple[float,...] = ()
    component_2: tuple[float,...] = ()
    component_labels: tuple[str,...] = ()
    component_baseline: float = 0.


@dataclass(frozen=True, slots=True)
class DualHOAnalysisConfig:
    """Validated service request for the existing staged M2/M3 core fitter."""

    m2_model: DualHOModel
    m3_model: DualHOModel | None = None
    fit_config: DualHOFitConfig = field(default_factory=DualHOFitConfig)
    run_m3: bool | None = None
    m3_evidence: M3TriggerEvidence = field(default_factory=M3TriggerEvidence)

    def validate(self) -> None:
        if not isinstance(self.m2_model, DualHOModel):
            raise TypeError("m2_model must be a DualHOModel")
        if self.m3_model is not None and not isinstance(self.m3_model, DualHOModel):
            raise TypeError("m3_model must be a DualHOModel or None")
        if not isinstance(self.fit_config, DualHOFitConfig):
            raise TypeError("fit_config must be a DualHOFitConfig")
        if self.run_m3 is not None and not isinstance(self.run_m3, bool):
            raise TypeError("run_m3 must be True, False, or None")
        if not isinstance(self.m3_evidence, M3TriggerEvidence):
            raise TypeError("m3_evidence must be M3TriggerEvidence")

    @property
    def parameter_units(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "D_H_cm2_s": "cm^2/s",
                "D_O_cm2_s": "cm^2/s",
                "k_H_cm_s": "cm/s",
                "k_O_cm_s": "cm/s",
                "A_H": "signal",
                "A_O": "signal",
                "y0": "signal",
            }
        )

    @property
    def parameter_bounds(self) -> Mapping[str, tuple[float, float]]:
        d_bounds = tuple(10.0**value for value in self.fit_config.log10_D_bounds)
        k_bounds = tuple(10.0**value for value in self.fit_config.log10_k_bounds)
        amplitude_bounds = tuple(
            10.0**value for value in self.fit_config.log10_amplitude_bounds
        )
        result: dict[str, tuple[float, float]] = {
            "D_H_cm2_s": d_bounds,
            "D_O_cm2_s": d_bounds,
            "k_H_cm_s": k_bounds,
            "k_O_cm_s": k_bounds,
            "A_H": amplitude_bounds,
            "A_O": amplitude_bounds,
        }
        if self.fit_config.fit_baseline and self.fit_config.baseline_bounds is not None:
            result["y0"] = self.fit_config.baseline_bounds
        return MappingProxyType(result)


class DualHOBackend:
    """Production AnalysisConfig adapter for staged M2/M3 fitting."""

    def __init__(self, *, model_factory: ModelFactory | None = None) -> None:
        self._model_factory = model_factory or ModelFactory()

    def __call__(
        self,
        dataset: EcrDataset,
        config: object,
        *,
        cancel_token: CancellationToken,
        progress_callback: ProgressReporter,
    ) -> CoreFitOutput:
        if not isinstance(config, AnalysisConfig):
            raise TypeError("DualHOBackend requires AnalysisConfig")
        if "fixed_d" in config.model_specific:
            from pyecr.fitting.fixed_d import fit_fixed_diffusion
            fit=fit_fixed_diffusion(dataset,config,cancel_check=cancel_token.raise_if_cancelled,progress_callback=progress_callback)
            return CoreFitOutput(fit_result=fit,model_name=config.model_id.value)
        if config.model_id not in {
            ModelID.M2_DUAL_HO_D,
            ModelID.M3_DUAL_HO_DK,
        }:
            raise ValueError("DualHOBackend supports only M2_DUAL_HO_D and M3_DUAL_HO_DK")
        cancel_token.raise_if_cancelled()
        progress_callback(0.04, "Assembling production M2/M3 models")

        m2_config = _m2_factory_config(config.model)
        m2_model = self._model_factory.create(m2_config)
        if not isinstance(m2_model, DualHOModel):
            raise TypeError("ModelFactory did not return a DualHOModel for M2")
        m3_model: DualHOModel | None = None
        run_m3 = config.model_id is ModelID.M3_DUAL_HO_DK
        if run_m3:
            candidate = self._model_factory.create(config)
            if not isinstance(candidate, DualHOModel):
                raise TypeError("ModelFactory did not return a DualHOModel for M3")
            m3_model = candidate

        fit_config, evidence = _dual_fit_adapter(config)

        def fit_progress(fraction: float, message: str) -> None:
            cancel_token.raise_if_cancelled()
            progress_callback(0.06 + 0.74 * fraction, message)

        result = fit_dual_ho(
            dataset.processed_time,
            dataset.processed_signal,
            m2_model=m2_model,
            m3_model=m3_model,
            fit_config=fit_config,
            run_m3=run_m3,
            m3_evidence=evidence,
            cancel_check=cancel_token.raise_if_cancelled,
            progress_callback=fit_progress,
        )
        selected_candidates = (
            result.m3_candidates
            if result.selected_model_id is ModelID.M3_DUAL_HO_DK
            else result.m2a_candidates
        )
        selected_fit = replace(
            result.selected_fit,
            candidates=tuple(
                FitCandidate(
                    start_index=candidate.start_index,
                    params=candidate.params,
                    SSE=candidate.SSE,
                    exit_status=candidate.exit_status,
                    nfev=candidate.nfev,
                    success=candidate.success,
                )
                for candidate in selected_candidates
            ),
        )
        persisted_fit = _fit_with_persisted_artifacts(
            selected_fit,
            component_h=result.component_H,
            component_o=result.component_O,
            warnings=result.warnings,
            model_name=result.selected_model_id.value,
            model_diagnostics=result.diagnostics,
            dispatch={
                "requested_model_id": config.model_id.value,
                "selected_model_id": result.selected_model_id.value,
                "m3_ran": bool(result.diagnostics["m3_ran"]),
                "m3_decision": str(result.diagnostics["m3_decision"]),
            },
        )
        cancel_token.raise_if_cancelled()
        progress_callback(0.82, "Production dual H/O fit complete")
        return CoreFitOutput(
            fit_result=persisted_fit,
            component_H=result.component_H,
            component_O=result.component_O,
            warnings=result.warnings,
            model_name=result.selected_model_id.value,
        )


class AnalysisBackendNotRegisteredError(LookupError):
    """Raised when no production analysis backend exists for a ModelID."""


def _surface_k_backend(dataset, config, *, cancel_token, progress_callback):
    result = fit_surface_k(dataset, config, cancel_check=cancel_token.raise_if_cancelled,
                           progress_callback=progress_callback)
    cancel_token.raise_if_cancelled()
    return CoreFitOutput(fit_result=result, model_name=ModelID.M7_SINGLE_K.value)

def _dual_surface_k_backend(dataset,config,*,cancel_token,progress_callback):
    result=fit_dual_surface_k(dataset,config,cancel_check=cancel_token.raise_if_cancelled,progress_callback=progress_callback)
    cancel_token.raise_if_cancelled()
    return CoreFitOutput(fit_result=result,model_name=ModelID.M8_DUAL_K.value)


class AnalysisBackendRegistry:
    """Immutable ModelID-to-backend dispatcher used by AnalysisService."""

    def __init__(
        self,
        backends: Mapping[ModelID, AnalysisBackend] | None = None,
    ) -> None:
        if backends is None:
            single = SingleEcrBackend()
            dual = DualHOBackend()
            backends = {
                ModelID.M0_SINGLE_D: single,
                ModelID.M1_SINGLE_DK: single,
                ModelID.M2_DUAL_HO_D: dual,
                ModelID.M3_DUAL_HO_DK: dual,
                ModelID.M7_SINGLE_K: _surface_k_backend,
                ModelID.M8_DUAL_K: _dual_surface_k_backend,
            }
        if not isinstance(backends, Mapping):
            raise TypeError("backends must be a mapping")
        normalized: dict[ModelID, AnalysisBackend] = {}
        for model_id, backend in backends.items():
            if not isinstance(model_id, ModelID):
                raise TypeError("analysis backend keys must be ModelID values")
            if not callable(backend):
                raise TypeError(f"backend for {model_id.value} must be callable")
            normalized[model_id] = backend
        self._backends = MappingProxyType(normalized)

    @classmethod
    def production(cls) -> AnalysisBackendRegistry:
        """Create the production M0--M3 dispatch table."""

        return cls()

    @property
    def model_ids(self) -> tuple[ModelID, ...]:
        return tuple(model_id for model_id in ModelID if model_id in self._backends)

    def get(self, model_id: ModelID | str) -> AnalysisBackend:
        try:
            normalized = ModelID(model_id)
        except (TypeError, ValueError) as exc:
            raise AnalysisBackendNotRegisteredError(
                f"no analysis backend is registered for {model_id!r}"
            ) from exc
        try:
            return self._backends[normalized]
        except KeyError as exc:
            suffix = (
                "; M4 is a boundary composition capability, not a standalone "
                "fit backend"
                if normalized is ModelID.M4_FLUSH
                else ""
            )
            raise AnalysisBackendNotRegisteredError(
                f"no analysis backend is registered for {normalized.value}{suffix}"
            ) from exc


@dataclass(frozen=True, slots=True)
class _StoredArtifacts:
    component_H: NDArray[np.float64] | None
    component_O: NDArray[np.float64] | None
    uncertainty: LocalUncertaintyResult | None
    warnings: tuple[str, ...]
    model_name: str | None
    root_cache: RootCacheDiagnostics | None = None
    component_cache: ComponentCacheDiagnostics | None = None
    solver_runs: tuple[SolverRun, ...] = ()


class AnalysisService:
    """Validate, execute, and describe core fits for non-core clients.

    ``fit(dataset, config)`` always returns the shared :class:`FitResult`.
    H/O components and uncertainty are consumed through :meth:`describe_fit`.
    Production dual-model components are also embedded as JSON-compatible
    FitResult diagnostics, so project persistence does not depend on this
    service instance's bounded in-memory cache.
    """

    def __init__(
        self,
        *,
        backend: AnalysisBackend | None = None,
        solver_options: SolverOptions | None = None,
        coarse_to_fine_options: CoarseToFineOptions | None = None,
        solver_full_provider: Callable | None = None,
        backend_registry: AnalysisBackendRegistry | None = None,
        artifact_cache_size: int = 32,
        root_cache_enabled: bool = True,
        component_cache_enabled: bool = True,
        root_mode: str = "on_demand_roots",
    ) -> None:
        if isinstance(artifact_cache_size, bool) or not isinstance(
            artifact_cache_size, int
        ):
            raise TypeError("artifact_cache_size must be an integer")
        if artifact_cache_size <= 0:
            raise ValueError("artifact_cache_size must be positive")
        if backend is not None and backend_registry is not None:
            raise ValueError("backend and backend_registry cannot both be supplied")
        if backend is not None and not callable(backend):
            raise TypeError("backend must be callable")
        if backend_registry is not None and not isinstance(
            backend_registry, AnalysisBackendRegistry
        ):
            raise TypeError("backend_registry must be an AnalysisBackendRegistry")
        if not isinstance(root_cache_enabled, bool):
            raise TypeError("root_cache_enabled must be boolean")
        if solver_options is not None and not isinstance(solver_options, SolverOptions):
            raise TypeError("solver_options must be SolverOptions or None")
        self._solver_options = solver_options
        if coarse_to_fine_options is not None and not isinstance(coarse_to_fine_options, CoarseToFineOptions):
            raise TypeError("coarse_to_fine_options must be CoarseToFineOptions or None")
        self._coarse_to_fine_options = coarse_to_fine_options
        self._solver_full_provider = solver_full_provider
        if not isinstance(component_cache_enabled, bool):
            raise TypeError("component_cache_enabled must be boolean")
        self._root_cache_enabled = root_cache_enabled
        self._component_cache_enabled = component_cache_enabled
        if root_mode not in ROOT_MODES:
            raise ValueError(f"root_mode must be one of {ROOT_MODES}")
        self._root_mode = root_mode
        self._backend = backend
        self._backend_registry = (
            backend_registry or AnalysisBackendRegistry.production()
        )
        self._artifact_cache_size = artifact_cache_size
        self._artifacts: OrderedDict[int, _StoredArtifacts] = OrderedDict()
        self._lock = Lock()

    def fit(
        self,
        dataset: EcrDataset,
        config: object,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
        progress_event_callback: Callable[[FitProgressEvent], None] | None = None,
        job_id: str | None = None,
        progress_revision: object | None = None,
        iteration_observer=None,
    ) -> FitResult:
        """Fit all eligible rows, optionally screening experimental candidates."""
        import hashlib, json
        from pyecr.fitting.reproducibility import bind_config,random_job_scope
        config,random_job=bind_config(config)
        options = self._solver_options or SolverOptions.from_config(config)
        if iteration_observer is not None and hasattr(iteration_observer,'solve_monitored'):
            if self._backend is not None or self._solver_full_provider is not None:
                raise SolverRejected('Monitored Unsupported with custom backend/full-provider composition')
            from pyecr.fitting.production_stop import configure_production
            configure_production(iteration_observer,config,self._coarse_to_fine_options)
            if cancel_token is not None:iteration_observer.cancel_check=cancel_token.raise_if_cancelled
        screening = self._coarse_to_fine_options
        if isinstance(config, AnalysisConfig) and (config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} or "fixed_d" in config.model_specific):
            if (options.strategy != "trf" or self._solver_full_provider is not None
                    or (screening is not None and screening.enabled)):
                raise ValueError("M7 supports full-data TRF only; P60/portfolio/provider composition Unsupported")
        if screening is not None and screening.enabled:
            if not isinstance(dataset, EcrDataset) or not isinstance(config, AnalysisConfig):
                raise ValueError("P60 requires EcrDataset and AnalysisConfig")
            if self._backend is not None or options.strategy != "trf" or self._solver_full_provider is not None:
                raise ValueError("P60 requires production backend and full-data TRF")
            if config.fit.loss != "linear" or config.model.boundary_mode != "ideal_step":
                raise ValueError("P60 supports linear loss and ideal_step only")
            for section in ("single_ecr", "dual_ho"):
                if config.model_specific.get(section, {}).get("jacobian_mode", "legacy") != "legacy":
                    raise ValueError("P60 composition with experimental Jacobian is unsupported")
        if options.strategy == "lm_transformed_experimental" and getattr(getattr(config,"fit",None),"loss","linear") != "linear":
            raise SolverRejected("LM transformed experimental supports only linear loss")
        provenance = {"seed": random_job.get('resolved_random_seed'), "job_id": job_id,
            "config_SHA256": hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True,default=str).encode()).hexdigest() if hasattr(config,"to_dict") else None,
            "dataset_revision": getattr(progress_revision,"data_revision",None),
            "config_revision": getattr(progress_revision,"config_revision",None)}
        from pyecr.fitting.iteration_observation import observation_scope
        with observation_scope(iteration_observer,dataset,config,job_id=job_id,screening=screening,solver_options=options), \
                random_job_scope(random_job), coarse_to_fine_scope(screening, original_indices=(
                np.flatnonzero(dataset.mask) if screening is not None and screening.enabled
                and isinstance(dataset, EcrDataset) else None)), \
                solver_scope(options, full_provider=self._solver_full_provider, provenance=provenance) as ledger:
            with fit_telemetry_scope(progress_event_callback, job_id=job_id,
                                     revision=progress_revision):
                try:
                    fitted=self._fit_impl(dataset, config, cancel_token=cancel_token,
                                          progress_callback=progress_callback)
                    if iteration_observer is not None:iteration_observer.finish_fit(fitted)
                    return fitted
                except BaseException as exc:
                    if hasattr(exc,"__dict__"):
                        exc.solver_runs = ledger.snapshots()
                        exc.fit_random_job = random_job
                        if iteration_observer is not None:
                            exc.iteration_observer = iteration_observer
                    raise

    def _fit_impl(
        self,
        dataset: EcrDataset,
        config: object,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
    ) -> FitResult:
        """Validate then run a fit through the configured core backend."""

        if not isinstance(dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        _validate_config(config)
        token = cancel_token or CancellationToken()
        if not isinstance(token, CancellationToken):
            raise TypeError("cancel_token must be a CancellationToken")
        reporter = progress_callback or _ignore_progress

        token.raise_if_cancelled()
        reporter(0.02, "Configuration validated")

        def cancellable_reporter(fraction: float, message: str) -> None:
            token.raise_if_cancelled()
            reporter(fraction, message)

        backend = self._resolve_backend(config)
        model_id = config.model_id.value if isinstance(config, AnalysisConfig) else "legacy-dual"
        boundary_mode = config.model.boundary_mode if isinstance(config, AnalysisConfig) else "legacy-ideal-step"
        with RootCacheScope(enabled=self._root_cache_enabled,
                            cancel_check=token.raise_if_cancelled,
                            root_mode=self._root_mode) as root_cache:
            with ComponentCacheScope(
                enabled=self._component_cache_enabled,
                cancel_check=token.raise_if_cancelled,
                model_id=model_id,
                boundary_mode=boundary_mode,
                signal_type=str(dataset.metadata.get("signal_type", "unknown")),
                signal_unit=str(dataset.metadata.get("signal_unit", "unknown")),
            ) as component_cache:
                try:
                    output = backend(
                        dataset,
                        config,
                        cancel_token=token,
                        progress_callback=cancellable_reporter,
                    )
                    token.raise_if_cancelled()
                    fit_result, artifacts = _normalise_backend_output(output)
                    if fit_result.y_fit.size != dataset.processed_signal.size:
                        raise ValueError("fit result length does not match the processed dataset")
                finally:
                    capture_fit_cache()
        runs = current_solver_runs()
        warnings = tuple(dict.fromkeys(w for run in runs for w in run.payload.get("warnings",())))
        nonlegacy = self._solver_options or SolverOptions.from_config(config)
        screening = current_screening()
        if screening is not None:
            if not screening.stages or any(s["status"] != "completed" for s in screening.stages):
                raise RuntimeError("P60 backend did not complete full-data refinement")
            fit_result = replace(fit_result, diagnostics={**fit_result.diagnostics,
                "experimental_coarse_to_fine": screening.to_dict()})
        if nonlegacy.strategy != "trf":
            fit_result = replace(fit_result, diagnostics={**fit_result.diagnostics,
                "local_solver": {"strategy":nonlegacy.strategy,"version":SOLVER_VERSION,
                    "experimental":True,"loss":getattr(getattr(config,"fit",None),"loss","linear"),
                    "warnings":list(warnings),"no_automatic_fallback":True}})
        artifacts = replace(artifacts, root_cache=root_cache.diagnostics,
                            component_cache=component_cache.diagnostics,
                            solver_runs=runs, warnings=tuple(dict.fromkeys((*artifacts.warnings,*warnings))))
        self._store_artifacts(fit_result, artifacts)
        fit_stage("Finalizing", "Core result complete; preparing display artifacts")
        reporter(0.86, "Core fit completed; preparing uncertainty and display data")
        return fit_result

    def root_cache_diagnostics(self, fit_result: FitResult) -> RootCacheDiagnostics | None:
        """Transient task counters, separate from serialized scientific results.

        None means the task is no longer in this service's bounded artifact store
        (or the result was loaded from a project). Root values are never retained.
        """
        return self._lookup_artifacts(fit_result).root_cache

    def component_cache_diagnostics(self, fit_result: FitResult) -> ComponentCacheDiagnostics | None:
        """Transient worker-local counters, excluded from FitResult and project state."""
        return self._lookup_artifacts(fit_result).component_cache

    def solver_diagnostics(self, fit_result: FitResult) -> tuple[SolverRun, ...]:
        """Transient audit records; loaded projects retain method, not timings."""
        return self._lookup_artifacts(fit_result).solver_runs

    def _resolve_backend(self, config: object) -> AnalysisBackend:
        if self._backend is not None:
            return self._backend
        if isinstance(config, AnalysisConfig):
            return self._backend_registry.get(config.model_id)
        if isinstance(config, DualHOAnalysisConfig):
            return _default_core_backend
        raise TypeError(
            "default AnalysisService requires AnalysisConfig or "
            "DualHOAnalysisConfig"
        )

    def describe_fit(
        self,
        dataset: EcrDataset,
        config: object,
        fit_result: FitResult,
    ) -> FitPresentation:
        """Build detached plot/table values and a conservative reliability flag."""

        if not isinstance(dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        if not isinstance(fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult")
        if fit_result.y_fit.size != dataset.processed_signal.size:
            raise ValueError("fit result length does not match the processed dataset")
        artifacts = self._lookup_artifacts(fit_result)
        warnings = [*fit_result.warnings, *artifacts.warnings]
        uncertainty = artifacts.uncertainty
        if uncertainty is None and fit_result.jacobian is not None:
            try:
                uncertainty = estimate_local_uncertainty(fit_result)
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                warnings.append(
                    "LOCAL_UNCERTAINTY_UNAVAILABLE: "
                    f"{type(exc).__name__}: {exc}"
                )
        if uncertainty is not None:
            warnings.extend(uncertainty.warnings)
        else:
            warnings.append(
                "LOCAL_CI_UNAVAILABLE: parameter reliability cannot be confirmed."
            )

        bounds = _parameter_bounds(config)
        if isinstance(config, AnalysisConfig) and config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}:
            surface_record=fit_result.diagnostics.get("surface_k",fit_result.diagnostics.get("dual_surface_k",{}))
            bounds = surface_record["parameter_bounds"]
        units = _parameter_units(config)
        signal_unit = str(dataset.metadata.get("signal_unit", "signal"))
        severe_fit_warning = any(_warning_is_reliability_critical(item) for item in warnings)
        rows: list[FitParameterPresentation] = []
        for name, estimate in fit_result.params.items():
            interval = (
                None
                if uncertainty is None
                else uncertainty.confidence_intervals.get(name)
            )
            lower = None if interval is None else float(interval.lower)
            upper = None if interval is None else float(interval.upper)
            surface_record=fit_result.diagnostics.get("surface_k",fit_result.diagnostics.get("dual_surface_k",{}))
            if (isinstance(config, AnalysisConfig) and config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}
                    and not surface_record.get("ci_availability",{}).get(name,surface_record.get("ci_available",False))):
                interval = None
                lower = upper = None
            if (fit_result.diagnostics.get("fixed_d",{}).get("ci_available") is False):
                interval=None;lower=upper=None
            hit, side = _bound_contact(
                estimate,
                bounds.get(name),
                is_log=fit_result.param_space[name] == "log",
            )
            reasons: list[str] = []
            if interval is None:
                reasons.append("95% CI unavailable")
            elif not _finite_interval(lower, upper):
                reasons.append("95% CI is non-finite")
            elif fit_result.param_space[name] == "log" and _interval_decades(
                lower, upper
            ) > 2.0:
                reasons.append("95% CI spans more than 2 decades")
            if hit:
                reasons.append(f"estimate touches {side} bound")
                warnings.append(
                    f"PARAMETER_BOUND_HIT: {name} touches the {side} bound."
                )
            if uncertainty is not None and not uncertainty.reliable_separation:
                reasons.append("local parameter separation is unreliable")
            if fit_result.exit_status <= 0:
                reasons.append("optimizer did not report successful termination")
            if severe_fit_warning:
                reasons.append("fit carries a reliability-critical warning")
            row_reliable = not reasons
            unit = units.get(name, _infer_parameter_unit(name, signal_unit))
            if unit == "signal":
                unit = signal_unit
            rows.append(
                FitParameterPresentation(
                    name=name,
                    estimate=float(estimate),
                    unit=unit,
                    ci_lower=lower,
                    ci_upper=upper,
                    bound_hit=hit,
                    bound_side=side,
                    reliability_flag=row_reliable,
                    reliability_reason="reliable" if row_reliable else "; ".join(reasons),
                )
            )

        warnings = list(dict.fromkeys(warnings))
        model_reliable = bool(rows) and all(row.reliability_flag for row in rows)
        if not model_reliable:
            warnings.append(
                "MODEL_NOT_RELIABLE: inspect confidence intervals, bounds, and "
                "identifiability before interpreting fitted parameters."
            )
        warnings = list(dict.fromkeys(warnings))
        component_h = _component_tuple(artifacts.component_H)
        component_o = _component_tuple(artifacts.component_O)
        model_name = artifacts.model_name or _model_name(config)
        expects_ho_components = model_name in {
            "M2_DUAL_HO_D",
            "M3_DUAL_HO_DK",
        }
        if expects_ho_components and (not component_h or not component_o):
            warnings.append(
                "HO_COMPONENTS_NOT_AVAILABLE: this fit backend did not return both "
                "H and O components."
            )

        return FitPresentation(
            time_s=tuple(float(value) for value in dataset.processed_time),
            data=tuple(float(value) for value in dataset.processed_signal),
            fit=tuple(float(value) for value in fit_result.y_fit),
            component_H=component_h,
            component_O=component_o,
            residuals=tuple(float(value) for value in fit_result.residuals),
            parameters=tuple(rows),
            model_name=model_name,
            SSE=fit_result.SSE,
            RMSE=fit_result.RMSE,
            R2=fit_result.R2,
            exit_status=fit_result.exit_status,
            nfev=fit_result.nfev,
            reliable=model_reliable,
            reliability_label=(
                "Local parameter checks passed; review Diagnostics credibility gates"
                if model_reliable else "Local parameter checks warn; review Diagnostics"
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            component_1=tuple(fit_result.diagnostics.get("dual_surface_k",{}).get("component_1",())),
            component_2=tuple(fit_result.diagnostics.get("dual_surface_k",{}).get("component_2",())),
            component_labels=tuple(fit_result.diagnostics.get("dual_surface_k",{}).get("component_labels",())),
            component_baseline=float(fit_result.diagnostics.get("dual_surface_k",{}).get("component_baseline",0.)),
        )

    def _store_artifacts(
        self,
        fit_result: FitResult,
        artifacts: _StoredArtifacts,
    ) -> None:
        with self._lock:
            self._artifacts[id(fit_result)] = artifacts
            self._artifacts.move_to_end(id(fit_result))
            while len(self._artifacts) > self._artifact_cache_size:
                self._artifacts.popitem(last=False)

    def _lookup_artifacts(self, fit_result: FitResult) -> _StoredArtifacts:
        with self._lock:
            cached = self._artifacts.get(id(fit_result))
        if cached is not None:
            return cached
        return _persisted_artifacts_from_fit_result(fit_result)


def _m2_factory_config(model: ModelConfig) -> ModelConfig:
    """Build the M2 companion required by the staged M3 workflow."""

    if model.model_id is ModelID.M2_DUAL_HO_D:
        return model
    if model.model_id is not ModelID.M3_DUAL_HO_DK:
        raise ValueError("dual factory adaptation requires M2 or M3")
    m2_names = {"D_H_cm2_s", "D_O_cm2_s", "A_H", "A_O", "y0"}
    return ModelConfig(
        model_id=ModelID.M2_DUAL_HO_D,
        geometry=model.geometry,
        surface_mode="instant",
        boundary_mode=model.boundary_mode,
        step_type=model.step_type,
        sign_H=model.sign_H,
        sign_O=model.sign_O,
        initial_values={
            name: value
            for name, value in model.initial_values.items()
            if name in m2_names
        },
        parameter_bounds={
            name: bounds
            for name, bounds in model.parameter_bounds.items()
            if name in m2_names
        },
        free_parameters=tuple(
            name for name in model.free_parameters if name in m2_names
        ),
        fixed_parameters={
            name: value
            for name, value in model.fixed_parameters.items()
            if name in m2_names
        },
    )


def _dual_fit_adapter(
    config: AnalysisConfig,
) -> tuple[DualHOFitConfig, M3TriggerEvidence]:
    """Translate unified configuration into the existing staged fitter policy."""

    raw = config.model_specific.get("dual_ho", {})
    if not isinstance(raw, Mapping):
        raise TypeError("model_specific['dual_ho'] must be a mapping")
    allowed = {
        "bi_trigger_max",
        "condition_warning_threshold",
        "correlation_warning_threshold",
        "kernel_options",
        "label_constraint",
        "m2_n_starts",
        "m3_evidence",
        "m3_n_starts",
        "n_candidates",
        "random_seed",
        "residual_autocorrelation_threshold",
        "residual_relative_rmse_threshold",
        "shape_grid_points",
        "surface_sensitivity_trigger",
        "run_m3",
        "jacobian_mode",
        "start_policy",
        "linear_projection",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "unknown dual_ho options: " + ", ".join(str(name) for name in unknown)
        )
    if (raw.get("jacobian_mode", "legacy") == "experimental_hybrid_v1"
            and config.model_id is not ModelID.M3_DUAL_HO_DK):
        raise ValueError("experimental_hybrid_v1 is available only for M3, not M2")

    requested_run_m3 = raw.get(
        "run_m3", config.model_id is ModelID.M3_DUAL_HO_DK
    )
    if not isinstance(requested_run_m3, bool):
        raise TypeError("dual_ho.run_m3 must be boolean")
    expected_run_m3 = config.model_id is ModelID.M3_DUAL_HO_DK
    if requested_run_m3 is not expected_run_m3:
        raise ValueError(
            "dual_ho.run_m3 must agree with the selected M2/M3 model_id"
        )

    required = {"D_H_cm2_s", "D_O_cm2_s", "A_H", "A_O"}
    if config.model_id is ModelID.M3_DUAL_HO_DK:
        required.update({"k_H_cm_s", "k_O_cm_s"})
    fixed_conflicts = sorted(required.intersection(config.model.fixed_parameters))
    if fixed_conflicts:
        raise ValueError(
            "the production staged fitter requires these parameters to be free: "
            + ", ".join(fixed_conflicts)
        )
    declared_free = set(config.model.free_parameters)
    if declared_free and not required.issubset(declared_free):
        missing = sorted(required - declared_free)
        raise ValueError(
            "free_parameters is missing required dual parameters: "
            + ", ".join(missing)
        )

    d_bounds = _shared_log10_bounds(
        config.model,
        ("D_H_cm2_s", "D_O_cm2_s"),
        (1.0e-12, 1.0e-2),
    )
    k_bounds = _shared_log10_bounds(
        config.model,
        ("k_H_cm_s", "k_O_cm_s"),
        (1.0e-10, 1.0),
    )
    amplitude_bounds = _shared_log10_bounds(
        config.model,
        ("A_H", "A_O"),
        (1.0e-14, 10.0),
    )
    fit_baseline = "y0" in declared_free
    fixed_baseline = config.model.fixed_parameters.get("y0")
    baseline_bounds = (
        config.model.parameter_bounds.get("y0") if fit_baseline else None
    )
    kernel_options = raw.get("kernel_options", {})
    if not isinstance(kernel_options, Mapping):
        raise TypeError("dual_ho.kernel_options must be a mapping")
    merged_kernel_options = dict(kernel_options)
    for reserved in ("term_tol", "max_terms"):
        if reserved in merged_kernel_options:
            raise ValueError(
                f"dual_ho.kernel_options cannot override solver-owned {reserved}"
            )
    merged_kernel_options.update(
        {
            "term_tol": config.solver.relative_tolerance,
            "max_terms": config.solver.max_terms,
        }
    )
    label_constraint = raw.get("label_constraint", "D_H>=D_O")
    if label_constraint == "dh_ge_do":
        label_constraint = "D_H>=D_O"
    fit_config = DualHOFitConfig(
        fit_baseline=fit_baseline,
        fixed_baseline=fixed_baseline,
        log10_D_bounds=d_bounds,
        log10_k_bounds=k_bounds,
        log10_amplitude_bounds=amplitude_bounds,
        baseline_bounds=baseline_bounds,
        label_constraint=label_constraint,  # type: ignore[arg-type]
        m2_n_starts=raw.get("m2_n_starts", 8),  # type: ignore[arg-type]
        m3_n_starts=raw.get("m3_n_starts", 20),  # type: ignore[arg-type]
        n_candidates=raw.get("n_candidates", 5),  # type: ignore[arg-type]
        random_seed=raw.get("random_seed", 20260915),  # type: ignore[arg-type]
        max_nfev=config.fit.max_nfev,
        loss=config.fit.loss,
        residual_autocorrelation_threshold=raw.get(
            "residual_autocorrelation_threshold", 0.30
        ),  # type: ignore[arg-type]
        residual_relative_rmse_threshold=raw.get(
            "residual_relative_rmse_threshold", 1.0e-5
        ),  # type: ignore[arg-type]
        bi_trigger_max=raw.get("bi_trigger_max", 30.0),  # type: ignore[arg-type]
        surface_sensitivity_trigger=raw.get(
            "surface_sensitivity_trigger", 0.50
        ),  # type: ignore[arg-type]
        correlation_warning_threshold=raw.get(
            "correlation_warning_threshold", 0.95
        ),  # type: ignore[arg-type]
        condition_warning_threshold=raw.get(
            "condition_warning_threshold", 1.0e8
        ),  # type: ignore[arg-type]
        shape_grid_points=raw.get("shape_grid_points", 2_048),  # type: ignore[arg-type]
        kernel_options=merged_kernel_options,
        jacobian_mode=raw.get("jacobian_mode", "legacy"),  # type: ignore[arg-type]
        initial_values=config.model.initial_values,
        start_policy=raw.get("start_policy", "informed_v2"),
        linear_projection=raw.get("linear_projection", True),
    )

    raw_evidence = raw.get("m3_evidence", {})
    if not isinstance(raw_evidence, Mapping):
        raise TypeError("dual_ho.m3_evidence must be a mapping")
    evidence_unknown = sorted(
        set(raw_evidence) - {"bi_values", "surface_sensitivity_score"}
    )
    if evidence_unknown:
        raise ValueError(
            "unknown m3_evidence options: "
            + ", ".join(str(name) for name in evidence_unknown)
        )
    evidence = M3TriggerEvidence(
        bi_values=tuple(raw_evidence.get("bi_values", ())),  # type: ignore[arg-type]
        surface_sensitivity_score=raw_evidence.get(  # type: ignore[arg-type]
            "surface_sensitivity_score"
        ),
    )
    return fit_config, evidence


def _shared_log10_bounds(
    model: ModelConfig,
    names: tuple[str, str],
    default: tuple[float, float],
) -> tuple[float, float]:
    supplied = [model.parameter_bounds[name] for name in names if name in model.parameter_bounds]
    physical = default if not supplied else supplied[0]
    if any(bounds != physical for bounds in supplied[1:]):
        raise ValueError(
            f"the current staged fitter requires shared bounds for {names[0]} "
            f"and {names[1]}"
        )
    if physical[0] <= 0.0:
        raise ValueError(f"{names[0]}/{names[1]} bounds must be positive")
    return (log10(physical[0]), log10(physical[1]))


def _fit_with_persisted_artifacts(
    fit_result: FitResult,
    *,
    component_h: ArrayLike,
    component_o: ArrayLike,
    warnings: tuple[str, ...],
    model_name: str,
    model_diagnostics: Mapping[str, object],
    dispatch: Mapping[str, object],
) -> FitResult:
    component_h_array = _optional_component(
        "component_H", component_h, fit_result.y_fit.size
    )
    component_o_array = _optional_component(
        "component_O", component_o, fit_result.y_fit.size
    )
    diagnostics = dict(fit_result.diagnostics)
    diagnostics["analysis_artifacts"] = {
        "schema_version": 1,
        "component_H": component_h_array.tolist(),  # type: ignore[union-attr]
        "component_O": component_o_array.tolist(),  # type: ignore[union-attr]
        "warnings": list(warnings),
        "model_name": model_name,
    }
    diagnostics["dual_ho"] = _json_safe_artifact(model_diagnostics)
    diagnostics["analysis_dispatch"] = dict(dispatch)
    return replace(fit_result, diagnostics=diagnostics)


def _json_safe_artifact(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return [_json_safe_artifact(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_artifact(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe_artifact(item) for item in value]
    raise TypeError(
        f"dual H/O diagnostic value is not JSON-compatible: {type(value).__name__}"
    )


def _persisted_artifacts_from_fit_result(fit_result: FitResult) -> _StoredArtifacts:
    solver_warnings = tuple(fit_result.diagnostics.get("local_solver", {}).get("warnings", ()))
    payload = fit_result.diagnostics.get("analysis_artifacts")
    if payload is None:
        return _StoredArtifacts(None, None, None, solver_warnings, None)
    if not isinstance(payload, Mapping):
        raise TypeError("analysis_artifacts must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported analysis_artifacts schema_version")
    warnings = payload.get("warnings", ())
    if not isinstance(warnings, (tuple, list)) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise TypeError("analysis_artifacts warnings must be strings")
    model_name = payload.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        raise TypeError("analysis_artifacts model_name must be a string or None")
    return _StoredArtifacts(
        component_H=_optional_component(
            "component_H", payload.get("component_H"), fit_result.y_fit.size
        ),
        component_O=_optional_component(
            "component_O", payload.get("component_O"), fit_result.y_fit.size
        ),
        uncertainty=None,
        warnings=tuple(dict.fromkeys((*warnings,*solver_warnings))),
        model_name=model_name,
    )


def _default_core_backend(
    dataset: EcrDataset,
    config: object,
    *,
    cancel_token: CancellationToken,
    progress_callback: ProgressReporter,
) -> CoreFitOutput:
    if not isinstance(config, DualHOAnalysisConfig):
        raise TypeError(
            "the default analysis backend currently requires DualHOAnalysisConfig"
        )
    cancel_token.raise_if_cancelled()
    progress_callback(0.08, "Running staged M2/M3 core fit")

    def core_progress(fraction: float, message: str) -> None:
        progress_callback(0.08 + 0.72 * fraction, message)

    result = fit_dual_ho(
        dataset.processed_time,
        dataset.processed_signal,
        m2_model=config.m2_model,
        m3_model=config.m3_model,
        fit_config=config.fit_config,
        run_m3=config.run_m3,
        m3_evidence=config.m3_evidence,
        cancel_check=cancel_token.raise_if_cancelled,
        progress_callback=core_progress,
    )
    cancel_token.raise_if_cancelled()
    progress_callback(0.82, "Core optimizer finished")
    return CoreFitOutput(
        fit_result=result.selected_fit,
        component_H=result.component_H,
        component_O=result.component_O,
        warnings=result.warnings,
        model_name=result.selected_model_id.value,
    )


def _validate_config(config: object) -> None:
    validator = getattr(config, "validate", None)
    if not callable(validator):
        raise TypeError("analysis config must provide validate()")
    validator()


def _normalise_backend_output(
    output: FitResult | CoreFitOutput | DualHOFitResult,
) -> tuple[FitResult, _StoredArtifacts]:
    if isinstance(output, FitResult):
        return output, _StoredArtifacts(None, None, None, (), None)
    if isinstance(output, DualHOFitResult):
        return output.selected_fit, _StoredArtifacts(
            component_H=_optional_component(
                "component_H", output.component_H, output.selected_fit.y_fit.size
            ),
            component_O=_optional_component(
                "component_O", output.component_O, output.selected_fit.y_fit.size
            ),
            uncertainty=None,
            warnings=output.warnings,
            model_name=output.selected_model_id.value,
        )
    if isinstance(output, CoreFitOutput):
        return output.fit_result, _StoredArtifacts(
            component_H=output.component_H,
            component_O=output.component_O,
            uncertainty=output.uncertainty,
            warnings=output.warnings,
            model_name=output.model_name,
        )
    raise TypeError("analysis backend must return FitResult or CoreFitOutput")


def _optional_component(
    name: str,
    values: ArrayLike | None,
    expected_size: int,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if result.size != expected_size:
        raise ValueError(f"{name} must align with fitted observations")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    result.flags.writeable = False
    return result


def _component_tuple(values: NDArray[np.float64] | None) -> tuple[float, ...]:
    return () if values is None else tuple(float(value) for value in values)


def _parameter_units(config: object) -> Mapping[str, str]:
    value = getattr(config, "parameter_units", {})
    if not isinstance(value, Mapping):
        raise TypeError("config.parameter_units must be a mapping")
    return {str(name): str(unit) for name, unit in value.items()}


def _parameter_bounds(config: object) -> Mapping[str, tuple[float, float]]:
    value = getattr(config, "parameter_bounds", {})
    if not isinstance(value, Mapping):
        raise TypeError("config.parameter_bounds must be a mapping")
    result: dict[str, tuple[float, float]] = {}
    for name, bounds in value.items():
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"parameter_bounds[{name!r}] must contain two values")
        lower = _finite_scalar(f"parameter_bounds[{name!r}][0]", bounds[0])
        upper = _finite_scalar(f"parameter_bounds[{name!r}][1]", bounds[1])
        if lower >= upper:
            raise ValueError(f"parameter_bounds[{name!r}] must be increasing")
        result[str(name)] = (lower, upper)
    return result


def _finite_interval(lower: float | None, upper: float | None) -> bool:
    return (
        lower is not None
        and upper is not None
        and isfinite(lower)
        and isfinite(upper)
        and lower <= upper
    )


def _interval_decades(lower: float | None, upper: float | None) -> float:
    if lower is None or upper is None or lower <= 0.0 or upper <= 0.0:
        return float("inf")
    return log10(upper / lower)


def _warning_is_reliability_critical(warning: str) -> bool:
    upper = warning.upper()
    return any(
        token in upper
        for token in (
            "UNRELIABLE",
            "IDENTIFIAB",
            "RANK_DEFICIENT",
            "ILL_CONDITION",
            "PARAMETER_CORRELATION",
            "CORRELATION_WARNING",
            "PARAMETER_BOUND",
            "LABEL_SWITCHING",
        )
    )


def _infer_parameter_unit(name: str, signal_unit: str) -> str:
    lowered = name.lower()
    if lowered.startswith("d_") or lowered in {"d", "d_cm2_s"}:
        return "cm^2/s"
    if lowered.startswith("k_") or lowered in {"k", "k_cm_s"}:
        return "cm/s"
    if lowered.startswith("a_") or lowered in {"amplitude", "baseline", "y0"}:
        return signal_unit
    return "—"


def _model_name(config: object) -> str:
    value = getattr(config, "model_id", None)
    if value is None:
        return type(config).__name__
    return str(getattr(value, "value", value))


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ignore_progress(fraction: float, message: str) -> None:
    del fraction, message
