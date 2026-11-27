"""Worker-safe orchestration for P44's unchanged M3 H/O objective grids."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from math import isclose
from numbers import Integral
from threading import Lock
from typing import Callable, Mapping, TYPE_CHECKING

from pyecr.config import AnalysisConfig
from pyecr.diagnostics.dual_error_map import (
    DUAL_FIXED_OBJECTIVE_KIND,
    DualErrorMapConfig,
    DualErrorMapResult,
    compute_dual_error_map,
)
from pyecr.diagnostics.error_map import ErrorMapMode, ErrorMapResolution
from pyecr.fitting.base import FitResult
from pyecr.fitting.dual_objective import DualObjectiveRevision
from pyecr.io.dataset import EcrDataset
from pyecr.models.ids import ModelID
from pyecr.report.snapshot import dataset_sha256
from pyecr.services.analysis import CancellationToken

if TYPE_CHECKING:
    from pyecr.services.workflow import AnalysisSession

__all__ = [
    "DualErrorMapComputation", "DualErrorMapContext", "DualErrorMapService",
    "DualMapProgress",
]


@dataclass(frozen=True, slots=True)
class DualMapProgress:
    """Independent H/O completion fractions and their combined fraction."""

    h_fraction: float
    o_fraction: float
    combined_fraction: float
    phase: str
    message: str

    def __post_init__(self) -> None:
        for name in ("h_fraction", "o_fraction", "combined_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.phase not in {"queued", "H", "O", "complete", "cache"}:
            raise ValueError("unknown dual Error Map progress phase")


@dataclass(frozen=True, slots=True)
class DualErrorMapContext:
    """Immutable, private-worker inputs captured from one current M3 fit."""

    dataset: EcrDataset
    analysis_config: AnalysisConfig
    fit_result: FitResult
    dataset_sha256: str
    revision: DualObjectiveRevision
    fit_generation: int
    selected_model_id: ModelID

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, EcrDataset):
            raise TypeError("dataset must be EcrDataset")
        if not isinstance(self.analysis_config, AnalysisConfig):
            raise TypeError("analysis_config must be AnalysisConfig")
        self.analysis_config.validate()
        if self.analysis_config.model_id is not ModelID.M3_DUAL_HO_DK:
            raise ValueError("dual Error Map requires M3_DUAL_HO_DK configuration")
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be FitResult")
        if not isinstance(self.revision, DualObjectiveRevision):
            raise TypeError("revision must be DualObjectiveRevision")
        if self.dataset_sha256 != dataset_sha256(self.dataset):
            raise ValueError("dataset_sha256 does not match captured dataset")
        if isinstance(self.fit_generation, bool) or not isinstance(self.fit_generation, Integral) or self.fit_generation < 0:
            raise ValueError("fit_generation must be a nonnegative integer")
        if self.selected_model_id is not ModelID.M3_DUAL_HO_DK:
            raise ValueError("final selected model must be M3, not an M2 fallback")

    @classmethod
    def capture(cls, session: AnalysisSession) -> DualErrorMapContext:
        """Capture without evaluating a grid or sharing mutable session state."""

        from pyecr.services.workflow import AnalysisSession

        if not isinstance(session, AnalysisSession):
            raise TypeError("capture requires AnalysisSession")
        dataset, config, fit = (
            session.dataset, session.current_analysis_config, session.fit_result,
        )
        if dataset is None or config is None or fit is None:
            raise RuntimeError("M3 Error Map requires a dataset, AnalysisConfig and FitResult")
        if session.fit_stale or session.fit_revision != session.current_revision:
            raise ValueError("M3 FitResult is Out of date")
        if config.model_id is not ModelID.M3_DUAL_HO_DK:
            raise ValueError("dual Error Map requires M3_DUAL_HO_DK")
        dispatch = fit.diagnostics.get("analysis_dispatch")
        if not isinstance(dispatch, Mapping) or (
            dispatch.get("selected_model_id") != ModelID.M3_DUAL_HO_DK.value
            or dispatch.get("m3_ran") is not True
        ):
            raise ValueError("final selected model must be M3, not an M2 fallback")
        assert session.fit_revision is not None
        return cls(
            dataset=dataset,
            analysis_config=config,
            fit_result=fit,
            dataset_sha256=dataset_sha256(dataset),
            revision=DualObjectiveRevision(
                dataset_revision=session.data_revision,
                preprocessing_revision=session.preprocessing_revision,
                model_revision=session.config_revision,
                fit_revision=session.fit_revision,
            ),
            fit_generation=session.fit_generation,
            selected_model_id=ModelID.M3_DUAL_HO_DK,
        )

    def detached_session(self) -> AnalysisSession:
        """Build a worker-private input for the unchanged P44 core entry point."""

        from pyecr.services.workflow import AnalysisSession

        return AnalysisSession(
            dataset=self.dataset,
            current_analysis_config=self.analysis_config,
            fit_result=self.fit_result,
            fit_stale=False,
            data_revision=self.revision.dataset_revision,
            preprocessing_revision=self.revision.preprocessing_revision,
            config_revision=self.revision.model_revision,
            physical_input_revision=self.revision.fit_revision.physical_input_revision,
            fit_revision=self.revision.fit_revision,
            fit_generation=self.fit_generation,
        )


@dataclass(frozen=True, slots=True)
class DualErrorMapComputation:
    context: DualErrorMapContext
    config: DualErrorMapConfig
    result: DualErrorMapResult
    cache_hit: bool
    status: str = "completed"


class DualErrorMapService:
    """Cache only complete M3 pairs; publishing remains with WorkflowController."""

    def __init__(self, *, max_entries: int = 8) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = int(max_entries)
        self._cache: OrderedDict[str, DualErrorMapResult] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def default_config() -> DualErrorMapConfig:
        """Interactive runs begin at Quick; P44's scientific default is untouched."""

        return DualErrorMapConfig(resolution=ErrorMapResolution.QUICK)

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def parameter_policy(context: DualErrorMapContext):
        """Read the fitter's physical bounds/labels without a forward call.

        from_fit_result also checks SSE by evaluating the fitted point. That
        belongs to the existing compute path, never to view-only controls.
        """
        if not isinstance(context,DualErrorMapContext):
            raise TypeError('parameter_policy requires DualErrorMapContext')
        from pyecr.services.analysis import _dual_fit_adapter
        from pyecr.fitting.dual_ho import _Parameterization
        config,_=_dual_fit_adapter(context.analysis_config)
        parameters=_Parameterization(config,context.dataset.processed_signal,is_m3=True)
        return parameters.physical_bounds,config.label_constraint.value

    def validate_request(self, context: DualErrorMapContext, config: DualErrorMapConfig, *, validate_bounds: bool = True) -> None:
        if not isinstance(context, DualErrorMapContext) or not isinstance(config, DualErrorMapConfig):
            raise TypeError("M3 Error Map requires DualErrorMapContext and DualErrorMapConfig")
        if config.mode is ErrorMapMode.PROFILE_NUISANCE:
            raise NotImplementedError("PROFILE_NUISANCE is not implemented for M3")
        if config.mode is not ErrorMapMode.FIXED:
            raise ValueError("M3 Error Map currently supports FIXED nuisance only")
        if not validate_bounds:
            return  # A cache digest can describe a prospective revision without executing it.
        # Reuse the core's exact model/label range policy before queueing, never
        # evaluate an objective or silently clip a MANUAL request on the GUI.
        from pyecr.diagnostics.dual_error_map import _prepare_ranges
        bounds,label=self.parameter_policy(context)
        for process in ('H','O'):
            _prepare_ranges(process,context.fit_result.params,bounds,label,config)

    def cache_key(self, context: DualErrorMapContext, config: DualErrorMapConfig) -> str:
        self.validate_request(context, config, validate_bounds=False)
        fit_json = _canonical_json(context.fit_result.to_dict())
        payload = {
            "schema_version": 1,
            "dataset_sha256": context.dataset_sha256,
            "revision": asdict(context.revision),
            "fit_generation": context.fit_generation,
            "selected_model_id": context.selected_model_id.value,
            "fit_result_sha256": hashlib.sha256(fit_json).hexdigest(),
            "process_identity": {
                "H": ("D_H_cm2_s", "k_H_cm_s"),
                "O": ("D_O_cm2_s", "k_O_cm_s"),
            },
            "objective_kind": DUAL_FIXED_OBJECTIVE_KIND,
            "analysis_config": context.analysis_config.to_dict(),
            "map_config": asdict(config),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def compute(
        self,
        context: DualErrorMapContext,
        config: DualErrorMapConfig | None = None,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        phase_callback: Callable[[DualMapProgress], None] | None = None,
    ) -> DualErrorMapComputation:
        settings = self.default_config() if config is None else config
        self.validate_request(context, settings)
        _check_cancel(cancel_token)
        key = self.cache_key(context, settings)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is not None:
            _check_cancel(cancel_token)
            _emit(DualMapProgress(1.0, 1.0, 1.0, "cache", "M3 H/O maps reused validated cache"),
                  progress_callback, phase_callback)
            _check_cancel(cancel_token)
            return DualErrorMapComputation(context, settings, cached, True)

        _emit(DualMapProgress(0.0, 0.0, 0.0, "queued", "Evaluating M3 H/O full-signal grids"),
              progress_callback, phase_callback)
        _check_cancel(cancel_token)

        def forward_progress(fraction: float, message: str) -> None:
            _check_cancel(cancel_token)
            h_fraction = min(1.0, 2.0 * fraction)
            o_fraction = max(0.0, 2.0 * fraction - 1.0)
            _emit(DualMapProgress(h_fraction, o_fraction, fraction,
                                  "H" if fraction <= 0.5 else "O", message),
                  progress_callback, phase_callback)
            _check_cancel(cancel_token)

        result = compute_dual_error_map(
            context.detached_session(), settings,
            cancel_check=(None if cancel_token is None else cancel_token.raise_if_cancelled),
            progress_callback=forward_progress,
        )
        _check_cancel(cancel_token)
        if not (
            result.quality.fitted_point_parity_ok
            and isclose(result.h_map.fitted_point.raw_sse, context.fit_result.SSE,
                        rel_tol=1e-8, abs_tol=1e-12)
            and isclose(result.o_map.fitted_point.raw_sse, context.fit_result.SSE,
                        rel_tol=1e-8, abs_tol=1e-12)
        ):
            raise ValueError("M3 H/O fitted-point raw SSE does not match FitResult.SSE")
        if result.revision != context.revision:
            raise ValueError("M3 Error Map result revision differs from captured request")
        _emit(DualMapProgress(1.0, 1.0, 1.0, "complete",
                              "M3 H/O maps complete; fit-point parity verified"),
              progress_callback, phase_callback)
        _check_cancel(cancel_token)
        with self._lock:
            _check_cancel(cancel_token)
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return DualErrorMapComputation(context, settings, result, False)

    @staticmethod
    def current_result(session: AnalysisSession) -> DualErrorMapResult | None:
        """Only the fully published result for the still-current fit is visible."""

        from pyecr.services.workflow import AnalysisSession

        if not isinstance(session, AnalysisSession):
            raise TypeError("current_result requires AnalysisSession")
        context = session.dual_error_map_context
        if (
            session.dual_error_map_result is None
            or session.dual_error_map_stale
            or session.fit_stale
            or context is None
            or context.dataset is not session.dataset
            or context.analysis_config is not session.current_analysis_config
            or context.fit_result is not session.fit_result
            or context.fit_generation != session.fit_generation
            or session.fit_revision != session.current_revision
            or context.revision != DualObjectiveRevision(
                session.data_revision, session.preprocessing_revision,
                session.config_revision, session.fit_revision,
            )
        ):
            return None
        return session.dual_error_map_result

    @classmethod
    def status(cls, session: AnalysisSession) -> str:
        """Session-scoped status; never confuse stale data with a worker crash."""

        if session.dual_error_map_stale or (
            session.dual_error_map_result is not None and cls.current_result(session) is None
        ):
            return "out_of_date"
        if session.dual_error_map_running:
            return "updating" if cls.current_result(session) is not None else "running"
        return session.dual_error_map_last_status


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _check_cancel(token: CancellationToken | None) -> None:
    if token is not None:
        token.raise_if_cancelled()


def _emit(
    progress: DualMapProgress,
    progress_callback: Callable[[float, str], None] | None,
    phase_callback: Callable[[DualMapProgress], None] | None,
) -> None:
    if phase_callback is not None:
        phase_callback(progress)
    if progress_callback is not None:
        progress_callback(progress.combined_fraction, progress.message)
