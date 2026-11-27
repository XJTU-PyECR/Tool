"""Worker-safe M1 Error Map orchestration, parity gate, and bounded cache."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from math import isclose, isfinite, log10
from numbers import Integral
from threading import Lock
from typing import Callable

from pyecr.config import AnalysisConfig
from pyecr.diagnostics.error_map import (
    ErrorMapConfig, ErrorMapMode, ErrorMapResult, compute_error_map,
)
from pyecr.fitting.base import FitResult
from pyecr.fitting.single import prepare_single_objective
from pyecr.io.dataset import EcrDataset
from pyecr.report.snapshot import dataset_sha256
from pyecr.services.analysis import CancellationToken

__all__ = [
    "ErrorMapComputation", "ErrorMapContext", "ErrorMapService",
    "ObjectiveParityError",
]


class ObjectiveParityError(ValueError):
    """The fitted-point production objective differs from FitResult.SSE."""


@dataclass(frozen=True, slots=True)
class ErrorMapContext:
    """Frozen physical inputs and revisions captured before worker dispatch."""

    dataset: EcrDataset
    analysis_config: AnalysisConfig
    fit_result: FitResult
    dataset_sha256: str
    revision: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, EcrDataset):
            raise TypeError("dataset must be EcrDataset")
        if not isinstance(self.analysis_config, AnalysisConfig):
            raise TypeError("analysis_config must be AnalysisConfig")
        self.analysis_config.validate()
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be FitResult")
        if self.dataset_sha256 != dataset_sha256(self.dataset):
            raise ValueError("dataset_sha256 does not match immutable dataset")
        if (
            not isinstance(self.revision, tuple)
            or len(self.revision) != 3
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in self.revision)
        ):
            raise ValueError("revision must contain three nonnegative integers")

    @classmethod
    def capture(
        cls,
        dataset: EcrDataset,
        analysis_config: AnalysisConfig,
        fit_result: FitResult,
        *,
        revision: tuple[int, int, int],
    ) -> ErrorMapContext:
        return cls(dataset, analysis_config, fit_result, dataset_sha256(dataset), revision)


@dataclass(frozen=True, slots=True)
class ErrorMapComputation:
    context: ErrorMapContext
    config: ErrorMapConfig
    result: ErrorMapResult
    cache_hit: bool


class ErrorMapService:
    """Compute in a caller-provided worker; cache only complete parity-checked maps."""

    def __init__(self, *, max_entries: int = 8) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = int(max_entries)
        self._cache: OrderedDict[str, ErrorMapResult] = OrderedDict()
        self._lock = Lock()

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def kinetic_log_bounds(
        self, context: ErrorMapContext,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the production fitter's log10 D/k bounds for GUI range tools."""

        if not isinstance(context, ErrorMapContext):
            raise TypeError("context must be ErrorMapContext")
        evaluator = prepare_single_objective(context.dataset, context.analysis_config)
        if evaluator.context.parameter_names[:2] != ("log10_D", "log10_k"):
            raise ValueError("Error Map requires an M1 D-k objective")
        lower, upper = evaluator.context.lower_bounds, evaluator.context.upper_bounds
        return ((float(lower[0]), float(upper[0])),
                (float(lower[1]), float(upper[1])))

    def validate_request(self, context: ErrorMapContext, config: ErrorMapConfig) -> None:
        """Reject unsupported or out-of-bounds controls before worker submission.

        This constructs the production objective context to reuse its exact
        optimizer bounds. It does not call predict(), evaluate(), or compute a
        grid on the GUI thread.
        """

        if not isinstance(context, ErrorMapContext) or not isinstance(config, ErrorMapConfig):
            raise TypeError("validate_request requires ErrorMapContext and ErrorMapConfig")
        if config.mode is not ErrorMapMode.FIXED:
            raise NotImplementedError("PROFILE_NUISANCE is not implemented; use FIXED")
        bounds = self.kinetic_log_bounds(context)
        for index, (name, requested) in enumerate((
            ("D_cm2_s", config.log10_D_range),
            ("k_cm_s", config.log10_k_range),
        )):
            fitted = context.fit_result.params.get(name)
            if fitted is None or not isfinite(float(fitted)) or fitted <= 0.0:
                raise ValueError(f"FitResult lacks a positive finite {name}")
            center = log10(float(fitted))
            lower, upper = bounds[index]
            if center < lower - 1.0e-10 or center > upper + 1.0e-10:
                raise ValueError(f"fitted {name} lies outside validated bounds")
            if requested is not None and (
                requested[0] < lower - 1.0e-10 or requested[1] > upper + 1.0e-10
            ):
                raise ValueError(
                    f"MANUAL {name} Error Map range must stay within validated bounds "
                    f"[{lower:.8g}, {upper:.8g}] in log10 space"
                )

    def cache_key(self, context: ErrorMapContext, config: ErrorMapConfig) -> str:
        if not isinstance(context, ErrorMapContext) or not isinstance(config, ErrorMapConfig):
            raise TypeError("cache_key requires ErrorMapContext and ErrorMapConfig")
        payload = {
            "dataset_sha256": context.dataset_sha256,
            "revision": context.revision,
            "analysis_config": context.analysis_config.to_dict(),
            "fit_result": context.fit_result.to_dict(),
            "error_map_config": asdict(config),
        }
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, allow_nan=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def compute(
        self,
        context: ErrorMapContext,
        config: ErrorMapConfig,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ErrorMapComputation:
        if not isinstance(context, ErrorMapContext) or not isinstance(config, ErrorMapConfig):
            raise TypeError("compute requires ErrorMapContext and ErrorMapConfig")
        _check_cancel(cancel_token)
        key = self.cache_key(context, config)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is not None:
            _check_cancel(cancel_token)
            if progress_callback is not None:
                progress_callback(1.0, "Error Map reused validated cache entry")
            return ErrorMapComputation(context, config, cached, True)

        if progress_callback is not None:
            progress_callback(0.0, "Evaluating M1 D-k objective grid")

        def forward_progress(fraction: float, message: str) -> None:
            _check_cancel(cancel_token)
            if progress_callback is not None:
                progress_callback(0.95 * fraction, message)

        result = compute_error_map(
            context.dataset,
            context.analysis_config,
            context.fit_result,
            config,
            cancel_check=(None if cancel_token is None else cancel_token.raise_if_cancelled),
            progress_callback=forward_progress,
        )
        _check_cancel(cancel_token)
        if result.diagnostics.n_valid == 0:
            first_reason = (
                result.diagnostics.invalid_cells[0].reason
                if result.diagnostics.invalid_cells else "no grid cell succeeded"
            )
            raise RuntimeError(
                f"Error Map has no valid cells; first forward exception: {first_reason}"
            )
        fitted_sse = result.fitted_point.objective
        reported_sse = context.fit_result.SSE
        if not isclose(fitted_sse, reported_sse, rel_tol=1.0e-8, abs_tol=1.0e-12):
            raise ObjectiveParityError(
                "Error Map fit-point objective does not match FitResult.SSE: "
                f"evaluated={fitted_sse:.17g}, FitResult.SSE={reported_sse:.17g}"
            )
        warnings = list(result.diagnostics.warnings)
        if result.diagnostics.range_clipped:
            warnings.append(
                "RANGE_CLIPPED: AUTO range was limited by validated D/k bounds"
            )
        if config.grid_size <= 41:
            warnings.append(
                "RESOLUTION_PRELIMINARY: use Standard/Fine to verify narrow minima"
            )
        if tuple(warnings) != result.diagnostics.warnings:
            result = replace(
                result,
                diagnostics=replace(result.diagnostics, warnings=tuple(warnings)),
            )
        _check_cancel(cancel_token)
        with self._lock:
            _check_cancel(cancel_token)
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        if progress_callback is not None:
            progress_callback(1.0, "Error Map complete; objective parity verified")
        return ErrorMapComputation(context, config, result, False)


def _check_cancel(token: CancellationToken | None) -> None:
    if token is not None:
        token.raise_if_cancelled()
