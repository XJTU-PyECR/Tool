"""Fixed-nuisance M1 log(D)-log(k) raw-SSE surface.

Every cell goes through the same public SingleObjectiveEvaluator that drives
fit_single_ecr. This module does not contain an ECR forward equation or a
second residual/SSE implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import hypot, isclose, isfinite, log10
from numbers import Integral, Real
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.fitting.single import prepare_single_objective
from pyecr.io.dataset import EcrDataset
from pyecr.models.ids import ModelID

__all__ = [
    "ErrorMapCellFailure", "ErrorMapConfig", "ErrorMapDiagnostics", "ErrorMapMode",
    "ErrorMapPoint", "ErrorMapQuality", "ErrorMapResolution", "ErrorMapResult",
    "assess_error_map_quality", "compute_error_map",
]


class ErrorMapMode(str, Enum):
    FIXED = "fixed"
    PROFILE_NUISANCE = "profile_nuisance"


class ErrorMapResolution(int, Enum):
    QUICK = 41
    STANDARD = 81
    FINE = 151


@dataclass(frozen=True, slots=True)
class ErrorMapConfig:
    """Map bounds are log10 coordinates; omitted ranges use clipped AUTO range."""

    mode: ErrorMapMode = ErrorMapMode.FIXED
    resolution: ErrorMapResolution = ErrorMapResolution.STANDARD
    n_points: int | None = None
    log10_D_range: tuple[float, float] | None = None
    log10_k_range: tuple[float, float] | None = None
    auto_half_width_decades: float = 1.5
    narrow_range_decades: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ErrorMapMode):
            raise ValueError("mode must be ErrorMapMode.FIXED or PROFILE_NUISANCE")
        if not isinstance(self.resolution, ErrorMapResolution):
            raise ValueError("resolution must be Quick, Standard, or Fine")
        if self.n_points is not None and (
            isinstance(self.n_points, bool) or not isinstance(self.n_points, Integral)
            or self.n_points < 3
        ):
            raise ValueError("n_points must be an integer >= 3")
        for name in ("log10_D_range", "log10_k_range"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(f"{name} must be a two-point range")
            lo, hi = value
            if not all(isinstance(item, Real) and not isinstance(item, bool) and isfinite(float(item)) for item in (lo, hi)):
                raise ValueError(f"{name} must contain finite real values")
            if lo >= hi:
                raise ValueError(f"{name} must be strictly increasing")
            object.__setattr__(self, name, (float(lo), float(hi)))
        for name in ("auto_half_width_decades", "narrow_range_decades"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @property
    def grid_size(self) -> int:
        return int(self.resolution.value if self.n_points is None else self.n_points)


@dataclass(frozen=True, slots=True)
class ErrorMapPoint:
    log10_D: float
    log10_k: float
    objective: float
    d_index: int | None = None
    k_index: int | None = None


@dataclass(frozen=True, slots=True)
class ErrorMapCellFailure:
    d_index: int
    k_index: int
    log10_D: float
    log10_k: float
    reason: str


@dataclass(frozen=True, slots=True)
class ErrorMapDiagnostics:
    objective_kind: str
    nuisance_policy: str
    range_clipped: bool
    n_valid: int
    n_total: int
    invalid_cells: tuple[ErrorMapCellFailure, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ErrorMapQuality:
    """Quality of a *visualized objective grid*, not parameter confidence."""

    optimum_in_range: bool
    grid_min_on_edge: bool
    distance_to_fit_log10_decades: float | None
    distance_to_fit_in_cells: float | None
    invalid_fraction: float
    objective_parity_ok: bool
    range_warning: str | None
    resolution_warning: str | None
    status: str
    warnings: tuple[str, ...]
    interpretation: str = "Objective visualization only; not a confidence interval."

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warning", "fail"}:
            raise ValueError("Error Map quality status must be pass, warning, or fail")
        if not isfinite(self.invalid_fraction) or not 0.0 <= self.invalid_fraction <= 1.0:
            raise ValueError("invalid_fraction must lie in [0, 1]")
        for name in ("distance_to_fit_log10_decades", "distance_to_fit_in_cells"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative or None")


@dataclass(frozen=True, slots=True)
class ErrorMapResult:
    log10_D: NDArray[np.float64]
    log10_k: NDArray[np.float64]
    objective_grid: NDArray[np.float64]
    relative_grid: NDArray[np.float64]
    log_ratio_grid: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    fitted_point: ErrorMapPoint
    grid_minimum: ErrorMapPoint | None
    diagnostics: ErrorMapDiagnostics
    quality: ErrorMapQuality | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the evaluated grid, never an instruction to recompute it."""

        return {
            "schema_version": 1,
            "log10_D": self.log10_D.tolist(),
            "log10_k": self.log10_k.tolist(),
            "objective_grid": _json_grid(self.objective_grid),
            "relative_grid": _json_grid(self.relative_grid),
            "log_ratio_grid": _json_grid(self.log_ratio_grid),
            "valid_mask": self.valid_mask.tolist(),
            "fitted_point": asdict(self.fitted_point),
            "grid_minimum": None if self.grid_minimum is None else asdict(self.grid_minimum),
            "diagnostics": asdict(self.diagnostics),
            "quality": None if self.quality is None else asdict(self.quality),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ErrorMapResult:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported ErrorMapResult schema_version")
        d_axis = np.asarray(value["log10_D"], dtype=np.float64)
        k_axis = np.asarray(value["log10_k"], dtype=np.float64)
        if d_axis.ndim != 1 or k_axis.ndim != 1 or min(d_axis.size, k_axis.size) < 3:
            raise ValueError("Error Map axes must be one-dimensional with at least 3 points")
        if not np.all(np.isfinite(d_axis)) or not np.all(np.isfinite(k_axis)) or not np.all(np.diff(d_axis) > 0) or not np.all(np.diff(k_axis) > 0):
            raise ValueError("Error Map axes must be finite and strictly increasing")
        shape = (d_axis.size, k_axis.size)
        grids = [np.asarray(value[name], dtype=np.float64) for name in
                 ("objective_grid", "relative_grid", "log_ratio_grid")]
        valid = np.asarray(value["valid_mask"], dtype=np.bool_)
        if any(grid.shape != shape for grid in (*grids, valid)):
            raise ValueError("Error Map grid shapes do not match axes")
        if not np.all(np.isfinite(grids[0][valid])) or np.any(grids[0][valid] < 0):
            raise ValueError("valid Error Map objectives must be finite and nonnegative")
        point = value["fitted_point"]
        minimum = value["grid_minimum"]
        diagnostic = value["diagnostics"]
        if not isinstance(point, dict) or not isinstance(diagnostic, dict) or not (minimum is None or isinstance(minimum, dict)):
            raise TypeError("Error Map point and diagnostics must be objects")
        failures = tuple(ErrorMapCellFailure(**item) for item in diagnostic["invalid_cells"])
        quality_payload = value.get("quality")
        if quality_payload is not None and not isinstance(quality_payload, dict):
            raise TypeError("Error Map quality must be an object")
        quality = None if quality_payload is None else ErrorMapQuality(
            optimum_in_range=bool(quality_payload["optimum_in_range"]),
            grid_min_on_edge=bool(quality_payload["grid_min_on_edge"]),
            distance_to_fit_log10_decades=(None if quality_payload["distance_to_fit_log10_decades"] is None else float(quality_payload["distance_to_fit_log10_decades"])),
            distance_to_fit_in_cells=(None if quality_payload["distance_to_fit_in_cells"] is None else float(quality_payload["distance_to_fit_in_cells"])),
            invalid_fraction=float(quality_payload["invalid_fraction"]),
            objective_parity_ok=bool(quality_payload["objective_parity_ok"]),
            range_warning=(None if quality_payload["range_warning"] is None else str(quality_payload["range_warning"])),
            resolution_warning=(None if quality_payload["resolution_warning"] is None else str(quality_payload["resolution_warning"])),
            status=str(quality_payload["status"]),
            warnings=tuple(str(item) for item in quality_payload["warnings"]),
            interpretation=str(quality_payload.get("interpretation", "Objective visualization only; not a confidence interval.")),
        )
        result = cls(
            log10_D=d_axis, log10_k=k_axis, objective_grid=grids[0],
            relative_grid=grids[1], log_ratio_grid=grids[2], valid_mask=valid,
            fitted_point=ErrorMapPoint(**point),
            grid_minimum=None if minimum is None else ErrorMapPoint(**minimum),
            diagnostics=ErrorMapDiagnostics(
                objective_kind=str(diagnostic["objective_kind"]),
                nuisance_policy=str(diagnostic["nuisance_policy"]),
                range_clipped=bool(diagnostic["range_clipped"]),
                n_valid=int(diagnostic["n_valid"]), n_total=int(diagnostic["n_total"]),
                invalid_cells=failures,
                warnings=tuple(str(item) for item in diagnostic["warnings"]),
            ),
            quality=quality,
        )
        if result.diagnostics.n_total != valid.size or result.diagnostics.n_valid != int(valid.sum()):
            raise ValueError("Error Map diagnostic counts disagree with valid_mask")
        for array in (d_axis, k_axis, *grids, valid):
            array.flags.writeable = False
        return result


def _json_grid(grid: NDArray[np.float64]) -> list[list[float | str]]:
    def scalar(value: float) -> float | str:
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return float(value)

    return [[scalar(float(item)) for item in row] for row in grid]


def assess_error_map_quality(
    result: ErrorMapResult, config: ErrorMapConfig, fit_sse: float,
) -> ErrorMapQuality:
    """Assess stored cells and fit-point parity without evaluating any model."""

    if not isinstance(result, ErrorMapResult) or not isinstance(config, ErrorMapConfig):
        raise TypeError("quality assessment requires ErrorMapResult and ErrorMapConfig")
    if not isfinite(fit_sse) or fit_sse < 0:
        raise ValueError("fit_sse must be finite and nonnegative")
    d_axis, k_axis = result.log10_D, result.log10_k
    fitted = result.fitted_point
    d_step = float((d_axis[-1] - d_axis[0]) / (d_axis.size - 1))
    k_step = float((k_axis[-1] - k_axis[0]) / (k_axis.size - 1))
    optimum_in_range = bool(
        d_axis[0] <= fitted.log10_D <= d_axis[-1]
        and k_axis[0] <= fitted.log10_k <= k_axis[-1]
    )
    minimum = result.grid_minimum
    on_edge = bool(minimum is not None and (
        minimum.d_index in (0, d_axis.size - 1)
        or minimum.k_index in (0, k_axis.size - 1)
    ))
    log_distance = None if minimum is None else hypot(
        minimum.log10_D - fitted.log10_D,
        minimum.log10_k - fitted.log10_k,
    )
    cell_distance = None if minimum is None else hypot(
        (minimum.log10_D - fitted.log10_D) / d_step,
        (minimum.log10_k - fitted.log10_k) / k_step,
    )
    invalid_fraction = 1.0 - result.diagnostics.n_valid / result.diagnostics.n_total
    parity = isclose(fitted.objective, fit_sse, rel_tol=1.0e-8, abs_tol=1.0e-12)
    range_codes: list[str] = []
    if not optimum_in_range:
        range_codes.append("OPTIMUM_OUTSIDE_RANGE")
    elif min(
        (fitted.log10_D - d_axis[0]) / d_step,
        (d_axis[-1] - fitted.log10_D) / d_step,
        (fitted.log10_k - k_axis[0]) / k_step,
        (k_axis[-1] - fitted.log10_k) / k_step,
    ) < 0.5:
        range_codes.append("OPTIMUM_NEAR_EDGE")
    if on_edge:
        range_codes.append("GRID_MIN_ON_EDGE")
    if result.diagnostics.range_clipped:
        range_codes.append("AUTO_RANGE_CLIPPED")
    if min(d_axis[-1] - d_axis[0], k_axis[-1] - k_axis[0]) < config.narrow_range_decades:
        range_codes.append("RANGE_TOO_NARROW")
    resolution_codes: list[str] = []
    if max(d_step, k_step) > 0.05:
        resolution_codes.append("COARSE_GRID_STEP")
    if config.grid_size <= ErrorMapResolution.QUICK.value:
        resolution_codes.append("QUICK_IS_PRELIMINARY")
    warnings = [*range_codes, *resolution_codes]
    if not parity:
        warnings.append("OBJECTIVE_PARITY_MISMATCH")
    if minimum is None:
        warnings.append("NO_VALID_CELLS")
    if invalid_fraction > 0:
        warnings.append(f"INVALID_CELLS: {invalid_fraction:.1%} of grid")
    if cell_distance is not None and cell_distance > 1.5:
        warnings.append("GRID_FIT_SEPARATION: inspect flat/multimodal objective valley")
    status = "fail" if not parity or minimum is None else "warning" if warnings else "pass"
    return ErrorMapQuality(
        optimum_in_range=optimum_in_range,
        grid_min_on_edge=on_edge,
        distance_to_fit_log10_decades=log_distance,
        distance_to_fit_in_cells=cell_distance,
        invalid_fraction=invalid_fraction,
        objective_parity_ok=parity,
        range_warning="; ".join(range_codes) or None,
        resolution_warning="; ".join(resolution_codes) or None,
        status=status,
        warnings=tuple(warnings),
    )


def compute_error_map(
    dataset: EcrDataset,
    config: AnalysisConfig,
    fit_result: FitResult,
    map_config: ErrorMapConfig | None = None,
    *,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> ErrorMapResult:
    """Evaluate the exact fitter-reported SSE on a fixed-nuisance M1 grid.

    PROFILE_NUISANCE is reserved and always fails explicitly in this release.
    Invalid forward calls are represented by NaN *and* a per-cell reason.
    """

    settings = ErrorMapConfig() if map_config is None else map_config
    if not isinstance(settings, ErrorMapConfig):
        raise TypeError("map_config must be ErrorMapConfig")
    if settings.mode is ErrorMapMode.PROFILE_NUISANCE:
        raise NotImplementedError("PROFILE_NUISANCE is not implemented; use FIXED")
    if not isinstance(config, AnalysisConfig) or config.model_id is not ModelID.M1_SINGLE_DK:
        raise ValueError("Error Map currently supports only M1_SINGLE_DK")
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be FitResult")
    evaluator = prepare_single_objective(dataset, config)
    names = evaluator.context.parameter_names
    if names[:2] != ("log10_D", "log10_k"):
        raise RuntimeError("M1 objective layout must start with log10_D, log10_k")
    if fit_result.y_fit.size != evaluator.context.observed.size:
        raise ValueError("FitResult and dataset observation counts differ")
    if "D_cm2_s" not in fit_result.params or "k_cm_s" not in fit_result.params:
        raise ValueError("FitResult lacks fitted M1 D/k parameters")
    fit_coordinates = np.empty(len(names), dtype=np.float64)
    for index, name in enumerate(names):
        if name == "log10_D":
            fit_coordinates[index] = log10(fit_result.params["D_cm2_s"])
        elif name == "log10_k":
            fit_coordinates[index] = log10(fit_result.params["k_cm_s"])
        else:
            if name not in fit_result.params:
                raise ValueError(f"FitResult lacks fitted nuisance parameter {name}")
            fit_coordinates[index] = fit_result.params[name]
    lower = evaluator.context.lower_bounds
    upper = evaluator.context.upper_bounds
    if np.any(fit_coordinates < lower - 1.0e-10) or np.any(fit_coordinates > upper + 1.0e-10):
        raise ValueError("FitResult parameters lie outside validated bounds")
    if cancel_check is not None:
        cancel_check()
    fitted_sse = evaluator.evaluate(fit_coordinates).sse
    fitted_point = ErrorMapPoint(float(fit_coordinates[0]), float(fit_coordinates[1]), fitted_sse)
    d_range, d_clipped = _axis_range(settings.log10_D_range, fit_coordinates[0], lower[0], upper[0], settings.auto_half_width_decades)
    k_range, k_clipped = _axis_range(settings.log10_k_range, fit_coordinates[1], lower[1], upper[1], settings.auto_half_width_decades)
    d_axis = np.linspace(*d_range, settings.grid_size)
    k_axis = np.linspace(*k_range, settings.grid_size)
    objectives = np.full((d_axis.size, k_axis.size), np.nan, dtype=np.float64)
    valid = np.zeros(objectives.shape, dtype=np.bool_)
    failures: list[ErrorMapCellFailure] = []
    coordinates = fit_coordinates.copy()
    for d_index, d_value in enumerate(d_axis):
        coordinates[0] = d_value
        for k_index, k_value in enumerate(k_axis):
            if cancel_check is not None:
                cancel_check()
            coordinates[1] = k_value
            try:
                sse = evaluator.evaluate(coordinates).sse
                if not isfinite(sse) or sse < 0.0:
                    raise FloatingPointError("objective SSE is non-finite or negative")
            except Exception as exc:
                failures.append(ErrorMapCellFailure(
                    d_index, k_index, float(d_value), float(k_value),
                    f"{type(exc).__name__}: {exc}",
                ))
            else:
                objectives[d_index, k_index] = sse
                valid[d_index, k_index] = True
        if progress_callback is not None:
            progress_callback((d_index + 1) / d_axis.size, f"Error Map row {d_index + 1}/{d_axis.size}")

    warnings: list[str] = []
    if failures:
        warnings.append(f"INVALID_CELLS: {len(failures)}/{objectives.size} grid evaluations failed; inspect invalid_cells")
    if d_range[1] - d_range[0] < settings.narrow_range_decades or k_range[1] - k_range[0] < settings.narrow_range_decades:
        warnings.append("RANGE_TOO_NARROW: at least one axis spans less than the configured minimum decades")
    if not np.isclose(fitted_sse, fit_result.SSE, rtol=1.0e-8, atol=1.0e-12):
        warnings.append("GRID_VS_FIT_MISMATCH: current objective at fitted point differs from FitResult.SSE")

    relative = np.full_like(objectives, np.nan)
    log_ratio = np.full_like(objectives, np.nan)
    grid_minimum: ErrorMapPoint | None = None
    if valid.any():
        minimum_index = np.unravel_index(int(np.nanargmin(objectives)), objectives.shape)
        d_index, k_index = map(int, minimum_index)
        minimum_sse = float(objectives[minimum_index])
        grid_minimum = ErrorMapPoint(float(d_axis[d_index]), float(k_axis[k_index]), minimum_sse, d_index, k_index)
        if d_index in {0, d_axis.size - 1} or k_index in {0, k_axis.size - 1}:
            warnings.append("EDGE_MINIMUM: grid minimum is on the evaluated boundary")
        if minimum_sse == 0.0:
            relative[valid] = np.where(objectives[valid] == 0.0, 1.0, np.inf)
            log_ratio[valid] = np.where(objectives[valid] == 0.0, 0.0, np.inf)
            warnings.append("ZERO_MINIMUM_SSE: nonzero/zero ratios are infinite")
        else:
            relative[valid] = objectives[valid] / minimum_sse
            log_ratio[valid] = np.log10(relative[valid])
        scale = max(fitted_sse, np.finfo(np.float64).eps**2 * max(float(evaluator.context.observed @ evaluator.context.observed), 1.0))
        if minimum_sse < fitted_sse - 0.01 * scale or minimum_sse > fitted_sse + 0.25 * scale:
            warnings.append("GRID_VS_FIT_MISMATCH: grid minimum and fitted-point objectives differ materially")
    else:
        warnings.append("NO_VALID_CELLS: no grid forward evaluation succeeded")

    for array in (d_axis, k_axis, objectives, relative, log_ratio, valid):
        array.flags.writeable = False
    diagnostics = ErrorMapDiagnostics(
        objective_kind="unweighted_raw_sse",
        nuisance_policy="fixed_at_fit",
        range_clipped=d_clipped or k_clipped,
        n_valid=int(valid.sum()),
        n_total=int(valid.size),
        invalid_cells=tuple(failures),
        warnings=tuple(warnings),
    )
    result = ErrorMapResult(
        log10_D=d_axis, log10_k=k_axis, objective_grid=objectives,
        relative_grid=relative, log_ratio_grid=log_ratio, valid_mask=valid,
        fitted_point=fitted_point, grid_minimum=grid_minimum, diagnostics=diagnostics,
    )
    quality = assess_error_map_quality(result, settings, fit_result.SSE)
    return replace(
        result,
        quality=quality,
        diagnostics=replace(
            diagnostics,
            warnings=tuple(dict.fromkeys((*diagnostics.warnings, *quality.warnings))),
        ),
    )


def _axis_range(
    explicit: tuple[float, float] | None,
    center: float,
    lower: float,
    upper: float,
    half_width: float,
) -> tuple[tuple[float, float], bool]:
    if explicit is not None:
        lo, hi = explicit
        if lo < lower - 1.0e-10 or hi > upper + 1.0e-10:
            raise ValueError("explicit Error Map range must stay within validated bounds")
        return (lo, hi), False
    requested_lo, requested_hi = center - half_width, center + half_width
    lo, hi = max(lower, requested_lo), min(upper, requested_hi)
    if lo >= hi:
        raise ValueError("AUTO Error Map range has no width within validated bounds")
    return (float(lo), float(hi)), bool(lo != requested_lo or hi != requested_hi)
