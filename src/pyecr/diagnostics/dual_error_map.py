"""M3 H/O fixed-nuisance objective maps over the complete observed signal.

Every cell delegates to the P43 DualHOObjectiveEvaluator. This module has no
H/O forward equation, component-level objective, fitting, or GUI dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isclose, isfinite, log10
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyecr.diagnostics.error_map import ErrorMapMode, ErrorMapResolution
from pyecr.fitting.dual_objective import DualHOObjectiveEvaluator, DualObjectiveRevision

if TYPE_CHECKING:
    from pyecr.services.workflow import AnalysisSession

__all__ = [
    "DUAL_FIXED_OBJECTIVE_KIND",
    "DualErrorMapCellFailure", "DualErrorMapConfig", "DualErrorMapPoint",
    "DualErrorMapQuality", "DualErrorMapResult", "DualErrorMapSlice",
    "DualMapRangeMode", "DualSliceQuality", "compute_dual_error_map",
]

DUAL_FIXED_OBJECTIVE_KIND = "unweighted_raw_sse_full_signal_fixed_nuisance"
_OBJECTIVE_KIND = DUAL_FIXED_OBJECTIVE_KIND
_PARITY_RTOL = 1.0e-8
_PARITY_ATOL = 1.0e-12


class DualMapRangeMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class DualErrorMapConfig:
    """Separate H/O log10 ranges; omitted ranges are fitted-pair-centred AUTO."""

    mode: ErrorMapMode = ErrorMapMode.FIXED
    range_mode: DualMapRangeMode = DualMapRangeMode.AUTO
    resolution: ErrorMapResolution = ErrorMapResolution.STANDARD
    n_points: int | None = None
    h_log10_D_range: tuple[float, float] | None = None
    h_log10_k_range: tuple[float, float] | None = None
    o_log10_D_range: tuple[float, float] | None = None
    o_log10_k_range: tuple[float, float] | None = None
    auto_half_width_decades: float = 1.5
    narrow_range_decades: float = 0.5
    near_label_decades: float = 0.1
    weak_amplitude_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ErrorMapMode):
            raise ValueError("mode must be ErrorMapMode")
        if not isinstance(self.range_mode, DualMapRangeMode):
            raise ValueError("range_mode must be AUTO or MANUAL")
        if not isinstance(self.resolution, ErrorMapResolution):
            raise ValueError("resolution must be Quick, Standard, or Fine")
        if self.n_points is not None and (
            isinstance(self.n_points, bool) or not isinstance(self.n_points, Integral)
            or self.n_points < 3
        ):
            raise ValueError("n_points must be an integer >= 3")
        names = (
            "h_log10_D_range", "h_log10_k_range",
            "o_log10_D_range", "o_log10_k_range",
        )
        for name in names:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(f"{name} must be a two-point range")
            lo, hi = value
            if not all(isinstance(item, Real) and not isinstance(item, bool)
                       and isfinite(float(item)) for item in (lo, hi)) or lo >= hi:
                raise ValueError(f"{name} range must be finite and increasing")
            object.__setattr__(self, name, (float(lo), float(hi)))
        selected = [getattr(self, name) is not None for name in names]
        if self.range_mode is DualMapRangeMode.AUTO and any(selected):
            raise ValueError("AUTO range must not provide MANUAL endpoints")
        if self.range_mode is DualMapRangeMode.MANUAL and not all(selected):
            raise ValueError("MANUAL range requires all four H/O axis endpoints")
        for name in ("auto_half_width_decades", "narrow_range_decades",
                     "near_label_decades", "weak_amplitude_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.weak_amplitude_fraction >= 1:
            raise ValueError("weak_amplitude_fraction must be below one")

    @property
    def grid_size(self) -> int:
        return int(self.resolution.value if self.n_points is None else self.n_points)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly capture of numerical range and resolution choices."""

        return {
            **asdict(self),
            "mode": self.mode.value,
            "range_mode": self.range_mode.value,
            "resolution": self.resolution.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DualErrorMapConfig:
        if not isinstance(value, Mapping):
            raise TypeError("dual Error Map config must be an object")
        return cls(
            mode=ErrorMapMode(value["mode"]),
            range_mode=DualMapRangeMode(value["range_mode"]),
            resolution=ErrorMapResolution(value["resolution"]),
            n_points=value.get("n_points"),
            h_log10_D_range=value.get("h_log10_D_range"),
            h_log10_k_range=value.get("h_log10_k_range"),
            o_log10_D_range=value.get("o_log10_D_range"),
            o_log10_k_range=value.get("o_log10_k_range"),
            auto_half_width_decades=float(value["auto_half_width_decades"]),
            narrow_range_decades=float(value["narrow_range_decades"]),
            near_label_decades=float(value["near_label_decades"]),
            weak_amplitude_fraction=float(value["weak_amplitude_fraction"]),
        )


@dataclass(frozen=True, slots=True)
class DualErrorMapPoint:
    log10_D: float
    log10_k: float
    raw_sse: float
    d_index: int | None = None
    k_index: int | None = None


@dataclass(frozen=True, slots=True)
class DualErrorMapCellFailure:
    d_index: int
    k_index: int
    log10_D: float
    log10_k: float
    reason: str


@dataclass(frozen=True, slots=True)
class DualSliceQuality:
    optimum_in_range: bool
    grid_min_on_edge: bool
    coarse_grid: bool
    invalid_fraction: float
    elongated_valley: bool
    elongation_ratio: float | None
    sse_mismatch: bool
    label_risk: bool
    weak_process_amplitude: bool
    warnings: tuple[str, ...]
    interpretation: str = "Objective visualization only; not a confidence interval."


@dataclass(frozen=True, slots=True)
class DualErrorMapQuality:
    fitted_point_parity_ok: bool
    invalid_fraction_max: float
    label_risk: bool
    weak_process_amplitude: bool
    warnings: tuple[str, ...]
    interpretation: str = "Two fixed-nuisance full-signal SSE surfaces; not confidence intervals."


@dataclass(frozen=True, slots=True)
class DualErrorMapSlice:
    process: str
    log10_D: NDArray[np.float64]
    log10_k: NDArray[np.float64]
    raw_sse_grid: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    fitted_point: DualErrorMapPoint
    grid_minimum: DualErrorMapPoint | None
    fixed_nuisance: Mapping[str, float]
    objective_kind: str
    revision: DualObjectiveRevision
    label_constraint: str
    range_clipped: bool
    invalid_cells: tuple[DualErrorMapCellFailure, ...]
    quality: DualSliceQuality
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.process not in {"H", "O"}:
            raise ValueError("process must be H or O")
        d_axis = np.asarray(self.log10_D, dtype=np.float64).copy()
        k_axis = np.asarray(self.log10_k, dtype=np.float64).copy()
        grid = np.asarray(self.raw_sse_grid, dtype=np.float64).copy()
        mask = np.asarray(self.valid_mask, dtype=np.bool_).copy()
        if (d_axis.ndim != 1 or k_axis.ndim != 1 or min(d_axis.size, k_axis.size) < 3
                or not np.all(np.isfinite(d_axis)) or not np.all(np.isfinite(k_axis))
                or not np.all(np.diff(d_axis) > 0) or not np.all(np.diff(k_axis) > 0)):
            raise ValueError("dual Error Map axes must be finite and increasing")
        if grid.shape != (d_axis.size, k_axis.size) or mask.shape != grid.shape:
            raise ValueError("dual Error Map grid/mask shape mismatch")
        if not np.all(np.isfinite(grid[mask])) or np.any(grid[mask] < 0):
            raise ValueError("valid raw SSE cells must be finite and nonnegative")
        if not np.all(np.isnan(grid[~mask])):
            raise ValueError("invalid raw SSE cells must be NaN and masked")
        if not isinstance(self.revision, DualObjectiveRevision):
            raise TypeError("revision must be DualObjectiveRevision")
        if self.objective_kind != _OBJECTIVE_KIND:
            raise ValueError("unsupported dual Error Map objective kind")
        nuisance = {str(name): float(value) for name, value in self.fixed_nuisance.items()}
        if not all(isfinite(value) for value in nuisance.values()):
            raise ValueError("fixed nuisance values must be finite")
        for array in (d_axis, k_axis, grid, mask):
            array.flags.writeable = False
        object.__setattr__(self, "log10_D", d_axis)
        object.__setattr__(self, "log10_k", k_axis)
        object.__setattr__(self, "raw_sse_grid", grid)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "fixed_nuisance", MappingProxyType(nuisance))
        object.__setattr__(self, "invalid_cells", tuple(self.invalid_cells))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "process": self.process,
            "log10_D": self.log10_D.tolist(), "log10_k": self.log10_k.tolist(),
            "raw_sse_grid": [[float(value) if self.valid_mask[i, j] else None
                              for j, value in enumerate(row)]
                             for i, row in enumerate(self.raw_sse_grid)],
            "valid_mask": self.valid_mask.tolist(),
            "fitted_point": asdict(self.fitted_point),
            "grid_minimum": None if self.grid_minimum is None else asdict(self.grid_minimum),
            "fixed_nuisance": dict(self.fixed_nuisance),
            "objective_kind": self.objective_kind,
            "revision": asdict(self.revision),
            "label_constraint": self.label_constraint,
            "range_clipped": self.range_clipped,
            "invalid_cells": [asdict(cell) for cell in self.invalid_cells],
            "quality": {**asdict(self.quality), "warnings": list(self.quality.warnings)},
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DualErrorMapSlice:
        grid = np.asarray([[np.nan if item is None else float(item) for item in row]
                           for row in value["raw_sse_grid"]], dtype=np.float64)
        quality = value["quality"]
        return cls(
            process=str(value["process"]),
            log10_D=np.asarray(value["log10_D"], dtype=np.float64),
            log10_k=np.asarray(value["log10_k"], dtype=np.float64),
            raw_sse_grid=grid,
            valid_mask=np.asarray(value["valid_mask"], dtype=np.bool_),
            fitted_point=DualErrorMapPoint(**value["fitted_point"]),
            grid_minimum=(None if value["grid_minimum"] is None else
                          DualErrorMapPoint(**value["grid_minimum"])),
            fixed_nuisance=value["fixed_nuisance"],
            objective_kind=str(value["objective_kind"]),
            revision=_revision_from_dict(value["revision"]),
            label_constraint=str(value["label_constraint"]),
            range_clipped=bool(value["range_clipped"]),
            invalid_cells=tuple(DualErrorMapCellFailure(**cell)
                                for cell in value["invalid_cells"]),
            quality=DualSliceQuality(
                **{**quality, "warnings": tuple(quality["warnings"])}),
            warnings=tuple(value["warnings"]),
        )


@dataclass(frozen=True, slots=True)
class DualErrorMapResult:
    h_map: DualErrorMapSlice
    o_map: DualErrorMapSlice
    objective_kind: str
    revision: DualObjectiveRevision
    label_constraint: str
    quality: DualErrorMapQuality
    warnings: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported DualErrorMapResult schema_version")
        if self.h_map.process != "H" or self.o_map.process != "O":
            raise ValueError("DualErrorMapResult requires H and O maps")
        if (self.h_map.revision != self.revision or self.o_map.revision != self.revision
                or self.h_map.objective_kind != self.objective_kind
                or self.o_map.objective_kind != self.objective_kind
                or self.h_map.label_constraint != self.label_constraint
                or self.o_map.label_constraint != self.label_constraint):
            raise ValueError("dual Error Map slice provenance mismatch")
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "h_map": self.h_map.to_dict(), "o_map": self.o_map.to_dict(),
            "objective_kind": self.objective_kind,
            "revision": asdict(self.revision),
            "label_constraint": self.label_constraint,
            "quality": {**asdict(self.quality), "warnings": list(self.quality.warnings)},
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DualErrorMapResult:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported DualErrorMapResult schema_version")
        quality = value["quality"]
        return cls(
            h_map=DualErrorMapSlice.from_dict(value["h_map"]),
            o_map=DualErrorMapSlice.from_dict(value["o_map"]),
            objective_kind=str(value["objective_kind"]),
            revision=_revision_from_dict(value["revision"]),
            label_constraint=str(value["label_constraint"]),
            quality=DualErrorMapQuality(
                **{**quality, "warnings": tuple(quality["warnings"])}),
            warnings=tuple(value["warnings"]),
        )


def compute_dual_error_map(
    session: AnalysisSession,
    map_config: DualErrorMapConfig | None = None,
    *,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> DualErrorMapResult:
    """Compute two fixed-nuisance full-signal SSE surfaces for current M3."""

    settings = DualErrorMapConfig() if map_config is None else map_config
    if not isinstance(settings, DualErrorMapConfig):
        raise TypeError("map_config must be DualErrorMapConfig")
    if settings.mode is ErrorMapMode.PROFILE_NUISANCE:
        raise NotImplementedError("PROFILE_NUISANCE is not implemented for M3")
    if settings.mode is not ErrorMapMode.FIXED:
        raise ValueError("unsupported M3 Error Map mode")
    fit_result = getattr(session, "fit_result", None)
    evaluator = DualHOObjectiveEvaluator.from_fit_result(fit_result, session)
    context = evaluator.context
    assert context.revision is not None
    fit_params = dict(fit_result.params)
    if "y0" not in fit_params:
        assert context.fixed_baseline is not None
        fit_params["y0"] = context.fixed_baseline
    label_risk = (
        context.label_constraint == "none" or
        abs(log10(fit_params["D_H_cm2_s"]) - log10(fit_params["D_O_cm2_s"]))
        <= settings.near_label_decades
    )
    amplitude_max = max(fit_params["A_H"], fit_params["A_O"])
    weak_amplitude = bool(amplitude_max > 0 and
                          min(fit_params["A_H"], fit_params["A_O"])
                          / amplitude_max < settings.weak_amplitude_fraction)

    prepared = []
    for process in ("H", "O"):
        prepared.append(_prepare_ranges(process, fit_params, context.parameter_bounds,
                                        context.label_constraint, settings))
    if cancel_check is not None:
        cancel_check()
    fitted_sse_h = evaluator.evaluate_raw_sse(fit_params)
    fitted_sse_o = evaluator.evaluate_raw_sse(fit_params)
    parity = all(isclose(value, fit_result.SSE, rel_tol=_PARITY_RTOL, abs_tol=_PARITY_ATOL)
                 for value in (fitted_sse_h, fitted_sse_o)) and isclose(
                     fitted_sse_h, fitted_sse_o, rel_tol=_PARITY_RTOL, abs_tol=_PARITY_ATOL)
    if not parity:
        raise ValueError("SSE_MISMATCH: H/O fitted-point raw SSE differs from FitResult.SSE")

    maps = []
    for index, process in enumerate(("H", "O")):
        d_range, k_range, clipped = prepared[index]
        maps.append(_compute_slice(
            process, evaluator, fit_params, settings, d_range, k_range, clipped,
            fitted_sse_h if process == "H" else fitted_sse_o,
            label_risk=label_risk, weak_amplitude=weak_amplitude,
            cancel_check=cancel_check, progress_callback=progress_callback,
            progress_offset=index,
        ))
    h_map, o_map = maps
    warnings = tuple(dict.fromkeys((*h_map.warnings, *o_map.warnings)))
    quality = DualErrorMapQuality(
        fitted_point_parity_ok=parity,
        invalid_fraction_max=max(h_map.quality.invalid_fraction,
                                 o_map.quality.invalid_fraction),
        label_risk=label_risk,
        weak_process_amplitude=weak_amplitude,
        warnings=warnings,
    )
    return DualErrorMapResult(
        h_map=h_map, o_map=o_map,
        objective_kind=_OBJECTIVE_KIND,
        revision=context.revision,
        label_constraint=context.label_constraint,
        quality=quality, warnings=warnings,
    )


def _prepare_ranges(
    process: str,
    params: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    label_constraint: str,
    config: DualErrorMapConfig,
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    d_name, k_name = f"D_{process}_cm2_s", f"k_{process}_cm_s"
    d_limits = tuple(log10(value) for value in bounds[d_name])
    k_limits = tuple(log10(value) for value in bounds[k_name])
    if label_constraint == "D_H>=D_O":
        if process == "H":
            d_limits = (max(d_limits[0], log10(params["D_O_cm2_s"])), d_limits[1])
        else:
            d_limits = (d_limits[0], min(d_limits[1], log10(params["D_H_cm2_s"])))
    if process == "H":
        requested_d, requested_k = config.h_log10_D_range, config.h_log10_k_range
    else:
        requested_d, requested_k = config.o_log10_D_range, config.o_log10_k_range
    d_range, d_clipped = _axis_range(
        requested_d, log10(params[d_name]), d_limits,
        config.auto_half_width_decades,
    )
    k_range, k_clipped = _axis_range(
        requested_k, log10(params[k_name]), k_limits,
        config.auto_half_width_decades,
    )
    return d_range, k_range, d_clipped or k_clipped


def _axis_range(
    manual: tuple[float, float] | None,
    center: float,
    limits: tuple[float, float],
    half_width: float,
) -> tuple[tuple[float, float], bool]:
    lower, upper = limits
    if lower >= upper:
        raise ValueError("validated bounds leave no M3 Error Map range")
    if manual is not None:
        if manual[0] < lower or manual[1] > upper:
            raise ValueError("MANUAL Error Map range lies outside validated bounds or label constraint")
        return manual, False
    requested = center - half_width, center + half_width
    clipped = max(requested[0], lower), min(requested[1], upper)
    if clipped[0] >= clipped[1]:
        raise ValueError("AUTO Error Map range has no width within validated bounds")
    return clipped, clipped != requested


def _compute_slice(
    process: str,
    evaluator: DualHOObjectiveEvaluator,
    fit_params: Mapping[str, float],
    config: DualErrorMapConfig,
    d_range: tuple[float, float],
    k_range: tuple[float, float],
    range_clipped: bool,
    fitted_sse: float,
    *,
    label_risk: bool,
    weak_amplitude: bool,
    cancel_check: Callable[[], None] | None,
    progress_callback: Callable[[float, str], None] | None,
    progress_offset: int,
) -> DualErrorMapSlice:
    d_name, k_name = f"D_{process}_cm2_s", f"k_{process}_cm_s"
    d_axis = np.linspace(*d_range, config.grid_size)
    k_axis = np.linspace(*k_range, config.grid_size)
    grid = np.full((d_axis.size, k_axis.size), np.nan, dtype=np.float64)
    valid = np.zeros(grid.shape, dtype=np.bool_)
    invalid_cells: list[DualErrorMapCellFailure] = []
    params = dict(fit_params)
    for d_index, d_value in enumerate(d_axis):
        params[d_name] = 10.0 ** float(d_value)
        for k_index, k_value in enumerate(k_axis):
            if cancel_check is not None:
                cancel_check()
            params[k_name] = 10.0 ** float(k_value)
            try:
                sse = evaluator.evaluate_raw_sse(params)
                if not isfinite(sse) or sse < 0:
                    raise FloatingPointError("full-signal raw SSE is nonfinite or negative")
            except Exception as exc:
                invalid_cells.append(DualErrorMapCellFailure(
                    d_index, k_index, float(d_value), float(k_value),
                    f"{type(exc).__name__}: {exc}",
                ))
            else:
                grid[d_index, k_index] = sse
                valid[d_index, k_index] = True
        if progress_callback is not None:
            progress_callback((progress_offset + (d_index + 1) / d_axis.size) / 2,
                              f"M3 {process} Error Map row {d_index + 1}/{d_axis.size}")
    if not valid.any():
        first = invalid_cells[0].reason if invalid_cells else "no valid cell"
        raise RuntimeError(f"M3 {process} Error Map has no valid cells; first: {first}")
    minimum_index = np.unravel_index(int(np.nanargmin(grid)), grid.shape)
    d_index, k_index = (int(minimum_index[0]), int(minimum_index[1]))
    fit_point = DualErrorMapPoint(log10(fit_params[d_name]),
                                 log10(fit_params[k_name]), fitted_sse)
    grid_min = DualErrorMapPoint(float(d_axis[d_index]), float(k_axis[k_index]),
                                float(grid[d_index, k_index]), d_index, k_index)
    quality = _assess_quality(
        d_axis, k_axis, grid, valid, fit_point, grid_min, config,
        range_clipped=range_clipped, fit_sse=fitted_sse,
        label_risk=label_risk, weak_amplitude=weak_amplitude,
    )
    fixed_nuisance = {name: float(value) for name, value in fit_params.items()
                      if name not in {d_name, k_name}}
    context = evaluator.context
    assert context.revision is not None
    return DualErrorMapSlice(
        process=process, log10_D=d_axis, log10_k=k_axis,
        raw_sse_grid=grid, valid_mask=valid,
        fitted_point=fit_point, grid_minimum=grid_min,
        fixed_nuisance=fixed_nuisance, objective_kind=_OBJECTIVE_KIND,
        revision=context.revision, label_constraint=context.label_constraint,
        range_clipped=range_clipped,
        invalid_cells=tuple(invalid_cells), quality=quality,
        warnings=quality.warnings,
    )


def _assess_quality(
    d_axis: NDArray[np.float64], k_axis: NDArray[np.float64],
    grid: NDArray[np.float64], valid: NDArray[np.bool_],
    fitted: DualErrorMapPoint, minimum: DualErrorMapPoint,
    config: DualErrorMapConfig, *, range_clipped: bool,
    fit_sse: float, label_risk: bool, weak_amplitude: bool,
) -> DualSliceQuality:
    in_range = bool(d_axis[0] <= fitted.log10_D <= d_axis[-1] and
                    k_axis[0] <= fitted.log10_k <= k_axis[-1])
    on_edge = bool(minimum.d_index in (0, len(d_axis) - 1) or
                   minimum.k_index in (0, len(k_axis) - 1))
    d_step = float((d_axis[-1] - d_axis[0]) / (len(d_axis) - 1))
    k_step = float((k_axis[-1] - k_axis[0]) / (len(k_axis) - 1))
    coarse = max(d_step, k_step) > 0.05 or config.grid_size <= ErrorMapResolution.QUICK.value
    invalid_fraction = float(1.0 - np.count_nonzero(valid) / valid.size)
    sse_mismatch = not isclose(fitted.raw_sse, fit_sse,
                               rel_tol=_PARITY_RTOL, abs_tol=_PARITY_ATOL)
    elongated, ratio = _valley_elongation(d_axis, k_axis, grid, valid)
    warnings: list[str] = []
    if not in_range:
        warnings.append("FIT_OUTSIDE_RANGE")
    if on_edge:
        warnings.append("EDGE_MINIMUM")
    if range_clipped:
        warnings.append("AUTO_RANGE_CLIPPED")
    if min(d_axis[-1] - d_axis[0], k_axis[-1] - k_axis[0]) < config.narrow_range_decades:
        warnings.append("RANGE_TOO_NARROW")
    if coarse:
        warnings.append("GRID_COARSE")
    if invalid_fraction > 0:
        warnings.append(f"INVALID_CELLS: {invalid_fraction:.1%}")
    if elongated:
        warnings.append("ELONGATED_VALLEY")
    if sse_mismatch:
        warnings.append("SSE_MISMATCH")
    if label_risk:
        warnings.append("LABEL_RISK")
    if weak_amplitude:
        warnings.append("WEAK_PROCESS_AMPLITUDE")
    return DualSliceQuality(
        optimum_in_range=in_range, grid_min_on_edge=on_edge,
        coarse_grid=coarse, invalid_fraction=invalid_fraction,
        elongated_valley=elongated, elongation_ratio=ratio,
        sse_mismatch=sse_mismatch, label_risk=label_risk,
        weak_process_amplitude=weak_amplitude,
        warnings=tuple(warnings),
    )


def _valley_elongation(
    d_axis: NDArray[np.float64], k_axis: NDArray[np.float64],
    grid: NDArray[np.float64], valid: NDArray[np.bool_],
) -> tuple[bool, float | None]:
    finite = grid[valid]
    if finite.size < 6:
        return False, None
    low, high = float(np.min(finite)), float(np.max(finite))
    # A display diagnostic, not a likelihood contour or CI threshold.
    threshold = low + max(0.05 * low, 0.005 * (high - low))
    near = valid & (grid <= threshold)
    indices = np.argwhere(near)
    if len(indices) < 6:
        return False, None
    coordinates = np.column_stack((d_axis[indices[:, 0]], k_axis[indices[:, 1]]))
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[1] <= 0:
        return False, None
    ratio = float(eigenvalues[1] / max(eigenvalues[0], np.finfo(float).eps))
    return ratio >= 25.0, ratio


def _revision_from_dict(value: Mapping[str, object]) -> DualObjectiveRevision:
    from pyecr.services.workflow import WorkflowRevision

    return DualObjectiveRevision(
        dataset_revision=int(value["dataset_revision"]),
        preprocessing_revision=int(value["preprocessing_revision"]),
        model_revision=int(value["model_revision"]),
        fit_revision=WorkflowRevision(**value["fit_revision"]),
    )
