"""Shared, read-only presentation of an already evaluated M1 Error Map."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from pyecr.diagnostics.error_map import ErrorMapResult


class ErrorMapMetric(str, Enum):
    LOG_RATIO = "log10(SSE/SSE_min)"
    RELATIVE = "SSE/SSE_min"
    SSE = "SSE"


class ErrorMapClipMode(str, Enum):
    """Optional display contrast limit; never changes stored objective arrays."""

    FULL = "Full"
    P95 = "P95"
    P99 = "P99"

    @property
    def percentile(self) -> float | None:
        return {self.FULL: None, self.P95: 95.0, self.P99: 99.0}[self]


@dataclass(frozen=True, slots=True)
class ErrorMapClippingMetadata:
    """Audit of the color/heat display transformation, not scientific data."""

    metric: ErrorMapMetric
    mode: ErrorMapClipMode
    raw_min: float | None
    raw_max: float | None
    display_min: float | None
    display_max: float | None
    clipped_cells: int
    nonfinite_metric_cells: int


@dataclass(frozen=True, slots=True)
class ErrorMapPlotData:
    D_cm2_s: NDArray[np.float64]
    k_cm_s: NDArray[np.float64]
    values: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    fitted_D_cm2_s: float
    fitted_k_cm_s: float
    grid_minimum: tuple[float, float] | None
    label: str
    display_values: NDArray[np.float64]
    display_valid_mask: NDArray[np.bool_]
    clipping: ErrorMapClippingMetadata

    def nearest_cell(self, D_cm2_s: float, k_cm_s: float) -> tuple[int, int, float] | None:
        """Hover only inspects stored cells; no objective evaluation occurs."""

        if D_cm2_s <= 0 or k_cm_s <= 0:
            return None
        if not (self.D_cm2_s[0] <= D_cm2_s <= self.D_cm2_s[-1]
                and self.k_cm_s[0] <= k_cm_s <= self.k_cm_s[-1]):
            return None
        d_index = int(np.abs(np.log10(self.D_cm2_s) - np.log10(D_cm2_s)).argmin())
        k_index = int(np.abs(np.log10(self.k_cm_s) - np.log10(k_cm_s)).argmin())
        if not self.display_valid_mask[d_index, k_index]:
            return None
        return d_index, k_index, float(self.values[d_index, k_index])


def error_map_plot_data(
    result: ErrorMapResult, metric: ErrorMapMetric = ErrorMapMetric.LOG_RATIO,
    *, clip: ErrorMapClipMode = ErrorMapClipMode.FULL,
) -> ErrorMapPlotData:
    if (not isinstance(result, ErrorMapResult)
            or not isinstance(metric, ErrorMapMetric)
            or not isinstance(clip, ErrorMapClipMode)):
        raise TypeError("requires ErrorMapResult, ErrorMapMetric, and ErrorMapClipMode")
    values = {
        ErrorMapMetric.LOG_RATIO: result.log_ratio_grid,
        ErrorMapMetric.RELATIVE: result.relative_grid,
        ErrorMapMetric.SSE: result.objective_grid,
    }[metric]
    display_mask = np.asarray(result.valid_mask & np.isfinite(values), dtype=np.bool_)
    displayed = np.full(values.shape, np.nan, dtype=np.float64)
    finite = values[display_mask]
    if finite.size:
        upper = (float(np.max(finite)) if clip.percentile is None else
                 float(np.percentile(finite, clip.percentile)))
        displayed[display_mask] = np.minimum(finite, upper)
        clipping = ErrorMapClippingMetadata(
            metric=metric, mode=clip,
            raw_min=float(np.min(finite)), raw_max=float(np.max(finite)),
            display_min=float(np.min(displayed[display_mask])),
            display_max=float(np.max(displayed[display_mask])),
            clipped_cells=int(np.count_nonzero(finite > upper)),
            nonfinite_metric_cells=int(np.count_nonzero(
                result.valid_mask & ~np.isfinite(values))),
        )
    else:
        clipping = ErrorMapClippingMetadata(
            metric=metric, mode=clip,
            raw_min=None, raw_max=None, display_min=None, display_max=None,
            clipped_cells=0,
            nonfinite_metric_cells=int(np.count_nonzero(
                result.valid_mask & ~np.isfinite(values))),
        )
    displayed.flags.writeable = False
    display_mask.flags.writeable = False
    minimum = result.grid_minimum
    return ErrorMapPlotData(
        D_cm2_s=np.power(10.0, result.log10_D),
        k_cm_s=np.power(10.0, result.log10_k),
        values=values,
        valid_mask=result.valid_mask,
        fitted_D_cm2_s=10.0 ** result.fitted_point.log10_D,
        fitted_k_cm_s=10.0 ** result.fitted_point.log10_k,
        grid_minimum=None if minimum is None else (
            10.0 ** minimum.log10_D, 10.0 ** minimum.log10_k,
        ),
        label=metric.value,
        display_values=displayed,
        display_valid_mask=display_mask,
        clipping=clipping,
    )


def log_axis_edges(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cell edges in decade coordinates, shared by GUI and static figures."""

    middle = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - 0.5 * (centers[1] - centers[0])],
                           middle,
                           [centers[-1] + 0.5 * (centers[-1] - centers[-2])]))
