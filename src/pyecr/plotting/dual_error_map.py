"""Read-only display arrays for already computed M3 H/O objective maps.

This module changes only color values. It never evaluates the fit objective or
mutates the saved raw-SSE grid. Relative colors are not confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from pyecr.diagnostics.dual_error_map import DualErrorMapSlice


class DualMapMetric(str, Enum):
    LOG_RATIO = "log10(SSE/SSE_min)"
    RAW_SSE = "SSE"


@dataclass(frozen=True, slots=True)
class DualMapHoverCell:
    log10_D: float
    log10_k: float
    D_cm2_s: float
    k_cm_s: float
    valid: bool
    metric_value: float | None
    raw_sse: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class DualMapPlotData:
    process: str
    log10_D: NDArray[np.float64]
    log10_k: NDArray[np.float64]
    values: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    display_values: NDArray[np.float64]
    display_valid_mask: NDArray[np.bool_]
    fitted_point: tuple[float, float]
    grid_minimum: tuple[float, float] | None
    metric: DualMapMetric
    label: str
    display_warning: str
    invalid_reasons: dict[tuple[int, int], str]

    def nearest_cell(self, log10_D: float, log10_k: float) -> DualMapHoverCell | None:
        """Inspect one stored cell, including its invalidity, without computing SSE."""

        if not (np.isfinite(log10_D) and np.isfinite(log10_k)):
            return None
        if not (self.log10_D[0] <= log10_D <= self.log10_D[-1]
                and self.log10_k[0] <= log10_k <= self.log10_k[-1]):
            return None
        i = int(np.abs(self.log10_D - log10_D).argmin())
        j = int(np.abs(self.log10_k - log10_k).argmin())
        valid = bool(self.valid_mask[i, j])
        metric_finite = bool(self.display_valid_mask[i, j])
        return DualMapHoverCell(
            log10_D=float(self.log10_D[i]), log10_k=float(self.log10_k[j]),
            D_cm2_s=10.0 ** float(self.log10_D[i]),
            k_cm_s=10.0 ** float(self.log10_k[j]),
            valid=valid,
            metric_value=float(self.values[i, j]) if metric_finite else None,
            raw_sse=float(self.raw_sse_at(i, j)) if valid else None,
            reason=(self.invalid_reasons.get((i, j), "invalid numerical cell")
                    if not valid else self.display_warning if not metric_finite else None),
        )

    def raw_sse_at(self, i: int, j: int) -> float:
        return float(self.raw_sse_grid[i, j])

    raw_sse_grid: NDArray[np.float64]


def dual_map_plot_data(
    result: DualErrorMapSlice,
    metric: DualMapMetric = DualMapMetric.LOG_RATIO,
) -> DualMapPlotData:
    if not isinstance(result, DualErrorMapSlice) or not isinstance(metric, DualMapMetric):
        raise TypeError("requires DualErrorMapSlice and DualMapMetric")
    raw = result.raw_sse_grid
    mask = result.valid_mask
    warning = ""
    if metric is DualMapMetric.RAW_SSE:
        values = raw
    else:
        values = np.full(raw.shape, np.nan, dtype=np.float64)
        finite_raw = raw[mask]
        if finite_raw.size:
            minimum = float(np.min(finite_raw))
            if minimum > 0:
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    values[mask] = np.log10(finite_raw / minimum)
            else:
                warning = "SSE minimum is zero; log-ratio colors are undefined. Select raw SSE."
        values.flags.writeable = False
    display_mask = np.asarray(mask & np.isfinite(values), dtype=np.bool_)
    display_mask.flags.writeable = False
    display = np.full(raw.shape, np.nan, dtype=np.float64)
    display[display_mask] = values[display_mask]
    display.flags.writeable = False
    reasons = {(cell.d_index, cell.k_index): cell.reason for cell in result.invalid_cells}
    minimum_point = result.grid_minimum
    return DualMapPlotData(
        process=result.process, log10_D=result.log10_D, log10_k=result.log10_k,
        values=values, valid_mask=mask, display_values=display,
        display_valid_mask=display_mask,
        fitted_point=(result.fitted_point.log10_D, result.fitted_point.log10_k),
        grid_minimum=(None if minimum_point is None else
                      (minimum_point.log10_D, minimum_point.log10_k)),
        metric=metric, label=metric.value, display_warning=warning,
        invalid_reasons=reasons, raw_sse_grid=raw,
    )


def dual_map_color_limits(
    h_data: DualMapPlotData, o_data: DualMapPlotData, *, shared: bool,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Color-scale ranges only; grids and objective values stay untouched."""

    if h_data.metric is not o_data.metric:
        raise ValueError("shared H/O scale requires the same display metric")
    h_values = h_data.display_values[h_data.display_valid_mask]
    o_values = o_data.display_values[o_data.display_valid_mask]
    if shared:
        combined = np.concatenate((h_values, o_values))
        limits = _color_limits(combined)
        return (limits if h_values.size else None,
                limits if o_values.size else None)
    return _color_limits(h_values), _color_limits(o_values)


def _color_limits(values: NDArray[np.float64]) -> tuple[float, float] | None:
    if values.size == 0:
        return None
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        high = low + max(abs(low), 1.0) * 1.0e-12
    return low, high
