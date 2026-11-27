"""Explicit, reviewable optional preprocessing; no smoothing or automatic commit."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import theilslopes

from pyecr.fitting.base import FitResult
from pyecr.io.dataset import EcrDataset
from pyecr.preprocessing.pipeline import ProcessingStep

OPTIONAL_OPERATIONS = ("drift_correction", "outlier_mask", "unmask_points", "adaptive_downsample")


@dataclass(frozen=True, slots=True)
class OptionalPreview:
    step: ProcessingStep
    dataset_hash: str
    notes: tuple[str, ...]

    def accept(self, dataset: EcrDataset) -> ProcessingStep:
        from pyecr.report.snapshot import dataset_sha256
        if dataset_sha256(dataset) != self.dataset_hash:
            raise ValueError("Optional preprocessing preview is stale; preview again")
        return self.step


def _preview(dataset: EcrDataset, operation: str, parameters: Mapping[str, object], *notes: str) -> OptionalPreview:
    from pyecr.report.snapshot import dataset_sha256
    return OptionalPreview(ProcessingStep.create(operation, parameters), dataset_sha256(dataset), tuple(notes))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} requires start/end")
    start, end = (_finite(v, name) for v in value)
    if end <= start:
        raise ValueError(f"{name} must have positive duration")
    return start, end


def _indices(values: Sequence[int], size: int, *, empty: bool = False) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise ValueError("indices must be integers")
    indices = list(values)
    if (not indices and not empty) or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= size for i in indices):
        raise ValueError("indices must be valid original row indices")
    if len(set(indices)) != len(indices):
        raise ValueError("indices must be unique")
    return sorted(int(i) for i in indices)


def preview_drift(dataset: EcrDataset, *, reference_window: Sequence[float], scope: Sequence[float], method: str = "linear", anchor_time: float | None = None) -> OptionalPreview:
    """Fit a reference plateau trend. Scope and reference use processed time units."""
    start, end = _interval(reference_window, "reference_window")
    scope_start, scope_end = _interval(scope, "scope")
    t, y = dataset.processed_time, dataset.processed_signal
    reference = (t >= start) & (t <= end)
    affected = (t >= scope_start) & (t <= scope_end)
    if reference.sum() < 3 or np.ptp(t[reference]) <= 0:
        raise ValueError("Drift reference window needs at least 3 points and positive duration")
    if not affected.any():
        raise ValueError("Drift scope contains no processed points")
    anchor = float(t[reference][0]) if anchor_time is None else _finite(anchor_time, "anchor_time")
    if method == "linear":
        slope, intercept = np.linalg.lstsq(np.column_stack((t[reference] - anchor, np.ones(reference.sum()))), y[reference], rcond=None)[0]
    elif method == "robust_linear":
        # scipy's deterministic Theil-Sen median pairwise slope.
        if reference.sum() > 3000:
            raise ValueError("Theil-Sen reference is limited to 3000 points; select a smaller reference window")
        fit = theilslopes(y[reference], t[reference] - anchor, method="separate")
        slope, intercept = fit.slope, fit.intercept
    else:
        raise ValueError("Drift method must be linear or robust_linear")
    residual = y[reference] - (slope * (t[reference] - anchor) + intercept)
    parameters = {
        "method": method, "estimator": "least_squares" if method == "linear" else "scipy.stats.theilslopes",
        "reference_window": [start, end], "scope": [scope_start, scope_end], "anchor_time": anchor,
        "slope": float(slope), "intercept_at_anchor": float(intercept),
        "reference_std": float(np.std(residual)), "reference_points": int(reference.sum()),
        "reference_raw_indices": np.flatnonzero(dataset.mask)[reference].tolist(),
        "affected_raw_indices": np.flatnonzero(dataset.mask)[affected].tolist(),
        "formula": "y_corrected = y - slope * (t - anchor_time) within scope; intercept retained",
        "time_unit": str(dataset.metadata.get("time_unit", "s")), "signal_unit": str(dataset.metadata.get("signal_unit", "")),
    }
    return _preview(dataset, "drift_correction", parameters, "Choose a stable reference plateau: a transient slope can remove material response.", f"slope={slope:.6g}; reference noise={np.std(residual):.6g}; scope=[{scope_start:g}, {scope_end:g}]")


def preview_outliers(dataset: EcrDataset, *, threshold: float = 6.0, half_window: int = 3, noise_floor: float = 0.0) -> OptionalPreview:
    """Local time-linear interpolation residual + MAD, preserving nonuniform time."""
    threshold = _finite(threshold, "threshold")
    noise_floor = _finite(noise_floor, "noise_floor")
    if threshold <= 0 or noise_floor < 0:
        raise ValueError("threshold must be positive and noise_floor nonnegative")
    if isinstance(half_window, bool) or not isinstance(half_window, Integral) or half_window < 2:
        raise ValueError("half_window must be an integer >= 2")
    t, y = dataset.processed_time, dataset.processed_signal
    if t.size < 2 * half_window + 1:
        raise ValueError("Outlier preview needs more points for the local window")
    residual = np.zeros(t.size)
    residual[1:-1] = y[1:-1] - (y[:-2] + (y[2:] - y[:-2]) * (t[1:-1] - t[:-2]) / (t[2:] - t[:-2]))
    eps_floor = np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))) * 32
    rows = []
    raw = np.flatnonzero(dataset.mask)
    for i in range(1, t.size - 1):
        local = residual[max(1, i - half_window):min(t.size - 1, i + half_window + 1)]
        center = float(np.median(local))
        scale = max(float(1.4826 * np.median(np.abs(local - center))), noise_floor, eps_floor)
        z = abs(float(residual[i]) - center) / scale
        # Require a local extremum to avoid marking the two neighbors of a spike.
        extremum = (y[i] - y[i-1]) * (y[i+1] - y[i]) < 0
        peak_residual = abs(residual[i]) > max(abs(residual[i-1]), abs(residual[i+1]))
        if extremum and peak_residual and z > threshold:
            rows.append({"raw_index": int(raw[i]), "reason": "local extremum exceeds robust interpolation-residual threshold", "score": float(z), "residual": float(residual[i]), "noise": scale})
    return _preview(dataset, "outlier_mask", {"method": "local_linear_residual_mad", "threshold": threshold, "half_window": int(half_window), "noise_floor": noise_floor, "raw_indices": [r["raw_index"] for r in rows], "reasons": rows}, "Review original indices before Accept; endpoints are never suggested.", "Sharp physical transitions may be flagged; edit the reviewed index list to retain them.")


def preview_downsampling(dataset: EcrDataset, *, target_points: int, early_until: float, late_power: float = 2.0) -> OptionalPreview:
    """Keep every early point; fill increasing time targets with existing later rows."""
    t = dataset.processed_time
    if isinstance(target_points, bool) or not isinstance(target_points, Integral) or target_points < 3 or target_points >= t.size:
        raise ValueError("target_points must be >= 3 and smaller than current point count")
    early_until = _finite(early_until, "early_until")
    power = _finite(late_power, "late_power")
    if early_until < t[0] or early_until >= t[-1] or power <= 1:
        raise ValueError("early_until must lie in the time range; late_power must exceed 1")
    early = np.flatnonzero(t <= early_until)
    remaining = int(target_points) - early.size
    if remaining < 2:
        raise ValueError("target_points must retain every early point and at least two later points")
    late = np.arange(early.size, t.size)
    targets = t[late[0]] + (t[-1] - t[late[0]]) * np.linspace(0, 1, remaining) ** power
    selected = set(int(v) for v in early)
    selected.update(int(late[min(np.searchsorted(t[late], v), late.size - 1)]) for v in targets)
    # Collisions are filled deterministically in earliest unselected order.
    for i in late:
        if len(selected) >= target_points:
            break
        selected.add(int(i))
    positions = np.array(sorted(selected), dtype=int)
    original = np.flatnonzero(dataset.mask)
    return _preview(dataset, "adaptive_downsample", {"method": "early_dense_power_time", "target_points": int(target_points), "early_until": early_until, "late_power": power, "selected_indices": original[positions].tolist(), "input_indices": original.tolist(), "early_indices": original[early].tolist(), "before_points": int(t.size), "after_points": int(positions.size), "interpolation": False}, "All early points and both endpoints retained. Later rows are selected, never averaged.", "Refit explicitly to compare parameters and residuals; SSE on different point sets is not directly comparable.")


def active_outlier_indices(steps: Sequence[ProcessingStep]) -> tuple[int, ...]:
    active: set[int] = set()
    for step in steps:
        if step.operation == "outlier_mask":
            active.update(step.parameters["raw_indices"])
        elif step.operation == "unmask_points":
            active.difference_update(step.parameters["raw_indices"])
    return tuple(sorted(active))


def apply_optional(dataset: EcrDataset, step: ProcessingStep, *, suppressed: set[int] | None = None) -> EcrDataset:
    """Apply stored coefficients/indices exactly. Replay never re-detects outliers."""
    p = step.to_dict()["parameters"]
    operation = step.operation
    kwargs: dict[str, object] = {}
    if operation == "drift_correction":
        if p.get("method") not in ("linear", "robust_linear") or not p.get("formula"):
            raise ValueError("Drift requires a preview with stored method/formula/coefficients")
        _interval(p["reference_window"], "reference_window")
        start, end = _interval(p["scope"], "scope")
        slope, anchor = _finite(p["slope"], "slope"), _finite(p["anchor_time"], "anchor_time")
        _finite(p["intercept_at_anchor"], "intercept_at_anchor")
        selected = (dataset.processed_time >= start) & (dataset.processed_time <= end)
        y = np.array(dataset.processed_signal, copy=True)
        y[selected] -= slope * (dataset.processed_time[selected] - anchor)
        kwargs["processed_signal"] = y
        p["applied_raw_indices"] = np.flatnonzero(dataset.mask)[selected].tolist()
    elif operation == "outlier_mask":
        indices = _indices(p["raw_indices"], dataset.raw_time.size)
        if p.get("method") != "local_linear_residual_mad" or _finite(p["threshold"], "threshold") <= 0 or "reasons" not in p:
            raise ValueError("Outlier mask requires method, positive threshold and reasons")
        if sorted(row["raw_index"] for row in p["reasons"]) != indices or any(not row.get("reason") for row in p["reasons"]):
            raise ValueError("Outlier mask must record a reason for each original index")
        effective = sorted(set(indices) - (suppressed or set()))
        keep = ~np.isin(np.flatnonzero(dataset.mask), effective)
        mask = np.array(dataset.mask, copy=True)
        mask[effective] = False
        kwargs.update(processed_time=dataset.processed_time[keep], processed_signal=dataset.processed_signal[keep], mask=mask)
        p["applied_raw_indices"] = effective
    elif operation == "adaptive_downsample":
        if p.get("method") != "early_dense_power_time" or p.get("interpolation") is not False:
            raise ValueError("Downsample requires stored exact-selection preview parameters")
        indices = _indices(p["selected_indices"], dataset.raw_time.size)
        if len(indices) != p["target_points"] or len(indices) != p["after_points"]:
            raise ValueError("Downsample point count disagrees with selected indices")
        raw = np.flatnonzero(dataset.mask)
        if not set(indices).issubset(set(raw)):
            raise ValueError("Downsample selected indices are not available in the replayed view")
        _finite(p["early_until"], "early_until")
        early = _indices(p["early_indices"], dataset.raw_time.size, empty=True)
        if len(indices) < 3 or not set(early).issubset(indices) or raw[0] not in indices or raw[-1] not in indices:
            raise ValueError("Downsample must preserve early points and endpoints")
        keep = np.isin(raw, indices)
        mask = np.array(dataset.mask, copy=True)
        mask[raw[~keep]] = False
        kwargs.update(processed_time=dataset.processed_time[keep], processed_signal=dataset.processed_signal[keep], mask=mask)
    elif operation == "unmask_points":
        _indices(p["raw_indices"], dataset.raw_time.size)
        p["method"] = "replay_cancel_prior_masks"
    else:
        raise ValueError("Unknown optional operation")
    return dataset._with_operation(operation=operation, parameters=p, **kwargs)


def fit_summary(result: FitResult, *, data_revision: int | None = None, dataset: EcrDataset | None = None) -> dict[str, object]:
    """Persist the before-fit evidence without mutating or refitting it."""
    residual = np.asarray(result.residuals)
    summary = {"params": dict(result.params), "SSE": float(result.SSE), "points": int(residual.size), "residual_rms": float(np.sqrt(np.mean(residual**2))), "residual_max_abs": float(np.max(np.abs(residual))), "data_revision": data_revision}
    if dataset is not None:
        if residual.size != dataset.processed_time.size:
            raise ValueError("Residual comparison needs matching fitted dataset")
        summary["residual_curve"] = {"time": dataset.processed_time.tolist(), "residual": residual.tolist(), "original_indices": np.flatnonzero(dataset.mask).tolist(), "time_unit": str(dataset.metadata.get("time_unit", "s")), "signal_unit": str(dataset.metadata.get("signal_unit", ""))}
    return summary


def preprocessing_comparison(dataset: EcrDataset, result: FitResult | None) -> dict[str, object] | None:
    before = next((r.parameters.get("before_fit") for r in dataset.history if r.operation in OPTIONAL_OPERATIONS and r.parameters.get("before_fit")), None)
    if before is None:
        return None
    return {"before": before, "after": None if result is None else fit_summary(result, dataset=dataset), "note": "Manual refit. Point sets or signal scales may differ; compare residual RMS and parameters, and interpret SSE with point count. Raw data and the original fit are retained."}
