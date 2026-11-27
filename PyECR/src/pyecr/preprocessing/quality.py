"""Read-only plateau suggestions and the pre-fit data quality gate.

The suggestion functions never change an EcrDataset or a processing recipe. A
caller must explicitly create and apply a ``choose_*_stability_window`` step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pyecr.config import AnalysisConfig
from pyecr.boundary.assistant import validate_boundary_fit_coverage
from pyecr.io.dataset import EcrDataset

WindowKind = Literal["initial", "final"]

SIGNED_PROFILE_FIT_REASON = (
    "Signed-DCT normalization is a terminal-referenced diagnostic profile; "
    "standard Fit requires a supported 0-to-1 progress representation. "
    "Undo the signed step (or Reset to the imported source), review physical/source units, "
    "then explicitly choose initial/final plateaus and normalize as progress when appropriate. "
    "Do not relabel signed values as progress."
)


@dataclass(frozen=True, slots=True)
class StabilityWindowSuggestion:
    kind: WindowKind
    start_time: float
    end_time: float
    score: float
    slope: float
    std: float
    reason: str
    confidence_class: Literal["high", "medium", "low"]
    n_points: int


@dataclass(frozen=True, slots=True)
class FitQualityReport:
    ready: bool
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _robust_sigma(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return 0.0
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def _local_noise(time: NDArray[np.float64], signal: NDArray[np.float64]) -> float:
    """Estimate measurement noise from linear interpolation residuals.

    This is a diagnostic calculation; it does not smooth or replace any data.
    """
    if signal.size < 5:
        return 0.0
    fraction = (time[1:-1] - time[:-2]) / (time[2:] - time[:-2])
    predicted = signal[:-2] + fraction * (signal[2:] - signal[:-2])
    return _robust_sigma(signal[1:-1] - predicted) / np.sqrt(1.5)


def _transition_index(time: NDArray[np.float64], signal: NDArray[np.float64]) -> int:
    slopes = np.abs(np.diff(signal) / np.diff(time))
    if slopes.size >= 5:
        padded = np.pad(slopes, (2, 2), mode="edge")
        slopes = np.median(
            np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1
        )
    return int(np.argmax(slopes))


def suggest_stability_window(
    dataset: EcrDataset,
    kind: WindowKind,
    *,
    min_points: int = 5,
) -> StabilityWindowSuggestion | None:
    """Rank contiguous edge windows by flatness, noise, and transition distance.

    A low-confidence candidate is still only a suggestion for review. ``None``
    means that no interval satisfies the minimum point and duration rules.
    """
    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if kind not in {"initial", "final"}:
        raise ValueError("kind must be 'initial' or 'final'")
    if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 3:
        raise ValueError("min_points must be an integer of at least 3")
    time = np.asarray(dataset.processed_time, dtype=np.float64)
    signal = np.asarray(dataset.processed_signal, dtype=np.float64)
    n = time.size
    if n < 2 * min_points or not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        return None
    if np.any(np.diff(time) <= 0.0):
        return None
    total_duration = float(time[-1] - time[0])
    if total_duration <= 0.0:
        return None
    minimum_duration = 0.025 * total_duration
    robust_amplitude = float(np.percentile(signal, 90) - np.percentile(signal, 10))
    amplitude = max(abs(robust_amplitude), float(np.ptp(signal)), np.finfo(float).eps)
    transition = _transition_index(time, signal)
    transition_time = float((time[transition] + time[transition + 1]) / 2.0)

    # Search the first or last third. Several widths prevent a short, quiet
    # stretch from automatically beating a longer and scientifically useful one.
    edge_count = max(min_points, int(np.ceil(0.35 * n)))
    lengths = sorted({min_points, max(min_points, int(np.ceil(0.08 * n))),
                      max(min_points, int(np.ceil(0.15 * n)))})
    best: StabilityWindowSuggestion | None = None
    for length in lengths:
        if length > edge_count:
            continue
        first_start = 0 if kind == "initial" else n - edge_count
        last_start = edge_count - length if kind == "initial" else n - length
        # Bound GUI click cost for very long traces while retaining both edge
        # positions and uniformly covering the candidate region.
        candidate_count = min(256, last_start - first_start + 1)
        starts = np.unique(
            np.rint(np.linspace(first_start, last_start, candidate_count)).astype(int)
        )
        for start in starts:
            end = start + length
            local_time = time[start:end]
            local_signal = signal[start:end]
            duration = float(local_time[-1] - local_time[0])
            if duration < minimum_duration:
                continue
            midpoint = float((local_time[0] + local_time[-1]) / 2.0)
            if (kind == "initial" and midpoint >= transition_time) or (
                kind == "final" and midpoint <= transition_time
            ):
                continue
            slope = float(np.polyfit(local_time - local_time[0], local_signal, 1)[0])
            std = float(np.std(local_signal, ddof=1))
            residual = local_signal - (local_signal.mean() + slope * (local_time - local_time.mean()))
            robust_noise = _robust_sigma(residual)
            slope_penalty = abs(slope) * duration / amplitude
            noise_penalty = robust_noise / amplitude
            edge_gap = (start if kind == "initial" else n - end) / max(n - 1, 1)
            transition_gap = abs(midpoint - transition_time) / total_duration
            duration_reward = min(duration / max(0.10 * total_duration, minimum_duration), 1.0)
            score = (
                exp(-3.0 * slope_penalty - 8.0 * noise_penalty - 3.0 * edge_gap)
                * (0.65 + 0.35 * min(transition_gap / 0.45, 1.0))
                * (0.65 + 0.35 * duration_reward)
            )
            score = float(np.clip(score, 0.0, 1.0))
            confidence: Literal["high", "medium", "low"] = (
                "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
            )
            reason = (
                f"{length} points over {duration:.4g} {dataset.metadata.get('time_unit', 's')}; "
                f"relative drift {slope_penalty:.3g}, robust noise {noise_penalty:.3g}, "
                f"transition distance {transition_gap:.2f} of trace"
            )
            candidate = StabilityWindowSuggestion(
                kind, float(local_time[0]), float(local_time[-1]), score,
                slope, std, reason, confidence, length,
            )
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def _selected_window_values(
    dataset: EcrDataset, kind: WindowKind
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    windows = dataset.metadata.get("stability_windows")
    if not isinstance(windows, Mapping):
        return None
    window = windows.get(kind)
    if not isinstance(window, Mapping):
        return None
    try:
        indices = np.asarray(window.get("raw_indices", ()), dtype=int)
    except (TypeError, ValueError):
        return None
    if indices.ndim != 1:
        return None
    current = np.flatnonzero(dataset.mask)
    selected = np.isin(current, indices)
    if not np.any(selected):
        return None
    return dataset.processed_time[selected], dataset.processed_signal[selected]


def evaluate_fit_quality(
    dataset: EcrDataset | None,
    config: AnalysisConfig | None,
) -> FitQualityReport:
    """Return deterministic blocking reasons and nonblocking review warnings."""
    blockers: list[str] = []
    warnings: list[str] = []
    if dataset is None:
        blockers.append("Import a dataset before fitting.")
    if config is None:
        blockers.append("Apply an analysis configuration before fitting.")
    if dataset is None:
        return FitQualityReport(False, tuple(blockers), ())
    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if dataset.metadata.get("normalization_mode") == "signed_terminal":
        blockers.append(SIGNED_PROFILE_FIT_REASON)
    time = np.asarray(dataset.processed_time, dtype=np.float64)
    signal = np.asarray(dataset.processed_signal, dtype=np.float64)
    if time.size != signal.size or not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        blockers.append("Fit time and signal must be finite, aligned arrays.")
        return FitQualityReport(False, tuple(blockers), tuple(warnings))
    free_parameters = (
        getattr(getattr(config, "model", None), "free_parameters", ())
        if isinstance(config, AnalysisConfig) else ()
    )
    minimum_points = max(3, 1 + len(free_parameters))
    if time.size < minimum_points:
        blockers.append(
            f"Fit requires at least {minimum_points} retained observations "
            "to exceed the free parameter count."
        )
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        blockers.append("Fit time must be strictly increasing.")
    if np.any(time < 0.0):
        blockers.append("Fit time must be nonnegative; review t0 or crop.")
    if dataset.metadata.get("time_unit", "s") != "s":
        blockers.append("Convert fit time to seconds before fitting.")
    if isinstance(config, AnalysisConfig):
        try:
            config.model.geometry.validate()
        except (TypeError, ValueError, AttributeError) as exc:
            blockers.append(f"Active sample geometry is invalid: {exc}")
        if time.size and np.all(np.isfinite(time)):
            try:
                validate_boundary_fit_coverage(dataset, config)
            except (TypeError, ValueError) as exc:
                blockers.append(str(exc))

    if signal.size >= 2:
        robust_amplitude = abs(float(np.percentile(signal, 90) - np.percentile(signal, 10)))
        amplitude = max(robust_amplitude, abs(float(signal[-1] - signal[0])))
        noise = _local_noise(time, signal) if time.size >= 5 and np.all(np.diff(time) > 0) else 0.0
        scale = max(float(np.max(np.abs(signal))), 1.0)
        if amplitude <= max(1.0e-12 * scale, 5.0 * noise):
            blockers.append("Signal amplitude is zero or no greater than estimated noise.")

    reference = None
    try:
        reference = dataset.normalization_reference()
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        blockers.append(str(exc))
    windows = dataset.metadata.get("stability_windows")
    if isinstance(windows, Mapping):
        initial = windows.get("initial")
        final = windows.get("final")
        if isinstance(initial, Mapping) and isinstance(final, Mapping):
            try:
                first_indices = set(initial.get("raw_indices", ()))
                last_indices = set(final.get("raw_indices", ()))
                if (first_indices & last_indices or
                        max(first_indices, default=-1) >= min(last_indices, default=np.inf)):
                    blockers.append("Initial and final stability windows overlap or are out of order.")
            except (TypeError, ValueError):
                blockers.append("Stability window indices are invalid.")
        for kind in ("initial", "final"):
            window = windows.get(kind)
            if isinstance(window, Mapping):
                try:
                    dataset._validate_stability_window_is_current(
                        kind, window,
                        require_unchanged_signal=True,
                        allow_normalized_signal=(
                            dataset.metadata.get("normalization_source") == "stability_windows"
                        ),
                    )
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    blockers.append(str(exc))

    if dataset.metadata.get("signal_type") == "normalized" and signal.size:
        initial_values = _selected_window_values(dataset, "initial")
        final_values = _selected_window_values(dataset, "final")
        initial_mean = float(np.mean(initial_values[1])) if initial_values else float(signal[0])
        final_mean = float(np.mean(final_values[1])) if final_values else float(signal[-1])
        first_value, last_value = float(signal[0]), float(signal[-1])
        if reference is not None:
            # Fixed plateau calibration belongs to the full normalization view,
            # not to endpoints of a subsequently truncated fit interval.
            initial_mean, final_mean = (0.0, 1.0) if reference["mode"] == "progress" else (1.0, 0.0)
            endpoints = reference["endpoints"]
            if isinstance(endpoints, Mapping):
                first_value, last_value = float(endpoints["initial"]), float(endpoints["final"])
            else:
                # Old records have no full-view endpoint snapshot. Retained
                # ends can no longer certify the original endpoint after crop.
                changed_rows = any(r.operation in {
                    "crop", "apply_start_index", "mask_points", "outlier_mask", "adaptive_downsample"
                } for r in dataset.history[reference["history_index"] + 1:])
                if changed_rows:
                    first_value, last_value = initial_mean, final_mean
                    warnings.append("Legacy normalization reference retained; original endpoint check is unavailable after selection. Review the original plateaus.")
            for kind in ("initial", "final"):
                w = reference["windows"][kind]
                retained = int(np.count_nonzero(dataset.mask[list(w["raw_indices"])]))
                if retained < w["n_points"]:
                    warnings.append(f"Fixed {kind} normalization reference retained ({retained}/{w['n_points']} plateau rows in fit interval); no re-normalization. The fit interval need not start at 0 or end at 1.")
            if time[0] > 0:
                warnings.append("Fit starts after t=0; the selected time origin is preserved. Missing early transient data can weaken D/k identification.")
            if float(signal[-1]) < 0.85:
                warnings.append("Fit interval ends before the final plateau; the fixed reference is retained, but late-time information is reduced.")
        # New records keep this check on the original full normalization view,
        # so wrong interior windows cannot be hidden by a later crop.
        if abs(initial_mean) > 0.15 or abs(first_value) > 0.15:
            blockers.append("Normalized initial plateau must be approximately 0.")
        if abs(final_mean - 1.0) > 0.15 or abs(last_value - 1.0) > 0.15:
            blockers.append("Normalized final plateau must be approximately 1.")

    if time.size >= 3 and np.all(np.diff(time) > 0):
        for kind in ("initial", "final"):
            selected = _selected_window_values(dataset, kind)
            if selected is None:
                if reference is not None:
                    # A missing calibration platform is not the fast transient
                    # at the first retained fit point.
                    continue
                count = max(2, min(5, time.size // 10))
                edge = slice(0, count) if kind == "initial" else slice(-count, None)
                local_time, local_signal = time[edge], signal[edge]
            else:
                local_time, local_signal = selected
            if local_time.size >= 2:
                duration = float(local_time[-1] - local_time[0])
                slope = float(np.polyfit(local_time - local_time[0], local_signal, 1)[0])
                if abs(slope * duration) > 0.05 * max(float(np.ptp(signal)), 1.0e-12):
                    warnings.append(f"Mild drift in the {kind} plateau; review the window.")
            if kind == "final" and (
                local_time.size < 5 or local_time[-1] - local_time[0] < 0.03 * (time[-1] - time[0])
            ):
                warnings.append("Final plateau is short; confirm its stability.")
        if signal.size >= 7:
            fraction = (time[1:-1] - time[:-2]) / (time[2:] - time[:-2])
            residual = signal[1:-1] - (signal[:-2] + fraction * (signal[2:] - signal[:-2]))
            residual_noise = _robust_sigma(residual)
            if float(np.max(np.abs(residual - np.median(residual)))) > max(
                6.0 * residual_noise, 0.15 * float(np.ptp(signal))
            ):
                warnings.append("Suspect outlier or spike in the signal; inspect raw points.")
    if 1.0 - float(np.mean(dataset.mask)) > 0.20:
        warnings.append("Excessive mask: more than 20% of raw points are excluded.")
    return FitQualityReport(not blockers, tuple(dict.fromkeys(blockers)), tuple(dict.fromkeys(warnings)))


__all__ = [
    "FitQualityReport", "StabilityWindowSuggestion", "evaluate_fit_quality",
    "suggest_stability_window",
]
