"""Read-only time-origin and gas-flush guidance for an ECR analysis.

The assistant never edits a dataset, processing recipe, or analysis config.
An accepted time origin is applied separately through ``set_manual_t0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from pyecr.boundary.composition import (
    BoundaryComposedSingleEcrModel,
    dominant_material_time_constant,
)
from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.io.dataset import EcrDataset
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.models.single import SingleEcrContext, SingleEcrModel

__all__ = [
    "FlushAssessment",
    "T0Suggestion",
    "assess_flush",
    "suggest_t0",
    "validate_boundary_fit_coverage",
]


@dataclass(frozen=True, slots=True)
class FlushAssessment:
    """Interpret a flush time against the fitted material's dominant mode.

    ``tau_material_s`` is the slowest whole-specimen modal characteristic time:
    one divided by the sum of the first active-axis modal decay rates. It is
    not inferred from the boundary-corrected curve's 63.2% crossing.
    """

    status: Literal["available", "not_applicable", "unsupported", "unavailable"]
    tau_material_s: float | None
    tau_flush_s: float | None
    flush_to_material_ratio: float | None
    severity: str
    explanation: str


@dataclass(frozen=True, slots=True)
class T0Suggestion:
    """A reviewable onset estimate in the *current processed* seconds axis."""

    t0_s: float
    onset_index: int
    suggested_start_index: int
    requires_pre_onset_crop: bool
    score: float
    confidence_class: Literal["high", "medium", "low"]
    reason: str
    tau_material_s: float | None = None
    tau_flush_s: float | None = None
    flush_to_material_ratio: float | None = None
    boundary_status: str = "unavailable"


def suggest_t0(
    dataset: EcrDataset,
    config: AnalysisConfig | None = None,
    fit_result: FitResult | None = None,
) -> T0Suggestion | None:
    """Suggest the first sustained departure from the initial signal level.

    The proposal is a heuristic onset, estimated by extrapolating the early
    rise to the initial level. Neither source nor processed arrays are changed.
    The caller must explicitly apply a ``set_manual_t0`` ProcessingStep. If
    earlier retained rows become negative after the shift, they must also be
    removed by a separately confirmed crop or mask before fitting.
    """

    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if config is not None and not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig or None")
    if fit_result is not None and not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult or None")
    if dataset.metadata.get("time_unit", "s") != "s":
        return None
    time = np.asarray(dataset.processed_time, dtype=np.float64)
    signal = np.asarray(dataset.processed_signal, dtype=np.float64)
    if (
        time.size < 8
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(signal))
        or np.any(np.diff(time) <= 0.0)
    ):
        return None

    n = time.size
    baseline_count = min(max(3, int(np.ceil(0.03 * n))), 25)
    baseline = float(np.median(signal[:baseline_count]))
    final = float(np.median(signal[-baseline_count:]))
    signed_amplitude = final - baseline
    amplitude = abs(signed_amplitude)
    if amplitude <= np.finfo(float).eps * max(1.0, abs(baseline), abs(final)):
        return None
    baseline_noise = _robust_sigma(signal[:baseline_count])
    derivative_noise = _robust_sigma(np.diff(signal[: max(6, baseline_count + 2)])) / np.sqrt(2.0)
    noise = max(baseline_noise, derivative_noise)
    if amplitude <= 8.0 * noise:
        return None

    direction = 1.0 if signed_amplitude > 0.0 else -1.0
    progress = direction * (signal - baseline) / amplitude
    threshold = max(0.05, 3.0 * noise / amplitude)
    if threshold >= 0.3:
        return None
    first_crossing: int | None = None
    for index in range(baseline_count, n - 2):
        if progress[index] >= threshold and np.count_nonzero(progress[index:index + 3] >= threshold) >= 2:
            first_crossing = index
            break
    if first_crossing is None or first_crossing < baseline_count + 2:
        # With no pre-step observations, the origin cannot be separated from
        # the material's early response on signal evidence alone.
        return None
    if first_crossing >= 5:
        baseline_rates = np.diff(signal[:first_crossing]) / np.diff(time[:first_crossing])
        baseline_drift_over_trace = abs(float(np.median(baseline_rates))) * float(time[-1] - time[0])
        if baseline_drift_over_trace > 0.5 * amplitude:
            # A trace with a continuing linear drift has no identifiable
            # gas-step onset for this assistant to mark.
            return None

    # A local early-rise extrapolation is less biased than naming the 5%
    # crossing itself as the physical step time. This remains a suggestion.
    early_end = first_crossing
    while early_end < n and progress[early_end] <= 0.35 and early_end - first_crossing < 12:
        early_end += 1
    early_end = max(early_end, min(n, first_crossing + 3))
    early_time = time[first_crossing:early_end]
    early_progress = progress[first_crossing:early_end]
    prior_time = float(time[first_crossing - 1])
    if early_time.size >= 3 and float(np.ptp(early_progress)) > 0.05:
        slope, intercept = np.polyfit(early_time - early_time[0], early_progress, 1)
        origin = float(early_time[0] - intercept / slope) if slope > 0.0 else prior_time
        predicted = slope * (early_time - early_time[0]) + intercept
        linearity = max(0.0, 1.0 - float(np.sqrt(np.mean((early_progress - predicted) ** 2))) / 0.2)
    else:
        origin = prior_time
        linearity = 0.5
    origin = float(np.clip(origin, time[0], time[first_crossing]))
    retained_index = min(int(np.searchsorted(time, origin, side="left")), n - 1)
    raw_index = int(np.flatnonzero(dataset.mask)[retained_index])
    requires_crop = bool(np.any(time < origin))
    signal_to_noise = amplitude / max(noise, np.finfo(float).eps)
    score = float(np.clip(0.35 + 0.35 * min(signal_to_noise / 20.0, 1.0) + 0.3 * linearity, 0.0, 1.0))
    confidence: Literal["high", "medium", "low"] = (
        "high" if score >= 0.75 else "medium" if score >= 0.55 else "low"
    )
    reason = (
        f"Estimated onset by extrapolating the first sustained rise above "
        f"{threshold:.1%} of the initial-to-final amplitude; "
        f"signal/noise ≈ {signal_to_noise:.2g}. "
        "Review against the measured gas step."
    )
    if requires_crop:
        reason += " Earlier retained times would be negative; explicitly crop or mask them before Fit."

    assessment = (
        assess_flush(config, fit_result)
        if config is not None and fit_result is not None
        else None
    )
    return T0Suggestion(
        t0_s=origin,
        onset_index=first_crossing,
        suggested_start_index=raw_index,
        requires_pre_onset_crop=requires_crop,
        score=score,
        confidence_class=confidence,
        reason=reason,
        tau_material_s=None if assessment is None else assessment.tau_material_s,
        tau_flush_s=None if assessment is None else assessment.tau_flush_s,
        flush_to_material_ratio=None if assessment is None else assessment.flush_to_material_ratio,
        boundary_status="unavailable" if assessment is None else assessment.status,
    )


def assess_flush(config: AnalysisConfig, fit_result: FitResult) -> FlushAssessment:
    """Explain tau_flush/tau_material using the production single-process kernel.

    Threshold labels are screening heuristics, not model-adequacy or
    identifiability decisions. M2/M3 composition remains unsupported.
    """

    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult")
    if config.model.model_id in {ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK}:
        return FlushAssessment(
            "unsupported", None, None, None, "Unsupported",
            "M2/M3 non-ideal gas-boundary composition is Unsupported; no dual-process flush mathematics is defined.",
        )
    boundary = config.boundary
    if boundary is None or boundary.mode != "exponential_flush":
        mode = "ideal_step" if boundary is None else boundary.mode
        return FlushAssessment(
            "not_applicable", None, None, None, "not_applicable",
            f"{mode} has no exponential tau_flush; tau_flush/tau_material is not defined.",
        )
    try:
        tau_material = _material_time_constant(config, fit_result)
    except (TypeError, ValueError, OverflowError, ArithmeticError, RuntimeError) as exc:
        return FlushAssessment(
            "unavailable", None, boundary.flush_time_s, None, "unavailable",
            f"Material characteristic time is unavailable: {exc}",
        )
    tau_flush = float(boundary.flush_time_s)  # validated by BoundaryConfig
    ratio = tau_flush / tau_material
    if ratio <= 0.1:
        severity = "no_correction_needed"
        advice = "flush is short relative to the dominant material mode"
    elif ratio >= 3.0:
        severity = "severe_limitation"
        advice = "flush is slower than the dominant material mode"
    else:
        severity = "correction_recommended"
        advice = "flush and material times are comparable"
    return FlushAssessment(
        "available", tau_material, tau_flush, ratio, severity,
        "tau_material = 1 / sum(first active-axis modal decay rates) from the "
        "production M0/M1 material kernel; "
        f"tau_flush/tau_material = {ratio:.4g}. {advice.capitalize()}. "
        "The 0.1 and 3 thresholds are heuristic screening only; faster modes "
        "can matter even when this ratio is small.",
    )


def validate_boundary_fit_coverage(dataset: EcrDataset, config: AnalysisConfig) -> None:
    """Reject any fit time outside measured boundary coverage, with no extrapolation."""

    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")
    boundary = config.boundary
    if boundary is None or boundary.mode == "ideal_step":
        return
    if config.model.model_id in {ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK}:
        raise ValueError("M2/M3 non-ideal boundary composition is Unsupported")
    if boundary.mode != "measured_history":
        return
    boundary.validate()
    if dataset.metadata.get("time_unit", "s") != "s":
        raise ValueError("measured boundary coverage requires fit time in seconds")
    fit_time = np.asarray(dataset.processed_time, dtype=np.float64)
    if fit_time.size == 0 or not np.all(np.isfinite(fit_time)):
        raise ValueError("fit time must be finite for measured boundary coverage")
    if (
        float(np.min(fit_time)) < boundary.measured_time_s[0]
        or float(np.max(fit_time)) > boundary.measured_time_s[-1]
    ):
        raise ValueError(
            "fit time is outside measured boundary coverage; the complete "
            "fit interval must be covered and extrapolation is not allowed"
        )


def _material_time_constant(config: AnalysisConfig, fit_result: FitResult) -> float:
    model = ModelFactory().create(config)
    material = model.material_model if isinstance(model, BoundaryComposedSingleEcrModel) else model
    if not isinstance(material, SingleEcrModel):
        raise TypeError("production single-process model is unavailable")
    d = fit_result.params.get("D_cm2_s", config.model.fixed_parameters.get("D_cm2_s"))
    if d is None or not isfinite(float(d)) or float(d) <= 0.0:
        raise ValueError("fitted D_cm2_s must be positive and finite")
    theta = {"log10_D": float(np.log10(float(d)))}
    if material.model_id is ModelID.M1_SINGLE_DK:
        k = fit_result.params.get("k_cm_s", config.model.fixed_parameters.get("k_cm_s"))
        if k is None or not isfinite(float(k)) or float(k) <= 0.0:
            raise ValueError("fitted k_cm_s must be positive and finite")
        theta["log10_k"] = float(np.log10(float(k)))
    context = SingleEcrContext(
        term_tol=config.solver.relative_tolerance,
        max_terms=config.solver.max_terms,
    )
    prediction = material.evaluate(np.array([0.0]), theta, context)
    return dominant_material_time_constant(prediction.kernel_result.axis_results)


def _robust_sigma(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))
