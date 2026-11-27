"""Model suggestions derived from a DCT result.

The assistant is intentionally diagnostic-only: it does not instantiate,
configure, or fit a physical model, and it never assigns a DCT peak to a
chemical species.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10
from numbers import Real
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from pyecr.config import SampleGeometry
from pyecr.dct.core import CandidateTimescale, DCTResult
from pyecr.models.ids import ModelID

__all__ = [
    "ClassicalBiAssessment",
    "DCTModelSuggestion",
    "LambdaStabilityAssessment",
    "suggest_models_from_dct",
]


@dataclass(frozen=True, slots=True)
class LambdaStabilityAssessment:
    """Peak persistence under the stored regularization-lambda perturbations."""

    status: Literal["stable", "unstable", "not_assessed"]
    snapshot_count: int
    selected_peak_count: int
    stable_peak_count: int
    stable_peak_positions_s: tuple[float, ...]
    max_position_shift_decades: tuple[float, ...]
    max_area_relative_change: tuple[float, ...]
    count_consistent: bool
    artifact_risk: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClassicalBiAssessment:
    """Qualitative compatibility with a classical two-timescale Bi criterion."""

    tau1_s: float
    tau2_s: float
    tau1_over_tau2: float
    possible_bi_values: tuple[float, float]
    diffusion_consistency: float
    mixed_control_consistency: float
    surface_control_consistency: float
    active_dimension_count: int
    interpretation: str
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DCTModelSuggestion:
    """Read-only model suggestions and their supporting diagnostics."""

    recommended_model_ids: tuple[ModelID, ...]
    recommendation: str
    selected_peak_count: int
    dominant_peak_count: int
    stable_peak_count: int
    stable_peak_positions_s: tuple[float, ...]
    lambda_stability: LambdaStabilityAssessment
    artifact_risk: bool
    artifact_reasons: tuple[str, ...]
    classical_bi_assessment: ClassicalBiAssessment | None
    geometry_summary: Mapping[str, object]
    chemical_species_assignment: bool
    warnings: tuple[str, ...]
    reasoning: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "recommended_model_ids", tuple(self.recommended_model_ids)
        )
        object.__setattr__(
            self, "stable_peak_positions_s", tuple(self.stable_peak_positions_s)
        )
        object.__setattr__(self, "artifact_reasons", tuple(self.artifact_reasons))
        object.__setattr__(
            self, "geometry_summary", MappingProxyType(dict(self.geometry_summary))
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reasoning", tuple(self.reasoning))
        if self.chemical_species_assignment:
            raise ValueError("DCT suggestions must not assign chemical species")


def suggest_models_from_dct(
    result: DCTResult,
    geometry: SampleGeometry | None,
    *,
    use_classical_bi: bool = False,
    minimum_dominant_area: float = 0.10,
    position_stability_decades: float = 0.30,
    area_relative_tolerance: float = 0.50,
    surface_bi_threshold: float = 0.03,
    diffusion_bi_threshold: float = 30.0,
) -> DCTModelSuggestion:
    """Suggest model families without changing the DCT result or fit settings.

    A peak is considered stable only when it persists in the stored neighboring
    lambda solutions, stays within ``position_stability_decades``, and changes
    area by no more than ``area_relative_tolerance``.  The optional classical
    Bi assessment is a qualitative compatibility check, not a peak assignment.
    """

    if not isinstance(result, DCTResult):
        raise TypeError("result must be a DCTResult")
    if geometry is not None and not isinstance(geometry, SampleGeometry):
        raise TypeError("geometry must be a SampleGeometry")
    if geometry is not None: geometry.validate()
    minimum_area = _unit_interval(
        "minimum_dominant_area", minimum_dominant_area, include_zero=True
    )
    position_tolerance = _positive_finite(
        "position_stability_decades", position_stability_decades
    )
    area_tolerance = _nonnegative_finite(
        "area_relative_tolerance", area_relative_tolerance
    )
    surface_threshold = _positive_finite(
        "surface_bi_threshold", surface_bi_threshold
    )
    diffusion_threshold = _positive_finite(
        "diffusion_bi_threshold", diffusion_bi_threshold
    )
    if surface_threshold >= 1.0:
        raise ValueError("surface_bi_threshold must be smaller than 1")
    if diffusion_threshold <= 1.0:
        raise ValueError("diffusion_bi_threshold must be greater than 1")

    dominant_candidates = _dominant_candidates(
        result.candidate_timescales, minimum_area
    )
    stability = _assess_lambda_stability(
        result,
        dominant_candidates,
        minimum_area=minimum_area,
        position_tolerance=position_tolerance,
        area_tolerance=area_tolerance,
    )

    artifact_reasons = list(stability.reasons)
    if any("DCT_GCV_BOUNDARY" in item for item in result.resolution_warnings):
        artifact_reasons.append("GCV_LAMBDA_AT_SEARCH_BOUNDARY")
    artifact_reasons = list(dict.fromkeys(artifact_reasons))
    artifact_risk = stability.artifact_risk or bool(
        "GCV_LAMBDA_AT_SEARCH_BOUNDARY" in artifact_reasons
    )

    stable_count = stability.stable_peak_count
    if stable_count >= 2:
        recommended = (ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK)
        recommendation = (
            "Legacy trial list: M2/M3 may be tested explicitly. Multiple peaks "
            "do not determine process count; one diffusion process has multiple modes."
        )
        reasoning = [
            "Multiple stable time scales can include eigenmodes of a single process; they are not independent process evidence.",
            "M2/M3 are trial model families only; no DCT peak is assigned to a "
            "chemical species.",
        ]
    else:
        recommended = (ModelID.M0_SINGLE_D, ModelID.M1_SINGLE_DK)
        recommendation = (
            "Legacy trial list: M0/M1 may be tested explicitly; a single resolved "
            "time scale does not establish the number of physical processes."
            if stable_count == 1
            else "No lambda-stable multi-peak structure was established; use M0/M1 "
            "only as provisional trials and review the DCT settings."
        )
        reasoning = [
            "Resolved peak count never determines physical process count.",
            "This suggestion ranks model trials and is not evidence for a chemical "
            "species assignment.",
        ]

    warnings = list(result.resolution_warnings)
    if artifact_risk:
        warnings.append(
            "DCT_ARTIFACT_RISK: peak count, position, or area is not robust to "
            "regularization lambda, or the lambda search boundary was reached."
        )
        reasoning.append(
            "Unstable lambda behavior reduces confidence in using peak count for "
            "model selection."
        )

    classical_assessment: ClassicalBiAssessment | None = None
    if use_classical_bi and geometry is not None:
        stable_candidates = _stable_candidates(
            dominant_candidates, stability.stable_peak_positions_s
        )
        if len(stable_candidates) >= 2:
            classical_assessment = _classical_bi_assessment(
                stable_candidates,
                active_dimension_count=sum(geometry.active_dimensions),
                surface_threshold=surface_threshold,
                diffusion_threshold=diffusion_threshold,
            )
            reasoning.append(classical_assessment.interpretation)
        else:
            warnings.append(
                "DCT_BI_NOT_ASSESSED: at least two lambda-stable candidate time "
                "scales are required for tau1/tau2."
            )

    if use_classical_bi and geometry is None:
        warnings.append('LEGACY_HEURISTIC_UNAVAILABLE: no physical geometry; whole time-scale inversion remains available.')
    active_count = 0 if geometry is None else sum(geometry.active_dimensions)
    geometry_summary: Mapping[str, object] = {
        "full_dims_cm": None if geometry is None else tuple(float(value) for value in geometry.full_dims_cm),
        "half_dims_cm": None if geometry is None else tuple(float(value) / 2.0 for value in geometry.full_dims_cm),
        "active_dimensions": None if geometry is None else tuple(geometry.active_dimensions),
        "active_dimension_count": active_count,
        "classical_bi_geometry_scope": (
            "legacy_heuristic_1d"
            if active_count == 1
            else "unavailable" if geometry is None else "legacy_heuristic_multi_axis"
        ),
    }

    return DCTModelSuggestion(
        recommended_model_ids=recommended,
        recommendation=recommendation,
        selected_peak_count=len(result.candidate_timescales),
        dominant_peak_count=len(dominant_candidates),
        stable_peak_count=stable_count,
        stable_peak_positions_s=stability.stable_peak_positions_s,
        lambda_stability=stability,
        artifact_risk=artifact_risk,
        artifact_reasons=tuple(artifact_reasons),
        classical_bi_assessment=classical_assessment,
        geometry_summary=geometry_summary,
        chemical_species_assignment=False,
        warnings=tuple(dict.fromkeys(warnings)),
        reasoning=tuple(reasoning),
    )


def _dominant_candidates(
    candidates: tuple[CandidateTimescale, ...],
    minimum_area: float,
) -> tuple[CandidateTimescale, ...]:
    retained = tuple(candidate for candidate in candidates if candidate.area >= minimum_area)
    if retained or not candidates:
        return retained
    return (max(candidates, key=lambda candidate: candidate.area),)


def _assess_lambda_stability(
    result: DCTResult,
    selected: tuple[CandidateTimescale, ...],
    *,
    minimum_area: float,
    position_tolerance: float,
    area_tolerance: float,
) -> LambdaStabilityAssessment:
    raw_snapshots = result.diagnostics.get("lambda_peak_snapshots", ())
    snapshots = _validated_snapshots(raw_snapshots, minimum_area=minimum_area)
    selected_sorted = tuple(sorted(selected, key=lambda candidate: candidate.position_s))
    selected_count = len(selected_sorted)

    if len(snapshots) < 3:
        reason = "LAMBDA_STABILITY_NOT_ASSESSED"
        return LambdaStabilityAssessment(
            status="not_assessed",
            snapshot_count=len(snapshots),
            selected_peak_count=selected_count,
            stable_peak_count=0,
            stable_peak_positions_s=(),
            max_position_shift_decades=(),
            max_area_relative_change=(),
            count_consistent=False,
            artifact_risk=True,
            reasons=(reason,),
        )

    count_consistent = all(len(snapshot) == selected_count for snapshot in snapshots)
    max_shifts: list[float] = []
    max_area_changes: list[float] = []
    stable_positions: list[float] = []
    position_unstable = False
    area_unstable = False

    for candidate in selected_sorted:
        shifts: list[float] = []
        area_changes: list[float] = []
        for snapshot in snapshots:
            if not snapshot:
                shifts.append(float("inf"))
                area_changes.append(float("inf"))
                continue
            match_position, match_area = min(
                snapshot,
                key=lambda peak: abs(log10(peak[0] / candidate.position_s)),
            )
            shifts.append(abs(log10(match_position / candidate.position_s)))
            denominator = max(candidate.area, np.finfo(np.float64).eps)
            area_changes.append(abs(match_area - candidate.area) / denominator)
        max_shift = max(shifts)
        max_area_change = max(area_changes)
        max_shifts.append(max_shift)
        max_area_changes.append(max_area_change)
        peak_is_stable = (
            max_shift <= position_tolerance and max_area_change <= area_tolerance
        )
        if peak_is_stable:
            stable_positions.append(candidate.position_s)
        position_unstable = position_unstable or max_shift > position_tolerance
        area_unstable = area_unstable or max_area_change > area_tolerance

    reasons: list[str] = []
    if not count_consistent:
        reasons.append("LAMBDA_PEAK_COUNT_UNSTABLE")
    if position_unstable:
        reasons.append("LAMBDA_PEAK_POSITION_UNSTABLE")
    if area_unstable:
        reasons.append("LAMBDA_PEAK_AREA_UNSTABLE")
    stable = count_consistent and len(stable_positions) == selected_count
    return LambdaStabilityAssessment(
        status="stable" if stable else "unstable",
        snapshot_count=len(snapshots),
        selected_peak_count=selected_count,
        stable_peak_count=len(stable_positions),
        stable_peak_positions_s=tuple(stable_positions),
        max_position_shift_decades=tuple(max_shifts),
        max_area_relative_change=tuple(max_area_changes),
        count_consistent=count_consistent,
        artifact_risk=not stable,
        reasons=tuple(reasons),
    )


def _validated_snapshots(
    raw_snapshots: object,
    *,
    minimum_area: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(raw_snapshots, (tuple, list)):
        return ()
    snapshots: list[tuple[tuple[float, float], ...]] = []
    for item in raw_snapshots:
        if not isinstance(item, Mapping):
            return ()
        positions = item.get("peak_positions_s", ())
        areas = item.get("peak_areas", ())
        if not isinstance(positions, (tuple, list)) or not isinstance(
            areas, (tuple, list)
        ):
            return ()
        if len(positions) != len(areas):
            return ()
        peaks: list[tuple[float, float]] = []
        for position, area in zip(positions, areas, strict=True):
            if not _is_positive_finite(position) or not _is_nonnegative_finite(area):
                return ()
            if float(area) >= minimum_area:
                peaks.append((float(position), float(area)))
        snapshots.append(tuple(sorted(peaks)))
    return tuple(snapshots)


def _stable_candidates(
    candidates: tuple[CandidateTimescale, ...],
    stable_positions: tuple[float, ...],
) -> tuple[CandidateTimescale, ...]:
    if not stable_positions:
        return ()
    return tuple(
        candidate
        for candidate in candidates
        if any(
            np.isclose(candidate.position_s, position, rtol=1.0e-12, atol=0.0)
            for position in stable_positions
        )
    )


def _classical_bi_assessment(
    candidates: tuple[CandidateTimescale, ...],
    *,
    active_dimension_count: int,
    surface_threshold: float,
    diffusion_threshold: float,
) -> ClassicalBiAssessment:
    strongest = sorted(candidates, key=lambda candidate: candidate.area, reverse=True)[:2]
    tau1, tau2 = sorted(
        (strongest[0].position_s, strongest[1].position_s), reverse=True
    )
    ratio = float(tau1 / tau2)
    possible_bi = (ratio, 1.0 / ratio)
    diffusion_score = max(
        _diffusion_consistency(value, diffusion_threshold) for value in possible_bi
    )
    surface_score = max(
        _surface_consistency(value, surface_threshold) for value in possible_bi
    )
    mixed_score = max(
        _mixed_consistency(value, surface_threshold, diffusion_threshold)
        for value in possible_bi
    )
    limitations = [
        "The time-scale ratio cannot identify a chemical species.",
        "No automatic assignment of either peak to diffusion, surface exchange, "
        "or another mechanism is made.",
        "The reciprocal Bi alternatives reflect the unknown ordering of the two "
        "candidate mechanisms.",
    ]
    if active_dimension_count > 1:
        limitations.append(
            "For multi-axis geometry, each active half-dimension has its own Bi; "
            "the ratio assessment is qualitative rather than axis-specific."
        )
    return ClassicalBiAssessment(
        tau1_s=tau1,
        tau2_s=tau2,
        tau1_over_tau2=ratio,
        possible_bi_values=possible_bi,
        diffusion_consistency=diffusion_score,
        mixed_control_consistency=mixed_score,
        surface_control_consistency=surface_score,
        active_dimension_count=active_dimension_count,
        interpretation=(
            "Legacy heuristic (legacy-peak-ratio/1): the tau1/tau2 check measures compatibility with limiting "
            "or mixed-control regimes only; it does not establish mechanism or "
            "chemical identity. possible_bi_values and reciprocals are historical heuristic numbers, not quantitative Bi."
        ),
        limitations=tuple(limitations),
    )


def _diffusion_consistency(bi: float, upper: float) -> float:
    if bi <= 1.0:
        return 0.0
    if bi >= upper:
        return 1.0
    return float(log10(bi) / log10(upper))


def _surface_consistency(bi: float, lower: float) -> float:
    if bi >= 1.0:
        return 0.0
    if bi <= lower:
        return 1.0
    return float(log10(1.0 / bi) / log10(1.0 / lower))


def _mixed_consistency(bi: float, lower: float, upper: float) -> float:
    mixed_low = max(lower, 0.1)
    mixed_high = min(upper, 10.0)
    if mixed_low <= bi <= mixed_high:
        return 1.0
    if lower < bi < mixed_low:
        return float(log10(bi / lower) / log10(mixed_low / lower))
    if mixed_high < bi < upper:
        return float(log10(upper / bi) / log10(upper / mixed_high))
    return 0.0


def _is_positive_finite(value: object) -> bool:
    return _is_real_finite(value) and float(value) > 0.0


def _is_nonnegative_finite(value: object) -> bool:
    return _is_real_finite(value) and float(value) >= 0.0


def _is_real_finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


def _finite_scalar(name: str, value: float) -> float:
    if not _is_real_finite(value):
        raise TypeError(f"{name} must be a finite real number")
    return float(value)


def _positive_finite(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _unit_interval(name: str, value: float, *, include_zero: bool) -> float:
    result = _finite_scalar(name, value)
    valid_lower = result >= 0.0 if include_zero else result > 0.0
    if not valid_lower or result > 1.0:
        bracket = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {bracket}")
    return result
