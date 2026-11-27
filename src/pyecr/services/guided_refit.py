"""Explicit, provenance-preserving M1 one-parameter guided refits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import isfinite, log10, sqrt
from typing import Mapping

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.models.ids import ModelID
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.report.snapshot import AnalysisSnapshot
from pyecr.services.diagnostics_service import DiagnosticsPresentation
from pyecr.uncertainty.local import estimate_local_uncertainty

__all__ = [
    "FitHistoryEntry",
    "GuidedRefitEligibility",
    "assess_guided_refit",
    "build_guided_refit_config",
    "fit_result_id",
    "refit_comparison",
]


@dataclass(frozen=True, slots=True)
class FitHistoryEntry:
    """An earlier fit plus its exact config and optional immutable report snapshot."""

    fit_id: str
    analysis_config: AnalysisConfig
    fit_result: FitResult
    report_snapshot: AnalysisSnapshot | None = None

    def __post_init__(self) -> None:
        if self.fit_id != fit_result_id(self.fit_result):
            raise ValueError("fit history ID does not match its FitResult")
        self.analysis_config.validate()
        if self.report_snapshot is not None and (
            fit_result_id(self.report_snapshot.fit_result) != self.fit_id
            or self.report_snapshot.analysis_config != self.analysis_config
        ):
            raise ValueError("fit history snapshot must capture the same fit and config")

    def to_dict(self) -> dict[str, object]:
        return {
            "fit_id": self.fit_id,
            "analysis_config": self.analysis_config.to_dict(),
            "fit_result": self.fit_result.to_dict(),
            "report_snapshot": (
                None if self.report_snapshot is None else self.report_snapshot.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FitHistoryEntry:
        if not isinstance(value, Mapping):
            raise TypeError("fit history entry must be a mapping")
        config = value.get("analysis_config")
        fit = value.get("fit_result")
        snapshot = value.get("report_snapshot")
        if not isinstance(config, Mapping) or not isinstance(fit, Mapping):
            raise TypeError("fit history requires config and fit mappings")
        return cls(
            fit_id=str(value.get("fit_id", "")),
            analysis_config=AnalysisConfig.from_dict(config),
            fit_result=FitResult.from_dict(fit),
            report_snapshot=(
                None if snapshot is None else AnalysisSnapshot.from_dict(snapshot)
            ),
        )


@dataclass(frozen=True, slots=True)
class GuidedRefitEligibility:
    available: bool
    reliable_parameter: str | None = None
    weak_parameter: str | None = None
    reason: str = ""
    evidence: tuple[str, ...] = ()


def fit_result_id(fit: FitResult) -> str:
    """Content ID for provenance, independent of in-memory object identity."""

    payload = json.dumps(
        fit.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assess_guided_refit(
    config: AnalysisConfig | None,
    fit: FitResult | None,
    diagnostics: DiagnosticsPresentation | None,
) -> GuidedRefitEligibility:
    """Identify one well-localized and one weak M1 kinetic parameter.

    Parameter-level intervals and bound contact are used here. A global model
    credibility gate can mark both display rows unreliable and does not by
    itself identify which parameter deserves to be fixed.
    """

    if config is None or fit is None or diagnostics is None:
        return GuidedRefitEligibility(False, reason="Current M1 Fit and Diagnostics are required.")
    if config.model_id is not ModelID.M1_SINGLE_DK:
        return GuidedRefitEligibility(False, reason="Guided Refit is supported only for M1.")
    names = ("D_cm2_s", "k_cm_s")
    if any(name not in fit.params for name in names):
        return GuidedRefitEligibility(False, reason="Both kinetic parameters must be free in the parent fit.")
    rows = {row.name: row for row in diagnostics.parameters}
    if any(name not in rows for name in names):
        return GuidedRefitEligibility(False, reason="Parameter diagnostics are incomplete.")
    verdict: dict[str, str] = {}
    evidence: list[str] = []
    for name in names:
        row = rows[name]
        if row.ci_lower is None or row.ci_upper is None or row.ci_lower <= 0 or row.ci_upper <= 0:
            verdict[name] = "weak"
            evidence.append(f"{name}: 95% CI unavailable")
            continue
        width = log10(row.ci_upper / row.ci_lower)
        evidence.append(f"{name}: 95% CI width {width:.3g} decades; bound_hit={row.bound_hit}")
        if row.bound_hit is True or width > 2.0:
            verdict[name] = "weak"
        elif row.bound_hit is False and width <= 1.0:
            verdict[name] = "reliable"
        else:
            verdict[name] = "ambiguous"
    reliable = [name for name in names if verdict[name] == "reliable"]
    weak = [name for name in names if verdict[name] == "weak"]
    if len(reliable) != 1 or len(weak) != 1:
        return GuidedRefitEligibility(
            False, reason="Guided Refit needs one localized and one weak kinetic parameter.",
            evidence=tuple(evidence),
        )
    return GuidedRefitEligibility(
        True, reliable[0], weak[0],
        f"{reliable[0]} is localized while {weak[0]} is weak; fixing the localized parameter can test the other parameter conditionally.",
        tuple(evidence),
    )


def build_guided_refit_config(
    parent: AnalysisConfig,
    parent_fit: FitResult,
    *,
    fixed_parameter: str,
    fixed_value: float,
    value_source: str,
    parent_fit_id: str,
) -> AnalysisConfig:
    """Build a new immutable M1 config, never a narrow-bound imitation of fixing."""

    if parent.model_id is not ModelID.M1_SINGLE_DK:
        raise ValueError("Guided Refit is supported only for M1")
    if fixed_parameter not in {"D_cm2_s", "k_cm_s"}:
        raise ValueError("fixed_parameter must be D_cm2_s or k_cm_s")
    if fixed_parameter not in parent_fit.params:
        raise ValueError("fixed parameter must be free in the parent FitResult")
    if value_source not in {"parent_fit", "manual"}:
        raise ValueError("value_source must be parent_fit or manual")
    if not isfinite(fixed_value) or fixed_value <= 0.0:
        raise ValueError("fixed_value must be positive and finite")
    if value_source == "parent_fit" and fixed_value != parent_fit.params[fixed_parameter]:
        raise ValueError("parent_fit value source must use the exact parent estimate")
    bounds = parent.parameter_bounds.get(fixed_parameter)
    if bounds is not None and not bounds[0] <= fixed_value <= bounds[1]:
        raise ValueError("fixed_value lies outside parameter bounds")
    free = tuple(name for name in parent_fit.params if name != fixed_parameter)
    model = replace(
        parent.model,
        free_parameters=free,
        fixed_parameters={**parent.model.fixed_parameters, fixed_parameter: float(fixed_value)},
        initial_values={
            name: value for name, value in parent.model.initial_values.items()
            if name != fixed_parameter
        },
    )
    model_specific = dict(parent.model_specific)
    model_specific["guided_refit"] = {
        "parent_fit_id": parent_fit_id,
        "fixed_parameter": fixed_parameter,
        "fixed_value": float(fixed_value),
        "value_source": value_source,
    }
    child = replace(parent, model=model, model_specific=model_specific)
    DEFAULT_MODEL_REGISTRY.validate_config(child.model)
    child.validate()
    return child


def refit_comparison(parent: FitResult, child: FitResult) -> dict[str, object]:
    """Metrics and conditional uncertainty for before/after reporting."""

    def summary(fit: FitResult) -> dict[str, object]:
        local = fit.diagnostics.get("local_uncertainty")
        if not isinstance(local, Mapping):
            local = {}
        if not local and fit.jacobian is not None:
            try:
                estimated = estimate_local_uncertainty(fit)
                local = {
                    "rank": estimated.rank,
                    "condition_number": (
                        estimated.condition_number
                        if isfinite(estimated.condition_number) else None
                    ),
                    "reliable_separation": estimated.reliable_separation,
                    "confidence_intervals": {
                        name: {
                            "estimate": interval.estimate if isfinite(interval.estimate) else None,
                            "lower": interval.lower if isfinite(interval.lower) else None,
                            "upper": interval.upper if isfinite(interval.upper) else None,
                        }
                        for name, interval in estimated.confidence_intervals.items()
                    },
                }
            except (ArithmeticError, RuntimeError, TypeError, ValueError):
                pass
        residual_mean = float(sum(fit.residuals) / fit.residuals.size)
        residual_sd = (
            sqrt(float(sum((value - residual_mean) ** 2 for value in fit.residuals)) / (fit.residuals.size - 1))
            if fit.residuals.size > 1 else 0.0
        )
        return {
            "SSE": fit.SSE,
            "RMSE": fit.RMSE,
            "R2": fit.R2,
            "confidence_intervals": dict(local.get("confidence_intervals", {})),
            "identifiability": {
                "rank": local.get("rank"),
                "condition_number": local.get("condition_number"),
                "reliable_separation": local.get("reliable_separation"),
            },
            "residual_mean": residual_mean,
            "residual_standard_deviation": residual_sd,
            "residual_max_abs": float(max(abs(value) for value in fit.residuals)),
        }

    return {"before": summary(parent), "after": summary(child)}
