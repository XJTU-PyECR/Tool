"""Information-criterion ranking with an explicit scientific reliability gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np

from pyecr.fitting.base import FitResult
from pyecr.uncertainty.local import (
    LocalUncertaintyResult,
    estimate_local_uncertainty,
)

__all__ = [
    "CandidateAssessment",
    "ModelSelectionConfig",
    "ModelSelectionResult",
    "ParameterBound",
    "select_model",
]


BoundSpace = Literal["coordinate", "physical"]
BOUND_POLICY_VERSION = "credibility-bound-proximity/1.0.0"
BoundInput = "ParameterBound | tuple[float, float]"
ModelKey = str | Enum


@dataclass(frozen=True, slots=True)
class ParameterBound:
    """One fitted parameter's bounds.

    ``space="coordinate"`` means log10 bounds for a parameter whose
    ``FitResult.param_space`` is ``"log"``.  ``space="physical"`` compares
    directly with the reported parameter value.
    """

    lower: float
    upper: float
    space: BoundSpace = "coordinate"

    def __post_init__(self) -> None:
        lower = _real_scalar("lower", self.lower, allow_infinite=True)
        upper = _real_scalar("upper", self.upper, allow_infinite=True)
        if lower >= upper:
            raise ValueError("parameter bound lower must be smaller than upper")
        if self.space not in {"coordinate", "physical"}:
            raise ValueError("bound space must be 'coordinate' or 'physical'")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class ModelSelectionConfig:
    """Thresholds and weights for ranking and reliability decisions."""

    residual_autocorrelation_limit: float = 0.30
    ci_width_decades_limit: float = 2.0
    correlation_limit: float = 0.98
    condition_number_limit: float = 1.0e12
    bound_proximity_fraction: float = 0.01
    aicc_weight: float = 0.60
    bic_weight: float = 0.40
    residual_penalty_weight: float = 4.0
    require_uncertainty: bool = True

    def __post_init__(self) -> None:
        _closed_interval(
            "residual_autocorrelation_limit",
            self.residual_autocorrelation_limit,
            0.0,
            1.0,
        )
        _positive_finite("ci_width_decades_limit", self.ci_width_decades_limit)
        _closed_interval("correlation_limit", self.correlation_limit, 0.0, 1.0)
        _positive_finite("condition_number_limit", self.condition_number_limit)
        _closed_interval(
            "bound_proximity_fraction",
            self.bound_proximity_fraction,
            0.0,
            0.5,
        )
        aicc_weight = _nonnegative_finite("aicc_weight", self.aicc_weight)
        bic_weight = _nonnegative_finite("bic_weight", self.bic_weight)
        if aicc_weight + bic_weight <= 0.0:
            raise ValueError("at least one information-criterion weight must be positive")
        _nonnegative_finite(
            "residual_penalty_weight",
            self.residual_penalty_weight,
        )
        if not isinstance(self.require_uncertainty, bool):
            raise TypeError("require_uncertainty must be boolean")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """Statistical evidence and Gate outcome for one candidate model."""

    model_id: str
    AICc: float
    BIC: float
    SSE: float
    delta_AICc: float
    delta_BIC: float
    statistical_score: float
    residual_lag1_autocorrelation: float
    condition_number: float | None
    max_abs_parameter_correlation: float | None
    ci_width_decades: Mapping[str, float]
    parameters_at_bounds: tuple[str, ...]
    parameters_outside_bounds: tuple[str, ...]
    failed_physical_checks: tuple[str, ...]
    reliable: bool
    gate_failures: tuple[str, ...]
    reasoning: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ci_width_decades",
            MappingProxyType(dict(self.ci_width_decades)),
        )


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    """Recommended statistical and reliability-qualified models."""

    best_statistical: str
    best_reliable: str | None
    ranking: tuple[str, ...]
    assessments: Mapping[str, CandidateAssessment]
    warnings: tuple[str, ...]
    reasoning: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assessments",
            MappingProxyType(dict(self.assessments)),
        )


def select_model(
    fit_results: Mapping[ModelKey, FitResult],
    *,
    uncertainties: Mapping[ModelKey, LocalUncertaintyResult] | None = None,
    parameter_bounds: Mapping[
        ModelKey,
        Mapping[str, ParameterBound | tuple[float, float]],
    ]
    | None = None,
    physical_checks: Mapping[ModelKey, Mapping[str, bool]] | None = None,
    key_parameters: Mapping[ModelKey, Sequence[str]] | None = None,
    config: ModelSelectionConfig | None = None,
) -> ModelSelectionResult:
    """Rank fitted models and apply a conservative scientific reliability Gate.

    The statistical score combines delta-AICc, delta-BIC, and excess residual
    lag-1 autocorrelation.  It never ranks by raw SSE alone.  ``best_reliable``
    is selected from candidates that additionally pass local-identifiability,
    confidence-width, correlation, bound, optimizer, and physical checks.

    If an uncertainty result is omitted but a fit stores a Jacobian, local
    uncertainty is calculated automatically.  With the default strict policy,
    a candidate lacking both is not allowed to become a trusted recommendation.
    """

    policy = ModelSelectionConfig() if config is None else config
    if not isinstance(policy, ModelSelectionConfig):
        raise TypeError("config must be a ModelSelectionConfig or None")
    fits = _normalise_fits(fit_results)
    uncertainty_map = _normalise_auxiliary(
        "uncertainties",
        uncertainties,
        model_ids=tuple(fits),
    )
    bounds_map = _normalise_auxiliary(
        "parameter_bounds",
        parameter_bounds,
        model_ids=tuple(fits),
    )
    physical_map = _normalise_auxiliary(
        "physical_checks",
        physical_checks,
        model_ids=tuple(fits),
    )
    key_map = _normalise_auxiliary(
        "key_parameters",
        key_parameters,
        model_ids=tuple(fits),
    )

    residual_correlations = {
        model_id: _lag1_autocorrelation(fit.residuals)
        for model_id, fit in fits.items()
    }
    delta_aicc = _metric_deltas({key: fit.AICc for key, fit in fits.items()})
    delta_bic = _metric_deltas({key: fit.BIC for key, fit in fits.items()})
    scores = {
        model_id: _statistical_score(
            delta_aicc[model_id],
            delta_bic[model_id],
            residual_correlations[model_id],
            policy,
        )
        for model_id in fits
    }
    used_sse_fallback = not any(np.isfinite(value) for value in scores.values())
    if used_sse_fallback:
        scores = _sse_fallback_scores(fits)

    assessments: dict[str, CandidateAssessment] = {}
    warnings: list[str] = []
    if used_sse_fallback:
        warnings.append(
            "INFORMATION_CRITERIA_UNAVAILABLE: statistical ordering fell back to "
            "SSE because neither AICc nor BIC could rank the candidates."
        )
    for model_id, fit in fits.items():
        uncertainty, uncertainty_note = _resolve_uncertainty(
            fit,
            uncertainty_map.get(model_id),
            policy,
        )
        assessment = _assess_candidate(
            model_id,
            fit,
            uncertainty=uncertainty,
            uncertainty_note=uncertainty_note,
            bounds=bounds_map.get(model_id),
            checks=physical_map.get(model_id),
            selected_key_parameters=key_map.get(model_id),
            residual_correlation=residual_correlations[model_id],
            delta_aicc=delta_aicc[model_id],
            delta_bic=delta_bic[model_id],
            statistical_score=scores[model_id],
            policy=policy,
        )
        assessments[model_id] = assessment
        if not assessment.reliable:
            warnings.append(
                f"MODEL_NOT_RELIABLE[{model_id}]: "
                + "; ".join(assessment.gate_failures)
            )

    ranking = tuple(
        sorted(
            fits,
            key=lambda name: (
                scores[name],
                _finite_or_infinity(fits[name].AICc),
                _finite_or_infinity(fits[name].BIC),
                _finite_or_infinity(fits[name].SSE),
                name,
            ),
        )
    )
    best_statistical = ranking[0]
    reliable_ranking = tuple(name for name in ranking if assessments[name].reliable)
    best_reliable = reliable_ranking[0] if reliable_ranking else None

    ranking_method = (
        "Information criteria were unavailable, so the provisional statistical "
        "ordering uses SSE and is not eligible for a trusted recommendation."
        if used_sse_fallback
        else "Statistical ranking combines delta-AICc, delta-BIC, and residual "
        "lag-1 autocorrelation; raw SSE is only a final tie-breaker."
    )
    reasoning = [
        ranking_method,
        _statistical_reason(best_statistical, assessments[best_statistical]),
    ]
    if best_reliable is None:
        warnings.append(
            "NO_RELIABLE_MODEL: every candidate failed at least one credibility Gate."
        )
        reasoning.append(
            "No model passed every identifiability, residual, bound, and physical Gate."
        )
    else:
        reasoning.append(
            f"{best_reliable} is the highest-ranked candidate that passed every Gate."
        )
    if not assessments[best_statistical].reliable:
        warnings.append(
            f"STATISTICAL_WINNER_NOT_RELIABLE: {best_statistical} has the best "
            "statistical score but failed the credibility Gate."
        )
    if best_reliable is not None and best_reliable != best_statistical:
        warnings.append(
            "STATISTICAL_AND_RELIABLE_SELECTION_DIFFER: prefer best_reliable for "
            "scientific reporting unless the failed Gates are resolved."
        )
        reasoning.append(
            f"{best_statistical} was not promoted to trusted selection; "
            f"{best_reliable} is recommended instead."
        )

    return ModelSelectionResult(
        best_statistical=best_statistical,
        best_reliable=best_reliable,
        ranking=ranking,
        assessments=assessments,
        warnings=tuple(dict.fromkeys(warnings)),
        reasoning=tuple(reasoning),
    )


def _assess_candidate(
    model_id: str,
    fit: FitResult,
    *,
    uncertainty: LocalUncertaintyResult | None,
    uncertainty_note: str | None,
    bounds: object,
    checks: object,
    selected_key_parameters: object,
    residual_correlation: float,
    delta_aicc: float,
    delta_bic: float,
    statistical_score: float,
    policy: ModelSelectionConfig,
) -> CandidateAssessment:
    failures: list[str] = []
    reasoning: list[str] = [
        f"delta-AICc={delta_aicc:.6g}, delta-BIC={delta_bic:.6g}, "
        f"statistical score={statistical_score:.6g}."
    ]
    if fit.exit_status <= 0:
        failures.append(f"OPTIMIZER_NOT_CONVERGED: exit_status={fit.exit_status}.")
    if not np.isfinite(fit.AICc) and not np.isfinite(fit.BIC):
        failures.append(
            "INFORMATION_CRITERIA_UNAVAILABLE: neither AICc nor BIC is finite."
        )

    invalid_parameters = _invalid_parameters(fit)
    if invalid_parameters:
        failures.append(
            "PHYSICAL_PARAMETER_INVALID: " + ", ".join(invalid_parameters) + "."
        )
    if np.isfinite(residual_correlation):
        reasoning.append(
            f"Residual lag-1 autocorrelation={residual_correlation:.6g}."
        )
        if abs(residual_correlation) > policy.residual_autocorrelation_limit:
            failures.append(
                "RESIDUAL_AUTOCORRELATION_EXCEEDS_LIMIT: "
                f"|rho1|={abs(residual_correlation):.6g} > "
                f"{policy.residual_autocorrelation_limit:.6g}."
            )
    else:
        reasoning.append("Residual lag-1 autocorrelation is unavailable.")

    condition_number: float | None = None
    max_correlation: float | None = None
    ci_widths: dict[str, float] = {}
    if uncertainty is None:
        if policy.require_uncertainty:
            failures.append(
                "MISSING_UNCERTAINTY: no local uncertainty result or Jacobian is available."
            )
        if uncertainty_note:
            reasoning.append(uncertainty_note)
    else:
        condition_number = uncertainty.condition_number
        max_correlation = _maximum_off_diagonal_correlation(
            uncertainty.correlation_matrix
        )
        keys = _key_parameter_names(fit, selected_key_parameters)
        ci_widths, missing_intervals = _ci_widths_decades(uncertainty, keys)
        if missing_intervals:
            failures.append(
                "MISSING_KEY_PARAMETER_CI: "
                + ", ".join(missing_intervals)
                + "."
            )
        wide = tuple(
            name
            for name, width in ci_widths.items()
            if not np.isfinite(width) or width > policy.ci_width_decades_limit
        )
        if wide:
            details = ", ".join(f"{name}={ci_widths[name]:.3g}" for name in wide)
            failures.append(
                "CI_WIDTH_EXCEEDS_LIMIT: "
                f"{details} decades; limit={policy.ci_width_decades_limit:.3g}."
            )
        if (
            not np.isfinite(condition_number)
            or condition_number > policy.condition_number_limit
        ):
            failures.append(
                "JACOBIAN_CONDITION_EXCEEDS_LIMIT: "
                f"cond(J^T J)={condition_number:.6g}; "
                f"limit={policy.condition_number_limit:.6g}."
            )
        if (
            max_correlation is not None
            and max_correlation > policy.correlation_limit
        ):
            failures.append(
                "PARAMETER_CORRELATION_EXCEEDS_LIMIT: "
                f"max |r|={max_correlation:.6g}; "
                f"limit={policy.correlation_limit:.6g}."
            )
        if not uncertainty.reliable_separation:
            failures.append(
                "LOCAL_UNCERTAINTY_UNRELIABLE: local covariance does not reliably "
                "separate all parameters."
            )
        reasoning.append(
            f"Local uncertainty: rank={uncertainty.rank}, "
            f"cond(J^T J)={condition_number:.6g}, "
            f"max |correlation|={max_correlation or 0.0:.6g}."
        )

    at_bounds, outside_bounds = _evaluate_bounds(fit, bounds, policy)
    if at_bounds:
        failures.append("PARAMETER_AT_BOUND: " + ", ".join(at_bounds) + ".")
    if outside_bounds:
        failures.append(
            "PARAMETER_OUTSIDE_BOUND: " + ", ".join(outside_bounds) + "."
        )

    failed_checks = _failed_physical_checks(checks)
    if failed_checks:
        failures.append(
            "PHYSICAL_CONSTRAINT_FAILED: " + ", ".join(failed_checks) + "."
        )
    for warning in fit.warnings:
        if any(
            token in warning
            for token in (
                "UNRELIABLE_PARAMETER_SEPARATION",
                "RANK_DEFICIENT_JACOBIAN",
            )
        ):
            failures.append(f"FIT_WARNING_GATE: {warning}")

    failures = list(dict.fromkeys(failures))
    reliable = not failures
    reasoning.append(
        "Passed every credibility Gate."
        if reliable
        else f"Failed {len(failures)} credibility Gate(s)."
    )
    return CandidateAssessment(
        model_id=model_id,
        AICc=float(fit.AICc),
        BIC=float(fit.BIC),
        SSE=float(fit.SSE),
        delta_AICc=float(delta_aicc),
        delta_BIC=float(delta_bic),
        statistical_score=float(statistical_score),
        residual_lag1_autocorrelation=float(residual_correlation),
        condition_number=condition_number,
        max_abs_parameter_correlation=max_correlation,
        ci_width_decades=ci_widths,
        parameters_at_bounds=at_bounds,
        parameters_outside_bounds=outside_bounds,
        failed_physical_checks=failed_checks,
        reliable=reliable,
        gate_failures=tuple(failures),
        reasoning=tuple(reasoning),
    )


def _resolve_uncertainty(
    fit: FitResult,
    supplied: object,
    policy: ModelSelectionConfig,
) -> tuple[LocalUncertaintyResult | None, str | None]:
    if supplied is not None:
        if not isinstance(supplied, LocalUncertaintyResult):
            raise TypeError("uncertainties values must be LocalUncertaintyResult")
        return supplied, "Local uncertainty was supplied by the caller."
    if fit.jacobian is None:
        return None, "Local uncertainty could not be derived because Jacobian is absent."
    try:
        result = estimate_local_uncertainty(
            fit,
            condition_threshold=policy.condition_number_limit,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return None, f"Automatic local uncertainty failed: {type(exc).__name__}: {exc}"
    return result, "Local uncertainty was derived from the stored Jacobian."


def _key_parameter_names(
    fit: FitResult,
    selected: object,
) -> tuple[str, ...]:
    if selected is None:
        return tuple(
            name for name in fit.params if fit.param_space[name] == "log"
        )
    if isinstance(selected, str) or not isinstance(selected, Sequence):
        raise TypeError("key_parameters values must be sequences of names")
    result = tuple(selected)
    unknown = set(result) - set(fit.params)
    if unknown:
        raise ValueError(f"unknown key parameters: {sorted(unknown)}")
    return result


def _ci_widths_decades(
    uncertainty: LocalUncertaintyResult,
    parameters: Sequence[str],
) -> tuple[dict[str, float], tuple[str, ...]]:
    widths: dict[str, float] = {}
    missing: list[str] = []
    for name in parameters:
        interval = uncertainty.confidence_intervals.get(name)
        if interval is None:
            missing.append(name)
            continue
        if interval.space == "log":
            width = interval.coordinate_upper - interval.coordinate_lower
        elif interval.lower > 0.0 and interval.upper > 0.0:
            width = np.log10(interval.upper) - np.log10(interval.lower)
        else:
            width = float("inf")
        widths[name] = float(width)
    return widths, tuple(missing)


def _maximum_off_diagonal_correlation(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = np.abs(matrix[mask])
    finite = values[np.isfinite(values)]
    return 0.0 if finite.size == 0 else float(np.max(finite))


def _evaluate_bounds(
    fit: FitResult,
    supplied: object,
    policy: ModelSelectionConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if supplied is None:
        return (), ()
    if not isinstance(supplied, Mapping):
        raise TypeError("parameter_bounds values must be mappings")
    unknown = set(supplied) - set(fit.params)
    if unknown:
        raise ValueError(f"bounds supplied for unknown parameters: {sorted(unknown)}")
    at_bounds: list[str] = []
    outside: list[str] = []
    for name, raw_bound in supplied.items():
        bound = _coerce_bound(raw_bound)
        value = float(fit.params[name])
        if bound.space == "coordinate" and fit.param_space[name] == "log":
            if value <= 0.0:
                outside.append(name)
                continue
            value = float(np.log10(value))
        if value < bound.lower or value > bound.upper:
            outside.append(name)
            continue
        tolerance = _bound_tolerance(bound, policy.bound_proximity_fraction)
        if (
            (np.isfinite(bound.lower) and value - bound.lower <= tolerance)
            or (np.isfinite(bound.upper) and bound.upper - value <= tolerance)
        ):
            at_bounds.append(name)
    return tuple(at_bounds), tuple(outside)


def _coerce_bound(value: object) -> ParameterBound:
    if isinstance(value, ParameterBound):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return ParameterBound(value[0], value[1])
    raise TypeError("bounds must be ParameterBound or (lower, upper) tuples")


def describe_bound_proximity(fit, name, bound, policy=None):
    """Explain the existing rule using its exact coordinate and tolerance.

    This does not change or replace the gate decision.
    """
    policy = policy or ModelSelectionConfig()
    value = float(fit.params[name])
    if bound.space == "coordinate" and fit.param_space[name] == "log":
        value = float(np.log10(value)) if value > 0 else float("-inf")
    tolerance = _bound_tolerance(bound, policy.bound_proximity_fraction)
    basis = "supplied range" if np.isfinite(bound.lower) and np.isfinite(bound.upper) else "max(1, abs(finite edge))"
    return (f"Credibility proximity: {bound.space} bounds [{bound.lower:.6g}, {bound.upper:.6g}]; "
            f"value={value:.6g}; tolerance={tolerance:.6g} "
            f"({policy.bound_proximity_fraction:.0%} of {basis}; {BOUND_POLICY_VERSION}). "
            "PARAMETER_AT_BOUND means near a bound under this policy, not numerical contact.")


def _bound_tolerance(bound: ParameterBound, fraction: float) -> float:
    if np.isfinite(bound.lower) and np.isfinite(bound.upper):
        return fraction * (bound.upper - bound.lower)
    finite_edge = bound.lower if np.isfinite(bound.lower) else bound.upper
    return fraction * max(1.0, abs(finite_edge))


def _failed_physical_checks(supplied: object) -> tuple[str, ...]:
    if supplied is None:
        return ()
    if not isinstance(supplied, Mapping):
        raise TypeError("physical_checks values must be mappings")
    failed: list[str] = []
    for name, passed in supplied.items():
        if not isinstance(name, str) or not name:
            raise ValueError("physical check names must be nonempty strings")
        if not isinstance(passed, bool):
            raise TypeError("physical check results must be boolean")
        if not passed:
            failed.append(name)
    return tuple(failed)


def _invalid_parameters(fit: FitResult) -> tuple[str, ...]:
    invalid: list[str] = []
    for name, value in fit.params.items():
        if not np.isfinite(value):
            invalid.append(name)
        elif fit.param_space[name] == "log" and value <= 0.0:
            invalid.append(name)
    return tuple(invalid)


def _lag1_autocorrelation(residuals: np.ndarray) -> float:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return float("nan")
    centred = values - float(np.mean(values))
    left = centred[:-1]
    right = centred[1:]
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _metric_deltas(values: Mapping[str, float]) -> dict[str, float]:
    finite_values = [float(value) for value in values.values() if np.isfinite(value)]
    if not finite_values:
        return {name: float("inf") for name in values}
    minimum = min(finite_values)
    return {
        name: float(value) - minimum if np.isfinite(value) else float("inf")
        for name, value in values.items()
    }


def _statistical_score(
    delta_aicc: float,
    delta_bic: float,
    residual_correlation: float,
    policy: ModelSelectionConfig,
) -> float:
    weighted_score = 0.0
    available_weight = 0.0
    if np.isfinite(delta_aicc) and policy.aicc_weight > 0.0:
        weighted_score += policy.aicc_weight * delta_aicc
        available_weight += policy.aicc_weight
    if np.isfinite(delta_bic) and policy.bic_weight > 0.0:
        weighted_score += policy.bic_weight * delta_bic
        available_weight += policy.bic_weight
    if available_weight <= 0.0:
        return float("inf")
    information_score = weighted_score / available_weight
    residual_penalty = 0.0
    if np.isfinite(residual_correlation):
        residual_penalty = policy.residual_penalty_weight * max(
            0.0,
            abs(residual_correlation) - policy.residual_autocorrelation_limit,
        )
    return float(information_score + residual_penalty)


def _sse_fallback_scores(fits: Mapping[str, FitResult]) -> dict[str, float]:
    finite = [fit.SSE for fit in fits.values() if np.isfinite(fit.SSE)]
    if not finite:
        return {name: float("inf") for name in fits}
    minimum = min(finite)
    scale = max(abs(minimum), np.finfo(np.float64).tiny)
    return {
        name: (fit.SSE - minimum) / scale if np.isfinite(fit.SSE) else float("inf")
        for name, fit in fits.items()
    }


def _normalise_fits(fit_results: Mapping[ModelKey, FitResult]) -> dict[str, FitResult]:
    if not isinstance(fit_results, Mapping):
        raise TypeError("fit_results must be a mapping of model IDs to FitResult")
    if not fit_results:
        raise ValueError("fit_results must contain at least one candidate")
    result: dict[str, FitResult] = {}
    for raw_key, fit in fit_results.items():
        key = _model_key(raw_key)
        if key in result:
            raise ValueError(f"duplicate normalized model ID: {key!r}")
        if not isinstance(fit, FitResult):
            raise TypeError(f"fit result for {key!r} must be FitResult")
        result[key] = fit
    return result


def _normalise_auxiliary(
    name: str,
    values: Mapping[ModelKey, object] | None,
    *,
    model_ids: tuple[str, ...],
) -> dict[str, object]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping or None")
    result = {_model_key(key): value for key, value in values.items()}
    unknown = set(result) - set(model_ids)
    if unknown:
        raise ValueError(f"{name} contains unknown model IDs: {sorted(unknown)}")
    return result


def _model_key(value: ModelKey) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str) or not value:
        raise ValueError("model IDs must be nonempty strings or string-valued enums")
    return value


def _statistical_reason(
    model_id: str,
    assessment: CandidateAssessment,
) -> str:
    return (
        f"{model_id} has the best statistical score "
        f"({assessment.statistical_score:.6g}); AICc={assessment.AICc:.6g}, "
        f"BIC={assessment.BIC:.6g}."
    )


def _finite_or_infinity(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


def _real_scalar(name: str, value: float, *, allow_infinite: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if np.isnan(result) or (not allow_infinite and not np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: float) -> float:
    result = _real_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(name: str, value: float) -> float:
    result = _real_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _closed_interval(
    name: str,
    value: float,
    lower: float,
    upper: float,
) -> float:
    result = _real_scalar(name, value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result
