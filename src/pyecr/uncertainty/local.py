"""Local covariance, confidence intervals, and identifiability diagnostics.

The Jacobian columns in this module always refer to the optimizer coordinates
declared by :class:`~pyecr.fitting.base.FitResult.param_space`.  Consequently,
parameters marked ``"log"`` are differentiated with respect to ``log10(p)``
and their covariance is first calculated in that coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from pyecr.fitting.base import FitResult, ParameterSpace

__all__ = [
    "ConfidenceInterval",
    "JacobianMethod",
    "LocalUncertaintyResult",
    "ReliabilityStatus",
    "estimate_local_uncertainty",
]


JacobianMethod = Literal["auto", "complex_step", "central_difference"]
PredictionFunction = Callable[
    [Mapping[str, float | complex]],
    ArrayLike,
]


class ReliabilityStatus(str, Enum):
    """Whether the local linearization can separate all fitted parameters."""

    RELIABLE = "reliable"
    UNRELIABLE_SEPARATION = "unreliable_parameter_separation"


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """One parameter's local confidence interval.

    ``coordinate_*`` values are in the optimizer coordinate.  ``estimate``,
    ``lower``, and ``upper`` are reported in physical (linear) units.  A log10
    interval therefore generally becomes asymmetric after back-transformation.
    """

    parameter: str
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    space: ParameterSpace
    coordinate_estimate: float
    coordinate_lower: float
    coordinate_upper: float
    coordinate_standard_error: float


@dataclass(frozen=True, slots=True)
class LocalUncertaintyResult:
    """Local least-squares uncertainty and identifiability summary."""

    jacobian: NDArray[np.float64]
    jacobian_source: str
    covariance: NDArray[np.float64]
    correlation_matrix: NDArray[np.float64]
    confidence_intervals: Mapping[str, ConfidenceInterval]
    sensitivity_norms: Mapping[str, float]
    condition_number: float
    rank: int
    degrees_of_freedom: int
    residual_variance: float
    confidence_level: float
    reliability_status: ReliabilityStatus
    reliable_separation: bool
    annotated_fit_result: FitResult
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "jacobian", _read_only_matrix(self.jacobian))
        object.__setattr__(
            self,
            "covariance",
            _read_only_matrix(self.covariance, allow_nan=True),
        )
        object.__setattr__(
            self,
            "correlation_matrix",
            _read_only_matrix(self.correlation_matrix, allow_nan=True),
        )
        object.__setattr__(
            self,
            "confidence_intervals",
            MappingProxyType(dict(self.confidence_intervals)),
        )
        object.__setattr__(
            self,
            "sensitivity_norms",
            MappingProxyType(dict(self.sensitivity_norms)),
        )


def estimate_local_uncertainty(
    fit_result: FitResult,
    *,
    prediction_function: PredictionFunction | None = None,
    jacobian: ArrayLike | None = None,
    jacobian_method: JacobianMethod = "auto",
    confidence_level: float = 0.95,
    condition_threshold: float = 1.0e12,
    finite_difference_step: float = 1.0e-5,
    complex_step: float = 1.0e-20,
    svd_rcond: float | None = None,
) -> LocalUncertaintyResult:
    """Estimate local parameter uncertainty from a completed least-squares fit.

    Jacobian selection is deliberately deterministic: the Jacobian stored by
    ``least_squares`` in ``fit_result`` is preferred, followed by an explicitly
    supplied matrix, then a numerical Jacobian of ``prediction_function``.
    ``prediction_function`` receives physical parameter values, while numerical
    perturbations are applied in each parameter's declared optimizer space.

    The covariance approximation is ``s^2 * pinv(J.T @ J)``, where
    ``s^2 = SSE / (n - p)``.  Rank and condition diagnostics are obtained from
    an SVD of ``J`` so that a deficient or nearly singular problem is not hidden
    by the pseudoinverse.
    """

    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult")
    _validate_options(
        jacobian_method=jacobian_method,
        confidence_level=confidence_level,
        condition_threshold=condition_threshold,
        finite_difference_step=finite_difference_step,
        complex_step=complex_step,
        svd_rcond=svd_rcond,
    )

    parameter_names = tuple(fit_result.params)
    n_observations = fit_result.y_fit.size
    n_parameters = len(parameter_names)
    if n_parameters == 0:
        raise ValueError("local uncertainty requires at least one fitted parameter")

    selected_jacobian, source = _select_jacobian(
        fit_result,
        jacobian=jacobian,
        prediction_function=prediction_function,
        jacobian_method=jacobian_method,
        finite_difference_step=finite_difference_step,
        complex_step=complex_step,
    )
    selected_jacobian = _validate_jacobian_shape(
        selected_jacobian,
        n_observations=n_observations,
        n_parameters=n_parameters,
    )

    _, singular_values, right_vectors = np.linalg.svd(
        selected_jacobian,
        # Only right singular vectors are consumed. For N>=p the reduced
        # decomposition preserves every right vector and avoids an unused N*N
        # U allocation when explicitly recomputing CI on full observations.
        # N<p still needs the complete right null space for the old N/A rule.
        full_matrices=n_observations < n_parameters,
    )
    singular_tolerance = _singular_tolerance(
        singular_values,
        selected_jacobian.shape,
        svd_rcond,
    )
    rank = int(np.count_nonzero(singular_values > singular_tolerance))
    condition_number = _normal_matrix_condition(singular_values, rank, n_parameters)

    degrees_of_freedom = n_observations - n_parameters
    residual_variance = (
        fit_result.SSE / degrees_of_freedom
        if degrees_of_freedom > 0
        else float("nan")
    )
    covariance = _covariance_matrix(
        selected_jacobian,
        residual_variance,
        svd_rcond,
    )
    correlation = _correlation_matrix(covariance)

    standard_errors = _coordinate_standard_errors(
        covariance,
        right_vectors,
        rank=rank,
        n_parameters=n_parameters,
    )
    intervals = _confidence_intervals(
        fit_result,
        standard_errors,
        confidence_level=confidence_level,
        degrees_of_freedom=degrees_of_freedom,
    )
    sensitivity_norms = {
        name: float(np.linalg.norm(selected_jacobian[:, index]))
        for index, name in enumerate(parameter_names)
    }

    warnings = _reliability_warnings(
        rank=rank,
        n_parameters=n_parameters,
        condition_number=condition_number,
        condition_threshold=condition_threshold,
        degrees_of_freedom=degrees_of_freedom,
    )
    reliable = not warnings
    status = (
        ReliabilityStatus.RELIABLE
        if reliable
        else ReliabilityStatus.UNRELIABLE_SEPARATION
    )
    annotated_fit = replace(
        fit_result,
        warnings=_append_unique(fit_result.warnings, warnings),
    )

    return LocalUncertaintyResult(
        jacobian=selected_jacobian,
        jacobian_source=source,
        covariance=covariance,
        correlation_matrix=correlation,
        confidence_intervals=intervals,
        sensitivity_norms=sensitivity_norms,
        condition_number=condition_number,
        rank=rank,
        degrees_of_freedom=degrees_of_freedom,
        residual_variance=float(residual_variance),
        confidence_level=float(confidence_level),
        reliability_status=status,
        reliable_separation=reliable,
        annotated_fit_result=annotated_fit,
        warnings=warnings,
    )


def _select_jacobian(
    fit_result: FitResult,
    *,
    jacobian: ArrayLike | None,
    prediction_function: PredictionFunction | None,
    jacobian_method: JacobianMethod,
    finite_difference_step: float,
    complex_step: float,
) -> tuple[NDArray[np.float64], str]:
    if fit_result.jacobian is not None:
        return np.asarray(fit_result.jacobian, dtype=np.float64), "least_squares"
    if jacobian is not None:
        return np.asarray(jacobian, dtype=np.float64), "provided"
    if prediction_function is None:
        raise ValueError(
            "prediction_function is required when no Jacobian is available"
        )

    coordinates = _optimizer_coordinates(fit_result)
    if jacobian_method in ("auto", "complex_step"):
        try:
            result = _complex_step_jacobian(
                fit_result,
                prediction_function,
                coordinates,
                step=complex_step,
            )
            return result, "complex_step"
        except (TypeError, ValueError, FloatingPointError):
            if jacobian_method == "complex_step":
                raise

    return (
        _central_difference_jacobian(
            fit_result,
            prediction_function,
            coordinates,
            relative_step=finite_difference_step,
        ),
        "central_difference",
    )


def _optimizer_coordinates(fit_result: FitResult) -> NDArray[np.float64]:
    coordinates: list[float] = []
    for name, value in fit_result.params.items():
        if fit_result.param_space[name] == "log":
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"log-space parameter {name!r} must be finite and positive"
                )
            coordinates.append(float(np.log10(value)))
        else:
            if not np.isfinite(value):
                raise ValueError(f"linear parameter {name!r} must be finite")
            coordinates.append(float(value))
    return np.asarray(coordinates, dtype=np.float64)


def _physical_parameters(
    fit_result: FitResult,
    coordinates: NDArray[np.float64] | NDArray[np.complex128],
) -> dict[str, float | complex]:
    params: dict[str, float | complex] = {}
    for index, name in enumerate(fit_result.params):
        coordinate = coordinates[index]
        if fit_result.param_space[name] == "log":
            params[name] = 10.0**coordinate
        else:
            params[name] = coordinate.item()
    return params


def _complex_step_jacobian(
    fit_result: FitResult,
    prediction_function: PredictionFunction,
    coordinates: NDArray[np.float64],
    *,
    step: float,
) -> NDArray[np.float64]:
    n_observations = fit_result.y_fit.size
    result = np.empty((n_observations, coordinates.size), dtype=np.float64)

    for index in range(coordinates.size):
        perturbed = coordinates.astype(np.complex128)
        perturbed[index] += 1j * step
        prediction = np.asarray(
            prediction_function(_physical_parameters(fit_result, perturbed))
        )
        _validate_prediction(prediction, n_observations, allow_complex=True)
        if not np.iscomplexobj(prediction):
            raise TypeError("prediction function discarded complex perturbations")
        result[:, index] = np.imag(prediction) / step

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("complex-step Jacobian contains non-finite values")
    return result


def _central_difference_jacobian(
    fit_result: FitResult,
    prediction_function: PredictionFunction,
    coordinates: NDArray[np.float64],
    *,
    relative_step: float,
) -> NDArray[np.float64]:
    n_observations = fit_result.y_fit.size
    result = np.empty((n_observations, coordinates.size), dtype=np.float64)

    for index in range(coordinates.size):
        step = relative_step * max(1.0, abs(float(coordinates[index])))
        upper = coordinates.copy()
        lower = coordinates.copy()
        upper[index] += step
        lower[index] -= step
        prediction_upper = np.asarray(
            prediction_function(_physical_parameters(fit_result, upper))
        )
        prediction_lower = np.asarray(
            prediction_function(_physical_parameters(fit_result, lower))
        )
        _validate_prediction(prediction_upper, n_observations, allow_complex=False)
        _validate_prediction(prediction_lower, n_observations, allow_complex=False)
        result[:, index] = np.asarray(
            (prediction_upper - prediction_lower) / (2.0 * step),
            dtype=np.float64,
        )

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("central-difference Jacobian contains non-finite values")
    return result


def _validate_prediction(
    prediction: NDArray[np.generic],
    n_observations: int,
    *,
    allow_complex: bool,
) -> None:
    if prediction.ndim != 1:
        raise ValueError("prediction_function must return a one-dimensional array")
    if prediction.size != n_observations:
        raise ValueError(
            "prediction_function output length must match the fitted observations"
        )
    if np.iscomplexobj(prediction) and not allow_complex:
        raise TypeError("central-difference predictions must be real")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("prediction_function returned non-finite values")


def _validate_jacobian_shape(
    jacobian: ArrayLike,
    *,
    n_observations: int,
    n_parameters: int,
) -> NDArray[np.float64]:
    try:
        result = np.asarray(jacobian, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("jacobian must contain real numeric values") from exc
    if result.ndim != 2 or result.shape != (n_observations, n_parameters):
        raise ValueError(
            "jacobian shape must be (number of observations, number of parameters)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("jacobian values must be finite")
    return result.copy()


def _singular_tolerance(
    singular_values: NDArray[np.float64],
    shape: tuple[int, int],
    svd_rcond: float | None,
) -> float:
    if singular_values.size == 0:
        return 0.0
    scale = float(singular_values[0])
    if svd_rcond is None:
        return np.finfo(np.float64).eps * max(shape) * scale
    return svd_rcond * scale


def _normal_matrix_condition(
    singular_values: NDArray[np.float64],
    rank: int,
    n_parameters: int,
) -> float:
    if rank < n_parameters or singular_values.size < n_parameters:
        return float("inf")
    smallest = float(singular_values[n_parameters - 1])
    largest = float(singular_values[0])
    if smallest <= 0.0:
        return float("inf")
    ratio = largest / smallest
    if ratio > np.sqrt(np.finfo(np.float64).max):
        return float("inf")
    return float(ratio * ratio)


def _covariance_matrix(
    jacobian: NDArray[np.float64],
    residual_variance: float,
    svd_rcond: float | None,
) -> NDArray[np.float64]:
    if not np.isfinite(residual_variance):
        return np.full((jacobian.shape[1], jacobian.shape[1]), np.nan)
    normal_matrix = jacobian.T @ jacobian
    if svd_rcond is None:
        inverse = np.linalg.pinv(normal_matrix)
    else:
        inverse = np.linalg.pinv(normal_matrix, rcond=svd_rcond)
    covariance = residual_variance * inverse
    return (covariance + covariance.T) / 2.0


def _correlation_matrix(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    diagonal = np.diag(covariance)
    standard_deviations = np.sqrt(np.maximum(diagonal, 0.0))
    denominator = np.outer(standard_deviations, standard_deviations)
    result = np.full_like(covariance, np.nan)
    np.divide(covariance, denominator, out=result, where=denominator > 0.0)
    finite_positive = np.isfinite(standard_deviations) & (standard_deviations > 0.0)
    indices = np.flatnonzero(finite_positive)
    result[indices, indices] = 1.0
    return np.clip(result, -1.0, 1.0, out=result, where=np.isfinite(result))


def _coordinate_standard_errors(
    covariance: NDArray[np.float64],
    right_vectors: NDArray[np.float64],
    *,
    rank: int,
    n_parameters: int,
) -> NDArray[np.float64]:
    diagonal = np.diag(covariance)
    standard_errors = np.sqrt(np.maximum(diagonal, 0.0))
    if rank < n_parameters:
        null_space = right_vectors[rank:, :]
        null_loadings = np.linalg.norm(null_space, axis=0)
        standard_errors[null_loadings > np.sqrt(np.finfo(np.float64).eps)] = np.inf
    return standard_errors


def _confidence_intervals(
    fit_result: FitResult,
    standard_errors: NDArray[np.float64],
    *,
    confidence_level: float,
    degrees_of_freedom: int,
) -> dict[str, ConfidenceInterval]:
    if degrees_of_freedom > 0:
        critical_value = float(
            student_t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom)
        )
    else:
        critical_value = float("inf")

    intervals: dict[str, ConfidenceInterval] = {}
    for index, (name, estimate) in enumerate(fit_result.params.items()):
        space = fit_result.param_space[name]
        coordinate_estimate = (
            float(np.log10(estimate)) if space == "log" else float(estimate)
        )
        standard_error = float(standard_errors[index])
        half_width = critical_value * standard_error
        if np.isnan(half_width):
            coordinate_lower = float("nan")
            coordinate_upper = float("nan")
        else:
            coordinate_lower = coordinate_estimate - half_width
            coordinate_upper = coordinate_estimate + half_width

        if space == "log":
            lower = _power_ten(coordinate_lower)
            upper = _power_ten(coordinate_upper)
        else:
            lower = coordinate_lower
            upper = coordinate_upper

        intervals[name] = ConfidenceInterval(
            parameter=name,
            estimate=float(estimate),
            lower=lower,
            upper=upper,
            confidence_level=float(confidence_level),
            space=space,
            coordinate_estimate=coordinate_estimate,
            coordinate_lower=coordinate_lower,
            coordinate_upper=coordinate_upper,
            coordinate_standard_error=standard_error,
        )
    return intervals


def _power_ten(exponent: float) -> float:
    if np.isnan(exponent):
        return float("nan")
    if exponent == float("-inf") or exponent < np.log10(np.finfo(np.float64).tiny):
        return 0.0
    if exponent == float("inf") or exponent > np.log10(np.finfo(np.float64).max):
        return float("inf")
    return float(10.0**exponent)


def _reliability_warnings(
    *,
    rank: int,
    n_parameters: int,
    condition_number: float,
    condition_threshold: float,
    degrees_of_freedom: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if rank < n_parameters:
        warnings.append(
            f"RANK_DEFICIENT_JACOBIAN: rank={rank}, parameters={n_parameters}."
        )
    if not np.isfinite(condition_number) or condition_number > condition_threshold:
        warnings.append(
            "ILL_CONDITIONED_INFORMATION_MATRIX: "
            f"cond(J^T J)={condition_number:.6g} exceeds "
            f"threshold={condition_threshold:.6g}."
        )
    if degrees_of_freedom <= 0:
        warnings.append(
            "NONPOSITIVE_DEGREES_OF_FREEDOM: covariance scale cannot be estimated."
        )
    if warnings:
        warnings.append(
            "UNRELIABLE_PARAMETER_SEPARATION: local covariance cannot reliably "
            "separate all fitted parameters."
        )
    return tuple(warnings)


def _append_unique(
    existing: tuple[str, ...],
    additions: tuple[str, ...],
) -> tuple[str, ...]:
    result = list(existing)
    for message in additions:
        if message not in result:
            result.append(message)
    return tuple(result)


def _read_only_matrix(
    values: ArrayLike,
    *,
    allow_nan: bool = False,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 2:
        raise ValueError("uncertainty matrices must be two-dimensional")
    if allow_nan:
        if np.any(np.isinf(result)):
            raise ValueError("uncertainty matrices cannot contain infinity")
    elif not np.all(np.isfinite(result)):
        raise ValueError("uncertainty matrices must be finite")
    result.flags.writeable = False
    return result


def _validate_options(
    *,
    jacobian_method: str,
    confidence_level: float,
    condition_threshold: float,
    finite_difference_step: float,
    complex_step: float,
    svd_rcond: float | None,
) -> None:
    if jacobian_method not in {"auto", "complex_step", "central_difference"}:
        raise ValueError(
            "jacobian_method must be 'auto', 'complex_step', or "
            "'central_difference'"
        )
    _positive_finite("condition_threshold", condition_threshold)
    _positive_finite("finite_difference_step", finite_difference_step)
    _positive_finite("complex_step", complex_step)
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not np.isfinite(confidence_level)
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    if svd_rcond is not None:
        _positive_finite("svd_rcond", svd_rcond, allow_zero=True)


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    invalid_sign = numeric < 0.0 if allow_zero else numeric <= 0.0
    if not np.isfinite(numeric) or invalid_sign:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
