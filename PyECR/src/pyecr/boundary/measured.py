"""Measured pO2/pH2O boundary histories for linear ECR forward models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

__all__ = [
    "BoundarySignalType",
    "MeasuredBoundary",
    "MeasuredBoundaryResult",
    "ResampledBoundary",
]


class BoundarySignalType(str, Enum):
    """Measured gas variables currently accepted by the boundary layer."""

    PO2 = "pO2"
    PH2O = "pH2O"


@dataclass(frozen=True, slots=True)
class ResampledBoundary:
    """Measured boundary values evaluated at a requested in-range time axis."""

    time_s: NDArray[np.float64]
    values: NDArray[np.float64]
    normalized_progress: NDArray[np.float64]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MeasuredBoundaryResult:
    """Material response driven by a measured, normalized boundary history."""

    time_s: NDArray[np.float64]
    normalized_response: NDArray[np.float64]
    ideal_response: NDArray[np.float64]
    boundary: ResampledBoundary
    diagnostics: Mapping[str, object]


IdealStepResponse = Callable[[NDArray[np.float64]], ArrayLike]

_INITIAL_PROGRESS_ATOL = 1.0e-10
_STEP_RESPONSE_ATOL = 1.0e-10
_DEFAULT_MAX_INTERNAL_POINTS = 200_000


@dataclass(frozen=True, slots=True)
class MeasuredBoundary:
    """A synchronized measured gas boundary for small-step linear response.

    The class assumes that, over the applied perturbation, normalized pO2 or
    pH2O progress is linearly proportional to the material's boundary drive
    and that the material is a linear time-invariant system.  It does not fit,
    smooth, clip, or force the measured history to an exponential profile.
    """

    time_s: ArrayLike
    values: ArrayLike
    signal_type: BoundarySignalType | str
    initial_value: float
    final_value: float
    value_unit: str = "unspecified"
    linear_small_step_assumption: bool = field(init=False, default=True)
    normalized_progress: NDArray[np.float64] = field(init=False, repr=False)
    diagnostics: Mapping[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        times = _one_dimensional_finite_array("time_s", self.time_s)
        values = _one_dimensional_finite_array("values", self.values)
        if times.size < 2:
            raise ValueError("time_s must contain at least two measurements")
        if values.size != times.size:
            raise ValueError("time_s and values must have the same length")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing with no duplicates")
        if np.any(values < 0.0):
            raise ValueError("measured partial-pressure values must be nonnegative")

        signal_type = _coerce_signal_type(self.signal_type)
        initial_value = _nonnegative_finite_scalar(
            "initial_value", self.initial_value
        )
        final_value = _nonnegative_finite_scalar("final_value", self.final_value)
        if final_value == initial_value:
            raise ValueError("initial_value and final_value must be different")
        if not isinstance(self.value_unit, str) or not self.value_unit.strip():
            raise ValueError("value_unit must be a nonempty string")

        progress = (values - initial_value) / (final_value - initial_value)
        if abs(float(progress[0])) > _INITIAL_PROGRESS_ATOL:
            raise ValueError(
                "measured boundary must begin at initial_value; earlier boundary "
                "history cannot be inferred"
            )

        times = _readonly_copy(times)
        values = _readonly_copy(values)
        progress = _readonly_copy(progress)
        progress_differences = np.diff(progress)
        monotonic = bool(
            np.all(progress_differences >= 0.0)
            or np.all(progress_differences <= 0.0)
        )
        diagnostics = MappingProxyType(
            {
                "signal_type": signal_type.value,
                "value_unit": self.value_unit.strip(),
                "n_measurements": int(times.size),
                "time_start_s": float(times[0]),
                "time_end_s": float(times[-1]),
                "minimum_time_step_s": float(np.min(np.diff(times))),
                "maximum_time_step_s": float(np.max(np.diff(times))),
                "is_monotonic_progress": monotonic,
                "progress_min": float(np.min(progress)),
                "progress_max": float(np.max(progress)),
                "outside_unit_interval_count": int(
                    np.count_nonzero((progress < 0.0) | (progress > 1.0))
                ),
                "linear_small_step_assumption": True,
                "input_was_fit": False,
                "smoothing_applied": False,
                "clipping_applied": False,
            }
        )

        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "signal_type", signal_type)
        object.__setattr__(self, "initial_value", initial_value)
        object.__setattr__(self, "final_value", final_value)
        object.__setattr__(self, "value_unit", self.value_unit.strip())
        object.__setattr__(self, "normalized_progress", progress)
        object.__setattr__(self, "diagnostics", diagnostics)

    def resample(self, target_time_s: ArrayLike) -> ResampledBoundary:
        """Linearly resample inside measured coverage without extrapolation."""

        target_times = _one_dimensional_finite_array("target_time_s", target_time_s)
        if target_times.size == 0:
            raise ValueError("target_time_s must contain at least one time")
        self._check_coverage(target_times)

        values = np.interp(target_times, self.time_s, self.values)
        progress = (values - self.initial_value) / (
            self.final_value - self.initial_value
        )
        target_times = _readonly_copy(target_times)
        values = _readonly_copy(values)
        progress = _readonly_copy(progress)

        return ResampledBoundary(
            time_s=target_times,
            values=values,
            normalized_progress=progress,
            diagnostics=MappingProxyType(
                {
                    "source_time_start_s": float(self.time_s[0]),
                    "source_time_end_s": float(self.time_s[-1]),
                    "n_requested": int(target_times.size),
                    "interpolation": "linear",
                    "extrapolation_used": False,
                    "smoothing_applied": False,
                }
            ),
        )

    def apply_to_step_response(
        self,
        response_time_s: ArrayLike,
        ideal_step_response: IdealStepResponse,
        *,
        resample_time_step_s: float | None = None,
        max_internal_points: int = _DEFAULT_MAX_INTERNAL_POINTS,
    ) -> MeasuredBoundaryResult:
        """Drive an ideal linear step response with this measured history.

        The measured profile is represented by piecewise-linear segments.
        Duhamel's integral is evaluated on a checked uniform grid using FFT
        convolution of boundary increments with interval-averaged values of
        the material step response.
        """

        target_times = _one_dimensional_finite_array(
            "response_time_s", response_time_s
        )
        if target_times.size == 0:
            raise ValueError("response_time_s must contain at least one time")
        self._check_coverage(target_times)
        if not callable(ideal_step_response):
            raise TypeError("ideal_step_response must be callable")
        point_limit = _positive_integer("max_internal_points", max_internal_points)

        requested_step: float | None
        if resample_time_step_s is None:
            requested_step = None
        else:
            requested_step = _positive_finite_scalar(
                "resample_time_step_s", resample_time_step_s
            )

        start_time = float(self.time_s[0])
        stop_time = float(np.max(target_times))
        duration = stop_time - start_time
        if duration == 0.0:
            internal_times = np.array([start_time], dtype=np.float64)
            internal_step = None
            auto_step = requested_step is None
        else:
            automatic_step = _automatic_resample_step(self.time_s, target_times)
            chosen_step = automatic_step if requested_step is None else requested_step
            interval_ratio = duration / chosen_step
            nearest_interval_count = round(interval_ratio)
            if np.isclose(
                interval_ratio,
                nearest_interval_count,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                n_intervals = max(1, int(nearest_interval_count))
            else:
                n_intervals = int(ceil(interval_ratio))
            n_points = n_intervals + 1
            if n_points > point_limit:
                raise ValueError(
                    "automatic/internal resampling would exceed "
                    f"max_internal_points={point_limit}; provide an explicit "
                    "resample_time_step_s or increase the limit"
                )
            internal_times = np.linspace(
                start_time, stop_time, n_points, dtype=np.float64
            )
            internal_step = duration / n_intervals
            auto_step = requested_step is None

        internal_boundary = self.resample(internal_times)
        lags = internal_times - start_time
        step_values = _evaluate_step_response(ideal_step_response, lags)
        if not np.isclose(
            step_values[0], 0.0, rtol=0.0, atol=_STEP_RESPONSE_ATOL
        ):
            raise ValueError(
                "ideal_step_response must have normalized response at lag zero "
                "equal to 0"
            )

        response_on_grid = np.zeros_like(internal_times)
        if internal_times.size > 1:
            boundary_increments = np.diff(
                internal_boundary.normalized_progress
            )
            interval_step_response = 0.5 * (
                step_values[:-1] + step_values[1:]
            )
            convolved = fftconvolve(
                boundary_increments,
                interval_step_response,
                mode="full",
            )
            response_on_grid[1:] = convolved[: internal_times.size - 1]

        initial_progress = float(internal_boundary.normalized_progress[0])
        if initial_progress != 0.0:
            response_on_grid += initial_progress * step_values

        response_values = np.interp(
            target_times, internal_times, response_on_grid
        )
        target_lags = target_times - start_time
        ideal_values = _evaluate_step_response(
            ideal_step_response, target_lags
        )
        target_boundary = self.resample(target_times)

        source_minimum_step = float(np.min(np.diff(self.time_s)))
        warnings: list[str] = []
        if internal_step is not None and internal_step > source_minimum_step:
            warnings.append(
                "Explicit internal step is coarser than the finest measured "
                "boundary interval."
            )

        response_values = _readonly_copy(response_values)
        ideal_values = _readonly_copy(ideal_values)
        result_times = _readonly_copy(target_times)
        diagnostics = MappingProxyType(
            {
                "method": "discrete_duhamel_fft_convolution",
                "boundary_interpolation": "piecewise_linear",
                "quadrature": "interval_trapezoid",
                "linear_small_step_assumption": True,
                "signal_type": self.signal_type.value,
                "value_unit": self.value_unit,
                "input_was_fit": False,
                "smoothing_applied": False,
                "clipping_applied": False,
                "extrapolation_used": False,
                "automatic_resampling": auto_step,
                "internal_time_step_s": internal_step,
                "n_internal_points": int(internal_times.size),
                "max_internal_points": point_limit,
                "source_minimum_time_step_s": source_minimum_step,
                "max_abs_difference_from_ideal": float(
                    np.max(np.abs(response_values - ideal_values))
                ),
                "warnings": tuple(warnings),
            }
        )

        return MeasuredBoundaryResult(
            time_s=result_times,
            normalized_response=response_values,
            ideal_response=ideal_values,
            boundary=target_boundary,
            diagnostics=diagnostics,
        )

    def _check_coverage(self, target_times: NDArray[np.float64]) -> None:
        if (
            float(np.min(target_times)) < float(self.time_s[0])
            or float(np.max(target_times)) > float(self.time_s[-1])
        ):
            raise ValueError(
                "target time axis is outside measured boundary coverage; "
                "extrapolation is not allowed"
            )


def _automatic_resample_step(
    source_times: NDArray[np.float64], target_times: NDArray[np.float64]
) -> float:
    candidates = [_representative_minimum_step(source_times)]
    unique_target_times = np.unique(target_times)
    if unique_target_times.size > 1:
        candidates.append(_representative_minimum_step(unique_target_times))
    return min(candidates)


def _representative_minimum_step(times: NDArray[np.float64]) -> float:
    differences = np.diff(times)
    median_step = float(np.median(differences))
    if np.allclose(
        differences,
        median_step,
        rtol=1.0e-10,
        atol=np.finfo(np.float64).eps * max(1.0, abs(median_step)) * 8.0,
    ):
        return median_step
    return float(np.min(differences))


def _evaluate_step_response(
    ideal_step_response: IdealStepResponse,
    lags: NDArray[np.float64],
) -> NDArray[np.float64]:
    try:
        raw_values = ideal_step_response(lags)
        values = np.asarray(raw_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "ideal_step_response must return real numeric values"
        ) from exc

    if values.ndim == 0 and lags.size == 1:
        values = values.reshape(1)
    elif values.ndim != 1:
        raise ValueError("ideal_step_response must return a one-dimensional array")
    if values.size != lags.size:
        raise ValueError(
            "ideal_step_response must return one value for every requested lag"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("ideal_step_response returned non-finite values")
    return values


def _coerce_signal_type(value: BoundarySignalType | str) -> BoundarySignalType:
    if isinstance(value, BoundarySignalType):
        return value
    if not isinstance(value, str):
        raise TypeError("signal_type must be pO2 or pH2O")
    aliases = {
        "po2": BoundarySignalType.PO2,
        "p_o2": BoundarySignalType.PO2,
        "ph2o": BoundarySignalType.PH2O,
        "p_h2o": BoundarySignalType.PH2O,
    }
    try:
        return aliases[value.strip().lower()]
    except KeyError as exc:
        raise ValueError("signal_type must be pO2 or pH2O") from exc


def _one_dimensional_finite_array(name: str, value: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    return result


def _readonly_copy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _nonnegative_finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _positive_finite_scalar(name: str, value: float) -> float:
    result = _nonnegative_finite_scalar(name, value)
    if result == 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result
