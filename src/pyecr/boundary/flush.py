"""Exponential CISTR flush-time correction for normalized ECR responses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

__all__ = [
    "FlushCorrectionResult",
    "FlushDiagnostic",
    "exponential_flush_correction",
]


class FlushDiagnostic(str, Enum):
    """Coarse screening label based on the first material time constant."""

    NO_CORRECTION_NEEDED = "No correction needed"
    CORRECTION_RECOMMENDED = "Correction recommended"
    SEVERE_LIMITATION = "Severely limited"


@dataclass(frozen=True, slots=True)
class FlushCorrectionResult:
    """Flush-corrected curve together with comparison and diagnostic data."""

    normalized_response: NDArray[np.float64]
    ideal_response: NDArray[np.float64]
    ratio: float
    diagnostic: FlushDiagnostic
    diagnostics: Mapping[str, object]


IdealStepResponse = Callable[[NDArray[np.float64]], ArrayLike]

_NO_CORRECTION_MAX_RATIO = 0.1
_SEVERE_LIMITATION_MIN_RATIO = 3.0
_INITIAL_RESPONSE_ATOL = 1.0e-10
_SOLVER_RTOL = 2.0e-10
_SOLVER_ATOL = 2.0e-12


def exponential_flush_correction(
    t_s: ArrayLike,
    ideal_step_response: IdealStepResponse,
    flush_time_s: float,
    *,
    tau1_s: float,
) -> FlushCorrectionResult:
    """Apply the den Otter single-exponential CISTR boundary response.

    ``ideal_step_response`` returns the normalized material response to an
    ideal gas step, with zero at ``t=0`` and unity at equilibrium.  The CISTR
    response is evaluated as the equivalent first-order convolution rather
    than by dividing modal expressions by ``tau_n - flush_time_s``.  It is
    therefore continuous when a material mode has ``tau_n == flush_time_s``.

    ``tau1_s`` is the slowest material-mode time constant.  It is used only
    for the requested screening ratio and does not alter the corrected curve.
    It is explicit because an arbitrary response callable does not uniquely
    expose its modal time constants.
    """

    times = _validated_times(t_s)
    flush_time = _nonnegative_finite_scalar("flush_time_s", flush_time_s)
    tau1 = _positive_finite_scalar("tau1_s", tau1_s)
    if not callable(ideal_step_response):
        raise TypeError("ideal_step_response must be callable")

    initial_ideal = _evaluate_ideal_response(
        ideal_step_response, np.array([0.0], dtype=np.float64)
    )[0]
    if not np.isclose(initial_ideal, 0.0, rtol=0.0, atol=_INITIAL_RESPONSE_ATOL):
        raise ValueError(
            "ideal_step_response must have normalized response at t=0 equal to 0"
        )

    ideal_values = _evaluate_ideal_response(ideal_step_response, times)
    ratio = flush_time / tau1
    diagnostic = _classify_ratio(ratio)

    if flush_time == 0.0:
        corrected_values = ideal_values.copy()
        limit_used: str | None = "ideal_step"
        solver_name: str | None = None
        nfev = 0
    else:
        corrected_values, nfev = _solve_cistr_convolution(
            times,
            ideal_step_response,
            flush_time,
            tau1,
        )
        limit_used = None
        solver_name = "LSODA"

    correction = corrected_values - ideal_values
    diagnostics = MappingProxyType(
        {
            "algorithm_version": "cistr-lsoda/1",
            "method": "den_otter_cistr_first_order_convolution",
            "flush_time_s": flush_time,
            "tau1_s": tau1,
            "ratio": ratio,
            "diagnostic": diagnostic.value,
            "diagnostic_basis": "tau_f/tau1 coarse screening",
            "diagnostic_note": (
                "Higher material modes can require correction even when the "
                "tau_f/tau1 ratio is small."
            ),
            "no_correction_max_ratio": _NO_CORRECTION_MAX_RATIO,
            "severe_limitation_min_ratio": _SEVERE_LIMITATION_MIN_RATIO,
            "limit_used": limit_used,
            "solver": solver_name,
            "solver_rtol": None if solver_name is None else _SOLVER_RTOL,
            "solver_atol": None if solver_name is None else _SOLVER_ATOL,
            "nfev": nfev,
            "max_abs_correction": float(np.max(np.abs(correction))),
            "tau_equal_safe_formulation": True,
        }
    )

    return FlushCorrectionResult(
        normalized_response=corrected_values,
        ideal_response=ideal_values,
        ratio=ratio,
        diagnostic=diagnostic,
        diagnostics=diagnostics,
    )


def _solve_cistr_convolution(
    times: NDArray[np.float64],
    ideal_step_response: IdealStepResponse,
    flush_time_s: float,
    tau1_s: float,
) -> tuple[NDArray[np.float64], int]:
    unique_times, inverse = np.unique(times, return_inverse=True)
    maximum_time = float(unique_times[-1])
    if maximum_time == 0.0:
        return np.zeros_like(times), 0

    evaluation_times = np.union1d(np.array([0.0]), unique_times)

    def right_hand_side(time: float, state: NDArray[np.float64]) -> list[float]:
        forcing = _evaluate_ideal_response(
            ideal_step_response, np.array([time], dtype=np.float64)
        )[0]
        return [(forcing - state[0]) / flush_time_s]

    first_step = min(flush_time_s, tau1_s, maximum_time) * 1.0e-3
    maximum_step = min(tau1_s / 20.0, maximum_time)
    solution = solve_ivp(
        right_hand_side,
        (0.0, maximum_time),
        np.array([0.0], dtype=np.float64),
        method="LSODA",
        t_eval=evaluation_times,
        rtol=_SOLVER_RTOL,
        atol=_SOLVER_ATOL,
        first_step=first_step,
        max_step=maximum_step,
    )
    if not solution.success:
        raise RuntimeError(f"CISTR convolution failed: {solution.message}")

    unique_positions = np.searchsorted(evaluation_times, unique_times)
    corrected_unique = solution.y[0, unique_positions]
    corrected = corrected_unique[inverse]
    corrected[times == 0.0] = 0.0
    return corrected.astype(np.float64, copy=False), int(solution.nfev)


def _evaluate_ideal_response(
    ideal_step_response: IdealStepResponse,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    try:
        raw_values = ideal_step_response(times)
        values = np.asarray(raw_values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "ideal_step_response must return real numeric values"
        ) from exc

    if values.ndim == 0 and times.size == 1:
        values = values.reshape(1)
    else:
        values = values.reshape(-1)
    if values.size != times.size:
        raise ValueError(
            "ideal_step_response must return one value for every requested time"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("ideal_step_response returned non-finite values")
    return values


def _classify_ratio(ratio: float) -> FlushDiagnostic:
    if ratio <= _NO_CORRECTION_MAX_RATIO:
        return FlushDiagnostic.NO_CORRECTION_NEEDED
    if ratio < _SEVERE_LIMITATION_MIN_RATIO:
        return FlushDiagnostic.CORRECTION_RECOMMENDED
    return FlushDiagnostic.SEVERE_LIMITATION


def _validated_times(t_s: ArrayLike) -> NDArray[np.float64]:
    try:
        times = np.asarray(t_s, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError("t_s must contain real numeric values") from exc

    if times.size == 0:
        raise ValueError("t_s must contain at least one time")
    if not np.all(np.isfinite(times)):
        raise ValueError("t_s values must be finite")
    if np.any(times < 0.0):
        raise ValueError("t_s values must be nonnegative")
    return times


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
