"""One-dimensional ECR forward kernels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.models.roots import ASYMPTOTIC_BI_THRESHOLD, solve_alpha_roots
from pyecr.models.root_cache import (resolve_alpha_roots, resolve_alpha_root_prefix,
                                    current_root_mode)

__all__ = [
    "KernelResult",
    "diffusion_controlled_fraction_1d",
    "finite_surface_fraction_1d",
]


@dataclass(frozen=True, slots=True)
class KernelResult:
    """Numerical result and convergence evidence for a one-dimensional kernel."""

    F: NDArray[np.float64]
    n_terms_used: int
    coefficient_sum: float
    diagnostics: Mapping[str, object]


def diffusion_controlled_fraction_1d(
    t_s: ArrayLike,
    diffusion_coefficient_cm2_s: float,
    full_dim_cm: float,
    *,
    term_tol: float = 1.0e-10,
    max_terms: int = 512,
) -> KernelResult:
    """Evaluate the explicit odd-series diffusion-controlled remaining fraction.

    The public geometry is the full dimension. Its half dimension is derived
    once and used by the modal decay rates. The finite-surface solver is not
    called and no artificial surface-exchange coefficient is introduced.
    """

    times = _validated_times(t_s)
    diffusion = _positive_finite_scalar(
        "diffusion_coefficient_cm2_s", diffusion_coefficient_cm2_s
    )
    full_dimension = _positive_finite_scalar("full_dim_cm", full_dim_cm)
    tolerance = _positive_finite_scalar("term_tol", term_tol)
    term_limit = _positive_integer("max_terms", max_terms)

    half_dim_cm = full_dimension / 2.0
    indices = np.arange(term_limit, dtype=np.float64)
    all_alphas = (indices + 0.5) * np.pi
    all_coefficients = 2.0 / all_alphas**2
    all_rates = all_alphas**2 * diffusion / half_dim_cm**2
    n_terms_used, truncation_error, converged = _select_diffusion_term_count(
        times,
        diffusion,
        half_dim_cm,
        all_coefficients,
        tolerance,
    )

    alphas = all_alphas[:n_terms_used].copy()
    coefficients = all_coefficients[:n_terms_used].copy()
    rates = all_rates[:n_terms_used].copy()
    coefficient_sum = float(np.sum(coefficients, dtype=np.float64))
    coefficient_sum_error = abs(1.0 - coefficient_sum)
    fraction = coefficients @ np.exp(-np.outer(rates, times))
    zero_time = times == 0.0
    fraction[zero_time] = 1.0
    truncation_error_by_time = _diffusion_truncation_error_by_time(
        times,
        diffusion,
        half_dim_cm,
        n_terms_used,
        coefficient_sum_error,
    )

    diagnostics = MappingProxyType(
        {
            "full_dim_cm": full_dimension,
            "half_dim_cm": half_dim_cm,
            "surface_mode": "instant",
            "alphas": alphas,
            "coefficients": coefficients,
            "rates_s_inv": rates,
            "coefficient_sum_error": coefficient_sum_error,
            "truncation_error": truncation_error,
            "truncation_error_by_time": truncation_error_by_time,
            "term_tol": tolerance,
            "max_terms": term_limit,
            "converged": converged,
            "has_t_zero": bool(np.any(zero_time)),
            "normalization_applied": False,
        }
    )
    return KernelResult(
        F=fraction.astype(np.float64, copy=False),
        n_terms_used=n_terms_used,
        coefficient_sum=coefficient_sum,
        diagnostics=diagnostics,
    )


def finite_surface_fraction_1d(
    t_s: ArrayLike,
    diffusion_coefficient_cm2_s: float,
    surface_exchange_coefficient_cm_s: float,
    full_dim_cm: float,
    *,
    term_tol: float = 1.0e-10,
    max_terms: int = 512,
) -> KernelResult:
    """Evaluate the raw one-dimensional finite-surface remaining fraction.

    The public geometry input is the full specimen dimension.  The half
    dimension is derived once and used for both Bi and modal decay rates.
    Positive-time values use the unnormalised truncated series.  At exactly
    zero time, the prescribed initial condition is returned directly while
    the missing coefficient mass is reported separately.
    """

    times = _validated_times(t_s)
    diffusion = _positive_finite_scalar(
        "diffusion_coefficient_cm2_s", diffusion_coefficient_cm2_s
    )
    surface_exchange = _positive_finite_scalar(
        "surface_exchange_coefficient_cm_s", surface_exchange_coefficient_cm_s
    )
    full_dimension = _positive_finite_scalar("full_dim_cm", full_dim_cm)
    tolerance = _positive_finite_scalar("term_tol", term_tol)
    term_limit = _positive_integer("max_terms", max_terms)

    half_dim_cm = full_dimension / 2.0
    bi = half_dim_cm * surface_exchange / diffusion

    if current_root_mode() == "on_demand_roots":
        def select_prefix(prefix):
            return _select_term_count(times, diffusion, half_dim_cm,
                                      _finite_surface_coefficients(prefix, bi), tolerance)
        (all_roots, all_root_residuals), selection = resolve_alpha_root_prefix(
            bi, term_limit, selector=select_prefix, solver=solve_alpha_roots)
    else:
        all_roots, all_root_residuals = resolve_alpha_roots(
            bi, term_limit, solver=solve_alpha_roots
        )
        selection = None
    all_coefficients = _finite_surface_coefficients(all_roots, bi)
    all_rates = all_roots**2 * diffusion / half_dim_cm**2

    n_terms_used, truncation_error, converged = selection or _select_term_count(
        times,
        diffusion,
        half_dim_cm,
        all_coefficients,
        tolerance,
    )

    roots = all_roots[:n_terms_used].copy()
    root_residuals = all_root_residuals[:n_terms_used].copy()
    coefficients = all_coefficients[:n_terms_used].copy()
    rates = all_rates[:n_terms_used].copy()
    coefficient_sum = float(np.sum(coefficients, dtype=np.float64))
    coefficient_sum_error = abs(1.0 - coefficient_sum)

    exponentials = np.exp(-np.outer(rates, times))
    fraction = coefficients @ exponentials
    zero_time = times == 0.0
    fraction[zero_time] = 1.0

    truncation_error_by_time = _truncation_error_by_time(
        times,
        diffusion,
        half_dim_cm,
        n_terms_used,
        coefficient_sum_error,
    )

    diagnostics = MappingProxyType(
        {
            "full_dim_cm": full_dimension,
            "half_dim_cm": half_dim_cm,
            "bi": bi,
            "roots": roots,
            "root_residuals": root_residuals,
            "root_residual_max_abs": float(np.max(np.abs(root_residuals))),
            "coefficients": coefficients,
            "rates_s_inv": rates,
            "coefficient_sum_error": coefficient_sum_error,
            "truncation_error": truncation_error,
            "truncation_error_by_time": truncation_error_by_time,
            "term_tol": tolerance,
            "max_terms": term_limit,
            "converged": converged,
            "has_t_zero": bool(np.any(zero_time)),
            "normalization_applied": False,
            "asymptotic_roots_used": bi > ASYMPTOTIC_BI_THRESHOLD,
        }
    )

    return KernelResult(
        F=fraction.astype(np.float64, copy=False),
        n_terms_used=n_terms_used,
        coefficient_sum=coefficient_sum,
        diagnostics=diagnostics,
    )


def _finite_surface_coefficients(
    roots: NDArray[np.float64], bi: float
) -> NDArray[np.float64]:
    scaled_denominator = 1.0 + 1.0 / bi + (roots / bi) ** 2
    return 2.0 / (roots**2 * scaled_denominator)


def _select_term_count(
    times: NDArray[np.float64],
    diffusion: float,
    half_dim_cm: float,
    coefficients: NDArray[np.float64],
    term_tol: float,
) -> tuple[int, float, bool]:
    positive_times = times[times > 0.0]
    if positive_times.size == 0:
        return 1, 0.0, True

    cumulative_coefficients = np.cumsum(coefficients, dtype=np.float64)
    smallest_positive_time = float(np.min(positive_times))
    selected_error = float("inf")

    for n_terms, coefficient_sum in enumerate(cumulative_coefficients, start=1):
        missing_coefficient_mass = abs(1.0 - float(coefficient_sum))
        next_root_lower_bound = n_terms * np.pi
        exponential_bound = np.exp(
            -(next_root_lower_bound**2)
            * diffusion
            * smallest_positive_time
            / half_dim_cm**2
        )
        selected_error = float(missing_coefficient_mass * exponential_bound)
        if selected_error <= term_tol:
            return n_terms, selected_error, True

    return coefficients.size, selected_error, False


def _select_diffusion_term_count(
    times: NDArray[np.float64],
    diffusion: float,
    half_dim_cm: float,
    coefficients: NDArray[np.float64],
    term_tol: float,
) -> tuple[int, float, bool]:
    positive_times = times[times > 0.0]
    if positive_times.size == 0:
        return 1, 0.0, True

    cumulative_coefficients = np.cumsum(coefficients, dtype=np.float64)
    smallest_positive_time = float(np.min(positive_times))
    selected_error = float("inf")
    for n_terms, coefficient_sum in enumerate(cumulative_coefficients, start=1):
        missing_coefficient_mass = abs(1.0 - float(coefficient_sum))
        next_alpha = (n_terms + 0.5) * np.pi
        selected_error = float(
            missing_coefficient_mass
            * np.exp(
                -(next_alpha**2)
                * diffusion
                * smallest_positive_time
                / half_dim_cm**2
            )
        )
        if selected_error <= term_tol:
            return n_terms, selected_error, True
    return coefficients.size, selected_error, False


def _truncation_error_by_time(
    times: NDArray[np.float64],
    diffusion: float,
    half_dim_cm: float,
    n_terms_used: int,
    coefficient_sum_error: float,
) -> NDArray[np.float64]:
    errors = np.full(times.shape, np.nan, dtype=np.float64)
    positive_time = times > 0.0
    next_root_lower_bound = n_terms_used * np.pi
    errors[positive_time] = coefficient_sum_error * np.exp(
        -(next_root_lower_bound**2)
        * diffusion
        * times[positive_time]
        / half_dim_cm**2
    )
    return errors


def _diffusion_truncation_error_by_time(
    times: NDArray[np.float64],
    diffusion: float,
    half_dim_cm: float,
    n_terms_used: int,
    coefficient_sum_error: float,
) -> NDArray[np.float64]:
    errors = np.full(times.shape, np.nan, dtype=np.float64)
    positive_time = times > 0.0
    next_alpha = (n_terms_used + 0.5) * np.pi
    errors[positive_time] = coefficient_sum_error * np.exp(
        -(next_alpha**2)
        * diffusion
        * times[positive_time]
        / half_dim_cm**2
    )
    return errors


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


def _positive_finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result
