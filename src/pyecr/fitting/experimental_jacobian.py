"""Opt-in P58B hybrid Jacobian for the production M1/M3 forward paths.

Kinetic columns are symmetric log10-coordinate differences of the real
production forward. Only signal amplitudes and baselines use exact algebraic
columns. This module does not approximate roots or alter the forward model.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from math import log
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from pyecr.fitting.progress import count_fit_call

EXPERIMENTAL_MODE = "experimental_hybrid_v1"
LEGACY_MODE = "legacy"
LOG10_STEP = 1.0e-5
_LN10 = log(10.0)


class HybridFallback(Exception):
    """Restart this MultiStart candidate with the unchanged legacy Jacobian."""


def validate_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {LEGACY_MODE, EXPERIMENTAL_MODE}:
        raise ValueError("jacobian_mode must be legacy or experimental_hybrid_v1")
    return value


def _kernel_signature(kernel_result: object) -> tuple[tuple[int | None, bool | None], ...]:
    axes = getattr(kernel_result, "axis_results", None)
    if axes is None:
        raise HybridFallback("kernel has no axis diagnostics")
    signature = []
    for axis in axes:
        if axis is None:
            signature.append((None, None))
            continue
        evidence = axis.diagnostics
        if evidence.get("asymptotic_roots_used", False):
            raise HybridFallback("large-Bi approximate-root branch")
        if not evidence.get("converged", False):
            raise HybridFallback("finite-axis term tolerance not met")
        signature.append((int(axis.n_terms_used), False))
    return tuple(signature)


def _checked_forward(call: Callable[[], object]) -> object:
    try:
        count_fit_call("forward")
        return call()
    except CancelledError:
        raise
    except HybridFallback:
        raise
    except (ArithmeticError, FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
        raise HybridFallback(f"production forward failed: {type(exc).__name__}: {exc}") from exc


def _near_bounds(q: NDArray[np.float64], lower: NDArray[np.float64],
                 upper: NDArray[np.float64], columns: tuple[int, ...]) -> None:
    if any(q[i] - lower[i] <= 4 * LOG10_STEP or
           upper[i] - q[i] <= 4 * LOG10_STEP for i in columns):
        raise HybridFallback("kinetic or label coordinate near bound")


def single_hybrid_jacobian(evaluator: object) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """M1/M0 Jacobian in the exact ordered free optimizer coordinates."""
    from pyecr.models.single import SingleEcrModel

    context = evaluator.context
    model = context.model
    names = context.parameter_names
    kinetic = tuple(i for i, name in enumerate(names) if name in {"log10_D", "log10_k"})
    if not isinstance(model, SingleEcrModel):
        raise HybridFallback("nonideal boundary is outside experimental derivative domain")

    def forward(q: NDArray[np.float64]) -> object:
        theta = dict(context.fixed_coordinates)
        theta.update(zip(names, map(float, q), strict=True))
        return _checked_forward(lambda: model.evaluate(
            context.time_s, theta, context.forward_context))

    def jac(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if evaluator.cancel_check is not None:
            evaluator.cancel_check()
        q = np.asarray(q, dtype=np.float64)
        _near_bounds(q, context.lower_bounds, context.upper_bounds, kinetic)
        base = forward(q)
        signature = _kernel_signature(base.kernel_result)
        result = np.empty((context.time_s.size, len(names)), dtype=np.float64)
        for column, name in enumerate(names):
            if name == "amplitude":
                result[:, column] = 1.0 - base.F
            elif name == "baseline":
                result[:, column] = 1.0
            else:
                plus = q.copy(); minus = q.copy()
                plus[column] += LOG10_STEP; minus[column] -= LOG10_STEP
                if evaluator.cancel_check is not None:
                    evaluator.cancel_check()
                high = forward(plus)
                low = forward(minus)
                if _kernel_signature(high.kernel_result) != signature or _kernel_signature(low.kernel_result) != signature:
                    raise HybridFallback("finite-axis selected-term count changed")
                result[:, column] = (high.y - low.y) / (2.0 * LOG10_STEP)
        if not np.all(np.isfinite(result)):
            raise HybridFallback("nonfinite hybrid Jacobian")
        return result

    return jac


def dual_hybrid_jacobian(evaluator: object, parameterization: object,
                         cancel_check: Callable[[], None] | None = None
                         ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """M3 Jacobian in q, including the nonlinear D_H>=D_O label map."""
    if not parameterization.is_m3:
        raise ValueError("P58B dual Jacobian applies only to M3")
    context = evaluator.context
    model = context.model
    lower, upper = parameterization.lower, parameterization.upper
    kinetic = (0, 1, 4, 5)

    def forward(q: NDArray[np.float64]) -> object:
        p: Mapping[str, float] = parameterization.decode(q)
        return _checked_forward(lambda: model.evaluate(
            context.processed_time_s,
            D_H_cm2_s=p["D_H_cm2_s"], D_O_cm2_s=p["D_O_cm2_s"],
            A_H=p["A_H"], A_O=p["A_O"], y0=p["y0"],
            k_H_cm_s=p["k_H_cm_s"], k_O_cm_s=p["k_O_cm_s"],
            kernel_options=context.kernel_options))

    def signature(result: object) -> tuple[object, object]:
        return (_kernel_signature(result.kernel_H_result),
                _kernel_signature(result.kernel_O_result))

    def jac(q: NDArray[np.float64]) -> NDArray[np.float64]:
        if cancel_check is not None:
            cancel_check()
        q = np.asarray(q, dtype=np.float64)
        _near_bounds(q, lower, upper, kinetic)
        base = forward(q)
        base_signature = signature(base)
        result = np.empty((context.processed_time_s.size, q.size), dtype=np.float64)
        for column in range(q.size):
            if column == 2:
                result[:, column] = _LN10 * base.component_H
            elif column == 3:
                result[:, column] = _LN10 * base.component_O
            elif column == 6:
                result[:, column] = 1.0
            else:
                plus = q.copy(); minus = q.copy()
                plus[column] += LOG10_STEP; minus[column] -= LOG10_STEP
                if cancel_check is not None:
                    cancel_check()
                high = forward(plus)
                low = forward(minus)
                if signature(high) != base_signature or signature(low) != base_signature:
                    raise HybridFallback("finite-axis selected-term count changed")
                result[:, column] = (high.total - low.total) / (2.0 * LOG10_STEP)
        if not np.all(np.isfinite(result)):
            raise HybridFallback("nonfinite hybrid Jacobian")
        return result

    return jac
