"""Shared-D/k inversion of multiple finite-surface ECR curves."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from pyecr.fitting.base import FitResult, ParameterSpace
from pyecr.models.ids import ModelID
from pyecr.models.rectangular import RectangularKernel
from pyecr.uncertainty.local import (
    ConfidenceInterval,
    LocalUncertaintyResult,
    estimate_local_uncertainty,
)

__all__ = [
    "GlobalCurve",
    "GlobalFitConfig",
    "GlobalFitResult",
    "fit_global_ecr",
]


@dataclass(frozen=True, slots=True)
class GlobalCurve:
    """One ECR curve and its curve-specific geometry/nuisance settings.

    ``noise_weight`` multiplies this curve's residuals in the global objective.
    For known standard deviation ``sigma``, use ``noise_weight=1/sigma``.
    The forward response is ``baseline + amplitude * (1 - F)``.
    """

    curve_id: str
    time_s: ArrayLike
    signal: ArrayLike
    geometry: RectangularKernel
    noise_weight: float = 1.0
    fit_baseline: bool = True
    fit_amplitude: bool = True
    baseline: float | None = None
    amplitude: float | None = None
    baseline_bounds: tuple[float, float] | None = None
    amplitude_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.curve_id, str) or not self.curve_id.strip():
            raise ValueError("curve_id must be a nonempty string")
        times = _vector("time_s", self.time_s)
        signal = _vector("signal", self.signal)
        if times.size != signal.size:
            raise ValueError("time_s and signal must have the same length")
        if times.size < 4:
            raise ValueError("each global curve requires at least four observations")
        if np.any(times < 0.0):
            raise ValueError("time_s values must be nonnegative")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if not isinstance(self.geometry, RectangularKernel):
            raise TypeError("geometry must be a RectangularKernel")
        weight = _positive_finite("noise_weight", self.noise_weight)
        if not isinstance(self.fit_baseline, bool):
            raise TypeError("fit_baseline must be boolean")
        if not isinstance(self.fit_amplitude, bool):
            raise TypeError("fit_amplitude must be boolean")
        baseline = (
            None if self.baseline is None else _finite_scalar("baseline", self.baseline)
        )
        amplitude = (
            None
            if self.amplitude is None
            else _finite_scalar("amplitude", self.amplitude)
        )
        baseline_bounds = _optional_bounds("baseline_bounds", self.baseline_bounds)
        amplitude_bounds = _optional_bounds(
            "amplitude_bounds", self.amplitude_bounds
        )
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "noise_weight", weight)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "baseline_bounds", baseline_bounds)
        object.__setattr__(self, "amplitude_bounds", amplitude_bounds)


@dataclass(frozen=True, slots=True)
class GlobalFitConfig:
    """Bounds, starts, kernel precision, and covariance policy."""

    log10_D_bounds: tuple[float, float] = (-12.0, -2.0)
    log10_k_bounds: tuple[float, float] = (-10.0, 0.0)
    initial_D_cm2_s: float | None = None
    initial_k_cm_s: float | None = None
    n_starts: int = 8
    random_seed: int = 20260916
    max_nfev: int = 3_000
    loss: str = "linear"
    term_tol: float = 1.0e-10
    max_terms: int = 128
    derivative_step_log10: float = 1.0e-5
    confidence_level: float = 0.95
    condition_number_limit: float = 1.0e12

    def __post_init__(self) -> None:
        d_bounds = _bounds("log10_D_bounds", self.log10_D_bounds)
        k_bounds = _bounds("log10_k_bounds", self.log10_k_bounds)
        initial_d = _optional_positive("initial_D_cm2_s", self.initial_D_cm2_s)
        initial_k = _optional_positive("initial_k_cm_s", self.initial_k_cm_s)
        if initial_d is not None and not d_bounds[0] <= np.log10(initial_d) <= d_bounds[1]:
            raise ValueError("initial_D_cm2_s lies outside log10_D_bounds")
        if initial_k is not None and not k_bounds[0] <= np.log10(initial_k) <= k_bounds[1]:
            raise ValueError("initial_k_cm_s lies outside log10_k_bounds")
        n_starts = _positive_integer("n_starts", self.n_starts)
        random_seed = _nonnegative_integer("random_seed", self.random_seed)
        max_nfev = _positive_integer("max_nfev", self.max_nfev)
        max_terms = _positive_integer("max_terms", self.max_terms)
        if self.loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("loss is not supported by scipy.optimize.least_squares")
        term_tol = _positive_finite("term_tol", self.term_tol)
        derivative_step = _positive_finite(
            "derivative_step_log10", self.derivative_step_log10
        )
        confidence = _finite_scalar("confidence_level", self.confidence_level)
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1")
        condition_limit = _positive_finite(
            "condition_number_limit", self.condition_number_limit
        )
        object.__setattr__(self, "log10_D_bounds", d_bounds)
        object.__setattr__(self, "log10_k_bounds", k_bounds)
        object.__setattr__(self, "initial_D_cm2_s", initial_d)
        object.__setattr__(self, "initial_k_cm_s", initial_k)
        object.__setattr__(self, "n_starts", n_starts)
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "max_nfev", max_nfev)
        object.__setattr__(self, "term_tol", term_tol)
        object.__setattr__(self, "max_terms", max_terms)
        object.__setattr__(self, "derivative_step_log10", derivative_step)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "condition_number_limit", condition_limit)


@dataclass(frozen=True, slots=True)
class GlobalFitResult:
    """Global fit, covariance, and curve-resolved residual output."""

    model_id: ModelID
    fit_result: FitResult
    uncertainty: LocalUncertaintyResult
    shared_params: Mapping[str, float]
    curve_params: Mapping[str, Mapping[str, float]]
    y_fit_by_curve: Mapping[str, NDArray[np.float64]]
    residuals_by_curve: Mapping[str, NDArray[np.float64]]
    weighted_residuals_by_curve: Mapping[str, NDArray[np.float64]]
    global_covariance: NDArray[np.float64]
    global_correlation: NDArray[np.float64]
    shared_covariance: NDArray[np.float64]
    shared_confidence_intervals: Mapping[str, ConfidenceInterval]
    diagnostics: Mapping[str, object]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_params", MappingProxyType(dict(self.shared_params)))
        object.__setattr__(
            self,
            "curve_params",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(values))
                    for key, values in self.curve_params.items()
                }
            ),
        )
        for field_name in (
            "y_fit_by_curve",
            "residuals_by_curve",
            "weighted_residuals_by_curve",
        ):
            values = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(
                    {key: _read_only_vector(array) for key, array in values.items()}
                ),
            )
        object.__setattr__(
            self,
            "global_covariance",
            _read_only_matrix(self.global_covariance, allow_nan=True),
        )
        object.__setattr__(
            self,
            "global_correlation",
            _read_only_matrix(self.global_correlation, allow_nan=True),
        )
        object.__setattr__(
            self,
            "shared_covariance",
            _read_only_matrix(self.shared_covariance, allow_nan=True),
        )
        object.__setattr__(
            self,
            "shared_confidence_intervals",
            MappingProxyType(dict(self.shared_confidence_intervals)),
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(slots=True)
class _CurveLayout:
    curve: GlobalCurve
    row_slice: slice
    baseline_index: int | None
    amplitude_index: int | None
    fixed_baseline: float
    fixed_amplitude: float


@dataclass(slots=True)
class _Candidate:
    q: NDArray[np.float64]
    residuals: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    SSE: float
    status: int
    nfev: int
    success: bool


class _Parameterization:
    def __init__(self, curves: tuple[GlobalCurve, ...], config: GlobalFitConfig) -> None:
        self.curves = curves
        self.config = config
        names = ["D_cm2_s", "k_cm_s"]
        spaces: list[ParameterSpace] = ["log", "log"]
        lower = [config.log10_D_bounds[0], config.log10_k_bounds[0]]
        upper = [config.log10_D_bounds[1], config.log10_k_bounds[1]]
        initial = [
            (
                np.log10(config.initial_D_cm2_s)
                if config.initial_D_cm2_s is not None
                else sum(config.log10_D_bounds) / 2.0
            ),
            (
                np.log10(config.initial_k_cm_s)
                if config.initial_k_cm_s is not None
                else sum(config.log10_k_bounds) / 2.0
            ),
        ]

        layouts: list[_CurveLayout] = []
        row_start = 0
        for curve in curves:
            baseline_initial, amplitude_initial = _local_initial_values(curve)
            baseline_index: int | None = None
            amplitude_index: int | None = None
            if curve.fit_baseline:
                baseline_bounds = curve.baseline_bounds or _automatic_baseline_bounds(
                    curve.signal
                )
                _ensure_inside(
                    f"{curve.curve_id} baseline",
                    baseline_initial,
                    baseline_bounds,
                )
                baseline_index = len(names)
                names.append(f"{curve.curve_id}:baseline")
                spaces.append("linear")
                lower.append(baseline_bounds[0])
                upper.append(baseline_bounds[1])
                initial.append(baseline_initial)
            if curve.fit_amplitude:
                amplitude_bounds = curve.amplitude_bounds or _automatic_amplitude_bounds(
                    curve.signal
                )
                _ensure_inside(
                    f"{curve.curve_id} amplitude",
                    amplitude_initial,
                    amplitude_bounds,
                )
                amplitude_index = len(names)
                names.append(f"{curve.curve_id}:amplitude")
                spaces.append("linear")
                lower.append(amplitude_bounds[0])
                upper.append(amplitude_bounds[1])
                initial.append(amplitude_initial)
            layouts.append(
                _CurveLayout(
                    curve=curve,
                    row_slice=slice(row_start, row_start + curve.time_s.size),
                    baseline_index=baseline_index,
                    amplitude_index=amplitude_index,
                    fixed_baseline=baseline_initial,
                    fixed_amplitude=amplitude_initial,
                )
            )
            row_start += curve.time_s.size

        self.names = tuple(names)
        self.spaces = tuple(spaces)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.initial = np.asarray(initial, dtype=np.float64)
        self.layouts = tuple(layouts)
        self.n_observations = row_start

    def start(self, index: int, generator: np.random.Generator) -> NDArray[np.float64]:
        if index == 0:
            return self.initial.copy()
        return generator.uniform(self.lower, self.upper)

    def decode(
        self, q: NDArray[np.float64]
    ) -> tuple[float, float, dict[str, dict[str, float]]]:
        diffusion = 10.0 ** float(q[0])
        surface_exchange = 10.0 ** float(q[1])
        local: dict[str, dict[str, float]] = {}
        for layout in self.layouts:
            baseline = (
                layout.fixed_baseline
                if layout.baseline_index is None
                else float(q[layout.baseline_index])
            )
            amplitude = (
                layout.fixed_amplitude
                if layout.amplitude_index is None
                else float(q[layout.amplitude_index])
            )
            local[layout.curve.curve_id] = {
                "baseline": baseline,
                "amplitude": amplitude,
            }
        return diffusion, surface_exchange, local

    def reported_params(self, q: NDArray[np.float64]) -> dict[str, float]:
        diffusion, surface_exchange, local = self.decode(q)
        result = {
            "D_cm2_s": diffusion,
            "k_cm_s": surface_exchange,
        }
        for layout in self.layouts:
            curve_values = local[layout.curve.curve_id]
            if layout.baseline_index is not None:
                result[f"{layout.curve.curve_id}:baseline"] = curve_values["baseline"]
            if layout.amplitude_index is not None:
                result[f"{layout.curve.curve_id}:amplitude"] = curve_values["amplitude"]
        return result


class _GlobalObjective:
    def __init__(self, parameterization: _Parameterization) -> None:
        self.parameterization = parameterization
        self._cached_q: NDArray[np.float64] | None = None
        self._cached_result: tuple[
            NDArray[np.float64],
            dict[str, NDArray[np.float64]],
            dict[str, NDArray[np.float64]],
        ] | None = None

    def residual(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        residuals, _, _ = self.evaluate(q)
        return residuals

    def evaluate(
        self, q: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        dict[str, NDArray[np.float64]],
        dict[str, NDArray[np.float64]],
    ]:
        if self._cached_q is not None and np.array_equal(q, self._cached_q):
            if self._cached_result is None:
                raise RuntimeError("global objective cache is inconsistent")
            return self._cached_result
        result = self._calculate(q)
        self._cached_q = q.copy()
        self._cached_result = result
        return result

    def _calculate(
        self, q: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.float64],
        dict[str, NDArray[np.float64]],
        dict[str, NDArray[np.float64]],
    ]:
        diffusion, surface_exchange, local = self.parameterization.decode(q)
        residuals = np.empty(
            self.parameterization.n_observations,
            dtype=np.float64,
        )
        predictions: dict[str, NDArray[np.float64]] = {}
        relaxed_fractions: dict[str, NDArray[np.float64]] = {}
        config = self.parameterization.config
        for layout in self.parameterization.layouts:
            curve = layout.curve
            kernel = curve.geometry.evaluate(
                curve.time_s,
                diffusion,
                surface_exchange,
                term_tol=config.term_tol,
                max_terms=config.max_terms,
            )
            relaxed = 1.0 - kernel.F
            values = local[curve.curve_id]
            prediction = values["baseline"] + values["amplitude"] * relaxed
            residuals[layout.row_slice] = curve.noise_weight * (
                prediction - curve.signal
            )
            predictions[curve.curve_id] = prediction
            relaxed_fractions[curve.curve_id] = relaxed
        return residuals, predictions, relaxed_fractions

    def jacobian(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        step = self.parameterization.config.derivative_step_log10
        jacobian = np.zeros(
            (self.parameterization.n_observations, q.size),
            dtype=np.float64,
        )
        for column in (0, 1):
            upper = q.copy()
            lower = q.copy()
            upper[column] += step
            lower[column] -= step
            upper_residuals, _, _ = self._calculate(upper)
            lower_residuals, _, _ = self._calculate(lower)
            jacobian[:, column] = (upper_residuals - lower_residuals) / (2.0 * step)

        _, _, relaxed_fractions = self.evaluate(q)
        for layout in self.parameterization.layouts:
            curve = layout.curve
            if layout.baseline_index is not None:
                jacobian[layout.row_slice, layout.baseline_index] = curve.noise_weight
            if layout.amplitude_index is not None:
                jacobian[layout.row_slice, layout.amplitude_index] = (
                    curve.noise_weight * relaxed_fractions[curve.curve_id]
                )
        return jacobian


def fit_global_ecr(
    curves: list[GlobalCurve] | tuple[GlobalCurve, ...],
    *,
    config: GlobalFitConfig | None = None,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> GlobalFitResult:
    """Fit shared ``D`` and ``k`` to concatenated weighted curve residuals."""

    validated_curves = _validate_curves(curves)
    policy = GlobalFitConfig() if config is None else config
    if not isinstance(policy, GlobalFitConfig):
        raise TypeError("config must be GlobalFitConfig or None")
    parameterization = _Parameterization(validated_curves, policy)
    if parameterization.n_observations <= len(parameterization.names):
        raise ValueError("global fit needs more observations than free parameters")
    objective = _GlobalObjective(parameterization)
    check_cancel = cancel_check or (lambda: None)
    report = progress_callback or (lambda fraction, message: None)
    check_cancel()
    report(0.02, "Global configuration validated")
    generator = np.random.default_rng(policy.random_seed)
    start_audit=[]
    candidates: list[_Candidate] = []
    failure_messages: list[str] = []

    for start_index in range(policy.n_starts):
        check_cancel()
        report(
            0.05 + 0.75 * start_index / policy.n_starts,
            f"Global multistart {start_index + 1}/{policy.n_starts}",
        )
        start = parameterization.start(start_index, generator)
        start_audit.append({'start_index':start_index,'coordinates':start.tolist()})

        def cancellable_residual(q):  # type: ignore[no-untyped-def]
            check_cancel()
            return objective.residual(q)

        def cancellable_jacobian(q):  # type: ignore[no-untyped-def]
            check_cancel()
            return objective.jacobian(q)

        try:
            optimizer = least_squares(
                cancellable_residual,
                start,
                jac=cancellable_jacobian,
                bounds=(parameterization.lower, parameterization.upper),
                loss=policy.loss,
                max_nfev=policy.max_nfev,
                x_scale="jac",
            )
            residuals = objective.residual(optimizer.x).copy()
            sse = float(np.dot(residuals, residuals))
            jacobian = objective.jacobian(optimizer.x)
            if not np.isfinite(sse) or not np.all(np.isfinite(jacobian)):
                raise FloatingPointError("optimizer returned non-finite output")
            candidates.append(
                _Candidate(
                    q=np.asarray(optimizer.x, dtype=np.float64).copy(),
                    residuals=residuals,
                    jacobian=jacobian,
                    SSE=sse,
                    status=int(optimizer.status),
                    nfev=int(optimizer.nfev),
                    success=bool(optimizer.success),
                )
            )
        except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
            failure_messages.append(
                f"start {start_index}: {type(exc).__name__}: {exc}"
            )
    if not candidates:
        raise RuntimeError("global fit produced no valid optimization candidate")
    check_cancel()
    report(0.84, "Calculating global covariance and curve residuals")
    candidates.sort(key=lambda candidate: candidate.SSE)
    best = candidates[0]
    _, predictions, _ = objective.evaluate(best.q)
    diffusion, surface_exchange, curve_params = parameterization.decode(best.q)

    warnings: list[str] = []
    if failure_messages:
        warnings.append(
            f"GLOBAL_MULTISTART_FAILURES: {len(failure_messages)} of "
            f"{policy.n_starts} starts failed."
        )
    if not best.success:
        warnings.append(
            f"GLOBAL_OPTIMIZER_NOT_CONVERGED: exit_status={best.status}."
        )
    if policy.loss != "linear":
        warnings.append(
            "GLOBAL_ROBUST_LOSS_COVARIANCE_APPROXIMATE: covariance uses the "
            "returned local Jacobian with an ordinary least-squares scale."
        )

    weighted_observed = np.concatenate(
        [curve.noise_weight * curve.signal for curve in validated_curves]
    )
    weighted_fitted = np.concatenate(
        [curve.noise_weight * predictions[curve.curve_id] for curve in validated_curves]
    )
    fit_result = _build_fit_result(
        weighted_observed,
        weighted_fitted,
        params=parameterization.reported_params(best.q),
        param_space={
            name: space
            for name, space in zip(
                parameterization.names,
                parameterization.spaces,
                strict=True,
            )
        },
        exit_status=best.status,
        nfev=best.nfev,
        jacobian=best.jacobian,
        warnings=warnings,
    )
    uncertainty = estimate_local_uncertainty(
        fit_result,
        confidence_level=policy.confidence_level,
        condition_threshold=policy.condition_number_limit,
    )
    fit_result = uncertainty.annotated_fit_result
    from pyecr.fitting.reproducibility import record_randomness
    fit_result=record_randomness(fit_result,seed=policy.random_seed,stochastic=policy.n_starts>1,
                                algorithm='global-first-initial-then-PCG64-v1',starts=start_audit)

    residuals_by_curve = {
        curve.curve_id: curve.signal - predictions[curve.curve_id]
        for curve in validated_curves
    }
    weighted_residuals = {
        curve.curve_id: curve.noise_weight * residuals_by_curve[curve.curve_id]
        for curve in validated_curves
    }
    covariance = uncertainty.covariance
    shared_intervals = {
        name: uncertainty.confidence_intervals[name]
        for name in ("D_cm2_s", "k_cm_s")
    }
    all_warnings = tuple(dict.fromkeys((*fit_result.warnings, *warnings)))
    result = GlobalFitResult(
        model_id=ModelID.M5_GLOBAL,
        fit_result=fit_result,
        uncertainty=uncertainty,
        shared_params={
            "D_cm2_s": diffusion,
            "k_cm_s": surface_exchange,
        },
        curve_params=curve_params,
        y_fit_by_curve=predictions,
        residuals_by_curve=residuals_by_curve,
        weighted_residuals_by_curve=weighted_residuals,
        global_covariance=covariance,
        global_correlation=uncertainty.correlation_matrix,
        shared_covariance=covariance[:2, :2],
        shared_confidence_intervals=shared_intervals,
        diagnostics={
            "n_curves": len(validated_curves),
            "curve_ids": tuple(curve.curve_id for curve in validated_curves),
            "n_observations": parameterization.n_observations,
            "n_free_parameters": len(parameterization.names),
            "parameter_order": parameterization.names,
            "noise_weights": MappingProxyType(
                {curve.curve_id: curve.noise_weight for curve in validated_curves}
            ),
            "weighted_objective": True,
            "shared_parameters": ("D_cm2_s", "k_cm_s"),
            "failed_start_messages": tuple(failure_messages),
            "candidate_SSE": tuple(candidate.SSE for candidate in candidates),
        },
        warnings=all_warnings,
    )
    check_cancel()
    report(1.0, "Global fit complete")
    return result


def _build_fit_result(
    observed: NDArray[np.float64],
    fitted: NDArray[np.float64],
    *,
    params: Mapping[str, float],
    param_space: Mapping[str, ParameterSpace],
    exit_status: int,
    nfev: int,
    jacobian: NDArray[np.float64],
    warnings: list[str],
) -> FitResult:
    residuals = observed - fitted
    sse = float(np.dot(residuals, residuals))
    n_observations = observed.size
    n_parameters = len(params)
    rmse = float(np.sqrt(sse / n_observations))
    centred = observed - float(np.mean(observed))
    sst = float(np.dot(centred, centred))
    r_squared = float("nan") if sst == 0.0 else 1.0 - sse / sst
    mean_sse = max(sse / n_observations, np.finfo(np.float64).tiny)
    aic = n_observations * log(mean_sse) + 2.0 * n_parameters
    bic = n_observations * log(mean_sse) + n_parameters * log(n_observations)
    if n_observations <= n_parameters + 1:
        aicc = float("inf")
        warnings.append("AICC_UNDEFINED: n_observations <= n_parameters + 1.")
    else:
        aicc = aic + (
            2.0 * n_parameters * (n_parameters + 1)
            / (n_observations - n_parameters - 1)
        )
    return FitResult(
        params=params,
        param_space=param_space,
        y_fit=fitted,
        residuals=residuals,
        SSE=sse,
        RMSE=rmse,
        R2=r_squared,
        AIC=aic,
        AICc=aicc,
        BIC=bic,
        exit_status=exit_status,
        nfev=nfev,
        jacobian=jacobian,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_curves(
    curves: list[GlobalCurve] | tuple[GlobalCurve, ...],
) -> tuple[GlobalCurve, ...]:
    if not isinstance(curves, (list, tuple)):
        raise TypeError("curves must be a list or tuple of GlobalCurve")
    result = tuple(curves)
    if not result:
        raise ValueError("at least one curve is required")
    if not all(isinstance(curve, GlobalCurve) for curve in result):
        raise TypeError("curves must contain only GlobalCurve")
    identifiers = tuple(curve.curve_id for curve in result)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("curve_id values must be unique")
    return result


def _local_initial_values(curve: GlobalCurve) -> tuple[float, float]:
    baseline = float(curve.signal[0]) if curve.baseline is None else curve.baseline
    amplitude = (
        float(curve.signal[-1] - curve.signal[0])
        if curve.amplitude is None
        else curve.amplitude
    )
    return baseline, amplitude


def _automatic_baseline_bounds(signal: NDArray[np.float64]) -> tuple[float, float]:
    span = float(np.ptp(signal))
    scale = max(span, float(np.max(np.abs(signal))), 1.0e-8)
    pad = 2.0 * scale
    return float(np.min(signal) - pad), float(np.max(signal) + pad)


def _automatic_amplitude_bounds(signal: NDArray[np.float64]) -> tuple[float, float]:
    span = float(np.ptp(signal))
    scale = max(span, float(np.max(np.abs(signal))), 1.0e-8)
    limit = 10.0 * scale
    return -limit, limit


def _ensure_inside(
    name: str,
    value: float,
    bounds: tuple[float, float],
) -> None:
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"initial {name} lies outside its bounds")


def _optional_bounds(
    name: str,
    value: tuple[float, float] | None,
) -> tuple[float, float] | None:
    return None if value is None else _bounds(name, value)


def _bounds(name: str, value: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    lower = _finite_scalar(f"{name}[0]", value[0])
    upper = _finite_scalar(f"{name}[1]", value[1])
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    return lower, upper


def _optional_positive(name: str, value: float | None) -> float | None:
    return None if value is None else _positive_finite(name, value)


def _vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    result.flags.writeable = False
    return result


def _read_only_vector(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    result.flags.writeable = False
    return result


def _read_only_matrix(
    values: ArrayLike,
    *,
    allow_nan: bool,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 2:
        raise ValueError("covariance and correlation arrays must be two-dimensional")
    if allow_nan:
        if np.any(np.isinf(result)):
            raise ValueError("covariance and correlation cannot contain infinity")
    elif not np.all(np.isfinite(result)):
        raise ValueError("covariance and correlation must be finite")
    result.flags.writeable = False
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
