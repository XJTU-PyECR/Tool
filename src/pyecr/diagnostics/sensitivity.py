"""Per-decade signal sensitivities evaluated with the production M1/M3 models.

``S_logD(t)`` means ``d y(t) / d log10(D)`` with every other fitted parameter
held fixed.  It differs by a factor of ``ln(10)`` from the literature's
``D * d y / d D`` convention, so the two names must not be interchanged.
The same definition applies to k and to the H/O parameters in M3.

Production forward models require real scalars and cast intermediate arrays to
float, so complex-step perturbations are unsuitable here.  Symmetric finite
differences use a fixed step in decades; a half-step calculation records an
independent convergence check for each curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10
from numbers import Real
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.fitting.single import prepare_single_objective
from pyecr.io.dataset import EcrDataset
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID

__all__ = [
    "SensitivityCurve",
    "SensitivityPresentation",
    "compute_sensitivity",
]


DEFAULT_STEP_DECADES = 1.0e-3

_M1_PARAMETERS = (
    ("D_cm2_s", "S_logD(t) = ∂y/∂log10(D)"),
    ("k_cm_s", "S_logk(t) = ∂y/∂log10(k)"),
)
_M3_PARAMETERS = (
    ("D_H_cm2_s", "S_logD_H(t) = ∂y/∂log10(D_H)"),
    ("k_H_cm_s", "S_logk_H(t) = ∂y/∂log10(k_H)"),
    ("D_O_cm2_s", "S_logD_O(t) = ∂y/∂log10(D_O)"),
    ("k_O_cm_s", "S_logk_O(t) = ∂y/∂log10(k_O)"),
)


@dataclass(frozen=True, slots=True)
class SensitivityCurve:
    """One signed signal derivative and its squared-sensitivity time profile.

    ``normalized_values`` divide the signed curve by its own peak absolute
    value.  Early/mid/late information fractions sum to one when the curve
    contains information.  The fractions partition observed times into thirds
    of the elapsed span and sum squared sensitivities of the observed points;
    this corresponds to equal independent observation noise.
    """

    parameter: str
    label: str
    values: tuple[float, ...]
    normalized_values: tuple[float, ...]
    peak_absolute: float
    peak_time_s: float
    early_information_fraction: float
    mid_information_fraction: float
    late_information_fraction: float
    convergence_relative_error: float


@dataclass(frozen=True, slots=True)
class SensitivityPresentation:
    """Read-only sensitivity result shared by Diagnostics, GUI, and Report."""

    available: bool
    curves: tuple[SensitivityCurve, ...] = ()
    reason: str | None = None
    method: str = "central_finite_difference"
    step_decades: float = DEFAULT_STEP_DECADES


def compute_sensitivity(
    dataset: EcrDataset,
    config: AnalysisConfig,
    fit_result: FitResult,
    *,
    step_decades: float = DEFAULT_STEP_DECADES,
) -> SensitivityPresentation:
    """Evaluate ``d y / d log10(parameter)`` at the fitted parameter values.

    The returned values have signal units per decade.  No signal equations are
    reimplemented here: both model branches call their production forward
    objects with the same solver settings and fixed nuisance parameters as the
    fit.  Derivatives are computed independently of ``FitResult.jacobian``,
    whose residual sign and nuisance columns vary by fitter.
    """

    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult")
    if isinstance(step_decades, bool) or not isinstance(step_decades, Real):
        raise TypeError("step_decades must be a real number")
    step = float(step_decades)
    if not isfinite(step) or step <= 0.0 or step >= 1.0:
        raise ValueError("step_decades must be finite and within (0, 1)")

    model_id = config.model_id
    if "fixed_d" in config.model_specific:
        times=dataset.processed_time
        curves=[]
        for name in fit_result.params:
            if not name.startswith("k_"):continue
            values=fit_result.jacobian[:,tuple(fit_result.params).index(name)]
            peak_index=int(np.argmax(np.abs(values)));peak=float(abs(values[peak_index]))
            norm=np.zeros_like(values) if peak==0 else values/peak
            early,mid,late=_information_fractions(times,values)
            curves.append(SensitivityCurve(name,f"Conditional sensitivity {name}; D fixed",tuple(map(float,values)),tuple(map(float,norm)),
                peak,float(times[peak_index]),early,mid,late,0.))
        return SensitivityPresentation(True,tuple(curves),reason="Full-row conditional optimizer Jacobian; finite-D kernel",method="conditional_full_jacobian")
    if model_id is ModelID.M8_DUAL_K:
        record=fit_result.diagnostics["dual_surface_k"]
        from pyecr.models.dual_surface_k import DualSurfaceKModel
        model=DualSurfaceKModel(ModelFactory().create(config).kernel,tuple(record["signs"]))
        times=dataset.processed_time
        params=record["reported_parameters"]
        j=model.jacobian(times,params,("k_1_cm_s","k_2_cm_s"))
        curves=[]
        for i in range(2):
            values=j[:,i]
            peak_index=int(np.argmax(np.abs(values)))
            peak=float(abs(values[peak_index]))
            norm=np.zeros_like(values) if peak==0 else values/peak
            early,mid,late=_information_fractions(times,values)
            curves.append(SensitivityCurve(f"k_{i+1}_cm_s",record["component_labels"][i],tuple(map(float,values)),
                tuple(map(float,norm)),peak,float(times[peak_index]),early,mid,late,0.))
        return SensitivityPresentation(True,tuple(curves),reason="Analytic whole-time effective-mode derivatives",method="analytic_dual_surface_k")
    if model_id is ModelID.M7_SINGLE_K:
        times = np.asarray(dataset.processed_time, dtype=np.float64)
        if times.size != fit_result.y_fit.size:
            raise ValueError("M7 sensitivity needs all aligned fit-eligible rows")
        parameters = {**fit_result.diagnostics["surface_k"]["fixed_parameters"], **fit_result.params}
        model = ModelFactory().create(config)
        values = model.jacobian(times, parameters, ("k_cm_s",))[:, 0]
        peak_index = int(np.argmax(np.abs(values)))
        peak = float(abs(values[peak_index]))
        normalized = np.zeros_like(values) if peak == 0 else values / peak
        early, mid, late = _information_fractions(times, values)
        curve = SensitivityCurve("k_cm_s", "S_logk(t) = dy/dlog10(k), analytic surface exchange",
            tuple(map(float, values)), tuple(map(float, normalized)), peak, float(times[peak_index]),
            early, mid, late, 0.0)
        return SensitivityPresentation(True, (curve,), reason="Exact analytic derivative; finite-difference step unused",
                                       method="analytic_surface_k", step_decades=step)
    if model_id not in {ModelID.M1_SINGLE_DK, ModelID.M3_DUAL_HO_DK}:
        return SensitivityPresentation(
            available=False,
            reason="Sensitivity curves require a finite-surface M1 or M3 fit.",
            step_decades=step,
        )

    times = np.asarray(dataset.processed_time, dtype=np.float64)
    if times.size != fit_result.y_fit.size:
        raise ValueError("fit_result and processed time must have equal length")
    if times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("processed time must have at least two increasing finite values")

    definitions = _M1_PARAMETERS if model_id is ModelID.M1_SINGLE_DK else _M3_PARAMETERS
    physical_params = {**config.model.fixed_parameters, **fit_result.params}
    missing = [name for name, _ in definitions if name not in physical_params]
    if missing:
        return SensitivityPresentation(
            available=False,
            reason="Selected fit lacks finite-surface parameter(s): " + ", ".join(missing),
            step_decades=step,
        )
    for name, _ in definitions:
        value = physical_params[name]
        if not isfinite(value) or value <= 0.0:
            return SensitivityPresentation(
                available=False,
                reason=f"{name} must be positive and finite for log sensitivity.",
                step_decades=step,
            )

    try:
        predictor = (
            _m1_predictor(dataset, config, fit_result)
            if model_id is ModelID.M1_SINGLE_DK
            else _m3_predictor(dataset, config, fit_result)
        )
        curves = tuple(
            _curve(name, label, times, physical_params, predictor, step)
            for name, label in definitions
        )
    except (ArithmeticError, OverflowError, TypeError, ValueError, RuntimeError) as exc:
        return SensitivityPresentation(
            available=False,
            reason=f"Production forward sensitivity evaluation failed: {exc}",
            step_decades=step,
        )
    return SensitivityPresentation(available=True, curves=curves, step_decades=step)


def _m1_predictor(
    dataset: EcrDataset,
    config: AnalysisConfig,
    fit_result: FitResult,
) -> Callable[[Mapping[str, float]], NDArray[np.float64]]:
    # The public objective builder supplies precisely the model and signal
    # context used by production M1 fitting, including nonideal gas boundaries.
    objective = prepare_single_objective(dataset, config)
    context = objective.context
    model = context.model
    times = context.time_s

    def predict(params: Mapping[str, float]) -> NDArray[np.float64]:
        theta = {
            "log10_D": log10(params["D_cm2_s"]),
            "log10_k": log10(params["k_cm_s"]),
            "amplitude": params.get("amplitude", fit_result.params.get("amplitude", context.forward_context.amplitude)),
            "baseline": params.get("baseline", fit_result.params.get("baseline", context.forward_context.baseline)),
        }
        return _prediction(model.predict(times, theta, context.forward_context), times.size)

    return predict


def _m3_predictor(
    dataset: EcrDataset,
    config: AnalysisConfig,
    fit_result: FitResult,
) -> Callable[[Mapping[str, float]], NDArray[np.float64]]:
    # Reuse the same validated adapter that sets up production M3 fitting.  In
    # particular this preserves explicit kernel options plus solver tolerances.
    from pyecr.services.analysis import _dual_fit_adapter

    fit_config, _ = _dual_fit_adapter(config)
    model = ModelFactory().create(config)
    if not isinstance(model, DualHOModel):
        raise TypeError("ModelFactory did not construct a production M3 model")
    times = np.asarray(dataset.processed_time, dtype=np.float64)
    baseline = fit_result.params.get(
        "y0",
        config.model.fixed_parameters.get("y0", float(dataset.processed_signal[0])),
    )

    def predict(params: Mapping[str, float]) -> NDArray[np.float64]:
        result = model.evaluate(
            times,
            D_H_cm2_s=params["D_H_cm2_s"],
            D_O_cm2_s=params["D_O_cm2_s"],
            A_H=params["A_H"],
            A_O=params["A_O"],
            y0=params.get("y0", baseline),
            k_H_cm_s=params["k_H_cm_s"],
            k_O_cm_s=params["k_O_cm_s"],
            kernel_options=fit_config.kernel_options,
        )
        return _prediction(result.total, times.size)

    return predict


def _curve(
    name: str,
    label: str,
    times: NDArray[np.float64],
    fitted_params: Mapping[str, float],
    predict: Callable[[Mapping[str, float]], NDArray[np.float64]],
    step: float,
) -> SensitivityCurve:
    values = _central_difference(name, fitted_params, predict, step)
    half_step_values = _central_difference(name, fitted_params, predict, step / 2.0)
    difference_norm = float(np.linalg.norm(values - half_step_values))
    reference_norm = float(np.linalg.norm(half_step_values))
    convergence = difference_norm / max(reference_norm, np.finfo(np.float64).tiny)
    if not isfinite(convergence):
        raise FloatingPointError(f"{name} convergence estimate is nonfinite")
    peak_index = int(np.argmax(np.abs(values)))
    peak = float(abs(values[peak_index]))
    normalized = np.zeros_like(values) if peak == 0.0 else values / peak
    early, mid, late = _information_fractions(times, values)
    return SensitivityCurve(
        parameter=name,
        label=label,
        values=tuple(float(value) for value in values),
        normalized_values=tuple(float(value) for value in normalized),
        peak_absolute=peak,
        peak_time_s=float(times[peak_index]),
        early_information_fraction=early,
        mid_information_fraction=mid,
        late_information_fraction=late,
        convergence_relative_error=convergence,
    )


def _central_difference(
    name: str,
    fitted_params: Mapping[str, float],
    predict: Callable[[Mapping[str, float]], NDArray[np.float64]],
    step: float,
) -> NDArray[np.float64]:
    center = float(fitted_params[name])
    factor = 10.0**step
    upper = dict(fitted_params)
    lower = dict(fitted_params)
    upper[name] = center * factor
    lower[name] = center / factor
    return (predict(upper) - predict(lower)) / (2.0 * step)


def _prediction(values: object, expected_size: int) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (expected_size,) or not np.all(np.isfinite(result)):
        raise FloatingPointError("production forward prediction is nonfinite or misaligned")
    return result


def _information_fractions(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[float, float, float]:
    span = float(times[-1] - times[0])
    first_cut = float(times[0] + span / 3.0)
    second_cut = float(times[0] + 2.0 * span / 3.0)
    weights = np.square(values)
    totals = np.asarray(
        (
            float(np.sum(weights[times < first_cut])),
            float(np.sum(weights[(times >= first_cut) & (times < second_cut)])),
            float(np.sum(weights[times >= second_cut])),
        ),
        dtype=np.float64,
    )
    total = float(np.sum(totals))
    if total == 0.0:
        return 0.0, 0.0, 0.0
    if not isfinite(total):
        raise FloatingPointError("squared sensitivity information is nonfinite")
    fractions = totals / total
    return tuple(float(value) for value in fractions)  # type: ignore[return-value]
