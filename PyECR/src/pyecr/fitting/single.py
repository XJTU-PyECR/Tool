"""Production inversion for M0/M1 single-process ECR models."""

from __future__ import annotations

from pyecr.fitting.reproducibility import seeded_analysis_fit,record_randomness

from dataclasses import dataclass, replace
from math import isfinite, log
from numbers import Integral, Real
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from pyecr.fitting.solver_strategy import least_squares
from pyecr.fitting.coarse_to_fine import current_screening, screen_and_refine

from pyecr.boundary.composition import BoundaryComposedSingleEcrModel
from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitCandidate, FitResult, ParameterSpace
from pyecr.fitting.objective import ObjectiveContext, SingleObjectiveEvaluator
from pyecr.fitting.experimental_jacobian import (
    EXPERIMENTAL_MODE, HybridFallback, single_hybrid_jacobian, validate_mode,
)
from pyecr.io.dataset import EcrDataset
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.models.single import SingleEcrContext, SingleEcrModel
from pyecr.uncertainty.local import LocalUncertaintyResult, estimate_local_uncertainty

__all__ = ["fit_single_ecr", "prepare_single_objective"]


CancelCheck = Callable[[], None]
ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class _SingleOptions:
    normalized_data: bool
    fit_amplitude: bool
    fit_baseline: bool
    n_starts: int
    n_candidates: int
    random_seed: int
    condition_threshold: float
    bi_surface_limit: float
    bi_diffusion_limit: float
    jacobian_mode: str


@dataclass(slots=True)
class _Candidate:
    start_index: int
    initial_coordinates: tuple[float, ...]
    coordinates: NDArray[np.float64]
    params: dict[str, float]
    prediction: NDArray[np.float64]
    residuals: NDArray[np.float64]
    SSE: float
    exit_status: int
    nfev: int
    success: bool
    jacobian: NDArray[np.float64]


@seeded_analysis_fit
def fit_single_ecr(
    dataset: EcrDataset,
    config: AnalysisConfig,
    *,
    cancel_check: CancelCheck | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit an M0 or M1 model to one immutable ECR dataset.

    D and k are optimized in log10 coordinates. Normalized data fix amplitude
    to one and baseline to zero. Raw-signal nuisance parameters are introduced
    only through explicit ``single_ecr`` model-specific options.
    """

    evaluator = prepare_single_objective(
        dataset, config, cancel_check=cancel_check
    )
    from pyecr.fitting.progress import (
        fit_counters_available, fit_start, fit_candidate, fit_start_finished, fit_stage,
    )
    fit_counters_available()
    options = _options(config)
    objective_context = evaluator.context
    model = objective_context.model
    times = objective_context.time_s
    observed = objective_context.observed
    names = objective_context.parameter_names
    spaces = objective_context.parameter_spaces
    lower = objective_context.lower_bounds
    upper = objective_context.upper_bounds
    context = objective_context.forward_context

    rng = np.random.default_rng(options.random_seed)
    inward_lower = np.nextafter(lower, upper)
    inward_upper = np.nextafter(upper, lower)
    starts = rng.uniform(inward_lower, inward_upper, size=(options.n_starts, len(names)))
    candidates: list[_Candidate] = []
    total_nfev = 0
    experimental_starts: list[dict[str, object]] = []
    hybrid_jac = None
    initial_fallback: str | None = None
    if options.jacobian_mode == EXPERIMENTAL_MODE:
        try:
            hybrid_jac = single_hybrid_jacobian(evaluator)
        except HybridFallback as exc:
            initial_fallback = str(exc)
    progress = progress_callback or _ignore_progress

    refined = None
    screening_stage = None
    if current_screening() is not None:
        if current_screening().original_indices is None:
            current_screening().original_indices = np.flatnonzero(dataset.mask)
        if config.fit.loss != "linear" or config.model.boundary_mode != "ideal_step" or options.jacobian_mode != "legacy":
            raise ValueError("P60 supports linear loss, ideal_step and legacy numeric Jacobian only")
        def subset_factory(indices):
            return replace(evaluator, context=replace(
                objective_context, time_s=times[indices], observed=observed[indices]
            )).optimizer_residual
        def physical_parameters(q):
            params = _reported_params(names, q)
            scale = max(float(np.ptp(observed)), np.finfo(float).eps)
            return params, [float(v) if space == "log" else float(v) / scale
                            for v, space in zip(q, spaces, strict=True)]
        refined, screening_stage = screen_and_refine(
            times=times, starts=starts, lower=lower, upper=upper,
            subset_factory=subset_factory, full_residual=evaluator.optimizer_residual,
            full_solver=least_squares, max_nfev=config.fit.max_nfev,
            model_id=config.model_id.value, seed=options.random_seed,
            physical_parameters=physical_parameters, cancel_check=cancel_check,
            progress=progress,
        )

    for start_index, start in enumerate(starts):
        if refined is not None and start_index not in refined:
            continue
        _check_cancel(cancel_check)
        fit_start("M1" if config.model_id.value == "M1_SINGLE_DK" else "M0",
                  start_index + 1, options.n_starts, config.fit.loss)
        if refined is None:
            progress(
                start_index / options.n_starts,
                f"M0/M1 MultiStart {start_index + 1}/{options.n_starts}",
            )
        try:
            fallback_reason = initial_fallback
            if refined is not None:
                optimized = refined[start_index]
            elif hybrid_jac is not None:
                try:
                    optimized = least_squares(
                        evaluator.optimizer_residual, start,
                        bounds=(lower, upper), loss=config.fit.loss,
                        max_nfev=config.fit.max_nfev, jac=hybrid_jac,
                    )
                except HybridFallback as exc:
                    fallback_reason = str(exc)
                    optimized = least_squares(
                        evaluator.optimizer_residual, start,
                        bounds=(lower, upper), loss=config.fit.loss,
                        max_nfev=config.fit.max_nfev,
                    )
            else:
                optimized = least_squares(
                    evaluator.optimizer_residual,
                    start,
                    bounds=(lower, upper),
                    loss=config.fit.loss,
                    max_nfev=config.fit.max_nfev,
                )
            total_nfev += int(optimized.nfev)
            evaluation = evaluator.evaluate(np.asarray(optimized.x, dtype=np.float64))
            fitted = evaluation.prediction
            residuals = evaluation.residuals
            sse = evaluation.sse
            if options.jacobian_mode == EXPERIMENTAL_MODE:
                experimental_starts.append({
                    "start_index": start_index, "method": (
                        "legacy_fallback" if fallback_reason is not None else EXPERIMENTAL_MODE),
                    "fallback_reason": fallback_reason, "nfev": int(optimized.nfev),
                    "njev": int(optimized.njev), "raw_sse": float(sse),
                })
            params = _reported_params(names, optimized.x)
            jacobian = np.asarray(optimized.jac, dtype=np.float64)
            if (
                not np.all(np.isfinite(optimized.x))
                or not np.all(np.isfinite(fitted))
                or not np.all(np.isfinite(jacobian))
                or not isfinite(sse)
                or sse < 0.0
            ):
                fit_start_finished()
                continue
            candidates.append(
                _Candidate(
                    start_index=start_index,
                    initial_coordinates=tuple(float(value) for value in start),
                    coordinates=np.asarray(optimized.x, dtype=np.float64).copy(),
                    params=params,
                    prediction=fitted.copy(),
                    residuals=residuals.copy(),
                    SSE=sse,
                    exit_status=int(optimized.status),
                    nfev=int(optimized.nfev),
                    success=bool(optimized.success),
                    jacobian=jacobian.copy(),
                )
            )
            fit_candidate(sse, optimized.cost, params=params, model_id=config.model_id,
                          observation_count=len(observed), native_status=int(optimized.status))
            fit_start_finished()
        except (ArithmeticError, FloatingPointError, ValueError):
            fit_start_finished()
            continue

    if not candidates:
        raise RuntimeError("all M0/M1 MultiStart candidates failed")
    candidates.sort(key=lambda candidate: (candidate.SSE, candidate.start_index))
    best = candidates[0]
    from .production_stop import polish_ensemble
    polished=polish_ensemble('M1' if config.model_id.value=='M1_SINGLE_DK' else 'M0')
    if polished is not None:
        native,index=polished;endpoint=evaluator.evaluate(np.asarray(native.x,dtype=np.float64))
        best=replace(next(c for c in candidates if c.start_index==index),coordinates=np.array(native.x),
            params=_reported_params(names,native.x),prediction=endpoint.prediction.copy(),residuals=endpoint.residuals.copy(),
            SSE=endpoint.sse,exit_status=int(native.status),nfev=int(native.nfev),success=bool(native.success),jacobian=np.array(native.jac))
        total_nfev+=int(native.nfev)
    fit_stage("Diagnostics", "Estimating local uncertainty and fit statistics")
    warnings: list[str] = []
    if len(candidates) < options.n_candidates:
        warnings.append(
            "MULTISTART_INSUFFICIENT_VALID_CANDIDATES: fewer valid candidates "
            "than requested"
        )
    warnings.extend(_multistart_identifiability(candidates, observed))
    warnings.extend(_bound_warnings(names, best.coordinates, lower, upper))

    retained = tuple(
        FitCandidate(
            start_index=candidate.start_index,
            params=candidate.params,
            SSE=candidate.SSE,
            exit_status=candidate.exit_status,
            nfev=candidate.nfev,
            success=candidate.success,
            initial_coordinates=candidate.initial_coordinates,
        )
        for candidate in candidates[: options.n_candidates]
    )
    statistics, statistic_warnings = _fit_statistics(
        observed,
        best.residuals,
        n_parameters=len(names),
    )
    warnings.extend(statistic_warnings)
    base_diagnostics: dict[str, object] = {
        "model_id": config.model.model_id.value,
        "random_seed": options.random_seed,
        "n_starts": options.n_starts,
        "n_valid_candidates": len(candidates),
        "n_candidates_retained": len(retained),
        "total_nfev": total_nfev,
        "normalized_data": options.normalized_data,
        "fit_amplitude": options.fit_amplitude,
        "fit_baseline": options.fit_baseline,
        "optimizer_parameter_names": list(names),
    }
    if screening_stage is not None:
        base_diagnostics["experimental_coarse_to_fine"] = current_screening().to_dict()
    if options.jacobian_mode == EXPERIMENTAL_MODE:
        base_diagnostics["experimental_jacobian"] = {
            "mode": EXPERIMENTAL_MODE, "log10_step": 1.0e-5,
            "starts": experimental_starts,
        }
    if isinstance(model, BoundaryComposedSingleEcrModel):
        from pyecr.fitting.iteration_observation import observe_direct_forward
        observe_direct_forward()
        boundary_prediction = model.evaluate(
            times,
            {
                **objective_context.fixed_coordinates,
                **_theta_from_coordinates(names, best.coordinates),
            },
            context,
        )
        boundary_diagnostics = dict(boundary_prediction.diagnostics)
        if boundary_diagnostics.get("severity") == "severe_limitation":
            warnings.append(
                "BOUNDARY_SEVERE_LIMITATION: gas flush is slower than the "
                "dominant material response; interpret transport parameters cautiously."
            )
    else:
        boundary_diagnostics = {
            "type": "ideal_step",
            "severity": "none",
            "composition_applied": False,
        }
    base_diagnostics["boundary"] = boundary_diagnostics
    bi_diagnostics, bi_warnings = _bi_diagnostics(
        config, {**config.model.fixed_parameters, **best.params}, options
    )
    base_diagnostics["bi"] = bi_diagnostics
    warnings.extend(bi_warnings)

    preliminary = FitResult(
        params=best.params,
        param_space={name: space for name, space in zip(best.params, spaces, strict=True)},
        y_fit=best.prediction,
        residuals=best.residuals,
        SSE=best.SSE,
        RMSE=statistics["RMSE"],
        R2=statistics["R2"],
        AIC=statistics["AIC"],
        AICc=statistics["AICc"],
        BIC=statistics["BIC"],
        exit_status=best.exit_status,
        nfev=best.nfev,
        jacobian=best.jacobian,
        warnings=tuple(dict.fromkeys(warnings)),
        candidates=retained,
        diagnostics=base_diagnostics,
    )
    uncertainty = estimate_local_uncertainty(
        preliminary,
        condition_threshold=options.condition_threshold,
    )
    warnings.extend(uncertainty.warnings)
    warnings.extend(_correlation_warnings(uncertainty))
    base_diagnostics["local_uncertainty"] = _uncertainty_summary(uncertainty)
    progress(1.0, "M0/M1 fit and local uncertainty complete")
    final = FitResult(
        params=preliminary.params,
        param_space=preliminary.param_space,
        y_fit=preliminary.y_fit,
        residuals=preliminary.residuals,
        SSE=preliminary.SSE,
        RMSE=preliminary.RMSE,
        R2=preliminary.R2,
        AIC=preliminary.AIC,
        AICc=preliminary.AICc,
        BIC=preliminary.BIC,
        exit_status=preliminary.exit_status,
        nfev=preliminary.nfev,
        jacobian=preliminary.jacobian,
        warnings=tuple(dict.fromkeys(warnings)),
        candidates=retained,
        diagnostics=base_diagnostics,
    )
    return record_randomness(final,seed=options.random_seed,stochastic=True,algorithm="single-PCG64-uniform-log-starts-v1",starts=[{"start_index":i,"coordinates":q.tolist()} for i,q in enumerate(starts)])


def prepare_single_objective(
    dataset: EcrDataset,
    config: AnalysisConfig,
    *,
    cancel_check: CancelCheck | None = None,
) -> SingleObjectiveEvaluator:
    """Build the exact production residual/SSE evaluator used by M0/M1 fits."""

    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")
    config.validate()
    if config.model.model_id not in {ModelID.M0_SINGLE_D, ModelID.M1_SINGLE_DK}:
        raise ValueError("fit_single_ecr supports only M0_SINGLE_D or M1_SINGLE_DK")
    if dataset.metadata.get("time_unit", "s") != "s":
        raise ValueError("dataset time must be converted to seconds before fitting")

    options = _options(config)
    signal_type = str(dataset.metadata.get("signal_type", "unknown"))
    if options.normalized_data and signal_type != "normalized":
        raise ValueError(
            "normalized_data=True requires an EcrDataset with signal_type='normalized'"
        )
    if options.normalized_data and (options.fit_amplitude or options.fit_baseline):
        raise ValueError(
            "normalized data fixes amplitude=1 and baseline=0; set "
            "normalized_data=False to fit raw-signal nuisance parameters"
        )

    model = ModelFactory().create(config)
    if not isinstance(model, (SingleEcrModel, BoundaryComposedSingleEcrModel)):
        raise TypeError("ModelFactory did not return a single-process ECR model")
    times = np.asarray(dataset.processed_time, dtype=np.float64)
    observed = np.asarray(dataset.processed_signal, dtype=np.float64)
    if observed.size < 3:
        raise ValueError("single ECR fitting requires at least three observations")
    if isinstance(model, BoundaryComposedSingleEcrModel):
        model.validate_time_coverage(times)

    names, spaces, lower, upper = _parameter_layout(config, options, observed)
    fixed_coordinates = {
        coordinate: float(np.log10(config.model.fixed_parameters[physical]))
        for physical, coordinate in (
            ("D_cm2_s", "log10_D"), ("k_cm_s", "log10_k")
        )
        if physical in config.model.fixed_parameters
    }
    if observed.size <= len(names):
        raise ValueError("number of observations must exceed free parameter count")
    fixed_amplitude = float(config.model.fixed_parameters.get("amplitude", 1.0))
    fixed_baseline = float(config.model.fixed_parameters.get("baseline", 0.0))
    if options.fit_amplitude and "amplitude" in config.model.fixed_parameters:
        raise ValueError("amplitude cannot be both fitted and fixed")
    if options.fit_baseline and "baseline" in config.model.fixed_parameters:
        raise ValueError("baseline cannot be both fitted and fixed")
    context = SingleEcrContext(
        amplitude=1.0 if options.normalized_data else fixed_amplitude,
        baseline=0.0 if options.normalized_data else fixed_baseline,
        term_tol=config.solver.relative_tolerance,
        max_terms=config.solver.max_terms,
    )

    return SingleObjectiveEvaluator(
        ObjectiveContext(
            model=model,
            time_s=times,
            observed=observed,
            forward_context=context,
            parameter_names=names,
            parameter_spaces=spaces,
            lower_bounds=lower,
            upper_bounds=upper,
            fixed_coordinates=fixed_coordinates,
        ),
        cancel_check=cancel_check,
    )


def _options(config: AnalysisConfig) -> _SingleOptions:
    raw = config.model_specific.get("single_ecr", {})
    if not isinstance(raw, Mapping):
        raise TypeError("model_specific['single_ecr'] must be a mapping")
    free = set(config.model.free_parameters)
    fit_amplitude = _boolean_option(
        raw, "fit_amplitude", "amplitude" in free
    ) or "amplitude" in free
    fit_baseline = _boolean_option(
        raw, "fit_baseline", "baseline" in free
    ) or "baseline" in free
    options = _SingleOptions(
        normalized_data=_boolean_option(raw, "normalized_data", True),
        fit_amplitude=fit_amplitude,
        fit_baseline=fit_baseline,
        n_starts=_positive_integer(raw.get("n_starts", 20), "n_starts"),
        n_candidates=_positive_integer(
            raw.get("n_candidates", 5), "n_candidates"
        ),
        random_seed=_nonnegative_integer(
            raw.get("random_seed", 20260915), "random_seed"
        ),
        condition_threshold=_positive_float(
            raw.get("condition_threshold", 1.0e12), "condition_threshold"
        ),
        bi_surface_limit=_positive_float(
            raw.get("bi_surface_limit", 0.03), "bi_surface_limit"
        ),
        bi_diffusion_limit=_positive_float(
            raw.get("bi_diffusion_limit", 30.0), "bi_diffusion_limit"
        ),
        jacobian_mode=validate_mode(raw.get("jacobian_mode", "legacy")),
    )
    if options.jacobian_mode == EXPERIMENTAL_MODE and config.model_id is not ModelID.M1_SINGLE_DK:
        raise ValueError("experimental_hybrid_v1 is available only for M1, not M0")
    if options.n_candidates > options.n_starts:
        raise ValueError("n_candidates must not exceed n_starts")
    if options.bi_surface_limit >= options.bi_diffusion_limit:
        raise ValueError("bi_surface_limit must be below bi_diffusion_limit")
    return options


def _parameter_layout(
    config: AnalysisConfig,
    options: _SingleOptions,
    observed: NDArray[np.float64],
) -> tuple[
    tuple[str, ...],
    tuple[ParameterSpace, ...],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    model_id = config.model.model_id
    required_physical = ["D_cm2_s"]
    default_physical_bounds = {"D_cm2_s": (1.0e-9, 1.0e-3)}
    if model_id is ModelID.M1_SINGLE_DK:
        required_physical.append("k_cm_s")
        default_physical_bounds["k_cm_s"] = (1.0e-10, 1.0)
    fixed_kinetic = set(required_physical) & set(config.model.fixed_parameters)
    if model_id is ModelID.M0_SINGLE_D and fixed_kinetic:
        raise ValueError("M0 diffusion must remain a free parameter")
    if len(fixed_kinetic) > 1:
        raise ValueError("M1 Guided Refit requires exactly one free kinetic parameter")
    free_kinetic = [name for name in required_physical if name not in fixed_kinetic]
    names = ["log10_D" if name == "D_cm2_s" else "log10_k" for name in free_kinetic]
    spaces: list[ParameterSpace] = ["log"] * len(names)
    declared_free = set(config.model.free_parameters)
    if declared_free and not set(free_kinetic).issubset(declared_free):
        raise ValueError("free_parameters must include all unfixed M0/M1 kinetic parameters")

    lower: list[float] = []
    upper: list[float] = []
    for parameter in free_kinetic:
        physical_bounds = config.model.parameter_bounds.get(
            parameter, default_physical_bounds[parameter]
        )
        if physical_bounds[0] <= 0.0:
            raise ValueError(f"{parameter} lower bound must be positive")
        lower.append(float(np.log10(physical_bounds[0])))
        upper.append(float(np.log10(physical_bounds[1])))

    scale = max(float(np.ptp(observed)), float(np.max(np.abs(observed))), 1.0)
    if options.fit_amplitude:
        names.append("amplitude")
        spaces.append("linear")
        bounds = config.model.parameter_bounds.get(
            "amplitude", (-10.0 * scale, 10.0 * scale)
        )
        lower.append(bounds[0])
        upper.append(bounds[1])
    if options.fit_baseline:
        names.append("baseline")
        spaces.append("linear")
        bounds = config.model.parameter_bounds.get(
            "baseline",
            (float(np.min(observed) - 2.0 * scale), float(np.max(observed) + 2.0 * scale)),
        )
        lower.append(bounds[0])
        upper.append(bounds[1])
    return (
        tuple(names),
        tuple(spaces),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def _theta_from_coordinates(
    names: tuple[str, ...],
    coordinates: NDArray[np.float64],
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(names, coordinates, strict=True)
    }


def _reported_params(
    names: tuple[str, ...],
    coordinates: NDArray[np.float64],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in zip(names, coordinates, strict=True):
        if name == "log10_D":
            result["D_cm2_s"] = float(10.0**value)
        elif name == "log10_k":
            result["k_cm_s"] = float(10.0**value)
        else:
            result[name] = float(value)
    return result


def _fit_statistics(
    observed: NDArray[np.float64],
    residuals: NDArray[np.float64],
    *,
    n_parameters: int,
) -> tuple[dict[str, float], tuple[str, ...]]:
    n_observations = observed.size
    sse = float(residuals @ residuals)
    rmse = float(np.sqrt(sse / n_observations))
    centered = observed - float(np.mean(observed))
    sst = float(centered @ centered)
    warnings: list[str] = []
    if sst <= np.finfo(np.float64).eps * max(float(observed @ observed), 1.0):
        r2 = float("nan")
        warnings.append("R2_UNDEFINED: observed signal has zero variance")
    else:
        r2 = 1.0 - sse / sst
    signal_scale = max(float(observed @ observed), 1.0)
    sse_floor = np.finfo(np.float64).eps**2 * signal_scale
    log_variance = log(max(sse, sse_floor) / n_observations)
    aic = n_observations * log_variance + 2.0 * n_parameters
    bic = n_observations * log_variance + n_parameters * log(n_observations)
    if n_observations <= n_parameters + 1:
        aicc = float("inf")
        warnings.append("AICC_UNDEFINED: insufficient observations")
    else:
        aicc = aic + (
            2.0 * n_parameters * (n_parameters + 1)
            / (n_observations - n_parameters - 1)
        )
    return (
        {"RMSE": rmse, "R2": r2, "AIC": aic, "AICc": aicc, "BIC": bic},
        tuple(warnings),
    )


def _multistart_identifiability(
    candidates: list[_Candidate],
    observed: NDArray[np.float64],
) -> tuple[str, ...]:
    best_sse = candidates[0].SSE
    machine_scale = np.finfo(np.float64).eps**2 * max(
        float(observed @ observed), 1.0
    )
    comparable = [
        candidate
        for candidate in candidates
        if (candidate.SSE - best_sse) / max(best_sse, machine_scale) < 0.01
    ]
    warnings: list[str] = []
    for parameter in ("D_cm2_s", "k_cm_s"):
        values = [candidate.params[parameter] for candidate in comparable if parameter in candidate.params]
        if len(values) >= 2:
            spread = float(np.ptp(np.log10(values)))
            if spread > 1.0:
                warnings.append(
                    f"IDENTIFIABILITY_MULTISTART: {parameter} spans >1 decade "
                    "while SSE differs by <1%"
                )
    return tuple(warnings)


def _bound_warnings(
    names: tuple[str, ...],
    coordinates: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> tuple[str, ...]:
    warnings: list[str] = []
    for name, value, low, high in zip(names, coordinates, lower, upper, strict=True):
        tolerance = max(1.0e-7, 1.0e-5 * (high - low))
        if value - low <= tolerance:
            warnings.append(f"PARAMETER_BOUND_HIT: {name} touches lower bound")
        elif high - value <= tolerance:
            warnings.append(f"PARAMETER_BOUND_HIT: {name} touches upper bound")
    return tuple(warnings)


def _bi_diagnostics(
    config: AnalysisConfig,
    params: Mapping[str, float],
    options: _SingleOptions,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if config.model.model_id is ModelID.M0_SINGLE_D:
        return {"applicable": False, "regime": "diffusion_controlled"}, ()
    diffusion = params["D_cm2_s"]
    surface_exchange = params["k_cm_s"]
    values = [
        (dimension / 2.0) * surface_exchange / diffusion
        for dimension, active in zip(
            config.model.geometry.full_dims_cm,
            config.model.geometry.active_dimensions,
            strict=True,
        )
        if active
    ]
    representative = float(np.exp(np.mean(np.log(values))))
    if representative < options.bi_surface_limit:
        regime = "surface_limited"
    elif representative > options.bi_diffusion_limit:
        regime = "diffusion_limited"
    else:
        regime = "mixed_control"
    warning = ()
    if regime != "mixed_control":
        weak_parameter = "D" if regime == "surface_limited" else "k"
        warning = (
            f"IDENTIFIABILITY_BI: {regime} regime; {weak_parameter} is weakly "
            "separable from this curve",
        )
    return (
        {
            "applicable": True,
            "by_active_axis": values,
            "representative": representative,
            "regime": regime,
            "surface_limit": options.bi_surface_limit,
            "diffusion_limit": options.bi_diffusion_limit,
        },
        warning,
    )


def _correlation_warnings(
    uncertainty: LocalUncertaintyResult,
) -> tuple[str, ...]:
    matrix = uncertainty.correlation_matrix
    if matrix.shape[0] < 2:
        return ()
    off_diagonal = np.abs(matrix - np.eye(matrix.shape[0]))
    finite = off_diagonal[np.isfinite(off_diagonal)]
    maximum = 0.0 if finite.size == 0 else float(np.max(finite))
    if maximum > 0.98:
        return (
            "IDENTIFIABILITY_CORRELATION: maximum absolute parameter correlation "
            "exceeds 0.98",
        )
    return ()


def _uncertainty_summary(result: LocalUncertaintyResult) -> dict[str, object]:
    intervals: dict[str, object] = {}
    for name, interval in result.confidence_intervals.items():
        intervals[name] = {
            "estimate": _finite_or_none(interval.estimate),
            "lower": _finite_or_none(interval.lower),
            "upper": _finite_or_none(interval.upper),
            "coordinate_standard_error": _finite_or_none(
                interval.coordinate_standard_error
            ),
        }
    return {
        "condition_number": _finite_or_none(result.condition_number),
        "rank": result.rank,
        "degrees_of_freedom": result.degrees_of_freedom,
        "reliable_separation": result.reliable_separation,
        "correlation_matrix": [
            [_finite_or_none(value) for value in row]
            for row in result.correlation_matrix
        ],
        "confidence_intervals": intervals,
    }


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if isfinite(numeric) else None


def _boolean_option(
    values: Mapping[str, object],
    name: str,
    default: bool,
) -> bool:
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"single_ecr.{name} must be boolean")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"single_ecr.{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"single_ecr.{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"single_ecr.{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"single_ecr.{name} must be nonnegative")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"single_ecr.{name} must be a real number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"single_ecr.{name} must be positive and finite")
    return result


def _check_cancel(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _ignore_progress(fraction: float, message: str) -> None:
    del fraction, message
