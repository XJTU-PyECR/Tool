"""Profile-likelihood scans for single-process ECR parameters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from os import cpu_count
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from pyecr.fitting.base import FitResult
from pyecr.uncertainty.base import (
    ProgressCallback,
    ProgressUpdate,
    ProfileTrace,
    UncertaintyInterval,
    UncertaintyMethod,
    UncertaintyResult,
    _validate_n_jobs,
)

__all__ = [
    "ProfileRefitFunction",
    "profile_likelihood_dual_ho",
    "profile_likelihood_single",
]


ProfileRefitFunction = Callable[[str, float, Mapping[str, float]], FitResult]
_DEFAULT_SINGLE_PARAMETERS = ("D_cm2_s", "k_cm_s")


def profile_likelihood_single(
    fit_result: FitResult,
    *,
    refit_function: ProfileRefitFunction,
    parameters: Sequence[str] | None = None,
    span_decades: float = 1.0,
    n_points: int = 41,
    coordinate_grids: Mapping[str, ArrayLike] | None = None,
    confidence_level: float = 0.95,
    n_jobs: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> UncertaintyResult:
    """Profile selected log10 parameters while nuisance parameters are refitted.

    This function is intentionally opt-in.  For each fixed physical parameter
    value, ``refit_function`` must re-optimize every other free parameter and
    return a ``FitResult`` containing the same complete parameter mapping.  The
    callback's reported fixed value is checked, preventing an unconstrained
    ordinary refit from masquerading as a profile point.

    By default, present single-process ``D_cm2_s`` and ``k_cm_s`` parameters are
    profiled.  ``coordinate_grids`` values are always log10 coordinates.
    """

    selected = _validate_inputs(
        fit_result,
        refit_function=refit_function,
        parameters=parameters,
        span_decades=span_decades,
        n_points=n_points,
        coordinate_grids=coordinate_grids,
        confidence_level=confidence_level,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
    )
    workers = _resolved_workers(_validate_n_jobs(n_jobs))
    grids = {
        name: _profile_grid(
            fit_result,
            name,
            span_decades=span_decades,
            n_points=n_points,
            supplied=None if coordinate_grids is None else coordinate_grids.get(name),
        )
        for name in selected
    }
    tasks = [
        (name, index, float(coordinate))
        for name in selected
        for index, coordinate in enumerate(grids[name])
    ]
    total = len(tasks)
    parameter_names = tuple(fit_result.params)
    cutoff = float(chi2.ppf(confidence_level, df=1))
    base_start = MappingProxyType(dict(fit_result.params))

    storage = {
        name: {
            "SSE": np.full(grids[name].size, np.nan),
            "successful": np.zeros(grids[name].size, dtype=np.bool_),
            "optimized": np.full((grids[name].size, len(parameter_names)), np.nan),
        }
        for name in selected
    }
    failures: list[str] = []
    _notify(
        progress_callback,
        ProgressUpdate("profile_likelihood", 0, total, "Preparing profile grid"),
    )

    def run_one(
        task: tuple[str, int, float],
    ) -> tuple[str, int, float, NDArray[np.float64]]:
        name, index, coordinate = task
        fixed_value = _coordinate_to_physical(coordinate)
        refitted = refit_function(name, fixed_value, base_start)
        values = _validated_profile_refit(
            fit_result,
            refitted,
            fixed_parameter=name,
            fixed_value=fixed_value,
        )
        return name, index, float(refitted.SSE), values

    completed = 0
    if workers == 1:
        for task in tasks:
            _consume_profile_task(run_one, task, storage, failures)
            completed += 1
            _notify(
                progress_callback,
                ProgressUpdate(
                    "profile_likelihood",
                    completed,
                    total,
                    "Refitting nuisance parameters",
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    name, index, sse, values = future.result()
                    _store_profile_success(storage, name, index, sse, values)
                except Exception as exc:
                    name, index, _ = task
                    failures.append(
                        f"{name}[{index}]: {type(exc).__name__}: {exc}"
                    )
                completed += 1
                _notify(
                    progress_callback,
                    ProgressUpdate(
                        "profile_likelihood",
                        completed,
                        total,
                        "Refitting nuisance parameters",
                    ),
                )

    profiles: dict[str, ProfileTrace] = {}
    intervals: dict[str, UncertaintyInterval] = {}
    warnings: list[str] = []
    n_successful = 0
    truncated_parameters: list[str] = []

    for name in selected:
        stored = storage[name]
        successful = stored["successful"]
        successful_count = int(np.count_nonzero(successful))
        n_successful += successful_count
        if successful_count == 0:
            warnings.append(f"PROFILE_EMPTY: no successful points for {name}.")
            continue

        sse = stored["SSE"]
        likelihood_ratio, reference_sse = _likelihood_ratio(
            sse,
            successful,
            base_sse=fit_result.SSE,
            n_observations=fit_result.y_fit.size,
        )
        values = np.power(10.0, grids[name])
        trace = ProfileTrace(
            parameter=name,
            space=fit_result.param_space[name],
            parameter_names=parameter_names,
            coordinates=grids[name],
            values=values,
            SSE=sse,
            likelihood_ratio=likelihood_ratio,
            successful=successful,
            optimized_parameters=stored["optimized"],
            cutoff=cutoff,
            reference_SSE=reference_sse,
        )
        profiles[name] = trace
        interval, truncated = _profile_interval(
            fit_result,
            name,
            trace,
            confidence_level=confidence_level,
        )
        if interval is not None:
            intervals[name] = interval
        if truncated:
            truncated_parameters.append(name)

    if failures:
        warnings.append(
            f"PROFILE_REFIT_FAILURES: {len(failures)} of {total} profile "
            "refits failed or violated the fixed-parameter constraint."
        )
    if truncated_parameters:
        warnings.append(
            "PROFILE_INTERVAL_TRUNCATED: extend the scan for "
            f"{', '.join(truncated_parameters)}."
        )

    return UncertaintyResult(
        method=UncertaintyMethod.PROFILE_LIKELIHOOD,
        parameter_names=parameter_names,
        estimates=fit_result.params,
        param_space=fit_result.param_space,
        intervals=intervals,
        confidence_level=float(confidence_level),
        n_requested=total,
        n_successful=n_successful,
        profiles=profiles,
        diagnostics={
            "scope": "single_ecr",
            "profiled_parameters": selected,
            "n_jobs": workers,
            "likelihood_ratio_definition": "n*log(SSE/SSE_min)",
            "chi_square_degrees_of_freedom": 1,
            "likelihood_ratio_cutoff": cutoff,
            "failed_refit_messages": tuple(failures[:20]),
        },
        warnings=tuple(warnings),
    )


def profile_likelihood_dual_ho(
    fit_result: FitResult,
    *,
    refit_function: ProfileRefitFunction,
    parameters: Sequence[str] | None = None,
    span_decades: float = 1.0,
    n_points: int = 41,
    coordinate_grids: Mapping[str, ArrayLike] | None = None,
    confidence_level: float = 0.95,
    n_jobs: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> UncertaintyResult:
    """Profile ``log10(D_H)`` and ``log10(D_O)`` using the shared engine."""

    required = ("D_H_cm2_s", "D_O_cm2_s")
    missing = set(required) - set(fit_result.params)
    if missing:
        raise ValueError(
            "dual H/O profile requires parameters: "
            f"{', '.join(required)}; missing {', '.join(sorted(missing))}"
        )
    selected = required if parameters is None else parameters
    result = profile_likelihood_single(
        fit_result,
        refit_function=refit_function,
        parameters=selected,
        span_decades=span_decades,
        n_points=n_points,
        coordinate_grids=coordinate_grids,
        confidence_level=confidence_level,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["scope"] = "dual_ho"
    return replace(result, diagnostics=diagnostics)


def _consume_profile_task(
    run_one: Callable[
        [tuple[str, int, float]],
        tuple[str, int, float, NDArray[np.float64]],
    ],
    task: tuple[str, int, float],
    storage: dict[str, dict[str, NDArray[np.generic]]],
    failures: list[str],
) -> None:
    try:
        name, index, sse, values = run_one(task)
        _store_profile_success(storage, name, index, sse, values)
    except Exception as exc:
        name, index, _ = task
        failures.append(f"{name}[{index}]: {type(exc).__name__}: {exc}")


def _store_profile_success(
    storage: dict[str, dict[str, NDArray[np.generic]]],
    name: str,
    index: int,
    sse: float,
    values: NDArray[np.float64],
) -> None:
    storage[name]["SSE"][index] = sse
    storage[name]["successful"][index] = True
    storage[name]["optimized"][index, :] = values


def _validated_profile_refit(
    base_fit: FitResult,
    refitted: FitResult,
    *,
    fixed_parameter: str,
    fixed_value: float,
) -> NDArray[np.float64]:
    if not isinstance(refitted, FitResult):
        raise TypeError("refit_function must return FitResult")
    if refitted.exit_status <= 0 or not np.isfinite(refitted.SSE):
        raise RuntimeError("profile optimizer did not converge to a finite solution")
    if tuple(refitted.params) != tuple(base_fit.params):
        raise ValueError("profile refit must report the complete base parameter mapping")
    if dict(refitted.param_space) != dict(base_fit.param_space):
        raise ValueError("profile refit parameter spaces differ from the base fit")
    reported_fixed = refitted.params[fixed_parameter]
    tolerance = max(abs(fixed_value) * 1.0e-9, np.finfo(np.float64).tiny)
    if not np.isclose(reported_fixed, fixed_value, rtol=1.0e-9, atol=tolerance):
        raise ValueError(
            f"refit did not keep {fixed_parameter} at the requested profile value"
        )
    values = np.asarray(list(refitted.params.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("profile parameter estimates must be finite")
    for index, name in enumerate(base_fit.params):
        if base_fit.param_space[name] == "log" and values[index] <= 0.0:
            raise ValueError(f"log-space parameter {name!r} must remain positive")
    return values


def _profile_grid(
    fit_result: FitResult,
    parameter: str,
    *,
    span_decades: float,
    n_points: int,
    supplied: ArrayLike | None,
) -> NDArray[np.float64]:
    if supplied is None:
        centre = float(np.log10(fit_result.params[parameter]))
        return np.linspace(centre - span_decades, centre + span_decades, n_points)
    grid = np.asarray(supplied, dtype=np.float64).reshape(-1).copy()
    if grid.size < 3:
        raise ValueError(f"coordinate grid for {parameter} needs at least 3 points")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError(
            f"coordinate grid for {parameter} must be finite and strictly increasing"
        )
    return grid


def _likelihood_ratio(
    sse: NDArray[np.float64],
    successful: NDArray[np.bool_],
    *,
    base_sse: float,
    n_observations: int,
) -> tuple[NDArray[np.float64], float]:
    valid_sse = sse[successful]
    minimum_profile_sse = float(np.min(valid_sse))
    positive_candidates = [
        value for value in (float(base_sse), minimum_profile_sse) if value > 0.0
    ]
    result = np.full_like(sse, np.nan)
    if positive_candidates:
        reference = min(positive_candidates)
        ratios = np.maximum(valid_sse / reference, 1.0)
        result[successful] = n_observations * np.log(ratios)
    else:
        reference = 0.0
        result[successful] = np.where(valid_sse == 0.0, 0.0, np.nan)
    finite = np.isfinite(result)
    if np.any(finite):
        result[finite] -= float(np.min(result[finite]))
    return result, reference


def _profile_interval(
    fit_result: FitResult,
    parameter: str,
    trace: ProfileTrace,
    *,
    confidence_level: float,
) -> tuple[UncertaintyInterval | None, bool]:
    valid = trace.successful & np.isfinite(trace.likelihood_ratio)
    if not np.any(valid):
        return None, True
    valid_indices = np.flatnonzero(valid)
    minimum_index = int(
        valid_indices[np.argmin(trace.likelihood_ratio[valid_indices])]
    )
    if trace.likelihood_ratio[minimum_index] > trace.cutoff:
        return None, True

    left = minimum_index
    while (
        left > 0
        and valid[left - 1]
        and trace.likelihood_ratio[left - 1] <= trace.cutoff
    ):
        left -= 1
    right = minimum_index
    while (
        right < trace.coordinates.size - 1
        and valid[right + 1]
        and trace.likelihood_ratio[right + 1] <= trace.cutoff
    ):
        right += 1

    lower_truncated = left == 0 or not valid[left - 1]
    upper_truncated = right == trace.coordinates.size - 1 or not valid[right + 1]
    coordinate_lower = float(trace.coordinates[left])
    coordinate_upper = float(trace.coordinates[right])
    if not lower_truncated:
        coordinate_lower = _crossing_coordinate(
            float(trace.coordinates[left - 1]),
            float(trace.likelihood_ratio[left - 1]),
            float(trace.coordinates[left]),
            float(trace.likelihood_ratio[left]),
            trace.cutoff,
        )
    if not upper_truncated:
        coordinate_upper = _crossing_coordinate(
            float(trace.coordinates[right]),
            float(trace.likelihood_ratio[right]),
            float(trace.coordinates[right + 1]),
            float(trace.likelihood_ratio[right + 1]),
            trace.cutoff,
        )

    estimate = float(fit_result.params[parameter])
    interval = UncertaintyInterval(
        parameter=parameter,
        estimate=estimate,
        lower=_coordinate_to_physical(coordinate_lower),
        upper=_coordinate_to_physical(coordinate_upper),
        confidence_level=float(confidence_level),
        space=fit_result.param_space[parameter],
        coordinate_estimate=float(np.log10(estimate)),
        coordinate_lower=coordinate_lower,
        coordinate_upper=coordinate_upper,
    )
    return interval, lower_truncated or upper_truncated


def _crossing_coordinate(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cutoff: float,
) -> float:
    if y2 == y1:
        return (x1 + x2) / 2.0
    fraction = (cutoff - y1) / (y2 - y1)
    return x1 + float(np.clip(fraction, 0.0, 1.0)) * (x2 - x1)


def _coordinate_to_physical(coordinate: float) -> float:
    value = float(10.0**coordinate)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("profile coordinate maps outside positive finite range")
    return value


def _resolved_workers(n_jobs: int) -> int:
    return max(1, cpu_count() or 1) if n_jobs == -1 else n_jobs


def _notify(callback: ProgressCallback | None, update: ProgressUpdate) -> None:
    if callback is not None:
        callback(update)


def _validate_inputs(
    fit_result: FitResult,
    *,
    refit_function: ProfileRefitFunction,
    parameters: Sequence[str] | None,
    span_decades: float,
    n_points: int,
    coordinate_grids: Mapping[str, ArrayLike] | None,
    confidence_level: float,
    n_jobs: int,
    progress_callback: ProgressCallback | None,
) -> tuple[str, ...]:
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult")
    if not callable(refit_function):
        raise TypeError("refit_function must be callable")
    if parameters is None:
        selected = tuple(
            name for name in _DEFAULT_SINGLE_PARAMETERS if name in fit_result.params
        )
    else:
        if isinstance(parameters, str):
            raise TypeError("parameters must be a sequence, not one string")
        selected = tuple(parameters)
    if not selected:
        raise ValueError("no single-ECR D or k parameter was selected for profiling")
    if len(set(selected)) != len(selected):
        raise ValueError("profile parameters must not contain duplicates")
    for name in selected:
        if name not in fit_result.params:
            raise ValueError(f"unknown profile parameter: {name!r}")
        if fit_result.param_space[name] != "log":
            raise ValueError(f"profile parameter {name!r} must use log10 space")
        if fit_result.params[name] <= 0.0:
            raise ValueError(f"profile parameter {name!r} must be positive")
    if not np.isfinite(span_decades) or span_decades <= 0.0:
        raise ValueError("span_decades must be positive and finite")
    if (
        isinstance(n_points, bool)
        or not isinstance(n_points, int)
        or n_points < 3
        or n_points % 2 == 0
    ):
        raise ValueError("n_points must be an odd integer of at least 3")
    if coordinate_grids is not None:
        unknown = set(coordinate_grids) - set(selected)
        if unknown:
            raise ValueError(
                f"coordinate grids supplied for unselected parameters: {sorted(unknown)}"
            )
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    _validate_n_jobs(n_jobs)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    return selected
