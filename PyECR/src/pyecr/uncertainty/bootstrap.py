"""Residual-bootstrap uncertainty for a completed single-process ECR fit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from os import cpu_count
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from pyecr.fitting.base import FitResult
from pyecr.uncertainty.base import (
    ProgressCallback,
    ProgressUpdate,
    UncertaintyInterval,
    UncertaintyMethod,
    UncertaintyResult,
    _integer_in_range,
    _validate_n_jobs,
)

__all__ = [
    "BootstrapRefitFunction",
    "residual_bootstrap_dual_ho",
    "residual_bootstrap_single",
]


BootstrapRefitFunction = Callable[[NDArray[np.float64]], FitResult]


def residual_bootstrap_single(
    fit_result: FitResult,
    *,
    refit_function: BootstrapRefitFunction,
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
    n_jobs: int = 1,
    progress_callback: ProgressCallback | None = None,
    minimum_success_fraction: float = 0.80,
    residual_convention: str = 'observed_minus_fit',
) -> UncertaintyResult:
    """Run an explicitly requested residual bootstrap.

    The observed signal is reconstructed using PyECR's residual convention,
    ``observed = y_fit + residuals``.  Residuals are centred before sampling.
    ``refit_function`` must perform the same single-ECR fit on the supplied
    synthetic signal and return parameters in the same order and spaces.

    Parallel execution uses threads so closures containing model context remain
    usable on Windows.  Random seeds are generated before dispatch, making the
    sample matrix reproducible across worker counts.
    """

    _validate_inputs(
        fit_result,
        refit_function=refit_function,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_seed=random_seed,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
        minimum_success_fraction=minimum_success_fraction,
    )
    n_resamples = _integer_in_range("n_resamples", n_resamples, 200, 1000)
    workers = _resolved_workers(_validate_n_jobs(n_jobs))

    parameter_names = tuple(fit_result.params)
    if residual_convention not in {'observed_minus_fit','fit_minus_observed'}:
        raise ValueError('unknown parent residual convention')
    residual_pool = np.asarray(fit_result.residuals, dtype=np.float64)
    if residual_convention=='fit_minus_observed':residual_pool=-residual_pool
    residual_pool = residual_pool - float(np.mean(residual_pool))
    fitted = np.asarray(fit_result.y_fit, dtype=np.float64)

    from pyecr.fitting.reproducibility import integer_seed
    random_seed=integer_seed(20260915 if random_seed is None else random_seed)
    generator = np.random.default_rng(random_seed)
    initial_rng_state = generator.bit_generator.state
    seeds = generator.integers(
        0,
        np.iinfo(np.uint64).max,
        size=n_resamples,
        dtype=np.uint64,
    )
    sample_rows = np.full((n_resamples, len(parameter_names)), np.nan)
    successful = np.zeros(n_resamples, dtype=np.bool_)
    failures: list[str] = []

    _notify(
        progress_callback,
        ProgressUpdate("residual_bootstrap", 0, n_resamples, "Preparing samples"),
    )

    def run_one(index: int) -> tuple[int, NDArray[np.float64]]:
        replicate_generator = np.random.default_rng(int(seeds[index]))
        sampled_indices = replicate_generator.integers(
            0,
            residual_pool.size,
            size=residual_pool.size,
        )
        synthetic_signal = fitted + residual_pool[sampled_indices]
        refitted = refit_function(synthetic_signal.copy())
        return index, _validated_refit_parameters(fit_result, refitted)

    completed = 0
    if workers == 1:
        for index in range(n_resamples):
            try:
                sample_index, values = run_one(index)
                sample_rows[sample_index, :] = values
                successful[sample_index] = True
            except Exception as exc:  # one failed optimizer must not abort all samples
                failures.append(f"replicate {index}: {type(exc).__name__}: {exc}")
            completed += 1
            _notify(
                progress_callback,
                ProgressUpdate(
                    "residual_bootstrap",
                    completed,
                    n_resamples,
                    "Refitting bootstrap samples",
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one, index): index
                for index in range(n_resamples)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    sample_index, values = future.result()
                    sample_rows[sample_index, :] = values
                    successful[sample_index] = True
                except Exception as exc:
                    failures.append(f"replicate {index}: {type(exc).__name__}: {exc}")
                completed += 1
                _notify(
                    progress_callback,
                    ProgressUpdate(
                        "residual_bootstrap",
                        completed,
                        n_resamples,
                        "Refitting bootstrap samples",
                    ),
                )

    physical_samples = sample_rows[successful]
    n_successful = int(np.count_nonzero(successful))
    coordinate_samples = _to_coordinate_samples(fit_result, physical_samples)
    intervals = {} if n_successful < 2 else _percentile_intervals(
        fit_result,
        coordinate_samples,
        confidence_level=confidence_level,
    )

    mode_counts = {
        name: (0 if n_successful < 2 else _estimated_mode_count(coordinate_samples[:, index]))
        for index, name in enumerate(parameter_names)
    }
    multimodal = tuple(name for name, count in mode_counts.items() if count > 1)
    warnings: list[str] = []
    if n_successful < 2:
        warnings.append('LOW_BOOTSTRAP_SUCCESS_COUNT: fewer than two successful refits; samples retained, no interval.')
    if failures:
        warnings.append(
            f"BOOTSTRAP_REFIT_FAILURES: {len(failures)} of {n_resamples} "
            "replicates failed."
        )
    if n_successful / n_resamples < minimum_success_fraction:
        warnings.append(
            "LOW_BOOTSTRAP_SUCCESS_RATE: percentile intervals may be biased by "
            "failed refits."
        )
    if multimodal:
        warnings.append(
            "BOOTSTRAP_MULTIMODAL: "
            f"{', '.join(multimodal)} contain multiple resolved modes; "
            "LOCAL_GAUSSIAN_CI_UNRELIABLE."
        )

    return UncertaintyResult(
        method=UncertaintyMethod.RESIDUAL_BOOTSTRAP,
        parameter_names=parameter_names,
        estimates=fit_result.params,
        param_space=fit_result.param_space,
        intervals=intervals,
        confidence_level=float(confidence_level),
        n_requested=n_resamples,
        n_successful=n_successful,
        physical_samples=physical_samples,
        coordinate_samples=coordinate_samples,
        diagnostics={
            "scope": "single_ecr",
            "stochastic": True, "random_seed": random_seed, "seed_policy": "bootstrap-default-v1",
            "replicate_seeds": [int(v) for v in seeds],
            "rng": type(generator.bit_generator).__name__, "numpy_version": np.__version__,
            "rng_state_before": initial_rng_state, "quantile_method": "linear",
            "minimum_success_fraction": minimum_success_fraction,
            "successful_replicate_indices": np.flatnonzero(successful).tolist(),
            "successful_mask": successful.tolist(), "residual_convention": residual_convention,
            "residuals_centered": True,
            "n_jobs": workers,
            "mode_detection": "gaussian_kde_prominent_peaks",
            "mode_counts": mode_counts,
            "multimodal_parameters": multimodal,
            "failed_refit_messages": tuple(failures[:20]),
        },
        warnings=tuple(warnings),
    )


def residual_bootstrap_dual_ho(
    fit_result: FitResult,
    *,
    refit_function: BootstrapRefitFunction,
    n_resamples: int = 200,
    confidence_level: float = 0.95,
    random_seed: int | None = None,
    n_jobs: int = 1,
    progress_callback: ProgressCallback | None = None,
    minimum_success_fraction: float = 0.80,
    residual_convention: str = 'observed_minus_fit',
) -> UncertaintyResult:
    """Run residual bootstrap for an H/O fit using the shared engine."""

    required = {"D_H_cm2_s", "D_O_cm2_s"}
    missing = required - set(fit_result.params)
    if missing:
        raise ValueError(
            "dual H/O bootstrap requires parameters: "
            f"{', '.join(sorted(required))}; missing {', '.join(sorted(missing))}"
        )
    result = residual_bootstrap_single(
        fit_result,
        refit_function=refit_function,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_seed=random_seed,
        n_jobs=n_jobs,
        progress_callback=progress_callback,
        minimum_success_fraction=minimum_success_fraction,
        residual_convention=residual_convention,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["scope"] = "dual_ho"
    return replace(result, diagnostics=diagnostics)


def _validated_refit_parameters(
    base_fit: FitResult,
    refitted: FitResult,
) -> NDArray[np.float64]:
    if not isinstance(refitted, FitResult):
        raise TypeError("refit_function must return FitResult")
    if refitted.exit_status <= 0:
        raise RuntimeError(f"optimizer exit_status={refitted.exit_status}")
    if tuple(refitted.params) != tuple(base_fit.params):
        raise ValueError("refitted parameter names/order differ from the base fit")
    if dict(refitted.param_space) != dict(base_fit.param_space):
        raise ValueError("refitted parameter spaces differ from the base fit")
    values = np.asarray(list(refitted.params.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("refitted parameters must be finite")
    for index, name in enumerate(base_fit.params):
        if base_fit.param_space[name] == "log" and values[index] <= 0.0:
            raise ValueError(f"log-space parameter {name!r} must remain positive")
    return values


def _to_coordinate_samples(
    fit_result: FitResult,
    physical_samples: NDArray[np.float64],
) -> NDArray[np.float64]:
    result = physical_samples.copy()
    for index, name in enumerate(fit_result.params):
        if fit_result.param_space[name] == "log":
            if np.any(result[:, index] <= 0.0):
                raise ValueError(f"bootstrap parameter {name!r} must be positive")
            result[:, index] = np.log10(result[:, index])
    return result


def _percentile_intervals(
    fit_result: FitResult,
    coordinate_samples: NDArray[np.float64],
    *,
    confidence_level: float,
) -> dict[str, UncertaintyInterval]:
    tail = (1.0 - confidence_level) / 2.0
    intervals: dict[str, UncertaintyInterval] = {}
    for index, (name, estimate) in enumerate(fit_result.params.items()):
        coordinate_lower, coordinate_upper = np.quantile(
            coordinate_samples[:, index],
            [tail, 1.0 - tail],
            method='linear',
        )
        space = fit_result.param_space[name]
        coordinate_estimate = (
            float(np.log10(estimate)) if space == "log" else float(estimate)
        )
        lower = (
            float(10.0**coordinate_lower)
            if space == "log"
            else float(coordinate_lower)
        )
        upper = (
            float(10.0**coordinate_upper)
            if space == "log"
            else float(coordinate_upper)
        )
        intervals[name] = UncertaintyInterval(
            parameter=name,
            estimate=float(estimate),
            lower=lower,
            upper=upper,
            confidence_level=float(confidence_level),
            space=space,
            coordinate_estimate=coordinate_estimate,
            coordinate_lower=float(coordinate_lower),
            coordinate_upper=float(coordinate_upper),
        )
    return intervals


def _estimated_mode_count(samples: NDArray[np.float64]) -> int:
    if samples.size < 20 or np.ptp(samples) <= np.finfo(np.float64).eps:
        return 1
    standardized = (samples - np.mean(samples)) / np.std(samples, ddof=1)
    if not np.all(np.isfinite(standardized)):
        return 1
    try:
        density_model = gaussian_kde(standardized)
    except np.linalg.LinAlgError:
        return 1
    margin = max(0.25, 0.10 * float(np.ptp(standardized)))
    grid = np.linspace(
        float(np.min(standardized)) - margin,
        float(np.max(standardized)) + margin,
        512,
    )
    density = density_model(grid)
    maximum = float(np.max(density))
    if maximum <= 0.0 or not np.isfinite(maximum):
        return 1
    peaks, properties = find_peaks(
        density,
        height=0.10 * maximum,
        prominence=0.05 * maximum,
        distance=25,
    )
    if peaks.size <= 1:
        return 1

    heights = properties["peak_heights"]
    resolved = 1
    previous_peak = int(peaks[0])
    previous_height = float(heights[0])
    for peak, height in zip(peaks[1:], heights[1:], strict=True):
        valley = float(np.min(density[previous_peak : int(peak) + 1]))
        if valley <= 0.75 * min(previous_height, float(height)):
            resolved += 1
            previous_peak = int(peak)
            previous_height = float(height)
        elif height > previous_height:
            previous_peak = int(peak)
            previous_height = float(height)
    return resolved


def _resolved_workers(n_jobs: int) -> int:
    return max(1, cpu_count() or 1) if n_jobs == -1 else n_jobs


def _notify(callback: ProgressCallback | None, update: ProgressUpdate) -> None:
    if callback is not None:
        callback(update)


def _validate_inputs(
    fit_result: FitResult,
    *,
    refit_function: BootstrapRefitFunction,
    n_resamples: int,
    confidence_level: float,
    random_seed: int | None,
    n_jobs: int,
    progress_callback: ProgressCallback | None,
    minimum_success_fraction: float,
) -> None:
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult")
    if not callable(refit_function):
        raise TypeError("refit_function must be callable")
    _integer_in_range("n_resamples", n_resamples, 200, 1000)
    _validate_n_jobs(n_jobs)
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int)
    ):
        raise TypeError("random_seed must be an integer or None")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if not 0.0 < float(minimum_success_fraction) <= 1.0:
        raise ValueError("minimum_success_fraction must be in (0, 1]")
