"""Explicit terminal-referenced normalization, shared by recipes and DCT copies.

There is no spectrum solver here. Diagnostics inspect the observations without
replacing them, and a prepared result never changes the input dataset.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import fsum, isclose
from types import MappingProxyType

import numpy as np

from pyecr.contracts import SignedNormalizationResult, SignedNormalizationSpec
from pyecr.contracts.common import ResultState, SourceIdentity, canonical_sha256, require
from pyecr.io.dataset import EcrDataset, _TIME_FACTORS_TO_SECONDS
from pyecr.preprocessing.pipeline import PreprocessConfig, ProcessingStep
from pyecr.preprocessing.quality import _local_noise, _robust_sigma


# Fixed review policy, separate from the frozen mathematical profile. No caller
# threshold overrides: a changed policy requires a new policy identifier.
SIGNED_NORMALIZATION_CHECKS = MappingProxyType({
    "version": "signed-normalization-checks/1",
    "roundoff_eps_factor": 32,
    "low_information_noise_factor": 5,
    "minimum_terminal_points": 3,
    "terminal_drift_fraction": 0.05,
    "terminal_scatter_fraction": 0.05,
    "short_terminal_points": 5,
    "short_terminal_duration_fraction": 0.03,
    "scale_second_largest_ratio": 1 / 3,
})


def raw_observations_sha256(dataset: EcrDataset) -> str:
    """Canonical JSON hash of both complete raw arrays, independent of a view."""
    return canonical_sha256({"raw_time": dataset.raw_time.tolist(),
                             "raw_signal": dataset.raw_signal.tolist()})


def cumulative_t0_s(dataset: EcrDataset) -> float:
    """Read the recorded cumulative origin in seconds, without changing time."""
    origins = []
    for record in dataset.history:
        if record.operation == "set_manual_t0":
            unit = record.parameters.get("time_unit")
            require(unit in _TIME_FACTORS_TO_SECONDS, "time-origin record has no supported unit")
            origins.append(float(record.parameters["t0"]) * _TIME_FACTORS_TO_SECONDS[unit])
    return fsum(origins)


def _validate_binding(dataset, spec, recipe):
    require(isinstance(dataset, EcrDataset), "dataset must be EcrDataset")
    require(isinstance(spec, SignedNormalizationSpec), "spec must be SignedNormalizationSpec")
    require(isinstance(recipe, PreprocessConfig), "recipe must be the actual PreprocessConfig")
    actual = SourceIdentity.capture(dataset, recipe)
    require(actual.dataset_sha256 == spec.identity.dataset_sha256
            and actual.recipe_sha256 == spec.identity.recipe_sha256,
            "signed normalization dataset/recipe identity conflict; prepare again")
    require(spec.raw_count == dataset.raw_time.size
            and spec.raw_indices == tuple(int(i) for i in np.flatnonzero(dataset.mask)),
            "normalization must use the complete eligible observation set")
    require(spec.signal_type == dataset.metadata.get("signal_type")
            and spec.signal_unit == dataset.metadata.get("signal_unit"),
            "normalization signal type/unit conflict")
    require(dataset.metadata.get("time_unit") in _TIME_FACTORS_TO_SECONDS,
            "normalization requires an explicit supported time unit")
    # Each t0 is recorded in the then-current time unit. Sum in seconds; never
    # use metadata['manual_t0'], which only stores the most recent shift.
    origin = cumulative_t0_s(dataset)
    require(isclose(origin, spec.t0_s, rel_tol=0,
                    abs_tol=32 * np.finfo(float).eps * max(abs(origin), abs(spec.t0_s))),
            "normalization cumulative t0_s conflict")
    if recipe.steps:
        require(len(dataset.history) >= len(recipe.steps), "recipe is absent from dataset history")
        require(all(r.operation == s.operation and r.timestamp == s.timestamp
                    for r, s in zip(dataset.history[-len(recipe.steps):], recipe.steps)),
                "recipe does not match dataset operation history")


def _unavailable(spec, reason, warnings=(), *, status="unsupported"):
    return SignedNormalizationResult(spec=spec, state=ResultState(
        status=status, warnings=tuple(warnings), blocking_reasons=(reason,)))


def _terminal_check(time, signal, selected, warnings):
    """Return instability reasons. Signal is already divided by the true max."""
    p = SIGNED_NORMALIZATION_CHECKS
    t, y = time[selected], signal[selected]
    if len(y) < p["minimum_terminal_points"]:
        return ["terminal_insufficient_points: at least 3 observations are required"]
    x = (t - t[0]) / (t[-1] - t[0])
    # Dimensionless end-to-end linear drift; no detrending of the observations.
    slope = float(np.polyfit(x, y - y[0], 1)[0])
    residual = y - (float(np.mean(y)) + slope * (x - float(np.mean(x))))
    scatter = _robust_sigma(residual)
    reasons = []
    if abs(slope) > p["terminal_drift_fraction"]:
        reasons.append(f"terminal_unstable_drift: abs(slope*duration)/S={abs(slope):.6g} > 0.05")
    if scatter > p["terminal_scatter_fraction"]:
        reasons.append(f"terminal_unstable_scatter: robust_sigma/S={scatter:.6g} > 0.05")
    if len(y) < p["short_terminal_points"] or (
        t[-1] - t[0] < p["short_terminal_duration_fraction"] * (time[-1] - time[0])
    ):
        warnings.append("terminal_short: fewer than 5 points or less than 3% of trace duration; confirm stability")
    return reasons


def normalize_signed_dct(
    dataset: EcrDataset,
    spec: SignedNormalizationSpec,
    *,
    recipe: PreprocessConfig,
) -> SignedNormalizationResult:
    """Compute (sigma - terminal)/max(abs(sigma - terminal)) on every eligible row.

    Malformed/stale requests raise ValueError. Scientifically unavailable cases
    return an explicit state without numerical payload. Model-inferred terminals
    remain a distinct, unsupported source until a real parent-evidence adapter
    exists; a hash alone is not evidence that a Fit is current or reliable.
    """
    _validate_binding(dataset, spec, recipe)
    if spec.reference_kind == "parent_fit_asymptote":
        return _unavailable(spec, "model_terminal_requires_parent_evidence_adapter: P02 accepts a final window or external known terminal only")

    warnings = []
    values = dataset.processed_signal
    time = dataset.processed_time
    final_window = dataset.metadata.get("stability_windows", {})
    final_window = final_window.get("final") if isinstance(final_window, Mapping) else None
    if spec.reference_kind == "accepted_final_window":
        require(isinstance(final_window, Mapping), "accept a final stability window before normalization")
        dataset._validate_stability_window_is_current("final", final_window)
        require(tuple(final_window["raw_indices"]) == spec.reference_rows,
                "accepted final window raw rows conflict")
        require(spec.reference_rows[-1] == spec.raw_indices[-1],
                "accepted final window must include the last eligible observation")
        selected = np.isin(spec.raw_indices, spec.reference_rows)
        # Recompute from current values, not potentially stale cached statistics
        # (e.g. after an explicitly applied drift correction).
        with np.errstate(over="ignore", invalid="ignore"):
            terminal = float(np.mean(values[selected]))
    else:
        terminal = spec.reference_value
        warnings.append("external_known_terminal: conditional on the explicit value and evidence; not an observed or model-inferred plateau")
        selected = None
        if isinstance(final_window, Mapping):
            try:
                dataset._validate_stability_window_is_current("final", final_window)
                if final_window["raw_indices"][-1] == spec.raw_indices[-1]:
                    selected = np.isin(spec.raw_indices, final_window["raw_indices"])
            except (ValueError, TypeError, KeyError, IndexError):
                warnings.append("terminal_window_stale: excluded from stability evidence; reselect explicitly")
        if selected is None:
            # This tail is a diagnostic only. It never supplies the terminal.
            selected = np.arange(len(values)) >= len(values) - max(3, min(20, int(np.ceil(0.1 * len(values)))))
            warnings.append("terminal_unverified_observed: no current accepted final window; inspecting tail only")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        residual = values - terminal
        scale = float(np.max(np.abs(residual)))
    if not np.isfinite(terminal) or not np.isfinite(scale):
        return _unavailable(spec, "numeric_range: terminal mean or residual overflow", warnings, status="error")
    magnitude = max(float(np.max(np.abs(values))), abs(terminal))
    floor = SIGNED_NORMALIZATION_CHECKS["roundoff_eps_factor"] * np.finfo(float).eps * magnitude
    if scale <= floor:
        return _unavailable(spec, f"near_zero_scale: S={scale:.6g} <= 32*eps*max(abs(signal),abs(terminal))={floor:.6g}", warnings)
    normalized = residual / scale
    if float(np.ptp(normalized)) <= floor / scale:
        return _unavailable(spec, "no_temporal_information: signal variation is at or below roundoff", warnings)

    # Compute diagnostics in normalized coordinates to avoid large DC offsets.
    # Quantiles here assess information only; they NEVER replace the max scale.
    with np.errstate(over="ignore", invalid="ignore"):
        duration = float(time[-1] - time[0])
    if not np.isfinite(duration):
        return _unavailable(spec, "numeric_range: time span overflow", warnings, status="error")
    local_time = (time - time[0]) / duration
    if np.any(np.diff(local_time) <= 0):
        return _unavailable(spec, "numeric_range: time resolution lost in stability diagnostics", warnings, status="error")
    noise = _local_noise(local_time, normalized)
    span = float(np.percentile(normalized, 90) - np.percentile(normalized, 10))
    if noise > 0 and span <= SIGNED_NORMALIZATION_CHECKS["low_information_noise_factor"] * noise:
        warnings.append(f"low_information: P90-P10={span:.6g} <= 5*local_noise={5*noise:.6g}; heuristic, not DCT reliability")
    if len(values) < 5:
        warnings.append("noise_unassessed: fewer than 5 observations")

    unstable = _terminal_check(local_time, normalized, selected, warnings)
    if spec.reference_kind == "accepted_final_window" and unstable:
        return _unavailable(spec, "; ".join(unstable), warnings)
    warnings.extend(unstable)
    if spec.reference_kind == "external_known_terminal":
        observed = float(np.mean(normalized[selected]))
        if abs(observed) > 0.05:
            warnings.append(f"observed_terminal_mismatch: abs(tail_mean-terminal)/S={abs(observed):.6g} > 0.05; external reference retained")

    peak = int(np.argmax(np.abs(residual)))  # deterministic: first eligible tie
    second = float(np.partition(np.abs(normalized), -2)[-2])
    if second <= SIGNED_NORMALIZATION_CHECKS["scale_second_largest_ratio"]:
        warnings.append(f"scale_single_point_dominance: raw row {spec.raw_indices[peak]}, second_abs/S={second:.6g} <= 1/3; inspect outlier or sparse sampling, max retained")
    return SignedNormalizationResult(
        spec=spec, state=ResultState(status="warning" if warnings else "completed", warnings=tuple(warnings)),
        normalized_signal=tuple(float(v) for v in normalized), reference_value=terminal,
        scale_value=scale, scale_raw_index=spec.raw_indices[peak],
    )


def inverse_signed_dct(result: SignedNormalizationResult) -> tuple[float, ...]:
    """Restore pre-transform processed signal values; never modify a dataset."""
    require(isinstance(result, SignedNormalizationResult) and result.state.ready,
            "inverse requires an available, current signed normalization result")
    with np.errstate(over="ignore", invalid="ignore"):
        signal = np.asarray(result.normalized_signal) * result.scale_value + result.reference_value
    require(bool(np.all(np.isfinite(signal))), "inverse numeric range overflow")
    return tuple(float(v) for v in signal)


def _apply_signed_step(dataset, step, *, recipe):
    require(set(step.parameters) == {"spec"}, "normalize_signed_dct requires exactly a versioned spec")
    spec = SignedNormalizationSpec.from_dict(step.parameters["spec"])
    result = normalize_signed_dct(dataset, spec, recipe=recipe)
    require(result.state.ready, "; ".join(result.state.blocking_reasons))
    audit = {
        "normalization": result.to_dict(),
        "checks": dict(SIGNED_NORMALIZATION_CHECKS),
        "input_recipe": recipe.to_dict(),
        "input_mask": dataset.mask.tolist(),
        "raw_observations_sha256": raw_observations_sha256(dataset),
        "processed_time_sha256": canonical_sha256(dataset.processed_time.tolist()),
        "time_unit": str(dataset.metadata["time_unit"]),
        "inverse_formula": "sigma_i = Y_i * S + sigma_infinity (input signal unit)",
        "clipped": False, "resampled": False,
    }
    return dataset._with_operation(
        operation="normalize_signed_dct", parameters=audit,
        processed_signal=result.normalized_signal,
        metadata_updates={"signal_type": "normalized", "signal_unit": "1",
                          **dataset._current_unit_key_update("dimensionless"),
                          "normalization_mode": "signed_terminal",
                          "normalization_source": spec.reference_kind,
                          "signed_dct_normalization": audit},
    )


def applied_signed_normalization(
    dataset: EcrDataset, *, recipe: PreprocessConfig,
) -> SignedNormalizationResult:
    """Bind a just-applied result to its real output identity, without self-hashing.

    The audit embedded in the dataset retains output_identity=null; storing the
    dataset's own hash inside itself would be recursive. Bind only on read.
    Further processing requires a fresh explicit preparation, not relabeling.
    """
    require(isinstance(dataset, EcrDataset) and isinstance(recipe, PreprocessConfig), "dataset/recipe required")
    require(bool(recipe.steps) and recipe.steps[-1].operation == "normalize_signed_dct"
            and bool(dataset.history) and dataset.history[-1].operation == "normalize_signed_dct",
            "signed result is not the current last operation")
    record = dataset.history[-1]
    audit = dataset.metadata.get("signed_dct_normalization")
    require(isinstance(audit, Mapping) and record.parameters == audit, "signed audit/history conflict")
    require(audit["checks"] == SIGNED_NORMALIZATION_CHECKS, "unknown signed normalization check policy")
    result = SignedNormalizationResult.from_dict(audit["normalization"])
    prefix = PreprocessConfig(recipe.steps[:-1])
    require(prefix.to_dict() == _plain(audit["input_recipe"])
            and canonical_sha256(prefix.to_dict()) == result.spec.identity.recipe_sha256,
            "signed input recipe conflict")
    require(record.timestamp == recipe.steps[-1].timestamp
            and _plain(recipe.steps[-1].parameters) == {"spec": result.spec.to_dict()}, "signed operation/spec conflict")
    require(raw_observations_sha256(dataset) == audit["raw_observations_sha256"], "signed raw hash conflict")
    require(dataset.mask.tolist() == list(audit["input_mask"])
            and tuple(np.flatnonzero(dataset.mask)) == result.spec.raw_indices,
            "signed mask conflict")
    require(canonical_sha256(dataset.processed_time.tolist()) == audit["processed_time_sha256"]
            and dataset.metadata.get("time_unit") == audit["time_unit"], "signed time axis conflict")
    require(dataset.metadata.get("signal_type") == "normalized" and dataset.metadata.get("signal_unit") == "1"
            and dataset.metadata.get("normalization_mode") == "signed_terminal"
            and tuple(dataset.processed_signal) == result.normalized_signal, "signed output signal/unit conflict")
    return replace(result, output_identity=SourceIdentity.capture(dataset, recipe))


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value
