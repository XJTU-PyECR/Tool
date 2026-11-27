"""Display-only preprocessing form and provenance helpers.

This module never commits a step or changes the scientific algorithms.  A
preview is replayed through the production PreprocessConfig before Accept.
"""
from __future__ import annotations

import json
from math import isfinite
from typing import Mapping

import numpy as np

from pyecr.io.dataset import EcrDataset
from pyecr.preprocessing import PreprocessConfig, ProcessingStep
from pyecr.gui.preprocessing_plots import window_rows, window_is_current


FORM_FIELDS: dict[str, tuple[str, ...]] = {
    "convert_time_unit": ("to_unit",),
    "resistance_to_conductance": (),
    "conductance_to_conductivity": ("path_length_cm", "cross_section_area_cm2"),
    "set_manual_t0": ("t0",),
    "crop": ("start_time", "end_time"),
    "mask_points": ("raw_indices",),
    "unmask_points": ("raw_indices",),
    "choose_initial_stability_window": ("start_time", "end_time"),
    "choose_final_stability_window": ("start_time", "end_time"),
    "normalize": ("mode",),
}

FIELD_DOMAINS: dict[str, str] = {
    "to_unit": "s, ms, min, h or us; conversion uses the current time unit",
    "path_length_cm": "finite number > 0, cm",
    "cross_section_area_cm2": "finite number > 0, cm²",
    "t0": "finite time in the current processed unit; negative fit time is blocked",
    "start_time": "finite inclusive start in the current processed time unit",
    "end_time": "finite inclusive end ≥ start in the current processed time unit",
    "raw_indices": "unique zero-based original row indices, comma separated",
    "mode": "progress or remaining; the Fit gate requires a valid progression",
    "minimum_points": "integer ≥ 3; suggestion only, service default is 5",
    "reference_start": "finite reference start in current processed time unit",
    "reference_end": "finite reference end > start; at least 3 points",
    "scope_start": "finite correction start in current processed time unit",
    "scope_end": "finite correction end > start",
    "drift_method": "Least squares (linear) or robust Theil-Sen (robust_linear)",
    "anchor_time": "finite time in current processed unit; optional",
    "threshold": "finite robust score > 0; default 6",
    "half_window": "integer ≥ 2; default 3",
    "noise_floor": "finite ≥ 0; default 0",
    "target_points": "integer ≥ 3 and < current point count",
    "early_until": "finite time in the current range; all early points retained",
    "late_power": "finite > 1; default 2",
}


def finite_field(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {FIELD_DOMAINS[name]}") from exc
    if not isfinite(number):
        raise ValueError(f"{name}: {FIELD_DOMAINS[name]}")
    return number


def integer_field(value: object, name: str, minimum: int) -> int:
    try:
        text = str(value).strip()
        number = int(text)
        if str(number) != text or number < minimum:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {FIELD_DOMAINS[name]}") from exc
    return number


def parse_indices(value: object) -> list[int]:
    """Accept a comma list or JSON array; never silently deduplicate rows."""
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parts = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(FIELD_DOMAINS["raw_indices"]) from exc
            if not isinstance(parts, list):
                raise ValueError(FIELD_DOMAINS["raw_indices"])
        else:
            parts = [part.strip() for part in text.split(",")]
    indices: list[int] = []
    for part in parts:
        if isinstance(part, bool) or not str(part).isdigit():
            raise ValueError(FIELD_DOMAINS["raw_indices"])
        indices.append(int(part))
    if len(indices) != len(set(indices)):
        raise ValueError("raw_indices: duplicate original row index")
    return indices


def generic_parameters(operation: str, values: Mapping[str, object]) -> dict[str, object]:
    """Parse visible typed fields into the existing ProcessingStep schema."""
    if operation not in FORM_FIELDS:
        raise ValueError(f"unsupported form operation: {operation}")
    parameters: dict[str, object] = {}
    for name in FORM_FIELDS[operation]:
        value = values[name]
        if name in ("to_unit", "mode"):
            choice = str(value).strip()
            allowed = ("s", "ms", "min", "h", "us") if name == "to_unit" else ("progress", "remaining")
            if choice not in allowed:
                raise ValueError(f"{name}: {FIELD_DOMAINS[name]}")
            parameters[name] = choice
        elif name == "raw_indices":
            parameters[name] = parse_indices(value)
        else:
            parameters[name] = finite_field(value, name)
    if "start_time" in parameters and parameters["end_time"] < parameters["start_time"]:
        raise ValueError("end_time must be ≥ start_time")
    for positive in ("path_length_cm", "cross_section_area_cm2"):
        if positive in parameters and parameters[positive] <= 0:
            raise ValueError(f"{positive}: {FIELD_DOMAINS[positive]}")
    if operation == 'conductance_to_conductivity':
        parameters['unit_policy'] = 'recorded_conductance_to_S'
    return parameters


def optional_parameters(kind: str, values: Mapping[str, object]) -> dict[str, object]:
    if kind == "drift_correction":
        method = {'Least squares':'linear','Robust (Theil-Sen)':'robust_linear'}.get(str(values['drift_method']),str(values['drift_method']))
        if method not in ("linear", "robust_linear"):
            raise ValueError(f"drift_method: {FIELD_DOMAINS['drift_method']}")
        reference = [finite_field(values[key], key) for key in ("reference_start", "reference_end")]
        scope = [finite_field(values[key], key) for key in ("scope_start", "scope_end")]
        if reference[1] <= reference[0]:
            raise ValueError("reference_end must be > reference_start")
        if scope[1] <= scope[0]:
            raise ValueError("scope_end must be > scope_start")
        result: dict[str, object] = {"method": method, "reference_window": reference, "scope": scope}
        anchor = str(values.get("anchor_time", "")).strip()
        if anchor:
            result["anchor_time"] = finite_field(anchor, "anchor_time")
        return result
    if kind == "outlier_mask":
        threshold = finite_field(values["threshold"], "threshold")
        half_window = integer_field(values["half_window"], "half_window", 2)
        noise_floor = finite_field(values["noise_floor"], "noise_floor")
        if threshold <= 0 or noise_floor < 0:
            raise ValueError("threshold must be > 0 and noise_floor must be ≥ 0")
        return {"threshold": threshold, "half_window": half_window, "noise_floor": noise_floor}
    if kind == "adaptive_downsample":
        target = integer_field(values["target_points"], "target_points", 3)
        early = finite_field(values["early_until"], "early_until")
        power = finite_field(values["late_power"], "late_power")
        if power <= 1:
            raise ValueError(f"late_power: {FIELD_DOMAINS['late_power']}")
        return {"target_points": target, "early_until": early, "late_power": power}
    raise ValueError(f"unsupported optional operation: {kind}")


def excluded_by_reason(source: EcrDataset, config: PreprocessConfig) -> dict[str, list[int]]:
    """Explain current exclusions from replayed masks, with independent reasons."""
    categories: dict[int, str] = {}
    previous = source
    for position, step in enumerate(config.steps, 1):
        # Unmask acts retroactively on earlier explicit masks. Include future
        # unmask steps while attributing each intermediate exclusion.
        future_unmask = tuple(s for s in config.steps[position:] if s.operation == "unmask_points")
        current = PreprocessConfig((*config.steps[:position], *future_unmask)).replay(source)
        before = set(np.flatnonzero(previous.mask).tolist())
        after = set(np.flatnonzero(current.mask).tolist())
        for index in after - before:
            categories.pop(index, None)
        for index in before - after:
            categories[index] = step.operation
        previous = current
    result: dict[str, list[int]] = {}
    for index, reason in sorted(categories.items()):
        result.setdefault(reason, []).append(index)
    return result


def reviewable_mask_indices(config: PreprocessConfig) -> tuple[int, ...]:
    """Rows explicitly masked by a user or accepted outlier step, excluding crops."""
    indices: set[int] = set()
    for step in config.steps:
        if step.operation in ("mask_points", "outlier_mask"):
            indices.update(step.parameters["raw_indices"])
        elif step.operation == "unmask_points":
            indices.difference_update(step.parameters["raw_indices"])
    return tuple(sorted(indices))


def preview_evidence(source: EcrDataset, current: EcrDataset, proposed: EcrDataset,
                     step: ProcessingStep, config: PreprocessConfig) -> str:
    """Readable review of arrays, windows, statistics, indices and t0."""
    kept = np.flatnonzero(proposed.mask).tolist()
    removed = excluded_by_reason(source, config.append(step))
    time = proposed.processed_time
    signal = proposed.processed_signal
    lines = [
        f"Preview only · {step.operation} · no state changed",
        f"Raw points: {source.raw_time.size}; current: {current.processed_time.size}; proposed: {time.size}",
        f"Selected original indices ({len(kept)}): {kept}",
        f"Excluded by reason: {json.dumps(removed, ensure_ascii=False)}",
        f"Current time unit: {proposed.metadata.get('time_unit', 's')}; manual t0: {proposed.metadata.get('manual_t0', 'not set')}",
    ]
    windows = proposed.metadata.get("stability_windows", {})
    for kind in ("initial", "final"):
        item = windows.get(kind) if isinstance(windows, Mapping) else None
        if isinstance(item, Mapping):
            start, end = item["start_time"], item["end_time"]
            selected = window_rows(proposed,item)
            if np.count_nonzero(selected) >= 2:
                slope = float(np.polyfit(time[selected], signal[selected], 1)[0])
                noise = float(np.std(signal[selected] - np.polyval(np.polyfit(time[selected], signal[selected], 1), time[selected])))
            else:
                slope = noise = float("nan")
            status='' if window_is_current(proposed,kind,item) else '; recorded window is stale — reselect before use'
            if proposed.metadata.get('normalization_source') == 'stability_windows' and window_is_current(proposed,kind,item):
                status='; fixed normalization reference, including rows outside fit interval'
            current_bounds=proposed.stability_window_bounds(kind,item)
            status+=f'; current bounds [{current_bounds[0]:g}, {current_bounds[1]:g}] {proposed.metadata.get("time_unit","s")}'
            lines.append(f"{kind} window recorded [{start:g}, {end:g}] {item.get('time_unit','unknown')}: {int(selected.sum())} retained original rows on current {proposed.metadata.get('time_unit','s')} axis; slope={slope:.6g}; residual std={noise:.6g}; original indices={item.get('raw_indices', [])}{status}")
    lines.append("Review the overlay and Fit quality before Accept. Reject leaves the recipe unchanged.")
    return "\n".join(lines)


_FORMULAS = {
    "convert_time_unit": "t_new = convert(t_current, from_unit, to_unit)",
    "resistance_to_conductance": "G = 1 / R after unit conversion",
    "conductance_to_conductivity": "sigma = G × path_length / cross_section_area",
    "set_manual_t0": "t_new = t_current - t0",
    "crop": "retain inclusive processed time interval",
    "mask_points": "exclude original rows from processed view; raw retained",
    "choose_initial_stability_window": "record initial plateau from selected processed interval",
    "choose_final_stability_window": "record final plateau from selected processed interval",
    "normalize": "use stored initial/final window means; see recorded mode",
    "normalize_signed_dct": "Y=(sigma-sigma_inf)/max_all_eligible(abs(sigma-sigma_inf)); terminal reference",
    "outlier_mask": "exclude reviewed original rows; raw retained",
    "unmask_points": "replay with prior explicit masks cancelled for these rows",
    "adaptive_downsample": "select stored original rows; no interpolation or averaging",
}


def history_evidence(config: PreprocessConfig, revision: int, source: EcrDataset | None = None) -> str:
    if not config.steps:
        return f"No preprocessing steps. Pipeline revision {revision}."
    lines = [f"Current pipeline revision: {revision}; {len(config.steps)} immutable steps.\n"]
    for number, step in enumerate(config.steps, 1):
        p = step.to_dict()["parameters"]
        before = PreprocessConfig(config.steps[:number - 1]).replay(source) if source is not None else None
        after = PreprocessConfig(config.steps[:number]).replay(source) if source is not None else None
        actual = after.history[-1].parameters if after is not None else {}
        origin = actual.get("source", p.get("source", "not recorded by step schema"))
        signed = actual.get("normalization") if step.operation == "normalize_signed_dct" else None
        if isinstance(signed, Mapping):
            origin = signed["spec"]["reference_kind"]
        original_indices = actual.get("raw_indices", actual.get("selected_indices", actual.get(
            "applied_raw_indices", p.get("raw_indices", p.get("selected_indices",
            np.flatnonzero(after.mask).tolist() if after is not None else "not recorded")))))
        excluded = (sorted(set(np.flatnonzero(before.mask)) - set(np.flatnonzero(after.mask)))
                    if before is not None and after is not None else [])
        restored = (sorted(set(np.flatnonzero(after.mask)) - set(np.flatnonzero(before.mask)))
                    if before is not None and after is not None else [])
        lines.append(
            f"{number}. {step.operation} | source={origin} | step schema={step.schema_version} | "
            f"recorded={step.timestamp}\n"
            f"   formula={p.get('formula', _FORMULAS.get(step.operation, 'production ' + step.operation))}; "
            f"scope={p.get('scope', [p.get('start_time'), p.get('end_time')])}; "
            f"original indices={original_indices}; excluded={excluded}; restored={restored}\n"
            f"   parameters={json.dumps(p, ensure_ascii=False, sort_keys=True)}"
        )
        if isinstance(signed, Mapping):
            lines.append(f"   sigma_inf={signed['reference_value']}; S={signed['scale_value']} {signed['spec']['signal_unit']}; "
                         f"max raw index={signed['scale_raw_index']}; profile={signed['spec']['profile_id']}; "
                         f"warnings={list(signed['state']['warnings'])}")
    return "\n".join(lines)
