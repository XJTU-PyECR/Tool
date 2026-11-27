"""Presentation adapters for P02; no independent normalization equations."""
from collections.abc import Mapping
from math import isfinite

import numpy as np

from pyecr.contracts import SignedNormalizationSpec, SourceIdentity
from pyecr.preprocessing import ProcessingStep, normalize_signed_dct
from pyecr.preprocessing.signed_normalization import cumulative_t0_s
from pyecr.preprocessing.quality import SIGNED_PROFILE_FIT_REASON


SIGNED_OPERATION = "normalize_signed_dct"
SIGNED_DISPLAY = "Signed-DCT normalization"
REFERENCE_CHOICES = ("Accepted final plateau", "External known terminal")
SIGNED_FIELDS = ("reference_source", "external_value", "reference_evidence", "external_declared")


def operation_id(display):
    return SIGNED_OPERATION if display == SIGNED_DISPLAY else str(display)


def form_signature(values):
    # Old non-UI callers declare the current unit via the P03 dataset contract.
    # R04 direct forms additionally freeze the visible, locked unit field.
    return tuple(values.get(name) for name in (*SIGNED_FIELDS, 'external_unit'))


def prepare_signed_normalization(dataset, recipe, values):
    """Prepare a spec/result only. Accept is the only UI commit boundary."""
    source = values.get("reference_source")
    if source not in REFERENCE_CHOICES:
        raise ValueError("Select an accepted final plateau or an explicitly known external terminal")
    evidence = str(values.get("reference_evidence", "")).strip()
    rows = ()
    terminal = None
    if source == REFERENCE_CHOICES[0]:
        windows = dataset.metadata.get("stability_windows", {})
        window = windows.get("final") if isinstance(windows, Mapping) else None
        if not isinstance(window, Mapping):
            raise ValueError("No accepted final plateau. Open Platforms, Preview and Accept the Final window first, or explicitly declare an external known terminal.")
        rows = tuple(int(i) for i in window["raw_indices"])
        evidence = evidence or "User accepted final stability window recorded in this dataset's processing history"
        reference_kind = "accepted_final_window"
    else:
        if 'external_unit' in values and values['external_unit'] != dataset.metadata['signal_unit']:
            raise ValueError('External terminal signal unit must match the current input unit; no implicit conversion')
        if values.get("external_declared") is not True:
            raise ValueError("Explicit declaration required: this terminal is externally known, not inferred from this curve or a model")
        raw_value = values.get("external_value", "")
        try:
            if isinstance(raw_value, bool):
                raise ValueError
            terminal = float(raw_value)
            if not isfinite(terminal):
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise ValueError("External terminal must be a finite value in the current input signal unit") from exc
        if not evidence:
            raise ValueError("External terminal requires an explicit source/evidence statement")
        reference_kind = "external_known_terminal"
    spec = SignedNormalizationSpec(
        identity=SourceIdentity.capture(dataset, recipe), raw_count=len(dataset.raw_time),
        raw_indices=tuple(int(i) for i in np.flatnonzero(dataset.mask)),
        signal_type=str(dataset.metadata["signal_type"]), signal_unit=str(dataset.metadata["signal_unit"]),
        reference_kind=reference_kind, reference_rows=rows, reference_value=terminal,
        reference_evidence=evidence, t0_s=cumulative_t0_s(dataset),
    )
    result = normalize_signed_dct(dataset, spec, recipe=recipe)
    return ProcessingStep.create(SIGNED_OPERATION, {"spec": spec.to_dict()}), result


def signed_review_text(dataset, result):
    spec, state = result.spec, result.state
    time = dataset.processed_time
    unit = str(dataset.metadata["time_unit"])
    lines = [
        f"{SIGNED_DISPLAY} | {'Accepted profile' if result.output_identity else 'Preview only'} | {state.status}",
        "Y=(sigma-sigma_inf)/S; S=max over ALL eligible observations | output unit 1",
        f"Source: {spec.reference_kind} | input {spec.signal_type} ({spec.signal_unit})",
        "Blocking: " + ("; ".join(state.blocking_reasons) or "none"),
        "Warnings: " + ("; ".join(state.warnings) or "none"),
        f"Original rows: {spec.raw_count}; complete eligible rows: {len(spec.raw_indices)}; mask unchanged",
        f"Eligible interval: [{time[0]:.9g}, {time[-1]:.9g}] {unit} | cumulative t0={spec.t0_s:.9g} s",
    ]
    if spec.reference_rows:
        selected = np.isin(spec.raw_indices, spec.reference_rows)
        times = time[selected]
        lines.append(f"Selected final interval: [{times[0]:.9g}, {times[-1]:.9g}] {unit}; {len(times)} points")
    else:
        lines.append("Selected final interval: none; terminal is explicitly external (tail checks do not set it)")
    if state.ready:
        lines.extend((f"sigma_inf (σ∞) = {result.reference_value:.12g} {spec.signal_unit}",
                      f"S = {result.scale_value:.12g} {spec.signal_unit}",
                      f"Maximum original row index = {result.scale_raw_index} (zero-based)",
                      f"Overlay: {len(result.normalized_signal)} input and {len(result.normalized_signal)} Y points; separate unit axes"))
    else:
        lines.append("sigma_inf (σ∞), S and Y: unavailable; no fabricated numerical payload")
    lines.extend((f"Evidence: {spec.reference_evidence}",
                  f"Input dataset SHA256: {spec.identity.dataset_sha256}",
                  f"Input recipe SHA256: {spec.identity.recipe_sha256}",
                  f"Profile: {spec.profile_id} | {spec.algorithm_version}", SIGNED_PROFILE_FIT_REASON))
    return "\n".join(lines)
