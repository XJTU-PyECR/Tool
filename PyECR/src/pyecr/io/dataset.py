"""Immutable ECR data container and auditable preprocessing operations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.units import factor_to_reference, get_default_unit, validate_pair

__all__ = ["EcrDataset", "ProcessingRecord"]


_TIME_FACTORS_TO_SECONDS = {
    "us": 1.0e-6,
    "ms": 1.0e-3,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}
_TIME_ALIASES = {
    "µs": "us",
    "μs": "us",
    "microsecond": "us",
    "microseconds": "us",
    "millisecond": "ms",
    "milliseconds": "ms",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hr": "h",
    "hour": "h",
    "hours": "h",
}
@dataclass(frozen=True, slots=True)
class ProcessingRecord:
    """One immutable and JSON-serializable preprocessing audit entry."""

    operation: str
    parameters: Mapping[str, object]
    timestamp: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        operation = str(self.operation).strip()
        if not operation:
            raise ValueError("operation must not be empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        timestamp = str(self.timestamp)
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO 8601 datetime") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "schema_version", 1)

    @classmethod
    def create(
        cls,
        operation: str,
        parameters: Mapping[str, object],
        *,
        timestamp: str | None = None,
        schema_version: int = 1,
    ) -> ProcessingRecord:
        """Create an entry stamped with the current UTC time."""

        return cls(
            operation=operation,
            parameters=parameters,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a mutable JSON-compatible representation."""

        return {
            "operation": self.operation,
            "parameters": _thaw_json(self.parameters),
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProcessingRecord:
        """Restore an audit entry from :meth:`to_dict` output."""

        try:
            operation = value["operation"]
            parameters = value["parameters"]
            timestamp = value["timestamp"]
        except KeyError as exc:
            raise ValueError(f"missing processing-record field: {exc.args[0]}") from exc
        if not isinstance(operation, str) or not isinstance(parameters, Mapping):
            raise TypeError("invalid processing-record value types")
        if not isinstance(timestamp, str):
            raise TypeError("processing-record timestamp must be a string")
        schema_version = value.get("schema_version", 1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, Integral):
            raise TypeError("processing-record schema_version must be an integer")
        return cls(operation, parameters, timestamp, int(schema_version))


@dataclass(frozen=True, slots=True)
class EcrDataset:
    """Raw ECR observations plus an immutable processed view.

    ``mask`` is aligned with the raw arrays.  The two processed arrays contain
    exactly the currently retained observations after all recorded operations.
    Raw arrays are copied on construction and are never changed by a method.
    """

    raw_time: ArrayLike
    raw_signal: ArrayLike
    processed_time: ArrayLike | None = None
    processed_signal: ArrayLike | None = None
    mask: ArrayLike | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    history: tuple[ProcessingRecord, ...] = ()

    def __post_init__(self) -> None:
        raw_time = _read_only_float_vector("raw_time", self.raw_time)
        raw_signal = _read_only_float_vector("raw_signal", self.raw_signal)
        if raw_time.size != raw_signal.size:
            raise ValueError("raw_time and raw_signal must have the same length")
        if raw_time.size == 0:
            raise ValueError("raw arrays must contain at least one observation")
        _require_strictly_increasing("raw_time", raw_time)

        mask = (
            np.ones(raw_time.size, dtype=np.bool_)
            if self.mask is None
            else _boolean_mask(self.mask, raw_time.size)
        )
        if not np.any(mask):
            raise ValueError("mask must retain at least one observation")
        # A view backed by immutable bytes cannot be made writable later with
        # ``setflags(write=True)`` through the public ndarray attribute.
        mask = np.frombuffer(mask.tobytes(), dtype=np.bool_)

        processed_time = _read_only_float_vector(
            "processed_time",
            raw_time[mask] if self.processed_time is None else self.processed_time,
        )
        processed_signal = _read_only_float_vector(
            "processed_signal",
            raw_signal[mask] if self.processed_signal is None else self.processed_signal,
        )
        retained_count = int(np.count_nonzero(mask))
        if processed_time.size != retained_count or processed_signal.size != retained_count:
            raise ValueError(
                "processed arrays must contain one value for every True mask entry"
            )
        _require_strictly_increasing("processed_time", processed_time)

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        history = tuple(
            item
            if isinstance(item, ProcessingRecord)
            else ProcessingRecord.from_dict(item)
            for item in self.history
        )
        object.__setattr__(self, "raw_time", raw_time)
        object.__setattr__(self, "raw_signal", raw_signal)
        object.__setattr__(self, "processed_time", processed_time)
        object.__setattr__(self, "processed_signal", processed_signal)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        object.__setattr__(self, "history", history)

    @classmethod
    def from_raw(
        cls,
        raw_time: ArrayLike,
        raw_signal: ArrayLike,
        *,
        time_unit: str = "s",
        signal_type: str = "unknown",
        signal_unit: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> EcrDataset:
        """Create an unprocessed dataset while recording input unit semantics."""

        canonical_time_unit = _canonical_time_unit(time_unit)
        normalized_signal_type = str(signal_type).strip().lower()
        if not normalized_signal_type:
            raise ValueError("signal_type must not be empty")
        if signal_unit is None:
            try:
                signal_unit = get_default_unit(normalized_signal_type).label
            except ValueError:
                signal_unit = "arbitrary"  # Legacy API accepts other signal types.
        try:
            signal_unit_key = validate_pair(normalized_signal_type, str(signal_unit))
        except ValueError:
            signal_unit_key = None  # Legacy API/project data remain representable.
        values = dict(metadata or {})
        values.update(
            {
                "raw_time_unit": canonical_time_unit,
                "time_unit": canonical_time_unit,
                "raw_signal_type": normalized_signal_type,
                "signal_type": normalized_signal_type,
                "raw_signal_unit": str(signal_unit),
                "signal_unit": str(signal_unit),
            }
        )
        if signal_unit_key is not None:
            values.update({"raw_signal_unit_key": signal_unit_key,
                           "signal_unit_key": signal_unit_key})
        else:
            values.pop("raw_signal_unit_key", None)
            values.pop("signal_unit_key", None)
        return cls(raw_time=raw_time, raw_signal=raw_signal, metadata=values)

    def convert_time_unit(
        self,
        to_unit: str = "s",
        *,
        from_unit: str | None = None,
    ) -> EcrDataset:
        """Convert processed time values while preserving the imported time array."""

        target = _canonical_time_unit(to_unit)
        recorded_source = self.metadata.get("time_unit")
        source = _canonical_time_unit(
            from_unit if from_unit is not None else str(recorded_source or "s")
        )
        if from_unit is not None and recorded_source is not None:
            recorded = _canonical_time_unit(str(recorded_source))
            if source != recorded:
                raise ValueError(
                    f"from_unit {source!r} conflicts with recorded unit {recorded!r}"
                )
        scale = _TIME_FACTORS_TO_SECONDS[source] / _TIME_FACTORS_TO_SECONDS[target]
        return self._with_operation(
            processed_time=self.processed_time * scale,
            operation="convert_time_unit",
            parameters={
                "from_unit": source,
                "to_unit": target,
                "scale_factor": scale,
            },
            metadata_updates={"time_unit": target},
        )

    def resistance_to_conductance(
        self,
        *,
        resistance_unit: str | None = None,
    ) -> EcrDataset:
        """Convert positive resistance observations to conductance in siemens."""

        current_type = str(self.metadata.get("signal_type", "unknown")).lower()
        if current_type not in {"unknown", "resistance"}:
            raise ValueError(
                "resistance_to_conductance requires resistance or unknown signal_type"
            )
        recorded_unit = str(self.metadata.get("signal_unit", "ohm"))
        supplied_unit = resistance_unit or (
            "ohm" if current_type == "unknown" else recorded_unit
        )
        canonical_unit = validate_pair("resistance", supplied_unit)
        factor = factor_to_reference(canonical_unit)
        # Keep old replay records byte-compatible; new imports carry the key.
        record_unit = (
            "Mohm"
            if canonical_unit == "megaohm" and "signal_unit_key" not in self.metadata
            else canonical_unit
        )
        resistance_ohm = self.processed_signal * factor
        if np.any(resistance_ohm <= 0.0):
            raise ValueError("resistance values must be strictly positive")
        return self._with_operation(
            processed_signal=1.0 / resistance_ohm,
            operation="resistance_to_conductance",
            parameters={
                "input_unit": record_unit,
                "output_unit": "S",
                "ohm_scale_factor": factor,
            },
            metadata_updates={"signal_type": "conductance", "signal_unit": "S",
                              **self._current_unit_key_update("siemens")},
        )

    def conductance_to_conductivity(
        self,
        *,
        path_length_cm: float,
        cross_section_area_cm2: float,
        unit_policy: str = 'recorded_conductance_to_S',
    ) -> EcrDataset:
        """Apply sigma = G L/A using the electrical measurement geometry."""

        current_type = str(self.metadata.get("signal_type", "unknown")).lower()
        if current_type not in {"unknown", "conductance"}:
            raise ValueError(
                "conductance_to_conductivity requires conductance or unknown signal_type"
            )
        length = _positive_finite("path_length_cm", path_length_cm)
        area = _positive_finite(
            "cross_section_area_cm2", cross_section_area_cm2
        )
        geometry_factor = length / area
        if unit_policy not in {'recorded_conductance_to_S','legacy_assume_S'}:
            raise ValueError('Unknown conductance conversion unit policy')
        scale = 1.0
        if unit_policy == 'recorded_conductance_to_S':
            unit = 'S' if current_type == 'unknown' else str(self.metadata.get('signal_unit','S'))
            scale = factor_to_reference(validate_pair('conductance',unit))
        return self._with_operation(
            processed_signal=self.processed_signal * scale * geometry_factor,
            operation="conductance_to_conductivity",
            parameters={
                "path_length_cm": length,
                "cross_section_area_cm2": area,
                "geometry_factor_cm_inverse": geometry_factor,
                **({'input_unit': str(self.metadata.get('signal_unit','S')), 'siemens_scale_factor':scale,
                    'unit_policy':unit_policy} if scale != 1.0 else {}),
            },
            metadata_updates={
                "signal_type": "conductivity",
                "signal_unit": "S/cm",
                **self._current_unit_key_update("S_per_cm"),
                "electrical_path_length_cm": length,
                "electrical_cross_section_area_cm2": area,
            },
        )

    def apply_start_index(self, start_index: int) -> EcrDataset:
        """Exclude raw rows before a zero-based index without deleting them."""

        if isinstance(start_index, bool) or not isinstance(start_index, Integral):
            raise TypeError("start_index must be an integer")
        index = int(start_index)
        if index < 0 or index >= self.raw_time.size:
            raise ValueError(
                f"start_index must be between 0 and {self.raw_time.size - 1}"
            )
        raw_indices = np.flatnonzero(self.mask)
        keep_processed = raw_indices >= index
        new_mask = np.array(self.mask, copy=True)
        new_mask[:index] = False
        if not np.any(new_mask):
            raise ValueError("start_index would remove every observation")
        return self._with_operation(
            processed_time=self.processed_time[keep_processed],
            processed_signal=self.processed_signal[keep_processed],
            mask=new_mask,
            operation="apply_start_index",
            parameters={
                "start_index": index,
                "excluded_raw_observations": int(np.count_nonzero(~new_mask)),
            },
            metadata_updates={"start_index": index},
        )

    def set_manual_t0(self, t0: float) -> EcrDataset:
        """Shift processed times by a user-selected origin in the current unit."""

        origin = _finite_scalar("t0", t0)
        current_unit = str(self.metadata.get("time_unit", "s"))
        return self._with_operation(
            processed_time=self.processed_time - origin,
            operation="set_manual_t0",
            parameters={"t0": origin, "time_unit": current_unit},
            metadata_updates={"manual_t0": origin},
        )

    def crop(self, *, start_time: float, end_time: float) -> EcrDataset:
        """Retain an inclusive interval in the current processed time axis."""

        start = _finite_scalar("start_time", start_time)
        end = _finite_scalar("end_time", end_time)
        if end < start:
            raise ValueError("end_time must be greater than or equal to start_time")
        keep_processed = (self.processed_time >= start) & (self.processed_time <= end)
        if not np.any(keep_processed):
            raise ValueError("crop interval contains no processed observations")
        raw_indices = np.flatnonzero(self.mask)
        new_mask = np.array(self.mask, copy=True)
        new_mask[raw_indices[~keep_processed]] = False
        return self._with_operation(
            processed_time=self.processed_time[keep_processed],
            processed_signal=self.processed_signal[keep_processed],
            mask=new_mask,
            operation="crop",
            parameters={
                "start_time": start,
                "end_time": end,
                "time_unit": str(self.metadata.get("time_unit", "s")),
                "retained_observations": int(np.count_nonzero(keep_processed)),
            },
        )

    def mask_points(self, raw_indices: ArrayLike) -> EcrDataset:
        """Exclude explicit raw-row indices without deleting raw observations."""

        indices = _raw_index_list(raw_indices, self.raw_time.size)
        new_mask = np.array(self.mask, copy=True)
        new_mask[indices] = False
        if not np.any(new_mask):
            raise ValueError("mask_points would remove every observation")
        current_raw_indices = np.flatnonzero(self.mask)
        keep_processed = ~np.isin(current_raw_indices, indices)
        return self._with_operation(
            processed_time=self.processed_time[keep_processed],
            processed_signal=self.processed_signal[keep_processed],
            mask=new_mask,
            operation="mask_points",
            parameters={
                "raw_indices": indices.tolist(),
                "retained_observations": int(np.count_nonzero(new_mask)),
            },
        )

    def choose_initial_stability_window(
        self,
        *,
        start_time: float,
        end_time: float,
    ) -> EcrDataset:
        """Record the initial plateau interval and its signal statistics."""

        return self._choose_stability_window(
            "initial", start_time=start_time, end_time=end_time
        )

    def choose_final_stability_window(
        self,
        *,
        start_time: float,
        end_time: float,
    ) -> EcrDataset:
        """Record the final plateau interval and its signal statistics."""

        return self._choose_stability_window(
            "final", start_time=start_time, end_time=end_time
        )

    def _choose_stability_window(
        self,
        kind: str,
        *,
        start_time: float,
        end_time: float,
    ) -> EcrDataset:
        start = _finite_scalar("start_time", start_time)
        end = _finite_scalar("end_time", end_time)
        if end < start:
            raise ValueError("end_time must be greater than or equal to start_time")
        selected = (self.processed_time >= start) & (self.processed_time <= end)
        if not np.any(selected):
            raise ValueError(
                f"{kind} stability window contains no processed observations"
            )
        values = self.processed_signal[selected]
        current_raw_indices = np.flatnonzero(self.mask)
        descriptor = {
            "start_time": start,
            "end_time": end,
            "time_unit": str(self.metadata.get("time_unit", "s")),
            "raw_indices": current_raw_indices[selected].tolist(),
            "n_points": int(values.size),
            "mean": float(np.mean(values)),
            "standard_deviation": (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ),
            "signal_type": str(self.metadata.get("signal_type", "unknown")),
            "signal_unit": str(self.metadata.get("signal_unit", "arbitrary")),
        }
        windows_value = self.metadata.get("stability_windows", {})
        windows = (
            _thaw_json(windows_value) if isinstance(windows_value, Mapping) else {}
        )
        windows[kind] = descriptor
        operation = f"choose_{kind}_stability_window"
        return self._with_operation(
            operation=operation,
            parameters=descriptor,
            metadata_updates={"stability_windows": windows},
        )

    def normalize_from_stability_windows(
        self,
        *,
        mode: str = "progress",
        record_reference: bool = True,
    ) -> EcrDataset:
        """Normalize using the means of explicit initial and final plateaus."""

        windows = self.metadata.get("stability_windows")
        if not isinstance(windows, Mapping) or not all(
            key in windows for key in ("initial", "final")
        ):
            raise ValueError(
                "initial and final stability windows are required before normalization"
            )
        initial_window = windows["initial"]
        final_window = windows["final"]
        if not isinstance(initial_window, Mapping) or not isinstance(
            final_window, Mapping
        ):
            raise ValueError("stability window metadata is invalid")
        self._validate_stability_window_is_current("initial", initial_window, require_unchanged_signal=True)
        self._validate_stability_window_is_current("final", final_window, require_unchanged_signal=True)
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"progress", "remaining"}:
            raise ValueError("mode must be 'progress' or 'remaining'")
        initial = _finite_scalar("initial stability mean", initial_window["mean"])
        final = _finite_scalar("final stability mean", final_window["mean"])
        scale = max(abs(initial), abs(final), 1.0)
        if abs(final - initial) <= np.finfo(np.float64).eps * scale:
            raise ValueError("initial and final normalization values must be different")
        progress = (self.processed_signal - initial) / (final - initial)
        normalized = progress if normalized_mode == "progress" else 1.0 - progress
        reference = {}
        if record_reference:
            reference = {
                "reference_policy": "fixed-plateaus/1",
                "reference_windows": {
                    "initial": _thaw_json(initial_window),
                    "final": _thaw_json(final_window),
                },
                "reference_endpoints": {
                    "initial": float(normalized[0]),
                    "final": float(normalized[-1]),
                },
            }
        return self._with_operation(
            processed_signal=normalized,
            operation="normalize",
            parameters={
                "mode": normalized_mode,
                "initial_value": initial,
                "final_value": final,
                "initial_points": int(initial_window["n_points"]),
                "final_points": int(final_window["n_points"]),
                "source": "stability_windows",
                "clipped": False,
                **reference,
            },
            metadata_updates={
                "signal_type": "normalized",
                "signal_unit": "1",
                **self._current_unit_key_update("dimensionless"),
                "normalization_mode": normalized_mode,
                "normalization_source": "stability_windows",
            },
        )

    def _validate_stability_window_is_current(
        self,
        kind: str,
        window: Mapping[str, object],
        *,
        allow_normalized_signal: bool = False,
        require_unchanged_signal: bool = False,
    ) -> None:
        if allow_normalized_signal:
            reference = self.normalization_reference()
            if reference is not None:
                recorded = reference["windows"].get(kind)
                if recorded != window:
                    raise ValueError(f"{kind} stability window differs from the fixed normalization reference")
                return
        if not (allow_normalized_signal and self.metadata.get("signal_type") == "normalized") and (
            window.get("signal_type") != self.metadata.get("signal_type")
            or window.get("signal_unit") != self.metadata.get("signal_unit")
        ):
            raise ValueError(
                f"{kind} stability window is Out of date after signal conversion; choose it again"
            )
        indices = _raw_index_list(window.get("raw_indices", ()), self.raw_time.size)
        if not np.all(self.mask[indices]):
            raise ValueError(
                f"{kind} stability window is Out of date after crop/mask; choose it again"
            )
        start, end = self.stability_window_bounds(kind, window)
        # Membership is tied to original rows; affine time conversions preserve it.
        tolerance = 32 * np.finfo(float).eps * max(abs(start), abs(end), 1.0)
        selected = (self.processed_time >= start - tolerance) & (self.processed_time <= end + tolerance)
        if not np.array_equal(np.flatnonzero(self.mask)[selected], indices):
            raise ValueError(
                f"{kind} stability window is Out of date on the current time axis; choose it again"
            )
        latest_choice = self._stability_choice_index(kind)
        if require_unchanged_signal and any(record.operation not in {
            "choose_initial_stability_window", "choose_final_stability_window",
            "set_manual_t0", "convert_time_unit", "crop", "apply_start_index",
            "mask_points", "outlier_mask", "unmask_points", "adaptive_downsample",
        } for record in self.history[latest_choice + 1:]):
            raise ValueError(f"{kind} stability window is Out of date after a signal change; choose it again")

    def _stability_choice_index(self, kind: str) -> int:
        return next((i for i in range(len(self.history) - 1, -1, -1)
                     if self.history[i].operation == f"choose_{kind}_stability_window"), -1)

    def stability_window_bounds(self, kind: str, window: Mapping[str, object]) -> tuple[float, float]:
        """Current coordinates of recorded bounds; original audit stays unchanged."""
        start, end = float(window["start_time"]), float(window["end_time"])
        unit = str(window["time_unit"])
        for record in self.history[self._stability_choice_index(kind) + 1:]:
            if record.operation == "set_manual_t0":
                if record.parameters["time_unit"] != unit:
                    raise ValueError("stability window time unit conflicts with recorded t0")
                start -= float(record.parameters["t0"])
                end -= float(record.parameters["t0"])
            elif record.operation == "convert_time_unit":
                if record.parameters["from_unit"] != unit:
                    raise ValueError("stability window time unit conflicts with conversion")
                factor = float(record.parameters["scale_factor"])
                start, end = start * factor, end * factor
                unit = str(record.parameters["to_unit"])
        if unit != self.metadata.get("time_unit"):
            raise ValueError("stability window is Out of date on the current time axis")
        return start, end

    def normalization_reference(self) -> Mapping[str, object] | None:
        """Fixed accepted plateaus survive view selection, never signal edits.

        Legacy records recover their descriptors from prior choice records without
        rewriting the saved recipe, dataset metadata or history.
        """
        if (self.metadata.get("signal_type") != "normalized" or
                self.metadata.get("normalization_source") != "stability_windows"):
            return None
        index = next((i for i in range(len(self.history) - 1, -1, -1)
                      if self.history[i].operation == "normalize"), -1)
        if index < 0:
            raise ValueError("Fixed normalization reference is missing; normalize again from the source")
        record = self.history[index]
        p = record.parameters
        if p.get("source") != "stability_windows" or p.get("mode") != self.metadata.get("normalization_mode"):
            raise ValueError("Normalization reference conflicts with the recorded representation")
        allowed = {"set_manual_t0", "convert_time_unit", "crop", "apply_start_index",
                   "mask_points", "outlier_mask", "unmask_points", "adaptive_downsample"}
        if any(r.operation not in allowed for r in self.history[index + 1:]):
            raise ValueError("Fixed normalization reference is Out of date after signal/platform changes; Undo and review normalization")
        windows = p.get("reference_windows")
        if windows is None:
            windows = {}
            for old in self.history[:index]:
                for kind in ("initial", "final"):
                    if old.operation == f"choose_{kind}_stability_window":
                        windows[kind] = old.parameters
        if not isinstance(windows, Mapping) or not all(k in windows for k in ("initial", "final")):
            raise ValueError("Fixed normalization plateau evidence is incomplete")
        for kind, key in (("initial", "initial_value"), ("final", "final_value")):
            window = windows[kind]
            if not isinstance(window, Mapping):
                raise ValueError("Fixed normalization plateau evidence is invalid")
            indices = _raw_index_list(window.get("raw_indices", ()), self.raw_time.size)
            if indices.size == 0 or len(indices) != window.get("n_points") or window.get("mean") != p.get(key):
                raise ValueError(f"Fixed {kind} normalization reference is inconsistent")
        endpoints = p.get("reference_endpoints")
        if endpoints is not None:
            if not isinstance(endpoints, Mapping) or not all(k in endpoints for k in ("initial", "final")):
                raise ValueError("Fixed normalization endpoint evidence is incomplete")
            for kind in ("initial", "final"):
                _finite_scalar("normalization reference endpoint", endpoints[kind])
        return {"windows": windows, "endpoints": p.get("reference_endpoints"),
                "mode": p["mode"], "history_index": index}

    def normalize(
        self,
        *,
        mode: str = "progress",
        initial_value: float | None = None,
        final_value: float | None = None,
        initial_points: int = 1,
        final_points: int = 1,
    ) -> EcrDataset:
        """Normalize to progress (0 to 1) or remaining fraction (1 to 0).

        Endpoints default to means of the requested leading and trailing windows.
        Values are not clipped, so experimental overshoot remains observable.
        """

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"progress", "remaining"}:
            raise ValueError("mode must be 'progress' or 'remaining'")
        n_initial = _window_size("initial_points", initial_points, self.processed_signal.size)
        n_final = _window_size("final_points", final_points, self.processed_signal.size)
        initial = (
            float(np.mean(self.processed_signal[:n_initial]))
            if initial_value is None
            else _finite_scalar("initial_value", initial_value)
        )
        final = (
            float(np.mean(self.processed_signal[-n_final:]))
            if final_value is None
            else _finite_scalar("final_value", final_value)
        )
        scale = max(abs(initial), abs(final), 1.0)
        if abs(final - initial) <= np.finfo(np.float64).eps * scale:
            raise ValueError("initial and final normalization values must be different")
        progress = (self.processed_signal - initial) / (final - initial)
        normalized = progress if normalized_mode == "progress" else 1.0 - progress
        return self._with_operation(
            processed_signal=normalized,
            operation="normalize",
            parameters={
                "mode": normalized_mode,
                "initial_value": initial,
                "final_value": final,
                "initial_points": n_initial,
                "final_points": n_final,
                "clipped": False,
            },
            metadata_updates={
                "signal_type": "normalized",
                "signal_unit": "1",
                **self._current_unit_key_update("dimensionless"),
                "normalization_mode": normalized_mode,
            },
        )

    def metadata_as_dict(self) -> dict[str, object]:
        """Return a mutable, JSON-compatible copy of metadata."""

        return _thaw_json(self.metadata)

    def _current_unit_key_update(self, key: str) -> dict[str, str]:
        return {"signal_unit_key": key} if "signal_unit_key" in self.metadata else {}

    def to_metadata_json(
        self,
        *,
        indent: int | None = None,
        sort_keys: bool = True,
    ) -> str:
        """Serialize metadata only; numerical arrays remain separate project data."""

        return json.dumps(
            self.metadata_as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=sort_keys,
        )

    @staticmethod
    def metadata_from_json(payload: str) -> Mapping[str, object]:
        """Deserialize and freeze metadata produced by :meth:`to_metadata_json`."""

        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid metadata JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("metadata JSON root must be an object")
        return _freeze_mapping(value)

    def _with_operation(
        self,
        *,
        operation: str,
        parameters: Mapping[str, object],
        processed_time: ArrayLike | None = None,
        processed_signal: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        metadata_updates: Mapping[str, object] | None = None,
    ) -> EcrDataset:
        metadata = self.metadata_as_dict()
        if metadata_updates:
            metadata.update(metadata_updates)
        record = ProcessingRecord.create(operation, parameters)
        return EcrDataset(
            raw_time=self.raw_time,
            raw_signal=self.raw_signal,
            processed_time=(
                self.processed_time if processed_time is None else processed_time
            ),
            processed_signal=(
                self.processed_signal if processed_signal is None else processed_signal
            ),
            mask=self.mask if mask is None else mask,
            metadata=metadata,
            history=(*self.history, record),
        )


def _read_only_float_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    return np.frombuffer(result.tobytes(), dtype=np.float64)


def _boolean_mask(values: ArrayLike, expected_size: int) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError("mask must be one-dimensional and aligned with raw arrays")
    if array.dtype.kind != "b":
        raise TypeError("mask entries must be boolean")
    return np.asarray(array, dtype=np.bool_).copy()


def _require_strictly_increasing(name: str, values: NDArray[np.float64]) -> None:
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _canonical_time_unit(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("time unit must be a string")
    normalized = value.strip().lower()
    normalized = _TIME_ALIASES.get(normalized, normalized)
    if normalized not in _TIME_FACTORS_TO_SECONDS:
        allowed = ", ".join(_TIME_FACTORS_TO_SECONDS)
        raise ValueError(f"unsupported time unit {value!r}; expected one of: {allowed}")
    return normalized


def _window_size(name: str, value: int, available: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0 or result > available:
        raise ValueError(f"{name} must be between 1 and {available}")
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _raw_index_list(values: object, available: int) -> NDArray[np.int64]:
    if isinstance(values, (str, bytes)):
        raise TypeError("raw_indices must be a sequence of integers")
    try:
        supplied = list(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("raw_indices must be a sequence of integers") from exc
    if not supplied:
        raise ValueError("raw_indices must contain at least one index")
    result: list[int] = []
    for value in supplied:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("raw_indices entries must be integers")
        index = int(value)
        if index < 0 or index >= available:
            raise ValueError(f"raw index {index} is outside 0..{available - 1}")
        result.append(index)
    return np.asarray(sorted(set(result)), dtype=np.int64)


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(value, path="metadata")
    if not isinstance(frozen, Mapping):  # pragma: no cover - guaranteed by input
        raise TypeError("metadata must be a mapping")
    return frozen


def _freeze_json(value: object, *, path: str) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"{path} numeric values must be finite")
        return number
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
