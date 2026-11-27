"""Serializable, deterministic orchestration of ECR preprocessing steps."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from pyecr.io.dataset import EcrDataset, ProcessingRecord

__all__ = [
    "DownstreamStaleEvent",
    "PREPROCESSING_OPERATIONS",
    "PreprocessConfig",
    "PreprocessingPipeline",
    "ProcessingStep",
]


PIPELINE_SCHEMA_VERSION = 1
PREPROCESSING_OPERATIONS: tuple[str, ...] = (
    "convert_time_unit",
    "resistance_to_conductance",
    "conductance_to_conductivity",
    "set_manual_t0",
    "crop",
    "mask_points",
    "choose_initial_stability_window",
    "choose_final_stability_window",
    "normalize",
    "normalize_signed_dct",
    "drift_correction",
    "outlier_mask",
    "unmask_points",
    "adaptive_downsample",
)


@dataclass(frozen=True, slots=True)
class ProcessingStep:
    """One user-requested operation suitable for project-file persistence."""

    operation: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = PIPELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        operation = str(self.operation).strip()
        if operation not in PREPROCESSING_OPERATIONS:
            raise ValueError(f"unsupported preprocessing operation: {operation!r}")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        timestamp = _timezone_timestamp(self.timestamp)
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "schema_version", PIPELINE_SCHEMA_VERSION)

    @classmethod
    def create(
        cls,
        operation: str,
        parameters: Mapping[str, object] | None = None,
        *,
        timestamp: str | None = None,
    ) -> ProcessingStep:
        values = dict(parameters or {})
        if operation == "normalize":
            # Version new requests only. Deserialized older recipes retain their
            # original records so project integrity checks remain exact.
            values.setdefault("reference_policy", "fixed-plateaus/1")
        return cls(
            operation=operation,
            parameters=values,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "parameters": _thaw_json(self.parameters),
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProcessingStep:
        if not isinstance(value, Mapping):
            raise TypeError("processing step must be a mapping")
        missing = {
            "operation",
            "parameters",
            "timestamp",
        }.difference(value)
        if missing:
            raise ValueError(f"missing processing-step field: {sorted(missing)[0]}")
        parameters = value["parameters"]
        if not isinstance(parameters, Mapping):
            raise TypeError("processing-step parameters must be a mapping")
        schema_version = value.get("schema_version", PIPELINE_SCHEMA_VERSION)
        if isinstance(schema_version, bool) or not isinstance(schema_version, Integral):
            raise TypeError("processing-step schema_version must be an integer")
        return cls(
            operation=str(value["operation"]),
            parameters=parameters,
            timestamp=str(value["timestamp"]),
            schema_version=int(schema_version),
        )


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """An ordered preprocessing recipe that can be replayed from source data."""

    steps: tuple[ProcessingStep, ...] = ()
    schema_version: int = PIPELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        steps = tuple(
            step if isinstance(step, ProcessingStep) else ProcessingStep.from_dict(step)
            for step in self.steps
        )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "schema_version", PIPELINE_SCHEMA_VERSION)

    def append(self, step: ProcessingStep) -> PreprocessConfig:
        if not isinstance(step, ProcessingStep):
            raise TypeError("step must be a ProcessingStep")
        return PreprocessConfig((*self.steps, step), self.schema_version)

    def without_last(self) -> PreprocessConfig:
        if not self.steps:
            raise RuntimeError("cannot undo an empty preprocessing pipeline")
        return PreprocessConfig(self.steps[:-1], self.schema_version)

    def replay(self, source: EcrDataset) -> EcrDataset:
        if not isinstance(source, EcrDataset):
            raise TypeError("source must be an EcrDataset")
        # An unmask cancels earlier explicit masks only. Replaying the recipe
        # restores transformed values; crop/downsample exclusions stay in force.
        from pyecr.preprocessing.optional import _indices
        suppressed: set[int] = set()
        overrides: dict[int, set[int]] = {}
        for index in range(len(self.steps) - 1, -1, -1):
            step = self.steps[index]
            if step.operation == "unmask_points":
                suppressed.update(_indices(step.parameters["raw_indices"], source.raw_time.size))
            elif step.operation in ("mask_points", "outlier_mask"):
                overrides[index] = set(suppressed)
        result = source
        for index, step in enumerate(self.steps):
            result = _apply_processing_step(
                result, step, suppressed=overrides.get(index),
                recipe=PreprocessConfig(self.steps[:index]),
            )
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreprocessConfig:
        if not isinstance(value, Mapping):
            raise TypeError("preprocess config must be a mapping")
        schema_version = value.get("schema_version", PIPELINE_SCHEMA_VERSION)
        if isinstance(schema_version, bool) or not isinstance(schema_version, Integral):
            raise TypeError("preprocess schema_version must be an integer")
        raw_steps = value.get("steps", ())
        if isinstance(raw_steps, (str, bytes)) or not isinstance(raw_steps, Sequence):
            raise TypeError("preprocess steps must be an array")
        return cls(
            steps=tuple(ProcessingStep.from_dict(step) for step in raw_steps),
            schema_version=int(schema_version),
        )

    @classmethod
    def from_json(cls, payload: str) -> PreprocessConfig:
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid preprocessing JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("preprocessing JSON root must be an object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class DownstreamStaleEvent:
    """Notification that derived model/fit results must no longer be trusted."""

    revision: int
    action: str
    operation: str | None
    timestamp: str
    downstream_stale: bool = True
    schema_version: int = PIPELINE_SCHEMA_VERSION


StaleListener = Callable[[DownstreamStaleEvent], None]


class PreprocessingPipeline:
    """Stateful Apply/Undo/Reset facade backed by deterministic full replay."""

    def __init__(
        self,
        source: EcrDataset,
        config: PreprocessConfig | None = None,
    ) -> None:
        if not isinstance(source, EcrDataset):
            raise TypeError("source must be an EcrDataset")
        self._source = source
        self._config = config or PreprocessConfig()
        self._dataset = self._config.replay(source)
        self._listeners: list[StaleListener] = []
        # A restored recipe already represents committed preprocessing work.
        # Starting at its step count keeps subsequent Apply/Undo revisions
        # monotonic after a project is reopened.
        self._revision = len(self._config.steps)

    @property
    def source(self) -> EcrDataset:
        return self._source

    @property
    def config(self) -> PreprocessConfig:
        return self._config

    @property
    def dataset(self) -> EcrDataset:
        return self._dataset

    @property
    def revision(self) -> int:
        return self._revision

    def subscribe(self, listener: StaleListener) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: StaleListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def apply(self, step: ProcessingStep) -> EcrDataset:
        candidate_config = self._config.append(step)
        candidate_dataset = candidate_config.replay(self._source)
        self._config = candidate_config
        self._dataset = candidate_dataset
        self._emit("apply", step.operation)
        return candidate_dataset

    def undo(self) -> EcrDataset:
        removed = self._config.steps[-1] if self._config.steps else None
        candidate_config = self._config.without_last()
        candidate_dataset = candidate_config.replay(self._source)
        self._config = candidate_config
        self._dataset = candidate_dataset
        self._emit("undo", removed.operation if removed is not None else None)
        return candidate_dataset

    def reset(self) -> EcrDataset:
        self._config = PreprocessConfig()
        self._dataset = self._source
        self._emit("reset", None)
        return self._dataset

    def _emit(self, action: str, operation: str | None) -> None:
        self._revision += 1
        event = DownstreamStaleEvent(
            revision=self._revision,
            action=action,
            operation=operation,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        for listener in tuple(self._listeners):
            listener(event)


def _apply_processing_step(dataset: EcrDataset, step: ProcessingStep, *, suppressed: set[int] | None = None, recipe: PreprocessConfig | None = None) -> EcrDataset:
    parameters = step.parameters
    operation = step.operation
    if operation == "convert_time_unit":
        result = dataset.convert_time_unit(
            str(parameters.get("to_unit", "s")),
            from_unit=_optional_string(parameters, "from_unit"),
        )
    elif operation == "resistance_to_conductance":
        result = dataset.resistance_to_conductance(
            resistance_unit=_optional_string(parameters, "resistance_unit")
        )
    elif operation == "conductance_to_conductivity":
        result = dataset.conductance_to_conductivity(
            path_length_cm=_required_float(parameters, "path_length_cm"),
            cross_section_area_cm2=_required_float(
                parameters, "cross_section_area_cm2"
            ),
            # Old saved steps retain their numerical interpretation. New GUI
            # requests explicitly select unit-aware conversion before L/A.
            unit_policy=str(parameters.get('unit_policy','legacy_assume_S')),
        )
    elif operation == "set_manual_t0":
        result = dataset.set_manual_t0(_required_float(parameters, "t0"))
    elif operation == "crop":
        result = dataset.crop(
            start_time=_required_float(parameters, "start_time"),
            end_time=_required_float(parameters, "end_time"),
        )
    elif operation == "mask_points":
        if "raw_indices" not in parameters:
            raise ValueError("mask_points requires raw_indices")
        if suppressed:
            from pyecr.preprocessing.optional import _indices
            import numpy as np
            original = _indices(parameters["raw_indices"], dataset.raw_time.size)
            effective = sorted(set(original) - suppressed)
            keep = ~np.isin(np.flatnonzero(dataset.mask), effective)
            mask = np.array(dataset.mask, copy=True)
            mask[effective] = False
            result = dataset._with_operation(
                operation=operation, parameters={"raw_indices": original, "applied_raw_indices": effective},
                processed_time=dataset.processed_time[keep], processed_signal=dataset.processed_signal[keep], mask=mask,
            )
        else:
            result = dataset.mask_points(parameters["raw_indices"])
    elif operation == "choose_initial_stability_window":
        result = dataset.choose_initial_stability_window(
            start_time=_required_float(parameters, "start_time"),
            end_time=_required_float(parameters, "end_time"),
        )
    elif operation == "choose_final_stability_window":
        result = dataset.choose_final_stability_window(
            start_time=_required_float(parameters, "start_time"),
            end_time=_required_float(parameters, "end_time"),
        )
    elif operation == "normalize":
        policy = parameters.get("reference_policy")
        if policy not in (None, "fixed-plateaus/1"):
            raise ValueError("unknown normalization reference policy")
        result = dataset.normalize_from_stability_windows(
            mode=str(parameters.get("mode", "progress")),
            record_reference=policy == "fixed-plateaus/1",
        )
    elif operation == "normalize_signed_dct":
        from pyecr.preprocessing.signed_normalization import _apply_signed_step
        result = _apply_signed_step(dataset, step, recipe=recipe)
    elif operation in ("drift_correction", "outlier_mask", "unmask_points", "adaptive_downsample"):
        from pyecr.preprocessing.optional import apply_optional
        result = apply_optional(dataset, step, suppressed=suppressed)
    else:  # pragma: no cover - ProcessingStep rejects this before dispatch
        raise ValueError(f"unsupported preprocessing operation: {operation!r}")
    return _stamp_last_record(result, step)


def _stamp_last_record(dataset: EcrDataset, step: ProcessingStep) -> EcrDataset:
    actual = dataset.history[-1]
    record = ProcessingRecord.create(
        actual.operation,
        actual.parameters,
        timestamp=step.timestamp,
        schema_version=step.schema_version,
    )
    return EcrDataset(
        raw_time=dataset.raw_time,
        raw_signal=dataset.raw_signal,
        processed_time=dataset.processed_time,
        processed_signal=dataset.processed_signal,
        mask=dataset.mask,
        metadata=dataset.metadata,
        history=(*dataset.history[:-1], record),
    )


def _required_float(parameters: Mapping[str, object], name: str) -> float:
    if name not in parameters:
        raise ValueError(f"{name} is required")
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_string(parameters: Mapping[str, object], name: str) -> str | None:
    value = parameters.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _timezone_timestamp(value: str) -> str:
    timestamp = str(value)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO 8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return timestamp


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(value, path="parameters")
    assert isinstance(frozen, Mapping)
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
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{path} numeric values must be finite")
        return result
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
