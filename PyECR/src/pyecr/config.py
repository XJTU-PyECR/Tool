"""Immutable configuration objects for PyECR.

This module stores validated inputs and numerical settings only.  It does not
evaluate scientific models or calculate diffusion solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import ClassVar, Mapping

from pyecr.models import ModelID
from pyecr.units import DEFAULT_UNITS

__all__ = [
    "AnalysisConfig",
    "BoundaryConfig",
    "DataConfig",
    "ExperimentMetadata",
    "FitConfig",
    "ModelConfig",
    "SampleGeometry",
    "SolverConfig",
]


def _require_canonical_unit(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}; received {actual!r}")


def _require_positive_finite(name: str, value: Real) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")


def _require_mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _freeze_json(value: object, *, path: str) -> object:
    """Return an immutable representation restricted to JSON-compatible data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return int(value) if isinstance(value, int) else numeric
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be nonempty strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains a value that is not JSON-compatible")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _numeric_mapping(
    name: str,
    values: Mapping[str, float],
) -> Mapping[str, float]:
    supplied = _require_mapping(values, name)
    normalized: dict[str, float] = {}
    for parameter, value in supplied.items():
        if not isinstance(parameter, str) or not parameter:
            raise ValueError(f"{name} keys must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{parameter!r}] must be a real number")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{name}[{parameter!r}] must be finite")
        normalized[parameter] = numeric
    return MappingProxyType(normalized)


def _bounds_mapping(
    values: Mapping[str, tuple[float, float]],
) -> Mapping[str, tuple[float, float]]:
    supplied = _require_mapping(values, "parameter_bounds")
    normalized: dict[str, tuple[float, float]] = {}
    for parameter, bounds in supplied.items():
        if not isinstance(parameter, str) or not parameter:
            raise ValueError("parameter_bounds keys must be nonempty strings")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"parameter_bounds[{parameter!r}] must contain lower and upper"
            )
        lower, upper = bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, Real)
            or not isinstance(upper, Real)
        ):
            raise TypeError(
                f"parameter_bounds[{parameter!r}] must contain real numbers"
            )
        pair = (float(lower), float(upper))
        if not all(isfinite(value) for value in pair):
            raise ValueError(
                f"parameter_bounds[{parameter!r}] must contain finite values"
            )
        if pair[0] >= pair[1]:
            raise ValueError(
                f"parameter_bounds[{parameter!r}] lower bound must be below upper"
            )
        normalized[parameter] = pair
    return MappingProxyType(normalized)


def _parameter_names(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of parameter names")
    normalized = tuple(values)
    if not all(isinstance(value, str) and value for value in normalized):
        raise ValueError(f"{name} must contain nonempty strings")
    return normalized


def _model_id_value(value: object) -> object:
    return value.value if isinstance(value, ModelID) else value


def _model_id_from_dict(value: object) -> ModelID:
    try:
        return ModelID(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid model_id: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class SampleGeometry:
    """Complete three-axis specimen geometry.

    ``full_dims_cm`` is ordered as the specimen's x, y, and z full dimensions.
    ``active_dimensions`` uses the same order and indicates which dimensions
    participate in the selected transport model.  Inactive dimensions must
    still have their measured, positive full size recorded.
    """

    full_dims_cm: tuple[float, float, float]
    active_dimensions: tuple[bool, bool, bool] = field(
        default=(True, False, False),
        metadata={
            "meaning": "Default one-dimensional transport along the x dimension."
        },
    )
    length_unit: str = field(
        default=DEFAULT_UNITS.length,
        metadata={"meaning": "All recorded specimen dimensions are in centimetres."},
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "full_dims_cm", tuple(self.full_dims_cm))
        object.__setattr__(self, "active_dimensions", tuple(self.active_dimensions))

    def validate(self) -> None:
        """Validate complete dimensions, active axes, and the length unit."""

        if len(self.full_dims_cm) != 3:
            raise ValueError("full_dims_cm must contain all three x, y, z dimensions")
        if len(self.active_dimensions) != 3:
            raise ValueError(
                "active_dimensions must contain one flag for each x, y, z dimension"
            )

        for axis, dimension in zip("xyz", self.full_dims_cm, strict=True):
            _require_positive_finite(f"full_dims_cm[{axis}]", dimension)

        if not all(isinstance(flag, bool) for flag in self.active_dimensions):
            raise TypeError("active_dimensions entries must be bool values")
        if not any(self.active_dimensions):
            raise ValueError("at least one active dimension is required")

        _require_canonical_unit(
            "SampleGeometry.length_unit", self.length_unit, DEFAULT_UNITS.length
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "full_dims_cm": list(self.full_dims_cm),
            "active_dimensions": list(self.active_dimensions),
            "length_unit": self.length_unit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SampleGeometry:
        supplied = _require_mapping(value, "SampleGeometry")
        return cls(
            full_dims_cm=tuple(supplied["full_dims_cm"]),  # type: ignore[arg-type]
            active_dimensions=tuple(  # type: ignore[arg-type]
                supplied["active_dimensions"]
            ),
            length_unit=str(supplied.get("length_unit", DEFAULT_UNITS.length)),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Electrical input-data semantics without boundary-history ownership.

    Gas-boundary settings belong to :class:`BoundaryConfig` within an
    :class:`AnalysisConfig`; they are not inferred from the ECR signal columns.
    """

    VALID_SIGNAL_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"conductivity", "resistance", "normalized"}
    )

    time_column: str = field(
        default="time",
        metadata={"meaning": "Column containing elapsed experimental time."},
    )
    signal_column: str = field(
        default="signal",
        metadata={"meaning": "Column containing the measured electrical response."},
    )
    signal_type: str = field(
        default="conductivity",
        metadata={"meaning": "The measured response is electrical conductivity."},
    )
    time_unit: str = field(
        default=DEFAULT_UNITS.time,
        metadata={"meaning": "Elapsed experimental time is expressed in seconds."},
    )

    def validate(self) -> None:
        """Validate column semantics, signal type, and time unit."""

        if not self.time_column.strip():
            raise ValueError("time_column must not be empty")
        if not self.signal_column.strip():
            raise ValueError("signal_column must not be empty")
        if self.time_column == self.signal_column:
            raise ValueError("time_column and signal_column must be different")
        if self.signal_type not in self.VALID_SIGNAL_TYPES:
            allowed = ", ".join(sorted(self.VALID_SIGNAL_TYPES))
            raise ValueError(
                f"unknown signal_type {self.signal_type!r}; expected one of: {allowed}"
            )

        _require_canonical_unit(
            "DataConfig.time_unit", self.time_unit, DEFAULT_UNITS.time
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "time_column": self.time_column,
            "signal_column": self.signal_column,
            "signal_type": self.signal_type,
            "time_unit": self.time_unit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DataConfig:
        supplied = _require_mapping(value, "DataConfig")
        return cls(
            time_column=str(supplied.get("time_column", "time")),
            signal_column=str(supplied.get("signal_column", "signal")),
            signal_type=str(supplied.get("signal_type", "conductivity")),
            time_unit=str(supplied.get("time_unit", DEFAULT_UNITS.time)),
        )


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    """Serializable gas-boundary condition composed with a material model."""

    VALID_MODES: ClassVar[frozenset[str]] = frozenset(
        {"ideal_step", "exponential_flush", "measured_history"}
    )

    mode: str = field(
        default="ideal_step",
        metadata={"meaning": "Ideal, CISTR exponential, or measured gas history."},
    )
    flush_time_s: float | None = field(
        default=None,
        metadata={"meaning": "CISTR residence/flush time in seconds."},
    )
    measured_time_s: tuple[float, ...] = field(
        default=(),
        metadata={"meaning": "Strictly increasing measured-boundary time axis in s."},
    )
    measured_values: tuple[float, ...] = field(
        default=(),
        metadata={"meaning": "Measured pO2 or pH2O values aligned with time."},
    )
    measured_signal_type: str | None = field(
        default=None,
        metadata={"meaning": "Measured boundary variable: pO2 or pH2O."},
    )
    measured_initial_value: float | None = field(
        default=None,
        metadata={"meaning": "Declared gas value before the applied step."},
    )
    measured_final_value: float | None = field(
        default=None,
        metadata={"meaning": "Declared final gas value after the applied step."},
    )
    measured_value_unit: str = field(
        default="unspecified",
        metadata={"meaning": "Unit label retained with the measured gas history."},
    )
    resample_time_step_s: float | None = field(
        default=None,
        metadata={"meaning": "Optional explicit convolution grid step in seconds."},
    )
    max_internal_points: int = field(
        default=200_000,
        metadata={"meaning": "Safety limit for measured-boundary convolution grid."},
    )

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode == "measured":
            mode = "measured_history"
        signal_type = self.measured_signal_type
        if signal_type is not None:
            aliases = {
                "po2": "pO2",
                "p_o2": "pO2",
                "ph2o": "pH2O",
                "p_h2o": "pH2O",
            }
            signal_type = aliases.get(str(signal_type).strip().lower(), str(signal_type))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "measured_time_s", tuple(self.measured_time_s))
        object.__setattr__(self, "measured_values", tuple(self.measured_values))
        object.__setattr__(self, "measured_signal_type", signal_type)

    def validate(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"unknown boundary mode: {self.mode!r}")
        if (
            isinstance(self.max_internal_points, bool)
            or not isinstance(self.max_internal_points, int)
            or self.max_internal_points <= 0
        ):
            raise ValueError("max_internal_points must be a positive integer")
        if self.resample_time_step_s is not None:
            _require_positive_finite(
                "resample_time_step_s", self.resample_time_step_s
            )

        has_measured_payload = bool(
            self.measured_time_s
            or self.measured_values
            or self.measured_signal_type is not None
            or self.measured_initial_value is not None
            or self.measured_final_value is not None
        )
        if self.mode == "ideal_step":
            if self.flush_time_s is not None or has_measured_payload:
                raise ValueError("ideal_step must not contain flush or measured options")
            return
        if self.mode == "exponential_flush":
            if self.flush_time_s is None:
                raise ValueError("exponential_flush requires flush_time_s")
            _require_positive_finite("flush_time_s", self.flush_time_s)
            if has_measured_payload:
                raise ValueError(
                    "exponential_flush must not contain measured-history options"
                )
            return

        if self.flush_time_s is not None:
            raise ValueError("measured_history must not contain flush_time_s")
        if len(self.measured_time_s) < 2:
            raise ValueError("measured_history requires at least two time values")
        if len(self.measured_values) != len(self.measured_time_s):
            raise ValueError("measured boundary time and values must have equal length")
        times: list[float] = []
        values: list[float] = []
        for index, value in enumerate(self.measured_time_s):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"measured_time_s[{index}] must be a real number")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError("measured boundary times must be finite")
            times.append(numeric)
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("measured boundary time must be strictly increasing")
        for index, value in enumerate(self.measured_values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"measured_values[{index}] must be a real number")
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0.0:
                raise ValueError("measured boundary values must be finite and nonnegative")
            values.append(numeric)
        if self.measured_signal_type not in {"pO2", "pH2O"}:
            raise ValueError("measured_signal_type must be pO2 or pH2O")
        if self.measured_initial_value is None or self.measured_final_value is None:
            raise ValueError(
                "measured_history requires measured_initial_value and measured_final_value"
            )
        for name, value in (
            ("measured_initial_value", self.measured_initial_value),
            ("measured_final_value", self.measured_final_value),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.measured_initial_value == self.measured_final_value:
            raise ValueError("measured initial and final values must differ")
        if not self.measured_value_unit.strip():
            raise ValueError("measured_value_unit must not be empty")
        boundary_span = abs(
            float(self.measured_final_value) - float(self.measured_initial_value)
        )
        if (
            abs(values[0] - float(self.measured_initial_value))
            > 1.0e-10 * boundary_span
        ):
            raise ValueError(
                "measured boundary must begin at measured_initial_value"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "flush_time_s": self.flush_time_s,
            "measured_time_s": list(self.measured_time_s),
            "measured_values": list(self.measured_values),
            "measured_signal_type": self.measured_signal_type,
            "measured_initial_value": self.measured_initial_value,
            "measured_final_value": self.measured_final_value,
            "measured_value_unit": self.measured_value_unit,
            "resample_time_step_s": self.resample_time_step_s,
            "max_internal_points": self.max_internal_points,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BoundaryConfig:
        supplied = _require_mapping(value, "BoundaryConfig")
        return cls(
            mode=str(supplied.get("mode", "ideal_step")),
            flush_time_s=(
                None
                if supplied.get("flush_time_s") is None
                else float(supplied["flush_time_s"])
            ),
            measured_time_s=tuple(supplied.get("measured_time_s", ())),  # type: ignore[arg-type]
            measured_values=tuple(supplied.get("measured_values", ())),  # type: ignore[arg-type]
            measured_signal_type=(
                None
                if supplied.get("measured_signal_type") is None
                else str(supplied["measured_signal_type"])
            ),
            measured_initial_value=(
                None
                if supplied.get("measured_initial_value") is None
                else float(supplied["measured_initial_value"])
            ),
            measured_final_value=(
                None
                if supplied.get("measured_final_value") is None
                else float(supplied["measured_final_value"])
            ),
            measured_value_unit=str(
                supplied.get("measured_value_unit", "unspecified")
            ),
            resample_time_step_s=(
                None
                if supplied.get("resample_time_step_s") is None
                else float(supplied["resample_time_step_s"])
            ),
            max_internal_points=int(supplied.get("max_internal_points", 200_000)),
        )


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Numerical controls and canonical units shared by model solvers."""

    VALID_SURFACE_MODES: ClassVar[frozenset[str]] = frozenset({"instant", "finite", "surface_controlled"})

    surface_mode: str = field(
        default="instant",
        metadata={
            "meaning": "The specimen surface is assumed to equilibrate instantaneously."
        },
    )
    relative_tolerance: float = field(
        default=1.0e-8,
        metadata={"meaning": "Relative numerical convergence tolerance."},
    )
    absolute_tolerance: float = field(
        default=1.0e-12,
        metadata={"meaning": "Absolute numerical convergence tolerance."},
    )
    max_terms: int = field(
        default=200,
        metadata={"meaning": "Maximum number of terms allowed in a numerical series."},
    )
    time_unit: str = field(
        default=DEFAULT_UNITS.time,
        metadata={"meaning": "Solver time inputs are expressed in seconds."},
    )
    length_unit: str = field(
        default=DEFAULT_UNITS.length,
        metadata={"meaning": "Solver length inputs are expressed in centimetres."},
    )
    diffusion_coefficient_unit: str = field(
        default=DEFAULT_UNITS.diffusion_coefficient,
        metadata={
            "meaning": "Diffusion coefficients are expressed in square centimetres per second."
        },
    )
    surface_exchange_coefficient_unit: str = field(
        default=DEFAULT_UNITS.surface_exchange_coefficient,
        metadata={
            "meaning": "Surface exchange coefficients are expressed in centimetres per second."
        },
    )

    def validate(self) -> None:
        """Validate numerical controls, surface mode, and canonical units."""

        if self.surface_mode not in self.VALID_SURFACE_MODES:
            allowed = ", ".join(sorted(self.VALID_SURFACE_MODES))
            raise ValueError(
                f"unknown surface_mode {self.surface_mode!r}; expected: {allowed}"
            )
        _require_positive_finite("relative_tolerance", self.relative_tolerance)
        _require_positive_finite("absolute_tolerance", self.absolute_tolerance)
        if isinstance(self.max_terms, bool) or not isinstance(self.max_terms, int):
            raise TypeError("max_terms must be an integer")
        if self.max_terms <= 0:
            raise ValueError("max_terms must be greater than zero")

        _require_canonical_unit("SolverConfig.time_unit", self.time_unit, DEFAULT_UNITS.time)
        _require_canonical_unit(
            "SolverConfig.length_unit", self.length_unit, DEFAULT_UNITS.length
        )
        _require_canonical_unit(
            "SolverConfig.diffusion_coefficient_unit",
            self.diffusion_coefficient_unit,
            DEFAULT_UNITS.diffusion_coefficient,
        )
        _require_canonical_unit(
            "SolverConfig.surface_exchange_coefficient_unit",
            self.surface_exchange_coefficient_unit,
            DEFAULT_UNITS.surface_exchange_coefficient,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "surface_mode": self.surface_mode,
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "max_terms": self.max_terms,
            "time_unit": self.time_unit,
            "length_unit": self.length_unit,
            "diffusion_coefficient_unit": self.diffusion_coefficient_unit,
            "surface_exchange_coefficient_unit": (
                self.surface_exchange_coefficient_unit
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SolverConfig:
        supplied = _require_mapping(value, "SolverConfig")
        return cls(
            surface_mode=str(supplied.get("surface_mode", "instant")),
            relative_tolerance=float(supplied.get("relative_tolerance", 1.0e-8)),
            absolute_tolerance=float(supplied.get("absolute_tolerance", 1.0e-12)),
            max_terms=int(supplied.get("max_terms", 200)),
            time_unit=str(supplied.get("time_unit", DEFAULT_UNITS.time)),
            length_unit=str(supplied.get("length_unit", DEFAULT_UNITS.length)),
            diffusion_coefficient_unit=str(
                supplied.get(
                    "diffusion_coefficient_unit",
                    DEFAULT_UNITS.diffusion_coefficient,
                )
            ),
            surface_exchange_coefficient_unit=str(
                supplied.get(
                    "surface_exchange_coefficient_unit",
                    DEFAULT_UNITS.surface_exchange_coefficient,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Model selection and optimizer controls for one fit."""

    _INSTANT_MODELS: ClassVar[frozenset[ModelID]] = frozenset(
        {ModelID.M0_SINGLE_D, ModelID.M2_DUAL_HO_D}
    )
    _FINITE_MODELS: ClassVar[frozenset[ModelID]] = frozenset(
        {ModelID.M1_SINGLE_DK, ModelID.M3_DUAL_HO_DK}
    )
    VALID_LOSSES: ClassVar[frozenset[str]] = frozenset(
        {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    )

    model_id: ModelID = field(
        default=ModelID.M0_SINGLE_D,
        metadata={
            "meaning": "Default fit uses one transport process with diffusion only."
        },
    )
    max_nfev: int = field(
        default=5_000,
        metadata={"meaning": "Maximum optimizer model evaluations for one fit."},
    )
    loss: str = field(
        default="linear",
        metadata={"meaning": "Residuals receive ordinary least-squares weighting."},
    )

    def validate(self, solver: SolverConfig | None = None) -> None:
        """Validate the model ID, optimizer controls, and optional solver pairing."""

        if not isinstance(self.model_id, ModelID):
            raise ValueError(f"invalid model_id: {self.model_id!r}")
        if self.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} and self.loss != "linear":
            raise ValueError("M7 supports linear loss only")
        if isinstance(self.max_nfev, bool) or not isinstance(self.max_nfev, int):
            raise TypeError("max_nfev must be an integer")
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be greater than zero")
        if self.loss not in self.VALID_LOSSES:
            allowed = ", ".join(sorted(self.VALID_LOSSES))
            raise ValueError(f"unknown loss {self.loss!r}; expected one of: {allowed}")

        if solver is None:
            return

        solver.validate()
        if self.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} and solver.surface_mode != "surface_controlled":
            raise ValueError("M7 requires surface_mode='surface_controlled'")
        if self.model_id not in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} and solver.surface_mode == "surface_controlled":
            raise ValueError("surface_controlled is supported only by M7")
        if self.model_id in self._INSTANT_MODELS and solver.surface_mode != "instant":
            raise ValueError(f"{self.model_id.value} requires surface_mode='instant'")
        if self.model_id in self._FINITE_MODELS and solver.surface_mode != "finite":
            raise ValueError(f"{self.model_id.value} requires surface_mode='finite'")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "model_id": _model_id_value(self.model_id),
            "max_nfev": self.max_nfev,
            "loss": self.loss,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FitConfig:
        supplied = _require_mapping(value, "FitConfig")
        return cls(
            model_id=_model_id_from_dict(
                supplied.get("model_id", ModelID.M0_SINGLE_D.value)
            ),
            max_nfev=int(supplied.get("max_nfev", 5_000)),
            loss=str(supplied.get("loss", "linear")),
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Serializable physical-model selection and parameter policy.

    This object contains no solver and no equations. Parameter names are
    checked against the selected model descriptor by :class:`ModelRegistry`.
    """

    VALID_SURFACE_MODES: ClassVar[frozenset[str]] = frozenset({"instant", "finite", "surface_controlled"})
    VALID_BOUNDARY_MODES: ClassVar[frozenset[str]] = frozenset(
        {"ideal_step", "exponential_flush", "measured_history"}
    )
    VALID_STEP_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"hydration", "dehydration", "custom"}
    )
    _INSTANT_MODELS: ClassVar[frozenset[ModelID]] = frozenset(
        {ModelID.M0_SINGLE_D, ModelID.M2_DUAL_HO_D}
    )
    _FINITE_MODELS: ClassVar[frozenset[ModelID]] = frozenset(
        {ModelID.M1_SINGLE_DK, ModelID.M3_DUAL_HO_DK}
    )

    model_id: ModelID = field(
        metadata={"meaning": "Stable identifier of the selected physical model."}
    )
    geometry: SampleGeometry = field(
        metadata={"meaning": "Complete specimen dimensions and active axes."}
    )
    surface_mode: str = field(
        default="instant",
        metadata={"meaning": "Surface transport is instantaneous or finite-rate."},
    )
    boundary_mode: str = field(
        default="ideal_step",
        metadata={"meaning": "Gas boundary is ideal, exponential-flush, or measured."},
    )
    step_type: str = field(
        default="hydration",
        metadata={"meaning": "Hydration, dehydration, or explicit custom H/O signs."},
    )
    sign_H: float | None = field(
        default=None,
        metadata={"meaning": "Custom signed H contribution, unused for standard steps."},
    )
    sign_O: float | None = field(
        default=None,
        metadata={"meaning": "Custom signed O contribution, unused for standard steps."},
    )
    initial_values: Mapping[str, float] = field(
        default_factory=dict,
        metadata={"meaning": "Optional physical-space initial parameter values."},
    )
    parameter_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=dict,
        metadata={"meaning": "Optional physical-space lower and upper parameter bounds."},
    )
    free_parameters: tuple[str, ...] = field(
        default=(),
        metadata={"meaning": "Parameters designated for optimization."},
    )
    fixed_parameters: Mapping[str, float] = field(
        default_factory=dict,
        metadata={"meaning": "Parameters held at explicit physical values."},
    )

    def __post_init__(self) -> None:
        boundary_mode = str(self.boundary_mode).strip().lower()
        if boundary_mode == "measured":
            boundary_mode = "measured_history"
        step_type = getattr(self.step_type, "value", self.step_type)
        if isinstance(step_type, str):
            step_type = step_type.strip().lower()
        object.__setattr__(self, "step_type", step_type)
        object.__setattr__(self, "boundary_mode", boundary_mode)
        object.__setattr__(
            self,
            "initial_values",
            _numeric_mapping("initial_values", self.initial_values),
        )
        object.__setattr__(
            self,
            "parameter_bounds",
            _bounds_mapping(self.parameter_bounds),
        )
        object.__setattr__(
            self,
            "free_parameters",
            _parameter_names("free_parameters", self.free_parameters),
        )
        object.__setattr__(
            self,
            "fixed_parameters",
            _numeric_mapping("fixed_parameters", self.fixed_parameters),
        )

    def validate(self) -> None:
        """Validate generic geometry, boundary, step, and parameter invariants."""

        if not isinstance(self.model_id, ModelID):
            raise ValueError(f"invalid model_id: {self.model_id!r}")
        if not isinstance(self.geometry, SampleGeometry):
            raise TypeError("geometry must be a SampleGeometry")
        self.geometry.validate()
        if self.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}:
            if self.surface_mode != "surface_controlled" or self.boundary_mode != "ideal_step":
                raise ValueError("M7 supports surface_controlled and ideal_step only; other boundaries Unsupported")
            if self.step_type == "custom" and self.model_id is ModelID.M7_SINGLE_K:
                raise ValueError("M7 uses signed amplitude, not custom H/O signs")
            if self.model_id is ModelID.M8_DUAL_K and self.step_type == "custom" and (
                    self.sign_H not in (-1.,1.) or self.sign_O not in (-1.,1.)):
                raise ValueError("M8 custom input-process signs must each be +/-1")
        elif self.surface_mode == "surface_controlled":
            raise ValueError("surface_controlled is supported only by M7")
        if self.surface_mode not in self.VALID_SURFACE_MODES:
            raise ValueError(f"unknown surface_mode: {self.surface_mode!r}")
        if self.boundary_mode not in self.VALID_BOUNDARY_MODES:
            raise ValueError(f"unknown boundary_mode: {self.boundary_mode!r}")
        if self.step_type not in self.VALID_STEP_TYPES:
            raise ValueError(f"unknown step_type: {self.step_type!r}")

        if self.model_id in self._INSTANT_MODELS and self.surface_mode != "instant":
            raise ValueError(f"{self.model_id.value} requires surface_mode='instant'")
        if self.model_id in self._FINITE_MODELS and self.surface_mode != "finite":
            raise ValueError(f"{self.model_id.value} requires surface_mode='finite'")

        if self.step_type == "custom":
            if self.sign_H is None or self.sign_O is None:
                raise ValueError("custom step_type requires both sign_H and sign_O")
            for name, value in (("sign_H", self.sign_H), ("sign_O", self.sign_O)):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"{name} must be a real number")
                if not isfinite(float(value)):
                    raise ValueError(f"{name} must be finite")
        elif self.sign_H is not None or self.sign_O is not None:
            raise ValueError("sign_H/sign_O require step_type='custom'")

        if len(set(self.free_parameters)) != len(self.free_parameters):
            raise ValueError("free_parameters must not contain duplicates")
        conflicts = sorted(set(self.free_parameters) & set(self.fixed_parameters))
        if conflicts:
            raise ValueError(
                "parameters cannot be both free and fixed: " + ", ".join(conflicts)
            )

        for source_name, source in (
            ("initial_values", self.initial_values),
            ("fixed_parameters", self.fixed_parameters),
        ):
            for parameter, value in source.items():
                bounds = self.parameter_bounds.get(parameter)
                if bounds is not None and not bounds[0] <= value <= bounds[1]:
                    raise ValueError(
                        f"{source_name}[{parameter!r}] lies outside parameter_bounds"
                    )
        overlap = set(self.initial_values) & set(self.fixed_parameters)
        disagree = sorted(
            parameter
            for parameter in overlap
            if self.initial_values[parameter] != self.fixed_parameters[parameter]
        )
        if disagree:
            raise ValueError(
                "initial and fixed values disagree for: " + ", ".join(disagree)
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "model_id": _model_id_value(self.model_id),
            "geometry": self.geometry.to_dict(),
            "surface_mode": self.surface_mode,
            "boundary_mode": self.boundary_mode,
            "step_type": self.step_type,
            "sign_H": self.sign_H,
            "sign_O": self.sign_O,
            "initial_values": dict(self.initial_values),
            "parameter_bounds": {
                name: list(bounds) for name, bounds in self.parameter_bounds.items()
            },
            "free_parameters": list(self.free_parameters),
            "fixed_parameters": dict(self.fixed_parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelConfig:
        supplied = _require_mapping(value, "ModelConfig")
        geometry = supplied.get("geometry")
        if not isinstance(geometry, Mapping):
            raise TypeError("ModelConfig.geometry must be a mapping")
        return cls(
            model_id=_model_id_from_dict(supplied.get("model_id")),
            geometry=SampleGeometry.from_dict(geometry),
            surface_mode=str(supplied.get("surface_mode", "instant")),
            boundary_mode=str(supplied.get("boundary_mode", "ideal_step")),
            step_type=str(supplied.get("step_type", "hydration")),
            sign_H=supplied.get("sign_H"),  # type: ignore[arg-type]
            sign_O=supplied.get("sign_O"),  # type: ignore[arg-type]
            initial_values=supplied.get("initial_values", {}),  # type: ignore[arg-type]
            parameter_bounds=supplied.get("parameter_bounds", {}),  # type: ignore[arg-type]
            free_parameters=tuple(  # type: ignore[arg-type]
                supplied.get("free_parameters", ())
            ),
            fixed_parameters=supplied.get("fixed_parameters", {}),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    """Descriptive experimental context kept separate from numerical settings."""

    experiment_id: str = field(
        default="",
        metadata={"meaning": "No laboratory experiment identifier has been assigned."},
    )
    sample_id: str = field(
        default="",
        metadata={"meaning": "No specimen identifier has been recorded."},
    )
    temperature_k: float | None = field(
        default=None,
        metadata={"meaning": "Experimental temperature in kelvin is not yet recorded."},
    )
    atmosphere: str = field(
        default="",
        metadata={"meaning": "The surrounding gas atmosphere is not yet recorded."},
    )
    operator: str = field(
        default="",
        metadata={"meaning": "The experiment operator is not yet recorded."},
    )
    notes: str = field(
        default="",
        metadata={"meaning": "No additional experimental notes are present."},
    )

    def validate(self) -> None:
        """Validate optional physical metadata when it is supplied."""

        if self.temperature_k is not None:
            _require_positive_finite("temperature_k", self.temperature_k)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "experiment_id": self.experiment_id,
            "sample_id": self.sample_id,
            "temperature_k": self.temperature_k,
            "atmosphere": self.atmosphere,
            "operator": self.operator,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExperimentMetadata:
        supplied = _require_mapping(value, "ExperimentMetadata")
        temperature = supplied.get("temperature_k")
        return cls(
            experiment_id=str(supplied.get("experiment_id", "")),
            sample_id=str(supplied.get("sample_id", "")),
            temperature_k=None if temperature is None else float(temperature),
            atmosphere=str(supplied.get("atmosphere", "")),
            operator=str(supplied.get("operator", "")),
            notes=str(supplied.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Serializable composition of model, solver, fit, and model-specific options."""

    model: ModelConfig = field(
        metadata={"meaning": "Physical model, geometry, boundary, and parameters."}
    )
    solver: SolverConfig = field(
        metadata={"meaning": "Numerical tolerances and canonical solver units."}
    )
    fit: FitConfig = field(
        metadata={"meaning": "Optimizer controls paired with the selected model."}
    )
    model_specific: Mapping[str, object] = field(
        default_factory=dict,
        metadata={
            "meaning": "Optional JSON-compatible settings owned by a model-specific adapter."
        },
    )
    boundary: BoundaryConfig | None = field(
        default=None,
        metadata={"meaning": "Gas-boundary condition composed with the model."},
    )

    def __post_init__(self) -> None:
        frozen = _freeze_json(self.model_specific, path="model_specific")
        if not isinstance(frozen, Mapping):
            raise TypeError("model_specific must be a mapping")
        object.__setattr__(self, "model_specific", frozen)
        boundary = self.boundary
        if boundary is None and isinstance(self.model, ModelConfig):
            boundary = BoundaryConfig(mode=self.model.boundary_mode)
        object.__setattr__(self, "boundary", boundary)

    def validate(self) -> None:
        """Validate nested configs and require one consistent model/surface choice."""

        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        if not isinstance(self.solver, SolverConfig):
            raise TypeError("solver must be a SolverConfig")
        if not isinstance(self.fit, FitConfig):
            raise TypeError("fit must be a FitConfig")
        if not isinstance(self.boundary, BoundaryConfig):
            raise TypeError("boundary must be a BoundaryConfig")
        self.model.validate()
        self.solver.validate()
        self.boundary.validate()
        if self.boundary.mode != self.model.boundary_mode:
            raise ValueError("model.boundary_mode and boundary.mode must match")
        if (
            self.model.model_id in {ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK}
            and self.boundary.mode != "ideal_step"
        ):
            raise ValueError(
                "M2/M3 non-ideal boundary composition is unsupported because no "
                "strict dual-process boundary definition is implemented"
            )
        if self.fit.model_id != self.model.model_id:
            raise ValueError("model.model_id and fit.model_id must match")
        if self.solver.surface_mode != self.model.surface_mode:
            raise ValueError("model and solver surface_mode must match")
        self.fit.validate(self.solver)

    @property
    def model_id(self) -> ModelID:
        """Expose the selected model for services and presentation clients."""

        return self.model.model_id

    @property
    def parameter_units(self) -> Mapping[str, str]:
        """Return registry-derived units without duplicating unit strings."""

        from pyecr.models.registry import DEFAULT_MODEL_REGISTRY

        descriptor = DEFAULT_MODEL_REGISTRY.get(self.model.model_id)
        return MappingProxyType(
            {parameter.name: parameter.unit for parameter in descriptor.parameters}
        )

    @property
    def parameter_bounds(self) -> Mapping[str, tuple[float, float]]:
        """Return explicit bounds with descriptor defaults filling omissions."""

        from pyecr.models.registry import DEFAULT_MODEL_REGISTRY

        descriptor = DEFAULT_MODEL_REGISTRY.get(self.model.model_id)
        values: dict[str, tuple[float, float]] = {}
        for parameter in descriptor.parameters:
            bounds = self.model.parameter_bounds.get(
                parameter.name, parameter.default_bounds
            )
            if bounds is not None:
                values[parameter.name] = bounds
        return MappingProxyType(values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "model": self.model.to_dict(),
            "solver": self.solver.to_dict(),
            "fit": self.fit.to_dict(),
            "model_specific": _thaw_json(self.model_specific),
            "boundary": self.boundary.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AnalysisConfig:
        supplied = _require_mapping(value, "AnalysisConfig")
        model = supplied.get("model")
        solver = supplied.get("solver")
        fit = supplied.get("fit")
        boundary = supplied.get("boundary")
        if not isinstance(model, Mapping):
            raise TypeError("AnalysisConfig.model must be a mapping")
        if not isinstance(solver, Mapping):
            raise TypeError("AnalysisConfig.solver must be a mapping")
        if not isinstance(fit, Mapping):
            raise TypeError("AnalysisConfig.fit must be a mapping")
        model_config = ModelConfig.from_dict(model)
        return cls(
            model=model_config,
            solver=SolverConfig.from_dict(solver),
            fit=FitConfig.from_dict(fit),
            model_specific=supplied.get("model_specific", {}),  # type: ignore[arg-type]
            boundary=(
                BoundaryConfig(mode=model_config.boundary_mode)
                if boundary is None
                else BoundaryConfig.from_dict(
                    _require_mapping(boundary, "AnalysisConfig.boundary")
                )
            ),
        )
