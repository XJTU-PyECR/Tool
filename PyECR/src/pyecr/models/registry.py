"""Immutable metadata registry for PyECR model identifiers.

Descriptors contain names, units, and compatibility metadata only. They do not
evaluate a model and do not duplicate any ECR equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, TYPE_CHECKING

from pyecr.models.ids import ModelID
from pyecr.units import DEFAULT_UNITS

if TYPE_CHECKING:
    from pyecr.config import ModelConfig

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "ModelDescriptor",
    "ModelRegistry",
    "ParameterDescriptor",
]


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    """One named model parameter and its presentation/validation metadata."""

    name: str
    unit: str
    role: str
    default_initial: float | None = None
    default_bounds: tuple[float, float] | None = None
    default_free: bool = True
    constraint: str = "real"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be a nonempty string")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("parameter unit must be a nonempty string")
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("parameter role must be a nonempty string")
        if not isinstance(self.default_free, bool):
            raise TypeError("default_free must be boolean")
        if self.constraint not in {"real", "nonnegative", "positive"}:
            raise ValueError("constraint must be real, nonnegative, or positive")
        if self.default_initial is not None:
            self._validate_value(self.default_initial, "default_initial")
        if self.default_bounds is not None:
            if len(self.default_bounds) != 2:
                raise ValueError("default_bounds must contain lower and upper")
            lower, upper = self.default_bounds
            if not all(
                isinstance(value, Real)
                and not isinstance(value, bool)
                and isfinite(float(value))
                for value in (lower, upper)
            ):
                raise ValueError("default_bounds must contain finite real values")
            if lower >= upper:
                raise ValueError("default_bounds lower must be below upper")
            self._validate_value(lower, "default_bounds lower")
            if self.constraint == "positive" and lower <= 0.0:
                raise ValueError("positive parameter bounds must have a positive lower")

    def _validate_value(self, value: float, source: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{source} must be a real number")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{source} must be finite")
        if self.constraint == "positive" and numeric <= 0.0:
            raise ValueError(f"{source} must be positive")
        if self.constraint == "nonnegative" and numeric < 0.0:
            raise ValueError(f"{source} must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "role": self.role,
            "default_initial": self.default_initial,
            "default_bounds": (
                None if self.default_bounds is None else list(self.default_bounds)
            ),
            "default_free": self.default_free,
            "constraint": self.constraint,
        }


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Static capabilities and parameters associated with one ``ModelID``."""

    model_id: ModelID
    display_name: str
    description: str
    parameters: tuple[ParameterDescriptor, ...]
    supported_surface_modes: tuple[str, ...]
    supported_boundary_modes: tuple[str, ...]
    supported_step_types: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    geometry_unit: str = DEFAULT_UNITS.length
    factory_strategy: str | None = None
    factory_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, ModelID):
            raise TypeError("model_id must be a ModelID")
        if not self.display_name.strip() or not self.description.strip():
            raise ValueError("display_name and description must not be empty")
        parameters = tuple(self.parameters)
        if not all(isinstance(item, ParameterDescriptor) for item in parameters):
            raise TypeError("parameters must contain ParameterDescriptor values")
        names = tuple(item.name for item in parameters)
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate parameter in descriptor {self.model_id.value}")
        for field_name in (
            "supported_surface_modes",
            "supported_boundary_modes",
            "supported_step_types",
            "assumptions",
            "limitations",
        ):
            values = tuple(getattr(self, field_name))
            if field_name.startswith("supported_") and not values:
                raise ValueError(f"{field_name} must not be empty")
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError(f"{field_name} must contain nonempty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
            object.__setattr__(self, field_name, values)
        if self.geometry_unit != DEFAULT_UNITS.length:
            raise ValueError("geometry_unit must use DEFAULT_UNITS.length")
        if self.factory_strategy is None and not self.factory_unavailable_reason:
            raise ValueError(
                "descriptors without a factory strategy require an explicit reason"
            )
        object.__setattr__(self, "parameters", parameters)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def parameters_by_name(self) -> Mapping[str, ParameterDescriptor]:
        return MappingProxyType(
            {parameter.name: parameter for parameter in self.parameters}
        )

    @property
    def factory_supported(self) -> bool:
        return self.factory_strategy is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id.value,
            "display_name": self.display_name,
            "description": self.description,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "supported_surface_modes": list(self.supported_surface_modes),
            "supported_boundary_modes": list(self.supported_boundary_modes),
            "supported_step_types": list(self.supported_step_types),
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
            "geometry_unit": self.geometry_unit,
            "factory_supported": self.factory_supported,
            "factory_unavailable_reason": self.factory_unavailable_reason,
        }


class ModelRegistry:
    """Read-only, complete mapping whose only keys are ``ModelID`` values."""

    def __init__(
        self,
        descriptors: Iterable[ModelDescriptor],
        *,
        require_complete: bool = True,
    ) -> None:
        values: dict[ModelID, ModelDescriptor] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, ModelDescriptor):
                raise TypeError("registry entries must be ModelDescriptor values")
            if descriptor.model_id in values:
                raise ValueError(
                    f"duplicate model descriptor: {descriptor.model_id.value}"
                )
            values[descriptor.model_id] = descriptor
        if require_complete:
            missing = tuple(model_id for model_id in ModelID if model_id not in values)
            if missing:
                raise ValueError(
                    "registry is missing ModelID values: "
                    + ", ".join(model_id.value for model_id in missing)
                )
        self._descriptors = MappingProxyType(values)

    @property
    def descriptors(self) -> Mapping[ModelID, ModelDescriptor]:
        return self._descriptors

    @property
    def model_ids(self) -> tuple[ModelID, ...]:
        return tuple(model_id for model_id in ModelID if model_id in self._descriptors)

    def __getitem__(self, model_id: ModelID | str) -> ModelDescriptor:
        return self.get(model_id)

    def __iter__(self) -> Iterator[ModelID]:
        return iter(self.model_ids)

    def __len__(self) -> int:
        return len(self._descriptors)

    def get(self, model_id: ModelID | str) -> ModelDescriptor:
        try:
            normalized = ModelID(model_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown model_id: {model_id!r}") from exc
        try:
            return self._descriptors[normalized]
        except KeyError as exc:
            raise KeyError(f"model is not registered: {normalized.value}") from exc

    def validate_config(self, config: ModelConfig) -> ModelDescriptor:
        """Validate a model config against its registered descriptor."""

        from pyecr.config import ModelConfig

        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        config.validate()
        descriptor = self.get(config.model_id)
        if config.surface_mode not in descriptor.supported_surface_modes:
            raise ValueError(
                f"{config.model_id.value} does not support surface_mode="
                f"{config.surface_mode!r}"
            )
        if config.boundary_mode not in descriptor.supported_boundary_modes:
            raise ValueError(
                f"{config.model_id.value} does not support boundary_mode="
                f"{config.boundary_mode!r}"
            )
        if config.step_type not in descriptor.supported_step_types:
            raise ValueError(
                f"{config.model_id.value} does not support step_type="
                f"{config.step_type!r}"
            )

        declared = set(descriptor.parameter_names)
        referenced = (
            set(config.initial_values)
            | set(config.parameter_bounds)
            | set(config.free_parameters)
            | set(config.fixed_parameters)
        )
        unknown = sorted(referenced - declared)
        if unknown:
            raise ValueError(
                f"{config.model_id.value} has undeclared parameters: "
                + ", ".join(unknown)
            )

        for parameter_name, value in (
            *config.initial_values.items(),
            *config.fixed_parameters.items(),
        ):
            descriptor.parameters_by_name[parameter_name]._validate_value(
                value, parameter_name
            )
        for parameter_name, bounds in config.parameter_bounds.items():
            parameter = descriptor.parameters_by_name[parameter_name]
            parameter._validate_value(bounds[0], f"{parameter_name} lower bound")
            if parameter.constraint == "positive" and bounds[0] <= 0.0:
                raise ValueError(f"{parameter_name} lower bound must be positive")
        return descriptor


_ALL_BOUNDARIES = ("ideal_step", "exponential_flush", "measured_history")
_STANDARD_STEPS = ("hydration", "dehydration")
_DUAL_STEPS = (*_STANDARD_STEPS, "custom")


def _diffusion(name: str, *, default_free: bool = True) -> ParameterDescriptor:
    return ParameterDescriptor(
        name=name,
        unit=DEFAULT_UNITS.diffusion_coefficient,
        role="transport",
        default_initial=1.0e-6,
        default_bounds=(1.0e-12, 1.0e-2),
        default_free=default_free,
        constraint="positive",
    )


def _surface_exchange(name: str) -> ParameterDescriptor:
    return ParameterDescriptor(
        name=name,
        unit=DEFAULT_UNITS.surface_exchange_coefficient,
        role="transport",
        default_initial=1.0e-4,
        default_bounds=(1.0e-10, 1.0),
        constraint="positive",
    )


def _signal_parameter(
    name: str,
    *,
    baseline: bool = False,
    default_free: bool = True,
    allow_negative: bool = False,
    positive: bool = False,
) -> ParameterDescriptor:
    if allow_negative and positive:
        raise ValueError("signal parameter cannot be both signed and positive")
    unconstrained = baseline or allow_negative
    return ParameterDescriptor(
        name=name,
        unit=DEFAULT_UNITS.signal,
        role="baseline" if baseline else "amplitude",
        default_initial=0.0 if baseline else 0.1,
        default_bounds=(
            (-1.0e6, 1.0e6)
            if unconstrained
            else (1.0e-14, 1.0e6) if positive else (0.0, 1.0e6)
        ),
        default_free=default_free,
        constraint=(
            "real" if unconstrained else "positive" if positive else "nonnegative"
        ),
    )


def _default_descriptors() -> tuple[ModelDescriptor, ...]:
    return (
        ModelDescriptor(
            model_id=ModelID.M8_DUAL_K,
            display_name="M8 — dual surface-controlled effective modes",
            description="Joint y0+B*X1+C*X2 over every fit-eligible time; no D or chemical attribution by speed.",
            parameters=(
                ParameterDescriptor("k_1_cm_s",DEFAULT_UNITS.surface_exchange_coefficient,"transport",1e-3,(1e-10,1.),True,"positive"),
                _surface_exchange("k_2_cm_s"),
                ParameterDescriptor("A_1",DEFAULT_UNITS.signal,"amplitude",.1,(0.,1e6),True,"nonnegative"),
                ParameterDescriptor("A_2",DEFAULT_UNITS.signal,"amplitude",.1,(0.,1e6),True,"nonnegative"),
                _signal_parameter("y0",baseline=True,default_free=False),
            ),
            supported_surface_modes=("surface_controlled",),supported_boundary_modes=("ideal_step",),
            supported_step_types=_DUAL_STEPS,
            assumptions=("Two independent linear effective modes; uniform internal state and ideal gas step.",
                         "Both faces of each active axis exposed with uniform k per mode."),
            limitations=("Near rates and weak amplitudes can prevent separate kinetic inference.",
                         "Effective fast/slow labels do not identify H/O; no default M7 replacement.",
                         "Independent Batch supports both k; effective labels may reorder with temperature.",
                         "Pair/Global/DCT and nonideal boundaries Unsupported."),
            factory_strategy="dual_surface_k",
        ),
        ModelDescriptor(
            model_id=ModelID.M7_SINGLE_K,
            display_name="M7 — single surface-controlled k",
            description="Analytic uniform-state surface exchange; k is the only kinetic parameter.",
            parameters=(
                _surface_exchange("k_cm_s"),
                ParameterDescriptor("amplitude", DEFAULT_UNITS.signal, "amplitude", 1.0,
                                    (-1e6, 1e6), False, "real"),
                _signal_parameter("baseline", baseline=True, default_free=False),
            ),
            supported_surface_modes=("surface_controlled",),
            supported_boundary_modes=("ideal_step",),
            supported_step_types=_STANDARD_STEPS,
            assumptions=("Uniform internal state, linear small perturbation and ideal gas step.",
                         "Both faces of each active axis are exposed with uniform constant k."),
            limitations=("No fitted D: Bi cannot verify the surface-control assumption.",
                         "Raw amplitude/baseline may be free; count all free parameters.",
                         "Independent Batch supports k; Pair, Global, DCT and nonideal boundaries are unsupported."),
            factory_strategy="single_surface_k",
        ),
        ModelDescriptor(
            model_id=ModelID.M0_SINGLE_D,
            display_name="M0 — single-process diffusion",
            description="Single-process diffusion-controlled ECR model.",
            parameters=(
                _diffusion("D_cm2_s"),
                _signal_parameter(
                    "amplitude", default_free=False, allow_negative=True
                ),
                _signal_parameter("baseline", baseline=True, default_free=False),
            ),
            supported_surface_modes=("instant",),
            supported_boundary_modes=_ALL_BOUNDARIES,
            supported_step_types=_STANDARD_STEPS,
            assumptions=(
                "Rectangular specimen with transport only along active dimensions.",
                "A linear small-step response is appropriate for the selected data.",
            ),
            limitations=(
                "Surface exchange is treated as instantaneous.",
                "A single transport process may not represent multi-stage relaxation.",
            ),
            factory_strategy="single_diffusion",
        ),
        ModelDescriptor(
            model_id=ModelID.M1_SINGLE_DK,
            display_name="M1 — single-process diffusion and surface exchange",
            description="Single-process finite-surface rectangular ECR kernel.",
            parameters=(
                _diffusion("D_cm2_s"),
                _surface_exchange("k_cm_s"),
                _signal_parameter(
                    "amplitude", default_free=False, allow_negative=True
                ),
                _signal_parameter("baseline", baseline=True, default_free=False),
            ),
            supported_surface_modes=("finite",),
            supported_boundary_modes=_ALL_BOUNDARIES,
            supported_step_types=_STANDARD_STEPS,
            assumptions=(
                "Rectangular specimen with transport only along active dimensions.",
                "One common surface-exchange coefficient applies to active faces.",
            ),
            limitations=(
                "Strongly diffusion- or surface-limited data may not separate D and k.",
                "The model contains only one material relaxation process.",
            ),
            factory_strategy="single_finite",
        ),
        ModelDescriptor(
            model_id=ModelID.M2_DUAL_HO_D,
            display_name="M2 — dual H/O diffusion",
            description="Independent H/O diffusion-controlled process composition.",
            parameters=(
                _diffusion("D_H_cm2_s"),
                _diffusion("D_O_cm2_s"),
                _signal_parameter("A_H", positive=True),
                _signal_parameter("A_O", positive=True),
                _signal_parameter("y0", baseline=True, default_free=False),
            ),
            supported_surface_modes=("instant",),
            supported_boundary_modes=("ideal_step",),
            supported_step_types=_DUAL_STEPS,
            assumptions=(
                "H and O contributions are represented by independent kernels.",
                "Both processes share the recorded active specimen dimensions.",
            ),
            limitations=(
                "Surface exchange is treated as instantaneous for both processes.",
                "Closely spaced processes can be weakly identifiable or label-sensitive.",
            ),
            factory_strategy="dual_ho_diffusion",
        ),
        ModelDescriptor(
            model_id=ModelID.M3_DUAL_HO_DK,
            display_name="M3 — dual H/O diffusion and surface exchange",
            description="Independent H/O finite-surface process composition.",
            parameters=(
                _diffusion("D_H_cm2_s"),
                _diffusion("D_O_cm2_s"),
                _surface_exchange("k_H_cm_s"),
                _surface_exchange("k_O_cm_s"),
                _signal_parameter("A_H", positive=True),
                _signal_parameter("A_O", positive=True),
                _signal_parameter("y0", baseline=True, default_free=False),
            ),
            supported_surface_modes=("finite",),
            supported_boundary_modes=("ideal_step",),
            supported_step_types=_DUAL_STEPS,
            assumptions=(
                "H and O contributions are represented by independent kernels.",
                "Each process has one surface coefficient shared by its active faces.",
            ),
            limitations=(
                "The larger parameter set requires strong data sensitivity and diagnostics.",
                "Closely spaced processes can be weakly identifiable or label-sensitive.",
            ),
            factory_strategy="dual_ho_finite",
        ),
        ModelDescriptor(
            model_id=ModelID.M4_FLUSH,
            display_name="M4 — flush-corrected boundary",
            description="Boundary-correction capability, not a standalone model object.",
            parameters=(
                ParameterDescriptor(
                    name="tau_f_s",
                    unit=DEFAULT_UNITS.time,
                    role="boundary",
                    default_initial=1.0,
                    default_bounds=(1.0e-12, 1.0e12),
                    constraint="positive",
                ),
            ),
            supported_surface_modes=("instant", "finite"),
            supported_boundary_modes=("exponential_flush", "measured_history"),
            supported_step_types=_STANDARD_STEPS,
            assumptions=(
                "A separately defined material model supplies the ideal response.",
            ),
            limitations=(
                "This registry entry is a boundary adapter, not a standalone fit model.",
            ),
            factory_unavailable_reason=(
                "M4 is a boundary adapter and has no standalone model assembly"
            ),
        ),
        ModelDescriptor(
            model_id=ModelID.M5_GLOBAL,
            display_name="M5 — global ECR",
            description="Multi-curve fitting workflow, not a standalone model object.",
            parameters=(_diffusion("D_cm2_s"), _surface_exchange("k_cm_s")),
            supported_surface_modes=("finite",),
            supported_boundary_modes=_ALL_BOUNDARIES,
            supported_step_types=_STANDARD_STEPS,
            assumptions=(
                "Selected curves share explicitly declared physical parameters.",
            ),
            limitations=(
                "This entry requires a multi-curve workflow rather than one model object.",
            ),
            factory_unavailable_reason=(
                "M5 is constructed through GlobalCurve and fit_global_ecr"
            ),
        ),
        ModelDescriptor(
            model_id=ModelID.M6_DCT,
            display_name="M6 — characteristic-time distribution",
            description="Diagnostic time-scale distribution, not a physical model object.",
            parameters=(
                ParameterDescriptor(
                    name="regularization_lambda",
                    unit=DEFAULT_UNITS.dimensionless,
                    role="regularization",
                    default_initial=1.0e-4,
                    default_bounds=(1.0e-12, 1.0e4),
                    constraint="positive",
                ),
            ),
            supported_surface_modes=("instant", "finite"),
            supported_boundary_modes=_ALL_BOUNDARIES,
            supported_step_types=_STANDARD_STEPS,
            assumptions=(
                "The measured relaxation can be represented by nonnegative time scales.",
            ),
            limitations=(
                "Candidate time scales do not identify chemical species by themselves.",
            ),
            factory_unavailable_reason=(
                "M6 is a diagnostic workflow and has no physical model assembly"
            ),
        ),
    )


DEFAULT_MODEL_REGISTRY = ModelRegistry(_default_descriptors())
