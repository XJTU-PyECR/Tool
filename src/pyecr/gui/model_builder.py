"""Registry-driven Model-page schema values and AnalysisConfig construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from pyecr.config import (
    AnalysisConfig,
    BoundaryConfig,
    FitConfig,
    ModelConfig,
    SampleGeometry,
    SolverConfig,
)
from pyecr.models.ids import ModelID
from pyecr.models.registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelDescriptor,
    ModelRegistry,
)

__all__ = [
    "ModelFormValues",
    "ModelPageConfigBuilder",
    "ModelParameterValue",
]


@dataclass(frozen=True, slots=True)
class ModelParameterValue:
    """Editable values for one descriptor-defined parameter row."""

    value: float
    lower_bound: float
    upper_bound: float
    free: bool

    def __post_init__(self) -> None:
        for name in ("value", "lower_bound", "upper_bound"):
            supplied = getattr(self, name)
            if isinstance(supplied, bool) or not isinstance(supplied, Real):
                raise TypeError(f"{name} must be a real number")
            numeric = float(supplied)
            if not isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, numeric)
        if not isinstance(self.free, bool):
            raise TypeError("free must be boolean")


@dataclass(frozen=True, slots=True)
class ModelFormValues:
    """Presentation-neutral values collected by the schema-driven Model page."""

    model_id: ModelID
    full_dims_cm: tuple[float, float, float]
    active_dimensions: tuple[bool, bool, bool]
    surface_mode: str
    boundary_mode: str
    step_type: str
    parameters: Mapping[str, ModelParameterValue]
    sign_H: float | None = None
    sign_O: float | None = None
    relative_tolerance: float = 1.0e-8
    absolute_tolerance: float = 1.0e-12
    max_terms: int = 200
    max_nfev: int = 5_000
    loss: str = "linear"
    model_specific: Mapping[str, object] = field(default_factory=dict)
    flush_time_s: float | None = None
    measured_time_s: tuple[float, ...] = ()
    measured_values: tuple[float, ...] = ()
    measured_signal_type: str | None = None
    measured_initial_value: float | None = None
    measured_final_value: float | None = None
    measured_value_unit: str = "unspecified"
    resample_time_step_s: float | None = None
    max_internal_points: int = 200_000
    random_seed: int | None = None

    def __post_init__(self) -> None:
        try:
            model_id = ModelID(self.model_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown model_id: {self.model_id!r}") from exc
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        if not isinstance(self.model_specific, Mapping):
            raise TypeError("model_specific must be a mapping")
        normalized: dict[str, ModelParameterValue] = {}
        for name, value in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError("parameter names must be nonempty strings")
            if not isinstance(value, ModelParameterValue):
                raise TypeError(f"parameters[{name!r}] must be ModelParameterValue")
            normalized[name] = value
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "full_dims_cm", tuple(self.full_dims_cm))
        object.__setattr__(self, "active_dimensions", tuple(self.active_dimensions))
        object.__setattr__(self, "measured_time_s", tuple(self.measured_time_s))
        object.__setattr__(self, "measured_values", tuple(self.measured_values))
        object.__setattr__(self, "parameters", MappingProxyType(normalized))
        object.__setattr__(
            self, "model_specific", MappingProxyType(dict(self.model_specific))
        )


class ModelPageConfigBuilder:
    """Build validated core configuration solely from registry metadata."""

    def __init__(self, registry: ModelRegistry = DEFAULT_MODEL_REGISTRY) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError("registry must be a ModelRegistry")
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return tuple(
            descriptor
            for descriptor in (
                self._registry.get(model_id) for model_id in self._registry
            )
            if descriptor.factory_supported
        )

    def descriptor(self, model_id: ModelID | str) -> ModelDescriptor:
        descriptor = self._registry.get(model_id)
        if not descriptor.factory_supported:
            raise ValueError(
                f"{descriptor.model_id.value} is a workflow/boundary capability, "
                "not a selectable material model"
            )
        return descriptor

    def default_form(self, model_id: ModelID | str) -> ModelFormValues:
        descriptor = self.descriptor(model_id)
        parameter_values: dict[str, ModelParameterValue] = {}
        for parameter in descriptor.parameters:
            if parameter.default_initial is None or parameter.default_bounds is None:
                raise ValueError(
                    f"{descriptor.model_id.value}.{parameter.name} lacks GUI defaults"
                )
            parameter_values[parameter.name] = ModelParameterValue(
                value=parameter.default_initial,
                lower_bound=parameter.default_bounds[0],
                upper_bound=parameter.default_bounds[1],
                free=parameter.default_free,
            )
        return ModelFormValues(
            model_id=descriptor.model_id,
            full_dims_cm=(0.1, 0.1, 0.1),
            active_dimensions=(True, False, False),
            surface_mode=descriptor.supported_surface_modes[0],
            boundary_mode=descriptor.supported_boundary_modes[0],
            step_type=descriptor.supported_step_types[0],
            parameters=parameter_values,
        )

    def build(self, values: ModelFormValues, *, parent_fit=None) -> AnalysisConfig:
        if not isinstance(values, ModelFormValues):
            raise TypeError("values must be ModelFormValues")
        descriptor = self.descriptor(values.model_id)
        expected = set(descriptor.parameter_names)
        supplied = set(values.parameters)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError(
                f"parameter controls do not match {values.model_id.value}: "
                + "; ".join(details)
            )

        initial_values = {
            name: control.value for name, control in values.parameters.items()
        }
        parameter_bounds = {
            name: (control.lower_bound, control.upper_bound)
            for name, control in values.parameters.items()
        }
        free_parameters = tuple(
            parameter.name
            for parameter in descriptor.parameters
            if values.parameters[parameter.name].free
        )
        fixed_parameters = {
            parameter.name: values.parameters[parameter.name].value
            for parameter in descriptor.parameters
            if not values.parameters[parameter.name].free
        }
        model = ModelConfig(
            model_id=descriptor.model_id,
            geometry=SampleGeometry(
                full_dims_cm=values.full_dims_cm,
                active_dimensions=values.active_dimensions,
            ),
            surface_mode=values.surface_mode,
            boundary_mode=values.boundary_mode,
            step_type=values.step_type,
            sign_H=values.sign_H,
            sign_O=values.sign_O,
            initial_values=initial_values,
            parameter_bounds=parameter_bounds,
            free_parameters=free_parameters,
            fixed_parameters=fixed_parameters,
        )
        solver = SolverConfig(
            surface_mode=values.surface_mode,
            relative_tolerance=values.relative_tolerance,
            absolute_tolerance=values.absolute_tolerance,
            max_terms=values.max_terms,
        )
        fit = FitConfig(
            model_id=descriptor.model_id,
            max_nfev=values.max_nfev,
            loss=values.loss,
        )
        config = AnalysisConfig(
            model=model,
            solver=solver,
            fit=fit,
            model_specific=values.model_specific,
            boundary=BoundaryConfig(
                mode=values.boundary_mode,
                flush_time_s=values.flush_time_s,
                measured_time_s=values.measured_time_s,
                measured_values=values.measured_values,
                measured_signal_type=values.measured_signal_type,
                measured_initial_value=values.measured_initial_value,
                measured_final_value=values.measured_final_value,
                measured_value_unit=values.measured_value_unit,
                resample_time_step_s=values.resample_time_step_s,
                max_internal_points=values.max_internal_points,
            ),
        )
        if values.random_seed is not None:
            from dataclasses import replace
            from pyecr.fitting.reproducibility import integer_seed
            config=replace(config,model_specific={**config.model_specific,'fit_random_seed':integer_seed(values.random_seed)})
        self._registry.validate_config(model)
        config.validate()
        from pyecr.fitting.solver_strategy import SolverOptions
        SolverOptions.from_config(config)
        if "fixed_d_sources" in values.model_specific:
            from pyecr.services.fixed_d import with_fixed_diffusion
            sources=values.model_specific["fixed_d_sources"]
            if sources is None:
                from dataclasses import replace
                config=replace(config,model_specific={k:v for k,v in config.model_specific.items() if k not in {"fixed_d_sources","fixed_d"}})
            else:
                record=values.model_specific.get("fixed_d",{})
                if record.get("parent_fit") is not None:
                    from pyecr.fitting.base import FitResult
                    parent_fit=FitResult.from_dict(record["parent_fit"])
                config=with_fixed_diffusion(config,sources,parent_fit=parent_fit)
        if config.model_id is ModelID.M7_SINGLE_K:
            if "k_cm_s" not in config.model.free_parameters:
                raise ValueError("M7 requires k to be free; observation parameters may be fixed")
            if SolverOptions.from_config(config).strategy != "trf":
                raise ValueError("M7 local solver must be TRF; other strategies Unsupported")
        if config.model_id is ModelID.M8_DUAL_K:
            if not {"k_1_cm_s","k_2_cm_s"}.issubset(config.model.free_parameters):
                raise ValueError("M8 requires both k free; observations can be fixed")
            if SolverOptions.from_config(config).strategy!="trf": raise ValueError("M8 requires TRF")
        return config
