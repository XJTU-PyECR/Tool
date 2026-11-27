"""Central assembly point for existing PyECR kernels and model objects."""

from __future__ import annotations

from pyecr.boundary.composition import BoundaryComposedSingleEcrModel
from pyecr.config import AnalysisConfig, BoundaryConfig, ModelConfig
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.rectangular import (
    DiffusionControlledRectangularKernel,
    RectangularKernel,
)
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY, ModelRegistry
from pyecr.models.single import SingleEcrModel
from pyecr.models.surface_k import SurfaceKKernel, SurfaceKModel
from pyecr.models.dual_surface_k import DualSurfaceKModel, input_signs

__all__ = ["ModelFactory", "ModelNotImplementedError"]


class ModelNotImplementedError(NotImplementedError):
    """Raised when a registered model has no honest existing assembly path."""


class ModelFactory:
    """Validate configuration and assemble only already-implemented objects.

    The factory contains no forward equations. It maps registered assembly
    strategies to the existing single-process kernels and ``DualHOModel``.
    """

    def __init__(self, registry: ModelRegistry = DEFAULT_MODEL_REGISTRY) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError("registry must be a ModelRegistry")
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def create(
        self,
        config: ModelConfig | AnalysisConfig,
        *,
        boundary: BoundaryConfig | None = None,
    ) -> SingleEcrModel | BoundaryComposedSingleEcrModel | DualHOModel | SurfaceKModel | DualSurfaceKModel:
        """Create an existing kernel/model or fail explicitly if unavailable."""

        if isinstance(config, AnalysisConfig):
            config.validate()
            model_config = config.model
            analysis_config: AnalysisConfig | None = config
        elif isinstance(config, ModelConfig):
            model_config = config
            analysis_config = None
        else:
            raise TypeError("config must be ModelConfig or AnalysisConfig")

        # A forward request may supply its boundary without inventing an
        # AnalysisConfig (and therefore a Fit/observation policy).
        if boundary is not None:
            if not isinstance(boundary, BoundaryConfig):
                raise TypeError("boundary must be BoundaryConfig")
            boundary.validate()
            if boundary.mode != model_config.boundary_mode:
                raise ValueError("explicit boundary/model mode conflict")
            if analysis_config is not None and boundary != analysis_config.boundary:
                raise ValueError("explicit boundary/AnalysisConfig conflict")
        elif analysis_config is not None:
            boundary = analysis_config.boundary

        descriptor = self._registry.validate_config(model_config)
        strategy = descriptor.factory_strategy
        if strategy is None:
            reason = descriptor.factory_unavailable_reason or "not implemented"
            raise ModelNotImplementedError(
                f"{descriptor.model_id.value} cannot be constructed: {reason}"
            )

        geometry = model_config.geometry
        if strategy == "single_surface_k":
            return SurfaceKModel(SurfaceKKernel(geometry))
        if strategy == "dual_surface_k":
            return DualSurfaceKModel(SurfaceKKernel(geometry),input_signs(model_config))
        if strategy == "single_diffusion":
            material = SingleEcrModel(
                model_id=descriptor.model_id,
                kernel=DiffusionControlledRectangularKernel(
                    full_dims_cm=geometry.full_dims_cm,
                    active_dimensions=geometry.active_dimensions,
                ),
            )
            return self._compose_single(material, model_config, boundary)
        if strategy == "single_finite":
            material = SingleEcrModel(
                model_id=descriptor.model_id,
                kernel=RectangularKernel(
                    full_dims_cm=geometry.full_dims_cm,
                    active_dimensions=geometry.active_dimensions,
                ),
            )
            return self._compose_single(material, model_config, boundary)
        if strategy == "dual_ho_diffusion":
            kernel = DiffusionControlledRectangularKernel(
                full_dims_cm=geometry.full_dims_cm,
                active_dimensions=geometry.active_dimensions,
            )
            return DualHOModel(
                kernel=kernel,
                step_type=model_config.step_type,
                sign_H=model_config.sign_H,
                sign_O=model_config.sign_O,
            )
        if strategy == "dual_ho_finite":
            kernel = RectangularKernel(
                full_dims_cm=geometry.full_dims_cm,
                active_dimensions=geometry.active_dimensions,
            )
            return DualHOModel(
                kernel=kernel,
                step_type=model_config.step_type,
                sign_H=model_config.sign_H,
                sign_O=model_config.sign_O,
            )
        raise RuntimeError(
            f"unknown registered factory strategy {strategy!r} for "
            f"{descriptor.model_id.value}"
        )

    @staticmethod
    def _compose_single(
        material: SingleEcrModel,
        model_config: ModelConfig,
        boundary: BoundaryConfig | None,
    ) -> SingleEcrModel | BoundaryComposedSingleEcrModel:
        if model_config.boundary_mode == "ideal_step":
            return material
        if boundary is None:
            raise ModelNotImplementedError(
                "AnalysisConfig carrying validated BoundaryConfig options is "
                f"required for {model_config.boundary_mode!r} boundary composition; "
                "a forward caller may instead supply boundary=BoundaryConfig"
            )
        return BoundaryComposedSingleEcrModel(
            material_model=material,
            boundary=boundary,
        )
