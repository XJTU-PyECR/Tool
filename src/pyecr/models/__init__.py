"""Scientific model identifiers and model interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyecr.models.dual_ho import (
    DualHOModel,
    DualHOResult,
    FractionKernelProtocol,
    StepType,
)
from pyecr.models.ids import ModelID

if TYPE_CHECKING:
    from pyecr.models.factory import ModelFactory, ModelNotImplementedError
    from pyecr.models.registry import (
        DEFAULT_MODEL_REGISTRY,
        ModelDescriptor,
        ModelRegistry,
        ParameterDescriptor,
    )
    from pyecr.models.single import (
        SingleEcrContext,
        SingleEcrModel,
        SingleEcrPrediction,
    )

__all__ = [
    "DualHOModel",
    "DualHOResult",
    "DEFAULT_MODEL_REGISTRY",
    "FractionKernelProtocol",
    "ModelDescriptor",
    "ModelFactory",
    "ModelID",
    "ModelNotImplementedError",
    "ModelRegistry",
    "ParameterDescriptor",
    "SingleEcrContext",
    "SingleEcrModel",
    "SingleEcrPrediction",
    "StepType",
]


def __getattr__(name: str) -> object:
    """Load registry/factory APIs lazily to keep config imports acyclic."""

    if name in {"ModelFactory", "ModelNotImplementedError"}:
        from pyecr.models.factory import ModelFactory, ModelNotImplementedError

        return {
            "ModelFactory": ModelFactory,
            "ModelNotImplementedError": ModelNotImplementedError,
        }[name]
    if name in {
        "DEFAULT_MODEL_REGISTRY",
        "ModelDescriptor",
        "ModelRegistry",
        "ParameterDescriptor",
    }:
        from pyecr.models.registry import (
            DEFAULT_MODEL_REGISTRY,
            ModelDescriptor,
            ModelRegistry,
            ParameterDescriptor,
        )

        return {
            "DEFAULT_MODEL_REGISTRY": DEFAULT_MODEL_REGISTRY,
            "ModelDescriptor": ModelDescriptor,
            "ModelRegistry": ModelRegistry,
            "ParameterDescriptor": ParameterDescriptor,
        }[name]
    if name in {"SingleEcrContext", "SingleEcrModel", "SingleEcrPrediction"}:
        from pyecr.models.single import (
            SingleEcrContext,
            SingleEcrModel,
            SingleEcrPrediction,
        )

        return {
            "SingleEcrContext": SingleEcrContext,
            "SingleEcrModel": SingleEcrModel,
            "SingleEcrPrediction": SingleEcrPrediction,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
