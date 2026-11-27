"""PyECR public package interface."""

from pyecr.config import AnalysisConfig, BoundaryConfig, ModelConfig
from pyecr.io import EcrDataset, ProcessingRecord
from pyecr.models import (
    DEFAULT_MODEL_REGISTRY,
    ModelDescriptor,
    ModelFactory,
    ModelID,
    ModelNotImplementedError,
    ModelRegistry,
    ParameterDescriptor,
)
from pyecr.services.workflow import AnalysisSession, WorkflowController, WorkflowState
from pyecr.units import DEFAULT_UNITS, UnitConvention

__all__ = [
    "AnalysisConfig",
    "AnalysisSession",
    "BoundaryConfig",
    "DEFAULT_UNITS",
    "DEFAULT_MODEL_REGISTRY",
    "EcrDataset",
    "ModelConfig",
    "ModelDescriptor",
    "ModelFactory",
    "ModelID",
    "ModelNotImplementedError",
    "ModelRegistry",
    "ParameterDescriptor",
    "ProcessingRecord",
    "UnitConvention",
    "WorkflowController",
    "WorkflowState",
]
__version__ = "1.0"
