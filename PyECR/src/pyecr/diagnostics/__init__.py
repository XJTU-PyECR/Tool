"""Fit diagnostics, model comparison, and reliability gates."""

from pyecr.diagnostics.dct_assistant import (
    ClassicalBiAssessment,
    DCTModelSuggestion,
    LambdaStabilityAssessment,
    suggest_models_from_dct,
)
from pyecr.diagnostics.dual_error_map import (
    DualErrorMapCellFailure,
    DualErrorMapConfig,
    DualErrorMapPoint,
    DualErrorMapQuality,
    DualErrorMapResult,
    DualErrorMapSlice,
    DualMapRangeMode,
    DualSliceQuality,
    compute_dual_error_map,
)
from pyecr.diagnostics.error_map import (
    ErrorMapCellFailure,
    ErrorMapConfig,
    ErrorMapDiagnostics,
    ErrorMapMode,
    ErrorMapPoint,
    ErrorMapQuality,
    ErrorMapResolution,
    ErrorMapResult,
    assess_error_map_quality,
    compute_error_map,
)
from pyecr.diagnostics.model_selection import (
    CandidateAssessment,
    ModelSelectionConfig,
    ModelSelectionResult,
    ParameterBound,
    select_model,
)

__all__ = [
    "CandidateAssessment",
    "ClassicalBiAssessment",
    "DCTModelSuggestion",
    "DualErrorMapCellFailure",
    "DualErrorMapConfig",
    "DualErrorMapPoint",
    "DualErrorMapQuality",
    "DualErrorMapResult",
    "DualErrorMapSlice",
    "DualMapRangeMode",
    "DualSliceQuality",
    "ErrorMapCellFailure",
    "ErrorMapConfig",
    "ErrorMapDiagnostics",
    "ErrorMapMode",
    "ErrorMapPoint",
    "ErrorMapQuality",
    "ErrorMapResolution",
    "ErrorMapResult",
    "assess_error_map_quality",
    "LambdaStabilityAssessment",
    "ModelSelectionConfig",
    "ModelSelectionResult",
    "ParameterBound",
    "select_model",
    "compute_error_map",
    "compute_dual_error_map",
    "suggest_models_from_dct",
]
