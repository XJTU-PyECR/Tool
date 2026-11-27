"""Data validation and deterministic preprocessing operations."""

from pyecr.io.dataset import EcrDataset, ProcessingRecord
from pyecr.preprocessing.pipeline import (
    PREPROCESSING_OPERATIONS,
    DownstreamStaleEvent,
    PreprocessConfig,
    PreprocessingPipeline,
    ProcessingStep,
)
from pyecr.preprocessing.quality import (
    FitQualityReport,
    StabilityWindowSuggestion,
    evaluate_fit_quality,
    suggest_stability_window,
)
from pyecr.preprocessing.optional import (
    OptionalPreview, active_outlier_indices, preview_drift, preview_outliers,
    preview_downsampling,
)
from pyecr.preprocessing.signed_normalization import (
    SIGNED_NORMALIZATION_CHECKS, applied_signed_normalization, inverse_signed_dct,
    normalize_signed_dct, raw_observations_sha256,
)

__all__ = [
    "PREPROCESSING_OPERATIONS",
    "DownstreamStaleEvent",
    "EcrDataset",
    "PreprocessConfig",
    "PreprocessingPipeline",
    "ProcessingRecord",
    "ProcessingStep",
    "FitQualityReport",
    "StabilityWindowSuggestion",
    "evaluate_fit_quality",
    "suggest_stability_window",
    "OptionalPreview",
    "active_outlier_indices",
    "preview_drift",
    "preview_outliers",
    "preview_downsampling",
    "SIGNED_NORMALIZATION_CHECKS",
    "normalize_signed_dct",
    "inverse_signed_dct",
    "applied_signed_normalization",
    "raw_observations_sha256",
]
