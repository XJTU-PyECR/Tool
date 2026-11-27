"""Batch fitting and temperature-series analysis."""

from pyecr.batch.temperature_series import (
    GAS_CONSTANT_J_MOL_K,
    ArrheniusResult,
    BatchFitOutcome,
    BatchFitRecord,
    ReliabilityDecision,
    TemperatureDataset,
    TemperatureSeriesResult,
    TemperatureUnit,
    fit_temperature_series,
)

__all__ = [
    "GAS_CONSTANT_J_MOL_K",
    "ArrheniusResult",
    "BatchFitOutcome",
    "BatchFitRecord",
    "ReliabilityDecision",
    "TemperatureDataset",
    "TemperatureSeriesResult",
    "TemperatureUnit",
    "fit_temperature_series",
]
