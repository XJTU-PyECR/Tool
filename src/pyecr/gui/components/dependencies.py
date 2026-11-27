"""Presentation-only imports and constants shared across page modules."""
from __future__ import annotations
import json
import traceback
from typing import Any
from pyecr.gui.model_builder import ModelFormValues, ModelParameterValue
from pyecr.gui.state import GUI_PAGES, GuiPage
from pyecr.io.importer import DelimiterMode, HeaderMode, ImportMapping, TextParseConfig
from pyecr.io.dataset import EcrDataset
from pyecr.preprocessing import PREPROCESSING_OPERATIONS, ProcessingStep
from pyecr.plotting import FitPlotData
from pyecr.services.batch_analysis import BatchAnalysisResult, GlobalAnalysisResult
from pyecr.services.dct_service import DctAnalysisConfig, DctPresentation
from pyecr.gui.components.tags import WidgetTags
from pyecr.units import SIGNAL_TYPES

_TIME_UNIT_CHOICES = ("s", "ms", "min", "h", "us")
_SIGNAL_TYPE_CHOICES = SIGNAL_TYPES
_HEADER_MODE_CHOICES = tuple(mode.name for mode in HeaderMode)
_TEXT_PARSE_MODE_CHOICES = tuple(mode.name for mode in DelimiterMode)
_DATA_PREVIEW_CONTAINER_TAG = WidgetTags.DATA_PREVIEW_CONTAINER
_DATA_PREVIEW_TABLE_TAG = WidgetTags.DATA_PREVIEW_TABLE
_PROCESSING_PARAMETER_EXAMPLES = {
    "convert_time_unit": {"to_unit": "s"},
    "resistance_to_conductance": {},
    "conductance_to_conductivity": {
        "path_length_cm": 1.0,
        "cross_section_area_cm2": 1.0,
    },
    "set_manual_t0": {"t0": 0.0},
    "crop": {"start_time": 0.0, "end_time": 100.0},
    "mask_points": {"raw_indices": [0]},
    "choose_initial_stability_window": {"start_time": 0.0, "end_time": 10.0},
    "choose_final_stability_window": {"start_time": 90.0, "end_time": 100.0},
    "normalize": {"mode": "progress"},
}


