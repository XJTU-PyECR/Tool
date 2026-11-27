"""Compatibility import for shared presentation-only model aliases."""
from pyecr.model_display import (
    VERSION, ModelDisplay, MODEL_DISPLAY, BY_ID, BY_LABEL,
    model_id_for_display, display_label, display_labels, display_name,
)

__all__ = [
    "VERSION", "ModelDisplay", "MODEL_DISPLAY", "BY_ID", "BY_LABEL",
    "model_id_for_display", "display_label", "display_labels", "display_name",
]
