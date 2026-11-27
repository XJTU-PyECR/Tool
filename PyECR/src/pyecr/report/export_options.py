"""Explicit v2 export choices; no solver or scientific-display precision state."""
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
import json

VERSION = "pyecr-export/2.0.0"


class ExportPreset(str, Enum):
    REVIEW = "review"
    EXCEL = "excel"
    PUBLICATION = "publication"


@dataclass(frozen=True, slots=True)
class ExportOptions:
    preset: ExportPreset = ExportPreset.REVIEW
    csv_encoding: str = "utf-8"
    csv_delimiter: str = ","
    image_dpi: int = 300
    width_in: float = 7.2
    height_in: float = 4.8
    font_family: str = "DejaVu Sans"
    max_data_rows_per_sheet: int = 1048575
    plot_views: str | None = None
    legend_policy: str = 'follow'

    def __post_init__(self):
        if self.legend_policy not in ('follow', 'auto'):
            raise ValueError('export legend policy must be follow or auto')
        if self.plot_views is not None:
            from pyecr.plotting.view_state import freeze_views
            object.__setattr__(self, 'plot_views', freeze_views(self.plot_views))
        object.__setattr__(self, "preset", ExportPreset(self.preset))
        if self.csv_encoding not in {"utf-8", "utf-8-sig"} or self.csv_delimiter not in {",", ";", "\t"}:
            raise ValueError("v2 CSV supports UTF-8/BOM and comma/semicolon/tab only")
        if isinstance(self.image_dpi, bool) or self.image_dpi not in {300, 600}:
            raise ValueError("publication DPI must be 300 or 600")
        for name in ("width_in", "height_in"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isfinite(v) or not 2 <= v <= 20:
                raise ValueError(f"{name} must be finite in [2, 20]")
        if not isinstance(self.font_family, str) or not self.font_family.strip():
            raise ValueError("font_family is required")
        if isinstance(self.max_data_rows_per_sheet, bool) or not isinstance(self.max_data_rows_per_sheet, int) or not 1 <= self.max_data_rows_per_sheet <= 1048575:
            raise ValueError("XLSX data sheet limit must be in [1, 1048575]")

    def to_dict(self):
        value = {**asdict(self), "preset": self.preset.value}
        if self.legend_policy == 'follow':
            value.pop('legend_policy')  # Preserve existing package option hashes.
        return value


def field_dictionary(*,scientific_extensions=False):
    value=json.loads(Path(__file__).with_name("export_schema_v2.json").read_text(encoding="utf-8"))
    if not scientific_extensions:return value
    value['scientific_extension']='scientific-result-evidence/1; analysis snapshot 3'
    value['tables']['ScientificEvidence']={'columns':['field','value_json','status','reason'],'types':['text','json','text','text?']}
    value['tables']['TransportRegime']={'columns':['process','axis','D_cm2_s','k_cm_s','full_size_cm','half_size_cm','Bi','half_length_D_over_k_cm','slab_full_thickness_2D_over_k_cm','regime','parent_reliability','source_kind','point_id','summary_id','reason'],
        'types':['text','text','float?','float?','float','float','float?','float?','float?','text','text','text','text','text','text?']}
    return value


def encode_text(value: str) -> str:
    if value == "":
        return "'"  # Reversible distinction from a nullable CSV cell.
    trimmed = value.lstrip()
    if value.startswith(("=", "+", "-", "@", "\t", "\r", "\n", "'")) or trimmed.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def decode_text(value: str) -> str:
    return value[1:] if value.startswith("'") else value
