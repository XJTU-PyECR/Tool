"""Strict tabular-file import primitives for raw ECR observations."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from math import isfinite
from numbers import Integral
from pathlib import Path
from io import StringIO
from typing import Mapping

import numpy as np
import pandas as pd

from pyecr.io.dataset import EcrDataset
from pyecr.units import get_default_unit, validate_pair

__all__ = [
    "ColumnRef",
    "DelimiterMode",
    "HeaderMode",
    "ImportMapping",
    "ImportPreview",
    "ImportResult",
    "ImportValidationError",
    "SourceFileInfo",
    "TextParseConfig",
    "import_tabular_file",
    "preview_tabular_file",
]


_SUPPORTED_SUFFIXES = frozenset({".csv", ".txt", ".xlsx", ".xls"})
_UNNAMED_HEADER = re.compile(r"^Unnamed:\s*\d+(?:_level_\d+)?$", re.IGNORECASE)
_UNCERTAIN_HEADER_MESSAGE = "Header detection uncertain — please confirm."
_UNCERTAIN_DELIMITER_MESSAGE = (
    "Delimiter detection uncertain — please select parsing mode."
)


class ImportValidationError(ValueError):
    """A user-correctable file, mapping, or raw-data validation failure."""


class HeaderMode(str, Enum):
    """How a tabular source identifies its header row."""

    AUTO = "auto"
    FIRST_ROW = "first_row"
    NONE = "none"
    ROW_INDEX = "row_index"


class DelimiterMode(str, Enum):
    """Supported text-table parsing strategies."""

    AUTO = "auto"
    WHITESPACE = "whitespace"
    TAB = "tab"
    COMMA = "comma"
    SEMICOLON = "semicolon"
    FIXED_WIDTH = "fixed_width"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class TextParseConfig:
    """Immutable parsing choice shared by text Preview and final Import."""

    mode: DelimiterMode = DelimiterMode.AUTO
    custom_delimiter: str | None = None
    auto_sample_rows: int = 100

    def __post_init__(self) -> None:
        mode = _delimiter_mode(self.mode)
        delimiter = self.custom_delimiter
        if delimiter is not None and not isinstance(delimiter, str):
            raise TypeError("custom_delimiter must be a string or None")
        if mode is DelimiterMode.CUSTOM:
            if delimiter is None or not delimiter:
                raise ValueError("custom_delimiter is required for CUSTOM mode")
            if len(delimiter) != 1:
                raise ValueError("custom_delimiter must be a single character")
            if delimiter in {"\r", "\n"}:
                raise ValueError("custom_delimiter cannot be a newline")
        else:
            delimiter = None
        if isinstance(self.auto_sample_rows, bool) or not isinstance(
            self.auto_sample_rows, Integral
        ):
            raise TypeError("auto_sample_rows must be an integer")
        sample_rows = int(self.auto_sample_rows)
        if sample_rows < 50 or sample_rows > 100:
            raise ValueError("auto_sample_rows must be between 50 and 100")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "custom_delimiter", delimiter)
        object.__setattr__(self, "auto_sample_rows", sample_rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "custom_delimiter": self.custom_delimiter,
            "auto_sample_rows": self.auto_sample_rows,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TextParseConfig:
        return cls(
            mode=str(payload.get("mode", DelimiterMode.AUTO.value)),
            custom_delimiter=(
                None
                if payload.get("custom_delimiter") is None
                else str(payload["custom_delimiter"])
            ),
            auto_sample_rows=int(payload.get("auto_sample_rows", 100)),
        )


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """Stable zero-based column identity independent of its displayed label."""

    index: int
    original_name: str | None
    display_name: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, Integral):
            raise TypeError("ColumnRef.index must be an integer")
        if int(self.index) < 0:
            raise ValueError("ColumnRef.index must be nonnegative")
        original_name = self.original_name
        if original_name is not None:
            original_name = str(original_name).strip() or None
        display_name = _nonempty_name("ColumnRef.display_name", self.display_name)
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "original_name", original_name)
        object.__setattr__(self, "display_name", display_name)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "original_name": self.original_name,
            "display_name": self.display_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ColumnRef:
        return cls(
            index=int(payload["index"]),
            original_name=(
                None
                if payload.get("original_name") is None
                else str(payload["original_name"])
            ),
            display_name=str(payload["display_name"]),
        )


@dataclass(frozen=True, slots=True)
class SourceFileInfo:
    """Immutable fingerprint of the exact source bytes used for import."""

    source_path: str
    format: str
    size_bytes: int
    mtime_ns: int
    mtime_utc: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "mtime_utc": self.mtime_utc,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True, init=False)
class ImportMapping:
    """Explicit index-based source-column, header, and unit mapping."""

    time_column_index: int | str
    signal_column_index: int | str
    time_unit: str
    signal_type: str
    signal_unit: str | None
    sheet_name: str | int | None
    header_mode: HeaderMode
    header_row_index: int | None
    text_parse_config: TextParseConfig

    def __init__(
        self,
        time_column_index: int | str | None = None,
        signal_column_index: int | str | None = None,
        time_unit: str = "s",
        signal_type: str = "unknown",
        signal_unit: str | None = None,
        sheet_name: str | int | None = None,
        header_mode: HeaderMode | str = HeaderMode.AUTO,
        header_row_index: int | None = None,
        text_parse_config: TextParseConfig | Mapping[str, object] | None = None,
        *,
        time_column: int | str | None = None,
        signal_column: int | str | None = None,
    ) -> None:
        """Create a mapping; legacy name selectors remain accepted at the boundary.

        New callers should pass integer ``*_column_index`` values. String selectors
        are resolved only when unique and the returned :class:`ImportResult` always
        contains a normalized index mapping.
        """

        if time_column_index is not None and time_column is not None:
            raise TypeError("provide time_column_index, not both index and legacy name")
        if signal_column_index is not None and signal_column is not None:
            raise TypeError(
                "provide signal_column_index, not both index and legacy name"
            )
        time_selector = time_column if time_column_index is None else time_column_index
        signal_selector = (
            signal_column if signal_column_index is None else signal_column_index
        )
        time_selector = _column_selector("time_column_index", time_selector)
        signal_selector = _column_selector("signal_column_index", signal_selector)
        if time_selector == signal_selector:
            raise ValueError("time and signal must map to different columns")
        normalized_sheet = _sheet_selector(sheet_name)
        normalized_mode = _header_mode(header_mode)
        normalized_row = _header_row_index(normalized_mode, header_row_index)
        normalized_text_config = _text_parse_config(text_parse_config)
        normalized_signal_unit = signal_unit
        if normalized_signal_unit is not None:
            normalized_signal_unit = _nonempty_name(
                "signal_unit", normalized_signal_unit
            )
        object.__setattr__(self, "time_column_index", time_selector)
        object.__setattr__(self, "signal_column_index", signal_selector)
        object.__setattr__(self, "time_unit", _nonempty_name("time_unit", time_unit))
        object.__setattr__(
            self,
            "signal_type",
            _nonempty_name("signal_type", signal_type).lower(),
        )
        object.__setattr__(self, "signal_unit", normalized_signal_unit)
        object.__setattr__(self, "sheet_name", normalized_sheet)
        object.__setattr__(self, "header_mode", normalized_mode)
        object.__setattr__(self, "header_row_index", normalized_row)
        object.__setattr__(self, "text_parse_config", normalized_text_config)

    @property
    def time_column(self) -> int | str:
        """Compatibility alias for pre-index importer clients."""

        return self.time_column_index

    @property
    def signal_column(self) -> int | str:
        """Compatibility alias for pre-index importer clients."""

        return self.signal_column_index

    @property
    def is_index_resolved(self) -> bool:
        return isinstance(self.time_column_index, int) and isinstance(
            self.signal_column_index, int
        )

    @property
    def signal_unit_key(self) -> str | None:
        try:
            return validate_pair(
                self.signal_type,
                self.signal_unit or get_default_unit(self.signal_type).label,
            )
        except ValueError:
            return None

    def with_indices(self, time_index: int, signal_index: int) -> ImportMapping:
        return ImportMapping(
            time_column_index=time_index,
            signal_column_index=signal_index,
            time_unit=self.time_unit,
            signal_type=self.signal_type,
            signal_unit=self.signal_unit,
            sheet_name=self.sheet_name,
            header_mode=self.header_mode,
            header_row_index=self.header_row_index,
            text_parse_config=self.text_parse_config,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "time_column_index": self.time_column_index,
            "signal_column_index": self.signal_column_index,
            "time_unit": self.time_unit,
            "signal_type": self.signal_type,
            "signal_unit": self.signal_unit,
            "signal_unit_key": self.signal_unit_key,
            "sheet_name": self.sheet_name,
            "header_mode": self.header_mode.value,
            "header_row_index": self.header_row_index,
            "text_parse_config": self.text_parse_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ImportMapping:
        return cls(
            time_column_index=payload.get(
                "time_column_index", payload.get("time_column")
            ),
            signal_column_index=payload.get(
                "signal_column_index", payload.get("signal_column")
            ),
            time_unit=str(payload.get("time_unit", "s")),
            signal_type=str(payload.get("signal_type", "unknown")),
            signal_unit=(
                None
                if payload.get("signal_unit") is None
                else str(payload["signal_unit"])
            ),
            sheet_name=payload.get("sheet_name"),
            header_mode=str(payload.get("header_mode", HeaderMode.AUTO.value)),
            header_row_index=(
                None
                if payload.get("header_row_index") is None
                else int(payload["header_row_index"])
            ),
            text_parse_config=(
                payload.get("text_parse_config")
                if isinstance(payload.get("text_parse_config"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ImportPreview:
    """Small JSON-friendly view using the same parse as final import."""

    source: SourceFileInfo
    column_refs: tuple[ColumnRef, ...]
    rows: tuple[tuple[object, ...], ...]
    row_numbers: tuple[int, ...]
    row_count: int
    requested_header_mode: HeaderMode
    detected_header_mode: HeaderMode
    header_row_index: int | None
    header_detection_uncertain: bool = False
    header_detection_message: str | None = None
    suggested_time_column_index: int | None = None
    sheet_names: tuple[str, ...] = ()
    active_sheet: str | None = None
    requested_text_parse_config: TextParseConfig | None = None
    detected_delimiter_mode: DelimiterMode | None = None
    delimiter_detection_uncertain: bool = False
    delimiter_detection_message: str | None = None
    parser_valid: bool = True
    parsed_column_count: int = 0
    field_counts: tuple[int, ...] = ()
    consistent_row_count: int = 0
    parse_warning: str | None = None
    column_dtypes: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """Friendly, unique labels suitable for GUI combo items."""

        return tuple(ref.display_name for ref in self.column_refs)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_count, self.parsed_column_count)

    @property
    def field_count_consistency(self) -> str:
        total = len(self.field_counts)
        if total == 0:
            return "0/0 rows have 0 fields"
        expected = _modal_field_count(self.field_counts)
        return (
            f"{self.consistent_row_count}/{total} rows have "
            f"{expected} fields"
        )

    def column_ref(self, index: int) -> ColumnRef:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("column index must be an integer")
        if int(index) < 0 or int(index) >= len(self.column_refs):
            raise IndexError(f"column index out of range: {index}")
        return self.column_refs[int(index)]

    def column_ref_for_display(self, display_name: str) -> ColumnRef:
        matches = [
            ref for ref in self.column_refs if ref.display_name == display_name
        ]
        if len(matches) != 1:
            raise ImportValidationError(
                f"display column {display_name!r} is not uniquely available"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.to_dict(),
            "column_refs": [ref.to_dict() for ref in self.column_refs],
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "row_numbers": list(self.row_numbers),
            "row_count": self.row_count,
            "requested_header_mode": self.requested_header_mode.value,
            "detected_header_mode": self.detected_header_mode.value,
            "header_row_index": self.header_row_index,
            "header_detection_uncertain": self.header_detection_uncertain,
            "header_detection_message": self.header_detection_message,
            "suggested_time_column_index": self.suggested_time_column_index,
            "sheet_names": list(self.sheet_names),
            "active_sheet": self.active_sheet,
            "requested_text_parse_config": (
                None
                if self.requested_text_parse_config is None
                else self.requested_text_parse_config.to_dict()
            ),
            "detected_delimiter_mode": (
                None
                if self.detected_delimiter_mode is None
                else self.detected_delimiter_mode.value
            ),
            "delimiter_detection_uncertain": self.delimiter_detection_uncertain,
            "delimiter_detection_message": self.delimiter_detection_message,
            "parser_valid": self.parser_valid,
            "parsed_column_count": self.parsed_column_count,
            "field_counts": list(self.field_counts),
            "consistent_row_count": self.consistent_row_count,
            "field_count_consistency": self.field_count_consistency,
            "parse_warning": self.parse_warning,
            "column_dtypes": list(self.column_dtypes),
        }


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Imported immutable dataset plus the exact mapping and preview evidence."""

    dataset: EcrDataset
    mapping: ImportMapping
    preview: ImportPreview

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        if not isinstance(self.mapping, ImportMapping):
            raise TypeError("mapping must be an ImportMapping")
        if not self.mapping.is_index_resolved:
            raise ValueError("ImportResult.mapping must contain column indexes")
        if not isinstance(self.preview, ImportPreview):
            raise TypeError("preview must be an ImportPreview")

    def to_dict(self) -> dict[str, object]:
        return {
            "mapping": self.mapping.to_dict(),
            "preview": self.preview.to_dict(),
            "dataset": {
                "raw_time": self.dataset.raw_time.tolist(),
                "raw_signal": self.dataset.raw_signal.tolist(),
                "metadata": _json_value(self.dataset.metadata),
                "history": [record.to_dict() for record in self.dataset.history],
            },
        }


@dataclass(frozen=True, slots=True)
class _ParsedTable:
    frame: pd.DataFrame
    column_refs: tuple[ColumnRef, ...]
    row_numbers: tuple[int, ...]
    sheet_names: tuple[str, ...]
    active_sheet: str | None
    requested_header_mode: HeaderMode
    detected_header_mode: HeaderMode
    header_row_index: int | None
    header_detection_uncertain: bool
    header_detection_message: str | None
    suggested_time_column_index: int | None
    requested_text_parse_config: TextParseConfig | None
    detected_delimiter_mode: DelimiterMode | None
    delimiter_detection_uncertain: bool
    delimiter_detection_message: str | None
    parser_valid: bool
    field_counts: tuple[int, ...]
    consistent_row_count: int
    parse_warning: str | None


@dataclass(frozen=True, slots=True)
class _RawReadResult:
    frame: pd.DataFrame
    row_numbers: tuple[int, ...]
    sheet_names: tuple[str, ...]
    active_sheet: str | None
    requested_text_parse_config: TextParseConfig | None
    detected_delimiter_mode: DelimiterMode | None
    delimiter_detection_uncertain: bool
    delimiter_detection_message: str | None
    parser_valid: bool
    field_counts: tuple[int, ...]
    consistent_row_count: int
    parse_warning: str | None


@dataclass(frozen=True, slots=True)
class _TextCandidate:
    mode: DelimiterMode
    rows: tuple[tuple[object, ...], ...]
    field_counts: tuple[int, ...]
    expected_field_count: int
    consistent_row_count: int
    parser_valid: bool
    evidence_rows: int


def preview_tabular_file(
    source_path: str | Path,
    *,
    max_rows: int = 20,
    sheet_name: str | int | None = None,
    header_mode: HeaderMode | str = HeaderMode.AUTO,
    header_row_index: int | None = None,
    text_parse_config: TextParseConfig | Mapping[str, object] | None = None,
) -> ImportPreview:
    """Read a bounded display preview without inferring scientific mappings."""

    if isinstance(max_rows, bool) or not isinstance(max_rows, Integral):
        raise TypeError("max_rows must be an integer")
    if int(max_rows) <= 0:
        raise ValueError("max_rows must be positive")
    path = _validated_source_path(source_path)
    source = _source_info(path)
    parsed = _parse_table(
        path,
        sheet_name=sheet_name,
        header_mode=header_mode,
        header_row_index=header_row_index,
        text_parse_config=text_parse_config,
        allow_invalid_parser=True,
    )
    return _preview_from_parsed(source, parsed, max_rows=int(max_rows))


def import_tabular_file(
    source_path: str | Path,
    mapping: ImportMapping,
    *,
    expected_preview: ImportPreview | None = None,
    preview_rows: int = 20,
) -> ImportResult:
    """Create an EcrDataset from explicitly mapped, untouched raw columns."""

    if not isinstance(mapping, ImportMapping):
        raise TypeError("mapping must be an ImportMapping")
    path = _validated_source_path(source_path)
    source = _source_info(path)
    if expected_preview is not None:
        if not isinstance(expected_preview, ImportPreview):
            raise TypeError("expected_preview must be an ImportPreview or None")
        _verify_unchanged_source(source, expected_preview.source)

    parsed = _parse_table(
        path,
        sheet_name=mapping.sheet_name,
        header_mode=mapping.header_mode,
        header_row_index=mapping.header_row_index,
        text_parse_config=mapping.text_parse_config,
        allow_invalid_parser=False,
    )
    preview = _preview_from_parsed(source, parsed, max_rows=preview_rows)
    if expected_preview is not None:
        _verify_same_parse(preview, expected_preview)

    time_ref = _resolve_column_ref(
        parsed.column_refs, mapping.time_column_index, "time"
    )
    signal_ref = _resolve_column_ref(
        parsed.column_refs, mapping.signal_column_index, "signal"
    )
    if time_ref.index == signal_ref.index:
        raise ImportValidationError("time and signal must map to different columns")
    normalized_mapping = mapping.with_indices(time_ref.index, signal_ref.index)
    source_unit = normalized_mapping.signal_unit or get_default_unit(normalized_mapping.signal_type).label
    try:
        validate_pair(normalized_mapping.signal_type, source_unit)
    except ValueError as exc:
        raise ImportValidationError(str(exc)) from exc
    times = _numeric_column(parsed, time_ref, "time")
    signal = _numeric_column(parsed, signal_ref, "signal")
    _validate_raw_vectors(times, signal, parsed.row_numbers)

    metadata: dict[str, object] = {
        "dataset_name": path.stem,
        "source_path": source.source_path,
        "source_format": source.format,
        "source_size_bytes": source.size_bytes,
        "source_mtime_ns": source.mtime_ns,
        "source_mtime_utc": source.mtime_utc,
        "source_sha256": source.sha256,
        "source_sheet": parsed.active_sheet,
        "source_header_mode_requested": parsed.requested_header_mode.value,
        "source_header_mode_detected": parsed.detected_header_mode.value,
        "source_header_row_index": parsed.header_row_index,
        "source_text_parse_config": (
            None
            if parsed.requested_text_parse_config is None
            else parsed.requested_text_parse_config.to_dict()
        ),
        "source_detected_delimiter_mode": (
            None
            if parsed.detected_delimiter_mode is None
            else parsed.detected_delimiter_mode.value
        ),
        "source_field_count_consistency": (
            f"{parsed.consistent_row_count}/{len(parsed.field_counts)}"
        ),
        "source_time_column_index": time_ref.index,
        "source_signal_column_index": signal_ref.index,
        "source_time_column": time_ref.original_name or time_ref.display_name,
        "source_signal_column": signal_ref.original_name or signal_ref.display_name,
        "source_time_column_display": time_ref.display_name,
        "source_signal_column_display": signal_ref.display_name,
    }
    dataset = EcrDataset.from_raw(
        times,
        signal,
        time_unit=normalized_mapping.time_unit,
        signal_type=normalized_mapping.signal_type,
        signal_unit=normalized_mapping.signal_unit,
        metadata=metadata,
    )
    return ImportResult(
        dataset=dataset,
        mapping=normalized_mapping,
        preview=preview,
    )


def _validated_source_path(source_path: str | Path) -> Path:
    try:
        path = Path(source_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as exc:
        raise ImportValidationError(f"source file is not accessible: {source_path}") from exc
    if not path.is_file():
        raise ImportValidationError(f"source path is not a file: {path}")
    if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ImportValidationError(
            "unsupported source format; select a CSV, TXT, XLSX, or XLS file"
        )
    return path


def _source_info(path: Path) -> SourceFileInfo:
    stat = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return SourceFileInfo(
        source_path=str(path),
        format=path.suffix.lower().removeprefix("."),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        mtime_utc=mtime,
        sha256=digest.hexdigest(),
    )


def _parse_table(
    path: Path,
    *,
    sheet_name: str | int | None,
    header_mode: HeaderMode | str,
    header_row_index: int | None,
    text_parse_config: TextParseConfig | Mapping[str, object] | None,
    allow_invalid_parser: bool,
) -> _ParsedTable:
    requested_mode = _header_mode(header_mode)
    requested_row = _header_row_index(requested_mode, header_row_index)
    raw_result = _read_raw_frame(path, sheet_name, text_parse_config)
    if not raw_result.parser_valid and not allow_invalid_parser:
        raise ImportValidationError(
            raw_result.parse_warning or "text parser produced ragged rows"
        )
    raw = raw_result.frame
    detected_mode = requested_mode
    detected_row = requested_row
    uncertain = False
    message = None
    if requested_mode is HeaderMode.AUTO:
        detected_mode, uncertain = _detect_header_mode(raw)
        detected_row = 0 if detected_mode is HeaderMode.FIRST_ROW else None
        message = _UNCERTAIN_HEADER_MESSAGE if uncertain else None
    elif requested_mode is HeaderMode.FIRST_ROW:
        detected_row = 0
    elif requested_mode is HeaderMode.NONE:
        detected_row = None

    if detected_row is not None:
        if detected_row >= raw.shape[0]:
            raise ImportValidationError(
                f"header row index {detected_row} is outside the source"
            )
        header_values = tuple(raw.iloc[detected_row, :].tolist())
        data_start = detected_row + 1
    else:
        header_values = tuple(None for _ in range(raw.shape[1]))
        data_start = 0
    frame = raw.iloc[data_start:, :].copy().reset_index(drop=True)
    if frame.empty:
        raise ImportValidationError("source contains no data rows after header parsing")
    frame.columns = list(range(frame.shape[1]))
    frame = _coerce_numeric_columns(frame)
    refs = _column_refs(header_values)
    row_numbers = raw_result.row_numbers[data_start:]
    return _ParsedTable(
        frame=frame,
        column_refs=refs,
        row_numbers=row_numbers,
        sheet_names=raw_result.sheet_names,
        active_sheet=raw_result.active_sheet,
        requested_header_mode=requested_mode,
        detected_header_mode=detected_mode,
        header_row_index=detected_row,
        header_detection_uncertain=uncertain,
        header_detection_message=message,
        suggested_time_column_index=_suggest_time_column(frame),
        requested_text_parse_config=raw_result.requested_text_parse_config,
        detected_delimiter_mode=raw_result.detected_delimiter_mode,
        delimiter_detection_uncertain=raw_result.delimiter_detection_uncertain,
        delimiter_detection_message=raw_result.delimiter_detection_message,
        parser_valid=raw_result.parser_valid,
        field_counts=raw_result.field_counts,
        consistent_row_count=raw_result.consistent_row_count,
        parse_warning=raw_result.parse_warning,
    )


def _read_raw_frame(
    path: Path,
    sheet_name: str | int | None,
    text_parse_config: TextParseConfig | Mapping[str, object] | None,
) -> _RawReadResult:
    """Read every source row as data; header interpretation happens later."""

    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                workbook_context = pd.ExcelFile(path, engine="xlrd" if path.suffix.lower() == ".xls" else "openpyxl")
            except ImportError as exc:
                raise ImportValidationError(
                    "Excel input requires openpyxl for XLSX or xlrd for XLS. "
                    "Use the complete PyECR package, including its bundled runtime."
                ) from exc
            selected: str | int = 0 if sheet_name is None else sheet_name
            with workbook_context as workbook:
                sheet_names = tuple(str(name) for name in workbook.sheet_names)
                frame = pd.read_excel(workbook, sheet_name=selected, header=None)
            active_sheet = (
                sheet_names[selected]
                if isinstance(selected, int) and 0 <= selected < len(sheet_names)
                else str(selected)
            )
            if frame.empty or frame.shape[1] == 0:
                raise ImportValidationError("source contains no rows or columns")
            count = int(frame.shape[1])
            field_counts = (count,) * int(frame.shape[0])
            return _RawReadResult(
                frame=frame,
                row_numbers=tuple(range(1, int(frame.shape[0]) + 1)),
                sheet_names=sheet_names,
                active_sheet=active_sheet,
                requested_text_parse_config=None,
                detected_delimiter_mode=None,
                delimiter_detection_uncertain=False,
                delimiter_detection_message=None,
                parser_valid=True,
                field_counts=field_counts,
                consistent_row_count=len(field_counts),
                parse_warning=None,
            )
        else:
            if sheet_name is not None:
                raise ImportValidationError("sheet_name is valid only for XLSX or XLS workbooks")
            config = _text_parse_config(text_parse_config)
            content = path.read_text(encoding="utf-8-sig")
            line_pairs = tuple(
                (line_number, line)
                for line_number, line in enumerate(content.splitlines(), start=1)
                if line.strip()
            )
            if not line_pairs:
                raise ImportValidationError("source contains no nonempty rows")
            return _read_text_rows(line_pairs, config)
    except ImportValidationError:
        raise
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ImportValidationError(
            f"could not read {path.name}: {type(exc).__name__}: {exc}"
        ) from exc


def _read_text_rows(
    line_pairs: tuple[tuple[int, str], ...],
    config: TextParseConfig,
) -> _RawReadResult:
    lines = tuple(line for _, line in line_pairs)
    detected_mode = config.mode
    uncertain = False
    message = None
    if config.mode is DelimiterMode.AUTO:
        detected_mode, uncertain = _detect_delimiter_mode(
            lines[: config.auto_sample_rows]
        )
        message = _UNCERTAIN_DELIMITER_MESSAGE if uncertain else None
    candidate = _text_candidate(lines, detected_mode, config.custom_delimiter)
    width = max(candidate.field_counts)
    padded_rows = tuple(
        row + (None,) * (width - len(row)) for row in candidate.rows
    )
    frame = pd.DataFrame(padded_rows, columns=range(width))
    warning = None
    if not candidate.parser_valid:
        differing = [
            f"row {line_pairs[index][0]}: {count} fields"
            for index, count in enumerate(candidate.field_counts)
            if count != candidate.expected_field_count
        ]
        detail = "; ".join(differing[:10])
        warning = (
            "ragged rows detected; parser is invalid. "
            f"Expected {candidate.expected_field_count} fields from the modal row "
            f"count, but {len(differing)} row(s) differ"
            f"{': ' + detail if detail else '.'}"
        )
    return _RawReadResult(
        frame=frame,
        row_numbers=tuple(line_number for line_number, _ in line_pairs),
        sheet_names=(),
        active_sheet=None,
        requested_text_parse_config=config,
        detected_delimiter_mode=detected_mode,
        delimiter_detection_uncertain=uncertain,
        delimiter_detection_message=message,
        parser_valid=candidate.parser_valid,
        field_counts=candidate.field_counts,
        consistent_row_count=candidate.consistent_row_count,
        parse_warning=warning,
    )


def _detect_delimiter_mode(
    sample_lines: tuple[str, ...],
) -> tuple[DelimiterMode, bool]:
    modes = (
        DelimiterMode.TAB,
        DelimiterMode.COMMA,
        DelimiterMode.SEMICOLON,
        DelimiterMode.WHITESPACE,
        DelimiterMode.FIXED_WIDTH,
    )
    candidates = tuple(_text_candidate(sample_lines, mode, None) for mode in modes)
    high_confidence = [
        candidate
        for candidate in candidates
        if candidate.parser_valid
        and candidate.expected_field_count >= 2
        and candidate.evidence_rows >= max(1, int(np.ceil(len(sample_lines) * 0.8)))
    ]
    if high_confidence:
        return high_confidence[0].mode, False
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.evidence_rows,
            candidate.consistent_row_count,
            candidate.expected_field_count,
            -modes.index(candidate.mode),
        ),
        reverse=True,
    )
    return ranked[0].mode, True


def _text_candidate(
    lines: tuple[str, ...],
    mode: DelimiterMode,
    custom_delimiter: str | None,
) -> _TextCandidate:
    if mode is DelimiterMode.AUTO:
        raise ValueError("AUTO must be resolved before parsing text rows")
    if mode is DelimiterMode.FIXED_WIDTH:
        frame = pd.read_fwf(
            StringIO("\n".join(lines)),
            header=None,
            infer_nrows=min(100, len(lines)),
        )
        rows = tuple(
            tuple(row)
            for row in frame.itertuples(index=False, name=None)
        )
        field_counts = (int(frame.shape[1]),) * len(rows)
        evidence = sum(bool(re.search(r"\S {2,}\S", line)) for line in lines)
        expected = int(frame.shape[1])
        return _TextCandidate(
            mode=mode,
            rows=rows,
            field_counts=field_counts,
            expected_field_count=expected,
            consistent_row_count=len(rows),
            parser_valid=bool(rows) and expected >= 1,
            evidence_rows=evidence,
        )

    if mode is DelimiterMode.WHITESPACE:
        rows = tuple(tuple(line.strip().split()) for line in lines)
        evidence = sum(bool(re.search(r"\S\s+\S", line.strip())) for line in lines)
    else:
        delimiter = {
            DelimiterMode.TAB: "\t",
            DelimiterMode.COMMA: ",",
            DelimiterMode.SEMICOLON: ";",
            DelimiterMode.CUSTOM: custom_delimiter,
        }[mode]
        if delimiter is None:
            raise ValueError("CUSTOM mode requires custom_delimiter")
        rows = tuple(
            tuple(next(csv.reader([line], delimiter=delimiter))) for line in lines
        )
        evidence = sum(delimiter in line for line in lines)
    field_counts = tuple(len(row) for row in rows)
    expected = _modal_field_count(field_counts)
    consistent = sum(count == expected for count in field_counts)
    return _TextCandidate(
        mode=mode,
        rows=rows,
        field_counts=field_counts,
        expected_field_count=expected,
        consistent_row_count=consistent,
        parser_valid=bool(rows) and consistent == len(rows),
        evidence_rows=evidence,
    )


def _detect_header_mode(raw: pd.DataFrame) -> tuple[HeaderMode, bool]:
    """Conservatively detect obvious headers and preserve data when uncertain."""

    if raw.shape[0] < 2:
        return HeaderMode.NONE, True
    first = raw.iloc[0, :]
    following = raw.iloc[1 : min(raw.shape[0], 21), :]
    nonblank = [index for index, value in enumerate(first) if not _is_blank(value)]
    if not nonblank:
        return HeaderMode.NONE, True

    first_numeric = {index for index in nonblank if _is_numeric_value(first.iloc[index])}
    first_text = set(nonblank) - first_numeric
    numeric_following: set[int] = set()
    for index in nonblank:
        values = [value for value in following.iloc[:, index] if not _is_blank(value)]
        if values and sum(_is_numeric_value(value) for value in values) / len(values) >= 0.8:
            numeric_following.add(index)

    if first_numeric == set(nonblank) and numeric_following == set(nonblank):
        return HeaderMode.NONE, False

    strong_header = first_text.intersection(numeric_following)
    required = max(1, int(np.ceil(len(nonblank) * 0.5)))
    if not first_numeric and len(strong_header) >= required:
        return HeaderMode.FIRST_ROW, False
    return HeaderMode.NONE, True


def _column_refs(header_values: tuple[object, ...]) -> tuple[ColumnRef, ...]:
    originals = tuple(_original_header_name(value) for value in header_values)
    bases = tuple(
        f"Column {index + 1}"
        if name is None or _UNNAMED_HEADER.fullmatch(name)
        else name
        for index, name in enumerate(originals)
    )
    counts = {base: bases.count(base) for base in set(bases)}
    return tuple(
        ColumnRef(
            index=index,
            original_name=originals[index],
            display_name=(
                f"{base} [Column {index + 1}]" if counts[base] > 1 else base
            ),
        )
        for index, base in enumerate(bases)
    )


def _coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for index in range(result.shape[1]):
        series = result.iloc[:, index].replace(
            [r"^\s*$", r"^\s*(?i:nan)\s*$"],
            np.nan,
            regex=True,
        )
        result[index] = series
        converted = pd.to_numeric(series, errors="coerce")
        source_nonmissing = ~series.isna()
        if bool((~converted[source_nonmissing].isna()).all()):
            result[index] = converted
    return result


def _suggest_time_column(frame: pd.DataFrame) -> int | None:
    for index in range(frame.shape[1]):
        try:
            values = pd.to_numeric(frame.iloc[:, index], errors="raise").to_numpy(
                dtype=np.float64,
                copy=False,
            )
        except (TypeError, ValueError):
            continue
        if (
            values.size >= 2
            and np.isfinite(values).all()
            and np.all(np.diff(values) > 0)
        ):
            return index
    return None


def _preview_from_parsed(
    source: SourceFileInfo,
    parsed: _ParsedTable,
    *,
    max_rows: int,
) -> ImportPreview:
    limited = parsed.frame.head(max_rows)
    rows = tuple(
        tuple(_preview_cell(value) for value in row)
        for row in limited.itertuples(index=False, name=None)
    )
    return ImportPreview(
        source=source,
        column_refs=parsed.column_refs,
        rows=rows,
        row_numbers=parsed.row_numbers[: len(rows)],
        row_count=int(parsed.frame.shape[0]),
        requested_header_mode=parsed.requested_header_mode,
        detected_header_mode=parsed.detected_header_mode,
        header_row_index=parsed.header_row_index,
        header_detection_uncertain=parsed.header_detection_uncertain,
        header_detection_message=parsed.header_detection_message,
        suggested_time_column_index=parsed.suggested_time_column_index,
        sheet_names=parsed.sheet_names,
        active_sheet=parsed.active_sheet,
        requested_text_parse_config=parsed.requested_text_parse_config,
        detected_delimiter_mode=parsed.detected_delimiter_mode,
        delimiter_detection_uncertain=parsed.delimiter_detection_uncertain,
        delimiter_detection_message=parsed.delimiter_detection_message,
        parser_valid=parsed.parser_valid,
        parsed_column_count=int(parsed.frame.shape[1]),
        field_counts=parsed.field_counts,
        consistent_row_count=parsed.consistent_row_count,
        parse_warning=parsed.parse_warning,
        column_dtypes=tuple(str(dtype) for dtype in parsed.frame.dtypes),
    )


def _preview_cell(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not isfinite(value):
        return str(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _resolve_column_ref(
    refs: tuple[ColumnRef, ...],
    selector: int | str,
    role: str,
) -> ColumnRef:
    if isinstance(selector, int):
        if selector < 0 or selector >= len(refs):
            raise ImportValidationError(
                f"missing {role} column index {selector}; source has {len(refs)} columns"
            )
        return refs[selector]
    matches = [
        ref
        for ref in refs
        if selector in {ref.original_name, ref.display_name}
    ]
    if not matches:
        available = ", ".join(ref.display_name for ref in refs)
        raise ImportValidationError(
            f"missing {role} column {selector!r}; available columns: {available}"
        )
    if len(matches) > 1:
        raise ImportValidationError(
            f"ambiguous {role} column name {selector!r}; select by column index"
        )
    return matches[0]


def _numeric_column(
    parsed: _ParsedTable,
    ref: ColumnRef,
    role: str,
) -> np.ndarray:
    try:
        values = pd.to_numeric(
            parsed.frame.iloc[:, ref.index], errors="raise"
        ).to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ImportValidationError(
            f"{role} column {ref.display_name!r} contains non-numeric values: {exc}"
        ) from exc
    invalid = np.flatnonzero(~np.isfinite(values))
    if invalid.size:
        rows = ", ".join(str(parsed.row_numbers[int(index)]) for index in invalid[:10])
        raise ImportValidationError(
            f"{role} column {ref.display_name!r} contains NaN or Inf at source rows: {rows}"
        )
    return values


def _validate_raw_vectors(
    times: np.ndarray,
    signal: np.ndarray,
    row_numbers: tuple[int, ...],
) -> None:
    if times.size != signal.size:
        raise ImportValidationError("mapped time and signal columns have different lengths")
    if times.size == 0:
        raise ImportValidationError("mapped columns contain no observations")
    differences = np.diff(times)
    duplicates = np.flatnonzero(differences == 0.0)
    if duplicates.size:
        pairs = ", ".join(
            f"{row_numbers[int(index)]}/{row_numbers[int(index) + 1]}"
            for index in duplicates[:10]
        )
        raise ImportValidationError(
            f"duplicate time values at source row pairs: {pairs}"
        )
    decreases = np.flatnonzero(differences < 0.0)
    if decreases.size:
        pairs = ", ".join(
            f"{row_numbers[int(index)]}/{row_numbers[int(index) + 1]}"
            for index in decreases[:10]
        )
        raise ImportValidationError(
            "time values must be strictly increasing; decreases occur at "
            f"source row pairs: {pairs}"
        )


def _verify_same_parse(actual: ImportPreview, expected: ImportPreview) -> None:
    same = (
        actual.active_sheet == expected.active_sheet
        and actual.column_refs == expected.column_refs
        and actual.requested_header_mode is expected.requested_header_mode
        and actual.detected_header_mode is expected.detected_header_mode
        and actual.header_row_index == expected.header_row_index
        and actual.row_count == expected.row_count
        and actual.rows[: len(expected.rows)] == expected.rows
        and actual.row_numbers[: len(expected.row_numbers)] == expected.row_numbers
        and actual.requested_text_parse_config
        == expected.requested_text_parse_config
        and actual.detected_delimiter_mode is expected.detected_delimiter_mode
        and actual.parser_valid == expected.parser_valid
        and actual.field_counts == expected.field_counts
    )
    if not same:
        raise ImportValidationError(
            "selected sheet, header mode, or parsed columns differ from the preview; "
            "preview the source again"
        )


def _verify_unchanged_source(actual: SourceFileInfo, expected: SourceFileInfo) -> None:
    if actual.source_path != expected.source_path:
        raise ImportValidationError("selected source path differs from the preview")
    if (
        actual.size_bytes != expected.size_bytes
        or actual.mtime_ns != expected.mtime_ns
        or actual.sha256 != expected.sha256
    ):
        raise ImportValidationError(
            "source file changed after preview; preview it again before import"
        )


def _column_selector(name: str, value: object) -> int | str:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{name} must be a nonnegative integer column index")
    if isinstance(value, Integral):
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be nonnegative")
        return result
    if isinstance(value, str):
        return _nonempty_name(name, value)
    raise TypeError(f"{name} must be a nonnegative integer column index")


def _sheet_selector(value: object) -> str | int | None:
    if isinstance(value, bool) or (
        value is not None and not isinstance(value, (str, Integral))
    ):
        raise TypeError("sheet_name must be a string, nonnegative integer, or None")
    if isinstance(value, Integral):
        if int(value) < 0:
            raise ValueError("sheet_name index must be nonnegative")
        return int(value)
    if isinstance(value, str):
        return _nonempty_name("sheet_name", value)
    return None


def _header_mode(value: HeaderMode | str) -> HeaderMode:
    if isinstance(value, HeaderMode):
        return value
    if not isinstance(value, str):
        raise TypeError("header_mode must be a HeaderMode or string")
    normalized = value.strip().lower()
    try:
        return HeaderMode(normalized)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in HeaderMode)
        raise ValueError(f"unknown header_mode {value!r}; expected one of: {valid}") from exc


def _delimiter_mode(value: DelimiterMode | str) -> DelimiterMode:
    if isinstance(value, DelimiterMode):
        return value
    if not isinstance(value, str):
        raise TypeError("delimiter mode must be a DelimiterMode or string")
    normalized = value.strip().lower()
    try:
        return DelimiterMode(normalized)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in DelimiterMode)
        raise ValueError(
            f"unknown delimiter mode {value!r}; expected one of: {valid}"
        ) from exc


def _text_parse_config(
    value: TextParseConfig | Mapping[str, object] | None,
) -> TextParseConfig:
    if value is None:
        return TextParseConfig()
    if isinstance(value, TextParseConfig):
        return value
    if isinstance(value, Mapping):
        return TextParseConfig.from_dict(value)
    raise TypeError("text_parse_config must be TextParseConfig, mapping, or None")


def _modal_field_count(field_counts: tuple[int, ...]) -> int:
    if not field_counts:
        return 0
    counts = Counter(field_counts)
    highest_frequency = max(counts.values())
    return min(
        count for count, frequency in counts.items() if frequency == highest_frequency
    )


def _header_row_index(mode: HeaderMode, value: int | None) -> int | None:
    if mode is not HeaderMode.ROW_INDEX:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("header_row_index is required for ROW_INDEX mode")
    if int(value) < 0:
        raise ValueError("header_row_index must be nonnegative")
    return int(value)


def _original_header_name(value: object) -> str | None:
    if _is_blank(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_numeric_value(value: object) -> bool:
    if _is_blank(value) or isinstance(value, bool):
        return False
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(converted)


def _nonempty_name(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
