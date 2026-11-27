"""Application service for previewing and importing raw ECR files."""

from __future__ import annotations

from pathlib import Path

from pyecr.io.importer import (
    HeaderMode,
    ImportMapping,
    ImportPreview,
    ImportResult,
    TextParseConfig,
    import_tabular_file,
    preview_tabular_file,
)

__all__ = ["ImportService"]


class ImportService:
    """Keep dataframe and workbook dependencies outside GUI code."""

    def preview(
        self,
        source_path: str | Path,
        *,
        max_rows: int = 20,
        sheet_name: str | int | None = None,
        header_mode: HeaderMode | str = HeaderMode.AUTO,
        header_row_index: int | None = None,
        text_parse_config: TextParseConfig | None = None,
    ) -> ImportPreview:
        return preview_tabular_file(
            source_path,
            max_rows=max_rows,
            sheet_name=sheet_name,
            header_mode=header_mode,
            header_row_index=header_row_index,
            text_parse_config=text_parse_config,
        )

    def import_dataset(
        self,
        source_path: str | Path,
        mapping: ImportMapping,
        *,
        expected_preview: ImportPreview | None = None,
    ) -> ImportResult:
        result = import_tabular_file(
            source_path,
            mapping,
            expected_preview=expected_preview,
        )
        from pyecr.services.simulation_transfer import recognize_simulation_csv
        return recognize_simulation_csv(result)
