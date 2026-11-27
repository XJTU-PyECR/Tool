"""One namespace for stable DearPyGui aliases; transient children use UUIDs."""

from __future__ import annotations


class WidgetTags:
    PREFIX = "pyecr_"
    PRIMARY_WINDOW = "pyecr_primary_window"
    PAGES = "pyecr_pages"
    ERROR_CONSOLE = "pyecr_error_console"
    DATA_PREVIEW_CONTAINER = "pyecr_data_preview_container"
    DATA_PREVIEW_TABLE = "pyecr_data_preview_table"

    @classmethod
    def page(cls, page: str) -> str:
        return f"{cls.PREFIX}tab_{page.lower()}"

    @classmethod
    def summary(cls, page: str) -> str:
        return f"{cls.PREFIX}summary_{page.lower()}"

    @classmethod
    def progress(cls, page: str) -> str:
        return f"{cls.PREFIX}progress_{page.lower()}"

    @classmethod
    def task_status(cls, page: str) -> str:
        return f"{cls.PREFIX}task_status_{page.lower()}"

    @classmethod
    def run(cls, page: str) -> str:
        return f"{cls.PREFIX}run_{page.lower()}"

    @classmethod
    def model_parameter(cls, name: str, field: str) -> str:
        return f"{cls.PREFIX}model_parameter_{name}_{field}"
