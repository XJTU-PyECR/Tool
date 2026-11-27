"""Reusable status error GUI controls."""

from __future__ import annotations
from pyecr.gui.theme import TOKENS, status_color

from pyecr.gui.components.dependencies import (
    Any, json, traceback, ModelFormValues, ModelParameterValue,
    GUI_PAGES, GuiPage, DelimiterMode, HeaderMode, ImportMapping,
    TextParseConfig, EcrDataset, PREPROCESSING_OPERATIONS, ProcessingStep,
    FitPlotData, BatchAnalysisResult, GlobalAnalysisResult,
    DctAnalysisConfig, DctPresentation,
    _TIME_UNIT_CHOICES, _SIGNAL_TYPE_CHOICES, _HEADER_MODE_CHOICES,
    _TEXT_PARSE_MODE_CHOICES, _DATA_PREVIEW_CONTAINER_TAG,
    _DATA_PREVIEW_TABLE_TAG, _PROCESSING_PARAMETER_EXAMPLES,
)
from pyecr.gui.components.scientific_input import (
    _model_parameter_tag, _parse_numeric_sequence, _file_dialog_path,
    _directory_dialog_path,
)
from pyecr.gui.components.results_table import (
    _render_bi_table, _render_parameter_diagnostics_table,
    _render_correlation_table, _render_multistart_table,
    _format_confidence_interval,
)
from pyecr.gui.components.plot_tabs import (
    _update_processed_plot, _replace_data_preview_table,
)
from pyecr.gui.components.tags import WidgetTags


def build_error_panel(dpg: Any, copy_callback: Any) -> None:
    """Create the one persistent, copyable traceback panel."""

    dpg.add_separator()
    dpg.add_text("Errors and tracebacks")
    dpg.add_input_text(
        tag=WidgetTags.ERROR_CONSOLE,
        default_value="No errors.",
        multiline=True,
        readonly=True,
        width=-1,
        height=TOKENS.heights.error_console,
    )
    dpg.add_button(label="Copy error text", callback=copy_callback)


def refresh_error_panel(dpg: Any, text: str) -> None:
    dpg.set_value(WidgetTags.ERROR_CONSOLE, text)


class StatusErrorMixin:
    def _build_current_error_panel(self) -> None:
        dpg = self._require_dpg()
        with dpg.group(tag="pyecr_error_current_group"):
            dpg.add_text("Current task error", tag="pyecr_current_task_error_label")
            dpg.add_input_text(
                tag="pyecr_current_task_error",
                default_value="No current task error.",
                multiline=True,
                readonly=True,
                width=-1,
                height=TOKENS.heights.status_small,
            )
        with dpg.group(tag="pyecr_warning_current_group"):
            dpg.add_text("Current warning", tag="pyecr_current_warning_label")
            dpg.add_input_text(
                tag="pyecr_current_warning",
                default_value="No current warning.",
                multiline=True,
                readonly=True,
                width=-1,
                height=TOKENS.heights.status_small,
            )
        with dpg.group(tag="pyecr_history_group"):
            dpg.add_text("History")
            dpg.add_input_text(
                tag="pyecr_activity_history",
                default_value="No warnings or errors.",
                multiline=True,
                readonly=True,
                width=-1,
                height=TOKENS.heights.history,
            )
        with dpg.collapsing_header(label="Developer traceback", tag="pyecr_developer_traceback_header", default_open=False):
            build_error_panel(dpg, self._copy_errors)

    def _refresh_current_error_panel(self) -> None:
        dpg = self._require_dpg()
        active = self.controller.state.active_page
        if active in {GuiPage.DATA, GuiPage.MODEL, GuiPage.FIT, GuiPage.DIAGNOSTICS}:
            relevant = {GuiPage.DATA, GuiPage.MODEL, GuiPage.FIT, GuiPage.DIAGNOSTICS}
        else:
            relevant = {active}
        active_errors = [
            error
            for page, error in self.controller.state.current_errors.items()
            if page in relevant
        ]
        dpg.set_value("pyecr_current_task_error", (
            f"{active_errors[-1].exception_type}: {active_errors[-1].message}"
            if active_errors else "No current task error."
        ))
        dpg.configure_item(
            "pyecr_current_task_error_label",
            color=status_color("error" if active_errors else "idle"),
        )
        notices = [notice for notice in self.controller.state.notices if notice.page in relevant]
        task_warnings = [
            (page, self.controller.state.tasks[page]) for page in relevant
            if self.controller.state.tasks[page].status in {"warning", "cancelled"}
        ]
        warning_text = (
            notices[-1].message if notices else
            task_warnings[-1][1].message if task_warnings else
            "No current warning."
        )
        dpg.set_value("pyecr_current_warning", warning_text)
        dpg.configure_item(
            "pyecr_current_warning_label",
            color=status_color("warning" if warning_text != "No current warning." else "idle"),
        )
        dpg.set_value(
            "pyecr_activity_history", self.controller.state.activity_history_text()
        )
        refresh_error_panel(dpg, self.controller.state.copyable_error_text())

    def _copy_errors(self, sender: Any, app_data: Any, user_data: Any) -> None:
        self._require_dpg().set_clipboard_text(
            self.controller.state.copyable_error_text()
        )

    def _refresh_summaries(self) -> None:
        if not self._built:
            return
        dpg = self._require_dpg()
        from pyecr.model_display import display_model_text
        for page in GUI_PAGES:
            tag = f"pyecr_summary_{page.value.lower()}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, display_model_text(self.controller.state.status_summary(page)))
