"""BatchViewMixin: page builders, callbacks, and refresh renderers."""

from __future__ import annotations
from pyecr.gui.theme import TOKENS

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
    format_scientific,
)
from pyecr.gui.components.plot_tabs import (
    _update_processed_plot, _replace_data_preview_table,
)
from pyecr.gui.components.action_state import add_action_tooltip


from pyecr.gui.components.batch_layout import BatchLayoutMixin


class BatchViewMixin(BatchLayoutMixin):
    def _build_batch_area(self) -> None:
        self._build_batch_layout()

    def _show_batch_project_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        self._require_dpg().show_item("pyecr_batch_project_dialog")

    def _batch_input_values(self) -> tuple[str, float, str, float, bool, bool]:
        dpg = self._require_dpg()
        return (
            str(dpg.get_value("pyecr_batch_record_id")).strip(),
            float(dpg.get_value("pyecr_batch_temperature")),
            str(dpg.get_value("pyecr_batch_temperature_unit")),
            float(dpg.get_value("pyecr_batch_noise_weight")),
            bool(dpg.get_value("pyecr_batch_fit_baseline")),
            bool(dpg.get_value("pyecr_batch_fit_amplitude")),
        )

    def _add_current_batch_dataset(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            record_id, temperature, unit, weight, fit_baseline, fit_amplitude = (
                self._batch_input_values()
            )
            if not record_id:
                record_id = (
                    self.controller.state.dataset_name
                    or f"dataset-{len(self.controller.session.batch_inputs) + 1}"
                )
            self.controller.add_current_dataset_to_batch(
                record_id=record_id,
                temperature=temperature,
                temperature_unit=unit,
                noise_weight=weight,
                fit_baseline=fit_baseline,
                fit_amplitude=fit_amplitude,
            )
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.BATCH, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _add_batch_project_from_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        try:
            record_id, temperature, unit, weight, fit_baseline, fit_amplitude = (
                self._batch_input_values()
            )
            self.controller.add_project_to_batch(
                _directory_dialog_path(app_data),
                record_id=record_id or None,
                temperature=temperature,
                temperature_unit=unit,
                noise_weight=weight,
                fit_baseline=fit_baseline,
                fit_amplitude=fit_amplitude,
            )
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.BATCH, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _clear_batch_inputs(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.clear_batch_inputs()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.BATCH, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _run_batch_independent(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.run_batch_independent()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.BATCH, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _run_global_analysis(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.run_global_analysis()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.BATCH, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _cancel_batch(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            self.controller.cancel_page(GuiPage.BATCH)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.BATCH, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _render_batch_results(self) -> None:
        self._refresh_batch_layout()
