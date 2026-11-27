"""ReportsViewMixin: page builders, callbacks, and refresh renderers."""

from __future__ import annotations
from pyecr.gui.theme import TOKENS
from pyecr.gui.components.report_layout import ReportLayoutMixin

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
from pyecr.gui.components.action_state import (
    add_action_tooltip, set_action_availability,
)


class ReportsViewMixin(ReportLayoutMixin):
    def _build_report_area(self) -> None:
        self._build_report_layout()

    def _export_options_changed(self, sender=None, app_data=None, user_data=None):
        dpg = self._require_dpg()
        value = dpg.get_value("pyecr_export_preset")
        extra = " + analysis.xlsx" if value == "excel" else " + PNG/SVG/PDF figures" if value == "publication" else ""
        dpg.set_value("pyecr_export_file_preview", "Files: legacy CSV/JSON + raw/audit/processed/fit/components/parameters/diagnostics/history/metadata/batch + dictionary/README/manifest" + extra)

    def _export_options(self):
        from pyecr.report import ExportOptions
        dpg = self._require_dpg()
        return ExportOptions(preset=dpg.get_value("pyecr_export_preset"), csv_encoding=dpg.get_value("pyecr_export_encoding"),
            csv_delimiter="\t" if dpg.get_value("pyecr_export_delimiter") == "tab" else dpg.get_value("pyecr_export_delimiter"),
            image_dpi=int(dpg.get_value("pyecr_export_dpi")), width_in=float(dpg.get_value("pyecr_export_width")),
            height_in=float(dpg.get_value("pyecr_export_height")), font_family=dpg.get_value("pyecr_export_font"),
            plot_views=self._plot_views.freeze(),
            legend_policy='auto' if dpg.get_value('pyecr_export_legend_auto') else 'follow')

    def _cancel_export(self, sender=None, app_data=None, user_data=None):
        self.controller.cancel_page(GuiPage.REPORT)
        self._refresh_widgets()

    def _capture_report_snapshot(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del app_data, user_data
        try:
            self.controller.capture_snapshot(include_export_context=True)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.REPORT, exc, traceback.format_exc()
            )
            self._reveal_report_failure(sender)
        self._refresh_widgets()

    def _report_destination(self) -> str:
        dpg = self._require_dpg()
        if getattr(dpg, "does_item_exist", lambda _: False)("pyecr_report_path_mode"):
            return self._resolve_report_destination()
        value = str(
            self._require_dpg().get_value("pyecr_report_destination")
        ).strip()
        if not value:
            raise ValueError("enter a new report directory before export")
        return value

    def _export_current_report(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del app_data, user_data
        try:
            self.controller.start_export(self._report_destination(), options=self._export_options())
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.REPORT, exc, traceback.format_exc()
            )
            self._reveal_report_failure(sender)
        self._refresh_widgets()

    def _reveal_report_failure(self, sender):
        """A toolbar failure opens its owning page; no service/gate is changed."""
        dpg = self._require_dpg()
        alias = sender if isinstance(sender, str) else dpg.get_item_alias(sender) if sender is not None else ''
        if alias in {'pyecr_analysis_capture_snapshot', 'pyecr_analysis_export_report'}:
            self._navigate_workflow(None, None, 'export')

    def _export_captured_report(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            snapshot = self.controller.session.report_snapshot
            if snapshot is None:
                raise RuntimeError("capture an AnalysisSnapshot before historical export")
            self.controller.start_export(self._report_destination(), options=self._export_options(), historical=True)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.REPORT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _refresh_report_widgets(self) -> None:
        dpg = self._require_dpg()
        if getattr(dpg,'does_item_exist',lambda _:False)('pyecr_reports_workflow'):
            return  # The three-step page owns its source/task/receipt refresh.
        session = self.controller.analysis_session
        snapshot = self.controller.session.report_snapshot
        export_result = self.controller.session.report_export_result
        current = self.controller.can_capture_snapshot
        from pyecr.fitting.reproducibility import reproducibility_issue
        rng_issue=reproducibility_issue(session.fit_result)
        current=current and not rng_issue
        running = self.controller.state.tasks[GuiPage.REPORT].status == "running"
        dpg.configure_item("pyecr_export_cancel", enabled=running)
        current_reason = (
            rng_issue or "Requires a current, non-Out-of-date Fit result."
        )
        set_action_availability(
            dpg, "pyecr_report_capture", enabled=current and not running, reason=current_reason,
        )
        set_action_availability(
            dpg, "pyecr_report_export_current", enabled=current and not running,
            reason=current_reason,
        )
        set_action_availability(
            dpg, "pyecr_report_export_captured", enabled=snapshot is not None and snapshot.fully_reproducible and not running,
            reason=rng_issue or "Capture a reproducible AnalysisSnapshot before historical export.",
        )
        dpg.set_value(
            "pyecr_report_disabled_reason",
            rng_issue or ("" if current and snapshot is not None else
            "Current capture/export requires a current Fit; Out of date results require a captured historical snapshot.")
        )

        if snapshot is None:
            snapshot_status = "No AnalysisSnapshot captured."
        else:
            is_current = bool(
                current
                and snapshot.dataset is session.dataset
                and snapshot.analysis_config == session.current_analysis_config
                and snapshot.fit_result is session.fit_result
            )
            kind = "current" if is_current else "historical"
            snapshot_status = (
                f"Captured {kind} snapshot | model="
                f"{snapshot.analysis_config.model_id.value} | stochastic={snapshot.stochastic} | seed={snapshot.seed if snapshot.stochastic else 'not used (deterministic)'} | "
                f"dataset SHA256={snapshot.dataset_sha256}"
            )
        dpg.set_value("pyecr_report_snapshot_status", snapshot_status)
        if snapshot is not None:
            from pyecr.report.snapshot import _sha256_json
            if getattr(self, "_export_preview_snapshot", None) is not snapshot:
                self._export_preview_snapshot = snapshot
                self._export_preview_fit_id = _sha256_json(snapshot.fit_result.to_dict())
                data = snapshot.fit_plot_data
                for tag, values in (("pyecr_export_preview_observed",data.observed),("pyecr_export_preview_fit",data.fitted)):
                    dpg.set_value(tag,[list(data.time_s),list(values)])
                dpg.configure_item("pyecr_export_preview_y", label=data.signal_label)
                if hasattr(dpg, "fit_axis_data"):
                    dpg.fit_axis_data("pyecr_export_preview_x")
                    dpg.fit_axis_data("pyecr_export_preview_y")
            dpg.configure_item("pyecr_export_preview_plot", show=True)
            dpg.set_value("pyecr_report_snapshot_status", snapshot_status + f" | FitResult={self._export_preview_fit_id}")
        else:
            dpg.configure_item("pyecr_export_preview_plot", show=False)

        if export_result is None:
            dpg.set_value("pyecr_report_pdf_status", "PDF: optional/not configured")
            dpg.set_value("pyecr_report_artifacts", "No report exported.")
            return
        dpg.set_value(
            "pyecr_report_pdf_status",
            f"PDF: {export_result.pdf_status}",
        )
        lines = [
            f"Bundle: {export_result.root}",
            f"Verification: {export_result.verification_status}",
            f"Dataset SHA256: {export_result.dataset_sha256}",
            "",
        ]
        lines.extend(
            f"{artifact.status} | {artifact.relative_path} | "
            f"{artifact.size_bytes} bytes | SHA256 {artifact.sha256}"
            for artifact in export_result.artifacts
        )
        dpg.set_value("pyecr_report_artifacts", "\n".join(lines))
