"""DctViewMixin: page builders, callbacks, and refresh renderers."""

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


from pyecr.gui.components.dct_layout import DctLayoutMixin
from pyecr.gui.views.signed_dct import SignedDctViewMixin
from pyecr.gui.views.component_dct import ComponentDctViewMixin
from pyecr.gui.views.dct_export import DctExportViewMixin


class DctViewMixin(DctExportViewMixin, ComponentDctViewMixin, SignedDctViewMixin, DctLayoutMixin):
    def _build_dct_area(self) -> None:
        self._build_dct_layout()

    def _select_dct_lambda_mode(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        self._require_dpg().configure_item(
            "pyecr_dct_lambda",
            enabled=str(app_data).strip().lower() == "manual",
        )

    def _dct_config_from_widgets(self) -> DctAnalysisConfig:
        dpg = self._require_dpg()
        mode = str(dpg.get_value("pyecr_dct_lambda_mode")).strip().lower()
        if mode=='known-noise auto':mode='manual' # convex options carry the actual selection policy separately
        return DctAnalysisConfig(
            gcv_lambdas=(tuple(float(v.strip()) for v in str(dpg.get_value('pyecr_dct_lambdas')).split(',') if v.strip())
                         if mode == 'gcv' and not self._dct_signed_mode() and dpg.does_item_exist('pyecr_dct_lambdas') else ()),
            tau_min_s=float(dpg.get_value("pyecr_dct_tau_min")),
            tau_max_s=float(dpg.get_value("pyecr_dct_tau_max")),
            n_tau=int(dpg.get_value("pyecr_dct_n_tau")),
            lambda_mode=mode,  # type: ignore[arg-type]
            regularization_lambda=(
                float(dpg.get_value("pyecr_dct_lambda"))
                if mode == "manual"
                else None
            ),
            use_classical_bi=bool(
                dpg.get_value("pyecr_dct_classical_bi")
            ),
        )

    def _run_dct(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._dct_search_message = ''
        if str(self._require_dpg().get_value('pyecr_dct_lambda_mode')).lower() in ('gcv','known-noise auto'):
            self._dct_scan_after_run = self._lambda_scan_target()[0]
        try:
            if self._dct_signed_mode():
                self._run_signed_dct_widgets()
            else:
                self.controller.run_dct(self._dct_config_from_widgets())
        except Exception as exc:
            self._dct_scan_after_run = None
            if self._dct_signed_mode():
                self._signed_dct_callback_error = str(exc)
            self.controller.record_callback_exception(
                GuiPage.DCT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _cancel_dct(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._dct_scan_after_run = None
        try:
            self.controller.cancel_active_dct()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DCT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _fit_dct_plot_axes(self, *args):
        """Discard only DCT view ranges; fit stored samples without a solve."""
        d = self._require_dpg()
        manager = getattr(self, '_plot_views', None)
        for kind in ('spectrum', 'reconstruction', 'residual', 'lambda', 'stability'):
            key = 'pyecr_dct_' + kind + '_plot'
            if manager is not None:
                manager.change(key, x_range=None, y_range=None)
                manager.range_applied.discard(key)
                manager.range_ready.pop(key, None)
                manager.layout_ranges.pop(key, None)
            for axis in ('x', 'y'):
                tag = 'pyecr_dct_' + kind + '_' + axis + '_axis'
                d.set_axis_limits_auto(tag)
                d.fit_axis_data(tag)
        self._dct_fitted_axes.clear()
        self._dct_pending_axes.clear()
        self._dct_release_axes.clear()
        self._refresh_widgets()

    def _render_dct_presentation(self) -> None:
        dpg = self._require_dpg()
        signed = self._dct_signed_mode()
        # PlotViews restores business availability before rendering. Route
        # callbacks run earlier, so reassert availability here, even on cached
        # results; PlotViews applies the user's hidden-series state afterwards.
        for name in ('positive', 'negative', 'zero'):
            dpg.configure_item('pyecr_signed_dct_' + name, show=signed)
        dpg.configure_item('pyecr_dct_spectrum_series', show=not signed)
        if signed:
            self._render_signed_dct()
            return
        presentation = self.controller.session.dct_result
        if presentation is None:
            if self._rendered_dct_presentation is not None:
                for tag in (
                    "pyecr_dct_spectrum_series",
                    "pyecr_dct_data_series",
                    "pyecr_dct_reconstruction_series",
                    "pyecr_dct_residual_series",
                ):
                    dpg.set_value(tag, [[], []])
                dpg.set_value("pyecr_dct_summary", "No current DCT result.")
                dpg.set_value(
                    "pyecr_dct_details",
                    "No candidate time scales available.",
                )
                self._rendered_dct_presentation = None
            return
        if presentation is self._rendered_dct_presentation:
            return
        if not isinstance(presentation, DctPresentation):
            raise TypeError("session DCT result must be a DctPresentation")
        times = list(presentation.time_s)
        dpg.set_value(
            "pyecr_dct_spectrum_series",
            [list(presentation.log10_tau), list(presentation.chi)],
        )
        dpg.set_value(
            "pyecr_dct_data_series",
            [times, list(presentation.signal)],
        )
        dpg.set_value(
            "pyecr_dct_reconstruction_series",
            [times, list(presentation.reconstruction)],
        )
        dpg.set_value(
            "pyecr_dct_residual_series",
            [times, list(presentation.residuals)],
        )
        result = presentation.result
        dpg.set_value(
            "pyecr_dct_summary",
            f"lambda={format_scientific(result.regularization_lambda)} "
            f"({result.lambda_source}) | RMSE={format_scientific(result.RMSE)} | "
            f"normalization integral={format_scientific(result.normalization_integral)}",
        )
        lines = ["Candidate time scales (diagnostic labels only):"]
        if result.candidate_timescales:
            lines.extend(
                f"Candidate {index}: tau={format_scientific(candidate.position_s)} s, "
                f"area={format_scientific(candidate.area)}, width="
                f"{format_scientific(candidate.width_decades)} decades"
                for index, candidate in enumerate(
                    result.candidate_timescales, start=1
                )
            )
        else:
            lines.append("No resolved candidate time scale.")
        suggestion = presentation.suggestion
        from pyecr.model_display import display_label, display_recommendation_text
        lines.extend(
            (
                "",
                "Suggested model trials (not applied automatically): "
                + ", ".join(display_label(name) for name in suggestion.recommended_model_ids),
                display_recommendation_text(suggestion.recommendation),
                f"Lambda stability: "
                f"{suggestion.lambda_stability.get('status', 'not_assessed')}",
                "Chemical-species assignment: none",
            )
        )
        warnings = tuple(
            dict.fromkeys((*result.resolution_warnings, *suggestion.warnings))
        )
        if warnings:
            lines.extend(("", "Resolution / artifact warnings:", *warnings))
        dpg.set_value("pyecr_dct_details", "\n".join(lines))
        for tag in (
            "pyecr_dct_spectrum_x_axis",
            "pyecr_dct_spectrum_y_axis",
            "pyecr_dct_reconstruction_x_axis",
            "pyecr_dct_reconstruction_y_axis",
            "pyecr_dct_residual_x_axis",
            "pyecr_dct_residual_y_axis",
        ):
            dpg.fit_axis_data(tag)
        self._rendered_dct_presentation = presentation
