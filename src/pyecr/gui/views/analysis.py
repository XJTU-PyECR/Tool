"""AnalysisViewMixin: page builders, callbacks, and refresh renderers."""

from __future__ import annotations
from collections.abc import Mapping
from time import monotonic
from pyecr.gui.theme import TOKENS
from pyecr.gui.solver_portfolio import config_selection, solver_details

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
    _project_directory_text,
)
from pyecr.gui.components.results_table import (
    _replace_diagnostics_table,
    _render_bi_table, _render_parameter_diagnostics_table,
    _render_correlation_table, _render_multistart_table,
    _format_confidence_interval,
    _clear_table_copy,
    format_scientific,
)
from pyecr.gui.components.plot_tabs import (
    _update_processed_plot, _replace_data_preview_table,
)
from pyecr.gui.components.scientific_plot import (
    clear_plot_series, residual_axis_label, robust_residual_limits,
    signal_axis_label, signal_display_unit, time_axis_label,
)
from pyecr.gui.components.action_state import (
    add_action_tooltip, set_action_availability,
)
from pyecr.plotting.forward_views import boundary_plot_data
from pyecr.units import get_default_unit, get_units, unit_label, validate_pair
from pyecr.gui.controller import UnitConfirmationRequired
from pyecr.gui.preprocessing_workflow import (
    FIELD_DOMAINS, FORM_FIELDS, excluded_by_reason, generic_parameters, integer_field,
    history_evidence, preview_evidence,
)
from pyecr.gui.preprocessing_plots import preview_uses_separate_axes
from pyecr.preprocessing import PreprocessConfig
from pyecr.gui.signed_normalization import SIGNED_DISPLAY, SIGNED_OPERATION, operation_id


_EXPLORER_DEBOUNCE_SECONDS = 0.25


def _advisor_display(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value or "N/A"
    if isinstance(value, Mapping):
        return "\n".join(f"{key}: {item}" for key, item in value.items()) or "N/A"
    if isinstance(value, (tuple, list)):
        return "\n".join(str(item) for item in value) or "N/A"
    return str(value)


def _information_fraction(value: float) -> str:
    return f"{value:.1%}"


class AnalysisViewMixin:
    def _build_project_area(self) -> None:
        dpg = self._require_dpg()
        dpg.add_input_text(
            label="Project directory",
            tag="pyecr_project_path",
            default_value=self.controller.state.project_path or "",
            readonly=True,
            width=TOKENS.widths.path,
        )
        with dpg.group(horizontal=True):
            dpg.add_button(label="New", callback=self._new_project)
            dpg.add_button(label="Open", callback=self._show_project_open_dialog)
            dpg.add_button(
                label="Save",
                tag="pyecr_project_save",
                callback=self._save_project,
            )
            dpg.add_button(
                label="Save As",
                tag="pyecr_project_save_as",
                callback=self._show_project_save_as_dialog,
            )
            dpg.add_button(
                label="Make Portable",
                tag="pyecr_project_make_portable",
                callback=self._make_project_portable,
            )
        dpg.add_text(
            "No project is open. Import data before the first save.",
            tag="pyecr_project_details",
            wrap=TOKENS.wraps.page,
        )
        dpg.add_text(
            "Linked projects verify the external raw file; portable projects copy it "
            "into data/raw.",
            wrap=TOKENS.wraps.page,
        )
        dpg.add_text("", tag="pyecr_project_disabled_reason", wrap=TOKENS.wraps.project)
        with dpg.window(label="Confirm legacy signal unit", modal=True, show=False,
                        tag="pyecr_project_unit_confirmation", width=TOKENS.widths.unit_confirmation,
                        height=TOKENS.heights.unit_confirmation):
            dpg.add_text("", tag="pyecr_project_unit_warning", wrap=TOKENS.wraps.unit_confirmation)
            dpg.add_button(label="Open without unit conversion", callback=self._confirm_project_open_units)
            dpg.add_button(label="Cancel", callback=self._cancel_project_open_units)

    def _build_model_config_area(self) -> None:
        dpg = self._require_dpg()
        descriptors = self.controller.model_descriptors
        if not descriptors:
            raise RuntimeError("ModelRegistry contains no model descriptors")
        model_ids = tuple(descriptor.model_id.value for descriptor in descriptors)
        selected = self.controller.state.model_name
        if selected not in model_ids:
            selected = model_ids[0]
        from pyecr.gui.model_display import display_labels,display_label
        dpg.add_combo(display_labels(),label='Model — single / dual relaxation',tag='pyecr_model_display',
                      default_value=display_label(selected),callback=self._select_model_descriptor,width=-1)
        dpg.add_combo(
            model_ids,
            label="Candidate model",
            tag="pyecr_model_id",
            default_value=selected,
            callback=self._select_model_descriptor,
            width=-1,
            show=False,
        )
        dpg.add_group(tag="pyecr_model_schema_container")
        self._render_model_controls(selected)

    def _build_data_import_area(self) -> None:
        dpg = self._require_dpg()
        self._build_import_setup()
        with dpg.plot(label="Raw measurement", tag="pyecr_data_raw_plot", height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Raw time",
                tag="pyecr_data_raw_x_axis",
            )
            dpg.add_plot_axis(
                dpg.mvYAxis,
                label="Raw signal",
                tag="pyecr_data_raw_y_axis",
            )
            dpg.add_line_series(
                [],
                [],
                label="Raw data",
                parent="pyecr_data_raw_y_axis",
                tag="pyecr_data_raw_series",
                show=False,
            )
            for reason, label in (("mask", "Masked original rows"), ("crop", "Cropped original rows"), ("downsample", "Downsampled original rows")):
                dpg.add_scatter_series([], [], label=label, parent="pyecr_data_raw_y_axis", tag=f"pyecr_raw_excluded_{reason}",show=False)
        dpg.add_separator(tag="pyecr_data_processing_separator")
        dpg.add_text("Replayable preprocessing", tag="pyecr_data_processing_header")
        dpg.add_text(
            "Choose typed fields, Preview, review the data overlay, then Accept or Reject. "
            "No smoothing is applied automatically.",
            tag="pyecr_data_processing_description",
            wrap=TOKENS.wraps.rail,
        )
        with dpg.group(tag="pyecr_data_stability_controls"):
            dpg.add_text("Stability window suggestion")
            dpg.add_text(
                "Suggestions use the current processed signal. Accept records a "
                "replayable window step; Reject leaves the data unchanged.",
                wrap=TOKENS.wraps.rail,
            )
            dpg.add_radio_button(
                ("Initial", "Final"),
                label="Plateau",
                tag="pyecr_data_stability_kind",
                default_value="Initial",
                callback=self._select_stability_kind,
            )
            dpg.add_input_text(label="Minimum points (≥3)", tag="pyecr_data_stability_min_points", default_value="5", width=TOKENS.widths.field_wide)
            dpg.add_text(FIELD_DOMAINS["minimum_points"], wrap=TOKENS.wraps.rail)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Suggest window",
                    tag="pyecr_data_stability_suggest",
                    callback=self._suggest_stability_window,
                    enabled=False,
                )
                dpg.add_button(
                    label="Accept",
                    tag="pyecr_data_stability_accept",
                    callback=self._accept_stability_window,
                    enabled=False,
                )
                dpg.add_button(
                    label="Reject",
                    tag="pyecr_data_stability_reject",
                    callback=self._reject_stability_window,
                    enabled=False,
                )
            dpg.add_text(
                "Select Initial or Final, then Suggest. Preview does not change data.",
                tag="pyecr_data_stability_preview",
                wrap=TOKENS.wraps.rail,
            )
        with dpg.group(tag="pyecr_data_t0_controls"):
            dpg.add_text("t0 / flush assistant")
            dpg.add_text(
                "The onset is a suggestion. Accept shifts the current processed time axis; "
                "negative pre-onset times must be explicitly cropped or masked before Fit.",
                wrap=TOKENS.wraps.rail,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Suggest t0", tag="pyecr_data_t0_suggest",
                    callback=self._suggest_t0, enabled=False,
                )
                dpg.add_button(
                    label="Accept", tag="pyecr_data_t0_accept",
                    callback=self._accept_t0, enabled=False,
                )
                dpg.add_button(
                    label="Reject", tag="pyecr_data_t0_reject",
                    callback=self._reject_t0, enabled=False,
                )
            dpg.add_text(
                "Suggest t0 to preview the detected onset; Accept records a preprocessing step.",
                tag="pyecr_data_t0_preview", wrap=TOKENS.wraps.rail,
            )
        self._build_optional_preprocessing()
        # Compatibility dispatch value, not a hidden/renamed tool combo.
        with dpg.value_registry():
            dpg.add_string_value(tag="pyecr_data_processing_operation", default_value="normalize")
            dpg.add_string_value(tag="pyecr_typed_mode", default_value="progress")
        with dpg.group(tag="pyecr_typed_processing_controls"):
            dpg.add_text("Time ranges use the current processed unit.", tag="pyecr_typed_current_unit", wrap=TOKENS.wraps.rail)
            for field in ("to_unit", "path_length_cm", "cross_section_area_cm2", "t0", "start_time", "end_time", "raw_indices", "mode"):
                with dpg.group(tag=f"pyecr_typed_group_{field}"):
                    if field == "mode":
                        dpg.add_text('', tag='pyecr_typed_error_mode')
                        continue  # Visible Progress/Remaining/Signed radio is built in R03 layout.
                    if field == "to_unit":
                        choices = _TIME_UNIT_CHOICES
                        dpg.add_combo(choices, label=field, tag=f"pyecr_typed_{field}", default_value=choices[0], width=TOKENS.widths.field_wide)
                    else:
                        defaults = {"path_length_cm": "1", "cross_section_area_cm2": "1", "t0": "0", "start_time": "0", "end_time": "100", "raw_indices": ""}
                        dpg.add_input_text(label=field, tag=f"pyecr_typed_{field}", default_value=defaults[field], hint='e.g. 4, 12, 20' if field=='raw_indices' else '',width=TOKENS.widths.field_wide)
                    dpg.add_text(FIELD_DOMAINS[field], wrap=TOKENS.wraps.rail)
                    dpg.add_text("", tag=f"pyecr_typed_error_{field}", wrap=TOKENS.wraps.rail)
            dpg.add_text("Input valid domain appears under each field. Preview errors appear below.", tag="pyecr_typed_error", wrap=TOKENS.wraps.rail)
        self._build_signed_normalization()
        self._show_typed_processing_fields(PREPROCESSING_OPERATIONS[0])
        dpg.add_input_text(
            label="Advanced recipe (copyable JSON)",
            tag="pyecr_data_processing_parameters",
            default_value=json.dumps(_PROCESSING_PARAMETER_EXAMPLES[PREPROCESSING_OPERATIONS[0]]),
            readonly=True,
            width=-1,
        )
        with dpg.group(horizontal=True, tag="pyecr_data_processing_buttons"):
            dpg.add_button(
                label="Preview",
                tag="pyecr_data_apply_processing",
                callback=self._preview_typed_preprocessing,
                enabled=False,
            )
            dpg.add_button(label="Accept", tag="pyecr_data_accept_processing", callback=self._accept_typed_preprocessing, enabled=False)
            dpg.add_button(label="Reject", tag="pyecr_data_reject_processing", callback=self._reject_typed_preprocessing, enabled=False)
            dpg.add_button(
                label="Undo",
                tag="pyecr_data_undo_processing",
                callback=self._undo_preprocessing,
                enabled=False,
            )
            dpg.add_button(
                label="Reset",
                tag="pyecr_data_reset_processing",
                callback=self._reset_preprocessing,
                enabled=False,
            )
        dpg.add_text("Import a dataset before applying preprocessing.", tag="pyecr_data_processing_reason", wrap=TOKENS.wraps.narrow)
        with dpg.plot(label="Raw context · original units / exclusion reasons", tag="pyecr_preprocess_raw_plot", height=TOKENS.heights.residual, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(dpg.mvXAxis, label="Raw time", tag="pyecr_preprocess_raw_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Raw signal", tag="pyecr_preprocess_raw_y_axis")
            dpg.add_line_series([], [], label="Raw", parent="pyecr_preprocess_raw_y_axis", tag="pyecr_preprocess_raw_series")
            for reason, label in (("mask", "Mask"), ("crop", "Crop"), ("downsample", "Downsample")):
                dpg.add_scatter_series([], [], label=label, parent="pyecr_preprocess_raw_y_axis", tag=f"pyecr_preprocess_excluded_{reason}")
        with dpg.plot(label="Processed signal", tag="pyecr_data_processed_plot", height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Processed time",
                tag="pyecr_data_processed_x_axis",
            )
            dpg.add_plot_axis(
                dpg.mvYAxis,
                label="Processed signal",
                tag="pyecr_data_processed_y_axis",
            )
            dpg.add_line_series(
                [],
                [],
                label="Processed data",
                parent="pyecr_data_processed_y_axis",
                tag="pyecr_data_processed_series",
            )
            dpg.add_line_series(
                [], [], label="Suggested start",
                parent="pyecr_data_processed_y_axis",
                tag="pyecr_data_stability_start_series", show=False,
            )
            dpg.add_line_series(
                [], [], label="Suggested end",
                parent="pyecr_data_processed_y_axis",
                tag="pyecr_data_stability_end_series", show=False,
            )
            dpg.add_line_series(
                [], [], label="Suggested t0",
                parent="pyecr_data_processed_y_axis",
                tag="pyecr_data_t0_series", show=False,
            )
            dpg.add_line_series([], [], label="Optional preprocessing preview", parent="pyecr_data_processed_y_axis", tag="pyecr_optional_preview_series", show=False)
            dpg.add_scatter_series([], [], label="Review outliers", parent="pyecr_data_processed_y_axis", tag="pyecr_optional_markers", show=False)
            dpg.add_scatter_series([], [], label="Retained preview", parent="pyecr_data_processed_y_axis", tag="pyecr_optional_retained_markers", show=False)
            dpg.add_line_series([], [], label="Typed operation preview", parent="pyecr_data_processed_y_axis", tag="pyecr_typed_preview_series", show=False)
        dpg.add_text(
            "Processed plot uses the current immutable dataset view.",
            tag="pyecr_data_processed_interpretation",
            wrap=TOKENS.wraps.page,
        )
        dpg.add_input_text(label="Preprocessing review", tag="pyecr_typed_preview_evidence", default_value="Preview does not change data.", multiline=True, readonly=True, height=TOKENS.heights.history, width=-1)
        dpg.add_input_text(label="Fit quality (same service result)", tag="pyecr_data_fit_quality", default_value="FIT READY NO · import data and configure a model.", multiline=True, readonly=True, height=TOKENS.heights.history, width=-1)
        dpg.add_input_text(
            label="Processing history",
            tag="pyecr_data_processing_history",
            default_value="No preprocessing steps.",
            multiline=True,
            readonly=True,
            width=-1,
            height=TOKENS.heights.history,
        )

    def _build_data_file_dialog(self) -> None:
        dpg = self._require_dpg()
        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._select_data_file,
            tag="pyecr_data_file_dialog",
            width=TOKENS.widths.path,
            height=TOKENS.heights.dialog,
        ):
            dpg.add_file_extension(".csv", color=TOKENS.colors.csv)
            dpg.add_file_extension(".txt", color=TOKENS.colors.txt)
            dpg.add_file_extension(".xlsx", color=TOKENS.colors.xlsx)
            dpg.add_file_extension(".xls", color=TOKENS.colors.xlsx)
            dpg.add_file_extension(".*")

    def _build_project_dialogs(self) -> None:
        dpg = self._require_dpg()
        with dpg.window(label='Open project',tag='pyecr_project_open_window',show=False,
                        width=TOKENS.widths.path,height=TOKENS.heights.preview):
            dpg.add_text('Full project directory')
            dpg.add_input_text(tag='pyecr_project_open_path',width=-1,
                hint='C:\\path\\to\\project',on_enter=True,callback=self._open_project_from_path)
            dpg.add_text('',tag='pyecr_project_open_reason',wrap=TOKENS.wraps.project)
            with dpg.group(horizontal=True):
                dpg.add_button(label='Open project',tag='pyecr_project_open_confirm',callback=self._open_project_from_path)
                dpg.add_button(label='Browse directories',tag='pyecr_project_open_browse',callback=self._browse_project_directory)
                dpg.add_button(label='Cancel',callback=lambda:dpg.hide_item('pyecr_project_open_window'))
        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=self._open_project_from_dialog,
            tag="pyecr_project_open_dialog",
            width=TOKENS.widths.path,
            height=TOKENS.heights.dialog,
        )
        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=self._add_batch_project_from_dialog,
            tag="pyecr_batch_project_dialog",
            width=TOKENS.widths.path,
            height=TOKENS.heights.dialog,
        )
        dpg.add_file_dialog(
            directory_selector=True,
            show=False,
            callback=self._save_project_as_from_dialog,
            tag="pyecr_project_save_as_dialog",
            width=TOKENS.widths.path,
            height=TOKENS.heights.dialog,
        )

    def _build_fit_output_area(self) -> None:
        dpg = self._require_dpg()
        from pyecr.model_display import display_model_text
        dpg.add_text("", tag="pyecr_fit_summary", wrap=TOKENS.wraps.rail_alt)
        dpg.add_text(
            display_model_text(self.controller.fit_context_summary),
            tag="pyecr_fit_context_summary",
            wrap=TOKENS.wraps.rail_alt,
        )
        dpg.add_text(
            "Run Fit uses the current imported/processed dataset and the validated "
            "Model-page AnalysisConfig.",
            tag="pyecr_fit_usage_help",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_text("No Fit result.", tag="pyecr_fit_reliability", wrap=TOKENS.wraps.rail_alt)
        dpg.add_input_text(
            tag="pyecr_fit_warnings",
            default_value="No Fit warnings.",
            multiline=True,
            readonly=True,
            width=-1,
            height=TOKENS.heights.status_short,
        )
        with dpg.plot(label="Observed vs fitted", tag="pyecr_fit_plot", height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="pyecr_fit_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Signal", tag="pyecr_fit_y_axis")
            dpg.add_scatter_series(
                [],
                [],
                label="Data",
                parent="pyecr_fit_y_axis",
                tag="pyecr_fit_data_series",
            )
            dpg.add_line_series(
                [],
                [],
                label="Fit",
                parent="pyecr_fit_y_axis",
                tag="pyecr_fit_curve_series",
            )
        from pyecr.gui.components.decomposition import build_controls, build_auxiliary
        build_controls(self)
        with dpg.plot(label="Continuous contributions", tag="pyecr_components_plot", height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="pyecr_components_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Contribution", tag="pyecr_components_y_axis")
            dpg.add_line_series([], [], label="H component", parent="pyecr_components_y_axis", tag="pyecr_fit_h_series")
            dpg.add_line_series([], [], label="O component", parent="pyecr_components_y_axis", tag="pyecr_fit_o_series")
            dpg.add_line_series([], [], label="Joint fit", parent="pyecr_components_y_axis", tag="pyecr_decomposition_total")
            dpg.add_scatter_series([], [], label="Observed", parent="pyecr_components_y_axis", tag="pyecr_decomposition_observed")
            dpg.add_line_series([], [], label="Zero", parent="pyecr_components_y_axis", tag="pyecr_decomposition_reference")
        build_auxiliary(self)
        with dpg.group(horizontal=True, tag="pyecr_residual_scale_controls"):
            dpg.add_text("Residual y range")
            dpg.add_button(label="Auto y", tag="pyecr_residual_auto_y", callback=self._set_residual_scale, user_data="auto")
            dpg.add_button(label="Robust y", tag="pyecr_residual_robust_y", callback=self._set_residual_scale, user_data="robust")
            dpg.add_text("Auto", tag="pyecr_residual_scale_status")
        with dpg.plot(label="Fit residual", tag="pyecr_residual_plot", height=TOKENS.heights.residual, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Time (s)",
                tag="pyecr_residual_x_axis",
            )
            dpg.add_plot_axis(
                dpg.mvYAxis,
                label="Observed - fitted",
                tag="pyecr_residual_y_axis",
            )
            dpg.add_scatter_series(
                [],
                [],
                label="Residuals",
                parent="pyecr_residual_y_axis",
                tag="pyecr_residual_series",
            )
            dpg.add_line_series(
                [], [], label="Zero", parent="pyecr_residual_y_axis",
                tag="pyecr_residual_zero_series", show=False,
            )
        with dpg.plot(label="Gas boundary response", tag="pyecr_boundary_plot", height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="pyecr_boundary_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Normalized response / input", tag="pyecr_boundary_y_axis")
            for label, tag in (
                ("Ideal material", "pyecr_boundary_ideal_series"),
                ("Boundary-corrected", "pyecr_boundary_corrected_series"),
                ("Gas input", "pyecr_boundary_input_series"),
            ):
                dpg.add_line_series([], [], label=label, parent="pyecr_boundary_y_axis", tag=tag, show=False)
        dpg.add_text("", tag="pyecr_boundary_plot_note")
        with dpg.plot(label="Parameter sensitivity preview", tag="pyecr_explorer_plot", height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="pyecr_explorer_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Signal", tag="pyecr_explorer_y_axis")
            dpg.add_line_series([], [], label="Current fit", parent="pyecr_explorer_y_axis", tag="pyecr_explorer_fit_series", show=False)
            dpg.add_line_series([], [], label="Parameter variant", parent="pyecr_explorer_y_axis", tag="pyecr_explorer_variant_series", show=False)
        with dpg.collapsing_header(label="Parameter preview", tag="pyecr_explorer_controls", default_open=False):
            dpg.add_combo([], tag="pyecr_explorer_parameter", width=TOKENS.widths.input_medium,
                          callback=self._schedule_parameter_variant)
            dpg.add_input_float(label="Multiplier", tag="pyecr_explorer_multiplier", default_value=1.2,
                                width=TOKENS.widths.input_small, callback=self._schedule_parameter_variant)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Preview", tag="pyecr_explorer_preview", callback=self._preview_parameter_variant)
                dpg.add_button(label="Cancel", tag="pyecr_explorer_cancel", callback=self._cancel_parameter_variant, enabled=False)
            add_action_tooltip(dpg, "pyecr_explorer_preview")
            add_action_tooltip(dpg, "pyecr_explorer_cancel")
            dpg.add_text("Requires a current Fit result.", tag="pyecr_explorer_status", wrap=TOKENS.wraps.advanced)
        dpg.add_text("Parameter estimates")
        dpg.add_group(tag="pyecr_fit_parameter_container")

    def _build_diagnostics_output_area(self) -> None:
        dpg = self._require_dpg()
        dpg.add_text(
            "Diagnostics require a current Fit. Optimizer success and "
            "parameter reliability are reported separately.",
            wrap=TOKENS.wraps.rail,
        )
        with dpg.group(horizontal=True, tag="pyecr_diagnostics_actions"):
            dpg.add_button(
                label="Run Diagnostics",
                tag="pyecr_run_diagnostics",
                callback=self._run_page,
                user_data=GuiPage.DIAGNOSTICS.value,
                enabled=False,
            )
            dpg.add_button(
                label="Bootstrap (advanced)",
                tag="pyecr_diagnostics_bootstrap",
                callback=self._run_advanced_diagnostics,
                user_data="bootstrap",
                enabled=False,
            )
            dpg.add_button(
                label="Profile Likelihood (advanced)",
                tag="pyecr_diagnostics_profile",
                callback=self._run_advanced_diagnostics,
                user_data="profile",
                enabled=False,
            )
            dpg.add_button(
                label="Deep uncertainty (single D+k, experimental)",
                tag="pyecr_diagnostics_experimental_m1",
                callback=self._run_advanced_diagnostics,
                user_data="experimental_m1",
                enabled=False,
            )
            dpg.add_button(
                label="Cancel",
                tag="pyecr_cancel_diagnostics",
                callback=self._cancel_diagnostics,
                enabled=False,
            )
            for tag in (
                "pyecr_run_diagnostics", "pyecr_diagnostics_bootstrap",
                "pyecr_diagnostics_profile", "pyecr_cancel_diagnostics",
                "pyecr_diagnostics_experimental_m1",
            ):
                add_action_tooltip(dpg, tag)
        with dpg.group(tag="pyecr_experimental_m1_controls"):
            dpg.add_checkbox(
                label="I acknowledge centered residuals are exchangeable; time correlation is not retained",
                tag="pyecr_experimental_m1_ack", default_value=False,
            )
            dpg.add_input_int(label="Bootstrap seed", tag="pyecr_experimental_m1_seed",
                              default_value=660101, min_value=0, min_clamped=True,
                              width=TOKENS.widths.input_medium)
            dpg.add_text(
                "Single D+k, linear loss and ideal step only; 200 refits, 90 s budget. Local CI remains. "
                "This experimental interval requires residual assumptions and may be unavailable.",
                wrap=TOKENS.wraps.rail,
            )
            dpg.add_input_text(label="Independent result JSON", tag="pyecr_experimental_m1_path",
                               default_value="experimental_m1_uncertainty.json", width=-1)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save deep result", callback=self._save_experimental_m1)
                dpg.add_button(label="Reopen deep result", callback=self._reopen_experimental_m1)
        with dpg.group(tag="pyecr_diagnostics_error_map_action"):
            dpg.add_button(
                label="Error Map settings (M2_SINGLE_DK)",
                tag="pyecr_diagnostics_error_map_run",
                callback=self._run_error_map,
                enabled=False,
            )
            add_action_tooltip(dpg, "pyecr_diagnostics_error_map_run")
            dpg.add_text(
                "Open range/grid settings; Generate Error Map starts the calculation.",
                tag="pyecr_diagnostics_error_map_reason",
                wrap=TOKENS.wraps.narrow,
            )
        with dpg.group(tag="pyecr_diagnostics_dual_map_action", show=False):
            dpg.add_button(
                label="Error Map settings (M5_DUAL_H2O_DK)",
                tag="pyecr_diagnostics_dual_map_run",
                callback=self._run_dual_error_map,
                enabled=False,
            )
            add_action_tooltip(dpg, "pyecr_diagnostics_dual_map_run")
            dpg.add_text(
                "Open range/grid settings; Generate Dual Error Maps starts the calculation.",
                tag="pyecr_diagnostics_dual_map_reason", wrap=TOKENS.wraps.narrow,
            )
        dpg.add_text("Requires a current Fit result; Out of date results are disabled.", tag="pyecr_diagnostics_disabled_reason", wrap=TOKENS.wraps.narrow)
        dpg.add_text("Cancel is available while diagnostics are running.", tag="pyecr_diagnostics_cancel_reason", wrap=TOKENS.wraps.narrow)
        dpg.add_text(
            "No Diagnostics result.",
            tag="pyecr_diagnostics_reliability",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_input_text(
            tag="pyecr_diagnostics_warnings",
            default_value="No Diagnostics warnings.",
            multiline=True,
            readonly=True,
            width=-1,
            height=TOKENS.heights.status_short,
        )
        with dpg.plot(label="Residuals", tag="pyecr_diagnostics_residual_plot", height=TOKENS.heights.diagnostics, width=-1, show=False):
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Time (s)",
                tag="pyecr_diagnostics_residual_x_axis",
            )
            dpg.add_plot_axis(
                dpg.mvYAxis,
                label="Observed - fitted",
                tag="pyecr_diagnostics_residual_y_axis",
            )
            dpg.add_scatter_series(
                [],
                [],
                label="Residuals",
                parent="pyecr_diagnostics_residual_y_axis",
                tag="pyecr_diagnostics_residual_series",
            )
        dpg.add_text("Residual summary: N/A — diagnostics have not run.", tag="pyecr_diagnostics_residual_summary", wrap=TOKENS.wraps.rail)
        dpg.add_text("Fit statistics: N/A — diagnostics have not run.", tag="pyecr_diagnostics_information_criteria", wrap=TOKENS.wraps.rail)
        dpg.add_text("Error Map: not run.", tag="pyecr_diagnostics_error_map_status", wrap=TOKENS.wraps.rail)
        dpg.add_input_text(
            tag="pyecr_diagnostics_error_map_warnings",
            default_value="No Error Map warnings.",
            multiline=True,
            readonly=True,
            width=-1,
            height=TOKENS.heights.status_short,
        )
        dpg.add_text(
            "M5_DUAL_H2O_DK Error Maps: not run.", tag="pyecr_diagnostics_dual_map_status",
            wrap=TOKENS.wraps.rail, show=False,
        )
        dpg.add_input_text(
            tag="pyecr_diagnostics_dual_map_warnings",
            default_value="No H/O Error Map warnings.", multiline=True,
            readonly=True, width=-1, height=TOKENS.heights.status_short,
            show=False,
        )
        dpg.add_text("Jacobian: N/A — diagnostics have not run.", tag="pyecr_diagnostics_jacobian", wrap=TOKENS.wraps.rail)
        dpg.add_text("Bi: N/A — diagnostics have not run.", tag="pyecr_diagnostics_bi_status", wrap=TOKENS.wraps.rail)
        dpg.add_group(tag="pyecr_diagnostics_bi_container")
        dpg.add_text("Parameter uncertainty: N/A — diagnostics have not run.", tag="pyecr_diagnostics_parameter_status", wrap=TOKENS.wraps.rail)
        dpg.add_group(tag="pyecr_diagnostics_parameter_container")
        dpg.add_text("Correlation matrix: N/A — diagnostics have not run.", tag="pyecr_diagnostics_correlation_status", wrap=TOKENS.wraps.rail)
        dpg.add_group(tag="pyecr_diagnostics_correlation_container")
        dpg.add_text("MultiStart: N/A — diagnostics have not run.", tag="pyecr_diagnostics_multistart_status", wrap=TOKENS.wraps.rail)
        dpg.add_group(tag="pyecr_diagnostics_multistart_container")
        dpg.add_separator()
        dpg.add_text(
            "Bootstrap: not run (advanced diagnostics are opt-in).",
            tag="pyecr_diagnostics_bootstrap_status",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_text(
            "Profile likelihood: not run (advanced diagnostics are opt-in).",
            tag="pyecr_diagnostics_profile_status",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_text(
            "Deep uncertainty: not run (explicit opt-in).",
            tag="pyecr_diagnostics_experimental_m1_status",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_input_text(
            label="Experimental evidence (Bi heuristic; rank; correlation; spread; map; amplitude)",
            tag="pyecr_diagnostics_experimental_m1_evidence",
            default_value="No experimental result for the current fit.",
            multiline=True, readonly=True, width=-1,
            height=TOKENS.heights.status_small,
        )
        dpg.add_separator()
        dpg.add_text("RELIABILITY SUMMARY")
        dpg.add_text(
            "Identifiability: N/A — diagnostics have not run.",
            tag="pyecr_reliability_identifiability",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_text(
            "Model adequacy: N/A — diagnostics have not run.",
            tag="pyecr_reliability_adequacy",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_text(
            "Bi guidance: N/A — diagnostics have not run.",
            tag="pyecr_reliability_bi_guidance",
            wrap=TOKENS.wraps.rail,
        )
        dpg.add_input_text(
            label="Evidence",
            tag="pyecr_reliability_evidence",
            default_value="No Reliability Advisor evidence.",
            multiline=True,
            readonly=True,
            width=-1,
            height=TOKENS.heights.status_small,
        )
        with dpg.group(tag="pyecr_guided_refit_controls"):
            dpg.add_separator()
            dpg.add_text("M1 GUIDED REFIT")
            dpg.add_text(
                "Run Diagnostics first. Fix the better identified parameter, then "
                "refit the weaker one. A refit runs only after an explicit button click.",
                wrap=TOKENS.wraps.rail,
            )
            dpg.add_text(
                "Guided Refit: run 03 M2_SINGLE_DK Diagnostics to assess eligibility.",
                tag="pyecr_guided_refit_eligibility", wrap=TOKENS.wraps.rail,
            )
            dpg.add_combo(
                ("Select source", "Parent fitted value", "Manual value"),
                label="Fixed value source", tag="pyecr_guided_refit_source",
                default_value="Select source",
                callback=self._select_guided_refit_source,
                width=TOKENS.widths.input_wide,
            )
            dpg.add_input_double(
                label="Manual fixed value", tag="pyecr_guided_refit_manual_value",
                default_value=1.0e-6, format="%.6e", show=False,
                width=TOKENS.widths.input_regular,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Fix D / refit k", tag="pyecr_guided_refit_fix_d",
                    callback=self._run_guided_refit, user_data="D_cm2_s",
                    enabled=False,
                )
                dpg.add_button(
                    label="Fix k / refit D", tag="pyecr_guided_refit_fix_k",
                    callback=self._run_guided_refit, user_data="k_cm_s",
                    enabled=False,
                )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Cancel Refit", tag="pyecr_guided_refit_cancel",
                    callback=self._cancel_guided_refit, enabled=False,
                )
                dpg.add_button(
                    label="Rollback to parent", tag="pyecr_guided_refit_rollback",
                    callback=self._rollback_guided_refit, enabled=False,
                )
            for tag in ("pyecr_guided_refit_fix_d", "pyecr_guided_refit_fix_k"):
                add_action_tooltip(dpg, tag)
            dpg.add_text(
                "No Guided Refit has been run.",
                tag="pyecr_guided_refit_status", wrap=TOKENS.wraps.rail,
            )
            dpg.add_input_text(
                label="Before / after evidence", tag="pyecr_guided_refit_comparison",
                default_value="No Guided Refit comparison.",
                multiline=True, readonly=True, width=-1,
                height=TOKENS.heights.status_small,
            )

    def _build_sensitivity_area(self) -> None:
        dpg = self._require_dpg()
        dpg.add_text(
            "S_logD(t) = ∂y/∂log10(D); S_logk(t) = ∂y/∂log10(k). "
            "M3 reports H and O parameters separately. These are not q∂σ/∂q.",
            wrap=TOKENS.wraps.map,
        )
        dpg.add_text(
            "Run Diagnostics to calculate sensitivity curves.",
            tag="pyecr_sensitivity_status",
            wrap=TOKENS.wraps.map,
        )
        dpg.add_checkbox(
            label="Normalize each curve by its own peak magnitude (view only)",
            tag="pyecr_sensitivity_normalized",
            default_value=False,
            callback=self._toggle_sensitivity_view,
        )
        with dpg.plot(label="Parameter sensitivity", tag="pyecr_sensitivity_plot",
                      height=TOKENS.heights.plot, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(
                dpg.mvXAxis, label="Time (s)", tag="pyecr_sensitivity_x_axis",
            )
            dpg.add_plot_axis(
                dpg.mvYAxis, label="∂y/∂log10(parameter)",
                tag="pyecr_sensitivity_y_axis",
            )
        dpg.add_text(
            "Early / mid / late information is the fraction of each curve's squared sensitivity.",
            wrap=TOKENS.wraps.map,
        )
        dpg.add_group(tag="pyecr_sensitivity_table_container")

    def _toggle_sensitivity_view(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._update_sensitivity_plot()

    def _update_sensitivity_plot(self) -> None:
        dpg = self._require_dpg()
        if not dpg.does_item_exist("pyecr_sensitivity_y_axis"):
            return
        for tag in getattr(self, "_sensitivity_series_tags", ()):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self._sensitivity_series_tags = ()
        presentation = self.controller.state.diagnostics_presentation
        section = None if presentation is None else getattr(presentation, "sensitivity", None)
        if section is None or not section.available:
            return
        normalized = bool(dpg.get_value("pyecr_sensitivity_normalized"))
        dpg.configure_item(
            "pyecr_sensitivity_y_axis",
            label="Normalized sensitivity (unit peak)" if normalized
            else "∂y/∂log10(parameter)",
        )
        times = list(presentation.time_s)
        tags = []
        for index, curve in enumerate(section.curves):
            tag = f"pyecr_sensitivity_series_{index}"
            values = curve.normalized_values if normalized else curve.values
            dpg.add_line_series(
                times, list(values), label=curve.label,
                parent="pyecr_sensitivity_y_axis", tag=tag,
            )
            tags.append(tag)
        self._sensitivity_series_tags = tuple(tags)
        if tags:
            dpg.fit_axis_data("pyecr_sensitivity_x_axis")
            dpg.fit_axis_data("pyecr_sensitivity_y_axis")

    def _run_page(self, sender: Any, app_data: Any, user_data: Any) -> None:
        page = GuiPage(str(user_data))
        try:
            binding=None
            if page is GuiPage.FIT and getattr(self,'_r05_controls_ready',False):
                binding=self._r05_control_binding()
                source=self.controller.analysis_session.dataset
                recipe=self.controller.analysis_session.preprocessing.config
            task_id=self.controller.run_page(page,**({} if binding is None else {'fit_control_binding':binding}))
            if binding is not None:
                self._record_r05_legacy_run(task_id,binding,source,recipe)
        except Exception as exc:
            self.controller.record_callback_exception(page, exc, traceback.format_exc())
        self._refresh_widgets()

    def _cancel_fit(self, sender: Any, app_data: Any, user_data: Any) -> None:
        try:
            self.controller.cancel_page(GuiPage.FIT)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.FIT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _run_advanced_diagnostics(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data
        try:
            method = str(user_data)
            from pyecr.contracts import parse_confidence_percent
            from pyecr.uncertainty import AdvancedUncertaintyConfig
            dpg=self._require_dpg()
            level=parse_confidence_percent(dpg.get_value('pyecr_fit_confidence_percent'))
            settings=self._r07_settings() if getattr(self,'_r07_ready',False) else AdvancedUncertaintyConfig(enabled=True)
            if method == "experimental_m1":
                dpg = self._require_dpg()
                self.controller.run_advanced_diagnostics(
                    method, seed=int(dpg.get_value("pyecr_experimental_m1_seed")),
                    acknowledge_exchangeable_residuals=bool(dpg.get_value("pyecr_experimental_m1_ack")),
                    confidence_level=level,
                )
            else:
                self.controller.run_advanced_diagnostics(method,confidence_level=level,options=settings,
                    seed=int(dpg.get_value('pyecr_advanced_seed')) if getattr(self,'_r07_ready',False) else None)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DIAGNOSTICS, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _save_experimental_m1(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            self.controller.save_experimental_m1_uncertainty(
                str(self._require_dpg().get_value("pyecr_experimental_m1_path"))
            )
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.DIAGNOSTICS, exc, traceback.format_exc())
        self._refresh_widgets()

    def _reopen_experimental_m1(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            self.controller.reopen_experimental_m1_uncertainty(
                str(self._require_dpg().get_value("pyecr_experimental_m1_path"))
            )
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.DIAGNOSTICS, exc, traceback.format_exc())
        self._refresh_widgets()

    def _cancel_diagnostics(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.cancel_page(GuiPage.DIAGNOSTICS)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DIAGNOSTICS, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _show_data_file_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        self._start_data_file_picker()

    def _new_project(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            self.controller.new_project()
            self._reset_data_render_state()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.PROJECT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _show_project_open_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        dpg=self._require_dpg()
        if not dpg.get_value('pyecr_project_open_path') and self.controller.state.project_path:
            dpg.set_value('pyecr_project_open_path',self.controller.state.project_path)
        dpg.set_value('pyecr_project_open_reason','Enter a full path, or browse and select a directory row.')
        dpg.show_item('pyecr_project_open_window')

    def _browse_project_directory(self, *args) -> None:
        self._require_dpg().hide_item('pyecr_project_open_window')
        self._require_dpg().show_item('pyecr_project_open_dialog')

    def _open_project_from_path(self, *args) -> None:
        dpg=self._require_dpg()
        try:
            path=_project_directory_text(dpg.get_value('pyecr_project_open_path'),require_absolute=True)
        except ValueError as exc:
            dpg.set_value('pyecr_project_open_reason',str(exc))
            return
        self._open_project_at_path(path)

    def _show_project_save_as_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        self._require_dpg().show_item("pyecr_project_save_as_dialog")

    def _open_project_from_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        try:
            path=_project_directory_text(_directory_dialog_path(app_data))
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.PROJECT,exc,traceback.format_exc())
            self._require_dpg().set_value('pyecr_project_open_reason',str(exc))
            self._require_dpg().show_item('pyecr_project_open_window')
            self._refresh_widgets()
            return
        self._open_project_at_path(path)

    def _open_project_at_path(self, path: str) -> None:
        """One explicit request, shared by the text entry and directory picker."""
        dpg=self._require_dpg()
        try:
            self.controller.open_project(path)
            self._reset_data_render_state()
            dpg.hide_item('pyecr_project_open_window')
            dpg.set_value('pyecr_project_open_reason','')
        except UnitConfirmationRequired as exc:
            self._pending_unit_project_path = path
            dpg.hide_item('pyecr_project_open_window')
            dpg.set_value("pyecr_project_unit_warning", str(exc))
            dpg.show_item("pyecr_project_unit_confirmation")
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.PROJECT, exc, traceback.format_exc()
            )
            dpg.set_value('pyecr_project_open_reason',str(exc))
        self._refresh_widgets()

    def _save_project(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        if self.controller.state.project_path is None:
            self._require_dpg().show_item("pyecr_project_save_as_dialog")
            return
        try:
            self.controller.save_project()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.PROJECT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _save_project_as_from_dialog(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        try:
            self.controller.save_project(_directory_dialog_path(app_data))
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.PROJECT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _make_project_portable(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.make_project_portable()
            self._reset_data_render_state()
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.PROJECT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _reset_data_render_state(self) -> None:
        dpg = self._require_dpg()
        self._rendered_import_preview = None
        self._rendered_raw_dataset = None
        self._rendered_processed_dataset = None
        self._rendered_processing_config = None
        self._pending_stability_suggestion = None
        self._rendered_stability_suggestion = None
        self._pending_t0_suggestion = None
        self._rendered_t0_suggestion = None
        self._t0_preview_message = (
            "Suggest t0 to preview the detected onset; Accept records a preprocessing step."
        )
        self._selected_time_column_index = None
        self._selected_signal_column_index = None
        for tag in (
            "pyecr_data_raw_series", "pyecr_data_processed_series",
            "pyecr_data_stability_start_series", "pyecr_data_stability_end_series",
            "pyecr_data_t0_series",
        ):
            dpg.set_value(tag, [[], []])
        dpg.set_value("pyecr_data_source_path", "")
        dpg.set_value("pyecr_data_preview_summary", "No file selected.")
        dpg.set_value("pyecr_data_processing_history", "No preprocessing steps.")
        if dpg.does_item_exist(_DATA_PREVIEW_TABLE_TAG):
            dpg.delete_item(_DATA_PREVIEW_TABLE_TAG)

    def _select_data_file(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        try:
            path = _file_dialog_path(app_data)
            self._selected_time_column_index = None
            self._selected_signal_column_index = None
            header_mode, header_row_index = self._data_header_options()
            text_parse_config = self._data_text_parse_config()
            self.controller.preview_data_file(
                path,
                header_mode=header_mode,
                header_row_index=header_row_index,
                text_parse_config=text_parse_config,
            )
            self._rendered_import_preview = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_data_sheet(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        preview = self.controller.session.import_preview
        if preview is None or not preview.sheet_names:
            return
        try:
            header_mode, header_row_index = self._data_header_options()
            text_parse_config = self._data_text_parse_config()
            self.controller.preview_data_file(
                preview.source.source_path,
                sheet_name=str(app_data),
                header_mode=header_mode,
                header_row_index=header_row_index,
                text_parse_config=text_parse_config,
            )
            self._rendered_import_preview = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_data_header_mode(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        preview = self.controller.session.import_preview
        mode, row_index = self._data_header_options()
        text_parse_config = self._data_text_parse_config()
        dpg.configure_item(
            "pyecr_data_header_row_index",
            enabled=mode is HeaderMode.ROW_INDEX,
        )
        if preview is None:
            return
        try:
            self.controller.preview_data_file(
                preview.source.source_path,
                sheet_name=preview.active_sheet,
                header_mode=mode,
                header_row_index=row_index,
                text_parse_config=text_parse_config,
            )
            self._rendered_import_preview = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_data_text_parse_mode(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        preview = self.controller.session.import_preview
        try:
            config = self._data_text_parse_config()
            dpg.configure_item(
                "pyecr_data_custom_delimiter",
                show=config.mode is DelimiterMode.CUSTOM,
            )
            if preview is None or preview.source.format in {"xlsx", "xls"}:
                return
            header_mode, header_row_index = self._data_header_options()
            self.controller.preview_data_file(
                preview.source.source_path,
                sheet_name=preview.active_sheet,
                header_mode=header_mode,
                header_row_index=header_row_index,
                text_parse_config=config,
            )
            self._rendered_import_preview = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_data_mapping_column(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender
        preview = self.controller.session.import_preview
        if preview is None:
            return
        try:
            ref = preview.column_ref_for_display(str(app_data))
            if str(user_data) == "time":
                self._selected_time_column_index = ref.index
            else:
                self._selected_signal_column_index = ref.index
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )

    def _data_header_options(self) -> tuple[HeaderMode, int | None]:
        dpg = self._require_dpg()
        mode_name = str(dpg.get_value("pyecr_data_header_mode"))
        try:
            mode = HeaderMode[mode_name]
        except KeyError as exc:
            raise ValueError(f"unknown Header mode: {mode_name}") from exc
        row_index = None
        if mode is HeaderMode.ROW_INDEX:
            row_index = int(dpg.get_value("pyecr_data_header_row_index")) - 1
        return mode, row_index

    def _data_text_parse_config(self) -> TextParseConfig:
        dpg = self._require_dpg()
        mode_name = str(dpg.get_value("pyecr_data_text_parse_mode"))
        try:
            mode = DelimiterMode[mode_name]
        except KeyError as exc:
            raise ValueError(f"unknown text parsing mode: {mode_name}") from exc
        delimiter = None
        if mode is DelimiterMode.CUSTOM:
            delimiter = str(dpg.get_value("pyecr_data_custom_delimiter"))
        return TextParseConfig(mode=mode, custom_delimiter=delimiter)

    def _import_previewed_data(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        preview = self.controller.session.import_preview
        try:
            if preview is None:
                raise RuntimeError("select and preview a data file before import")
            signal_type = str(dpg.get_value("pyecr_data_signal_type"))
            signal_unit = str(dpg.get_value("pyecr_data_signal_unit")).strip()
            if signal_unit == "Custom…":
                signal_unit = str(dpg.get_value("pyecr_data_custom_unit")).strip()
                if not signal_unit:
                    raise ValueError("enter a custom signal unit before import")
            validate_pair(signal_type, signal_unit)
            time_ref = preview.column_ref_for_display(
                str(dpg.get_value("pyecr_data_time_column"))
            )
            signal_ref = preview.column_ref_for_display(
                str(dpg.get_value("pyecr_data_signal_column"))
            )
            mapping = ImportMapping(
                time_column_index=time_ref.index,
                signal_column_index=signal_ref.index,
                time_unit=str(dpg.get_value("pyecr_data_time_unit")),
                signal_type=signal_type,
                signal_unit=signal_unit or None,
                sheet_name=preview.active_sheet,
                header_mode=preview.requested_header_mode,
                header_row_index=(
                    preview.header_row_index
                    if preview.requested_header_mode is HeaderMode.ROW_INDEX
                    else None
                ),
                text_parse_config=(
                    preview.requested_text_parse_config or TextParseConfig()
                ),
            )
            self.controller.import_previewed_data(mapping)
            self._rendered_raw_dataset = None
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_processing_operation(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        operation = operation_id(app_data)
        example = _PROCESSING_PARAMETER_EXAMPLES.get(operation, {})
        self._require_dpg().set_value(
            "pyecr_data_processing_parameters",
            json.dumps(example, ensure_ascii=False),
        )
        self._pending_typed_preview = None
        self._signed_review = None
        self._show_typed_processing_fields(operation)
        self._refresh_widgets()

    def _show_typed_processing_fields(self, operation: str) -> None:
        dpg = self._require_dpg()
        for field in ("to_unit", "path_length_cm", "cross_section_area_cm2", "t0", "start_time", "end_time", "raw_indices", "mode"):
            dpg.configure_item(f"pyecr_typed_group_{field}", show=field in FORM_FIELDS.get(operation, ()))
        dpg.configure_item("pyecr_signed_controls", show=operation == SIGNED_OPERATION)

    def _preview_typed_preprocessing(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        if operation_id(dpg.get_value("pyecr_data_processing_operation")) == SIGNED_OPERATION:
            self._preview_signed_normalization()
            return
        self._pending_typed_preview = None
        for field in ("to_unit", "path_length_cm", "cross_section_area_cm2", "t0", "start_time", "end_time", "raw_indices", "mode"):
            dpg.set_value(f"pyecr_typed_error_{field}", "")
        try:
            pipeline = self.controller.session.preprocessing
            if pipeline is None:
                raise ValueError("Import data before preprocessing")
            operation = str(dpg.get_value("pyecr_data_processing_operation"))
            fields = (self._direct_typed_values(operation) if getattr(self, '_direct_preprocess_ready', False) else
                      {name: dpg.get_value(f"pyecr_typed_{name}") for name in FORM_FIELDS[operation]})
            parameters = generic_parameters(operation, fields)
            if operation == 'unmask_points':
                from pyecr.gui.preprocessing_workflow import reviewable_mask_indices
                if not parameters['raw_indices'] or not set(parameters['raw_indices']).issubset(reviewable_mask_indices(pipeline.config)):
                    raise ValueError('Choose currently masked original indices; crop/downsample exclusions remain.')
                parameters['reason'] = 'user reviewed and unmasked'
            step = ProcessingStep.create(operation, parameters)
            proposed = pipeline.config.append(step).replay(pipeline.source)
            self._pending_typed_preview = (step, proposed, pipeline.dataset, pipeline.revision,
                                           self.controller.analysis_session.data_revision)
            dpg.set_value("pyecr_data_processing_parameters", json.dumps(step.to_dict(), ensure_ascii=False, indent=2))
            dpg.set_value("pyecr_typed_error", "Fields valid. Review the overlay and evidence before Accept.")
        except Exception as exc:
            message = str(exc)
            dpg.set_value("pyecr_typed_error", f"Input error: {message}")
            for field in ("to_unit", "path_length_cm", "cross_section_area_cm2", "t0", "start_time", "end_time", "raw_indices", "mode"):
                if field in message:
                    dpg.set_value(f"pyecr_typed_error_{field}", message)
        self._refresh_widgets()

    def _accept_typed_preprocessing(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            if getattr(self, '_direct_preprocess_ready', False):
                self._validate_direct_preview('typed')
            pending = getattr(self, "_pending_typed_preview", None)
            pipeline = self.controller.session.preprocessing
            if pending is None or pipeline is None:
                raise ValueError("Preview before Accept")
            step, _, dataset, revision, data_revision = pending
            if (dataset is not pipeline.dataset or revision != pipeline.revision
                    or data_revision != self.controller.analysis_session.data_revision):
                raise ValueError("Preview is stale; preview again")
            if step.operation == SIGNED_OPERATION:
                self._validate_signed_accept()
            self.controller.apply_preprocessing(step)
            self._pending_typed_preview = None
            self._signed_review = None
            self._signed_message = "Accepted explicitly. Review the source/profile and Fit compatibility; no Fit was launched."
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.DATA, exc, traceback.format_exc())
        self._refresh_widgets()

    def _reject_typed_preprocessing(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._pending_typed_preview = None
        self._signed_review = None
        self._signed_message = "Rejected. Data and recipe unchanged."
        self._require_dpg().set_value("pyecr_typed_preview_evidence", "Rejected. Data and recipe unchanged.")
        self._refresh_widgets()

    def _render_typed_preprocessing(self, dpg: Any, pipeline: Any) -> None:
        if not hasattr(dpg, "does_item_exist") or not dpg.does_item_exist("pyecr_typed_processing_controls"):
            return
        pending = getattr(self, "_pending_typed_preview", None)
        if pending is not None:
            _, proposed, dataset, revision, data_revision = pending
            if (pipeline is None or dataset is not pipeline.dataset or revision != pipeline.revision
                    or data_revision != self.controller.analysis_session.data_revision):
                pending = self._pending_typed_preview = None
                dpg.set_value("pyecr_typed_error", "Preview expired after a data revision; preview again.")
        if getattr(self,'_direct_preprocess_ready',False) and not self._direct_binding_current():
            pending=None
        dpg.configure_item("pyecr_data_accept_processing", enabled=pending is not None)
        dpg.configure_item("pyecr_data_reject_processing", enabled=pending is not None)
        dpg.set_value("pyecr_typed_current_unit", "Time ranges use the current processed unit: " +
                      (str(pipeline.dataset.metadata.get("time_unit", "s")) if pipeline else "import data first"))
        if pending is None:
            self._typed_preview_render_key=None
            dpg.set_value("pyecr_typed_preview_series", [[], []])
            dpg.configure_item("pyecr_typed_preview_series", show=False)
            return
        step, proposed, _, _, _ = pending
        render_key=(id(pipeline.dataset),id(proposed),id(step))
        if render_key==getattr(self,'_typed_preview_render_key',None):return
        self._typed_preview_render_key=render_key
        mixed_units = preview_uses_separate_axes(pipeline.dataset,proposed)
        if step.operation == SIGNED_OPERATION or mixed_units:
            dpg.set_value("pyecr_typed_preview_series", [[], []])
            dpg.configure_item("pyecr_typed_preview_series", show=False)
            dpg.set_value("pyecr_typed_preview_evidence", ("Signed-DCT review and overlay use separate input / unit-1 axes. Accept is required to commit."
                if step.operation == SIGNED_OPERATION else preview_evidence(pipeline.source, pipeline.dataset, proposed, step, pipeline.config)
                + '\nDifferent units or time origin: review separate labeled axes.'))
            return
        if step.operation.startswith('choose_') and step.operation.endswith('_stability_window'):
            dpg.set_value('pyecr_typed_preview_series',[[],[]]);dpg.configure_item('pyecr_typed_preview_series',show=False)
            dpg.set_value('pyecr_typed_preview_evidence',preview_evidence(pipeline.source,pipeline.dataset,proposed,step,pipeline.config))
            return
        dpg.set_value("pyecr_typed_preview_series", [proposed.processed_time.tolist(), proposed.processed_signal.tolist()])
        dpg.configure_item("pyecr_typed_preview_series", show=True)
        dpg.set_value("pyecr_typed_preview_evidence", preview_evidence(pipeline.source, pipeline.dataset, proposed, step, pipeline.config))

    def _select_stability_kind(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        self._pending_stability_suggestion = None
        self._stability_preview_message = (
            "Plateau selection changed. Suggest a window to preview it."
        )
        self._refresh_widgets()

    def _select_guided_refit_source(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        manual = dpg.get_value("pyecr_guided_refit_source") == "Manual value"
        dpg.configure_item("pyecr_guided_refit_manual_value", show=manual)

    def _run_guided_refit(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data
        dpg = self._require_dpg()
        try:
            source_label = str(dpg.get_value("pyecr_guided_refit_source"))
            if source_label == "Parent fitted value":
                value_source, fixed_value = "parent_fit", None
            elif source_label == "Manual value":
                value_source = "manual"
                fixed_value = float(dpg.get_value("pyecr_guided_refit_manual_value"))
            else:
                raise ValueError("select a fixed value source")
            self.controller.run_guided_refit(
                fixed_parameter=str(user_data),
                value_source=value_source,
                fixed_value=fixed_value,
            )
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.FIT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _cancel_guided_refit(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.cancel_page(GuiPage.FIT)
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.FIT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _rollback_guided_refit(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.rollback_guided_refit()
            self._rendered_fit_presentation = None
            self._rendered_diagnostics_presentation = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.FIT, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _current_pending_stability_suggestion(
        self,
    ) -> tuple[Any, EcrDataset, int, int] | None:
        pending = self._pending_stability_suggestion
        if pending is None:
            return None
        suggestion, dataset, revision, data_revision = pending
        pipeline = self.controller.session.preprocessing
        if (
            self.controller.session.dataset is not dataset
            or pipeline is None
            or pipeline.revision != revision
            or self.controller.analysis_session.data_revision != data_revision
        ):
            self._pending_stability_suggestion = None
            self._stability_preview_message = (
                "Window preview expired after the dataset changed. Suggest again."
            )
            return None
        return suggestion, dataset, revision, data_revision

    def _suggest_stability_window(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            dataset = self.controller.session.dataset
            pipeline = self.controller.session.preprocessing
            if dataset is None or pipeline is None:
                raise RuntimeError("import a dataset before suggesting windows")
            kind = str(
                self._require_dpg().get_value("pyecr_data_stability_kind")
            ).strip().lower()
            if kind not in {"initial", "final"}:
                raise ValueError("select Initial or Final")
            from pyecr.preprocessing.quality import suggest_stability_window

            self._pending_stability_suggestion = None
            self._stability_preview_message = "Suggestion failed. Data unchanged."
            try:
                minimum_input = self._require_dpg().get_value("pyecr_data_stability_min_points")
            except KeyError:  # Older headless preview doubles predate the typed field.
                minimum_input = "5"
            minimum = integer_field(minimum_input, "minimum_points", 3)
            suggestion = suggest_stability_window(dataset, kind=kind, min_points=minimum)
            if suggestion is None:
                self._stability_preview_message = (
                    f"No suitable {kind} stability window was found. Data unchanged."
                )
            else:
                if suggestion.kind != kind:
                    raise ValueError("suggested stability window has the wrong kind")
                self._pending_stability_suggestion = (
                    suggestion, dataset, pipeline.revision,
                    self.controller.analysis_session.data_revision,
                )
                self._stability_preview_message = ""
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _accept_stability_window(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            if getattr(self, '_direct_preprocess_ready', False):
                self._validate_direct_preview('stability')
            pending = self._current_pending_stability_suggestion()
            if pending is None:
                raise RuntimeError("window preview is no longer current; suggest again")
            suggestion, _, _, _ = pending
            kind = str(suggestion.kind)
            if kind not in {"initial", "final"}:
                raise ValueError("suggested stability window has an invalid kind")
            step = ProcessingStep.create(
                f"choose_{kind}_stability_window",
                {
                    "start_time": float(suggestion.start_time),
                    "end_time": float(suggestion.end_time),
                },
            )
            self.controller.apply_preprocessing(step)
            self._pending_stability_suggestion = None
            self._stability_preview_message = (
                f"Accepted {kind} stability window as a preprocessing step."
            )
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _reject_stability_window(
        self, sender: Any, app_data: Any, user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        self._pending_stability_suggestion = None
        self._stability_preview_message = "Window suggestion rejected. Data unchanged."
        self._refresh_widgets()

    def _render_stability_window_preview(
        self, dpg: Any, dataset: EcrDataset | None,
    ) -> None:
        item_exists = getattr(dpg, "does_item_exist", None)
        if callable(item_exists) and not item_exists("pyecr_data_stability_preview"):
            return
        pending = self._current_pending_stability_suggestion()
        dpg.configure_item(
            "pyecr_data_stability_suggest", enabled=dataset is not None,
        )
        for tag in ("pyecr_data_stability_accept", "pyecr_data_stability_reject"):
            dpg.configure_item(tag, enabled=pending is not None)
        if pending is None:
            dpg.set_value(
                "pyecr_data_stability_preview",
                self._stability_preview_message if dataset is not None else
                "Import a dataset before suggesting stability windows.",
            )
            if self._rendered_stability_suggestion is not None:
                for tag in (
                    "pyecr_data_stability_start_series",
                    "pyecr_data_stability_end_series",
                ):
                    dpg.set_value(tag, [[], []])
                    dpg.configure_item(tag, show=False)
                self._rendered_stability_suggestion = None
            return
        suggestion, current_dataset, _, _ = pending
        unit = str(current_dataset.metadata.get("time_unit", "s"))
        dpg.set_value(
            "pyecr_data_stability_preview",
            f"{suggestion.kind.title()} window: "
            f"{suggestion.start_time:.6g}–{suggestion.end_time:.6g} {unit} "
            f"({suggestion.n_points} points)\n"
            f"Score {suggestion.score:.3f} | slope {suggestion.slope:.3g} | "
            f"std {suggestion.std:.3g} | confidence {suggestion.confidence_class}\n"
            f"{suggestion.reason}\n"
            "Preview only. Accept records a preprocessing step.",
        )
        if self._rendered_stability_suggestion is suggestion:
            return
        values = current_dataset.processed_signal
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.05, abs(high) * 1.0e-6, 1.0e-9)
        for tag, time in (
            ("pyecr_data_stability_start_series", suggestion.start_time),
            ("pyecr_data_stability_end_series", suggestion.end_time),
        ):
            dpg.set_value(
                tag, [[float(time), float(time)], [low - padding, high + padding]],
            )
            dpg.configure_item(tag, show=True)
        dpg.fit_axis_data("pyecr_data_processed_y_axis")
        self._rendered_stability_suggestion = suggestion

    def _current_pending_t0_suggestion(
        self,
    ) -> tuple[Any, EcrDataset, int, int] | None:
        pending = self._pending_t0_suggestion
        if pending is None:
            return None
        suggestion, dataset, revision, data_revision = pending
        pipeline = self.controller.session.preprocessing
        if (
            self.controller.session.dataset is not dataset
            or pipeline is None
            or pipeline.revision != revision
            or self.controller.analysis_session.data_revision != data_revision
        ):
            self._pending_t0_suggestion = None
            self._t0_preview_message = (
                "t0 preview expired after the dataset changed. Suggest again."
            )
            return None
        return suggestion, dataset, revision, data_revision

    def _suggest_t0(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            dataset = self.controller.session.dataset
            pipeline = self.controller.session.preprocessing
            if dataset is None or pipeline is None:
                raise RuntimeError("import a dataset before suggesting t0")
            from pyecr.boundary.assistant import suggest_t0

            self._pending_t0_suggestion = None
            self._t0_preview_message = "No credible t0 suggestion. Data unchanged."
            suggestion = suggest_t0(
                dataset,
                self.controller.current_analysis_config,
                self.controller.analysis_session.fit_result
                if not self.controller.analysis_session.fit_stale else None,
            )
            if suggestion is not None:
                self._pending_t0_suggestion = (
                    suggestion, dataset, pipeline.revision,
                    self.controller.analysis_session.data_revision,
                )
                self._t0_preview_message = ""
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
            self._t0_preview_message = f"t0 suggestion failed: {exc}. Data unchanged."
        self._refresh_widgets()

    def _accept_t0(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            if getattr(self, '_direct_preprocess_ready', False):
                self._validate_direct_preview('t0')
            pending = self._current_pending_t0_suggestion()
            if pending is None:
                raise RuntimeError("t0 preview is no longer current; suggest again")
            suggestion, _, _, _ = pending
            self.controller.apply_preprocessing(ProcessingStep.create(
                "set_manual_t0", {"t0": float(suggestion.t0_s)},
            ))
            self._pending_t0_suggestion = None
            self._t0_preview_message = (
                "Accepted t0 as a replayable preprocessing step. "
                "Review Fit readiness; crop or mask negative times explicitly if needed."
            )
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _reject_t0(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._pending_t0_suggestion = None
        self._t0_preview_message = "t0 suggestion rejected. Data unchanged."
        self._refresh_widgets()

    def _render_t0_preview(self, dpg: Any, dataset: EcrDataset | None) -> None:
        pending = self._current_pending_t0_suggestion()
        dpg.configure_item("pyecr_data_t0_suggest", enabled=dataset is not None)
        for tag in ("pyecr_data_t0_accept", "pyecr_data_t0_reject"):
            dpg.configure_item(tag, enabled=pending is not None)
        if pending is None:
            dpg.set_value(
                "pyecr_data_t0_preview",
                self._t0_preview_message if dataset is not None else
                "Import a dataset before suggesting t0.",
            )
            if self._rendered_t0_suggestion is not None:
                dpg.set_value("pyecr_data_t0_series", [[], []])
                dpg.configure_item("pyecr_data_t0_series", show=False)
                self._rendered_t0_suggestion = None
            return
        suggestion, current_dataset, _, _ = pending
        ratio = suggestion.flush_to_material_ratio
        ratio_text = "N/A" if ratio is None else f"{ratio:.3g}"
        material = suggestion.tau_material_s
        material_text = "N/A" if material is None else f"{material:.3g} s"
        flush = suggestion.tau_flush_s
        flush_text = "N/A" if flush is None else f"{flush:.3g} s"
        dpg.set_value(
            "pyecr_data_t0_preview",
            f"Suggested t0 = {suggestion.t0_s:.6g} s (current processed axis); "
            f"score {suggestion.score:.3f}, confidence {suggestion.confidence_class}.\n"
            f"τ_material = {material_text}; τ_flush = {flush_text}; "
            f"τ_flush/τ_material = {ratio_text}.\n"
            f"{suggestion.reason}\nBoundary: {suggestion.boundary_status}. "
            + ("Pre-onset points will become negative: explicitly crop or mask "
               "before Fit. " if suggestion.requires_pre_onset_crop else "")
            + "Preview only; Accept records set_manual_t0.",
        )
        if self._rendered_t0_suggestion is suggestion:
            return
        values = current_dataset.processed_signal
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.05, abs(high) * 1.0e-6, 1.0e-9)
        dpg.set_value(
            "pyecr_data_t0_series",
            [[float(suggestion.t0_s)] * 2, [low - padding, high + padding]],
        )
        dpg.configure_item("pyecr_data_t0_series", show=True)
        dpg.fit_axis_data("pyecr_data_processed_y_axis")
        self._rendered_t0_suggestion = suggestion

    def _apply_preprocessing(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        try:
            parameters = json.loads(
                str(dpg.get_value("pyecr_data_processing_parameters"))
            )
            if not isinstance(parameters, dict):
                raise ValueError("preprocessing parameters JSON must be an object")
            step = ProcessingStep.create(
                str(dpg.get_value("pyecr_data_processing_operation")),
                parameters,
            )
            self.controller.apply_preprocessing(step)
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _undo_preprocessing(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.undo_preprocessing()
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _reset_preprocessing(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, app_data, user_data
        try:
            self.controller.reset_preprocessing()
            self._rendered_processed_dataset = None
            self._rendered_processing_config = None
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.DATA, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_model_descriptor(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        try:
            from pyecr.gui.model_display import model_id_for_display
            self._render_model_controls(model_id_for_display(app_data))
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.MODEL, exc, traceback.format_exc()
            )
        self._refresh_widgets()

    def _select_model_step_type(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        enabled = str(app_data) == "custom"
        dpg = self._require_dpg()
        dpg.configure_item("pyecr_model_sign_h", enabled=enabled)
        dpg.configure_item("pyecr_model_sign_o", enabled=enabled)
        dpg.configure_item('pyecr_model_sign_group',show=enabled)

    def _select_boundary_mode(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        del sender, user_data
        mode = str(app_data)
        dpg = self._require_dpg()
        dpg.configure_item(
            "pyecr_boundary_flush_time_s",
            show=mode == "exponential_flush",
        )
        dpg.configure_item(
            "pyecr_boundary_measured_group",
            show=mode == "measured_history",
        )

    def _apply_model_configuration(
        self,
        sender: Any,
        app_data: Any,
        user_data: Any,
    ) -> None:
        from pyecr.model_display import display_model_text

        del sender, app_data, user_data
        dpg = self._require_dpg()
        try:
            descriptor = self.controller.model_descriptor(
                str(dpg.get_value("pyecr_model_id"))
            )
            parameters = {
                parameter.name: ModelParameterValue(
                    value=float(
                        dpg.get_value(_model_parameter_tag(parameter.name, "value"))
                    ),
                    lower_bound=float(
                        dpg.get_value(_model_parameter_tag(parameter.name, "lower"))
                    ),
                    upper_bound=float(
                        dpg.get_value(_model_parameter_tag(parameter.name, "upper"))
                    ),
                    free=self._parameter_is_free(parameter.name),
                )
                for parameter in descriptor.parameters
            }
            fixed_sources={}
            if descriptor.model_id.value in {"M1_SINGLE_DK","M3_DUAL_HO_DK"} and dpg.get_value("pyecr_fixed_d_enabled"):
                for name in (("D_cm2_s",) if descriptor.model_id.value=="M1_SINGLE_DK" else ("D_H_cm2_s","D_O_cm2_s")):
                    control=parameters[name]
                    parameters[name]=ModelParameterValue(control.value,control.lower_bound,control.upper_bound,False)
                    dpg.set_value(_model_parameter_tag(name,"free"),False)
                    kind=str(dpg.get_value("pyecr_fixed_d_kind_"+name))
                    fixed_sources[name]={"value_cm2_s":control.value,"unit":"cm^2/s","kind":kind,
                        "reference":str(dpg.get_value("pyecr_fixed_d_reference_"+name)),"uncertainty":str(dpg.get_value("pyecr_fixed_d_uncertainty_"+name)),
                        "independence":"same_data_plugin" if kind=="parent_fit" else "unknown"}
            step_type = str(dpg.get_value("pyecr_model_step_type"))
            boundary_mode = str(dpg.get_value("pyecr_model_boundary_mode"))
            measured_mode = boundary_mode == "measured_history"
            resample_step = (
                float(dpg.get_value("pyecr_boundary_resample_step_s"))
                if measured_mode
                else 0.0
            )
            values = ModelFormValues(
                model_id=descriptor.model_id,
                full_dims_cm=tuple(
                    float(dpg.get_value(f"pyecr_model_dim_{axis}"))
                    for axis in "xyz"
                ),
                active_dimensions=tuple(
                    bool(dpg.get_value(f"pyecr_model_active_{axis}"))
                    for axis in "xyz"
                ),
                surface_mode=str(dpg.get_value("pyecr_model_surface_mode")),
                boundary_mode=boundary_mode,
                step_type=step_type,
                sign_H=(
                    float(dpg.get_value("pyecr_model_sign_h"))
                    if step_type == "custom"
                    else None
                ),
                sign_O=(
                    float(dpg.get_value("pyecr_model_sign_o"))
                    if step_type == "custom"
                    else None
                ),
                parameters=parameters,
                relative_tolerance=float(
                    dpg.get_value("pyecr_model_relative_tolerance")
                ),
                absolute_tolerance=self.controller.default_model_form(descriptor.model_id).absolute_tolerance if descriptor.model_id.value in {"M7_SINGLE_K","M8_DUAL_K"} else float(
                    dpg.get_value("pyecr_model_absolute_tolerance")
                ),
                max_terms=self.controller.default_model_form(descriptor.model_id).max_terms if descriptor.model_id.value in {"M7_SINGLE_K","M8_DUAL_K"} else int(dpg.get_value("pyecr_model_max_terms")),
                max_nfev=int(dpg.get_value("pyecr_model_max_nfev")),
                random_seed=(int(str(dpg.get_value('pyecr_model_random_seed') or '').strip()) if dpg.does_item_exist('pyecr_model_random_seed') and str(dpg.get_value('pyecr_model_random_seed') or '').strip() else None),
                loss=str(dpg.get_value("pyecr_model_loss")),
                model_specific={**config_selection(
                    str(dpg.get_value("pyecr_model_local_solver")),
                    "warn" if descriptor.model_id.value in {"M7_SINGLE_K","M8_DUAL_K"} else str(dpg.get_value("pyecr_model_dogbox_policy")),
                    self.controller.current_analysis_config.model_specific
                    if self.controller.current_analysis_config is not None
                    and self.controller.current_analysis_config.model_id == descriptor.model_id else None),
                    **({"fixed_d_sources":fixed_sources or None} if descriptor.model_id.value in {"M1_SINGLE_DK","M3_DUAL_HO_DK"} else {})},
                flush_time_s=(
                    float(dpg.get_value("pyecr_boundary_flush_time_s"))
                    if boundary_mode == "exponential_flush"
                    else None
                ),
                measured_time_s=(
                    _parse_numeric_sequence(
                        dpg.get_value("pyecr_boundary_measured_time_s"),
                        "measured boundary time",
                    )
                    if measured_mode
                    else ()
                ),
                measured_values=(
                    _parse_numeric_sequence(
                        dpg.get_value("pyecr_boundary_measured_values"),
                        "measured boundary values",
                    )
                    if measured_mode
                    else ()
                ),
                measured_signal_type=(
                    str(dpg.get_value("pyecr_boundary_measured_signal_type"))
                    if measured_mode
                    else None
                ),
                measured_initial_value=(
                    float(dpg.get_value("pyecr_boundary_measured_initial"))
                    if measured_mode
                    else None
                ),
                measured_final_value=(
                    float(dpg.get_value("pyecr_boundary_measured_final"))
                    if measured_mode
                    else None
                ),
                measured_value_unit=(
                    str(dpg.get_value("pyecr_boundary_measured_unit"))
                    if measured_mode
                    else "unspecified"
                ),
                resample_time_step_s=(
                    resample_step if measured_mode and resample_step > 0.0 else None
                ),
                max_internal_points=(
                    int(dpg.get_value("pyecr_boundary_max_internal_points"))
                    if measured_mode
                    else 200_000
                ),
            )
            from dataclasses import replace
            from pyecr.fitting.reproducibility import gui_seed_specific
            if dpg.does_item_exist('pyecr_model_random_seed'):
                values=replace(values,model_specific=gui_seed_specific(values.model_specific,dpg.get_value('pyecr_model_random_seed')))
            config = self.controller.apply_model_configuration(values)
            self._remember_applied_model_form()
            from pyecr.model_display import display_label
            dpg.set_value(
                "pyecr_model_apply_status",
                f"Validated and applied: {display_label(config.model_id)}",
            )
        except Exception as exc:
            self.controller.record_callback_exception(
                GuiPage.MODEL, exc, traceback.format_exc()
            )
            dpg.set_value(
                "pyecr_model_apply_status",
                display_model_text(f"Configuration rejected: {type(exc).__name__}: {exc}"),
            )
        self._refresh_widgets()

    def _set_dataset_name(self, sender: Any, app_data: Any, user_data: Any) -> None:
        self.controller.state.dataset_name = str(app_data).strip() or None
        self._refresh_summaries()

    def _refresh_project_widgets(self) -> None:
        dpg = self._require_dpg()
        snapshot = self.controller.session.project_snapshot
        has_dataset = self.controller.analysis_session.dataset is not None
        has_project = self.controller.state.project_path is not None
        dpg.set_value(
            "pyecr_project_path", self.controller.state.project_path or ""
        )
        dpg.configure_item("pyecr_project_save", enabled=has_dataset and has_project)
        dpg.configure_item("pyecr_project_save_as", enabled=has_dataset)
        dpg.configure_item(
            "pyecr_project_make_portable",
            enabled=(
                snapshot is not None and snapshot.raw_data_mode.value == "linked"
            ),
        )
        if snapshot is None:
            details = (
                "Unsaved analysis session. Import data, then use Save As to create "
                "a directory project."
                if has_dataset
                else "No project is open. Import data before the first save."
            )
        else:
            details = (
                f"Mode: {snapshot.raw_data_mode.value} | schema: "
                f"{snapshot.schema_version} | raw SHA256: "
                f"{snapshot.raw_data_sha256[:16]}… | PyECR"
            )
        dpg.set_value("pyecr_project_details", details)
        dpg.set_value(
            "pyecr_project_disabled_reason",
            "" if has_dataset and has_project else
            "Save requires a dataset and opened project; Save As requires a dataset. Make Portable requires a linked project.",
        )

    def _render_data_import(self) -> None:
        preview = self.controller.session.import_preview
        dpg = self._require_dpg()
        import_reason = (
            "Select a file, map two columns, and resolve parser/header errors."
            if preview is None else
            "Import disabled: select two columns and resolve parser/header errors."
        )
        can_import = bool(
            preview is not None and len(preview.column_refs) >= 2 and preview.parser_valid
        )
        set_action_availability(
            dpg, "pyecr_data_import", enabled=can_import,
            reason=import_reason,
        )
        if preview is not None and preview is not self._rendered_import_preview:
            dpg.set_value("pyecr_data_source_path", preview.source.source_path)
            dpg.set_value(
                "pyecr_data_preview_summary",
                f"{preview.row_count} rows | {preview.source.format.upper()} | "
                f"{preview.source.size_bytes} bytes | SHA256 "
                f"{preview.source.sha256[:16]}…",
            )
            dpg.set_value(
                "pyecr_data_header_mode",
                preview.requested_header_mode.name,
            )
            dpg.configure_item(
                "pyecr_data_header_row_index",
                enabled=preview.requested_header_mode is HeaderMode.ROW_INDEX,
            )
            if preview.requested_header_mode is HeaderMode.ROW_INDEX:
                dpg.set_value(
                    "pyecr_data_header_row_index",
                    int(preview.header_row_index or 0) + 1,
                )
            header_status = (
                f"Requested {preview.requested_header_mode.name}; "
                f"detected {preview.detected_header_mode.name}."
            )
            if preview.header_detection_message:
                header_status = (
                    f"{header_status} {preview.header_detection_message}"
                )
            dpg.set_value("pyecr_data_header_status", header_status)
            is_text = preview.requested_text_parse_config is not None
            dpg.configure_item("pyecr_data_text_parse_mode", enabled=is_text)
            if is_text:
                text_config = preview.requested_text_parse_config
                assert text_config is not None
                dpg.set_value("pyecr_data_text_parse_mode", text_config.mode.name)
                dpg.configure_item(
                    "pyecr_data_custom_delimiter",
                    show=text_config.mode is DelimiterMode.CUSTOM,
                    enabled=True,
                )
                if text_config.custom_delimiter is not None:
                    dpg.set_value(
                        "pyecr_data_custom_delimiter",
                        text_config.custom_delimiter,
                    )
                detected = (
                    "UNKNOWN"
                    if preview.detected_delimiter_mode is None
                    else preview.detected_delimiter_mode.name
                )
                parse_status = (
                    f"Detected parsing mode: {detected} | "
                    f"columns: {preview.parsed_column_count} | "
                    f"consistency: {preview.field_count_consistency}"
                )
                if preview.delimiter_detection_message:
                    parse_status = (
                        f"{parse_status} | {preview.delimiter_detection_message}"
                    )
                if preview.parse_warning:
                    parse_status = f"{parse_status} | WARNING: {preview.parse_warning}"
            else:
                dpg.configure_item(
                    "pyecr_data_custom_delimiter",
                    show=False,
                    enabled=False,
                )
                parse_status = (
                    "Excel workbook parsing | "
                    f"columns: {preview.parsed_column_count} | "
                    f"consistency: {preview.field_count_consistency}"
                )
            dpg.set_value("pyecr_data_text_parse_status", parse_status)
            suggestion = preview.suggested_time_column_index
            if suggestion is None:
                suggestion_text = "Time-column suggestion: none. Select explicitly."
            else:
                suggestion_ref = preview.column_ref(suggestion)
                suggestion_text = (
                    "Time-column suggestion: "
                    f"{suggestion_ref.display_name} (column {suggestion + 1}). "
                    "Suggestion only; current selection was not changed."
                )
            dpg.set_value("pyecr_data_time_suggestion", suggestion_text)
            columns = [ref.display_name for ref in preview.column_refs]
            for tag in ("pyecr_data_time_column", "pyecr_data_signal_column"):
                dpg.configure_item(tag, items=columns, enabled=bool(columns))
            if columns:
                if (
                    self._selected_time_column_index is None
                    or self._selected_time_column_index >= len(columns)
                ):
                    self._selected_time_column_index = 0
                if (
                    self._selected_signal_column_index is None
                    or self._selected_signal_column_index >= len(columns)
                ):
                    self._selected_signal_column_index = (
                        1 if len(columns) > 1 else 0
                    )
                dpg.set_value(
                    "pyecr_data_time_column",
                    preview.column_ref(
                        self._selected_time_column_index
                    ).display_name,
                )
                dpg.set_value(
                    "pyecr_data_signal_column",
                    preview.column_ref(
                        self._selected_signal_column_index
                    ).display_name,
                )
            if preview.sheet_names:
                dpg.configure_item(
                    "pyecr_data_sheet",
                    items=list(preview.sheet_names),
                    enabled=True,
                )
                dpg.set_value("pyecr_data_sheet", preview.active_sheet)
            else:
                dpg.configure_item(
                    "pyecr_data_sheet",
                    items=["No workbook sheets"],
                    enabled=False,
                )
                dpg.set_value("pyecr_data_sheet", "No workbook sheets")
            dpg.set_value(
                "pyecr_data_import_reason",
                "" if len(columns) >= 2 and preview.parser_valid else
                "Import disabled: select two columns and resolve parser/header errors.",
            )
            _replace_data_preview_table(dpg, preview)
            self._rendered_import_preview = preview

        pipeline = self.controller.session.preprocessing
        dataset = self.controller.session.dataset
        raw_dataset = pipeline.source if pipeline is not None else dataset
        if raw_dataset is not None and raw_dataset is not self._rendered_raw_dataset:
            simulation_kind = raw_dataset.metadata.get('source_kind')
            simulation_label = {'simulation_clean':'Simulated clean truth','simulation_observed':'Simulated observations'}.get(simulation_kind)
            dpg.configure_item('pyecr_data_raw_plot',label=simulation_label or 'Raw measurement')
            dpg.configure_item('pyecr_data_raw_series',label=simulation_label or 'Raw data',show=bool(raw_dataset.raw_time.size))
            dpg.configure_item('pyecr_preprocess_raw_plot',label=(simulation_label+' · original simulation units / exclusion reasons') if simulation_label else 'Raw context · original units / exclusion reasons')
            dpg.set_value(
                "pyecr_data_raw_series",
                [raw_dataset.raw_time.tolist(), raw_dataset.raw_signal.tolist()],
            )
            dpg.set_value("pyecr_preprocess_raw_series", [raw_dataset.raw_time.tolist(), raw_dataset.raw_signal.tolist()])
            dpg.configure_item(
                "pyecr_data_raw_x_axis",
                label=time_axis_label(raw_dataset.metadata, raw=True),
            )
            dpg.configure_item(
                "pyecr_data_raw_y_axis",
                label=signal_axis_label(raw_dataset.metadata, raw=True),
            )
            dpg.configure_item('pyecr_preprocess_raw_x_axis',label=time_axis_label(raw_dataset.metadata,raw=True))
            dpg.configure_item('pyecr_preprocess_raw_y_axis',label=signal_axis_label(raw_dataset.metadata,raw=True))
            dpg.fit_axis_data("pyecr_data_raw_x_axis")
            dpg.fit_axis_data("pyecr_data_raw_y_axis")
            dpg.fit_axis_data("pyecr_preprocess_raw_x_axis")
            dpg.fit_axis_data("pyecr_preprocess_raw_y_axis")
            self._rendered_raw_dataset = raw_dataset
        elif raw_dataset is None and self._rendered_raw_dataset is not None:
            clear_plot_series(dpg, ("pyecr_data_raw_series",), hide=True)
            clear_plot_series(dpg, ("pyecr_preprocess_raw_series",), hide=False)
            self._rendered_raw_dataset = None

        if pipeline is not None:
            display_config = pipeline.config
            exclusion_key = (id(pipeline.source), id(pipeline.config))
            if exclusion_key != getattr(self, "_rendered_exclusion_key", None):
                reasons = excluded_by_reason(pipeline.source, display_config)
                for category, operations in (("mask", ("mask_points", "outlier_mask")), ("crop", ("crop",)), ("downsample", ("adaptive_downsample",))):
                    indices = [i for operation in operations for i in reasons.get(operation, [])]
                    dpg.set_value(f"pyecr_raw_excluded_{category}", [pipeline.source.raw_time[indices].tolist(), pipeline.source.raw_signal[indices].tolist()])
                    dpg.configure_item(f"pyecr_raw_excluded_{category}",show=bool(indices))
                    dpg.set_value(f"pyecr_preprocess_excluded_{category}", [pipeline.source.raw_time[indices].tolist(), pipeline.source.raw_signal[indices].tolist()])
                self._rendered_exclusion_key = exclusion_key
        elif getattr(self, "_rendered_exclusion_key", None) is not None:
            for category in ("mask", "crop", "downsample"):
                dpg.set_value(f"pyecr_raw_excluded_{category}", [[], []])
                dpg.configure_item(f"pyecr_raw_excluded_{category}",show=False)
                dpg.set_value(f"pyecr_preprocess_excluded_{category}", [[], []])
            self._rendered_exclusion_key = None

        if dataset is not None and dataset is not self._rendered_processed_dataset:
            _update_processed_plot(dpg, dataset)
            self._rendered_processed_dataset = dataset
        elif dataset is None and self._rendered_processed_dataset is not None:
            clear_plot_series(dpg, ("pyecr_data_processed_series",), hide=False)
            self._rendered_processed_dataset = None

        self._render_stability_window_preview(dpg, dataset)
        self._render_t0_preview(dpg, dataset)
        self._render_optional_preprocessing(dpg, dataset)
        self._render_robustness(dpg)
        self._render_typed_preprocessing(dpg, pipeline)
        self._render_signed_normalization(dpg, pipeline)
        if hasattr(self,'_render_preprocess_regions'):self._render_preprocess_regions(dpg,pipeline)
        if hasattr(dpg, "does_item_exist") and dpg.does_item_exist("pyecr_data_fit_quality"):
            quality = self._fit_quality_for_display()
            dpg.set_value("pyecr_data_fit_quality", "\n".join((
                f"FIT READY {'YES' if quality.ready else 'NO'}",
                "Blocking reasons: " + ("; ".join(quality.blocking_reasons) or "none"),
                "Warnings: " + ("; ".join(quality.warnings) or "none"),
            )))

        has_dataset = pipeline is not None
        has_steps = bool(pipeline and pipeline.config.steps)
        dpg.configure_item("pyecr_data_apply_processing", enabled=has_dataset)
        dpg.configure_item("pyecr_data_undo_processing", enabled=has_steps)
        dpg.configure_item("pyecr_data_reset_processing", enabled=has_steps)
        dpg.set_value(
            "pyecr_data_processing_reason",
            "Import a dataset before applying preprocessing." if not has_dataset else
            "Undo and Reset become available after the first processing step." if not has_steps else "",
        )
        config = pipeline.config if pipeline is not None else None
        if config is not self._rendered_processing_config:
            if config is None or not config.steps:
                history_text = history_evidence(config, pipeline.revision, pipeline.source) if pipeline else "No preprocessing steps."
            else:
                history_text = history_evidence(config, pipeline.revision, pipeline.source)
            dpg.set_value("pyecr_data_processing_history", history_text)
            self._rendered_processing_config = config

    def _render_fit_presentation(self) -> None:
        presentation = self.controller.state.fit_presentation
        if presentation is None:
            if self._rendered_fit_presentation is not None:
                dpg = self._require_dpg()
                for tag in (
                    "pyecr_fit_data_series",
                    "pyecr_fit_curve_series",
                    "pyecr_fit_h_series",
                    "pyecr_fit_o_series",
                    "pyecr_residual_series",
                    "pyecr_residual_zero_series",
                    "pyecr_boundary_ideal_series",
                    "pyecr_boundary_corrected_series",
                    "pyecr_boundary_input_series",
                    "pyecr_explorer_fit_series",
                    "pyecr_explorer_variant_series",
                ):
                    clear_plot_series(dpg, (tag,))
                dpg.set_value("pyecr_explorer_status", "Requires a current Fit result.")
                dpg.set_value("pyecr_fit_reliability", "No current Fit result.")
                dpg.set_value("pyecr_fit_warnings", "No Fit warnings.")
                if dpg.does_item_exist("pyecr_fit_parameter_table"):
                    dpg.delete_item("pyecr_fit_parameter_table")
                dpg.set_value("pyecr_boundary_plot_note", "No current Fit result.")
                self._apply_decomposition_view()
                self._rendered_fit_presentation = None
            return
        if presentation is self._rendered_fit_presentation:
            return
        dpg = self._require_dpg()
        plot_data = FitPlotData.from_fit_presentation(presentation)
        times = list(plot_data.time_s)
        dataset = self.controller.analysis_session.dataset
        if dataset is not None:
            metadata = dataset.metadata
            axis_signal = signal_axis_label(metadata)
            axis_residual = residual_axis_label(metadata)
            for tag in (
                "pyecr_fit_y_axis", "pyecr_explorer_y_axis",
            ):
                dpg.configure_item(tag, label=axis_signal)
            dpg.configure_item("pyecr_residual_y_axis", label=axis_residual)
            dpg.configure_item("pyecr_components_y_axis", label=f"Contribution ({signal_display_unit(metadata)})")
        dpg.set_value("pyecr_fit_data_series", [times, list(plot_data.observed)])
        dpg.set_value("pyecr_fit_curve_series", [times, list(plot_data.fitted)])
        dpg.configure_item("pyecr_fit_data_series", show=True)
        dpg.configure_item("pyecr_fit_curve_series", show=True)
        h_values=plot_data.component_1 if plot_data.model_name=="M8_DUAL_K" else plot_data.component_H
        o_values=plot_data.component_2 if plot_data.model_name=="M8_DUAL_K" else plot_data.component_O
        h_times = times if plot_data.has_components and h_values else []
        o_times = times if plot_data.has_components and o_values else []
        dpg.set_value(
            "pyecr_fit_h_series", [h_times, list(h_values) if h_times else []]
        )
        dpg.set_value(
            "pyecr_fit_o_series", [o_times, list(o_values) if o_times else []]
        )
        dpg.configure_item("pyecr_fit_h_series", show=bool(h_times))
        dpg.configure_item("pyecr_fit_o_series", show=bool(o_times))
        labels=plot_data.component_labels if plot_data.model_name=="M8_DUAL_K" else ("H component","O component")
        if labels:
            dpg.configure_item("pyecr_fit_h_series",label=labels[0])
            dpg.configure_item("pyecr_fit_o_series",label=labels[1])
        dpg.set_value(
            "pyecr_residual_series", [times, list(plot_data.residuals)]
        )
        dpg.configure_item("pyecr_residual_series", show=True)
        dpg.set_value("pyecr_residual_zero_series", [[times[0], times[-1]], [0.0, 0.0]])
        dpg.configure_item("pyecr_residual_zero_series", show=True)
        dpg.set_value("pyecr_explorer_fit_series", [times, list(plot_data.fitted)])
        dpg.configure_item("pyecr_explorer_fit_series", show=True)
        dpg.set_value("pyecr_explorer_variant_series", [[], []])
        dpg.configure_item("pyecr_explorer_variant_series", show=False)
        fit_result = self.controller.analysis_session.fit_result
        if dpg.does_item_exist("pyecr_solver_details"):
            runs=() if fit_result is None else self.controller.analysis_service.solver_diagnostics(fit_result)
            dpg.set_value("pyecr_solver_details",solver_details(runs,fit_result))
        config = self.controller.current_analysis_config
        if config is not None and config.model_id.value == "M1_SINGLE_DK":
            for parameter in ("D_cm2_s", "k_cm_s"):
                free_tag = _model_parameter_tag(parameter, "free")
                value_tag = _model_parameter_tag(parameter, "value")
                if dpg.does_item_exist(free_tag):
                    dpg.set_value(
                        free_tag, parameter in config.model.free_parameters,
                    )
                value = config.model.fixed_parameters.get(
                    parameter, config.model.initial_values.get(parameter)
                )
                if value is not None and dpg.does_item_exist(value_tag):
                    dpg.set_value(value_tag, float(value))
            guided = None if fit_result is None else fit_result.diagnostics.get("guided_refit")
            if isinstance(guided, Mapping):
                dpg.set_value(
                    "pyecr_model_apply_status",
                    f"Guided Refit configuration active: fixed "
                    f"{guided.get('fixed_parameter')}={guided.get('fixed_value')} "
                    f"from {guided.get('value_source')}."
                )
        options = [] if fit_result is None else [
            name for name in fit_result.params if name.startswith(("D_", "k_"))
        ]
        dpg.configure_item("pyecr_explorer_parameter", items=options)
        if options:
            dpg.set_value("pyecr_explorer_parameter", options[0])
        dpg.set_value("pyecr_explorer_status", "Scale one parameter; no refit.")
        for tag in ("pyecr_boundary_ideal_series", "pyecr_boundary_corrected_series", "pyecr_boundary_input_series"):
            dpg.set_value(tag, [[], []])
            dpg.configure_item(tag, show=False)
        if fit_result is not None and config is not None and config.boundary is not None and config.boundary.mode != "ideal_step":
            if config.boundary.mode == "measured_history":
                try:
                    self.controller.run_boundary_plot(plot_data)
                    dpg.set_value(
                        "pyecr_boundary_plot_note",
                        "Loading measured boundary history in background…",
                    )
                except Exception as exc:
                    self.controller.record_callback_exception(GuiPage.FIT, exc, traceback.format_exc())
                    dpg.set_value("pyecr_boundary_plot_note", f"Boundary view unavailable: {exc}")
            else:
                try:
                    boundary = boundary_plot_data(plot_data, fit_result, config)
                except Exception as exc:
                    self.controller.record_callback_exception(GuiPage.FIT, exc, traceback.format_exc())
                    dpg.set_value("pyecr_boundary_plot_note", f"Boundary view unavailable: {exc}")
                else:
                    self._apply_boundary_plot_data(boundary)
        else:
            dpg.set_value("pyecr_boundary_plot_note", "Ideal-step boundary: no separate gas-history view.")
        status_text = (
            f"{presentation.reliability_label} | Model={presentation.model_name} | "
            f"RMSE={format_scientific(presentation.RMSE)}, R²={format_scientific(presentation.R2)}, "
            f"nfev={presentation.nfev}"
        )
        if presentation.model_name == "M7_SINGLE_K":
            fit = self.controller.analysis_session.fit_result
            info = fit.diagnostics.get("surface_k", {}) if fit is not None else {}
            status_text += (f" | kinetic free=1; total free={info.get('free_parameter_count')}"
                f" | fixed={dict(info.get('fixed_parameters', {}))}"
                f" | derived g={info.get('g_cm_inv')} cm^-1; tau_surface_exchange={info.get('tau_surface_exchange_s')} s"
                " | Bi N/A (D_not_estimated); CI conditional on fixed inputs")
        if presentation.model_name=="M8_DUAL_K":
            info=self.controller.analysis_session.fit_result.diagnostics["dual_surface_k"]
            status_text+=(f" | kinetic free=2; total free={info['free_parameter_count']}"
                f" | labels={list(info['component_labels'])}; fixed={dict(info['fixed_parameters'])}"
                f" | tau_surface_exchange={[info['tau_1_surface_exchange_s'],info['tau_2_surface_exchange_s']]} s"
                " | Bi N/A; conditional CI; contributions plus y0 once")
        conditional=self.controller.analysis_session.fit_result.diagnostics.get("fixed_d")
        if conditional:
            status_text+=(f" | fixed D={ {n:s['value_cm2_s'] for n,s in conditional['sources'].items()} } cm²/s"
                          f" | total free={conditional['free_parameter_count']}; CI conditional on D; source uncertainty not propagated")
        from pyecr.model_display import display_model_text
        dpg.set_value("pyecr_fit_reliability", display_model_text(status_text))
        dpg.configure_item(
            "pyecr_fit_reliability",
            color=TOKENS.colors.success if presentation.reliable else TOKENS.colors.warning,
        )
        dpg.set_value(
            "pyecr_fit_warnings",
            "No Fit warnings."
            if not presentation.warnings
            else "\n".join(presentation.warnings),
        )
        if dpg.does_item_exist("pyecr_fit_parameter_table"):
            dpg.delete_item("pyecr_fit_parameter_table")
        with dpg.group(parent="pyecr_fit_parameter_container", tag="pyecr_fit_parameter_table"):
            for row in presentation.parameters:
                with dpg.collapsing_header(
                    label=f"{row.name}  {format_scientific(row.estimate)} {row.unit}",
                    default_open=False,
                ):
                    dpg.add_text(f"Gate local 95% CI: {_format_confidence_interval(row.ci_lower, row.ci_upper)}", wrap=TOKENS.wraps.rail)
                    dpg.add_text('Selected output CI: not computed for this parent.', tag='pyecr_parameter_output_ci_'+row.name, wrap=TOKENS.wraps.rail)
                    dpg.add_text(
                        "Numerical contact: unknown"
                        if row.bound_hit is None
                        else f"Numerical contact: {row.bound_side if row.bound_hit else 'no'}",
                        tag=f"pyecr_parameter_bound_{row.name}",
                    )
                    dpg.add_text(
                        "Local parameter checks: passed; see Diagnostics gates"
                        if row.reliability_flag
                        else f"Local parameter checks: warning — {row.reliability_reason}",
                        tag=f"pyecr_parameter_reliable_{row.name}",
                        wrap=TOKENS.wraps.rail,
                    )
        dpg.fit_axis_data("pyecr_fit_x_axis")
        dpg.fit_axis_data("pyecr_fit_y_axis")
        if plot_data.has_components:
            dpg.fit_axis_data("pyecr_components_x_axis")
            dpg.fit_axis_data("pyecr_components_y_axis")
        dpg.fit_axis_data("pyecr_residual_x_axis")
        dpg.fit_axis_data("pyecr_residual_y_axis")
        self._apply_residual_scale()
        self._apply_decomposition_view()
        self._rendered_fit_presentation = presentation

    def _change_decomposition_view(self, sender=None, app_data=None, user_data=None):
        from pyecr.gui.components.decomposition import options_from_widgets, apply_scene
        try:
            self.controller.set_decomposition_view_options(options_from_widgets(self))
            self._apply_decomposition_view()
        except Exception as exc:
            self._visual_decomposition_scene=None
            apply_scene(self, None)
            self._require_dpg().set_value("pyecr_decomposition_formula", f"Decomposition unavailable: {exc}")

    def _apply_decomposition_view(self):
        from pyecr.gui.components.decomposition import apply_scene
        try:
            scene = self.controller.current_decomposition()
            self._visual_decomposition_scene=scene
            apply_scene(self, scene)
        except Exception as exc:
            self._visual_decomposition_scene=None
            apply_scene(self, None)
            self._require_dpg().set_value("pyecr_decomposition_formula", f"Decomposition unavailable: {exc}")

    def _change_sparse_marker_display(self,sender=None,app_data=None,user_data=None):
        self._display_sparse_markers=bool(self._require_dpg().get_value("pyecr_display_sparse_markers"))
        self._refresh_fit_visuals()
        from pyecr.gui.components.decomposition import apply_scene
        apply_scene(self,getattr(self,"_visual_decomposition_scene",None))

    def _refresh_fit_visuals(self):
        """Markers and residual mirror from captured rows; no fit or prediction."""
        from pyecr.plotting.visual_style import marker_indices
        dpg=self._require_dpg();p=self.controller.state.fit_presentation
        if p is None:
            dpg.set_value("pyecr_fit_residual_mirror_series",[[],[]])
            dpg.set_value("pyecr_fit_residual_mirror_zero",[[],[]])
            self._fit_visual_key=None;self._residual_mirror_tick_key=None
            return
        key=(id(p),getattr(self,"_display_sparse_markers",False))
        if key!=getattr(self,"_fit_visual_key",None):
            indices=marker_indices(p.time_s,p.data) if key[1] else range(len(p.time_s))
            dpg.set_value("pyecr_fit_data_series",[[p.time_s[i] for i in indices],[p.data[i] for i in indices]])
            dpg.set_value("pyecr_marker_display_status",f"Markers {len(indices)}/{len(p.time_s)}; residuals use all observations")
            dpg.set_value("pyecr_fit_residual_mirror_series",[list(p.time_s),list(p.residuals)])
            dpg.set_value("pyecr_fit_residual_mirror_zero",[[p.time_s[0],p.time_s[-1]],[0.,0.]])
            unit=signal_display_unit(self.controller.analysis_session.dataset.metadata)
            dpg.configure_item("pyecr_fit_residual_mirror_y",label=f"Residual ({unit})")
            # Hidden tabs cannot reliably auto-fit an ImPlot axis. Derive the
            # displayed limits from every captured residual, retaining zero.
            low=min(0.,min(p.residuals));high=max(0.,max(p.residuals))
            margin=max((high-low)*.08,1.e-12)
            dpg.set_axis_limits("pyecr_fit_residual_mirror_y",low-margin,high+margin)
            self._fit_visual_key=key
        # Two endpoint ticks stay legible in the compact strip; the zero line
        # supplies the reference. Taller windows can also label zero.
        height=dpg.get_item_configuration("pyecr_fit_residual_mirror")["height"]
        tick_key=(id(p),height)
        if tick_key!=getattr(self,"_residual_mirror_tick_key",None):
            low=min(0.,min(p.residuals));high=max(0.,max(p.residuals))
            ticks=[(f"{low:.1e}",low),(f"{high:.1e}",high)]
            if height>=160 and low<0<high:ticks.insert(1,("0",0.))
            if low==high:ticks=[("0",0.)]
            dpg.set_axis_ticks("pyecr_fit_residual_mirror_y",tuple((label,float(value)) for label,value in ticks))
            self._residual_mirror_tick_key=tick_key
        limits=dpg.get_axis_limits("pyecr_fit_x_axis")
        if limits[0]<limits[1]:dpg.set_axis_limits("pyecr_fit_residual_mirror_x",*limits)

    def _apply_boundary_plot_data(self, boundary: Any) -> None:
        dpg = self._require_dpg()
        for tag, x, y in (
            ("pyecr_boundary_ideal_series", boundary.time_s, boundary.ideal_response),
            ("pyecr_boundary_corrected_series", boundary.time_s, boundary.corrected_response),
            ("pyecr_boundary_input_series", boundary.input_time_s, boundary.input_progress),
        ):
            dpg.set_value(tag, [list(x), list(y)])
            dpg.configure_item(tag, show=bool(x))
        dpg.set_value("pyecr_boundary_plot_note", boundary.input_note)
        dpg.fit_axis_data("pyecr_boundary_x_axis")
        dpg.fit_axis_data("pyecr_boundary_y_axis")

    def _set_residual_scale(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data
        mode = str(user_data)
        if mode not in {"auto", "robust"}:
            raise ValueError(f"unknown residual display mode: {mode}")
        self._residual_scale_mode = mode
        self._apply_residual_scale()

    def _apply_residual_scale(self) -> None:
        dpg = self._require_dpg()
        presentation = self.controller.state.fit_presentation
        if presentation is None:
            dpg.set_value("pyecr_residual_scale_status", "No current residuals")
            return
        mode = getattr(self, "_residual_scale_mode", "auto")
        if mode == "robust":
            limits = robust_residual_limits(presentation.residuals)
            if limits is not None:
                dpg.set_axis_limits("pyecr_residual_y_axis", *limits)
                dpg.set_value("pyecr_residual_scale_status", "Robust (symmetric 95% |r|)")
                return
        dpg.set_axis_limits_auto("pyecr_residual_y_axis")
        dpg.fit_axis_data("pyecr_residual_y_axis")
        dpg.set_value("pyecr_residual_scale_status", "Auto (all points + zero)")

    def _preview_parameter_variant(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._schedule_parameter_variant(None, None, "immediate")
        self._dispatch_debounced_explorer()
        self._refresh_widgets()

    def _schedule_parameter_variant(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data
        dpg = self._require_dpg()
        try:
            session = self.controller.analysis_session
            presentation = self.controller.state.fit_presentation
            config = self.controller.current_analysis_config
            if (
                session.fit_result is None or presentation is None or config is None
                or session.fit_stale
            ):
                raise RuntimeError("Parameter explorer requires a current Fit result")
            parameter = str(dpg.get_value("pyecr_explorer_parameter"))
            multiplier = float(dpg.get_value("pyecr_explorer_multiplier"))
            if not parameter:
                raise ValueError("choose a D or k parameter")
            if multiplier <= 0.0:
                raise ValueError("multiplier must be positive")
            self.controller.cancel_explorer_preview()
            delay = 0.0 if user_data == "immediate" else _EXPLORER_DEBOUNCE_SECONDS
            self._explorer_debounce_deadline = monotonic() + delay
            dpg.set_value(
                "pyecr_explorer_status",
                f"Waiting to preview {parameter} × {format_scientific(multiplier)}…",
            )
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.FIT, exc, traceback.format_exc())
            dpg.set_value("pyecr_explorer_status", f"Preview failed: {exc}")

    def _dispatch_debounced_explorer(self) -> None:
        deadline = self._explorer_debounce_deadline
        if deadline is None or monotonic() < deadline:
            return
        self._explorer_debounce_deadline = None
        dpg = self._require_dpg()
        try:
            presentation = self.controller.state.fit_presentation
            if presentation is None:
                raise RuntimeError("Parameter explorer requires a current Fit result")
            plot = FitPlotData.from_fit_presentation(presentation)
            parameter = str(dpg.get_value("pyecr_explorer_parameter"))
            multiplier = float(dpg.get_value("pyecr_explorer_multiplier"))
            self.controller.run_explorer_preview(plot, parameter, multiplier)
            dpg.set_value("pyecr_explorer_status", "Running production forward preview…")
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.FIT, exc, traceback.format_exc())
            dpg.set_value("pyecr_explorer_status", f"Preview failed: {exc}")

    def _cancel_parameter_variant(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._explorer_debounce_deadline = None
        requested = self.controller.cancel_explorer_preview()
        self._require_dpg().set_value(
            "pyecr_explorer_status",
            "Cancellation requested…" if requested else "No preview is running.",
        )

    def _render_async_plot_presentations(self) -> None:
        dpg = self._require_dpg()
        boundary = self.controller.boundary_plot_result
        if boundary is not None and boundary is not self._rendered_boundary_computation:
            self._apply_boundary_plot_data(boundary.result)
            self._rendered_boundary_computation = boundary
        explorer = self.controller.explorer_plot_result
        if explorer is not None and explorer is not self._rendered_explorer_computation:
            dpg.set_value(
                "pyecr_explorer_variant_series",
                [list(explorer.context.plot.time_s), list(explorer.values)],
            )
            dpg.configure_item("pyecr_explorer_variant_series", show=True)
            dpg.set_value(
                "pyecr_explorer_status",
                f"{explorer.parameter} × {format_scientific(explorer.multiplier)} (forward only)",
            )
            dpg.fit_axis_data("pyecr_explorer_x_axis")
            dpg.fit_axis_data("pyecr_explorer_y_axis")
            self._rendered_explorer_computation = explorer
        can_preview = bool(
            self.controller.state.fit_presentation is not None
            and not self.controller.analysis_session.fit_stale
        )
        set_action_availability(
            dpg, "pyecr_explorer_preview", enabled=can_preview,
            reason="Run a current Fit before generating a parameter preview.",
        )
        preview_pending = (
            self._explorer_debounce_deadline is not None
            or self.controller.explorer_preview_running
        )
        set_action_availability(
            dpg, "pyecr_explorer_cancel", enabled=preview_pending,
            reason="No Explorer preview is queued or running.",
        )

    def _confirm_project_open_units(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        path = self._pending_unit_project_path
        self._pending_unit_project_path = None
        self._require_dpg().hide_item("pyecr_project_unit_confirmation")
        try:
            if path is None:
                raise RuntimeError("no legacy project is pending confirmation")
            self.controller.open_project(path, confirm_unknown_units=True)
            self._reset_data_render_state()
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.PROJECT, exc, traceback.format_exc())
        self._refresh_widgets()

    def _cancel_project_open_units(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._pending_unit_project_path = None
        self._require_dpg().hide_item("pyecr_project_unit_confirmation")

    def _select_signal_type(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        signal_type = str(dpg.get_value("pyecr_data_signal_type"))
        options = get_units(signal_type)
        current = str(dpg.get_value("pyecr_data_signal_unit") or "")
        try:
            current_key = validate_pair(signal_type, current)
        except ValueError:
            current_key = None
        selected = (
            unit_label(current) if current_key not in {None, "custom"}
            else ("Custom…" if current_key == "custom" else get_default_unit(signal_type).label)
        )
        dpg.configure_item("pyecr_data_signal_unit", items=[item.label for item in options],
                           enabled=signal_type != "normalized")
        dpg.set_value("pyecr_data_signal_unit", selected)
        dpg.configure_item("pyecr_data_custom_unit", show=selected == "Custom…")

    def _select_signal_unit(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        dpg.configure_item("pyecr_data_custom_unit",
                           show=dpg.get_value("pyecr_data_signal_unit") == "Custom…")

    def _run_error_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._require_dpg().set_value("pyecr_analysis_plot_tabs", "pyecr_analysis_error_map_tab")
        self._require_dpg().set_y_scroll("pyecr_analysis_center", 0)
        self._refresh_widgets()

    def _run_dual_error_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._require_dpg().set_value("pyecr_analysis_plot_tabs", "pyecr_analysis_error_map_tab")
        self._require_dpg().set_y_scroll("pyecr_analysis_center", 0)
        self._refresh_widgets()

    def _render_sensitivity_summary(self, presentation: Any) -> None:
        dpg = self._require_dpg()
        section = getattr(presentation, "sensitivity", None)
        if section is None or not section.available:
            reason = "not captured in this Diagnostics result" if section is None else section.reason
            dpg.set_value("pyecr_sensitivity_status", f"Sensitivity: N/A — {reason}")
            if dpg.does_item_exist("pyecr_sensitivity_table"):
                dpg.delete_item("pyecr_sensitivity_table")
            _clear_table_copy(dpg, "pyecr_sensitivity_table", reason)
            self._update_sensitivity_plot()
            return
        dpg.set_value(
            "pyecr_sensitivity_status",
            f"Sensitivity: {section.method}; central difference step="
            f"{format_scientific(section.step_decades)} decades. "
            "The normalized view rescales display values only.",
        )
        _replace_diagnostics_table(
            dpg,
            table_tag="pyecr_sensitivity_table",
            parent="pyecr_sensitivity_table_container",
            headings=(
                "Curve", "Peak |S|", "Peak time (s)",
                "Early info", "Mid info", "Late info", "Step convergence",
            ),
            rows=tuple(
                (
                    str(curve.label),
                    format_scientific(curve.peak_absolute),
                    format_scientific(curve.peak_time_s),
                    _information_fraction(curve.early_information_fraction),
                    _information_fraction(curve.mid_information_fraction),
                    _information_fraction(curve.late_information_fraction),
                    (
                        "N/A" if curve.convergence_relative_error is None else
                        format_scientific(curve.convergence_relative_error)
                    ),
                )
                for curve in section.curves
            ),
        )
        self._update_sensitivity_plot()

    def _render_reliability_advisor(self, presentation: Any) -> None:
        dpg = self._require_dpg()
        advisor = getattr(presentation, "reliability_advisor", None)
        if advisor is None:
            dpg.set_value("pyecr_reliability_identifiability", "Identifiability: N/A — no advisor result.")
            dpg.set_value("pyecr_reliability_adequacy", "Model adequacy: N/A — no advisor result.")
            dpg.set_value("pyecr_reliability_bi_guidance", "Bi guidance: N/A — no advisor result.")
            dpg.set_value("pyecr_reliability_evidence", "No Reliability Advisor evidence.")
            return
        dpg.set_value(
            "pyecr_reliability_identifiability",
            f"Parameter identifiability — {advisor.identifiability_status}: "
            f"{advisor.identifiability_explanation}",
        )
        dpg.set_value(
            "pyecr_reliability_adequacy",
            f"Model adequacy — {advisor.model_adequacy_status}: "
            f"{advisor.model_adequacy_explanation}",
        )
        dpg.set_value(
            "pyecr_reliability_bi_guidance",
            "Bi guidance (heuristic): " + _advisor_display(advisor.bi_guidance),
        )
        dpg.set_value(
            "pyecr_reliability_evidence",
            _advisor_display(advisor.evidence),
        )

    def _refresh_guided_refit_widgets(self) -> None:
        dpg = self._require_dpg()
        eligibility = self.controller.workflow.assess_guided_refit()
        fit_running = self.controller.state.tasks[GuiPage.FIT].status == "running"
        diagnostics_running = (
            self.controller.state.tasks[GuiPage.DIAGNOSTICS].status == "running"
        )
        source = str(dpg.get_value("pyecr_guided_refit_source"))
        source_selected = source in {"Parent fitted value", "Manual value"}
        dpg.configure_item(
            "pyecr_guided_refit_manual_value", show=source == "Manual value",
        )
        explanation = (
            f"Ready: {eligibility.reliable_parameter} is better localized; "
            f"refit {eligibility.weak_parameter}. {eligibility.reason}"
            if eligibility.available else
            f"Unavailable: {eligibility.reason}"
        )
        if eligibility.evidence:
            explanation += "\n" + "\n".join(eligibility.evidence)
        dpg.set_value("pyecr_guided_refit_eligibility", explanation)
        for tag, parameter in (
            ("pyecr_guided_refit_fix_d", "D_cm2_s"),
            ("pyecr_guided_refit_fix_k", "k_cm_s"),
        ):
            set_action_availability(
                dpg, tag,
                enabled=(eligibility.available
                         and eligibility.reliable_parameter == parameter
                         and source_selected and not fit_running
                         and not diagnostics_running),
                reason="Choose a fixed value source and review current 03 M2_SINGLE_DK Diagnostics.",
            )
        dpg.configure_item(
            "pyecr_guided_refit_cancel",
            enabled=self.controller.guided_refit_running,
        )
        session = self.controller.analysis_session
        fit = session.fit_result if not session.fit_stale else None
        guided = None if fit is None else fit.diagnostics.get("guided_refit")
        if not isinstance(guided, Mapping):
            guided = None
        dpg.configure_item(
            "pyecr_guided_refit_rollback",
            enabled=bool(guided and session.fit_history and not fit_running
                         and not diagnostics_running),
        )
        if self.controller.guided_refit_running:
            status = "Guided Refit running. Cancel leaves the parent Fit current."
        elif guided is not None:
            status = (
                f"Guided Refit active: fixed {guided.get('fixed_parameter')} = "
                f"{guided.get('fixed_value')} ({guided.get('value_source')}); "
                f"parent FitResult ID {guided.get('parent_fit_id')}."
            )
        else:
            status = self.controller.guided_refit_last_message
        dpg.set_value("pyecr_guided_refit_status", status)
        comparison = None if guided is None else guided.get("comparison")
        if isinstance(comparison, Mapping):
            before = comparison.get("before")
            after = comparison.get("after")
            if isinstance(before, Mapping) and isinstance(after, Mapping):
                lines = []
                for label, key in (
                    ("SSE", "SSE"), ("RMSE", "RMSE"),
                    ("Residual mean", "residual_mean"),
                    ("Residual SD", "residual_standard_deviation"),
                    ("Max |residual|", "residual_max_abs"),
                ):
                    lines.append(
                        f"{label}: {before.get(key, 'N/A')} → {after.get(key, 'N/A')}"
                    )
                for label, key in (
                    ("CI", "confidence_intervals"),
                    ("Identifiability", "identifiability"),
                ):
                    lines.append(
                        f"{label} before: {before.get(key, {})}\n"
                        f"{label} after: {after.get(key, {})}"
                    )
                comparison_text = "\n".join(lines)
            else:
                comparison_text = "Guided Refit comparison is incomplete."
        else:
            comparison_text = "No Guided Refit comparison."
        dpg.set_value("pyecr_guided_refit_comparison", comparison_text)

    def _render_diagnostics_presentation(self) -> None:
        from pyecr.model_display import display_model_text
        presentation = self.controller.state.diagnostics_presentation
        dpg = self._require_dpg()
        if presentation is None and self._rendered_diagnostics_presentation is not None:
            dpg.set_value("pyecr_diagnostics_residual_series", [[], []])
            dpg.set_value("pyecr_diagnostics_reliability", "No current Diagnostics result.")
            dpg.set_value("pyecr_diagnostics_warnings", "No Diagnostics warnings.")
            dpg.set_value("pyecr_sensitivity_status", "No current Diagnostics result.")
            dpg.set_value("pyecr_reliability_identifiability", "Identifiability: N/A — diagnostics have not run.")
            dpg.set_value("pyecr_reliability_adequacy", "Model adequacy: N/A — diagnostics have not run.")
            dpg.set_value("pyecr_reliability_bi_guidance", "Bi guidance: N/A — diagnostics have not run.")
            dpg.set_value("pyecr_reliability_evidence", "No Reliability Advisor evidence.")
            for table_tag in (
                "pyecr_diagnostics_bi_table",
                "pyecr_diagnostics_parameter_table",
                "pyecr_diagnostics_correlation_table",
                "pyecr_diagnostics_multistart_table",
                "pyecr_sensitivity_table",
            ):
                if dpg.does_item_exist(table_tag):
                    dpg.delete_item(table_tag)
                _clear_table_copy(dpg, table_tag, "No current Diagnostics result.")
            self._update_sensitivity_plot()
            self._rendered_diagnostics_presentation = None
        if presentation is not None and presentation is not self._rendered_diagnostics_presentation:
            times = list(presentation.time_s)
            dpg.set_value(
                "pyecr_diagnostics_residual_series",
                [times, list(presentation.residuals)],
            )
            dpg.set_value(
                "pyecr_diagnostics_reliability",
                f"{presentation.reliability_label.removeprefix('Diagnostics v2: ')} | optimizer success="
                f"{presentation.optimizer_success} | statistical candidate="
                f"{display_model_text(presentation.best_statistical)} | all-gates candidate="
                f"{display_model_text(presentation.best_reliable or 'none')}",
            )
            dpg.configure_item(
                "pyecr_diagnostics_reliability",
                color=TOKENS.colors.success if presentation.reliable else TOKENS.colors.warning,
            )
            warning_lines = (*presentation.gate_failures, *presentation.warnings)
            dpg.set_value(
                "pyecr_diagnostics_warnings",
                "No Diagnostics warnings."
                if not warning_lines
                else "\n".join(dict.fromkeys(warning_lines)),
            )
            residual = presentation.residual_summary
            evaluation = presentation.evaluation
            lag1 = (
                format_scientific(residual.lag1_autocorrelation)
                if residual.lag1_autocorrelation is not None
                else f"N/A — {residual.lag1_reason}"
            )
            dpg.set_value(
                "pyecr_diagnostics_residual_summary",
                f"Residual summary: n={residual.count}, mean={format_scientific(residual.mean)}, "
                f"SD={format_scientific(residual.standard_deviation)}, min={format_scientific(residual.minimum)}, "
                f"max={format_scientific(residual.maximum)}, lag-1={lag1}"
                + (f"\n{evaluation.residual.explanation}\nSpacing: {evaluation.residual.spacing}; "
                   f"ACF lags 1–{len(evaluation.residual.acf)}: "
                   + ', '.join(f'{v:.3g}' for v in evaluation.residual.acf)
                   + f"; reference band ±{format_scientific(evaluation.residual.reference_band) if evaluation.residual.reference_band is not None else 'N/A'}; "
                   f"simultaneous band ±{format_scientific(evaluation.residual.simultaneous_band) if evaluation.residual.simultaneous_band is not None else 'N/A'}"
                   if evaluation is not None else ""),
            )
            metrics = presentation.information_criteria
            dpg.set_value(
                "pyecr_diagnostics_information_criteria",
                f"Fit statistics: SSE={format_scientific(metrics.SSE)}, RMSE={format_scientific(metrics.RMSE)}, "
                f"R²={format_scientific(metrics.R2)}, AIC={format_scientific(metrics.AIC)}, "
                f"AICc={format_scientific(metrics.AICc)}, BIC={format_scientific(metrics.BIC)}",
            )
            jacobian = presentation.jacobian
            dpg.set_value(
                "pyecr_diagnostics_jacobian",
                (
                    f"Jacobian: rank={jacobian.rank}/{jacobian.parameter_count}, "
                    f"cond(JᵀJ)="
                    f"{format_scientific(jacobian.condition_number) if jacobian.condition_number is not None else '∞'}, "
                    f"source={jacobian.source}"
                    if jacobian.available
                    else f"Jacobian: N/A — {jacobian.reason}"
                ),
            )
            _render_bi_table(dpg, presentation)
            if evaluation is not None:
                dpg.set_value("pyecr_diagnostics_jacobian",
                    f"Jacobian scaled rank={evaluation.rank}/{evaluation.parameter_count}; "
                    f"cond(ZᵀZ)={evaluation.scaled_condition}; raw cond(JᵀJ)={evaluation.raw_condition}. "
                    "Z has unit column norms; precision and absolute sensitivity are evaluated separately.")
            _render_parameter_diagnostics_table(dpg, presentation)
            _render_correlation_table(dpg, presentation)
            _render_multistart_table(dpg, presentation)
            self._render_sensitivity_summary(presentation)
            self._render_reliability_advisor(presentation)
            dpg.fit_axis_data("pyecr_diagnostics_residual_x_axis")
            dpg.fit_axis_data("pyecr_diagnostics_residual_y_axis")
            self._rendered_diagnostics_presentation = presentation

        for method in ("bootstrap", "profile", "experimental_m1"):
            advanced = self.controller.state.advanced_diagnostics.get(method)
            if advanced is None and method in self._rendered_advanced_diagnostics:
                dpg.set_value(
                    f"pyecr_diagnostics_{method}_status",
                    f"{method.title()}: not run for the current fit (opt-in).",
                )
                self._rendered_advanced_diagnostics.pop(method, None)
                if method == "experimental_m1":
                    dpg.set_value("pyecr_diagnostics_experimental_m1_evidence",
                                  "No experimental result for the current fit.")
                continue
            if advanced is self._rendered_advanced_diagnostics.get(method):
                continue
            tag = f"pyecr_diagnostics_{method}_status"
            if advanced is None:
                continue
            if method == "experimental_m1" and advanced.experimental_result is not None:
                record = advanced.experimental_result
                intervals = ", ".join(
                    f"{name}=[{format_scientific(values['lower'])}, {format_scientific(values['upper'])}]"
                    for name, values in record.intervals.items()
                ) or "no complete interval"
                warning = "" if not record.warnings else "; " + "; ".join(record.warnings)
                dpg.set_value(tag, f"Deep uncertainty: {record.status}; "
                              f"actual C={100*record.confidence_level:.12g}% / {record.method}; "
                              f"{record.n_successful}/{record.n_requested} successful; "
                              f"elapsed {record.elapsed_seconds:.1f} s; {intervals}; "
                              f"conditional if Guided Refit; see JSON for evidence{warning}")
                dpg.set_value("pyecr_diagnostics_experimental_m1_evidence",
                              json.dumps(record.evidence, ensure_ascii=False, indent=2,
                                         allow_nan=False))
            elif not advanced.available:
                actual=(f'actual C={100*advanced.result.confidence_level:.12g}% / {advanced.result.method.value}' if advanced.result is not None else
                        f'requested C={100*advanced.binding.controls.output_ci.confidence_level:.12g}%' if advanced.binding is not None else 'legacy method record')
                dpg.set_value(tag, f"{method.title()} ({actual}): N/A — {advanced.reason}")
            else:
                result = advanced.result
                assert result is not None
                intervals = ", ".join(
                    f"{name}=[{format_scientific(interval.lower)}, {format_scientific(interval.upper)}]"
                    for name, interval in result.intervals.items()
                ) or "intervals unavailable"
                warning = "" if not advanced.warnings else " | " + "; ".join(advanced.warnings)
                dpg.set_value(
                    tag,
                    f"{method.title()} / {result.method.value} / actual C={100*result.confidence_level:.12g}%: {result.n_successful}/{result.n_requested} "
                    f"successful | {intervals}{warning}",
                )
            self._rendered_advanced_diagnostics[method] = advanced

        self._render_error_map_summary()
        self._render_dual_error_map_summary()

    def _render_error_map_summary(self) -> None:
        dpg = self._require_dpg()
        result = self.controller.current_error_map_result
        current_config = self.controller.current_analysis_config
        if result is None:
            reason = (
                "N/A — a current M1 fit is required."
                if current_config is None else
                "previous map is Out of date; refit before running again."
                if self.controller.analysis_session.fit_stale else
                "N/A — only M1 D+k is supported."
                if not self.controller.workflow.can_error_map else
                "not run (opt-in)."
            )
            dpg.set_value("pyecr_diagnostics_error_map_status", f"Error Map: {reason}")
            dpg.set_value("pyecr_diagnostics_error_map_warnings", "No current Error Map warnings.")
            return
        diagnostics = result.diagnostics
        minimum = result.grid_minimum
        minimum_text = (
            "N/A — no valid grid cell"
            if minimum is None else
            f"logD={format_scientific(minimum.log10_D)}, "
            f"logk={format_scientific(minimum.log10_k)}, "
            f"SSE={format_scientific(minimum.objective)}"
        )
        dpg.set_value(
            "pyecr_diagnostics_error_map_status",
            f"Error Map: {diagnostics.n_valid}/{diagnostics.n_total} valid; "
            f"fit-point SSE={format_scientific(result.fitted_point.objective)}; "
            f"grid minimum: {minimum_text}; "
            f"quality={result.quality.status if result.quality is not None else 'N/A (legacy)'}",
        )
        warning_lines = list(diagnostics.warnings)
        if diagnostics.invalid_cells:
            warning_lines.append(
                "First invalid cell: " + diagnostics.invalid_cells[0].reason
            )
        dpg.set_value(
            "pyecr_diagnostics_error_map_warnings",
            "No Error Map warnings." if not warning_lines else "\n".join(warning_lines),
        )

    def _render_dual_error_map_summary(self) -> None:
        dpg = self._require_dpg()
        is_m3 = self.controller.workflow.is_dual_error_map_model
        dpg.configure_item("pyecr_diagnostics_dual_map_status", show=is_m3)
        dpg.configure_item("pyecr_diagnostics_dual_map_warnings", show=is_m3)
        if not is_m3:
            return
        result = self.controller.current_dual_error_map_result
        status = self.controller.dual_error_map_status
        progress = self.controller.dual_error_map_progress
        progress_text = (
            "" if progress is None else
            f" | H {progress.h_fraction:.0%}, O {progress.o_fraction:.0%}, "
            f"combined {progress.combined_fraction:.0%}"
        )
        if result is None:
            reason = (
                "Out of date — fit/data/model changed."
                if status == "out_of_date" else
                "Final selected result is D-only; M5_DUAL_H2O_DK maps are unavailable."
                if not self.controller.workflow.can_dual_error_map else
                "not run (opt-in)."
                if status == "idle" else
                f"{status}{progress_text}"
            )
            dpg.set_value("pyecr_diagnostics_dual_map_status", f"M5_DUAL_H2O_DK Error Maps: {reason}")
            dpg.set_value("pyecr_diagnostics_dual_map_warnings", "No current H/O Error Map warnings.")
            return
        updating = "Updating — previous valid result retained. " if status == "updating" else ""
        h_valid = int(result.h_map.valid_mask.sum())
        o_valid = int(result.o_map.valid_mask.sum())
        dpg.set_value(
            "pyecr_diagnostics_dual_map_status",
            f"M5_DUAL_H2O_DK Error Maps: {updating}H {h_valid}/{result.h_map.valid_mask.size} "
            f"and O {o_valid}/{result.o_map.valid_mask.size} valid; "
            f"full-signal fitted SSE={format_scientific(result.h_map.fitted_point.raw_sse)}"
            f"{progress_text}",
        )
        dpg.set_value(
            "pyecr_diagnostics_dual_map_warnings",
            "No H/O Error Map warnings." if not result.warnings else "\n".join(result.warnings),
        )
