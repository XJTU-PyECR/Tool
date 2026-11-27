"""Analysis workspace: stage-driven setup, shared plots, and results."""

from __future__ import annotations

from pyecr.gui.components.tags import WidgetTags
from pyecr.gui.components.action_state import (
    add_action_tooltip, set_action_availability,
)
from pyecr.gui.layout import (
    analysis_columns, analysis_workspace_columns, analysis_plot_height, dual_error_map_geometry,
    error_map_geometry, residual_panel_height,
)
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.gui.components.step_panel import step_panel, step_status, details_text, file_details
from pyecr.services.workflow import WorkflowState
from pyecr.gui.workflow_display import (
    WORKFLOW_STEPS, step_labels, legacy_stage_visibility, next_step_text,
    step_available, step_number, boundary_support_message,
)


def stage_visibility(
    state: WorkflowState, *, dual_model: bool = False
) -> dict[str, bool]:
    """Presentation-only progression; controls already created remain reusable."""

    return legacy_stage_visibility(state, dual_model)


def plot_tab_visibility(
    state: WorkflowState, *, dual_model: bool, nonideal_boundary: bool,
    components_available: bool = True, preview_available: bool = False,
    diagnostics_available: bool = False,
) -> dict[str, bool]:
    """Visibility follows available, non-stale scientific arrays."""

    imported = state is not WorkflowState.EMPTY
    fitted = state in {WorkflowState.FITTED, WorkflowState.DIAGNOSED}
    return {
        "raw": imported or preview_available,
        "processed": imported,
        "fit": fitted,
        "residual": fitted,
        "components": fitted and dual_model and components_available,
        "boundary": fitted and nonideal_boundary and not dual_model,
        "explorer": fitted,
        "sensitivity": fitted and diagnostics_available,
    }


class AnalysisWorkspaceMixin:
    def _build_analysis_workspace(self) -> None:
        dpg = self._require_dpg()
        self._build_parameter_window()
        self._build_analysis_navigation()
        dpg.add_text(
            self.controller.state.status_summary(GuiPage.DATA),
            tag=WidgetTags.summary("analysis"),
            wrap=TOKENS.wraps.analysis,
        )
        with dpg.group(horizontal=True):
            dpg.add_text("Analysis")
            dpg.add_button(
                label="Show Results",
                tag="pyecr_analysis_toggle_results",
                callback=self._toggle_analysis_results,
            )
            self._register_owned_root('pyecr_analysis_width_controls')
            with dpg.group(horizontal=True,tag='pyecr_analysis_width_controls'):
                dpg.add_text('Workflow width')
                dpg.add_slider_int(tag='pyecr_workflow_width',default_value=TOKENS.analysis.left_compact,
                                   min_value=TOKENS.analysis.left_min,max_value=TOKENS.analysis.left_max,
                                   width=TOKENS.analysis.slider_width,callback=self._change_workflow_width)
                dpg.add_button(label='Auto width',callback=self._reset_workflow_width)
        dpg.add_group(horizontal=True, tag="pyecr_workflow_actions")
        self._build_page_status('analysis')
        with dpg.group(horizontal=True, tag="pyecr_analysis_columns"):
            with dpg.child_window(tag="pyecr_analysis_left", width=TOKENS.viewport.left_initial, height=-1):
                dpg.bind_item_theme('pyecr_analysis_left', 'pyecr_step_rail_theme')
                dpg.add_text("WORKFLOW SETUP")
                dpg.add_text("Next: import data.", tag="pyecr_workflow_next_step", wrap=TOKENS.wraps.stage, show=False)
                for step, label in zip(WORKFLOW_STEPS, step_labels()):
                    name = step.key
                    with step_panel(dpg, name, label):
                        pass
                with dpg.group(parent="pyecr_stage_data_content"):
                    self._build_data_import_area()
                with dpg.group(parent="pyecr_stage_model_content"):
                    self._build_model_config_area()
                    with dpg.collapsing_header(label=f"Boundary conditions · step {step_number('boundary')}", tag="pyecr_stage_boundary", default_open=True):
                        dpg.add_group(tag="pyecr_stage_boundary_content")
                        dpg.add_text("",tag="pyecr_stage_boundary_reason",wrap=TOKENS.wraps.stage)
                    dpg.add_text("", tag="pyecr_boundary_stage_note", wrap=TOKENS.wraps.stage)
                with dpg.group(parent="pyecr_stage_export_content"):
                    dpg.add_button(label="Export settings / captured results", callback=self._navigate_workflow, user_data="export")
                with dpg.group(parent="pyecr_stage_fit_content", horizontal=True):
                    dpg.add_button(
                        label="Run Fit",
                        tag=WidgetTags.run("fit"),
                        callback=self._run_page,
                        user_data=GuiPage.FIT.value,
                        enabled=False,
                    )
                    add_action_tooltip(dpg, WidgetTags.run("fit"))
                    dpg.add_button(
                        label="Cancel Fit",
                        tag="pyecr_cancel_fit",
                        callback=self._cancel_fit,
                        enabled=False,
                    )
                dpg.add_text("", tag="pyecr_fit_disabled_reason", parent="pyecr_stage_fit_content", wrap=TOKENS.wraps.stage)
                dpg.add_text(
                    "FIT READY NO", tag="pyecr_fit_ready_status",
                    parent="pyecr_stage_fit_content", wrap=TOKENS.wraps.stage,
                )
                dpg.add_text(
                    "Import a dataset and apply a Model configuration.",
                    tag="pyecr_fit_blocking_reasons",
                    parent="pyecr_stage_fit_content", wrap=TOKENS.wraps.stage,
                )
                dpg.add_text(
                    "", tag="pyecr_fit_quality_warnings",
                    parent="pyecr_stage_fit_content", wrap=TOKENS.wraps.stage,
                )
                dpg.add_progress_bar(tag=WidgetTags.progress("fit"), default_value=0.0, width=-1)
                dpg.add_text("Idle", tag=WidgetTags.task_status("fit"), wrap=TOKENS.wraps.rail)
                dpg.add_text("", tag="pyecr_fit_live_status", wrap=TOKENS.wraps.rail)
                with dpg.collapsing_header(label="Fit task Details", tag="pyecr_fit_progress_details_header"):
                    with dpg.child_window(width=-1, height=TOKENS.heights.fit_details):
                        dpg.add_text("No live task telemetry.", tag="pyecr_fit_progress_details",
                                     wrap=TOKENS.wraps.rail)
                with dpg.collapsing_header(label="Local solver portfolio Details",tag="pyecr_solver_details_header"):
                    dpg.add_input_text(default_value="No completed solver records.",tag="pyecr_solver_details",
                        multiline=True,readonly=True,width=-1,height=TOKENS.heights.fit_details)
                dpg.add_text("M3_DUAL_H2O_D / M5_DUAL_H2O_DK: seeded random starts; Model initial values do not seed staged MultiStart.",
                             tag="pyecr_fit_start_note", wrap=TOKENS.wraps.rail)
                with dpg.group(parent="pyecr_stage_advanced_content"):
                    self._build_preprocessing_robustness()
                dpg.add_progress_bar(tag=WidgetTags.progress("diagnostics"), default_value=0.0, width=-1)
                dpg.add_text("Idle", tag=WidgetTags.task_status("diagnostics"))

            dpg.add_button(label='||',tag='pyecr_workflow_splitter',width=TOKENS.analysis.gap,height=TOKENS.heights.plot)
            with dpg.child_window(tag="pyecr_analysis_center", width=TOKENS.viewport.center_initial, height=-1):
                dpg.add_text("VISUALIZATION")
                dpg.add_text('Import a dataset to view curves.',tag='pyecr_analysis_empty_hint',color=TOKENS.steps.secondary)
                with dpg.tab_bar(tag="pyecr_analysis_plot_tabs"):
                    for label, name in (
                        ("Raw", "raw"), ("Processed", "processed"),
                        ("Fit", "fit"), ("Residual", "residual"),
                        ("Components", "components"),
                        ("Boundary", "boundary"),
                        ("Explorer", "explorer"),
                        ("Sensitivity", "sensitivity"),
                        ("Error Map", "error_map"),
                    ):
                        with dpg.tab(label=label, tag=f"pyecr_analysis_{name}_tab"):
                            if name == "error_map":
                                with dpg.group(tag="pyecr_error_map_single_area"):
                                    self._build_error_map_area()
                                with dpg.group(tag="pyecr_dual_error_map_area", show=False):
                                    self._build_dual_error_map_area()
                            elif name == "sensitivity":
                                self._build_sensitivity_area()
                            else:
                                with dpg.group(tag=f"pyecr_analysis_{name}_plot_host"):
                                    if name == "fit":
                                        with dpg.group(horizontal=True):
                                            dpg.add_checkbox(label="Sparse observed markers (display only)",tag="pyecr_display_sparse_markers",default_value=False,callback=self._change_sparse_marker_display)
                                            dpg.add_text("",tag="pyecr_marker_display_status")
                                        dpg.add_group(tag="pyecr_main_fit_host")
                                        with dpg.plot(label="Observation residuals · full eligible rows",tag="pyecr_fit_residual_mirror",no_title=True,height=TOKENS.heights.residual,width=-1):
                                            dpg.add_plot_axis(dpg.mvXAxis,label="Time (s)",tag="pyecr_fit_residual_mirror_x")
                                            dpg.add_plot_axis(dpg.mvYAxis,label="Observed − fitted",tag="pyecr_fit_residual_mirror_y",tick_format="%.1e")
                                            dpg.add_scatter_series([],[],parent="pyecr_fit_residual_mirror_y",tag="pyecr_fit_residual_mirror_series")
                                            dpg.add_line_series([],[],parent="pyecr_fit_residual_mirror_y",tag="pyecr_fit_residual_mirror_zero")
                                    if name == "raw":
                                        dpg.add_collapsing_header(
                                            label="File preview · expand to inspect rows",
                                            tag="pyecr_analysis_raw_preview_header",
                                            default_open=True,
                                        )

                with dpg.collapsing_header(label='Results summary — expand below plots',tag='pyecr_analysis_summary_bottom',default_open=False):
                    dpg.add_group(tag='pyecr_analysis_summary_bottom_host')
            with dpg.child_window(tag="pyecr_analysis_right", width=TOKENS.viewport.right_initial, height=-1):
                dpg.add_text("RESULTS & STATUS")
                with dpg.group(tag="pyecr_fit_results_group"):
                    dpg.add_separator()
                    dpg.add_text("Parameters · Local intervals · Reliability",tag='pyecr_fit_result_heading')
                    self._build_fit_output_area()
                with dpg.group(tag="pyecr_diagnostic_results_group"):
                    dpg.add_separator()
                    dpg.add_text("PARAMETER IDENTIFIABILITY")
                    dpg.add_group(tag="pyecr_identifiability_panel")
                    dpg.add_separator()
                    dpg.add_text("MODEL ADEQUACY")
                    dpg.add_group(tag="pyecr_model_adequacy_panel")
                    self._build_diagnostics_output_area()
                dpg.add_button(
                    label="Capture Snapshot",
                    tag="pyecr_analysis_capture_snapshot",
                    callback=self._capture_report_snapshot,
                    enabled=False,
                )
                add_action_tooltip(dpg, "pyecr_analysis_capture_snapshot")
                dpg.add_button(
                    label="Export Report",
                    tag="pyecr_analysis_export_report",
                    callback=self._open_reports_workflow,
                    enabled=True,
                )
                add_action_tooltip(dpg, "pyecr_analysis_export_report")
                dpg.add_text("", tag="pyecr_analysis_report_reason", wrap=TOKENS.wraps.rail_alt)
                self._build_current_error_panel()

        # Move already-built presentation widgets without replacing their series,
        # callback bindings, or tag identities.
        for item, parent in (
            ("pyecr_data_preview_container", "pyecr_analysis_raw_preview_header"),
            ("pyecr_data_raw_plot", "pyecr_analysis_raw_plot_host"),
            ("pyecr_preprocess_raw_plot", "pyecr_analysis_processed_plot_host"),
            ("pyecr_data_processed_plot", "pyecr_analysis_processed_plot_host"),
            ("pyecr_data_processed_interpretation", "pyecr_analysis_processed_plot_host"),
            ("pyecr_fit_plot", "pyecr_main_fit_host"),
            ("pyecr_residual_scale_controls", "pyecr_analysis_residual_plot_host"),
            ("pyecr_residual_plot", "pyecr_analysis_residual_plot_host"),
            ("pyecr_decomposition_controls", "pyecr_analysis_components_plot_host"),
            ("pyecr_components_plot", "pyecr_analysis_components_plot_host"),
            ("pyecr_decomposition_early_plot", "pyecr_analysis_components_plot_host"),
            ("pyecr_decomposition_residual_plot", "pyecr_analysis_components_plot_host"),
            ("pyecr_boundary_plot", "pyecr_analysis_boundary_plot_host"),
            ("pyecr_boundary_plot_note", "pyecr_analysis_boundary_plot_host"),
            ("pyecr_explorer_plot", "pyecr_analysis_explorer_plot_host"),
            ("pyecr_data_processing_separator", "pyecr_stage_preprocess_content"),
            ("pyecr_data_processing_header", "pyecr_stage_preprocess_content"),
            ("pyecr_data_processing_description", "pyecr_stage_preprocess_content"),
            ("pyecr_data_stability_controls", "pyecr_stage_preprocess_content"),
            ("pyecr_data_t0_controls", "pyecr_stage_preprocess_content"),
            ("pyecr_optional_controls", "pyecr_stage_preprocess_content"),
            ("pyecr_typed_processing_controls", "pyecr_stage_preprocess_content"),
            ("pyecr_signed_controls", "pyecr_stage_preprocess_content"),
            ("pyecr_data_processing_parameters", "pyecr_stage_preprocess_content"),
            ("pyecr_data_processing_buttons", "pyecr_stage_preprocess_content"),
            ("pyecr_data_processing_reason", "pyecr_stage_preprocess_content"),
            ("pyecr_typed_preview_evidence", "pyecr_stage_preprocess_content"),
            ("pyecr_signed_review_group", "pyecr_stage_preprocess_content"),
            ("pyecr_data_fit_quality", "pyecr_stage_preprocess_content"),
            ("pyecr_data_processing_history", "pyecr_stage_preprocess_content"),
            ("pyecr_boundary_controls", "pyecr_stage_boundary_content"),
            ("pyecr_diagnostics_actions", "pyecr_stage_advanced_content"),
            ("pyecr_diagnostics_error_map_action", "pyecr_stage_advanced_content"),
            ("pyecr_diagnostics_dual_map_action", "pyecr_stage_advanced_content"),
            ("pyecr_diagnostics_disabled_reason", "pyecr_stage_advanced_content"),
            ("pyecr_diagnostics_cancel_reason", "pyecr_stage_advanced_content"),
            ("pyecr_explorer_controls", "pyecr_stage_advanced_content"),
            ("pyecr_diagnostics_jacobian", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_bi_status", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_bi_container", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_parameter_status", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_parameter_container", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_correlation_status", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_correlation_container", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_multistart_status", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_multistart_container", "pyecr_identifiability_panel"),
            ("pyecr_guided_refit_controls", "pyecr_identifiability_panel"),
            ("pyecr_diagnostics_reliability", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_warnings", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_residual_summary", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_information_criteria", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_error_map_status", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_error_map_warnings", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_dual_map_status", "pyecr_model_adequacy_panel"),
            ("pyecr_diagnostics_dual_map_warnings", "pyecr_model_adequacy_panel"),
        ):
            dpg.move_item(item, parent=parent)
        # The P33 quick-action remains addressable for compatibility, but the
        # complete control surface now lives in the Error Map tab.
        dpg.configure_item("pyecr_diagnostics_error_map_action", show=False)
        # Keep primary actions and cancellation reachable without scrolling a long model form.
        for tag in (WidgetTags.run("fit"),"pyecr_cancel_fit","pyecr_run_diagnostics","pyecr_cancel_diagnostics",
                    "pyecr_analysis_capture_snapshot","pyecr_analysis_export_report"):
            dpg.move_item(tag,parent="pyecr_workflow_actions")
        dpg.move_item("pyecr_analysis_report_reason",parent="pyecr_stage_export_content")
        dpg.move_item('pyecr_stage_boundary',parent='pyecr_model_experiment_inputs')
        self._build_preprocessing_layout()
        self._build_step_panel_details()
        self._build_r05_control_layout()
        with dpg.item_handler_registry(tag='pyecr_workflow_splitter_handlers'):
            dpg.add_item_activated_handler(callback=self._begin_workflow_resize)
        dpg.bind_item_handler_registry('pyecr_workflow_splitter','pyecr_workflow_splitter_handlers')
        with dpg.handler_registry(tag='pyecr_workflow_resize_handlers'):
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left,threshold=0,callback=self._drag_workflow_resize)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left,callback=self._end_workflow_resize)

    def _build_step_panel_details(self):
        dpg = self._require_dpg()
        # Original top-level shortcut tags/callbacks remain intact. In-step
        # buttons forward the original callback and mirror its availability.
        self._step_action_proxies = {}
        actions = {'fit': ('pyecr_run_fit', 'pyecr_cancel_fit'),
                   'advanced': ('pyecr_run_diagnostics', 'pyecr_cancel_diagnostics'),
                   'export': ('pyecr_analysis_capture_snapshot', 'pyecr_analysis_export_report')}
        for key, originals in actions.items():
            host = f'pyecr_stage_{key}_content'
            first = (dpg.get_item_children(host, 1) or [None])[0]
            row = dpg.add_group(parent=host, horizontal=True, **({'before': first} if first else {}))
            for original in originals:
                cfg = dpg.get_item_configuration(original)
                tag = 'pyecr_step_action_'+original
                dpg.add_button(parent=row, tag=tag, label=cfg['label'],
                               callback=self._invoke_step_action, user_data=original, enabled=False)
                self._step_action_proxies[tag] = original
        for key in ('preprocess', 'model', 'fit', 'advanced', 'export'):
            with dpg.collapsing_header(parent=f'pyecr_stage_{key}_content',
                                      label='Details', tag=f'pyecr_step_{key}_details'):
                dpg.add_button(label='Copy complete details', callback=self._copy_step_details, user_data=key)
                dpg.add_input_text(tag=f'pyecr_step_{key}_details_text', readonly=True, multiline=True,
                                   width=-1, height=TOKENS.steps.details_height)
        for tag in ('pyecr_fit_progress_details_header', 'pyecr_solver_details_header'):
            dpg.move_item(tag, parent='pyecr_step_fit_details')
        for tag in (WidgetTags.progress('fit'), WidgetTags.task_status('fit'), 'pyecr_fit_live_status',
                    'pyecr_fit_start_note'):
            dpg.move_item(tag, parent='pyecr_stage_fit_content')
        for tag in (WidgetTags.progress('diagnostics'), WidgetTags.task_status('diagnostics')):
            dpg.move_item(tag, parent='pyecr_stage_advanced_content')
        # The old long Next instruction is preserved on demand.
        dpg.move_item('pyecr_workflow_next_step', parent='pyecr_workflow_help_window')
        layout = 'pyecr_preprocessing_layout'
        dpg.add_text('', tag='pyecr_step_preprocess_units', parent='pyecr_stage_preprocess_content',
                     before=layout, wrap=TOKENS.wraps.stage)
        for step in WORKFLOW_STEPS:
            self._prepare_owned_form(f'pyecr_stage_{step.key}_content')

    def _invoke_step_action(self, sender=None, app_data=None, user_data=None):
        dpg = self._require_dpg()
        if not dpg.get_item_configuration(user_data).get('enabled', False):
            return
        callback = dpg.get_item_callback(user_data)
        if callback is not None:
            callback(user_data, app_data, dpg.get_item_user_data(user_data))

    def _copy_step_details(self, sender=None, app_data=None, user_data=None):
        tag = 'pyecr_file_details_text' if user_data == 'data' else f'pyecr_step_{user_data}_details_text'
        if user_data in ('fit','advanced'):
            self._fit_details_force=user_data
            try:self._refresh_widgets()
            finally:self._fit_details_force=None
        self._require_dpg().set_clipboard_text(self._require_dpg().get_value(tag))

    def _refresh_step_panels(self):
        """View-only refresh, after old action availability/disclosure refresh."""
        dpg = self._require_dpg()
        session, state = self.controller.analysis_session, self.controller.state
        choice = dpg.get_value('pyecr_workflow_step')
        current = WORKFLOW_STEPS[step_labels().index(choice)].key
        for step in WORKFLOW_STEPS:
            text, severity = step_status(step.key, session, state)
            quality = getattr(self, '_fit_quality_cache_report', None)
            if step.key == 'fit' and severity != 'error' and quality is not None and not quality.ready:
                text = 'Blocked: '+ '; '.join(quality.blocking_reasons)
            if severity != 'error' and not step_available(step, session.state):
                text = 'Unavailable.'
            warnings = () if step.key != 'fit' or quality is None else quality.warnings
            if step.key == 'advanced' and state.diagnostics_presentation is not None:
                diagnostic = state.diagnostics_presentation
                warnings = diagnostic.gate_failures + (() if diagnostic.evaluation is None else diagnostic.evaluation.warnings)
            warning_tag = f'pyecr_step_{step.key}_warning'
            from pyecr.gui.workflow_display import brief_text
            dpg.set_value(warning_tag, '' if not warnings else 'Warning: '+brief_text(warnings[0])+
                          (f' (+{len(warnings)-1}; Details)' if len(warnings)>1 else ' — Details'))
            dpg.configure_item(warning_tag, show=bool(warnings), color=TOKENS.colors.warning)
            if step.key == 'data' and severity != 'error':
                preview = self.controller.session.import_preview
                if preview is not None and not preview.parser_valid:
                    text = 'Import blocked: '+str(preview.parse_warning or preview.delimiter_detection_message or 'Review parsing in File details.')
                    severity = 'error'
                elif preview is not None and session.dataset is None:
                    text = 'Import draft ready; confirm columns and units.'
            dpg.set_value(f'pyecr_step_{step.key}_status', text)
            dpg.configure_item(f'pyecr_step_{step.key}_status', wrap=max(TOKENS.analysis.field_min,
                dpg.get_item_configuration('pyecr_analysis_left')['width']-TOKENS.analysis.form_gutter-2*TOKENS.steps.padding))
            theme = 'error' if severity == 'error' else 'current' if step.key == current else 'normal'
            dpg.bind_item_theme(f'pyecr_step_panel_{step.key}', 'pyecr_step_theme_'+theme)
            if step.key != 'data' and (step.key!='fit' or self._fit_details_requested()):
                dpg.set_value(f'pyecr_step_{step.key}_details_text', details_text(step.key, dpg))
        for proxy, original in self._step_action_proxies.items():
            cfg = dpg.get_item_configuration(original)
            dpg.configure_item(proxy, enabled=cfg['enabled'], show=cfg['show'])
        preview = self.controller.session.import_preview
        raw = None if session.preprocessing is None else session.preprocessing.source
        raw = session.dataset if raw is None else raw
        identity = (id(preview), id(raw), id(session.dataset))
        if getattr(self, '_step_file_details_identity', None) != identity:
            self._step_file_details_identity = identity
            dpg.set_value('pyecr_file_details_text', file_details(preview, raw, session.dataset))
            if preview is None:
                # Opening a project clears the controller's import draft. Do
                # not leave the previous preview's path/statistics on screen.
                dpg.set_value('pyecr_data_source_path', '')
                dpg.set_value('pyecr_data_preview_summary', 'No active import draft.')
                for tag in ('pyecr_data_header_status', 'pyecr_data_text_parse_status', 'pyecr_data_time_suggestion'):
                    dpg.set_value(tag, 'No active import draft.')
        dataset = session.dataset
        units = 'Units unconfirmed.' if dataset is None else f"Time: {dataset.metadata.get('time_unit', 'unknown')} | Signal: {dataset.metadata.get('signal_unit', 'unknown')}"
        dpg.set_value('pyecr_step_preprocess_units', units)
        # Only secondary presentation is hidden; old values and callbacks stay.
        for tag in ('pyecr_preprocess_overview', 'pyecr_diagnostics_cancel_reason',
                    'pyecr_model_apply_status', 'pyecr_model_derived_values', 'pyecr_model_derived_header',
                    'pyecr_fit_live_status', 'pyecr_fit_start_note'):
            dpg.configure_item(tag, show=False)
        for key in ('fit', 'diagnostics'):
            dpg.configure_item(WidgetTags.task_status(key), show=False)
            dpg.configure_item(WidgetTags.progress(key), show=state.tasks[GuiPage.FIT if key == 'fit' else GuiPage.DIAGNOSTICS].status == 'running')
        # Static readiness reasons remain in Details; one short prerequisite is visible.
        dpg.configure_item('pyecr_diagnostics_disabled_reason', show=False)

    def _refresh_analysis_workspace(self) -> None:
        dpg = self._require_dpg()
        session = self.controller.analysis_session
        model = str(dpg.get_value("pyecr_model_id"))
        selected_dual = model.startswith(("M2_", "M3_", "M8_"))
        active_model = session.current_analysis_config
        dual = bool(active_model is not None and active_model.model_id.value.startswith(("M2_", "M3_", "M8_")))
        dpg.set_value("pyecr_workflow_next_step", next_step_text(session.state))
        for step in WORKFLOW_STEPS:
            available = step_available(step,session.state)
            dpg.configure_item(f"pyecr_stage_{step.key}",show=True)
            dpg.configure_item(f"pyecr_stage_{step.key}_content",show=available)
            reason = {'imported': 'Import a dataset first.', 'configured': 'Import data and Apply a configuration first.',
                      'fitted': 'Requires a current Fit result.', 'none': ''}[step.prerequisite]
            dpg.set_value(f"pyecr_stage_{step.key}_reason", "" if available else reason)
            dpg.configure_item(f"pyecr_stage_{step.key}_reason",show=not available)
        boundary = session.current_analysis_config
        nonideal = bool(boundary is not None and boundary.boundary is not None
                        and boundary.boundary.mode != "ideal_step")
        presentation = self.controller.state.fit_presentation
        preview = self.controller.session.import_preview
        preview_available = preview is not None
        components_available = bool(
            presentation is not None and ((presentation.component_H and presentation.component_O) or (presentation.component_1 and presentation.component_2))
        )
        for name, show in plot_tab_visibility(
            session.state, dual_model=dual, nonideal_boundary=nonideal,
            components_available=components_available,
            preview_available=preview_available,
            diagnostics_available=self.controller.state.diagnostics_presentation is not None,
        ).items():
            dpg.configure_item(f"pyecr_analysis_{name}_tab", show=show)
        if preview_available and session.dataset is None and preview is not getattr(self, "_analysis_seen_preview", None):
            self._analysis_seen_preview = preview
            dpg.set_value("pyecr_analysis_raw_preview_header", True)
        if session.dataset is not None and getattr(self, "_analysis_seen_import_revision", None) != session.import_revision:
            self._analysis_seen_import_revision = session.import_revision
            dpg.set_value("pyecr_analysis_raw_preview_header", False)
            dpg.set_value("pyecr_stage_data", False)
            dpg.set_value("pyecr_stage_model", True)
            self._select_analysis_step(None, True, 'model')
            for step in WORKFLOW_STEPS:
                dpg.set_value('pyecr_stage_'+step.key, step.key == 'model')
        if active_model is not None and getattr(self, "_analysis_seen_config_revision", None) != session.config_revision:
            self._analysis_seen_config_revision = session.config_revision
            dpg.set_value("pyecr_stage_model", False)
            dpg.set_value("pyecr_stage_fit", True)
            self._select_analysis_step(None, True, 'fit')
            for step in WORKFLOW_STEPS:
                dpg.set_value('pyecr_stage_'+step.key, step.key == 'fit')
        result_identity = (id(session.fit_result), id(session.diagnostics))
        if result_identity != getattr(self, '_step_seen_results', None):
            self._step_seen_results = result_identity
            key = 'export' if session.diagnostics is not None and not session.diagnostics_stale else 'advanced'
            if session.fit_result is not None and not session.fit_stale:
                for step in WORKFLOW_STEPS:
                    dpg.set_value('pyecr_stage_'+step.key, step.key == key)
                self._select_analysis_step(None, True, key)
        fit = session.fit_result
        selected_m2_without_k = bool(
            fit is not None and not session.fit_stale
            and "D_H_cm2_s" in fit.params and "D_O_cm2_s" in fit.params
            and not ("k_H_cm_s" in fit.params and "k_O_cm_s" in fit.params)
        )
        dual_view = bool(
            self.controller.workflow.can_dual_error_map
            or selected_m2_without_k
            or (session.dual_error_map_result is not None and session.dual_error_map_stale
                and not self.controller.workflow.can_error_map)
        )
        single_view = self.controller.workflow.can_error_map
        dpg.configure_item("pyecr_analysis_error_map_tab", show=single_view or dual_view)
        dpg.configure_item("pyecr_error_map_single_area", show=single_view and not dual_view)
        dpg.configure_item("pyecr_dual_error_map_area", show=dual_view)
        dpg.set_value(WidgetTags.summary("analysis"), self.controller.state.status_summary(GuiPage.DATA))
        quality = self._fit_quality_for_display()
        fit_ready = quality.ready
        blocking = list(quality.blocking_reasons)
        if not fit_ready and not blocking:
            blocking.append(
                "Import a dataset and apply a validated Model configuration."
            )
        warnings = list(quality.warnings)
        dpg.set_value("pyecr_fit_ready_status", f"FIT READY {'YES' if fit_ready else 'NO'}")
        dpg.set_value(
            "pyecr_fit_blocking_reasons",
            "Blocking reasons: none" if not blocking else
            "Blocking reasons:\n" + "\n".join(f"• {reason}" for reason in blocking),
        )
        dpg.set_value(
            "pyecr_fit_quality_warnings",
            "Warnings: none" if not warnings else
            "Warnings:\n" + "\n".join(f"• {warning}" for warning in warnings),
        )
        disabled_reason = "\n".join(blocking)
        dpg.set_value(
            "pyecr_fit_disabled_reason",
            "" if fit_ready else "Run Fit is blocked; review the reasons below.",
        )
        set_action_availability(
            dpg, WidgetTags.run("fit"),
            enabled=(fit_ready
                     and self.controller.state.tasks[GuiPage.FIT].status != "running"),
            reason=disabled_reason or "A Fit task is already running.",
        )
        dpg.set_value(
            "pyecr_stage_boundary_reason",
            boundary_support_message(self.controller.model_descriptor(model)),
        )
        dpg.set_value(
            "pyecr_boundary_stage_note",
            "",
        )
        self._render_boundary_assessment(selected_dual=selected_dual)
        self._refresh_current_error_panel()
        self._refresh_error_map_widgets()
        self._refresh_dual_error_map_widgets()
        can_report = self.controller.can_capture_snapshot
        for tag in ("pyecr_analysis_capture_snapshot",):
            set_action_availability(
                dpg, tag, enabled=can_report,
                reason="Requires a current Fit result; Out of date results cannot be reported.",
            )
        set_action_availability(dpg,'pyecr_analysis_export_report',enabled=True,reason='Choose stored results and content in Reports.')
        dpg.set_value('pyecr_analysis_report_reason','')
        self._resize_analysis_columns()

    def _render_boundary_assessment(self, *, selected_dual: bool) -> None:
        dpg = self._require_dpg()
        descriptor=self.controller.model_descriptor(str(dpg.get_value("pyecr_model_id")))
        support=boundary_support_message(descriptor)
        if support:
            dpg.set_value(
                "pyecr_boundary_flush_assessment",
                support,
            )
            return
        session = self.controller.analysis_session
        config = session.current_analysis_config
        dataset = session.dataset
        if config is None or config.boundary is None or dataset is None:
            dpg.set_value(
                "pyecr_boundary_flush_assessment",
                "Apply 03 M2_SINGLE_DK configuration and import data to assess boundary coverage.",
            )
            return
        from pyecr.boundary.assistant import (
            assess_flush, validate_boundary_fit_coverage,
        )

        if config.boundary.mode == "measured_history":
            try:
                validate_boundary_fit_coverage(dataset, config)
            except ValueError as exc:
                coverage = f"Measured boundary coverage FAIL: {exc}. Extrapolation is prohibited."
            else:
                coverage = "Measured boundary covers the complete Fit time; no extrapolation."
        else:
            coverage = ""
        fit = session.fit_result if not session.fit_stale else None
        if fit is None:
            assessment_text = "Run a current 03 M2_SINGLE_DK Fit to assess τ_flush/τ_material."
        else:
            cache_key = self._flush_assessment_cache_key
            if (cache_key is None or cache_key[0] is not config
                    or cache_key[1] is not fit):
                self._flush_assessment_cache_result = assess_flush(config, fit)
                self._flush_assessment_cache_key = (config, fit)
            assessment = self._flush_assessment_cache_result
            assert assessment is not None
            ratio = assessment.flush_to_material_ratio
            ratio_text = "N/A" if ratio is None else f"{ratio:.3g}"
            material = assessment.tau_material_s
            material_text = "N/A" if material is None else f"{material:.3g} s"
            flush = assessment.tau_flush_s
            flush_text = "N/A" if flush is None else f"{flush:.3g} s"
            assessment_text = (
                f"Flush {assessment.status} ({assessment.severity}): "
                f"τ_material={material_text}, τ_flush={flush_text}, "
                f"τ_flush/τ_material={ratio_text}. {assessment.explanation}"
            )
        dpg.set_value(
            "pyecr_boundary_flush_assessment",
            assessment_text + ("\n" + coverage if coverage else ""),
        )

    def _resize_analysis_columns(self) -> None:
        dpg = self._require_dpg()
        pending=getattr(self,'_results_scroll_pending',None)
        if pending is not None and dpg.get_frame_count()>=pending:
            dpg.set_y_scroll('pyecr_analysis_center',dpg.get_y_scroll_max('pyecr_analysis_center'))
            self._results_scroll_pending=None
        viewport = int(dpg.get_viewport_width())
        height = int(dpg.get_viewport_height())
        override = getattr(self, "_analysis_results_override", None)
        task = self.controller.state.tasks[GuiPage.FIT]
        has_results = (self.controller.state.fit_presentation is not None
                       or (self.controller.live_fit_is_current and (task.status in {'running', 'cancelled'}
                           or self.controller.current_fit_candidate is not None)))
        from pyecr.gui.workflow_display import InformationLevel
        if override is None and not has_results and getattr(self,"_information_level",InformationLevel.DEFAULT)==InformationLevel.DEFAULT:
            override=False
        center_height=int(dpg.get_item_rect_size("pyecr_analysis_center")[1])
        width=int(dpg.get_item_rect_size(WidgetTags.PRIMARY_WINDOW)[0]) or viewport
        preferred=getattr(self,'_workflow_preferred_width',None)
        if getattr(self,'_results_docked_bottom',False):
            opened=bool(dpg.get_value('pyecr_analysis_summary_bottom'))
            if opened!=getattr(self,'_rendered_bottom_summary_open',opened):
                override=opened;self._analysis_results_override=opened
        key = (width, height, override,center_height,preferred)
        if getattr(self, "_last_analysis_layout_key", None) == key:
            return
        self._last_analysis_layout_key = key
        visible=bool(has_results) if override is None else override
        columns = analysis_workspace_columns(width,preferred_left=preferred,results_visible=visible)
        if columns.results_bottom:
            if dpg.get_item_parent('pyecr_analysis_right')!=dpg.get_alias_id('pyecr_analysis_summary_bottom_host'):
                dpg.move_item('pyecr_analysis_right',parent='pyecr_analysis_summary_bottom_host')
            dpg.configure_item('pyecr_analysis_right',show=True,width=-1,height=TOKENS.analysis.summary_height)
            dpg.configure_item('pyecr_analysis_summary_bottom',show=True)
            dpg.set_value('pyecr_analysis_summary_bottom',columns.results_visible)
            self._rendered_bottom_summary_open=columns.results_visible
        else:
            if dpg.get_item_parent('pyecr_analysis_right')!=dpg.get_alias_id('pyecr_analysis_columns'):
                dpg.move_item('pyecr_analysis_right',parent='pyecr_analysis_columns')
            dpg.configure_item('pyecr_analysis_right',show=columns.results_visible,width=columns.right,height=-1)
            dpg.configure_item('pyecr_analysis_summary_bottom',show=False)
        self._results_docked_bottom=columns.results_bottom
        # Wrap explanations inside the actual dock, also when it moves below
        # the plot in a narrow window. Font and scientific arrays stay intact.
        result_width = columns.center if columns.results_bottom else columns.right
        for tag in ('pyecr_fit_summary','pyecr_fit_reliability','pyecr_fit_context_summary',
                    'pyecr_diagnostics_reliability','pyecr_diagnostics_parameter_status',
                    'pyecr_reliability_identifiability','pyecr_reliability_adequacy'):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag,wrap=max(120,result_width-TOKENS.spacing.fit_details_gutter))
        dpg.configure_item(
            "pyecr_analysis_toggle_results",
            label="Hide Results" if columns.results_visible else "Show Results",
        )
        for tag, width in (
            ("pyecr_analysis_left", columns.left),
            ("pyecr_analysis_center", columns.center),
        ):
            dpg.configure_item(tag, width=width)
        dpg.configure_item('pyecr_workflow_width',max_value=columns.max_left)
        dpg.set_value('pyecr_workflow_width',columns.left)
        dpg.configure_item('pyecr_workflow_splitter',height=max(TOKENS.analysis.field_min,center_height))
        self._refresh_owned_analysis_forms()
        plot_height = analysis_plot_height(height)
        for tag in (
            "pyecr_data_raw_plot", "pyecr_data_processed_plot", "pyecr_fit_plot",
            "pyecr_residual_plot", "pyecr_components_plot", "pyecr_boundary_plot",
            "pyecr_explorer_plot", "pyecr_error_map_plot",
        ):
            dpg.configure_item(tag, height=plot_height)
        available=max(TOKENS.heights.fit_main_min+TOKENS.heights.fit_residual_min,center_height-TOKENS.plot.plot_chrome)
        dpg.configure_item("pyecr_fit_plot",height=max(TOKENS.heights.fit_main_min,int(available*TOKENS.plot.fit_main_fraction)))
        residual_height = residual_panel_height(available)
        # The standalone tab should fit the actual child height, not the
        # requested outer window (which includes chrome and navigation).
        dpg.configure_item("pyecr_residual_plot",height=max(TOKENS.heights.residual,
            min(plot_height,center_height-TOKENS.plot.plot_chrome)))
        dpg.configure_item("pyecr_fit_residual_mirror",height=residual_height)
        dpg.configure_item("pyecr_decomposition_residual_plot",height=residual_height)
        dpg.configure_item("pyecr_diagnostics_residual_plot",height=residual_height)
        # Leave room for the optional early inset and residual axis in Components.
        dpg.configure_item("pyecr_components_plot", height=max(TOKENS.heights.dct_plot, min(TOKENS.heights.decomposition_main_max, int(plot_height * .60))))
        raw_context_height = max(185, min(210, int(plot_height * 0.45)))
        dpg.configure_item("pyecr_preprocess_raw_plot", height=raw_context_height)
        dpg.configure_item("pyecr_data_processed_plot", height=max(250, plot_height - raw_context_height))
        map_geometry = error_map_geometry(columns.center, plot_height)
        dpg.configure_item("pyecr_error_map_plot", width=map_geometry.plot_width)
        dpg.configure_item("pyecr_error_map_left_spacer", width=map_geometry.left_spacer)
        dpg.configure_item(
            "pyecr_error_map_colorbar",
            width=map_geometry.colorbar_width,
            height=map_geometry.plot_height,
        )
        dual_geometry = dual_error_map_geometry(columns.center, plot_height)
        dpg.configure_item("pyecr_dual_map_plots", horizontal=dual_geometry.horizontal)
        for process in ("h", "o"):
            dpg.configure_item(f"pyecr_dual_map_{process}_plot",
                               width=dual_geometry.plot_width,
                               height=dual_geometry.plot_height)
            dpg.configure_item(f"pyecr_dual_map_{process}_colorbar",
                               width=dual_geometry.colorbar_width,
                               height=dual_geometry.plot_height)
        self._resize_map_focus()

    def _toggle_analysis_results(self, sender: object, app_data: object, user_data: object) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        current=bool(dpg.get_value('pyecr_analysis_summary_bottom')) if getattr(self,'_results_docked_bottom',False) else dpg.get_item_configuration("pyecr_analysis_right")["show"]
        self._analysis_results_override = not current
        self._resize_analysis_columns()
        if self._analysis_results_override and getattr(self,'_results_docked_bottom',False):
            self._results_scroll_pending=dpg.get_frame_count()+2

    def _change_workflow_width(self,sender=None,app_data=None,user_data=None):
        dpg=self._require_dpg()
        self._workflow_preferred_width=int(dpg.get_value('pyecr_workflow_width') if app_data is None else app_data)
        self._resize_analysis_columns()
        self._refresh_information_layer()

    def _reset_workflow_width(self,*args):
        self._workflow_preferred_width=None
        self._resize_analysis_columns()

    def _begin_workflow_resize(self,*args):
        self._workflow_resize_start=self._require_dpg().get_item_configuration('pyecr_analysis_left')['width']

    def _drag_workflow_resize(self,sender=None,app_data=None,user_data=None):
        dpg=self._require_dpg()
        if not dpg.is_item_active('pyecr_workflow_splitter'):return
        delta=app_data[1] if isinstance(app_data,(list,tuple)) and len(app_data)>1 else dpg.get_mouse_drag_delta()[0]
        start=getattr(self,'_workflow_resize_start',None)
        if start is None:
            start=dpg.get_item_configuration('pyecr_analysis_left')['width'];self._workflow_resize_start=start
        self._change_workflow_width(None,start+delta,None)

    def _end_workflow_resize(self,*args):
        self._workflow_resize_start=None
