"""DearPyGui desktop shell for PyECR.

Widget callbacks only edit presentation state or dispatch a service request.
Scientific implementations are deliberately absent from this module.
"""

from __future__ import annotations

import importlib
import traceback
from time import monotonic
from pyecr.gui.fit_progress import progress_label, progress_details
from pyecr.model_display import display_model_text, display_fit_worker_text
from types import ModuleType
from typing import Any

from pyecr.gui.components.plot_tabs import (
    _replace_data_preview_table,
    _update_processed_plot,
)
from pyecr.gui.components.dependencies import (
    _DATA_PREVIEW_CONTAINER_TAG,
    _DATA_PREVIEW_TABLE_TAG,
)
from pyecr.gui.components.scientific_input import ScientificInputMixin
from pyecr.gui.components.display_layout import DisplayLayoutMixin
from pyecr.gui.components.information_layer import InformationLayerMixin
from pyecr.gui.components.analysis_forms import AnalysisFormsMixin
from pyecr.gui.components.data_file_selection import DataFileSelectionMixin
from pyecr.gui.components.map_controls import MapControlsMixin
from pyecr.gui.components.figure_export import FigureExportViewMixin
from pyecr.gui.components.preprocessing_layout import PreprocessingLayoutMixin
from pyecr.gui.components.scientific_plot import bind_scientific_plot_styles
from pyecr.gui.components.status_error import (
    StatusErrorMixin,
    build_error_panel,
    refresh_error_panel,
)
from pyecr.gui.components.action_state import set_action_availability
from pyecr.gui.components.tags import WidgetTags
from pyecr.gui.theme import (
    THEME_TAG,
    TOKENS,
    bind_primary_actions,
    build_fonts,
    build_themes,
    status_color,
    FONT_TAG, NAVIGATION_FONT_TAG,
)
from pyecr.gui.controller import GuiController
from pyecr.gui.state import GUI_PAGES, GuiPage
from pyecr.gui.views.analysis import AnalysisViewMixin
from pyecr.gui.views.analysis_workspace import AnalysisWorkspaceMixin
from pyecr.gui.views.batch import BatchViewMixin
from pyecr.gui.views.dct import DctViewMixin
from pyecr.gui.views.reports import ReportsViewMixin
from pyecr.gui.views.error_map import ErrorMapViewMixin
from pyecr.gui.views.dual_error_map import DualErrorMapViewMixin
from pyecr.gui.views.pair_design import PairDesignViewMixin
from pyecr.gui.views.optional_preprocessing import OptionalPreprocessingViewMixin
from pyecr.gui.views.preprocessing_robustness import PreprocessingRobustnessViewMixin
from pyecr.gui.views.signed_normalization import SignedNormalizationViewMixin
from pyecr.gui.views.simulation import SimulationViewMixin

__all__ = ["PyecrGuiApp", "main"]


_PAGE_DESCRIPTIONS = {
    GuiPage.PROJECT: "Create, open, and save a PyECR project.",
    GuiPage.DATA: "Import datasets and review preprocessing choices.",
    GuiPage.MODEL: "Build and validate an AnalysisConfig from the model registry.",
    GuiPage.FIT: "Launch fitting tasks and inspect their status.",
    GuiPage.DIAGNOSTICS: "Review residual, identifiability, and reliability outputs.",
    GuiPage.BATCH: "Prepare temperature-series and other batch jobs.",
    GuiPage.PAIR: "Compare independently analyzed ECR curves and explore experimental design regions.",
    GuiPage.DCT: "Run characteristic-time-distribution diagnostics.",
    GuiPage.REPORT: "Assemble and export analysis reports.",
    GuiPage.SIMULATION: "Generate independent given-parameter simulation curves.",
}

class PyecrGuiApp(
    DisplayLayoutMixin,
    InformationLayerMixin,
    AnalysisFormsMixin,
    DataFileSelectionMixin,
    MapControlsMixin,
    FigureExportViewMixin,
    PreprocessingLayoutMixin,
    ScientificInputMixin,
    OptionalPreprocessingViewMixin,
    PreprocessingRobustnessViewMixin,
    SignedNormalizationViewMixin,
    SimulationViewMixin,
    AnalysisViewMixin,
    AnalysisWorkspaceMixin,
    ErrorMapViewMixin,
    DualErrorMapViewMixin,
    BatchViewMixin,
    PairDesignViewMixin,
    DctViewMixin,
    ReportsViewMixin,
    StatusErrorMixin,
):
    """Build and run the DearPyGui shell around a :class:`GuiController`."""

    def __init__(self, controller: GuiController | None = None) -> None:
        self.controller = controller or GuiController()
        self._dpg: ModuleType | None = None
        self._built = False
        self._rendered_fit_presentation: object | None = None
        self._rendered_diagnostics_presentation: object | None = None
        self._rendered_dct_presentation: object | None = None
        self._rendered_advanced_diagnostics: dict[str, object] = {}
        self._rendered_import_preview: object | None = None
        self._rendered_raw_dataset: object | None = None
        self._rendered_processed_dataset: object | None = None
        self._rendered_processing_config: object | None = None
        self._pending_typed_preview: tuple[object, object, object, int, int] | None = None
        self._rendered_exclusion_key: tuple[int, int, int] | None = None
        self._pending_stability_suggestion: tuple[object, object, int, int] | None = None
        self._rendered_stability_suggestion: object | None = None
        self._pending_t0_suggestion: tuple[object, object, int, int] | None = None
        self._rendered_t0_suggestion: object | None = None
        self._t0_preview_message = (
            "Suggest t0 to preview the detected onset; Accept records a preprocessing step."
        )
        self._fit_quality_cache_key: tuple[object, ...] | None = None
        self._fit_quality_cache_report: object | None = None
        self._flush_assessment_cache_key: tuple[object, object] | None = None
        self._flush_assessment_cache_result: object | None = None
        self._stability_preview_message = (
            "Select Initial or Final, then Suggest. Preview does not change data."
        )
        self._selected_time_column_index: int | None = None
        self._selected_signal_column_index: int | None = None
        self._pending_unit_project_path: str | None = None
        self._explorer_debounce_deadline: float | None = None
        self._rendered_boundary_computation: object | None = None
        self._rendered_explorer_computation: object | None = None
        self._fit_progress_render_key: object | None = None
        self._fit_progress_render_at = 0.0
        self._robustness_variations = []
        self._robustness_render_key = None
        self._robustness_last_rendered_terminal = None

    def build(self) -> None:
        """Create widgets; DearPyGui is imported only when the shell is launched."""

        if self._built:
            return
        dpg = _load_dearpygui()
        self._dpg = dpg
        dpg.create_context()
        # Dynamic map/table replacement and callbacks share the render owner.
        dpg.configure_app(manual_callback_management=True)
        build_fonts(dpg)
        build_themes(dpg)
        dpg.create_viewport(
            title="PyECR", width=TOKENS.viewport.width,
            height=TOKENS.viewport.height, min_width=TOKENS.viewport.min_width,
            min_height=TOKENS.viewport.min_height,
        )
        with dpg.window(tag=WidgetTags.PRIMARY_WINDOW, label="PyECR"):
            with dpg.menu_bar(tag='pyecr_main_menu'):
                if dpg.does_item_exist(NAVIGATION_FONT_TAG):dpg.bind_item_font('pyecr_main_menu',NAVIGATION_FONT_TAG)
                with dpg.menu(label="Project"):
                    self._build_project_area()
                self._build_display_menu()
            dpg.add_text(self._display_font_warning(), tag="pyecr_font_warning",
                         show=bool(self._display_font_warning()), color=TOKENS.colors.warning,
                         wrap=TOKENS.wraps.page)
            self._build_information_bar()
            if dpg.does_item_exist(NAVIGATION_FONT_TAG):dpg.bind_item_font('pyecr_information_controls',NAVIGATION_FONT_TAG)
            with dpg.group(horizontal=True,tag='pyecr_header_actions'):
                if dpg.does_item_exist(NAVIGATION_FONT_TAG):dpg.bind_item_font('pyecr_header_actions',NAVIGATION_FONT_TAG)
                dpg.add_button(label='Export report…',tag='pyecr_figure_open',callback=self._open_reports_workflow)
                dpg.add_button(label='Cancel figure export',tag='pyecr_figure_cancel_active',show=False,callback=lambda:self.controller.cancel_figure_export())
            dpg.add_button(label='Cancel map',tag='pyecr_map_cancel_active',show=False,
                           callback=lambda:self.controller.cancel_page(GuiPage.DIAGNOSTICS))
            with dpg.tab_bar(tag=WidgetTags.PAGES, callback=self._on_tab_change):
                with dpg.tab(label="Analysis", tag=WidgetTags.page("analysis")):
                    if dpg.does_item_exist(NAVIGATION_FONT_TAG):dpg.bind_item_font(WidgetTags.page('analysis'),NAVIGATION_FONT_TAG)
                    with dpg.group() as body:
                        if dpg.does_item_exist(FONT_TAG):dpg.bind_item_font(body,FONT_TAG)
                        self._build_analysis_workspace()
                for page in (GuiPage.BATCH, GuiPage.PAIR, GuiPage.DCT, GuiPage.SIMULATION, GuiPage.REPORT):
                    self._build_page(page)
        self._build_data_file_dialog()
        self._build_project_dialogs()
        self._build_figure_export_dialog()
        bind_scientific_plot_styles(dpg)
        from pyecr.gui.components.plot_views import PlotViews
        self._plot_views = PlotViews(dpg, lambda: self.controller.analysis_session.project_metadata)
        dpg.bind_theme(THEME_TAG)
        bind_primary_actions(dpg)
        dpg.setup_dearpygui()
        dpg.set_primary_window(WidgetTags.PRIMARY_WINDOW, True)
        self._built = True
        self._refresh_widgets()

    def run(self) -> None:
        """Render frames and drain worker events without blocking UI callbacks."""

        self.build()
        dpg = self._require_dpg()
        dpg.show_viewport()
        try:
            while dpg.is_dearpygui_running():
                self._drain_gui_callbacks()
                events = self.controller.poll_events()
                self._refresh_frame_widgets(events)
                dpg.render_dearpygui_frame()
        finally:
            self.controller.shutdown(wait=False)
            dpg.destroy_context()

    def _drain_gui_callbacks(self) -> None:
        dpg = self._require_dpg()
        for job in dpg.get_callback_queue() or ():
            try:
                dpg.run_callbacks([job])
            except Exception as exc:
                self.controller.record_callback_exception(
                    self.controller.state.active_page, exc, traceback.format_exc()
                )

    def _fit_quality_for_display(self) -> Any:
        """Evaluate the current gate once per revision, not on every GUI frame."""

        session = self.controller.analysis_session
        key = (
            id(session.dataset), id(session.current_analysis_config),
            session.current_revision,
            session.preprocessing_revision,
        )
        if key != self._fit_quality_cache_key:
            self._fit_quality_cache_report = self.controller.workflow.fit_quality
            self._fit_quality_cache_key = key
        return self._fit_quality_cache_report

    def _build_page(self, page: GuiPage) -> None:
        dpg = self._require_dpg()
        slug = page.value.lower()
        with dpg.tab(label="Reports" if page is GuiPage.REPORT else page.value,
                     tag=WidgetTags.page("reports" if page is GuiPage.REPORT else slug)):
            if dpg.does_item_exist(NAVIGATION_FONT_TAG):dpg.bind_item_font(WidgetTags.page('reports' if page is GuiPage.REPORT else slug),NAVIGATION_FONT_TAG)
            with dpg.group() as body:
                if dpg.does_item_exist(FONT_TAG):dpg.bind_item_font(body,FONT_TAG)
                if page is GuiPage.SIMULATION:
                    self._build_simulation_area()
                    return
                dpg.add_text(_PAGE_DESCRIPTIONS[page], wrap=TOKENS.wraps.page)
                dpg.add_spacer(height=TOKENS.spacing.spacer_small)
                dpg.add_text(
                    self.controller.state.status_summary(page),
                    tag=WidgetTags.summary(slug),
                    wrap=TOKENS.wraps.page,
                )
                dpg.add_separator()
                if page is GuiPage.BATCH:
                    self._build_batch_area()
                elif page is GuiPage.PAIR:
                    self._build_pair_design_area()
                elif page is GuiPage.DCT:
                    self._build_dct_area()
                elif page is GuiPage.REPORT:
                    self._build_report_area()
                dpg.add_spacer(height=TOKENS.spacing.spacer_medium)
                dpg.add_progress_bar(
                    tag=WidgetTags.progress(slug),
                    default_value=0.0,
                    width=TOKENS.widths.progress,
                    overlay="Idle",
                )
                dpg.add_text("Idle", tag=WidgetTags.task_status(slug))


























































    def _on_tab_change(self, sender: Any, app_data: Any, user_data: Any) -> None:
        selected = str(app_data).removeprefix("pyecr_tab_")
        if selected == "analysis":
            self.controller.state.active_page = GuiPage.DATA
            self._refresh_information_layer()
            return
        if selected == "reports":
            self.controller.state.active_page = GuiPage.REPORT
            self._refresh_information_layer()
            return
        for page in GUI_PAGES:
            if page.value.lower() == selected.lower() or page.value == str(app_data):
                self.controller.state.active_page = page
                break
        self._refresh_information_layer()


    def _refresh_fit_task_widgets(self) -> None:
        """Throttle the live fit presentation to 10 Hz; phase changes are immediate."""
        task = self.controller.state.tasks[GuiPage.FIT]
        snapshot = task.fit_progress
        now = monotonic()
        key = (id(snapshot), task.status, task.message, task.cancellation_requested)
        if key == self._fit_progress_render_key and now - self._fit_progress_render_at < 0.1:
            return
        self._fit_progress_render_key, self._fit_progress_render_at = key, now
        dpg = self._require_dpg()
        cancelling = task.cancellation_requested and task.status == "running"
        fraction = None if snapshot is None or cancelling else snapshot.fraction
        dpg.configure_item("pyecr_progress_fit", show=fraction is not None,
                           overlay=(f"{snapshot.completed_starts}/{snapshot.total_starts} starts finished"
                                    if fraction is not None else ""))
        dpg.set_value("pyecr_progress_fit", fraction or 0.)
        dpg.set_value("pyecr_task_status_fit", f"Status: {task.status} | {display_fit_worker_text(task.message)}")
        dpg.configure_item("pyecr_task_status_fit", color=status_color(task.status))
        elapsed = None if snapshot is None else snapshot.elapsed_s
        if elapsed is not None and task.status == "running":
            elapsed += max(0., now - (snapshot.emitted_monotonic_s
                                    if snapshot.emitted_monotonic_s is not None
                                    else task.progress_received_at))
        label = progress_label(snapshot, cancelling=cancelling, elapsed_s=elapsed)
        if snapshot is None and task.status != "running":
            label = "No live telemetry: idle or result loaded from project."
        dpg.set_value("pyecr_fit_live_status", label)
        dpg.set_value("pyecr_fit_progress_details", progress_details(snapshot))
        if dpg.does_item_exist("pyecr_analysis_left"):
            rail_width = dpg.get_item_rect_size("pyecr_analysis_left")[0]
            wrap = max(TOKENS.widths.input_medium,
                       int(rail_width) - TOKENS.spacing.fit_details_gutter)
            for tag in ("pyecr_fit_live_status", "pyecr_task_status_fit",
                        "pyecr_fit_progress_details", "pyecr_fit_start_note"):
                dpg.configure_item(tag, wrap=wrap)

    def _refresh_widgets(self) -> None:
        if not self._built:
            return
        if hasattr(self, '_plot_views'):
            self._plot_views.before_refresh()
        dpg=self._require_dpg()
        self._refresh_figure_export_widgets()
        for tag in ('pyecr_map_cancel_active','pyecr_map_focus_cancel'):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag,show=self.controller.is_error_map_updating or self.controller.analysis_session.dual_error_map_running)
        self._poll_data_file_picker()
        self._restore_opened_model_display()
        self._refresh_summaries()
        self._refresh_project_widgets()
        for page in GUI_PAGES:
            slug = page.value.lower()
            task = self.controller.state.tasks[page]
            if not dpg.does_item_exist(WidgetTags.progress(slug)):
                continue
            if page is GuiPage.FIT:
                self._refresh_fit_task_widgets()
                continue
            dpg.set_value(f"pyecr_progress_{slug}", task.progress)
            dpg.configure_item(
                f"pyecr_progress_{slug}",
                overlay=f"{task.progress * 100:.0f}% — {display_model_text(task.message)}",
            )
            dpg.set_value(
                f"pyecr_task_status_{slug}",
                f"Status: {task.status} | {display_model_text(task.message)}",
            )
            dpg.configure_item(
                f"pyecr_task_status_{slug}", color=status_color(task.status),
            )
            if dpg.does_item_exist(f"pyecr_run_{slug}"):
                dpg.configure_item(
                    f"pyecr_run_{slug}", enabled=task.status != "running"
                )
        fit_running = self.controller.state.tasks[GuiPage.FIT].status == "running"
        fit_quality = self._fit_quality_for_display()
        set_action_availability(
            dpg, "pyecr_run_fit",
            enabled=(not fit_running and fit_quality.ready),
            reason="Review the Fit quality checks and resolve blocking reasons.",
        )
        dpg.configure_item("pyecr_cancel_fit", enabled=fit_running)

        diagnostics_running = (
            self.controller.state.tasks[GuiPage.DIAGNOSTICS].status == "running"
        )
        can_diagnose = self.controller.workflow.can_diagnose
        for tag in (
            "pyecr_run_diagnostics",
            "pyecr_diagnostics_bootstrap",
            "pyecr_diagnostics_profile",
            "pyecr_diagnostics_experimental_m1",
        ):
            set_action_availability(
                dpg, tag, enabled=can_diagnose and not diagnostics_running,
                reason="Requires a current Fit and no running Diagnostics task.",
            )
        experimental_m1 = (
            self.controller.workflow.can_experimental_m1_uncertainty
            and not diagnostics_running
        )
        set_action_availability(
            dpg, "pyecr_diagnostics_experimental_m1", enabled=experimental_m1,
            reason="Experimental single D+k uncertainty requires linear loss and an ideal step; dual models and robust loss are unsupported. Requires a current fit and explicit assumption acknowledgement.",
        )
        can_error_map = self.controller.workflow.can_error_map
        set_action_availability(
            dpg, "pyecr_diagnostics_error_map_run",
            enabled=can_error_map and not diagnostics_running,
            reason="Requires a current M1 Fit and no running Diagnostics task.",
        )
        dpg.set_value(
            "pyecr_diagnostics_error_map_reason",
            "" if can_error_map else
            "Requires a current single D+k fit.",
        )
        # Retain the legacy action tag for compatibility; the M3 map tab now
        # owns the only visible Generate/Cancel control surface.
        dpg.configure_item("pyecr_diagnostics_dual_map_action", show=False)
        can_dual_map = self.controller.workflow.can_dual_error_map
        set_action_availability(
            dpg, "pyecr_diagnostics_dual_map_run",
            enabled=can_dual_map and not diagnostics_running,
            reason="Requires a current final-selected M3 fit and no running Diagnostics task.",
        )
        dpg.set_value(
            "pyecr_diagnostics_dual_map_reason",
            "" if can_dual_map else
            "Dual D+k maps require a current dual D+k fit with both surface coefficients.",
        )
        dpg.configure_item("pyecr_cancel_diagnostics", enabled=diagnostics_running)
        self._refresh_guided_refit_widgets()
        dpg.set_value(
            "pyecr_diagnostics_disabled_reason",
            "" if can_diagnose else "Requires a current Fit result; Out of date results cannot be diagnosed.",
        )
        batch_running = self.controller.state.tasks[GuiPage.BATCH].status == "running"
        set_action_availability(
            dpg, "pyecr_batch_add_current",
            enabled=(self.controller.analysis_session.dataset is not None and not batch_running),
            reason="Import a current dataset and wait for Batch work to finish.",
        )
        set_action_availability(
            dpg, "pyecr_batch_run_independent",
            enabled=(self.controller.can_run_batch_independent and not batch_running),
            reason="Add inputs, apply a configuration template, and wait for the current task.",
        )
        set_action_availability(
            dpg, "pyecr_batch_run_global",
            enabled=(
                self.controller.can_run_global_analysis and not batch_running
            ),
            reason="Requires compatible M1 inputs and no running Batch task.",
        )
        set_action_availability(
            dpg, "pyecr_batch_cancel", enabled=batch_running,
            reason="No Batch or Global task is running.",
        )
        dpg.set_value(
            "pyecr_batch_independent_reason",
            "" if self.controller.can_run_batch_independent else
            "Add datasets/projects and apply an AnalysisConfig template first.",
        )
        dpg.set_value(
            "pyecr_batch_global_reason",
            "" if self.controller.can_run_global_analysis else
            "Global fits compatible single or dual D+k curves at one temperature, using captured geometry and weights.",
        )
        signed_dct = self._dct_signed_mode()
        dct_running = (any(j.status in {'queued','running'} for j in self.controller.signed_dct_service.history)
                       if signed_dct else self.controller.state.tasks[GuiPage.DCT].status == "running")
        dct_ready = (
            not dct_running
            and (self.controller.analysis_session.dataset is not None if signed_dct else
                 not self.controller.dct_unavailable_reason(
                    use_classical_bi=bool(dpg.get_value("pyecr_dct_classical_bi"))))
        )
        set_action_availability(
            dpg, "pyecr_run_dct",
            enabled=(
                dct_ready
            ),
            reason=("Components require a current parent Fit and Diagnostics." if self._dct_component_mode() else
                    "Import a source and confirm normalization. Whole DCT does not require Fit."),
        )
        set_action_availability(
            dpg, "pyecr_cancel_dct", enabled=dct_running,
            reason="No DCT task is running.",
        )
        dpg.set_value(
            "pyecr_dct_run_reason",
            "" if dct_ready else
            "Import a source and review normalization before Run DCT.",
        )
        dpg.set_value("pyecr_fit_context_summary", display_model_text(self.controller.fit_context_summary))
        self._render_data_import()
        self._render_batch_results()
        self._refresh_pair_design_widgets()
        self._render_fit_presentation()
        self._dispatch_debounced_explorer()
        self._render_async_plot_presentations()
        self._render_diagnostics_presentation()
        self._render_dct_presentation()
        self._refresh_report_widgets()
        self._refresh_simulation()
        self._refresh_analysis_workspace()
        self._refresh_display_layout()
        self._refresh_information_layer()
        self._compact_fields.refresh()
        self._refresh_fit_visuals()
        if hasattr(self, '_plot_views'):
            self._plot_views.refresh()
        self._ui_last_full_refresh = monotonic()

    def _refresh_frame_widgets(self, events=()) -> None:
        """Keep input/plots live; reconcile the full form/status tree at 10 Hz.

        Explicit callbacks still call _refresh_widgets immediately. Task state,
        terminal results, navigation and viewport changes bypass the heartbeat.
        """
        if not self._built:return
        dpg=self._require_dpg();state=self.controller.state
        key=(dpg.get_viewport_client_width(),dpg.get_viewport_client_height(),
             dpg.get_value(WidgetTags.PAGES),state.active_page,
             tuple((t.status,t.task_id,t.cancellation_requested) for t in state.tasks.values()))
        now=monotonic()
        terminal=any(e.kind!='progress' for e in events)
        if (terminal or key!=getattr(self,'_ui_frame_key',None)
                or now-getattr(self,'_ui_last_full_refresh',float('-inf'))>=.1):
            self._ui_frame_key=key
            self._refresh_widgets()
            return
        # Gesture state and axis synchronization must never wait for the form
        # heartbeat. Poll callbacks/workers in the outer loop on every frame.
        if hasattr(self,'_plot_views'):self._plot_views.before_refresh()
        self._poll_data_file_picker()
        self._dispatch_debounced_explorer()
        self._refresh_fit_task_widgets()
        self._refresh_fit_visuals()
        if hasattr(self,'_plot_views'):self._plot_views.refresh()









    def _require_dpg(self) -> ModuleType:
        if self._dpg is None:
            raise RuntimeError("DearPyGui context has not been created")
        return self._dpg


















def _load_dearpygui() -> ModuleType:
    try:
        return importlib.import_module("dearpygui.dearpygui")
    except ImportError as exc:
        raise RuntimeError(
            "DearPyGui is optional. Install PyECR with the 'gui' extra before "
            "launching the desktop shell."
        ) from exc










def main() -> None:
    """Launch the PyECR desktop shell."""

    PyecrGuiApp().run()


if __name__ == "__main__":  # pragma: no cover - manual desktop entry point
    main()
