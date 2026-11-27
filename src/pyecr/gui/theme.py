"""Single source of DearPyGui visual tokens and build-once themes.

Presentation constants only: scientific values and units never pass through
this module. Fonts are coverage-checked by the display-only fonts module.
"""

from __future__ import annotations

import os
from pyecr.plotting.visual_style import SERIES
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any


@dataclass(frozen=True, slots=True)
class Colors:
    window: tuple[int, int, int] = (17, 24, 35)
    child: tuple[int, int, int] = (24, 34, 48)
    frame: tuple[int, int, int] = (34, 47, 63)
    header: tuple[int, int, int] = (42, 59, 77)
    tab_active: tuple[int, int, int] = (36, 107, 135)
    button: tuple[int, int, int] = (45, 67, 84)
    button_hover: tuple[int, int, int] = (60, 98, 121)
    text: tuple[int, int, int] = (224, 233, 240)
    primary: tuple[int, int, int] = (27, 121, 145)
    primary_hover: tuple[int, int, int] = (39, 150, 173)
    primary_active: tuple[int, int, int] = (19, 102, 129)
    export: tuple[int, int, int] = (48, 96, 160)
    export_hover: tuple[int, int, int] = (65, 120, 188)
    cancel: tuple[int, int, int] = (147, 61, 61)
    cancel_hover: tuple[int, int, int] = (177, 75, 75)
    success: tuple[int, int, int] = (90, 210, 120)
    warning: tuple[int, int, int] = (255, 175, 70)
    error: tuple[int, int, int] = (230, 95, 95)
    csv: tuple[int, int, int, int] = (80, 190, 255, 255)
    txt: tuple[int, int, int, int] = (120, 220, 150, 255)
    xlsx: tuple[int, int, int, int] = (100, 200, 120, 255)
    fit_marker: tuple[int, int, int, int] = (255, 255, 255, 255)
    fit_marker_outline: tuple[int, int, int, int] = (0, 0, 0, 255)
    decomposition_one: tuple[int, int, int, int] = SERIES['one'].rgba
    decomposition_two: tuple[int, int, int, int] = SERIES['two'].rgba
    decomposition_total: tuple[int, int, int, int] = SERIES['total'].rgba
    decomposition_observed: tuple[int, int, int, int] = SERIES['observed'].rgba
    decomposition_reference: tuple[int, int, int, int] = SERIES['reference'].rgba


@dataclass(frozen=True, slots=True)
class Spacing:
    window_padding: tuple[int, int] = (8, 8)
    frame_padding: tuple[int, int] = (6, 4)
    item_spacing: tuple[int, int] = (8, 5)
    cell_padding: tuple[int, int] = (4, 2)
    spacer_small: int = 4
    spacer_medium: int = 8
    fit_details_gutter: int = 55


@dataclass(frozen=True, slots=True)
class Typography:
    body_pt: int = 20
    plot_pt: int = 18
    section_pt: int = 23
    secondary_pt: int = 21
    navigation_pt: int = 22
    font_env: str = "PYECR_UI_FONT"


@dataclass(frozen=True, slots=True)
class CompactFormStyle:
    label_column: int = 128
    control_min: int = 112
    stepped_control_min: int = 160
    value_chars: int = 10
    content_gutter: int = 16
    label_max_chars: int = 38


@dataclass(frozen=True, slots=True)
class PlotStyle:
    legend_help_delay: float = .65
    padding: tuple[int, int] = (12, 12)
    label_padding: tuple[int, int] = (8, 8)
    legend_padding: tuple[int, int] = (8, 6)
    fit_padding: tuple[float, float] = (0.06, 0.08)
    map_width_to_height: float = 1.12
    map_colorbar_gap: int = 14
    map_horizontal_gutter: int = 24
    robust_quantile: float = 0.95
    robust_headroom: float = 1.25
    fit_main_fraction: float = .64
    fit_residual_fraction: float = .25
    plot_chrome: int = 110


@dataclass(frozen=True, slots=True)
class Widths:
    simulation_form: int = 340
    simulation_results: int = 600
    simulation_details: int = 900
    normalization_review: int = 1000
    simulation_exchange: int = 900
    toolbar_range: int = 100
    toolbar_resolution: int = 105
    toolbar_metric: int = 165
    input_micro: int = 130
    input_compact: int = 140
    input_narrow: int = 145
    input_small: int = 150
    input_medium_compact: int = 170
    input_medium: int = 180
    input_regular: int = 190
    input_wide: int = 220
    panel_field: int = 320
    field_wide: int = 340
    model_selector: int = 420
    progress: int = 480
    path: int = 760
    report_path: int = 820
    colorbar: int = 154
    unit_confirmation: int = 560


@dataclass(frozen=True, slots=True)
class Heights:
    simulation_panel: int = 500
    simulation_noise_review: int = 100
    simulation_details: int = 650
    normalization_review: int = 820
    simulation_exchange: int = 650
    simulation_send_review: int = 200
    simulation_results: int = 400
    simulation_history: int = 260
    transport_details: int = 220
    status_short: int = 65
    status_small: int = 75
    notes: int = 80
    batch_status: int = 110
    error_console: int = 130
    history: int = 150
    preview: int = 180
    residual: int = 240
    diagnostics: int = 240
    dct_plot: int = 240
    plot: int = 260
    fit_details: int = 280
    results: int = 300
    error_map: int = 340
    dialog: int = 520
    unit_confirmation: int = 190
    decomposition_early: int = 160
    decomposition_residual: int = 240
    decomposition_main_max: int = 320
    fit_main_min: int = 170
    fit_residual_min: int = 240
    residual_max: int = 280

    @property
    def colorbar(self) -> int:
        return self.error_console


@dataclass(frozen=True, slots=True)
class Wraps:
    compact_status: int = 280
    normalization_review: int = 900
    simulation_form: int = 900
    simulation_notice: int = 300
    simulation_help: int = 580
    simulation_exchange: int = 820
    rail: int = 320
    rail_alt: int = 330
    narrow: int = 340
    advanced: int = 350
    stage: int = 370
    map: int = 500
    unit_confirmation: int = 525
    project: int = 720
    batch: int = 900
    page: int = 1100
    analysis: int = 1300


@dataclass(frozen=True, slots=True)
class Viewport:
    width: int = 1366
    height: int = 768
    min_width: int = 1000
    min_height: int = 640
    outer_margin: int = 54
    min_usable: int = 970
    left_fraction: float = 0.30
    right_fraction: float = 0.27
    left_min: int = 330
    right_min: int = 320
    center_min: int = 310
    column_gap: int = 22
    left_initial: int = 390
    center_initial: int = 520
    right_initial: int = 350
    compact_breakpoint: int = 1600
    large_breakpoint: int = 2200
    compact_left: int = 350
    compact_right: int = 320
    medium_left: int = 370
    medium_right: int = 400
    large_left: int = 390
    large_right: int = 430
    plot_height_fraction: float = 0.62
    plot_height_min: int = 320


@dataclass(frozen=True, slots=True)
class TableStyle:
    header_row: bool = True
    borders_inner_h: bool = True
    borders_inner_v: bool = True
    borders_outer_h: bool = True
    borders_outer_v: bool = True
    resizable: bool = True


@dataclass(frozen=True, slots=True)
class AnalysisLayoutStyle:
    left_small: int = 360
    left_compact: int = 420
    left_medium: int = 480
    left_large: int = 500
    left_min: int = 350
    left_max: int = 700
    plot_min: int = 480
    plot_emergency_min: int = 320
    margin: int = 32
    gap: int = 12
    summary_width: int = 320
    summary_large: int = 340
    summary_height: int = 320
    parameter_window_width: int = 980
    parameter_window_height: int = 650
    form_gutter: int = 40
    form_breakpoint: int = 600
    parameter_breakpoint: int = 680
    hysteresis: int = 16
    slider_width: int = 170
    field_min: int = 100
    derived_height: int = 80


@dataclass(frozen=True, slots=True)
class SimulationLayoutStyle:
    left_compact: int = 340
    left_narrow: int = 320
    compact_breakpoint: int = 1100
    margin: int = 32
    gap: int = 12
    gutter: int = 28
    plot_min: int = 300
    history_input_height: int = 64
    parameter_name_width: int = 85
    unit_width: int = 80
    axis_width: int = 35
    active_width: int = 80
    run_fallback: int = 160


@dataclass(frozen=True, slots=True)
class DctLayoutStyle:
    diagnostic_height: int = 300
    review_height: int = 115
    control_wrap: int = 260
    context_wrap: int = 600
    context_budget_width: int = 110
    context_summary_height: int = 155
    context_plot_height: int = 240
    context_details_height: int = 330
    left_small: int = 320
    left_compact: int = 340
    left_medium: int = 360
    left_large: int = 380
    margin: int = 32
    gap: int = 12
    plot_gutter: int = 32
    spectrum_min: int = 300
    spectrum_max: int = 480
    spectrum_fraction: float = .45
    reconstruction_min: int = 300
    reconstruction_max: int = 480
    residual_min: int = 300
    residual_max: int = 850
    residual_chrome: int = 160
    aspect_max: float = 2.6
    panel_min: int = 320
    footer: int = 24


@dataclass(frozen=True, slots=True)
class ReportLayoutStyle:
    # U05: font-measured readable columns, natural cards and bounded real lists.
    content_share: float = .62
    content_readable_chars: int = 76
    destination_readable_chars: int = 50
    page_inset: int = 16
    column_gap: int = 12
    source_visible_rows: int = 5
    figure_visible_rows: int = 14
    source_height_fraction: float = .28
    figure_height_fraction: float = .62
    list_padding: int = 8
    label_chars: int = 72
    workflow_left: int = 680
    workflow_left_min: int = 440
    workflow_right_min: int = 300
    workflow_sources: int = 190
    workflow_source_min: int = 60
    workflow_source_row: int = 30
    workflow_figures: int = 220
    workflow_footer: int = 145
    workflow_top_gap: int = 12
    left_compact: int = 430
    left_medium: int = 500
    left_large: int = 540
    right_min: int = 420
    build_right: int = 550
    build_height: int = 500
    wrap_left: int = 380
    wrap_right: int = 480
    snapshot_height: int = 110
    preview_height: int = 66
    results_height: int = 260
    artifacts_height: int = 280
    plot_height: int = 300
    dialog_width: int = 850
    dialog_height: int = 480
    dialog_max_width: int = 1000
    dialog_min_height: int = 300
    panel_min: int = 320
    results_min: int = 100
    dialog_margin: int = 80
    dialog_footer: int = 120
    gutter: int = 48
    margin: int = 32
    gap: int = 12
    footer: int = 24


@dataclass(frozen=True, slots=True)
class MapControlsStyle:
    palette_width: int = 120
    numeric_width: int = 110
    range_value_width: int = 200
    custom_width: int = 115
    compact_width: int = 100
    limits_width: int = 180
    range_width: int = 200
    dpi_width: int = 85
    wrap: int = 700
    focus_wrap: int = 900
    focus_width: int = 1000
    focus_height: int = 700
    minimum_width: int = 700
    minimum_height: int = 500
    margin: int = 70
    chrome: int = 180
    plot_min: int = 300
    panel_reserve: int = 130
    horizontal_breakpoint: int = 1100

@dataclass(frozen=True, slots=True)
class FigureExportStyle:
    width: int = 860
    height: int = 760
    sources_height: int = 100
    list_height: int = 200
    picker_height: int = 460
    wrap: int = 760
    footer: int = 100
    margin: int = 20

@dataclass(frozen=True, slots=True)
class StepPanelStyle:
    # R02 revision tokens; scoped to Analysis, not the other five pages.
    page: tuple[int, int, int] = (16, 25, 35)
    body: tuple[int, int, int] = (32, 52, 73)
    header: tuple[int, int, int] = (41, 69, 94)
    normal_header: tuple[int, int, int] = (37, 58, 77)
    error_body: tuple[int, int, int] = (57, 39, 47)
    error_header: tuple[int, int, int] = (91, 48, 58)
    input: tuple[int, int, int] = (23, 42, 59)
    text: tuple[int, int, int] = (232, 238, 245)
    secondary: tuple[int, int, int] = (186, 200, 213)
    action: tuple[int, int, int] = (27, 113, 137)
    padding: int = 14
    details_height: int = 220
    preprocess_form_min: int = 80
    preprocess_form_max: int = 280
    preprocess_footer_reserve: int = 320
    preprocess_plot_width: int = 950
    preprocess_plot_height: int = 700
    preprocess_table_wrap_width: int = 330
    parameter_table_width: int = 840
    parameter_columns: tuple[int,...] = (124,128,128,128,64,100)
    parameter_table_height: int = 280
    parameter_table_min_height: int = 100
    parameter_caption_reserve: int = 40
    model_form_height: int = 310
    fit_controls_height: int = 260
    control_reserve: int = 240
    scrollbar_gutter: int = 18


@dataclass(frozen=True, slots=True)
class DesignTokens:
    colors: Colors = field(default_factory=Colors)
    spacing: Spacing = field(default_factory=Spacing)
    typography: Typography = field(default_factory=Typography)
    form: CompactFormStyle = field(default_factory=CompactFormStyle)
    widths: Widths = field(default_factory=Widths)
    heights: Heights = field(default_factory=Heights)
    wraps: Wraps = field(default_factory=Wraps)
    viewport: Viewport = field(default_factory=Viewport)
    table: TableStyle = field(default_factory=TableStyle)
    plot: PlotStyle = field(default_factory=PlotStyle)
    analysis: AnalysisLayoutStyle = field(default_factory=AnalysisLayoutStyle)
    dct: DctLayoutStyle = field(default_factory=DctLayoutStyle)
    simulation: SimulationLayoutStyle = field(default_factory=SimulationLayoutStyle)
    report: ReportLayoutStyle = field(default_factory=ReportLayoutStyle)
    map_controls: MapControlsStyle = field(default_factory=MapControlsStyle)
    figures: FigureExportStyle = field(default_factory=FigureExportStyle)
    steps: StepPanelStyle = field(default_factory=StepPanelStyle)


TOKENS = DesignTokens()
THEME_TAG = "pyecr_theme"
PRIMARY_THEME_TAG = "pyecr_primary_action_theme"
EXPORT_THEME_TAG = "pyecr_export_action_theme"
CANCEL_THEME_TAG = "pyecr_cancel_action_theme"
FIT_MARKER_THEME_TAG = "pyecr_error_map_fit_theme"
FONT_REGISTRY_TAG = "pyecr_font_registry"
FONT_TAG = "pyecr_ui_font"
PLOT_FONT_TAG = "pyecr_plot_font"
PLOT_THEME_TAG = "pyecr_scientific_plot_theme"
SECTION_FONT_TAG = 'pyecr_section_font'
SECONDARY_FONT_TAG = 'pyecr_secondary_font'
NAVIGATION_FONT_TAG = 'pyecr_navigation_font'


def status_color(status: str) -> tuple[int, int, int]:
    """One status palette for task labels, reliability, and current errors."""

    return {
        "completed": TOKENS.colors.success,
        "warning": TOKENS.colors.warning,
        "out_of_date": TOKENS.colors.warning,
        "cancelled": TOKENS.colors.warning,
        "error": TOKENS.colors.error,
    }.get(status, TOKENS.colors.text)


def build_fonts(dpg: Any) -> bool:
    """Build checked fonts once; invalid paths degrade with visible diagnostics."""
    from dataclasses import replace
    from pyecr.gui.fonts import choose_fonts, log, native_font_path
    global FONT_PLAN
    FONT_PLAN = choose_fonts(os.environ.get(TOKENS.typography.font_env, "").strip())
    FONT_PLAN = replace(FONT_PLAN, heading=FONT_PLAN.body)
    native_paths = {}
    for face in (FONT_PLAN.body, FONT_PLAN.scientific, FONT_PLAN.heading):
        if face is None or face.path in native_paths:
            continue
        try:
            native_paths[face.path] = native_font_path(face)
        except OSError as exc:
            native_paths[face.path] = None
            log.warning("Cannot prepare font %s: %s; using the built-in fallback.", face.path, exc)
    FONT_PLAN = replace(FONT_PLAN, **{
        role: face if face is not None and native_paths.get(face.path) else None
        for role in ("body", "scientific", "heading")
        for face in (getattr(FONT_PLAN, role),)
    })
    log.info("PyECR font loading\n%s", FONT_PLAN.describe())
    if FONT_PLAN.body is None:
        log.warning("PyECR CJK coverage unavailable; see Display > Font diagnostics")
    with dpg.font_registry(tag=FONT_REGISTRY_TAG):
        for face, tag, size in ((FONT_PLAN.body, FONT_TAG, TOKENS.typography.body_pt),
                                (FONT_PLAN.scientific, PLOT_FONT_TAG, TOKENS.typography.plot_pt),
                                (FONT_PLAN.heading or FONT_PLAN.body, SECTION_FONT_TAG, TOKENS.typography.section_pt),
                                (FONT_PLAN.body, SECONDARY_FONT_TAG, TOKENS.typography.secondary_pt),
                                (FONT_PLAN.body, NAVIGATION_FONT_TAG, TOKENS.typography.navigation_pt)):
            if face is not None:
                with dpg.font(native_paths[face.path], size, tag=tag):
                    # Load actual covered BMP glyphs: Latin, Greek, CJK, math,
                    # super/subscripts. Do not allocate nonexistent codepoints.
                    # DPG 2.3 uses automatic glyph ranges (old range APIs are
                    # deprecated no-ops); keep explicit ranges for older DPG.
                    if tuple(int(x) for x in version("dearpygui").split('.')[:2]) < (2, 3):
                        dpg.add_font_chars(sorted(c for c in face.codepoints if 32 <= c <= 0xFFFF))
    if FONT_PLAN.body is not None:
        dpg.bind_font(FONT_TAG)
    return FONT_PLAN.body is not None


def build_themes(dpg: Any) -> None:
    """Create stable global/action themes once per DearPyGui context."""

    color = TOKENS.colors
    spacing = TOKENS.spacing
    step = TOKENS.steps
    with dpg.theme(tag='pyecr_step_rail_theme'):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, step.page)
    with dpg.theme(tag='pyecr_preprocess_selected_theme'):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, step.header)
            dpg.add_theme_color(dpg.mvThemeCol_Border, step.secondary)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
    for state in ('normal', 'current', 'error'):
        with dpg.theme(tag='pyecr_step_theme_'+state):
            with dpg.theme_component(dpg.mvAll):
                for slot, value in (
                    (dpg.mvThemeCol_ChildBg, step.error_body if state == 'error' else step.body),
                    (dpg.mvThemeCol_WindowBg, step.error_body if state == 'error' else step.body),
                    (dpg.mvThemeCol_Header, step.error_header if state == 'error' else step.header if state == 'current' else step.normal_header),
                    (dpg.mvThemeCol_HeaderHovered, step.header),
                    (dpg.mvThemeCol_HeaderActive, step.header),
                    (dpg.mvThemeCol_FrameBg, step.input),
                    (dpg.mvThemeCol_Text, step.text),
                    (dpg.mvThemeCol_TextDisabled, step.secondary),
                    (dpg.mvThemeCol_Button, step.action),
                ):
                    dpg.add_theme_color(slot, value)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, step.padding, step.padding)
    with dpg.theme(tag=THEME_TAG):
        with dpg.theme_component(dpg.mvAll):
            for slot, value in (
                (dpg.mvThemeCol_WindowBg, color.window),
                (dpg.mvThemeCol_ChildBg, color.child),
                (dpg.mvThemeCol_FrameBg, color.frame),
                (dpg.mvThemeCol_Header, color.header),
                (dpg.mvThemeCol_Tab, color.frame),
                (dpg.mvThemeCol_TabActive, color.tab_active),
                (dpg.mvThemeCol_Button, color.button),
                (dpg.mvThemeCol_ButtonHovered, color.button_hover),
                (dpg.mvThemeCol_Text, color.text),
            ):
                dpg.add_theme_color(slot, value)
            for slot, value in (
                (dpg.mvStyleVar_WindowPadding, spacing.window_padding),
                (dpg.mvStyleVar_FramePadding, spacing.frame_padding),
                (dpg.mvStyleVar_ItemSpacing, spacing.item_spacing),
                (dpg.mvStyleVar_CellPadding, spacing.cell_padding),
            ):
                dpg.add_theme_style(slot, *value)
    with dpg.theme(tag=PRIMARY_THEME_TAG):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, color.primary)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, color.primary_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color.primary_active)
    with dpg.theme(tag=FIT_MARKER_THEME_TAG):
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color.fit_marker,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color.fit_marker_outline,
                                category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 7.0,
                                category=dpg.mvThemeCat_Plots)
    for tag,base,hover in ((EXPORT_THEME_TAG,color.export,color.export_hover),(CANCEL_THEME_TAG,color.cancel,color.cancel_hover)):
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,base)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,hover)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,base)
    with dpg.theme(tag=PLOT_THEME_TAG):
        with dpg.theme_component(dpg.mvPlot):
            for slot, value in (
                (dpg.mvPlotStyleVar_PlotPadding, TOKENS.plot.padding),
                (dpg.mvPlotStyleVar_LabelPadding, TOKENS.plot.label_padding),
                (dpg.mvPlotStyleVar_LegendPadding, TOKENS.plot.legend_padding),
                (dpg.mvPlotStyleVar_FitPadding, TOKENS.plot.fit_padding),
            ):
                dpg.add_theme_style(slot, *value, category=dpg.mvThemeCat_Plots)


def bind_primary_actions(dpg: Any) -> None:
    for tag in (
        "pyecr_run_fit",
        "pyecr_batch_run_independent",
        "pyecr_batch_run_global",
        "pyecr_run_dct",
        "pyecr_run_diagnostics",
        "pyecr_error_map_generate",
        "pyecr_dual_map_generate",
    ):
        dpg.bind_item_theme(tag, PRIMARY_THEME_TAG)
    for tag in ("pyecr_report_export_current","pyecr_report_export_captured","pyecr_analysis_export_report","pyecr_analysis_capture_snapshot",
                "pyecr_error_map_export","pyecr_dual_map_export"):
        if dpg.does_item_exist(tag):dpg.bind_item_theme(tag,EXPORT_THEME_TAG)
    for tag in ("pyecr_cancel_fit","pyecr_cancel_diagnostics","pyecr_batch_cancel","pyecr_cancel_dct","pyecr_report_cancel_export","pyecr_export_cancel",
                "pyecr_error_map_cancel","pyecr_dual_map_cancel"):
        if dpg.does_item_exist(tag):dpg.bind_item_theme(tag,CANCEL_THEME_TAG)
