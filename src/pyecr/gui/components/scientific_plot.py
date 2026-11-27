"""Shared DearPyGui scientific-plot presentation; no forward calculations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from pyecr.gui.theme import PLOT_FONT_TAG, PLOT_THEME_TAG, TOKENS
from pyecr.plotting.visual_style import SERIES


PLOT_TAGS = (
    "pyecr_data_raw_plot", "pyecr_data_processed_plot", "pyecr_fit_plot",
    "pyecr_residual_plot", "pyecr_components_plot", "pyecr_boundary_plot",
    "pyecr_explorer_plot", "pyecr_error_map_plot", "pyecr_diagnostics_residual_plot",
    "pyecr_dual_map_h_plot", "pyecr_dual_map_o_plot",
)
SCIENTIFIC_TICK_AXES = (
    "pyecr_data_raw_x_axis", "pyecr_data_raw_y_axis",
    "pyecr_data_processed_x_axis", "pyecr_data_processed_y_axis",
    "pyecr_fit_x_axis", "pyecr_fit_y_axis",
    "pyecr_residual_x_axis", "pyecr_residual_y_axis",
    "pyecr_components_x_axis", "pyecr_components_y_axis",
    "pyecr_boundary_x_axis", "pyecr_boundary_y_axis",
    "pyecr_explorer_x_axis", "pyecr_explorer_y_axis",
    "pyecr_diagnostics_residual_x_axis", "pyecr_diagnostics_residual_y_axis",
)


def bind_scientific_plot_styles(dpg: Any) -> None:
    """Bind one plot theme and the same tick/title/legend font on built plots."""

    tags = list(PLOT_TAGS)
    if hasattr(dpg, "get_all_items"):
        tags = [item for item in dpg.get_all_items()
                if dpg.get_item_info(item)["type"].split("::")[-1] in ("mvPlot", "mvColorScale")]
    for tag in tags:
        if not dpg.does_item_exist(tag):
            continue
        if "mvColorScale" not in (dpg.get_item_info(tag)["type"] if hasattr(dpg, "get_item_info") else ""):
            dpg.bind_item_theme(tag, PLOT_THEME_TAG)
        if dpg.does_item_exist(PLOT_FONT_TAG):
            dpg.bind_item_font(tag, PLOT_FONT_TAG)
    for tag in SCIENTIFIC_TICK_AXES:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, tick_format="%.2e")
    roles={
        'observed':('pyecr_fit_data_series','pyecr_decomposition_observed','pyecr_decomposition_early_observed'),
        'total':('pyecr_fit_curve_series','pyecr_decomposition_total','pyecr_decomposition_early_total'),
        'one':('pyecr_fit_h_series','pyecr_decomposition_early_one'),
        'two':('pyecr_fit_o_series','pyecr_decomposition_early_two'),
        'residual':('pyecr_residual_series','pyecr_fit_residual_mirror_series','pyecr_decomposition_residual_values','pyecr_diagnostics_residual_series'),
        'reference':('pyecr_decomposition_reference','pyecr_decomposition_early_reference'),
        'zero':('pyecr_residual_zero_series','pyecr_fit_residual_mirror_zero','pyecr_decomposition_residual_reference'),
    }
    for role,series_tags in roles.items():
        style=SERIES[role]
        with dpg.theme() as theme:
            for kind in (dpg.mvLineSeries,dpg.mvScatterSeries):
                with dpg.theme_component(kind):
                    for slot in (dpg.mvPlotCol_Line,dpg.mvPlotCol_MarkerFill,dpg.mvPlotCol_MarkerOutline):
                        dpg.add_theme_color(slot,style.rgba,category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight,style.width,category=dpg.mvThemeCat_Plots)
        for series_tag in series_tags:
            if dpg.does_item_exist(series_tag):dpg.bind_item_theme(series_tag,theme)


def signal_axis_label(metadata: Mapping[str, object], *, raw: bool = False) -> str:
    """Use declared dataset units; never infer a physical unit from magnitudes."""

    from pyecr.plotting.labels import signal_axis_label as title
    return title(metadata, raw=raw)


def signal_display_unit(metadata: Mapping[str, object], *, raw: bool = False) -> str:
    if str(metadata.get("raw_signal_type" if raw else "signal_type", "")).lower() == "normalized":
        return "dimensionless"
    return str(metadata.get("raw_signal_unit" if raw else "signal_unit", "arbitrary")).strip() or "arbitrary"


def time_axis_label(metadata: Mapping[str, object], *, raw: bool = False) -> str:
    unit = str(metadata.get("raw_time_unit" if raw else "time_unit", "s")).strip()
    return f"Time ({unit or 's'})"


def residual_axis_label(metadata: Mapping[str, object]) -> str:
    if metadata.get('signal_type') == 'normalized':
        return 'Observed − fitted'
    return f"Observed − fitted ({signal_display_unit(metadata)})"


def robust_residual_limits(residuals: Sequence[float]) -> tuple[float, float] | None:
    """Symmetric display limits; the underlying residual series is untouched."""

    values = np.asarray(residuals, dtype=float)
    if values.size == 0:
        return None
    if not np.all(np.isfinite(values)):
        raise ValueError("residual display requires finite values")
    span = max(
        float(np.quantile(np.abs(values), TOKENS.plot.robust_quantile))
        * TOKENS.plot.robust_headroom,
        float(np.max(np.abs(values))) * 1.0e-6,
        1.0e-12,
    )
    return (-span, span)


def clear_plot_series(dpg: Any, tags: Sequence[str], *, hide: bool = True) -> None:
    for tag in tags:
        dpg.set_value(tag, [[], []])
        if hide:
            dpg.configure_item(tag, show=False)
