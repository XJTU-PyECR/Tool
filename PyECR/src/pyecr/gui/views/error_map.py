"""M1 Error Map controls and visualization; never evaluates an objective."""

from __future__ import annotations
from pyecr.gui.theme import FIT_MARKER_THEME_TAG, TOKENS
from pyecr.gui.components.action_state import (
    add_action_tooltip, set_action_availability,
)

import traceback
from math import isfinite, log10
from typing import Any

import numpy as np

from pyecr.diagnostics.error_map import ErrorMapConfig, ErrorMapResolution
from pyecr.gui.state import GuiPage
from pyecr.plotting.error_map import (
    ErrorMapClipMode, ErrorMapMetric, error_map_plot_data, log_axis_edges,
)
from dataclasses import replace
from pyecr.plotting.map_display import color_limits, display_raster, padded_limits, grid_points


class ErrorMapViewMixin:
    def _build_error_map_area(self) -> None:
        dpg = self._require_dpg()
        self._build_map_current_fit("single")
        self._register_owned_root("pyecr_error_map_calculation")
        with dpg.collapsing_header(label="2 · Range & grid",tag="pyecr_error_map_calculation",default_open=True):
            with dpg.group(horizontal=True,tag='pyecr_error_map_calculation_columns'):
                with dpg.child_window(tag='pyecr_error_map_range_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    self._build_map_range_controls('single')
                with dpg.child_window(tag='pyecr_error_map_grid_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    self._build_map_resolution_controls("single")
            dpg.move_item('pyecr_error_map_calculation_advanced',parent='pyecr_error_map_calculation')
            with dpg.group(horizontal=True,show=False):
                dpg.add_button(label="Center on fit", tag="pyecr_error_map_center_fit",callback=self._center_error_map_on_fit)
                dpg.add_button(label="Expand range", tag="pyecr_error_map_expand_range",callback=self._expand_error_map_range)
                dpg.add_button(label="Reset auto range", tag="pyecr_error_map_reset_auto",callback=self._reset_error_map_auto_range)
            with dpg.group(horizontal=True, tag="pyecr_error_map_manual_ranges", show=False):
                dpg.add_input_text(label="log10 D (cm²/s) min,max", tag="pyecr_error_map_d_range",hint="e.g. -8,-5",
                                   width=TOKENS.widths.input_regular,callback=self._map_calculation_changed,user_data="single")
                dpg.add_input_text(label="log10 k (cm/s) min,max", tag="pyecr_error_map_k_range",hint="e.g. -6,-3",
                                   width=TOKENS.widths.input_regular,callback=self._map_calculation_changed,user_data="single")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Recalculate map", tag="pyecr_error_map_generate",callback=self._generate_error_map)
            add_action_tooltip(dpg, "pyecr_error_map_generate")
            dpg.add_button(label="Cancel calculation", tag="pyecr_error_map_cancel",callback=self._cancel_error_map,enabled=False)
        dpg.add_text("Choose a range. Apply view uses stored cells; Recalculate map evaluates a new grid. No refit.",tag="pyecr_error_map_status",wrap=TOKENS.wraps.map)
        dpg.add_text("",tag="pyecr_error_map_captured_grid",wrap=TOKENS.map_controls.wrap)
        self._register_owned_root("pyecr_error_map_display_controls")
        with dpg.collapsing_header(label="3 · Display",tag="pyecr_error_map_display_controls",default_open=False):
            dpg.add_combo(tuple(item.value for item in ErrorMapMetric),
                          default_value=ErrorMapMetric.LOG_RATIO.value,
                          width=TOKENS.widths.toolbar_metric, label="Metric", tag="pyecr_error_map_metric",
                          callback=self._error_map_metric_changed)
            dpg.add_combo(tuple(item.value for item in ErrorMapClipMode),
                          default_value=ErrorMapClipMode.FULL.value,
                          width=TOKENS.widths.toolbar_range, label="Contrast", tag="pyecr_error_map_clip",
                          callback=self._error_map_metric_changed)
            self._build_map_view_controls("single")
            with dpg.group(horizontal=True,parent='pyecr_error_map_metric_panel'):
                dpg.add_button(label="Expand view",callback=self._focus_map,user_data="single")
                dpg.add_checkbox(label="Grid min", tag="pyecr_error_map_show_grid_min",
                                 default_value=True, callback=self._error_map_metric_changed)
        with dpg.collapsing_header(label="4 Export captured map",default_open=False):
            dpg.add_input_text(label="New report directory", tag="pyecr_error_map_export_path",
                               hint="New nonexistent path; verified ReportBundle, no recalculation", width=-1)
            dpg.add_button(label="Export captured Error Map",tag="pyecr_error_map_export",callback=self._export_error_map,enabled=False)
            add_action_tooltip(dpg, "pyecr_error_map_export")
        dpg.add_text("", tag="pyecr_error_map_quality_banner", color=TOKENS.colors.warning,
                     wrap=TOKENS.wraps.map, show=False)
        dpg.add_text("Display: Full; stored objective grid is unchanged.",
                     tag="pyecr_error_map_display_info", wrap=TOKENS.wraps.map)
        dpg.add_text("", tag="pyecr_error_map_hover", wrap=TOKENS.wraps.map)
        with dpg.group(horizontal=True, tag="pyecr_error_map_visual_row"):
            dpg.add_spacer(tag="pyecr_error_map_left_spacer", width=TOKENS.plot.map_horizontal_gutter)
            with dpg.plot(label="D–k objective surface", tag="pyecr_error_map_plot",
                          height=TOKENS.heights.error_map, width=TOKENS.heights.error_map, no_mouse_pos=True):
                dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                # Cells are linear in log10 coordinates; ticks show physical values.
                dpg.add_plot_axis(dpg.mvXAxis, label="D (cm²/s; log10 axis)",
                                  tag="pyecr_error_map_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="k (cm/s; log10 axis)",
                                  tag="pyecr_error_map_y_axis")
                dpg.add_scatter_series([], [], label="Continuous fit (white)",
                                       parent="pyecr_error_map_y_axis", tag="pyecr_error_map_fit_point")
                dpg.add_line_series([], [], label="Grid minimum (cross)",
                                    parent="pyecr_error_map_y_axis", tag="pyecr_error_map_grid_cross_h",
                                    show=False)
                dpg.add_line_series([], [], label="##grid minimum vertical",
                                    parent="pyecr_error_map_y_axis", tag="pyecr_error_map_grid_cross_v",
                                    show=False)
            dpg.add_colormap_scale(label=ErrorMapMetric.LOG_RATIO.value,
                                   tag="pyecr_error_map_colorbar", height=TOKENS.heights.error_map,
                                   width=TOKENS.widths.colorbar,
                                   colormap=dpg.mvPlotColormap_Viridis, format="%.2e", show=False)
        dpg.bind_item_theme("pyecr_error_map_fit_point", FIT_MARKER_THEME_TAG)
        for tag in ("pyecr_error_map_grid_cross_h","pyecr_error_map_grid_cross_v"):
            dpg.bind_item_theme(tag,"pyecr_map_grid_min_theme")
        dpg.bind_colormap("pyecr_error_map_plot", dpg.mvPlotColormap_Viridis)
        dpg.add_text("Relative SSE colors are objective ratios, not confidence intervals.",
                     wrap=TOKENS.wraps.map,parent='pyecr_error_map_calculation_advanced')

    def _error_map_range_changed(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        dpg.configure_item("pyecr_error_map_manual_ranges",
                           show=dpg.get_value("pyecr_error_map_range_mode") == "MANUAL")
        self._refresh_map_instructions("single")

    def _fit_log_coordinates(self) -> tuple[float, float]:
        fit = self.controller.analysis_session.fit_result
        if fit is None or not self.controller.workflow.can_error_map:
            raise RuntimeError("range tools require a current M1 fit")
        return log10(float(fit.params["D_cm2_s"])), log10(float(fit.params["k_cm_s"]))

    def _axis_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        context = self.controller.workflow.prepare_error_map()
        return self.controller.error_map_service.kinetic_log_bounds(context)

    def _display_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        dpg = self._require_dpg()
        if dpg.get_value("pyecr_error_map_range_mode") == "FULL BOUNDS":
            return self._axis_bounds()
        if dpg.get_value("pyecr_error_map_range_mode") == "MANUAL":
            return self._error_map_range("pyecr_error_map_d_range"), self._error_map_range("pyecr_error_map_k_range")
        result = self.controller.current_error_map_result
        if result is not None:
            return ((float(result.log10_D[0]), float(result.log10_D[-1])),
                    (float(result.log10_k[0]), float(result.log10_k[-1])))
        centers = self._fit_log_coordinates()
        bounds = self._axis_bounds()
        half = dpg.get_value("pyecr_error_map_half_width") or ErrorMapConfig().auto_half_width_decades
        return tuple(
            _clipped_log_range(center, half, bound)
            for center, bound in zip(centers, bounds, strict=True)
        )  # type: ignore[return-value]

    def _show_manual_ranges(self, ranges: tuple[tuple[float, float], tuple[float, float]]) -> None:
        dpg = self._require_dpg()
        for tag, values in zip(("pyecr_error_map_d_range", "pyecr_error_map_k_range"), ranges, strict=True):
            dpg.set_value(tag, f"{values[0]:.8g},{values[1]:.8g}")
        dpg.set_value("pyecr_error_map_range_mode", "MANUAL")
        dpg.configure_item("pyecr_error_map_manual_ranges", show=True)
        self._error_map_range_note = "Range controls changed; Generate to evaluate a new map."
        self._refresh_error_map_widgets()

    def _center_error_map_on_fit(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            ranges = self._display_ranges()
            centers = self._fit_log_coordinates()
            bounds = self._axis_bounds()
            self._show_manual_ranges(tuple(
                _clipped_log_range(center, (span[1] - span[0]) / 2, bound)
                for center, span, bound in zip(centers, ranges, bounds, strict=True)
            ))
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.DIAGNOSTICS, exc, traceback.format_exc())

    def _expand_error_map_range(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            ranges = self._display_ranges()
            bounds = self._axis_bounds()
            self._show_manual_ranges(tuple(
                _clipped_log_range((span[0] + span[1]) / 2, (span[1] - span[0]) * 0.75, bound)
                for span, bound in zip(ranges, bounds, strict=True)
            ))
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.DIAGNOSTICS, exc, traceback.format_exc())

    def _reset_error_map_auto_range(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        dpg = self._require_dpg()
        dpg.set_value("pyecr_error_map_range_mode", "AUTO")
        for tag in ("pyecr_error_map_d_range", "pyecr_error_map_k_range"):
            dpg.set_value(tag, "")
        dpg.configure_item("pyecr_error_map_manual_ranges", show=False)
        self._error_map_range_note = "AUTO range selected; Generate to evaluate a new map."
        self._refresh_error_map_widgets()

    def _error_map_metric_changed(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._render_error_map_plot(force=True)

    def _generate_error_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        if self._has_range_controls('single'):
            self._recalculate_map_ranges('single')
            return
        dpg = self._require_dpg()
        try:
            label=str(dpg.get_value("pyecr_error_map_resolution"))
            n=grid_points(label,dpg.get_value("pyecr_error_map_points"))
            resolution=ErrorMapResolution[label.upper()] if label.upper() in ErrorMapResolution.__members__ else ErrorMapResolution.FINE
            mode=dpg.get_value("pyecr_error_map_range_mode")
            ranges=self._display_ranges() if mode!="AUTO" else (None,None)
            settings = ErrorMapConfig(
                resolution=resolution,n_points=n,
                log10_D_range=ranges[0],log10_k_range=ranges[1],
                auto_half_width_decades=dpg.get_value("pyecr_error_map_half_width") or 1.5,
            )
            self._map_view_changed(user_data="single")
            self.controller.run_error_map(settings)
            self._remember_map_attempt("single")
            self._error_map_range_note = ""
        except Exception as exc:
            self._error_map_range_note = f"Generate rejected: {exc}"
            self.controller.record_callback_exception(GuiPage.DIAGNOSTICS, exc, traceback.format_exc())
        self._refresh_widgets()

    def _error_map_range(self, tag: str) -> tuple[float, float]:
        raw = str(self._require_dpg().get_value(tag)).split(",")
        if len(raw) != 2:
            raise ValueError(f"{tag} requires two comma-separated log10 limits")
        return float(raw[0]), float(raw[1])

    def _cancel_error_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.controller.cancel_page(GuiPage.DIAGNOSTICS)
        self._refresh_widgets()

    def _export_error_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            if self.controller.current_error_map_result is None:
                raise RuntimeError("cannot export an Out of date or missing Error Map")
            destination = str(self._require_dpg().get_value("pyecr_error_map_export_path")).strip()
            if not destination:
                raise ValueError("enter a new report directory for Error Map export")
            snapshot = self.controller.capture_snapshot()
            if snapshot.error_map_result is not self.controller.current_error_map_result:
                raise RuntimeError("current Error Map was not captured; export refused")
            self.controller.export_report_bundle(destination, snapshot=snapshot)
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.REPORT, exc, traceback.format_exc())
        self._refresh_widgets()

    def _refresh_error_map_widgets(self) -> None:
        dpg = self._require_dpg()
        self._refresh_map_instructions("single")
        can_run = self.controller.workflow.can_error_map
        updating = self.controller.is_error_map_updating
        result = self.controller.current_error_map_result
        can_generate = (can_run and not updating
                        and self.controller.state.tasks[GuiPage.DIAGNOSTICS].status != "running")
        dpg.configure_item('pyecr_error_map_visual_row', show=self.controller.analysis_session.error_map_result is not None)
        set_action_availability(
            dpg, "pyecr_error_map_generate", enabled=can_generate,
            reason="Requires a current 03 M2_SINGLE_DK fit and no running Diagnostics task.",
        )
        dpg.configure_item("pyecr_error_map_cancel", enabled=updating, show=updating)
        set_action_availability(
            dpg, "pyecr_error_map_export", enabled=result is not None and can_run,
            reason="Generate a current Error Map; Out of date maps cannot be exported.",
        )
        for tag in ("pyecr_error_map_center_fit", "pyecr_error_map_expand_range", "pyecr_error_map_reset_auto"):
            dpg.configure_item(tag, enabled=can_run and not updating)
        if self.controller.analysis_session.error_map_stale:
            status = "Out of date — regenerate Error Map before viewing or export."
        elif not can_run:
            status = "Unavailable — requires a current 03 M2_SINGLE_DK fit; Out of date map export is disabled."
        elif updating and result is not None:
            status = "Updating — previous valid map remains visible until the new grid completes."
        elif updating:
            status = "Updating — grid is running in a worker."
        elif result is None:
            status = "Ready · Recalculate to evaluate this range."
        else:
            status = f"Current map: {result.diagnostics.n_valid}/{result.diagnostics.n_total} valid cells."
        note = getattr(self, "_error_map_range_note", "")
        dpg.set_value("pyecr_error_map_status", f"{status} {note} {self._map_attempt_summary('single')}".strip())
        warnings = () if result is None else (
            result.quality.warnings if result.quality is not None else result.diagnostics.warnings
        )
        show_quality = bool(result is not None and (
            warnings or (result.quality is not None and result.quality.status != "pass")
        ))
        dpg.configure_item("pyecr_error_map_quality_banner", show=show_quality,
                           color=TOKENS.colors.error if result is not None and result.quality is not None and result.quality.status == "fail" else TOKENS.colors.warning)
        dpg.set_value("pyecr_error_map_quality_banner",
                      "" if not show_quality else "Map quality: " + "; ".join(warnings or (result.quality.status,)))
        self._render_error_map_plot()

    def _render_error_map_plot(self, *, force: bool = False) -> None:
        dpg = self._require_dpg()
        result = self.controller.current_error_map_result or self.controller.analysis_session.error_map_result
        if result is None:
            self._clear_error_map_plot()
            return
        options=self._map_options_for_render("single")
        metric = ErrorMapMetric(str(dpg.get_value("pyecr_error_map_metric")))
        clip = ErrorMapClipMode(str(dpg.get_value("pyecr_error_map_clip") or ErrorMapClipMode.FULL.value))
        options=replace(options,contrast=clip.value,metric=metric.value)
        if (force or result is not getattr(self, "_rendered_error_map_result", None)
                or metric is not getattr(self, "_rendered_error_map_metric", None)
                or clip is not getattr(self, "_rendered_error_map_clip", None)
                or options.raster_options() != getattr(self,"_rendered_error_map_options",None)):
            data = error_map_plot_data(result, metric, clip=clip)
            finite = data.display_values[data.display_valid_mask]
            if finite.size == 0:
                # A zero minimum SSE can make every ratio but the minimum
                # non-finite; raw SSE remains available for inspection.
                dpg.set_value("pyecr_error_map_display_info",
                              f"Display {clip.value}: no finite {metric.value} cells; try SSE. "
                              "Stored objective grid is unchanged.")
                self._clear_error_map_plot(clear_info=False)
                return
            low,high=color_limits(data.values,data.display_valid_mask,options)
            tag = "pyecr_error_map_heat_series"
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            # ImPlot heatmap uses row-major y/x; source grid is D/k.
            display = display_raster(data.values,data.display_valid_mask,options,color_range=(low,high))
            d_edges = log_axis_edges(result.log10_D)
            k_edges = log_axis_edges(result.log10_k)
            dpg.add_heat_series(display.T.ravel().tolist(), display.shape[1], display.shape[0],
                                parent="pyecr_error_map_y_axis", tag=tag,
                                label=data.label, format="", scale_min=low, scale_max=high,
                                bounds_min=(float(d_edges[0]), float(k_edges[0])),
                                bounds_max=(float(d_edges[-1]), float(k_edges[-1])))
            # Heat is the background; keep the white fitted optimum and cross
            # above it even after repeated dynamic-series replacement.
            dpg.move_item(tag, parent="pyecr_error_map_y_axis",
                          before="pyecr_error_map_fit_point")
            dpg.set_value("pyecr_error_map_fit_point", [[result.fitted_point.log10_D], [result.fitted_point.log10_k]])
            dpg.configure_item("pyecr_error_map_fit_point", show=True)
            if data.grid_minimum is not None:
                minimum = result.grid_minimum
                assert minimum is not None
                d_half = 0.012 * (result.log10_D[-1] - result.log10_D[0])
                k_half = 0.012 * (result.log10_k[-1] - result.log10_k[0])
                dpg.set_value("pyecr_error_map_grid_cross_h", [[minimum.log10_D - d_half, minimum.log10_D + d_half], [minimum.log10_k, minimum.log10_k]])
                dpg.set_value("pyecr_error_map_grid_cross_v", [[minimum.log10_D, minimum.log10_D], [minimum.log10_k - k_half, minimum.log10_k + k_half]])
            show_cross = data.grid_minimum is not None and bool(dpg.get_value("pyecr_error_map_show_grid_min"))
            for cross in ("pyecr_error_map_grid_cross_h", "pyecr_error_map_grid_cross_v"):
                dpg.configure_item(cross, show=show_cross)
            dpg.configure_item("pyecr_error_map_colorbar", show=True, label=data.label,
                               min_scale=low, max_scale=high)
            self._bind_map_colors("single",options)
            self._rendered_error_map_result = result
            self._rendered_error_map_metric = metric
            self._rendered_error_map_clip = clip
            self._rendered_error_map_options = options.raster_options()
            metadata = data.clipping
            self._rendered_error_map_clipping = metadata
            dpg.set_value(
                "pyecr_error_map_display_info",
                f"Display {metadata.mode.value}: {metadata.clipped_cells} clipped cells; "
                f"{metadata.nonfinite_metric_cells} nonfinite metric cells; "
                f"raw range [{metadata.raw_min:.3e}, {metadata.raw_max:.3e}], "
                f"shown range [{metadata.display_min:.3e}, {metadata.display_max:.3e}]. "
                f"Evaluated {len(result.log10_D)} × {len(result.log10_k)}; "
                f"{options.sampling} view {display.shape[0]} × {display.shape[1]}, palette {options.palette}. "
                f"Color limits [{low:.3e}, {high:.3e}]. Stored objective grid is unchanged.",
            )
        self._render_map_axis_range('single','single',result,options,(log_axis_edges(result.log10_D),log_axis_edges(result.log10_k)))
        if dpg.is_item_hovered("pyecr_error_map_plot"):
            logD, logk = dpg.get_plot_mouse_pos()
            if (isfinite(logD) and isfinite(logk)
                    and result.log10_D[0] <= logD <= result.log10_D[-1]
                    and result.log10_k[0] <= logk <= result.log10_k[-1]):
                data = error_map_plot_data(result, metric, clip=clip)
                cell = data.nearest_cell(10.0 ** logD, 10.0 ** logk)
                if cell is not None:
                    i, j, value = cell
                    dpg.set_value("pyecr_error_map_hover", f"D={data.D_cm2_s[i]:.3e} cm²/s | k={data.k_cm_s[j]:.3e} cm/s | {data.label}={value:.3e}")
                    return
        dpg.set_value("pyecr_error_map_hover", "Hover: no valid computed grid cell selected.")

    def _clear_error_map_plot(self, *, clear_info: bool = True) -> None:
        dpg = self._require_dpg()
        if dpg.does_item_exist("pyecr_error_map_heat_series"):
            dpg.delete_item("pyecr_error_map_heat_series")
        for tag in ("pyecr_error_map_fit_point", "pyecr_error_map_grid_cross_h", "pyecr_error_map_grid_cross_v"):
            dpg.set_value(tag, [[], []])
            dpg.configure_item(tag, show=False)
        dpg.configure_item("pyecr_error_map_colorbar", show=False)
        dpg.set_value("pyecr_error_map_hover", "No current computed grid.")
        if clear_info:
            dpg.set_value("pyecr_error_map_display_info", "No current display clipping metadata.")
        self._rendered_error_map_result = None
        self._rendered_error_map_metric = None
        self._rendered_error_map_clip = None
        self._rendered_error_map_clipping = None


def _decade_ticks(axis: np.ndarray) -> tuple[tuple[str, float], ...]:
    lo, hi = float(axis[0]), float(axis[-1])
    integers = range(int(np.ceil(lo)), int(np.floor(hi)) + 1)
    ticks = tuple((f"1e{power}", float(power)) for power in integers)
    if ticks:
        return ticks
    return ((f"{10.0 ** lo:.2e}", lo), (f"{10.0 ** hi:.2e}", hi))


def _clipped_log_range(center: float, half_width: float, bounds: tuple[float, float]) -> tuple[float, float]:
    if not all(isfinite(value) for value in (center, half_width, *bounds)) or half_width <= 0:
        raise ValueError("Error Map display range must be finite and positive")
    lower, upper = max(bounds[0], center - half_width), min(bounds[1], center + half_width)
    if lower >= upper:
        raise ValueError("suggested Error Map range is outside validated bounds")
    return lower, upper
