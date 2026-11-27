"""DearPyGui M3 H/O maps: presentation of P45 service results only."""

from __future__ import annotations

import traceback
from math import isfinite,log10
from typing import Any

from pyecr.diagnostics.dual_error_map import DualErrorMapResult, DualErrorMapConfig, DualMapRangeMode
from pyecr.diagnostics.error_map import ErrorMapResolution
from pyecr.plotting.map_display import color_limits, display_raster, padded_limits, grid_points
from pyecr.gui.components.action_state import add_action_tooltip, set_action_availability
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import FIT_MARKER_THEME_TAG, TOKENS
from pyecr.gui.views.error_map import _decade_ticks
from pyecr.plotting.dual_error_map import (
    DualMapMetric, DualMapPlotData, dual_map_plot_data,
)
from pyecr.plotting.error_map import log_axis_edges


def _tag(process: str, name: str) -> str:
    return f"pyecr_dual_map_{process.lower()}_{name}"


class DualErrorMapViewMixin:
    def _build_dual_error_map_area(self) -> None:
        dpg = self._require_dpg()
        self._build_map_current_fit("dual")
        self._register_owned_root("pyecr_dual_map_calculation")
        with dpg.collapsing_header(label="2 · Range & grid",tag="pyecr_dual_map_calculation",default_open=True):
            with dpg.group(horizontal=True,tag='pyecr_dual_map_calculation_columns'):
                with dpg.child_window(tag='pyecr_dual_map_range_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    self._build_map_range_controls('dual')
                with dpg.child_window(tag='pyecr_dual_map_grid_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    self._build_map_resolution_controls("dual")
            dpg.move_item('pyecr_dual_map_calculation_advanced',parent='pyecr_dual_map_calculation')
            with dpg.group(tag="pyecr_dual_map_manual_ranges",show=False):
                for process in ("h","o"):
                    with dpg.group(horizontal=True):
                        for name in ("d","k"):
                            dpg.add_input_text(label=f"{process.upper()} log10({'D (cm²/s)' if name=='d' else 'k (cm/s)'}) min,max",
                                               tag=f"pyecr_dual_map_{process}_{name}_range",width=TOKENS.map_controls.range_width,
                                               callback=self._map_calculation_changed,user_data="dual")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Recalculate map", tag="pyecr_dual_map_generate",callback=self._generate_dual_map,enabled=False)
            add_action_tooltip(dpg, "pyecr_dual_map_generate")
            dpg.add_button(label="Cancel calculation", tag="pyecr_dual_map_cancel",callback=self._cancel_dual_map,enabled=False)
        dpg.add_text("Choose a range. Apply view uses stored cells; Recalculate map evaluates a new grid. No refit.",tag="pyecr_dual_map_status",wrap=TOKENS.wraps.map)
        dpg.add_progress_bar(tag="pyecr_dual_map_progress", default_value=0.0, width=-1)
        dpg.add_text("",tag="pyecr_dual_map_captured_grid",wrap=TOKENS.map_controls.wrap)
        self._register_owned_root("pyecr_dual_map_display_controls")
        with dpg.collapsing_header(label="3 · Display",tag="pyecr_dual_map_display_controls",default_open=False):
            with dpg.group(horizontal=True, tag="pyecr_dual_map_toolbar"):
                dpg.add_combo(
                    (DualMapMetric.LOG_RATIO.value, DualMapMetric.RAW_SSE.value),
                    default_value=DualMapMetric.LOG_RATIO.value,
                    label="Metric", width=TOKENS.widths.toolbar_metric,
                    tag="pyecr_dual_map_metric", callback=self._dual_map_display_changed,
                )
                dpg.add_combo(
                    ("Independent", "Shared"), default_value="Independent",
                    label="Color scale", width=TOKENS.widths.toolbar_resolution,
                    tag="pyecr_dual_map_color_scale", callback=self._dual_map_display_changed,
                )
                dpg.add_checkbox(
                    label="Grid min", default_value=True,
                    tag="pyecr_dual_map_grid_min", callback=self._dual_map_display_changed,
                )
            self._build_map_view_controls("dual")
            dpg.add_button(label="Expand view",parent='pyecr_dual_map_metric_panel',callback=self._focus_map,user_data="dual")
        with dpg.collapsing_header(label="4 Export captured maps",default_open=False):
            dpg.add_input_text(label="New report directory",tag="pyecr_dual_map_export_path",
                               hint="New nonexistent path; verified ReportBundle, no recalculation",width=-1)
            dpg.add_button(label="Export captured Dual Error Maps",tag="pyecr_dual_map_export",callback=self._export_dual_map,enabled=False)
            add_action_tooltip(dpg, "pyecr_dual_map_export")
        dpg.add_text("", tag="pyecr_dual_map_stale_banner", show=False,
                     color=TOKENS.colors.warning, wrap=TOKENS.wraps.map)
        dpg.add_text("", tag="pyecr_dual_map_warning_banner", show=False,
                     color=TOKENS.colors.warning, wrap=TOKENS.wraps.map)
        with dpg.group(tag="pyecr_dual_map_plots", horizontal=True):
            for process in ("H", "O"):
                with dpg.group(tag=_tag(process, "panel")):
                    dpg.add_text(f"{process} process · {DualMapMetric.LOG_RATIO.value}",
                                 tag=_tag(process, "title"))
                    with dpg.group(horizontal=True, tag=_tag(process, "visual_row")):
                        with dpg.plot(
                            label=f"{process} process objective surface",
                            tag=_tag(process, "plot"), height=TOKENS.heights.error_map,
                            width=TOKENS.heights.error_map, no_mouse_pos=True,
                        ):
                            dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                            dpg.add_plot_axis(dpg.mvXAxis, label="D [cm²/s] (log10 axis)",
                                              tag=_tag(process, "x_axis"))
                            dpg.add_plot_axis(dpg.mvYAxis, label="k [cm/s] (log10 axis)",
                                              tag=_tag(process, "y_axis"))
                            dpg.add_scatter_series([], [], label="Continuous fit (white dot)",
                                                   parent=_tag(process, "y_axis"),
                                                   tag=_tag(process, "fit_point"))
                            dpg.add_line_series([], [], label="Grid minimum (cross)",
                                                parent=_tag(process, "y_axis"),
                                                tag=_tag(process, "cross_h"), show=False)
                            dpg.add_line_series([], [], label="##grid minimum vertical",
                                                parent=_tag(process, "y_axis"),
                                                tag=_tag(process, "cross_v"), show=False)
                        dpg.add_colormap_scale(
                            label=DualMapMetric.LOG_RATIO.value,
                            tag=_tag(process, "colorbar"),
                            height=TOKENS.heights.error_map, width=TOKENS.widths.colorbar,
                            colormap=dpg.mvPlotColormap_Viridis, format="%.2e", show=False,
                        )
                    dpg.add_text("Hover for stored D/k/metric/validity.",
                                 tag=_tag(process, "hover"), wrap=TOKENS.wraps.map)
                    # DearPyGui plots accept axes/series, not tooltip children.
                    # Attach the tooltip to the enclosing group instead.
                    with dpg.tooltip(parent=_tag(process, "panel")):
                        dpg.add_text("Hover for a stored grid cell.",
                                     tag=_tag(process, "tooltip"), wrap=TOKENS.wraps.map)
                    dpg.bind_item_theme(_tag(process, "fit_point"), FIT_MARKER_THEME_TAG)
                    for part in ("cross_h","cross_v"):
                        dpg.bind_item_theme(_tag(process,part),"pyecr_map_grid_min_theme")
                    dpg.bind_colormap(_tag(process, "plot"), dpg.mvPlotColormap_Viridis)
        dpg.add_text(
            "Both maps use full-signal fixed-nuisance SSE. Colors are objective visualization, not confidence intervals. "
            "Invalid numerical cells are masked. Export consumes the captured H/O grids; it never recomputes the objective.",
            wrap=TOKENS.wraps.map,parent='pyecr_dual_map_calculation_advanced',
        )

    def _generate_dual_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        if self._has_range_controls('dual'):
            self._recalculate_map_ranges('dual')
            return
        try:
            dpg=self._require_dpg();label=dpg.get_value("pyecr_dual_map_resolution") or "Quick"
            n=grid_points(label,dpg.get_value("pyecr_dual_map_points"))
            resolution=ErrorMapResolution[label.upper()] if label.upper() in ErrorMapResolution.__members__ else ErrorMapResolution.FINE
            mode=dpg.get_value("pyecr_dual_map_range_mode") or "AUTO"
            ranges={}
            if mode!="AUTO":
                bounds=self._dual_map_bounds()
                for process in ("h","o"):
                    for axis in ("d","k"):
                        key=(process,axis)
                        raw=dpg.get_value(f"pyecr_dual_map_{process}_{axis}_range")
                        values=bounds[key] if mode=="FULL BOUNDS" else tuple(float(x.strip()) for x in str(raw).split(','))
                        if len(values)!=2 or values[0]<bounds[key][0]-1e-10 or values[1]>bounds[key][1]+1e-10:
                            raise ValueError(f"{process.upper()} {axis} range must stay within validated bounds {bounds[key]}")
                        ranges[f"{process}_log10_{'D' if axis=='d' else 'k'}_range"]=values
            settings=DualErrorMapConfig(resolution=resolution,n_points=n,
                auto_half_width_decades=dpg.get_value("pyecr_dual_map_half_width") or 1.5,
                range_mode=DualMapRangeMode.AUTO if mode=="AUTO" else DualMapRangeMode.MANUAL,**ranges)
            self._map_view_changed(user_data="dual")
            self.controller.run_dual_error_map(settings)
            self._remember_map_attempt("dual")
            self._dual_map_range_note = ""
        except Exception as exc:
            self._dual_map_range_note = f"Generate rejected: {exc}"
            self.controller.record_callback_exception(GuiPage.DIAGNOSTICS, exc, traceback.format_exc())
        self._refresh_widgets()

    def _dual_map_range_changed(self,sender=None,app_data=None,user_data=None):
        dpg=self._require_dpg()
        dpg.configure_item("pyecr_dual_map_manual_ranges",show=dpg.get_value("pyecr_dual_map_range_mode")=="MANUAL")
        self._refresh_map_instructions("dual")

    def _dual_map_bounds(self):
        session=self.controller.analysis_session
        physical,label=self.controller.dual_error_map_service.parameter_policy(self.controller.workflow.prepare_dual_error_map())
        bounds={(p.lower(),a):tuple(log10(x) for x in physical[f"{'D' if a=='d' else 'k'}_{p}_{'cm2_s' if a=='d' else 'cm_s'}"])
                for p in ("H","O") for a in ("d","k")}
        if label=="D_H>=D_O":
            lo,hi=bounds['h','d'];bounds['h','d']=(max(lo,log10(session.fit_result.params['D_O_cm2_s'])),hi)
            lo,hi=bounds['o','d'];bounds['o','d']=(lo,min(hi,log10(session.fit_result.params['D_H_cm2_s'])))
        return bounds

    def _cancel_dual_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self.controller.cancel_dual_error_map()
        self._refresh_widgets()

    def _export_dual_map(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        try:
            current = self.controller.current_dual_error_map_result
            if current is None:
                raise RuntimeError("cannot export an Out of date or missing H/O Error Map")
            destination = str(self._require_dpg().get_value("pyecr_dual_map_export_path") or "").strip()
            if not destination:
                raise ValueError("enter a new report directory for H/O Error Map export")
            snapshot = self.controller.capture_snapshot()
            if snapshot.dual_error_map_result is not current:
                raise RuntimeError("current H/O Error Map was not captured; export refused")
            self.controller.export_report_bundle(destination, snapshot=snapshot)
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.REPORT, exc, traceback.format_exc())
        self._refresh_widgets()

    def _dual_map_display_changed(self, sender: Any, app_data: Any, user_data: Any) -> None:
        del sender, app_data, user_data
        self._render_dual_error_map_plot(force=True)

    def _refresh_dual_error_map_widgets(self) -> None:
        dpg = self._require_dpg()
        self._refresh_map_instructions("dual")
        session = self.controller.analysis_session
        current = self.controller.current_dual_error_map_result
        saved = session.dual_error_map_result
        dpg.configure_item('pyecr_dual_map_plots',show=saved is not None)
        status = self.controller.dual_error_map_status
        running = self.controller.state.tasks[GuiPage.DIAGNOSTICS].status == "running"
        dpg.configure_item('pyecr_dual_map_progress',show=running and session.dual_error_map_running)
        can_run = self.controller.workflow.can_dual_error_map
        fit = session.fit_result
        no_k = bool(fit is not None and not session.fit_stale and
                    not ("k_H_cm_s" in fit.params and "k_O_cm_s" in fit.params))
        set_action_availability(
            dpg, "pyecr_dual_map_generate", enabled=can_run and not running,
            reason="Requires a current final-selected M3 fit and no running Diagnostics task; M2 has no k_H/k_O.",
        )
        dpg.configure_item("pyecr_dual_map_cancel", enabled=running and session.dual_error_map_running, show=running and session.dual_error_map_running)
        set_action_availability(
            dpg, "pyecr_dual_map_export", enabled=current is not None,
            reason=("STALE H/O maps cannot be exported as a current report."
                    if saved is not None and current is None else
                    "Generate a current H/O Error Map before export."),
        )
        stale = saved is not None and current is None
        dpg.configure_item("pyecr_dual_map_stale_banner", show=stale)
        dpg.set_value("pyecr_dual_map_stale_banner",
                      "STALE — showing an earlier H/O map; it is not current and cannot be exported as a current report."
                      if stale else "")
        progress = self.controller.dual_error_map_progress
        dpg.set_value("pyecr_dual_map_progress", 0.0 if progress is None else progress.combined_fraction)
        if stale:
            message = ("Out of date — old H/O maps remain viewable. "
                       + ("Current result has no k_H/k_O; dual D-k maps cannot be regenerated from this fit."
                          if no_k else "Regenerate after a current 06 M5_DUAL_H2O_DK fit."))
        elif current is not None and status == "updating":
            message = "Updating — previous valid H/O maps remain visible until both new maps succeed."
        elif current is not None:
            message = "Current 06 M5_DUAL_H2O_DK maps; white dots are fit points, crosses are grid minima."
        elif no_k:
            message = "Current result has no k_H/k_O — a D-only selected result cannot generate dual D-k maps."
        elif status == "running":
            message = "Generating H/O maps in a worker."
        else:
            message = ("Ready · Recalculate to evaluate this range." if can_run else
                       "Run a current 06 M5_DUAL_H2O_DK Fit first.")
        if progress is not None and status in {"running", "updating"}:
            message += (f" H {progress.h_fraction:.0%} | O {progress.o_fraction:.0%}"
                        f" | combined {progress.combined_fraction:.0%}.")
        dpg.set_value("pyecr_dual_map_status", " ".join(filter(None,(
            message, getattr(self,"_dual_map_range_note",""), self._map_attempt_summary("dual")))))
        result = current if current is not None else saved if stale else None
        warning_lines = [] if result is None else [
            *(f"H: {warning}" for warning in result.h_map.warnings),
            *(f"O: {warning}" for warning in result.o_map.warnings),
        ]
        dpg.configure_item("pyecr_dual_map_warning_banner", show=bool(warning_lines))
        dpg.set_value("pyecr_dual_map_warning_banner", "; ".join(warning_lines))
        self._render_dual_error_map_plot(result=result)

    def _render_dual_error_map_plot(
        self, *, result: DualErrorMapResult | None = None, force: bool = False,
    ) -> None:
        dpg = self._require_dpg()
        if result is None:
            result = self.controller.current_dual_error_map_result
            if result is None:
                result = self.controller.analysis_session.dual_error_map_result
        if result is None:
            self._clear_dual_error_map_plot()
            return
        options=self._map_options_for_render("dual")
        metric = DualMapMetric(str(dpg.get_value("pyecr_dual_map_metric")))
        shared = dpg.get_value("pyecr_dual_map_color_scale") == "Shared"
        show_grid_min = bool(dpg.get_value("pyecr_dual_map_grid_min"))
        key = (metric, shared, show_grid_min,options.raster_options())
        if (force or result is not getattr(self, "_rendered_dual_map_result", None)
                or key != getattr(self, "_rendered_dual_map_key", None)):
            h_data = dual_map_plot_data(result.h_map, metric)
            o_data = dual_map_plot_data(result.o_map, metric)
            if shared:
                import numpy as np
                values=np.r_[h_data.values[h_data.display_valid_mask],o_data.values[o_data.display_valid_mask]]
                scale=color_limits(values,np.ones(values.shape,dtype=bool),options)
                limits=(scale if h_data.display_valid_mask.any() else None,scale if o_data.display_valid_mask.any() else None)
            else:limits=tuple(color_limits(x.values,x.display_valid_mask,options) for x in (h_data,o_data))
            for data, scale in zip((h_data, o_data), limits, strict=True):
                self._draw_dual_map_slice(data, scale, show_grid_min=show_grid_min,options=options)
            self._bind_map_colors("dual",options)
            self._rendered_dual_map_key = key
            self._rendered_dual_map_result = result
            self._rendered_dual_map_data = {"H": h_data, "O": o_data}
        for process, data in getattr(self, "_rendered_dual_map_data", {}).items():
            self._render_map_axis_range('dual',process,result,options,(log_axis_edges(data.log10_D),log_axis_edges(data.log10_k)))
            if not dpg.is_item_hovered(_tag(process, "plot")):
                continue
            log_d, log_k = dpg.get_plot_mouse_pos()
            cell = data.nearest_cell(log_d, log_k) if isfinite(log_d) and isfinite(log_k) else None
            if cell is None:
                message = "Outside the computed grid."
            else:
                coordinates = f"D={cell.D_cm2_s:.3e} cm²/s | k={cell.k_cm_s:.3e} cm/s"
                metric_text = (f"{data.label}={cell.metric_value:.3e}" if cell.metric_value is not None
                               else f"{data.label}=N/A")
                message = (f"{coordinates} | {metric_text} | "
                           f"{'valid' if cell.valid else 'invalid'}"
                           f"{': ' + cell.reason if cell.reason else ''}")
            dpg.set_value(_tag(process, "hover"), message)
            dpg.set_value(_tag(process, "tooltip"), message)

    def _draw_dual_map_slice(
        self, data: DualMapPlotData, scale: tuple[float, float] | None,
        *, show_grid_min: bool, options=None,
    ) -> None:
        dpg = self._require_dpg()
        process = data.process
        if options is None:options=self.controller.map_view_options("dual")
        heat_tag = _tag(process, "heat_series")
        if dpg.does_item_exist(heat_tag):
            dpg.delete_item(heat_tag)
        dpg.set_value(_tag(process, "title"), f"{process} process · {data.label}")
        if scale is None:
            dpg.configure_item(_tag(process, "colorbar"), show=False)
            for part in ("fit_point", "cross_h", "cross_v"):
                dpg.set_value(_tag(process, part), [[], []])
                dpg.configure_item(_tag(process, part), show=False)
            dpg.set_value(_tag(process, "hover"), data.display_warning or "No finite cells for this metric; use raw SSE.")
            dpg.set_value(_tag(process, "tooltip"), data.display_warning or "No finite cells for this metric.")
            return
        d_edges = log_axis_edges(data.log10_D)
        k_edges = log_axis_edges(data.log10_k)
        shown=display_raster(data.values,data.display_valid_mask,options,color_range=scale)
        dpg.add_heat_series(
            shown.T.ravel().tolist(), shown.shape[1], shown.shape[0],
            parent=_tag(process, "y_axis"), tag=heat_tag, label=data.label,
            format="", scale_min=scale[0], scale_max=scale[1],
            bounds_min=(float(d_edges[0]), float(k_edges[0])),
            bounds_max=(float(d_edges[-1]), float(k_edges[-1])),
        )
        dpg.move_item(heat_tag, parent=_tag(process, "y_axis"), before=_tag(process, "fit_point"))
        dpg.set_value(_tag(process, "fit_point"), [[data.fitted_point[0]], [data.fitted_point[1]]])
        dpg.configure_item(_tag(process, "fit_point"), show=True)
        if data.grid_minimum is not None:
            d_center, k_center = data.grid_minimum
            d_half = 0.012 * float(data.log10_D[-1] - data.log10_D[0])
            k_half = 0.012 * float(data.log10_k[-1] - data.log10_k[0])
            dpg.set_value(_tag(process, "cross_h"),
                          [[d_center - d_half, d_center + d_half], [k_center, k_center]])
            dpg.set_value(_tag(process, "cross_v"),
                          [[d_center, d_center], [k_center - k_half, k_center + k_half]])
        for part in ("cross_h", "cross_v"):
            dpg.configure_item(_tag(process, part), show=show_grid_min and data.grid_minimum is not None)
        dpg.configure_item(_tag(process, "colorbar"), show=True, label=data.label,
                           min_scale=scale[0], max_scale=scale[1])
        dpg.set_value(_tag(process,"title"),f"{process} · {data.label} · evaluated {len(data.log10_D)}²; "
                      f"{options.sampling} view {shown.shape[0]}² · {options.palette}")
        dpg.set_value(_tag(process, "hover"), "Hover for stored D/k/metric/validity.")

    def _clear_dual_error_map_plot(self) -> None:
        dpg = self._require_dpg()
        for process in ("H", "O"):
            heat_tag = _tag(process, "heat_series")
            if dpg.does_item_exist(heat_tag):
                dpg.delete_item(heat_tag)
            for part in ("fit_point", "cross_h", "cross_v"):
                dpg.set_value(_tag(process, part), [[], []])
                dpg.configure_item(_tag(process, part), show=False)
            dpg.configure_item(_tag(process, "colorbar"), show=False)
            dpg.set_value(_tag(process, "hover"), "No computed grid.")
        self._rendered_dual_map_key = None
        self._rendered_dual_map_result = None
        self._rendered_dual_map_data = {}
