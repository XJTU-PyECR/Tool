"""Deterministic headless Matplotlib renderers for PyECR plot data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyecr.plotting.data import FitPlotData
from pyecr.diagnostics.error_map import ErrorMapResult
from pyecr.diagnostics.dual_error_map import DualErrorMapSlice
from pyecr.plotting.dual_error_map import DualMapMetric, dual_map_plot_data
from pyecr.plotting.error_map import ErrorMapMetric, error_map_plot_data, log_axis_edges
from pyecr.plotting.visual_style import SERIES,PUBLICATION,style_publication_figure

__all__ = [
    "render_fit_figure", "render_residual_figure", "render_error_map_figure",
    "render_dual_error_map_figure",
]

_PNG_METADATA = {"Software": "PyECR reproducible report"}


def render_fit_figure(data: FitPlotData, destination: str | Path, *, plot_views=None) -> Path:
    """Render observed, fitted, and optional component curves without a GUI."""

    target = _png_target(destination)
    figure = Figure(figsize=(7.2, 4.5), dpi=120, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.scatter(data.time_s, data.observed, s=SERIES['observed'].marker_size, color=SERIES['observed'].hex,label="Data", zorder=3)
    axis.plot(data.time_s, data.fitted, linewidth=SERIES['total'].width,color=SERIES['total'].hex,label="Fit")
    if data.has_components and data.component_H:
        axis.plot(data.time_s, data.component_H, linewidth=SERIES['one'].width,color=SERIES['one'].hex,linestyle=SERIES['one'].line,label="H component")
    if data.has_components and data.component_O:
        axis.plot(data.time_s, data.component_O, linewidth=SERIES['two'].width,color=SERIES['two'].hex,linestyle=SERIES['two'].line,label="O component")
    if data.model_name=="M8_DUAL_K" and data.has_components:
        for i,values in enumerate((data.component_1,data.component_2)):
            style=SERIES['one' if i==0 else 'two']
            axis.plot(data.time_s,values,linewidth=style.width,color=style.hex,linestyle=style.line,label=data.component_labels[i] if data.component_labels else f"Effective process {i+1}")
        axis.axhline(0.,linewidth=SERIES['reference'].width,color=SERIES['reference'].hex,linestyle=SERIES['reference'].line,label="Zero-baseline contributions")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(data.signal_label)
    axis.set_title(f"ECR fit — {data.model_name}")
    axis.grid(True,alpha=PUBLICATION.grid_alpha)
    axis.legend(frameon=False)
    style_publication_figure(figure)
    from pyecr.plotting.view_state import apply_axis_view
    apply_axis_view(axis,'pyecr_fit_plot',plot_views,('pyecr_fit_data_series','pyecr_fit_curve_series','pyecr_fit_h_series','pyecr_fit_o_series'))
    canvas.print_png(target, metadata=_PNG_METADATA)
    return target


def render_residual_figure(data: FitPlotData, destination: str | Path, *, plot_views=None) -> Path:
    """Render observed-minus-fitted residuals without a GUI."""

    target = _png_target(destination)
    figure = Figure(figsize=(7.2, 3.2), dpi=120, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.axhline(0.0,color=SERIES['zero'].hex,linewidth=SERIES['zero'].width,linestyle=SERIES['zero'].line,label='Zero')
    axis.scatter(data.time_s, data.residuals, s=SERIES['residual'].marker_size,color=SERIES['residual'].hex,label='Residual')
    axis.legend(frameon=False)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Observed - fitted")
    axis.set_title("Fit residuals")
    axis.grid(True,alpha=PUBLICATION.grid_alpha)
    style_publication_figure(figure)
    from pyecr.plotting.view_state import apply_axis_view
    apply_axis_view(axis,'pyecr_residual_plot',plot_views,('pyecr_residual_zero_series','pyecr_residual_series'))
    canvas.print_png(target, metadata=_PNG_METADATA)
    return target


def render_error_map_figure(result: ErrorMapResult, destination: str | Path, *, view_options=None, plot_views=None, range_metadata=None) -> Path:
    """Render the captured grid; this function never evaluates a model."""

    target = _png_target(destination)
    if view_options is not None:
        figure=objective_map_figure(result,view_options)
        FigureCanvasAgg(figure)
        from pyecr.plotting.view_state import apply_axis_view
        apply_axis_view(figure.axes[0],'pyecr_error_map_plot',plot_views,('pyecr_error_map_fit_point','pyecr_error_map_grid_cross_h'))
        map_fit_view_note(figure.axes[0],result.fitted_point.log10_D,result.fitted_point.log10_k)
        _capture_map_ranges(figure.axes[0],result,range_metadata)
        figure.canvas.print_png(target,metadata=_PNG_METADATA)
        return target
    data = error_map_plot_data(result, ErrorMapMetric.LOG_RATIO)
    figure = Figure(figsize=(7.2, 5.4), dpi=120, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    values = np.ma.masked_where(~data.valid_mask | ~np.isfinite(data.values), data.values)
    d_edges = np.power(10.0, log_axis_edges(result.log10_D))
    k_edges = np.power(10.0, log_axis_edges(result.log10_k))
    image = axis.pcolormesh(d_edges, k_edges, values.T, shading="flat", cmap="viridis")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.scatter([data.fitted_D_cm2_s], [data.fitted_k_cm_s], c="white", edgecolors="black", s=65, label="Continuous fit", zorder=5)
    if data.grid_minimum is not None:
        axis.scatter([data.grid_minimum[0]], [data.grid_minimum[1]], marker="x", c="red", s=65, label="Grid minimum", zorder=6)
    axis.set_xlabel("D (cm²/s)")
    axis.set_ylabel("k (cm/s)")
    axis.set_title("M1 fixed-nuisance D–k objective map")
    axis.legend(frameon=True)
    figure.colorbar(image, ax=axis, label=data.label)
    from pyecr.plotting.view_state import apply_axis_view
    apply_axis_view(axis,'pyecr_error_map_plot',plot_views,('pyecr_error_map_fit_point','pyecr_error_map_grid_cross_h'),log_coordinates=True)
    _capture_map_ranges(axis,result,range_metadata,physical=True)
    if range_metadata is not None:range_metadata.update(metric=data.label,color_limits=list(image.get_clim()))
    canvas.print_png(target, metadata=_PNG_METADATA)
    return target


def render_dual_error_map_figure(
    result: DualErrorMapSlice, destination: str | Path, *, view_options=None, color_range=None, plot_views=None, range_metadata=None,
) -> Path:
    """Render one captured H/O full-signal grid without evaluating an objective."""

    target = _png_target(destination)
    if view_options is not None:
        figure=objective_map_figure(result,view_options,color_range=color_range)
        FigureCanvasAgg(figure)
        from pyecr.plotting.view_state import apply_axis_view
        prefix='pyecr_dual_map_'+result.process.lower()+'_'
        apply_axis_view(figure.axes[0],prefix+'plot',plot_views,(prefix+'fit_point',prefix+'cross_h'))
        map_fit_view_note(figure.axes[0],result.fitted_point.log10_D,result.fitted_point.log10_k)
        _capture_map_ranges(figure.axes[0],result,range_metadata)
        figure.canvas.print_png(target,metadata=_PNG_METADATA)
        return target
    data = dual_map_plot_data(result, DualMapMetric.LOG_RATIO)
    if not data.display_valid_mask.any():
        data = dual_map_plot_data(result, DualMapMetric.RAW_SSE)
    figure = Figure(figsize=(7.2, 5.4), dpi=120, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    values = np.ma.masked_where(~data.display_valid_mask, data.display_values)
    d_edges = np.power(10.0, log_axis_edges(result.log10_D))
    k_edges = np.power(10.0, log_axis_edges(result.log10_k))
    image = axis.pcolormesh(d_edges, k_edges, values.T, shading="flat", cmap="viridis")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.scatter([10.0 ** data.fitted_point[0]], [10.0 ** data.fitted_point[1]],
                 c="white", edgecolors="black", s=65, label="Continuous fit", zorder=5)
    if data.grid_minimum is not None:
        axis.scatter([10.0 ** data.grid_minimum[0]], [10.0 ** data.grid_minimum[1]],
                     marker="x", c="red", s=65, label="Grid minimum", zorder=6)
    axis.set_xlabel("D (cm²/s)")
    axis.set_ylabel("k (cm/s)")
    axis.set_title(f"M3 {result.process} fixed-nuisance full-signal objective map")
    axis.legend(frameon=True)
    figure.colorbar(image, ax=axis, label=data.label)
    from pyecr.plotting.view_state import apply_axis_view
    prefix='pyecr_dual_map_'+result.process.lower()+'_'
    apply_axis_view(axis,prefix+'plot',plot_views,(prefix+'fit_point',prefix+'cross_h'),log_coordinates=True)
    _capture_map_ranges(axis,result,range_metadata,physical=True)
    if range_metadata is not None:range_metadata.update(metric=data.label,color_limits=list(image.get_clim()))
    canvas.print_png(target, metadata=_PNG_METADATA)
    return target


def objective_map_figure(result, options, *, color_range=None):
    """Captured cells only; smooth SVG/PDF embeds a raster with vector axes."""
    from pyecr.plotting.map_display import color_limits,display_raster,padded_limits,palette_colors
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    if isinstance(result,ErrorMapResult):
        data=error_map_plot_data(result,ErrorMapMetric(options.metric))
        values,mask=data.values,data.display_valid_mask
        fitted=(result.fitted_point.log10_D,result.fitted_point.log10_k)
        minimum=result.grid_minimum;title="M1 fixed-nuisance D–k objective map"
    else:
        data=dual_map_plot_data(result,DualMapMetric(options.metric))
        values,mask=data.values,data.display_valid_mask;fitted=data.fitted_point
        minimum=result.grid_minimum;title=f"M3 {result.process} fixed-nuisance full-signal objective map"
    limits=color_range or color_limits(values,mask,options)
    figure=Figure(figsize=(7.2,5.4),dpi=options.image_dpi,layout="constrained")
    figure._pyecr_map_color_range=limits
    axis=figure.add_subplot(111)
    d_edges=log_axis_edges(result.log10_D);k_edges=log_axis_edges(result.log10_k)
    shown=display_raster(values,mask,options,color_range=limits)
    cmap=ListedColormap(palette_colors(options.cmap_name)/255.,name=options.cmap_name)
    if limits is not None:
        if options.sampling=="Smooth":
            image=axis.imshow(shown.T,origin="lower",aspect="auto",interpolation="nearest",
                extent=(d_edges[0],d_edges[-1],k_edges[0],k_edges[-1]),cmap=cmap,vmin=limits[0],vmax=limits[1])
        else:
            image=axis.pcolormesh(d_edges,k_edges,np.ma.masked_invalid(shown.T),shading="flat",cmap=cmap,vmin=limits[0],vmax=limits[1])
        figure.colorbar(image,ax=axis,label=data.label)
    else:axis.text(.5,.5,"No finite cells for selected metric; use SSE",transform=axis.transAxes,ha="center")
    axis.scatter([fitted[0]],[fitted[1]],c="white",edgecolors="black",s=65,label="Continuous fit",zorder=5)
    if minimum is not None and options.show_grid_min:
        axis.scatter([minimum.log10_D],[minimum.log10_k],marker="x",c="#e64b46",s=65,label="Grid minimum",zorder=6)
    for coordinate in (axis.xaxis,axis.yaxis):
        coordinate.set_major_locator(MaxNLocator(nbins=4))
        coordinate.set_major_formatter(FuncFormatter(lambda v,p:f"{10**v:.1e}"))
    view_ranges=options.ranges('single' if isinstance(result,ErrorMapResult) else result.process)
    axis.set_xlim(*(view_ranges[0] if view_ranges else padded_limits(d_edges,options)))
    axis.set_ylim(*(view_ranges[1] if view_ranges else padded_limits(k_edges,options)))
    map_fit_view_note(axis,*fitted)
    axis.set_xlabel("D (cm²/s; log scale)");axis.set_ylabel("k (cm/s; log scale)")
    axis.set_title(title);axis.legend(frameon=False)
    figure.supxlabel(f"Evaluated {len(result.log10_D)}×{len(result.log10_k)}; {options.sampling} display. Objective colors, not CI.",fontsize=8)
    style_publication_figure(figure)
    return figure


def map_fit_view_note(axis,log_D,log_k):
    """Reflect final displayed axes, including a later frozen pan/zoom."""
    for item in tuple(axis.texts):
        if item.get_gid()=='pyecr_map_fit_outside':item.remove()
    x,y=axis.get_xlim(),axis.get_ylim()
    if not (x[0]<=log_D<=x[1] and y[0]<=log_k<=y[1]):
        axis.text(.02,.02,'Fit point outside view',transform=axis.transAxes,fontsize=8,gid='pyecr_map_fit_outside')


def _capture_map_ranges(axis,result,metadata,*,physical=False):
    if metadata is None:return
    x,y=axis.get_xlim(),axis.get_ylim()
    if physical:x,y=np.log10(x),np.log10(y)
    metadata.update(display_range_log10={'D':[float(v) for v in x],'k':[float(v) for v in y]},
        fit_point_in_view=bool(x[0]<=result.fitted_point.log10_D<=x[1] and y[0]<=result.fitted_point.log10_k<=y[1]))


def _png_target(value: str | Path) -> Path:
    target = Path(value)
    if target.suffix.lower() != ".png":
        raise ValueError("Matplotlib report figures must use a .png suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def decomposition_figure(data):
    """Build a figure from the same immutable scene used by the native view."""
    figure = Figure(figsize=(8.8, 6.6), dpi=140, layout="constrained")
    figure.get_layout_engine().set(rect=(0., .035, 1., .965))
    if data.options.show_residuals:
        axis, residual = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": PUBLICATION.residual_height_ratios})
        residual.scatter(data.observation.time_s, data.residuals,s=SERIES['residual'].marker_size,color=SERIES['residual'].hex)
        residual.axhline(0.,linewidth=SERIES['zero'].width,color=SERIES['zero'].hex,linestyle=SERIES['zero'].line)
        residual.set_ylabel(f"Residual ({data.signal_unit})")
        residual.set_xlabel("Time (s)")
        residual.grid(alpha=PUBLICATION.grid_alpha)
    else:
        axis = figure.subplots()
        axis.set_xlabel("Time (s)")
    a, b = data.display.contributions(data.y0, data.options.mode)
    reference = data.y0 if data.options.mode == "shifted" else 0.

    def draw(target):
        target.scatter(data.observation.time_s, data.observed,s=SERIES['observed'].marker_size,color=SERIES['observed'].hex,label="Observed", zorder=3)
        target.plot(data.display.time_s, data.display.total,color=SERIES['total'].hex,linewidth=SERIES['total'].width,label="Joint fit")
        for values,label,role in zip((a,b),data.labels,('one','two')):
            style=SERIES[role]
            target.plot(data.display.time_s,values,color=style.hex,linewidth=style.width,linestyle=style.line,label=label)
        target.axhline(reference,color=SERIES['reference'].hex,linewidth=SERIES['reference'].width,linestyle=SERIES['reference'].line)
        target.grid(alpha=PUBLICATION.grid_alpha)
    draw(axis)
    axis.set_ylabel(f"Signal / signed contributions ({data.signal_unit})")
    from pyecr.model_display import display_name
    axis.set_title(f"Continuous decomposition · {display_name(data.model_id)}\n{data.formula}",fontsize=PUBLICATION.title_pt)
    axis.legend(frameon=True,facecolor=PUBLICATION.face,edgecolor="none",framealpha=.9,fontsize=PUBLICATION.legend_pt,loc="upper right")
    if data.options.show_amplitudes:
        text = f"Signed plateau amplitudes: B={data.signed_amplitudes[0]:.4g}, C={data.signed_amplitudes[1]:.4g} {data.signal_unit}"
        axis.text(.02, .02, text, transform=axis.transAxes, fontsize=PUBLICATION.note_pt,
                  bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    if data.options.show_early_inset:
        inset = axis.inset_axes(PUBLICATION.inset_bounds)
        draw(inset)
        inset.set_xlim(data.observation.time_s[0], data.early_end_s)
        mask = np.asarray(data.display.time_s) <= data.early_end_s
        obs_mask = np.asarray(data.observation.time_s) <= data.early_end_s
        visible = np.r_[np.asarray(data.display.total)[mask], np.asarray(a)[mask], np.asarray(b)[mask], np.asarray(data.observed)[obs_mask], reference]
        lo, hi = float(visible.min()), float(visible.max())
        pad = max((hi - lo) * .08, abs(hi) * 1e-8, 1e-12)
        inset.set_ylim(lo - pad, hi + pad)
        inset.set_title("Early time",fontsize=PUBLICATION.note_pt)
        inset.tick_params(labelsize=PUBLICATION.inset_tick_pt)
    figure.text(.5,PUBLICATION.footer_fraction,"Signed effective contributions; physical attribution requires evidence. No component CI.",ha="center",fontsize=PUBLICATION.note_pt)
    style_publication_figure(figure)
    return figure


def render_decomposition_figure(data, destination: str | Path, *, plot_views=None) -> Path:
    target = _png_target(destination)
    figure=decomposition_figure(data)
    FigureCanvasAgg(figure)
    from pyecr.plotting.view_state import apply_axis_view
    apply_axis_view(figure.axes[0],'pyecr_components_plot',plot_views,('pyecr_decomposition_observed','pyecr_decomposition_total','pyecr_fit_h_series','pyecr_fit_o_series','pyecr_decomposition_reference'))
    figure.canvas.print_png(target, metadata=_PNG_METADATA)
    return target
