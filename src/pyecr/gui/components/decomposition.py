"""Native decomposition controls and renderer using the shared scientific scene."""
from dataclasses import replace

from pyecr.plotting.decomposition import DecompositionViewOptions, INTERPRETATION
from pyecr.gui.theme import TOKENS
from pyecr.plotting.visual_style import SERIES,native_line_geometry,marker_indices


def build_controls(view):
    dpg = view._require_dpg()
    with dpg.group(tag="pyecr_decomposition_controls"):
        with dpg.group(horizontal=True):
            dpg.add_combo(["zero", "shifted"], label="Baseline view", default_value="zero", width=TOKENS.widths.toolbar_range,
                          tag="pyecr_decomposition_mode", callback=view._change_decomposition_view)
            for label, tag, default in (("Early inset", "early", False), ("Residual axis", "residuals", True), ("Amplitudes", "amplitudes", False)):
                dpg.add_checkbox(label=label, tag=f"pyecr_decomposition_{tag}", default_value=default,
                                 callback=view._change_decomposition_view)
        dpg.add_slider_float(label="Early window fraction", tag="pyecr_decomposition_fraction", default_value=.15,
                             min_value=.01, max_value=1., width=TOKENS.widths.input_medium, callback=view._change_decomposition_view)
        dpg.add_text("Requires a current joint dual fit.", tag="pyecr_decomposition_formula", wrap=TOKENS.wraps.project)
        dpg.add_text(INTERPRETATION, tag="pyecr_decomposition_note", wrap=TOKENS.wraps.project)
        dpg.add_text("", tag="pyecr_decomposition_amplitude_note", show=False, wrap=TOKENS.wraps.project)


def build_auxiliary(view):
    dpg = view._require_dpg()
    for suffix, label, height in (("early", "Early inset · same joint model", TOKENS.heights.decomposition_early), ("residual", "Observation residuals", TOKENS.heights.decomposition_residual)):
        with dpg.plot(label=label, tag=f"pyecr_decomposition_{suffix}_plot", height=height, width=-1, show=False):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=f"pyecr_decomposition_{suffix}_x")
            axis = dpg.add_plot_axis(dpg.mvYAxis, label="Signal", tag=f"pyecr_decomposition_{suffix}_y")
            if suffix == "early":
                for name in ("total", "one", "two", "reference"):
                    dpg.add_line_series([], [], label=name, parent=axis, tag=f"pyecr_decomposition_early_{name}")
                dpg.add_scatter_series([], [], label="Observed", parent=axis, tag="pyecr_decomposition_early_observed")
            else:
                dpg.add_scatter_series([], [], label="Observed - fitted", parent=axis, tag="pyecr_decomposition_residual_values")
                dpg.add_line_series([], [], label="Zero", parent=axis, tag="pyecr_decomposition_residual_reference")
    # Explicit semantic colors keep native and exported figures consistent.
    for name, color, tags in (
        ("one", TOKENS.colors.decomposition_one, ("pyecr_fit_h_series", "pyecr_decomposition_early_one")),
        ("two", TOKENS.colors.decomposition_two, ("pyecr_fit_o_series", "pyecr_decomposition_early_two")),
        ("total", TOKENS.colors.decomposition_total, ("pyecr_decomposition_total", "pyecr_decomposition_early_total")),
        ("observed", TOKENS.colors.decomposition_observed, ("pyecr_decomposition_observed", "pyecr_decomposition_early_observed")),
        ("reference", TOKENS.colors.decomposition_reference, ("pyecr_decomposition_reference", "pyecr_decomposition_early_reference", "pyecr_decomposition_residual_reference", "pyecr_decomposition_residual_values")),
    ):
        with dpg.theme() as theme:
            for kind in (dpg.mvLineSeries, dpg.mvScatterSeries):
                with dpg.theme_component(kind):
                    for slot in (dpg.mvPlotCol_Line, dpg.mvPlotCol_MarkerFill, dpg.mvPlotCol_MarkerOutline):
                        dpg.add_theme_color(slot, color, category=dpg.mvThemeCat_Plots)
        for tag in tags:
            dpg.bind_item_theme(tag, theme)


def apply_scene(view, scene):
    dpg = view._require_dpg()
    tags = ("pyecr_fit_h_series", "pyecr_fit_o_series", "pyecr_decomposition_total",
            "pyecr_decomposition_observed", "pyecr_decomposition_reference",
            *(f"pyecr_decomposition_early_{s}" for s in ("total", "one", "two", "reference", "observed")),
            "pyecr_decomposition_residual_values", "pyecr_decomposition_residual_reference")
    if scene is None:
        for tag in tags:
            dpg.set_value(tag, [[], []])
        for suffix in ("early", "residual"):
            dpg.configure_item(f"pyecr_decomposition_{suffix}_plot", show=False)
        dpg.set_value("pyecr_decomposition_formula", "Requires a current joint dual fit.")
        dpg.configure_item("pyecr_decomposition_amplitude_note", show=False)
        return
    opts = scene.options
    for name, value in (("mode", opts.mode), ("early", opts.show_early_inset), ("residuals", opts.show_residuals),
                        ("amplitudes", opts.show_amplitudes), ("fraction", opts.early_fraction)):
        dpg.set_value(f"pyecr_decomposition_{name}", value)
    x = list(scene.display.time_s)
    ox = list(scene.observation.time_s)
    a, b = scene.display.contributions(scene.y0, opts.mode)
    reference = scene.y0 if opts.mode == "shifted" else 0.
    curves = (("pyecr_fit_h_series", a, scene.labels[0]), ("pyecr_fit_o_series", b, scene.labels[1]),
              ("pyecr_decomposition_total", scene.display.total, "Joint fit"))
    for tag, values, label in curves:
        role='one' if tag=='pyecr_fit_h_series' else 'two' if tag=='pyecr_fit_o_series' else 'total'
        dpg.set_value(tag, list(native_line_geometry(x,values,SERIES[role].line)))
        dpg.configure_item(tag, label=label, show=True)
    indices=marker_indices(ox,scene.observed) if getattr(view,'_display_sparse_markers',False) else range(len(ox))
    dpg.set_value("pyecr_decomposition_observed", [[ox[i] for i in indices], [scene.observed[i] for i in indices]])
    dpg.set_value("pyecr_decomposition_reference", list(native_line_geometry([x[0],x[-1]],[reference,reference],SERIES['reference'].line)))
    dpg.configure_item("pyecr_decomposition_reference", label="Common baseline y0" if opts.mode == "shifted" else "Zero")
    dpg.configure_item("pyecr_components_plot", label="Continuous signed contributions")
    dpg.configure_item("pyecr_components_y_axis", label=f"Signal / contributions ({scene.signal_unit})")
    dpg.set_value("pyecr_decomposition_formula", f"{scene.formula} | y0={scene.y0:.5g} {scene.signal_unit} | N={len(ox)} observations")
    dpg.set_value("pyecr_decomposition_amplitude_note",
                  f"Signed plateau amplitudes: B={scene.signed_amplitudes[0]:.5g}, C={scene.signed_amplitudes[1]:.5g} {scene.signal_unit}; plateau need not be observed.")
    dpg.configure_item("pyecr_decomposition_amplitude_note", show=opts.show_amplitudes)
    early = [i for i, t in enumerate(x) if t <= scene.early_end_s]
    eo = [i for i, t in enumerate(ox) if t <= scene.early_end_s]
    for name, values, label in (("total", scene.display.total, "Joint fit"), ("one", a, scene.labels[0]), ("two", b, scene.labels[1])):
        dpg.set_value(f"pyecr_decomposition_early_{name}",list(native_line_geometry([x[i] for i in early],[values[i] for i in early],SERIES[name].line)))
        dpg.configure_item(f"pyecr_decomposition_early_{name}", label=label)
    dpg.set_value("pyecr_decomposition_early_observed", [[ox[i] for i in eo], [scene.observed[i] for i in eo]])
    dpg.set_value("pyecr_decomposition_early_reference", [[ox[0], scene.early_end_s], [reference, reference]])
    dpg.set_value("pyecr_decomposition_residual_values", [ox, list(scene.residuals)])
    dpg.set_value("pyecr_decomposition_residual_reference", [[ox[0], ox[-1]], [0., 0.]])
    for suffix, visible in (("early", opts.show_early_inset), ("residual", opts.show_residuals)):
        dpg.configure_item(f"pyecr_decomposition_{suffix}_plot", show=visible)
        dpg.configure_item(f"pyecr_decomposition_{suffix}_y", label=f"{'Residual' if suffix == 'residual' else 'Signal / contributions'} ({scene.signal_unit})")
        dpg.fit_axis_data(f"pyecr_decomposition_{suffix}_x")
        dpg.fit_axis_data(f"pyecr_decomposition_{suffix}_y")
    dpg.set_axis_limits("pyecr_decomposition_early_x", ox[0], scene.early_end_s)
    dpg.fit_axis_data("pyecr_components_x_axis")
    dpg.fit_axis_data("pyecr_components_y_axis")


def options_from_widgets(view):
    dpg = view._require_dpg()
    return replace(view.controller.decomposition_view_options,
        mode=dpg.get_value("pyecr_decomposition_mode"),
        show_early_inset=dpg.get_value("pyecr_decomposition_early"),
        show_residuals=dpg.get_value("pyecr_decomposition_residuals"),
        show_amplitudes=dpg.get_value("pyecr_decomposition_amplitudes"),
        early_fraction=float(dpg.get_value("pyecr_decomposition_fraction")))
