"""Scientific figures from captured arrays, with explicit publication options."""
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pyecr.plotting.matplotlib import decomposition_figure
from pyecr.report.export_tables import json_value
from pyecr.plotting.visual_style import SERIES,PUBLICATION,publication_fonts,style_publication_figure


def write_figures(snapshot, decomposition, root, options, check=lambda: None):
    root.mkdir(parents=True, exist_ok=True)
    fonts,plan=publication_fonts(options.font_family)
    with rc_context({"font.family":fonts,"font.size":PUBLICATION.body_pt,"text.usetex": False,
                     "svg.fonttype": "path", "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        if decomposition is not None:
            figure = decomposition_figure(decomposition)
        else:
            figure = Figure(layout="constrained")
            axis, residual = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": PUBLICATION.residual_height_ratios})
            d, f = snapshot.dataset, snapshot.fit_result
            axis.scatter(d.processed_time,d.processed_signal,s=SERIES['observed'].marker_size,color=SERIES['observed'].hex,label="Observed")
            axis.plot(d.processed_time,f.y_fit,color=SERIES['total'].hex,linewidth=SERIES['total'].width,label="Captured joint fit")
            unit = str(d.metadata.get("signal_unit", "signal"))
            from pyecr.plotting.labels import signal_axis_label
            axis.set_ylabel(signal_axis_label(d.metadata))
            from pyecr.model_display import display_name
            axis.set_title(display_name(snapshot.analysis_config.model_id))
            axis.legend(frameon=False)
            residual.scatter(d.processed_time,f.residuals,color=SERIES['residual'].hex,s=SERIES['residual'].marker_size,label='Residual')
            residual.axhline(0,linewidth=SERIES['zero'].width,color=SERIES['zero'].hex,linestyle=SERIES['zero'].line,label='Zero')
            residual.legend(frameon=False)
            residual.set_ylabel(f"Residual ({unit})")
            residual.set_xlabel("Time (s)")
        figure.set_size_inches(options.width_in, options.height_in)
        style_publication_figure(figure,options.font_family)
        FigureCanvasAgg(figure)
        from pyecr.plotting.view_state import apply_axis_view
        apply_axis_view(figure.axes[0], 'pyecr_components_plot' if decomposition is not None else 'pyecr_fit_plot', options.plot_views,
                        ('pyecr_decomposition_observed','pyecr_decomposition_total','pyecr_fit_h_series','pyecr_fit_o_series','pyecr_decomposition_reference') if decomposition is not None else
                        ('pyecr_fit_data_series','pyecr_fit_curve_series','pyecr_fit_h_series','pyecr_fit_o_series'),
                        legend_policy=options.legend_policy)
        if len(figure.axes)>1:
            apply_axis_view(figure.axes[1], 'pyecr_residual_plot', options.plot_views,
                            ('pyecr_residual_series','pyecr_residual_zero_series'),legend_policy=options.legend_policy)
        for fmt in ("png", "svg", "pdf"):
            check()
            metadata = {"Software":"PyECR"} if fmt=="png" else {"Creator":"PyECR","Date":None} if fmt=="svg" else {"Creator":"PyECR","CreationDate":None,"ModDate":None}
            figure.savefig(root/f"fit.{fmt}", dpi=options.image_dpi, metadata=metadata)
        if decomposition is not None:
            (root.parent/"display_grid.json").write_text(json_value(decomposition.to_dict())+"\n", encoding="utf-8")
        from pyecr.report.map_figures import write_map_figures
        write_map_figures(snapshot,root,publication=options,check=check)
        check()
