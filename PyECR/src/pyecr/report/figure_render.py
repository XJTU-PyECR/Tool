"""Data rendering of frozen descriptors; no screen capture or scientific solve."""
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pyecr.plotting.view_state import apply_axis_view,decode_views
from pyecr.plotting.visual_style import style_publication_figure
from pyecr.report.figure_catalog import thaw

def build_figure(descriptor,style):
    d=descriptor
    if not d.available:raise ValueError(d.unavailable_reason)
    if d.renderer=='objective_map':
        from pyecr.diagnostics.error_map import ErrorMapResult
        from pyecr.diagnostics.dual_error_map import DualErrorMapSlice
        from pyecr.plotting.map_display import MapViewOptions
        from pyecr.plotting.matplotlib import objective_map_figure,map_fit_view_note
        p=thaw(d.payload);r=(ErrorMapResult if p['process']=='single' else DualErrorMapSlice).from_dict(p['result'])
        figure=objective_map_figure(r,MapViewOptions.from_dict(p['options']));axis=figure.axes[0]
        if p['options']['metric']=='SSE' and len(figure.axes)>1:figure.axes[1].set_ylabel('SSE ('+p['signal_unit']+'²)')
        prefix='pyecr_error_map_' if p['process']=='single' else 'pyecr_dual_map_'+p['process'].lower()+'_'
        ids=(prefix+'fit_point',prefix+('grid_cross_h' if p['process']=='single' else 'cross_h'))
    else:
        figure=Figure(layout='constrained');axis=figure.subplots();ids=[]
        for series in d.series:
            kw=dict(label=series.label)
            if series.color is not None:kw['color']=series.color
            if series.kind=='scatter':axis.scatter(series.x,series.y,s=12,**kw)
            elif series.kind=='hline':axis.axhline(series.y[0],linewidth=.7,**kw)
            elif series.kind=='bar':axis.bar(series.x,series.y,**kw)
            else:axis.plot(series.x,series.y,linewidth=1.4,**kw)
            ids.append(series.series_id)
        axis.set(xlabel=d.xlabel,ylabel=d.ylabel,title=d.title,xscale=d.xscale,yscale=d.yscale)
        axis.grid(True,alpha=.15);axis.legend(frameon=False,fontsize=8)
    figure.set_size_inches(style.width_in,style.height_in);FigureCanvasAgg(figure)
    eligibility=thaw(d.payload).get('eligibility')
    if eligibility is not None:
        from .capability_evidence import caption
        figure.supxlabel(caption(eligibility),fontsize=7)
    style_publication_figure(figure,style.font_family)
    apply_axis_view(axis,d.plot_id,d.view_state,ids,log_coordinates=d.view_log_coordinates,ranges=d.use_view_ranges,
                    legend_policy=style.legend_policy)
    if d.renderer=='objective_map':map_fit_view_note(axis,r.fitted_point.log10_D,r.fitted_point.log10_k)
    state=decode_views(d.view_state).get(d.plot_id)
    meta=dict(axis_limits=dict(x=list(axis.get_xlim()),y=list(axis.get_ylim())),xscale=axis.get_xscale(),yscale=axis.get_yscale(),
        series=[dict(id=s.series_id,label=s.label,sampling=s.sampling,count=len(s.y)) for s in d.series],
        frozen_view=None if state is None else state.to_dict(),x_unit=d.x_unit,y_unit=d.y_unit)
    if eligibility is not None:meta['eligibility']=eligibility
    if style.legend_policy == 'auto':
        meta['legend_placement'] = getattr(axis, '_pyecr_legend_placement', None)
    if d.renderer=='objective_map':
        meta.update(computed_range_log10=dict(D=[float(r.log10_D[0]),float(r.log10_D[-1])],k=[float(r.log10_k[0]),float(r.log10_k[-1])]),
            display_range_log10=dict(D=list(axis.get_xlim()),k=list(axis.get_ylim())),color_range=figure._pyecr_map_color_range,
            map_view=p['options'],scientific_shape=[len(r.log10_D),len(r.log10_k)],
            color_unit=p['signal_unit']+'²' if p['options']['metric']=='SSE' else '1',
            sampling='captured full objective grid; display interpolation only within computed cells' if p['options']['sampling']=='Smooth' else 'captured objective cells',
            mask='full valid_mask in source.json and map CSV; no interpolated scientific samples')
    return figure,meta

def write_figure(descriptor,root,formats,style,check=lambda:None,*,filename_stem=None):
    from pyecr.report.figure_names import validate_file_stem
    stem=validate_file_stem(descriptor.figure_id if filename_stem is None else filename_stem)
    figure,metadata=build_figure(descriptor,style)
    try:
        for fmt in formats:
            check()
            attributes={'Software':'PyECR'} if fmt=='png' else {'Creator':'PyECR','Date':None} if fmt=='svg' else {'Creator':'PyECR','CreationDate':None,'ModDate':None}
            figure.savefig(root/(stem+'.'+fmt),format=fmt,dpi=style.image_dpi,metadata=attributes)
    finally:figure.clear()
    return metadata
