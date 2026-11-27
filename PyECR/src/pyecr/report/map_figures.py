"""GUI-independent map figures from an immutable Snapshot and captured view."""
from dataclasses import replace
import json
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pyecr.plotting.map_display import MapViewOptions,snapshot_map_options,color_limits
from pyecr.plotting.dual_error_map import DualMapMetric,dual_map_plot_data
from pyecr.plotting.matplotlib import render_error_map_figure,render_dual_error_map_figure,objective_map_figure,map_fit_view_note
from pyecr.plotting.visual_style import style_publication_figure
from pyecr.report.snapshot import _sha256_json

def write_map_figures(snapshot,root,*,publication=None,check=lambda:None):
    root.mkdir(parents=True,exist_ok=True);records={}
    single=snapshot_map_options(snapshot,"single");dual=snapshot_map_options(snapshot,"dual")
    entries=[]
    if snapshot.error_map_result is not None:entries.append(("error_map",snapshot.error_map_result,single,None))
    if snapshot.dual_error_map_result is not None:
        slices=(snapshot.dual_error_map_result.h_map,snapshot.dual_error_map_result.o_map);shared=None
        if dual is not None and dual.shared_scale:
            data=[dual_map_plot_data(x,DualMapMetric(dual.metric)) for x in slices]
            values=np.concatenate([x.values[x.display_valid_mask] for x in data])
            shared=color_limits(values,np.ones(values.shape,dtype=bool),dual)
        entries.extend(("error_map_"+x.process,x,dual,shared) for x in slices)
    for name,result,view,shared in entries:
        check()
        frozen=(publication.plot_views if publication is not None else None) or snapshot.export_context.get('project_metadata',{}).get('plot_views')
        if view is None and publication is None:
            render=render_error_map_figure if name=="error_map" else render_dual_error_map_figure
            ranges={}
            render(result,root/(name+".png"),plot_views=frozen,range_metadata=ranges)
            records[name]={"captured_view":None,"rendered_view":MapViewOptions(metric=ranges.pop('metric'),image_dpi=120).to_dict(),
                "rendered_color_range":ranges.pop('color_limits'),**ranges,
                "computed_range_log10":{"D":[float(result.log10_D[0]),float(result.log10_D[-1])],"k":[float(result.log10_k[0]),float(result.log10_k[-1])]},
                "parent_dataset_sha256":snapshot.dataset_sha256,"parent_fit_sha256":_sha256_json(snapshot.fit_result.to_dict()),
                "parent_config_sha256":_sha256_json(snapshot.analysis_config.to_dict()),
                "valid_mask_source":"captured result valid_mask; exported with complete scientific grid",
                "evaluated_shape":[len(result.log10_D),len(result.log10_k)],"shared_color_range":shared,
                "science_source":"captured full-data objective cells; no objective/refit on export",
                "vector_format_note":"Legacy captured view absent: original PNG renderer retained."}
            continue
        actual=view or MapViewOptions()
        if publication is not None:actual=replace(actual,image_dpi=publication.image_dpi)
        figure=objective_map_figure(result,actual,color_range=shared)
        if publication is not None:
            figure.set_size_inches(publication.width_in,publication.height_in)
            style_publication_figure(figure,publication.font_family)
        FigureCanvasAgg(figure)
        from pyecr.plotting.view_state import apply_axis_view
        plot_id='pyecr_error_map_plot' if name=='error_map' else 'pyecr_dual_map_'+result.process.lower()+'_plot'
        prefix='pyecr_error_map_' if name=='error_map' else 'pyecr_dual_map_'+result.process.lower()+'_'
        apply_axis_view(figure.axes[0],plot_id,frozen,(prefix+'fit_point',prefix+('grid_cross_h' if name=='error_map' else 'cross_h')),
                        legend_policy=publication.legend_policy if publication is not None else 'follow')
        map_fit_view_note(figure.axes[0],result.fitted_point.log10_D,result.fitted_point.log10_k)
        for fmt in ("png","svg","pdf"):
            check()
            meta={"Software":"PyECR"} if fmt=="png" else {"Creator":"PyECR","Date":None} if fmt=="svg" else {"Creator":"PyECR","CreationDate":None,"ModDate":None}
            figure.savefig(root/(name+'.'+fmt),dpi=actual.image_dpi,metadata=meta)
        records[name]={"captured_view":None if view is None else view.to_dict(),"rendered_view":actual.to_dict(),
                       "rendered_color_range":figure._pyecr_map_color_range,
                       "computed_range_log10":{"D":[float(result.log10_D[0]),float(result.log10_D[-1])],"k":[float(result.log10_k[0]),float(result.log10_k[-1])]},
                       "display_range_log10":{"D":list(figure.axes[0].get_xlim()),"k":list(figure.axes[0].get_ylim())},
                       "parent_dataset_sha256":snapshot.dataset_sha256,
                       "parent_fit_sha256":_sha256_json(snapshot.fit_result.to_dict()),
                       "parent_config_sha256":_sha256_json(snapshot.analysis_config.to_dict()),
                       "fit_point_in_view":bool(figure.axes[0].get_xlim()[0]<=result.fitted_point.log10_D<=figure.axes[0].get_xlim()[1]
                           and figure.axes[0].get_ylim()[0]<=result.fitted_point.log10_k<=figure.axes[0].get_ylim()[1]),
                       "valid_mask_source":"captured result valid_mask; exported with complete scientific grid",
                       "evaluated_shape":[len(result.log10_D),len(result.log10_k)],"shared_color_range":shared,
                       "science_source":"captured full-data objective cells; no objective/refit on export",
                       "vector_format_note":"Cells: vector cells. Smooth: embedded raster, vector axes/text."}
    if records:
        (root.parent/'error_map_display.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
