"""Native Batch result plots use the same saved-array adapter as export."""
from types import SimpleNamespace
from pyecr.gui.theme import TOKENS

class BatchFiguresMixin:
    def _build_batch_figure_controls(self):
        d=self._require_dpg();self._batch_figure_key=None;self._batch_figure_plot_ids=[]
        d.add_combo((),tag='pyecr_batch_figure_choice',width=-1,show=False,callback=self._select_batch_figure)
        d.add_text('',tag='pyecr_batch_figure_notice',show=False,wrap=TOKENS.wraps.map)
        d.add_group(tag='pyecr_batch_figure_host')

    def _refresh_batch_figures(self,result,mode):
        d=self._require_dpg();context=None if result is None else self.controller.batch_result_context(result)
        persisted=self.controller.analysis_session.project_metadata.get('figure_source_contexts')
        key=(id(result),id(context),id(persisted));manager=getattr(self,'_plot_views',None)
        if key!=self._batch_figure_key:
            self._batch_figure_key=key
            for plot in self._batch_figure_plot_ids:
                if manager is not None:manager.forget(plot)
            self._batch_figure_plot_ids=[];d.delete_item('pyecr_batch_figure_host',children_only=True)
            self._batch_figures=[];self._batch_figure_to_fit=set()
            if result is not None:
                from pyecr.report.figure_sources import Builder
                from pyecr.report.figure_batch_pair import batch
                # This native-only adapter reads objects directly. It does not
                # serialize a result/config just because Details are closed.
                source=SimpleNamespace(module='Batch' if mode=='independent' else 'Global',result_type=type(result).__name__,source_id='native_result_view',view_state=None)
                payload=dict(entries=() if context is None else context[1],template=None if context is None else context[2])
                if context is None and persisted is not None:
                    from pyecr.report.figure_catalog import thaw
                    source=self.controller.capture_batch_figure_source(result,scope='historical');payload=thaw(source.payload)
                builder=Builder(source);batch(builder,result,payload)
                self._batch_figures=[f for f in builder.items if f.available and f.plot_id]
                for f in self._batch_figures:
                    with d.plot(label=f.title,tag=f.plot_id,parent='pyecr_batch_figure_host',width=-1,height=TOKENS.heights.plot,show=False):
                        d.add_plot_legend();d.add_plot_axis(d.mvXAxis,label=f.xlabel,tag=f.plot_id+'_x')
                        with d.plot_axis(d.mvYAxis,label=f.ylabel,tag=f.plot_id+'_y'):
                            xs=[v for s in f.series for v in s.x];extent=(min(xs),max(xs)) if xs else (0,1)
                            for s in f.series:
                                if s.kind=='scatter':d.add_scatter_series(s.x,s.y,label=s.label,tag=s.series_id)
                                else:d.add_line_series(extent if s.kind=='hline' else s.x,(s.y[0],)*2 if s.kind=='hline' else s.y,label=s.label,tag=s.series_id)
                    self._batch_figure_plot_ids.append(f.plot_id);self._batch_figure_to_fit.add(f.plot_id)
                if manager is not None:manager.discover()
            labels=[f.title for f in self._batch_figures]
            default=next((f.title for f in self._batch_figures if '.arrhenius.' in f.figure_id),labels[0] if labels else '')
            d.configure_item('pyecr_batch_figure_choice',items=labels,show=bool(labels));d.set_value('pyecr_batch_figure_choice',default)
            text='Original curve inputs are missing; only captured summaries can be plotted.' if result is not None and context is None and persisted is None else ''
            d.set_value('pyecr_batch_figure_notice',text);d.configure_item('pyecr_batch_figure_notice',show=bool(text));self._select_batch_figure()
        for plot in tuple(self._batch_figure_to_fit):
            if d.get_item_state(plot).get('visible'):
                for axis in ('x','y'):d.fit_axis_data(plot+'_'+axis)
                self._batch_figure_to_fit.remove(plot)

    def _select_batch_figure(self,*args):
        d=self._require_dpg();selected=d.get_value('pyecr_batch_figure_choice')
        for f in getattr(self,'_batch_figures',()):d.configure_item(f.plot_id,show=f.title==selected)
