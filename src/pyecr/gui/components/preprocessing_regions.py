"""Plot reviewed intervals and mask changes using original-row identities."""
import numpy as np

from pyecr.gui.preprocessing_plots import window_rows, window_is_current


class PreprocessingRegionsMixin:
    def _build_preprocess_regions(self):
        d=self._require_dpg()
        # Inserting a fill before the data must not shift its palette identity.
        if not d.get_item_info('pyecr_data_processed_series').get('theme'):
            with d.theme() as data_theme:
                with d.theme_component(d.mvLineSeries):
                    d.add_theme_color(d.mvPlotCol_Line,[x*255 for x in d.get_colormap_color(d.mvPlotColormap_Deep,0)],category=d.mvThemeCat_Plots)
            d.bind_item_theme('pyecr_data_processed_series',data_theme)
        # Keep the selection fill behind the existing data and line overlays.
        d.add_shade_series([],[],y2=[],label='',parent='pyecr_data_processed_y_axis',
            before='pyecr_data_processed_series',tag='pyecr_preprocess_window_band',show=False)
        for kind in ('initial','final'):
            d.add_scatter_series([],[],label=kind.title()+' accepted',parent='pyecr_data_processed_y_axis',
                tag='pyecr_preprocess_accepted_'+kind,show=False)
        d.add_scatter_series([],[],label='Window preview',parent='pyecr_data_processed_y_axis',
            tag='pyecr_preprocess_window_points',show=False)
        for prefix,parent in (('pyecr_preprocess','pyecr_preprocess_raw_y_axis'),('pyecr_data','pyecr_data_raw_y_axis')):
            for kind,label in (('window','Window preview'),('excluded','Exclude preview'),('restored','Restore preview')):
                d.add_scatter_series([],[],label=label,parent=parent,tag=prefix+'_pending_'+kind,show=False)
        with d.theme() as fill:
            with d.theme_component(d.mvShadeSeries):
                d.add_theme_color(d.mvPlotCol_Fill,(245,170,65,45),category=d.mvThemeCat_Plots)
        d.bind_item_theme('pyecr_preprocess_window_band',fill)
        with d.theme() as accent:
            for kind in (d.mvScatterSeries,d.mvLineSeries):
                with d.theme_component(kind):
                    for color in (d.mvPlotCol_Line,d.mvPlotCol_MarkerFill,d.mvPlotCol_MarkerOutline):
                        d.add_theme_color(color,(245,175,70,255),category=d.mvThemeCat_Plots)
        for tag in ('pyecr_preprocess_window_points','pyecr_preprocess_pending_window','pyecr_data_pending_window',
                    'pyecr_data_stability_start_series','pyecr_data_stability_end_series'):
            d.bind_item_theme(tag,accent)
        for tag in ('pyecr_data_stability_start_series','pyecr_data_stability_end_series'):
            d.configure_item(tag,label='')

    def _plot_preview_candidate(self):
        if getattr(self,'_direct_preprocess_ready',False):
            if self._direct_binding_current():
                b=self._direct_preview_binding
                return b['step'],b['proposed'],b['dataset']
            return None
        pending=getattr(self,'_pending_typed_preview',None)
        pipeline=self.controller.session.preprocessing
        if pending and pipeline and pending[2] is pipeline.dataset and pending[3]==pipeline.revision:
            return pending[0].to_dict(),pending[1],pending[2]
        return None

    def _render_preprocess_regions(self, d, pipeline):
        if not d.does_item_exist('pyecr_preprocess_window_points'):return
        candidate=self._plot_preview_candidate()
        dataset=None if pipeline is None else pipeline.dataset
        key=(id(dataset),None if candidate is None else id(candidate[1]))
        if key==getattr(self,'_preprocess_region_key',None):return
        self._preprocess_region_key=key
        def series(tag,x=(),y=()):
            d.set_value(tag,[list(x),list(y)]);d.configure_item(tag,show=bool(len(x)))
        for tag in ('pyecr_preprocess_window_points','pyecr_data_stability_start_series','pyecr_data_stability_end_series','pyecr_data_t0_series'):
            series(tag)
        d.set_value('pyecr_preprocess_window_band',[[],[],[]]);d.configure_item('pyecr_preprocess_window_band',show=False)
        for prefix in ('pyecr_preprocess','pyecr_data'):
            for kind in ('window','excluded','restored'):series(prefix+'_pending_'+kind)
        for kind in ('initial','final'):
            tag='pyecr_preprocess_accepted_'+kind
            window={} if dataset is None else dataset.metadata.get('stability_windows',{}).get(kind,{})
            if window:
                selected=window_rows(dataset,window)
                series(tag,dataset.processed_time[selected],dataset.processed_signal[selected])
                d.configure_item(tag,label=kind.title()+(' accepted' if window_is_current(dataset,kind,window) else ' recorded · stale'))
            else:series(tag)
        if candidate is None:return
        step,proposed,current=candidate
        def raw_series(kind,indices):
            for prefix in ('pyecr_preprocess','pyecr_data'):
                series(prefix+'_pending_'+kind,current.raw_time[indices],current.raw_signal[indices])
        raw_series('excluded',np.flatnonzero(current.mask & ~proposed.mask))
        raw_series('restored',np.flatnonzero(~current.mask & proposed.mask))
        operation=step['operation'];parameters=step['parameters']
        if operation.startswith('choose_') and operation.endswith('_stability_window'):
            kind='initial' if 'initial' in operation else 'final'
            window=proposed.metadata['stability_windows'][kind]
            selected=window_rows(current,window)
            x,y=current.processed_time[selected],current.processed_signal[selected]
            series('pyecr_preprocess_window_points',x,y)
            d.configure_item('pyecr_preprocess_window_points',label=kind.title()+' window preview')
            raw_series('window',np.asarray(window['raw_indices'],dtype=int))
            low,high=float(current.processed_signal.min()),float(current.processed_signal.max())
            pad=max((high-low)*.05,abs(high)*1e-6,1e-9)
            start,end=float(parameters['start_time']),float(parameters['end_time'])
            for tag,time in (('pyecr_data_stability_start_series',start),('pyecr_data_stability_end_series',end)):
                series(tag,[time,time],[low-pad,high+pad])
            d.set_value('pyecr_preprocess_window_band',[[start,end],[low-pad]*2,[high+pad]*2])
            d.configure_item('pyecr_preprocess_window_band',show=True)
        elif operation=='set_manual_t0':
            values=current.processed_signal
            low,high=float(values.min()),float(values.max());pad=max((high-low)*.05,1e-9)
            series('pyecr_data_t0_series',[parameters['t0']]*2,[low-pad,high+pad])

    def _preview_region_on_panels(self,binding):
        d=self._require_dpg();step=binding['step'];op=step['operation']
        for role in ('input','candidate'):
            tag=f'pyecr_preview_{role}_window'
            d.set_value(tag,[[],[]]);d.configure_item(tag,show=False)
            if op.startswith('choose_') and op.endswith('_stability_window'):
                dataset=binding['dataset'] if role=='input' else binding['proposed']
                kind='initial' if 'initial' in op else 'final'
                selected=window_rows(dataset,binding['proposed'].metadata['stability_windows'][kind])
                d.set_value(tag,[dataset.processed_time[selected].tolist(),dataset.processed_signal[selected].tolist()])
                d.configure_item(tag,show=True,label=kind.title()+' window preview')
