"""Typed adapters for captured results. Numerical services are never invoked."""
from dataclasses import replace
import numpy as np
from pyecr.report.figure_catalog import FigureDescriptor as FD,FigureSeries as FS,freeze,thaw,safe_id

def validate_source(source):
    p=thaw(source.payload)
    if not isinstance(p,dict) or 'result' not in p:raise ValueError('Figure source needs a captured result')
    from pyecr.report.snapshot import AnalysisSnapshot
    from pyecr.services.batch_analysis import BatchAnalysisResult,GlobalAnalysisResult,BatchDatasetInput
    from pyecr.services.pair_design_state import PairDesignState
    from pyecr.services.dct_service import DctPresentation
    from pyecr.project.dct_archive import DctArchive
    from pyecr.contracts import SimulationResult
    types={t.__name__:t for t in (AnalysisSnapshot,BatchAnalysisResult,GlobalAnalysisResult,PairDesignState,DctPresentation,DctArchive,SimulationResult)}
    expected={'Analysis':('AnalysisSnapshot',),'Batch':('BatchAnalysisResult',),'Global':('GlobalAnalysisResult',),'Pair':('PairDesignState',),'DCT':('DctPresentation','DctArchive'),'Simulation':('SimulationResult',)}
    if source.result_type not in expected[source.module]:raise ValueError('Result type/module conflict')
    result=types[source.result_type].from_dict(p['result'])
    if source.module in ('Batch','Global'):
        from pyecr.report.snapshot import _dataset_from_dict
        from pyecr.config import AnalysisConfig
        entries=tuple(BatchDatasetInput(**{**e,'dataset':_dataset_from_dict(e['dataset']),
            'analysis_config':None if e['analysis_config'] is None else AnalysisConfig.from_dict(e['analysis_config'])}) for e in p['entries'])
        if len({e.record_id for e in entries})!=len(entries):raise ValueError('Duplicate captured Batch source')
        if p['template'] is not None:AnalysisConfig.from_dict(p['template'])
    if source.module=='DCT' and p.get('interpretations') is not None:
        from pyecr.project.interpretation_archive import validate_interpretations
        validate_interpretations(p['interpretations'],result if source.result_type=='DctArchive' else None,result if source.result_type=='DctPresentation' else None)
    return result

class Builder:
    def __init__(self,source):self.source=source;self.items=[]
    def add(self,key,title,series=(),*,plot='',x='Time (s)',y='Signal',xu='s',yu='1',reason=None,**kwargs):
        obj=FD(key,self.source.module,self.source.result_type,self.source.source_id,title,plot,x,y,xu,yu,
            tuple(series),self.source.view_state,unavailable_reason=reason,**kwargs)
        if any(o.figure_id==key for o in self.items):raise ValueError('Duplicate figure_id '+key)
        self.items.append(obj);return obj
    def missing(self,key,title,reason):return self.add(key,title,reason=reason)

def S(label,x,y,tag='',kind='line',sampling='scientific',color=None):return FS(label,tuple(x),tuple(y),tag,kind,sampling,color)
def parameter_unit(name,config=None,signal_unit=None):
    from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
    units={p.unit for m in DEFAULT_MODEL_REGISTRY.descriptors.values() for p in m.parameters if p.name==name}
    unit=config.parameter_units.get(name,'signal') if config is not None else next(iter(units)) if len(units)==1 else 'captured parameter unit'
    return (signal_unit or 'source signal unit unavailable') if unit=='signal' else unit

def describe_source(source):
    result=validate_source(source);b=Builder(source)
    if source.module=='Analysis':analysis(b,result)
    elif source.module in ('Batch','Global'):
        from pyecr.report.figure_batch_pair import batch
        batch(b,result,thaw(source.payload))
    elif source.module=='Pair':
        from pyecr.report.figure_batch_pair import pair
        pair(b,result)
    elif source.module=='DCT':
        from pyecr.report.figure_dct_simulation import dct
        dct(b,result,thaw(source.payload).get('interpretations'))
    elif source.module=='Simulation':
        from pyecr.report.figure_dct_simulation import simulation
        simulation(b,result)
    if source.result_type=='DctArchive':
        from .capability_evidence import component_records
        for record in component_records(result):
            prefix='dct.'+safe_id(record['job_id'])+'.'
            b.items=[replace(f,payload=freeze(dict(eligibility=record['eligibility'])))
                     if f.figure_id.startswith(prefix) else f for f in b.items]
    return b.items

def analysis(b,snap):
    from pyecr.plotting.labels import signal_axis_label
    d,f=snap.dataset,snap.fit_result;t=d.processed_time;u=str(d.metadata.get('signal_unit','arbitrary'))
    raw=[S('Raw data',d.raw_time,d.raw_signal,'pyecr_data_raw_series')]
    reasons=snap.export_context.get('row_reasons',{})
    for reason,title in (('mask','Masked original rows'),('crop','Cropped original rows'),('downsample','Downsampled original rows')):
        indices=[int(i) for i,v in reasons.items() if (v['reason'] if v['reason'] in ('crop','downsample') else 'mask')==reason]
        if indices:raw.append(S(title,d.raw_time[indices],d.raw_signal[indices],'pyecr_raw_excluded_'+reason,'scatter'))
    b.add('analysis.raw','Raw measurement',raw,plot='pyecr_data_raw_plot',x='Raw time ('+str(d.metadata.get('raw_time_unit','s'))+')',y=signal_axis_label(d.metadata,raw=True),xu=str(d.metadata.get('raw_time_unit','s')),yu=str(d.metadata.get('raw_signal_unit',u)))
    b.add('analysis.processed','Processed observations',(S('Processed observations',t,d.processed_signal,'pyecr_data_processed_series',kind='scatter'),),plot='pyecr_data_processed_plot',y=signal_axis_label(d.metadata),yu=u)
    b.add('analysis.fit','Captured fit',(S('Observed',t,d.processed_signal,'pyecr_fit_data_series',kind='scatter'),S('Fit',t,f.y_fit,'pyecr_fit_curve_series')),plot='pyecr_fit_plot',y=signal_axis_label(d.metadata),yu=u)
    b.add('analysis.residuals','Observed - fitted',(S('Residual',t,f.residuals,'pyecr_residual_series',kind='scatter'),S('Zero',(),(0,),'pyecr_residual_zero_series',kind='hline',sampling='reference')),plot='pyecr_residual_plot',y='Residual ('+u+')',yu=u)
    decomposition(b,snap,u)
    sensitivity=snap.diagnostics_result.get('sensitivity',{})
    curves=sensitivity.get('curves',())
    normalized_view=thaw(b.source.display_options).get('sensitivity_normalized')
    series=[S(c.get('label',c['parameter']),t,c['values'],'pyecr_sensitivity_series_'+str(i)) for i,c in enumerate(curves) if len(c.get('values',()))==len(t)]
    b.add('analysis.sensitivity','Stored sensitivity curves',series,plot='pyecr_sensitivity_plot',y='Sensitivity ('+u+'/decade)',yu=u+'/decade',use_view_ranges=normalized_view is False,reason=None if series else 'No aligned stored sensitivity curves. Run Diagnostics explicitly to create them.')
    normalized=[S(c.get('label',c['parameter']),t,c['normalized_values'],'pyecr_sensitivity_series_'+str(i)) for i,c in enumerate(curves) if len(c.get('normalized_values',()))==len(t)]
    b.add('analysis.sensitivity_normalized','Stored normalized sensitivities',normalized,plot='pyecr_sensitivity_plot',y='Normalized sensitivity (unit peak)',yu='1',use_view_ranges=normalized_view is True,reason=None if normalized else 'No stored normalized sensitivity curves.')
    from pyecr.plotting.map_display import snapshot_map_options
    maps=[]
    if snap.error_map_result is not None:maps.append(('single',snap.error_map_result))
    if snap.dual_error_map_result is not None:maps.extend((p,pmap) for p,pmap in (('H',snap.dual_error_map_result.h_map),('O',snap.dual_error_map_result.o_map)))
    for name,result in maps:
        options=snapshot_map_options(snap,'single' if name=='single' else 'dual')
        from pyecr.plotting.map_display import MapViewOptions
        options=options or MapViewOptions()
        from pyecr.plotting.error_map import error_map_plot_data,ErrorMapMetric
        from pyecr.plotting.dual_error_map import dual_map_plot_data,DualMapMetric
        data=error_map_plot_data(result,ErrorMapMetric(options.metric)) if name=='single' else dual_map_plot_data(result,DualMapMetric(options.metric))
        reason=None if data.display_valid_mask.any() else 'No finite cells for this metric; select SSE explicitly.'
        b.add('error_map.'+name,'D-k objective map '+name,plot='pyecr_error_map_plot' if name=='single' else 'pyecr_dual_map_'+name.lower()+'_plot',
            x='D (cm²/s; log10 coordinates)',y='k (cm/s; log10 coordinates)',xu='cm²/s',yu='cm/s',renderer='objective_map',reason=reason,
            payload=freeze(dict(process=name,result=result.to_dict(),options=options.to_dict(),signal_unit=u)))
    if not maps:b.missing('error_map.unavailable','Error maps','No captured current objective grid; export does not calculate one.')
    from pyecr.report.figure_advanced import advanced
    advanced(b,snap)
    if snap.dct_evidence:
        from pyecr.project.dct_archive import DctArchive
        from pyecr.report.figure_dct_simulation import dct
        from pyecr.report.scientific_evidence import plain
        e=plain(snap.dct_evidence)
        if e.get('archive') is not None:dct(b,DctArchive.from_dict(e['archive']),e.get('interpretations'),prefix='captured_')

def decomposition(b,snap,unit):
    """Use persisted signed observation contributions, including the M8 contract.

    A dense model display grid is deliberately not regenerated during export.
    The baseline is recovered algebraically only after full-array parity checks.
    """
    t=snap.dataset.processed_time;f=snap.fit_result
    record=f.diagnostics.get('dual_surface_k',{})
    parts=(record.get('component_1',()),record.get('component_2',())) if record else (snap.component_H,snap.component_O)
    labels=record.get('component_labels',('Effective H-slot process','Effective O-slot process'))
    if any(len(p)!=len(t) or not len(p) for p in parts):
        b.missing('analysis.components','Captured components','Complete signed contribution arrays were not captured in this result.');return
    a,c=map(np.asarray,parts);baseline=float(record.get('component_baseline',f.y_fit[0]-a[0]-c[0]))
    if not np.allclose(f.y_fit,baseline+a+c,rtol=2e-12,atol=2e-12*max(1.,float(np.max(np.abs(f.y_fit))))):
        raise ValueError('Captured signed contributions do not reconstruct their parent Fit')
    from pyecr.plotting.decomposition import DecompositionViewOptions
    options=DecompositionViewOptions.from_dict(snap.decomposition_view) if snap.decomposition_view else DecompositionViewOptions()
    shifted=options.mode=='shifted';reference=baseline if shifted else 0.
    series=[S(labels[0],t,a+reference,'pyecr_fit_h_series'),S(labels[1],t,c+reference,'pyecr_fit_o_series'),
        S('Joint fit',t,f.y_fit,'pyecr_decomposition_total'),S('Observed',t,snap.dataset.processed_signal,'pyecr_decomposition_observed','scatter'),
        S('Common baseline y0' if shifted else 'Zero',(),(reference,),'pyecr_decomposition_reference','hline','reference')]
    b.add('analysis.components','Captured signed contributions ('+options.mode+')',series,plot='pyecr_components_plot',y='Signal / contributions ('+unit+')',yu=unit)
    # Full source rows remain in CSV. The early-time figure is a view, not a
    # shortened scientific dataset or an interpolated residual calculation.
    from pyecr.plotting.view_state import decode_views,encode_views,freeze_views,PlotViewState
    views=decode_views(b.source.view_state);plot='pyecr_decomposition_early_plot'
    if plot not in views:views[plot]=PlotViewState(x_range=(float(t[0]),float(t[0]+options.early_fraction*(t[-1]-t[0]))))
    early=[replace(s,series_id='pyecr_decomposition_early_'+name) for s,name in zip(series,('one','two','total','observed','reference'),strict=True)]
    item=b.add('analysis.components_early','Captured contributions — early-time view',early,plot=plot,y='Signal / contributions ('+unit+')',yu=unit)
    # Deriving this display-only early window must not upgrade the frozen wire
    # of an existing package: its catalog/hash includes the original v1 view.
    version=thaw(b.source.view_state)['version'] if b.source.view_state is not None else 1
    b.items[-1]=replace(item,view_state=freeze_views(encode_views(views,version=version)))
    b.add('analysis.components_residuals','Joint decomposition residuals',(S('Observed - joint fit',t,f.residuals,'pyecr_decomposition_residual_values','scatter'),
        S('Zero',(),(0.,),'pyecr_decomposition_residual_reference','hline','reference')),plot='pyecr_decomposition_residual_plot',y='Residual ('+unit+')',yu=unit)
