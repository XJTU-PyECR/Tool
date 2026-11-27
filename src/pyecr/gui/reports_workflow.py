"""Reports selection and display freezing over the T09 export contracts."""
from dataclasses import dataclass,replace
from pathlib import Path
import re
from pyecr.report.figure_catalog import FigureSource,catalog,freeze,thaw
from pyecr.report.figure_bundle import FigureExportOptions
from pyecr.plotting.view_state import freeze_views

@dataclass(frozen=True)
class ReportChoice:
    key: str
    module: str
    label: str
    capture: object
    guard: object
    live_view: bool=True
    details: str=''

def result_versions(c):
    """Presentation identities only; unrelated drafts do not invalidate history."""
    s=c.analysis_session
    values=dict(analysis=(id(s),id(s.fit_result),s.current_revision,s.fit_stale),analysis_history=id(c.session.report_snapshot),
        independent=id(c.session.batch_result),global_result=id(c.session.global_result),pair=id(c.session.pair_design_state),monotonic=id(c.session.dct_result))
    values['global']=values.pop('global_result')
    values.update(('dct_'+j.job_id,id(j)) for service in (c.signed_dct_service,c.component_dct_service) for j in service.history)
    values.update(('simulation_'+j.job_id,id(j.result)) for j in c.simulation.history)
    return values

def result_choices(controller):
    """No capture, serialization or science while listing available results."""
    values=[];s=controller.analysis_session
    def short(value):
        value=str(value);return value if len(value)<=52 else value[:49]+'…'
    def name(snap):
        m=snap.dataset.metadata
        return short(m.get('dataset_name') or Path(str(m.get('source_path') or 'Analysis')).name)
    for key,_,capture,guard in controller.figure_export_choices():
        module='Analysis';live=True;details=''
        if key=='analysis':label='Current · '+short(s.dataset.metadata.get('dataset_name') or Path(str(s.dataset.metadata.get('source_path') or 'Analysis')).name)+' · Ready'
        elif key=='analysis_history':label='Historical snapshot · '+name(controller.session.report_snapshot);live=False
        elif key in ('independent','global'):
            module='Batch';r=controller.session.batch_result if key=='independent' else controller.session.global_result
            label=('Independent' if key=='independent' else 'Global')+' · Stored · '+str(len(r.records if key=='independent' else r.curves))+' records'
        elif key=='pair':
            module='Pair';r=controller.session.pair_design_state
            names=' / '.join(m.label for m in (r.member_a,r.member_b) if m is not None)
            label=short(names)+' · Stored · '+('Compared' if r.comparison is not None and r.comparison.ready else 'Review required')
        elif key=='monotonic':module='DCT';label='Monotonic · Historical result'
        elif key.startswith('dct_'):
            module='DCT';jobs=(*controller.signed_dct_service.history,*controller.component_dct_service.history)
            job=next(j for j in jobs if 'dct_'+j.job_id==key)
            kind='Components' if hasattr(job,'bundle') else 'Signed'
            group=controller.component_dct_service.history if kind=='Components' else controller.signed_dct_service.history
            label=kind+' '+str(next(i for i,j in enumerate(group,1) if j is job))+' · History · '+job.status.title()
        else:
            module='Simulation';run=next(r for r in controller.simulation.history if 'simulation_'+r.job_id==key)
            label=short(run.request.name)+' · History '+str(next(i for i,r in enumerate(controller.simulation.history,1) if r is run))+' · '+run.status.title()
        if key=='analysis':details=str(s.dataset.metadata.get('dataset_name',''))+'\n'+str(s.dataset.metadata.get('source_path',''))
        elif key=='analysis_history':details=str(controller.session.report_snapshot.dataset.metadata.get('dataset_name',''))+'\n'+str(controller.session.report_snapshot.dataset.metadata.get('source_path',''))
        elif key=='pair':details='\n'.join(m.label for m in (r.member_a,r.member_b) if m is not None)
        elif key in ('independent','global'):
            context=controller.batch_result_context(r)
            if context is not None:details='\n'.join(e.record_id+'\n'+str(e.source_project or e.dataset.metadata.get('source_path','')) for e in context[1])
        elif key.startswith('simulation_'):details=run.request.name+'\n'+key
        elif key.startswith('dct_'):details=key
        values.append(ReportChoice(key,module,label,capture,guard,live,details))
    return tuple(values)

def figure_preview(sources,selection=None):
    selected=None if selection is None else set(selection)
    return tuple(dict(source_id=s.source_id,module=s.module,figure_id=f.figure_id,title=f.title,
        available=f.available,reason=f.unavailable_reason,selected=f.available and (selected is None or s.source_id+'/'+f.figure_id in selected))
        for s in sources for f in catalog(s))

def short_reason(reason):
    """Concise display only; exact native reason remains in the preview/index."""
    known=(('Complete signed contribution','No saved components.'),('No aligned stored sensitivity','No saved sensitivity. Run Diagnostics in Analysis.'),
        ('No stored normalized sensitivity','No saved normalized sensitivity.'),('No versioned advanced','No saved uncertainty result.'),
        ('No captured Transport','No saved Transport result.'),('No captured current objective','No saved error map.'),
        ('Select the actual fitted Analysis','Select the actual fitted Analysis result.'),('No explicitly applied','No applied physical context.'),
        ('No GCV candidate arrays','This result used manual lambda; no GCV scan.'),('Physical restoration arrays','No saved physical restoration.'))
    for prefix,text in known:
        if str(reason).startswith(prefix):return text
    value=str(reason);return value if len(value)<=130 else value[:127]+'…'

def reviewed_warnings(sources):
    """Read frozen original warnings once at Review; never reassess qualification."""
    result=[]
    for s in sources:
        p=thaw(s.payload)['result'];messages=[]
        if s.module=='Analysis':
            diag=p.get('diagnostics_result') or {}
            messages.extend(diag.get('gate_failures',()))
            messages.extend(p.get('fit_result',{}).get('warnings',()))
        elif s.module=='DCT' and 'component_jobs' in p:
            for job in p['component_jobs']:
                bundle=job.get('bundle') or {};capture=bundle.get('component_capture') or job.get('request',{}).get('capture') or {}
                e=capture.get('eligibility') or {}
                messages.extend(e.get('warnings',()))
        if messages:result.append(dict(module=s.module,label=s.label,warnings=list(dict.fromkeys(messages))))
    return tuple(result)

def report_notice(message):
    """Default short error; exact paths/long native reasons remain in Details."""
    text=str(message)
    if len(text)>130 or re.search(r'[A-Za-z]:[\\/]|\\\\',text):return 'Report action failed. Review Details for the complete reason.'
    return text

def freeze_report(sources,*,live_views,views=None,layers=None,map_views=None,decomposition_view=None,selection=None,formats=('png',),xlsx=False,style=None):
    """Freeze click-time display; scientific dataset/config/Fit stay unchanged.

    Updating native snapshot display metadata can change the full source SHA;
    remap qualified image selections while retaining stable figure IDs.
    """
    values=[];identities={}
    for source,live in zip(sources,live_views,strict=True):
        captured=source
        if live:
            payload=source.payload
            if source.module=='Analysis' and (map_views is not None or decomposition_view is not None):
                p=thaw(payload);result=p['result'];context=result.get('export_context');meta=context.setdefault('project_metadata',{}) if context else None
                if map_views is not None and (map_views or (meta is not None and 'error_map_view' in meta)):
                    if meta is None:raise ValueError('Review a current snapshot before applying current map display settings.')
                    from pyecr.plotting.map_display import MapViewOptions
                    for value in map_views.values():MapViewOptions.from_dict(value)
                    meta['error_map_view']=map_views
                from pyecr.config import AnalysisConfig
                from pyecr.plotting.decomposition import DUAL_IDS
                if decomposition_view is not None and AnalysisConfig.from_dict(result['analysis_config']).model_id.value in DUAL_IDS and (
                    decomposition_view!=result.get('decomposition_view') or (meta is not None and 'decomposition_view' in meta)):
                    from pyecr.plotting.decomposition import DecompositionViewOptions
                    DecompositionViewOptions.from_dict(decomposition_view)
                    result['decomposition_view']=decomposition_view
                    if meta is not None:meta['decomposition_view']=decomposition_view
                payload=freeze(p)
            captured=replace(source,payload=payload,view_state=freeze_views(views),display_options=freeze(layers or {}))
        values.append(captured);identities[source.source_id]=captured.source_id
    values=tuple(values)
    ids=None if selection is None else tuple(identities[key.split('/',1)[0]]+'/'+key.split('/',1)[1] for key in selection)
    kwargs={} if style is None else dict(style=style)
    options=FigureExportOptions(ids,tuple(formats),xlsx,allow_arrays_only=True,**kwargs)
    return values,options
