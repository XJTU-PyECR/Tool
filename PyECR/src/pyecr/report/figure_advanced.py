"""Optional saved uncertainty/Transport/Simulation-fit evidence, never inference."""
import numpy as np
from pyecr.report.figure_sources import S,parameter_unit
from pyecr.report.figure_catalog import safe_id

def advanced(b,snap):
    e=snap.scientific_evidence
    def unit(name):return parameter_unit(name,snap.analysis_config,str(snap.dataset.metadata.get('signal_unit','arbitrary')))
    # Display only captured endpoints at their captured C. In particular, do
    # not call the local interval service, recompute covariance or rescue gate.
    output=e.get('selected_output',{})
    for name,interval in output.get('intervals',{}).items():
        lower,upper=interval.get('lower'),interval.get('upper')
        if lower is None or upper is None or not np.isfinite((lower,upper)).all():continue
        estimate=interval.get('estimate',snap.fit_result.params.get(name));key='analysis.interval.'+safe_id(name)
        series=[S('Captured interval',(lower,upper),(0.,0.),key+'_endpoints',sampling='stored_interval_endpoints')]
        if estimate is not None and np.isfinite(estimate):series.append(S('Captured estimate',(estimate,),(0.,),key+'_estimate','scatter','stored_estimate'))
        confidence=interval.get('confidence_level',output.get('confidence_level'))
        b.add(key,'Stored interval — '+name+(f' (C={100*confidence:g}%)' if confidence is not None else ''),series,
            x=name+' ('+unit(name)+')',xu=unit(name),y='Captured interval / estimate',yu='index')
    if not e:
        b.missing('analysis.uncertainty','Saved uncertainty','No versioned advanced uncertainty evidence in this snapshot.')
    for method,p in e.get('advanced',{}).items():
        from pyecr.services.advanced_intervals import advanced_from_dict
        from pyecr.report.scientific_evidence import plain
        value=advanced_from_dict(plain(p));r=value.result
        if r is not None:
            for name,trace in r.profiles.items():
                key='analysis.profile.'+safe_id(name)
                b.add(key,'Stored profile — '+name+f' (C={100*r.confidence_level:g}%)',(S('Likelihood ratio',trace.values,trace.likelihood_ratio,key,sampling='stored_profile_scan'),
                    S('Captured cutoff',(),(trace.cutoff,),key+'_cutoff','hline','reference')),
                    x=name+' ('+unit(name)+')',xu=unit(name),y='Likelihood ratio (captured C)',yu='1')
            if r.physical_samples is not None:
                for i,name in enumerate(r.parameter_names):
                    key='analysis.bootstrap.'+safe_id(name)
                    b.add(key,'Stored bootstrap samples — '+name,(S('Successful saved samples',range(len(r.physical_samples)),r.physical_samples[:,i],key,'scatter','stored_bootstrap_samples'),),
                        x='Successful sample index',xu='index',y=name+' ('+unit(name)+')',yu=unit(name))
        elif value.experimental_result is not None:
            b.missing('analysis.experimental_m1','Experimental uncertainty','Complete experimental method evidence is in source.json; no validated general figure schema for this method.')
    # Transport has no pre-existing native chart. Present only recorded values,
    # no new regime or covariance calculation during export.
    tr=e.get('transport')
    if tr:
        series=[S(a['process']+' / '+a['axis'],(i,),(a['value'],),'transport_'+str(i),'scatter') for i,a in enumerate(tr['axes']) if a['value'] is not None]
        b.add('analysis.transport','Captured Transport Bi',series,x='Captured process/axis index (see source.json)',xu='index',y='Bi = k L / (2D)',yu='1',reason=None if series else 'No applicable captured Bi values.')
    else:b.missing('analysis.transport','Transport','No captured Transport summary. Export does not calculate it.')
    origin=snap.dataset.metadata.get('simulation_origin')
    if origin:
        from pyecr.report.simulation_fit import simulation_comparison
        comparison=simulation_comparison(snap)
        if comparison and not comparison['direct_comparison_reasons']:
            from pyecr.services.simulation_transfer import validate_simulation_origin
            r=validate_simulation_origin(snap.dataset);t=r.request.time_s;u=r.request.signal_unit
            b.add('simulation.given_estimated','Given truth / actual fitted estimate',
                (S('Given clean truth',t,r.signal_clean,'simulation_given'),S('Actual physical Fit',t,snap.fit_result.y_fit,'simulation_estimated')),
                y='Signal ('+u+')',yu=u)
        else:b.missing('simulation.given_estimated','Given / estimated comparison','; '.join(comparison['direct_comparison_reasons']) if comparison else 'No compatible simulation origin.')
