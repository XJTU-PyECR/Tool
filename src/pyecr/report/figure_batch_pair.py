"""Figures for original Batch/Global/Pair types, using stored values only."""
import csv,io
import numpy as np
from pyecr.report.figure_sources import S,parameter_unit
from pyecr.report.figure_catalog import safe_id
from pyecr.report.snapshot import _dataset_from_dict

def batch(b,result,payload):
    independent=b.source.module=='Batch';prefix='batch' if independent else 'global'
    entries={(e.record_id if hasattr(e,'dataset') else e['record_id']):e for e in payload['entries']}
    from pyecr.config import AnalysisConfig
    template=payload['template'] if isinstance(payload['template'],AnalysisConfig) else None if payload['template'] is None else AnalysisConfig.from_dict(payload['template'])
    signal_units={str((e.dataset.metadata if hasattr(e,'dataset') else e['dataset']['metadata']).get('signal_unit','arbitrary')) for e in entries.values()}
    signal_unit=next(iter(signal_units)) if len(signal_units)==1 else None
    records=result.records if independent else result.curves
    for row in records:
        name=row.record_id if independent else row.curve_id;key=prefix+'.'+safe_id(name)
        fit=row.fit_result if independent else row
        entry=entries.get(name);reason=None
        if fit is None:reason=row.error or 'No successful captured Fit for this record.'
        elif entry is None:reason='Original captured observations/time/units are missing in this historical result.'
        if reason:
            for what in ('curve','residuals'):b.missing(key+'.'+what,name+' — '+what,reason)
            continue
        d=entry.dataset if hasattr(entry,'dataset') else _dataset_from_dict(entry['dataset']);t=d.processed_time;y=d.processed_signal
        fy=np.asarray(fit.y_fit);res=np.asarray(fit.residuals)
        if len(t)!=len(fy) or not np.array_equal(y-fy,res):
            raise ValueError('Captured Batch observations do not belong to result: '+name)
        u=str(d.metadata.get('signal_unit','arbitrary'))
        b.add(key+'.curve',name+' — captured fit',(S('Observed',t,y,key+'_data','scatter'),S('Fit',t,fy,key+'_fit')),plot=key+'.curve',y='Signal ('+u+')',yu=u)
        b.add(key+'.residuals',name+' — residuals',(S('Observed - fitted',t,res,key+'_residual','scatter'),),plot=key+'.residuals',y='Residual ('+u+')',yu=u)
        if not independent:
            b.add(key+'.weighted_residuals',name+' — weighted residuals',(S('Weighted residual',t,row.weighted_residuals,key+'_weighted'),),plot=key+'.weighted_residuals',y='Weighted residual ('+u+' × captured weight)',yu=u+' × weight')
    if independent:
        names=sorted({p for r in result.records for p in r.params})
        for name in names:
            rows=[r for r in result.records if r.fit_success and name in r.params]
            u=parameter_unit(name,template,signal_unit);k='batch.parameters.'+safe_id(name)
            b.add(k,'Stored '+name,[S('Successful captured records',[r.temperature_K for r in rows],[r.params[name] for r in rows],k+'_series','scatter')],plot=k,x='Temperature (K)',y=name+' ('+u+')',xu='K',yu=u,
                reason='Original signal units are unavailable or differ across records.' if u=='source signal unit unavailable' else None)
        table=list(csv.DictReader(io.StringIO(result.reproducible_csv)))
        for key,a in result.arrhenius.items():
            def yes(value):return str(value).strip().lower() in ('true','1')
            rows=[r for r in table if r['model_id']==a.model_id and yes(r['reliable']) and yes(r[a.parameter+'_reliable'])
                  and r.get(a.parameter) and np.isfinite(float(r[a.parameter])) and float(r[a.parameter])>0]
            if len(rows)!=a.n_points:raise ValueError('Stored Arrhenius inclusion count conflicts with result')
            x=np.asarray([1/float(r['temperature_K']) for r in rows]);y=np.log([float(r[a.parameter]) for r in rows]);order=np.argsort(x)
            # Evaluate the saved straight line, never call regression/optimizer.
            line=a.intercept+a.slope*x[order];fid='batch.arrhenius.'+safe_id(key)
            b.add(fid,'Stored Arrhenius — '+a.parameter,(S('Original eligible points',x,y,fid+'_points','scatter'),S('Saved regression',x[order],line,fid+'_line',sampling='stored_regression')),
                plot=fid,x='1 / T (K⁻¹)',y='ln('+a.parameter+' / '+parameter_unit(a.parameter)+')',xu='K⁻¹',yu='ln(original parameter unit)')
        if not result.arrhenius:b.missing('batch.arrhenius.unavailable','Arrhenius','No completed eligible Arrhenius regression was captured.')
    else:
        for name,value in result.shared_params.items():
            u=parameter_unit(name,template,signal_unit);fid='global.shared.'+safe_id(name)
            b.add(fid,'Shared '+name,(S(name,(0,),(value,),fid+'_series','scatter'),),plot=fid,x='Shared result',y=name+' ('+u+')',xu='shared result index',yu=u)
        for field,title in (('global_covariance','Global covariance'),('global_correlation','Global correlation')):
            # A matrix is retained in JSON; don't imply a native chart existed.
            matrix=np.asarray(getattr(result,field));series=[S('Row '+str(i),range(len(row)),row,prefix+'_'+field+str(i)) for i,row in enumerate(matrix)]
            b.add('global.'+field,title,series,x='Parameter column index (order in source.json)',y='Captured '+field,xu='index',yu='mixed parameter covariance' if field.endswith('covariance') else '1')

def pair(b,state):
    from pyecr.services.paired_ecr import _unit_scale
    if state.comparison is None or not state.comparison.ready:
        b.missing('pair.comparison','Pair comparison','No completed compatible comparison was captured.')
    else:
        curves=[];residuals=[]
        for slot,member in (('a',state.member_a),('b',state.member_b)):
            snap=member.snapshot;factor,unit=_unit_scale(snap);t=snap.dataset.processed_time;fit=snap.fit_result
            curves.append(S(member.label,t,np.asarray(fit.y_fit)*factor,'pyecr_pair_fit_'+slot))
            residuals.append(S(member.label,t,np.asarray(fit.residuals)*factor,'pyecr_pair_residual_'+slot))
        b.add('pair.curves','Independent A / B fits',curves,plot='pyecr_pair_fit_plot',y='Signal ('+unit+')',yu=unit)
        b.add('pair.residuals','Independent A / B residuals',residuals,plot='pyecr_pair_residual_plot',y='Residual ('+unit+')',yu=unit)
        for p in state.comparison.parameters:
            fid='pair.parameter.'+safe_id(p.parameter)
            label=state.comparison.parameter_label(p.parameter)
            b.add(fid,label+' — A / B',(S('A / B',(0,1),(p.value_a,p.value_b),fid,'scatter'),),x='A=0; B=1',y=label+' ('+p.unit+')',xu='member index',yu=p.unit)
        early=state.comparison.early_time_mismatch
        b.add('pair.early_mismatch','Stored early-time mismatch',() if not early.available else
            (S('Fit B - A',early.time_s,early.fit_progress_difference_b_minus_a,'pair_early_fit',sampling='stored_comparison_grid'),
             S('Observed B - A',early.time_s,early.observed_progress_difference_b_minus_a,'pair_early_observed',sampling='stored_comparison_grid')),
             y='Normalized progress difference',yu='1',reason=None if early.available else early.reason or 'No captured common early-time comparison grid.')
    if not state.plans:b.missing('pair.design.unavailable','Experiment Design','No completed planning scan; export does not run one.')
    for n,plan in enumerate(state.plans):
        key='pair.design.'+str(n+1)
        b.add(key+'.candidates','Stored design candidates',(S('Supported scenario fraction',range(len(plan.candidates)),[c.supported_fraction for c in plan.candidates],key+'_fraction','scatter'),),
            x='Candidate index (order in source.json)',y='Supported fraction',xu='index',yu='1')
        for c in plan.candidates:
            for i,scenario in enumerate(c.scenarios):
                fid=key+'.'+safe_id(c.candidate_id)+'.scenario_'+str(i+1)
                curves=[S(curve.label,c.time_s,curve.values,'pyecr_design_sensitivity_'+('d' if 'D' in curve.parameter else 'k'),sampling='stored_design_grid') for curve in scenario.sensitivity.curves]
                b.add(fid+'.sensitivity','Design '+c.candidate_id+' / scenario '+str(i+1),curves,plot='pyecr_design_sensitivity_plot',
                    y='Sensitivity (signal / decade)',yu='signal/decade',reason=None if curves else '; '.join(scenario.reasons) or 'No saved sensitivity curves for this scenario.')
