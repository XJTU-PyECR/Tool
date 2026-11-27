"""Reuse DCT/Simulation figure builders, exposing their individual data panels."""
from dataclasses import replace
import numpy as np
from matplotlib.colors import to_hex
from pyecr.report.figure_sources import S
from pyecr.report.figure_catalog import safe_id,digest,thaw
from pyecr.report.export_options import ExportOptions
from pyecr.plotting.labels import signal_unit_axis_label

def axis_series(ax,ids=(),*,sampling='scientific'):
    values=[]
    for i,line in enumerate(ax.lines):
        label=line.get_label();tag=ids[i] if i<len(ids) else label
        if line.get_transform()==ax.get_yaxis_transform():
            values.append(S(label,(),(line.get_ydata()[0],),tag,'hline','reference',to_hex(line.get_color())))
        else:
            values.append(S(label,line.get_xdata(),line.get_ydata(),tag,'scatter' if line.get_linestyle() in ('None','none','') else 'line',sampling,to_hex(line.get_color())))
    return values

def simulation(b,result):
    if not result.state.ready:
        b.missing('simulation.unavailable','Simulation','; '.join(result.state.blocking_reasons) or 'No completed Simulation arrays were captured.')
        return
    from pyecr.report.simulation_figures import write_simulation_figures
    unit=result.request.signal_unit;t=result.request.time_s
    def emit(name,fig):
        ax,lower=fig.axes
        if name=='simulation_observations':
            b.add('simulation.overlay','Simulation clean / observed',axis_series(ax,('pyecr_sim_total_series','pyecr_sim_observed_series')),
                plot='pyecr_sim_total_plot',y=signal_unit_axis_label(unit),yu=unit)
            b.add('simulation.noise','Actual total observation noise',axis_series(lower,('pyecr_sim_noise_series','__zero')),
                plot='pyecr_sim_noise_plot',y='Noise ('+unit+')',yu=unit)
        else:
            b.add('simulation.contributions','Clean signed contributions',axis_series(lower,tuple('pyecr_sim_component_'+str(i+1) for i in range(len(result.contributions)))),
                plot='pyecr_sim_components_plot',y='Contribution ('+unit+')',yu=unit)
    write_simulation_figures(result,None,(),ExportOptions(),lambda:None,emit=emit)
    for what,values,tag in (('clean',result.signal_clean,'pyecr_sim_total_series'),('observed',result.signal_observed,'pyecr_sim_observed_series')):
        b.add('simulation.'+what,'Simulation '+what,(S('Simulation '+what,t,values,tag,'scatter' if what=='observed' else 'line'),),plot='pyecr_sim_total_plot',y=signal_unit_axis_label(unit),yu=unit)
    b.missing('simulation.given_estimated','Given / estimated comparison','Select the actual fitted Analysis source to export its given/estimated comparison. Simulation has no estimated Fit.')

def dct(b,result,interpretations=None,*,prefix=''):
    from pyecr.services.dct_service import DctPresentation
    if isinstance(result,DctPresentation):monotonic(b,result,prefix)
    else:
        from pyecr.report.dct_figures import write_dct_figures
        from pyecr.report.dct_tables import records
        jobs={key:job for key,job in records(result)}
        display=thaw(b.source.display_options);physical=display.get('dct_physical');component_view=display.get('component_view')
        for key,job in jobs.items():
            if (job.analysis if key.startswith('w') else job.bundle) is None:
                b.missing(prefix+'dct.'+safe_id(job.job_id)+'.unavailable','DCT '+job.job_id,job.failure or 'No complete captured result: '+job.status)
        def emit(name,fig):
            record=next(k for k in jobs if name.startswith(k+'_'));job=jobs[record];kind=name[len(record)+1:]
            key=prefix+'dct.'+safe_id(job.job_id)
            if kind=='whole_signed':
                labels=('spectrum','reconstruction','residuals');plots=('pyecr_dct_spectrum_plot','pyecr_dct_reconstruction_plot','pyecr_dct_residual_plot')
                ids=(('pyecr_signed_dct_positive','pyecr_signed_dct_negative','pyecr_signed_dct_zero'),('pyecr_dct_data_series','pyecr_dct_reconstruction_series'),('pyecr_dct_residual_series',))
            elif kind=='component_comparison':
                labels=('independent_shapes','common_scale','component_time','component_residuals');plots=('pyecr_dct_spectrum_plot','pyecr_dct_spectrum_plot','pyecr_dct_reconstruction_plot','pyecr_dct_residual_plot')
                cids=tuple('pyecr_component_dct_component_'+str(i) for i,c in enumerate(job.bundle.independent_results,1) if c is not None)
                ids=(cids+('__zero',),cids+('__whole','__zero'),('pyecr_dct_data_series','pyecr_component_dct_parent_fit','pyecr_component_dct_delta','pyecr_dct_reconstruction_series'),('pyecr_component_dct_model_error','pyecr_dct_residual_series'))
            else:
                labels=(kind,);plots=({'lambda':'pyecr_dct_lambda_plot','peak_stability':'pyecr_dct_stability_plot','physical_dct':key+'.physical'}[kind],);ids=((),)
            for i,(ax,label,plot,tags) in enumerate(zip(fig.axes,labels,plots,ids)):
                xunit='decade(tau/s)' if 'log10' in ax.get_xlabel() else 's' if 'Time' in ax.get_xlabel() else 'lambda penalty' if kind=='lambda' else 'captured coordinate'
                yunit='1/decade' if label in ('spectrum','independent_shapes','common_scale') else 'decade' if kind=='peak_stability' else 'weighted signal²' if kind=='lambda' else '1'
                if kind=='physical_dct':
                    a=job.analysis if record.startswith('w') else job.bundle.overall_analysis
                    yunit=a.selected_result.physical_unit or 'unavailable'
                series=axis_series(ax,tags,sampling='stored_solver_grid' if xunit=='decade(tau/s)' else 'scientific')
                if kind=='lambda' and series:
                    points=series[0];series=[replace(points,series_id='pyecr_signed_dct_lambda_series'),replace(points,series_id='pyecr_signed_dct_lambda_points',kind='scatter',label='Stored lambda candidates')]
                if kind=='peak_stability':series=[replace(s,series_id='pyecr_signed_dct_stability_series') for s in series]
                use_ranges=kind in ('lambda','peak_stability')
                if kind=='whole_signed':use_ranges=(physical is False) if record.startswith('w') else component_view=='Overall'
                if kind=='component_comparison':
                    use_ranges=label!='independent_shapes' and component_view=='Comparison'
                    if label=='common_scale':
                        # The native comparison displays the whole signed curve
                        # as positive/negative series alongside both components.
                        whole=series[-2];series=series[:-2]+[
                            replace(whole,y=tuple(max(v,0) for v in whole.y),series_id='pyecr_signed_dct_positive',label='Positive whole signed density'),
                            replace(whole,y=tuple(min(v,0) for v in whole.y),series_id='pyecr_signed_dct_negative',label='Negative whole signed density'),series[-1]]
                if kind=='physical_dct':plot='pyecr_dct_reconstruction_plot';use_ranges=physical is True and record.startswith('w')
                reason=None if any(s.kind!='hline' and s.y for s in series) else ('Physical restoration was not captured.' if kind=='physical_dct' else 'No stored finite '+label+' evidence.')
                b.add(key+'.'+label,ax.get_title() or 'DCT '+label,series,plot=plot,x=ax.get_xlabel(),y=ax.get_ylabel(),xu=xunit,yu=yunit,
                    xscale=ax.get_xscale(),yscale=ax.get_yscale(),view_log_coordinates=(ax.get_xscale()=='log',False),use_view_ranges=use_ranges,reason=reason)
        write_dct_figures(result,None,(),ExportOptions(),lambda:None,emit=emit)
        for record,job in jobs.items():
            a=job.analysis if record.startswith('w') else job.bundle.overall_analysis if job.bundle is not None else None
            if a is None or a.selected_result is None:continue
            r=a.selected_result;key=prefix+'dct.'+safe_id(job.job_id);unit=r.physical_unit
            for label,series,plot,yu,x,xu in (
                ('physical_spectrum',() if r.physical_chi is None else (S('Positive physical density',r.log10_tau,np.maximum(r.physical_chi,0),'pyecr_signed_dct_positive',sampling='stored_solver_grid'),S('Negative physical density',r.log10_tau,np.minimum(r.physical_chi,0),'pyecr_signed_dct_negative',sampling='stored_solver_grid')),'pyecr_dct_spectrum_plot',(unit or 'unavailable')+'/decade','log10(tau / s)','decade(tau/s)'),
                ('physical_residuals',() if r.physical_residuals is None else (S('Physical residual',r.source.time_s,r.physical_residuals,'pyecr_dct_residual_series','scatter'),),'pyecr_dct_residual_plot',unit or 'unavailable','Time (s)','s')):
                b.add(key+'.'+label,'DCT '+label,series,plot=plot,x=x,xu=xu,y=label+' ('+yu+')',yu=yu,use_view_ranges=physical is True and record.startswith('w'),
                    reason=None if series else 'Physical restoration arrays were not captured.')
            if not record.startswith('w') and job.bundle is not None:
                display=thaw(b.source.display_options)
                for i,c in enumerate(job.bundle.independent_results,1):
                    ck=key+'.component_'+str(i);selected=display.get('component_view')=='component_'+str(i)
                    if c is None:
                        b.missing(ck+'.unavailable','Component '+str(i),job.bundle.unavailable_reasons[i-1] or 'No captured independent component inversion.')
                        continue
                    for name,values in (('shape',c.chi),('common_scale',c.common_chi)):
                        signed=(S('Positive density',c.log10_tau,tuple(max(v,0) for v in values),'pyecr_signed_dct_positive',sampling='stored_solver_grid'),
                            S('Negative density',c.log10_tau,tuple(min(v,0) for v in values),'pyecr_signed_dct_negative',sampling='stored_solver_grid'))
                        b.add(ck+'.'+name,'Component '+str(i)+' — '+name,signed,
                            plot='pyecr_dct_spectrum_plot',x='log10(tau / s)',xu='decade(tau/s)',y='chi (1/decade)',yu='1/decade',
                            use_view_ranges=selected and (display.get('component_scale')=='Independent shape')==(name=='shape'))
                    b.add(ck+'.reconstruction','Component '+str(i)+' — independent reconstruction',
                        (S('Model-derived component shape',c.source.time_s,c.source.signal,'pyecr_dct_data_series','scatter'),S('Independent DCT',c.source.time_s,c.reconstruction,'pyecr_dct_reconstruction_series')),
                        plot='pyecr_dct_reconstruction_plot',y='Normalized component shape',yu='1',use_view_ranges=selected and display.get('component_scale')=='Independent shape')
                    b.add(ck+'.residuals','Component '+str(i)+' — independent residuals',(S('Shape - DCT',c.source.time_s,c.residuals,'pyecr_dct_residual_series','scatter'),),
                        plot='pyecr_dct_residual_plot',y='Component shape residual',yu='1',use_view_ranges=selected and display.get('component_scale')=='Independent shape')
        if not jobs:b.missing(prefix+'dct.unavailable','DCT','No selected completed DCT run.')
    if interpretations:
        # Only explicit applied records are stored in this R13 archive. Review
        # drafts and a currently different Analysis never become background.
        for row in interpretations['history']:
            m=row['modal_reference'];processes=m['processes'];key=prefix+'dct.context.'+row.get('target_sha256',digest(row))[:16]+'.'+digest(row)[:10]
            masses=[];forward=[]
            for i,p in enumerate(processes):
                masses.append(S(p['label'],[v['log10_tau'] for v in p['modes']],[v['mass'] for v in p['modes']],f'pyecr_dct_context_modes_{i}','scatter','conditional_modal_mass'))
                for tag,field in (('full','production_remaining'),('partial','truncated_remaining')):
                    forward.append(S(tag+' '+p['label'],m['time_s'],p[field],f'pyecr_dct_context_{tag}_{i}',sampling='captured_conditional_forward'))
            for label,series,plot,x,xu,y,yu in (('modes',masses,'pyecr_dct_context_modes_plot','log10(tau / s)','decade(tau/s)','Discrete mass (not chi height)','1'),
                ('forward',forward,'pyecr_dct_context_forward_plot','Time (s)','s','Remaining fraction','1')):
                b.add(key+'.'+label,'Applied physical context — '+label,series,plot=plot,x=x,xu=xu,y=y,yu=yu,reason=None if series else '; '.join(m['reasons']) or 'No captured modal reference for the applied context.')
    else:b.missing(prefix+'dct.context.unavailable','Applied physical context','No explicitly applied physical context was captured.')

def monotonic(b,p,prefix):
    r=p.result;key=prefix+'dct.monotonic'
    for label,series,plot,x,xu,y,yu in (
        ('spectrum',(S('Nonnegative density',np.log10(r.tau_s),r.chi,'pyecr_dct_spectrum_series',sampling='stored_solver_grid'),),'pyecr_dct_spectrum_plot','log10(tau / s)','decade(tau/s)','chi (1/decade)','1/decade'),
        ('reconstruction',(S('Observed',p.time_s,p.signal,'pyecr_dct_data_series','scatter'),S('Reconstruction',p.time_s,r.y_fit,'pyecr_dct_reconstruction_series')),'pyecr_dct_reconstruction_plot','Time (s)','s','Normalized progress','1'),
        ('residuals',(S('Observed - reconstructed',p.time_s,r.residuals,'pyecr_dct_residual_series','scatter'),),'pyecr_dct_residual_plot','Time (s)','s','Residual','1')):
        b.add(key+'.'+label,'Monotonic DCT — '+label,series,plot=plot,x=x,xu=xu,y=y,yu=yu)
    lambdas=r.gcv_lambdas;scores=r.gcv_scores
    series=() if lambdas is None else (S('Saved GCV',lambdas,scores,'dct_monotonic_gcv',sampling='stored_lambda_candidates'),)
    b.add(key+'.lambda','Stored GCV candidates',series,plot='pyecr_dct_lambda_plot',x='lambda (penalty multiplier)',y='GCV score',xu='lambda penalty',yu='weighted signal²',xscale='log' if lambdas is not None and np.all(lambdas>0) else 'linear',view_log_coordinates=(True,False),reason=None if series else 'No GCV candidate arrays in this manual-lambda result.')
    snapshots=r.diagnostics.get('lambda_peak_snapshots',())
    curves=[]
    for i,row in enumerate(snapshots):
        curves.append(S('Stored lambda '+str(row.get('regularization_lambda',i)),row.get('peak_positions_s',()),row.get('peak_areas',()),key+'_stability_'+str(i),'scatter','stored_lambda_sensitivity'))
    b.add(key+'.stability','Stored lambda peak sensitivity',curves,x='Peak timescale (s)',y='Peak area (not CI)',xu='s',yu='1',reason=None if any(s.y for s in curves) else 'No stored lambda peak snapshots.')
