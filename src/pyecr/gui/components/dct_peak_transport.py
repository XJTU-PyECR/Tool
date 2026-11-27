"""Explicit read-only guide; quantitative interpretation gates stay separate."""
from math import log10
from pyecr.gui.theme import TOKENS
from pyecr.diagnostics.dct_transport import transport_guide, constant_tau_locus
from pyecr.services.transport_summary import summarize_session


class DctPeakTransportMixin:
    def _build_peak_transport_guide(self):
        d = self._require_dpg()
        with d.collapsing_header(label='Peak / Bi / D / k guide (conditional)',tag='pyecr_dct_peak_guide_header',default_open=False):
            d.add_text('Compare this spectrum with model time scales. This does not estimate D or k.',wrap=TOKENS.dct.context_wrap)
            d.add_combo((),tag='pyecr_dct_peak_guide_component',width=-1,show=False,callback=self._select_peak_guide_component)
            d.add_button(label='Preview peak / Bi / D / k guide',tag='pyecr_dct_peak_guide_preview',callback=self._preview_peak_transport)
            d.add_text('Select a stored DCT spectrum, then preview a matching current Fit. Given backgrounds require Review first.',tag='pyecr_dct_peak_guide_status',wrap=TOKENS.dct.context_wrap)
            d.add_text('',tag='pyecr_dct_peak_guide_scope',wrap=TOKENS.dct.context_wrap)
            d.add_group(tag='pyecr_dct_peak_guide_processes')
            with d.table(header_row=False,width=-1,policy=d.mvTable_SizingStretchProp):
                d.add_table_column(init_width_or_weight=1.);d.add_table_column(init_width_or_weight=1.)
                with d.table_row():
                    with d.group():
                        d.add_text('Recovered peak')
                        d.add_combo((),tag='pyecr_dct_peak_guide_peak',width=-1,callback=self._change_peak_locus)
                    with d.group():
                        d.add_text('Declared model role')
                        d.add_combo((),tag='pyecr_dct_peak_guide_process',width=-1,callback=self._change_peak_locus)
            d.add_text('',tag='pyecr_dct_peak_guide_takeaway',wrap=TOKENS.dct.context_wrap)
            d.add_text('Horizontal proximity means compatible times, not species identification. Model row: five largest retained modal masses. Rows do not represent spectral height.',wrap=TOKENS.dct.context_wrap)
            with d.plot(label='DCT peaks vs model times',tag='pyecr_dct_peak_alignment_plot',height=240,width=-1):
                d.add_plot_legend(location=d.mvPlot_Location_East)
                d.add_plot_axis(d.mvXAxis,label='log10(tau / s)',tag='pyecr_dct_peak_alignment_x')
                with d.plot_axis(d.mvYAxis,label='Reference row',tag='pyecr_dct_peak_alignment_y'):
                    d.add_scatter_series([],[],label='DCT peak candidates',tag='pyecr_dct_peak_alignment_observed')
                    for i in range(2):d.add_scatter_series([],[],label=f'Parent process {i+1} modes',tag=f'pyecr_dct_peak_alignment_{i}')
            with d.collapsing_header(label='Why one peak cannot determine both D and k',tag='pyecr_dct_peak_locus_details',default_open=False):
                d.add_text('Hypothesis: the selected peak is the fundamental mode. The curve contains alternative D,k pairs with this time; it is not an estimate or a confidence region.',wrap=TOKENS.dct.context_wrap)
                self._build_peak_locus_plot()
            with d.collapsing_header(label='All peak/model numbers and limitations (scrollable)',default_open=False):
                d.add_group(tag='pyecr_dct_peak_guide_support')
                d.add_group(tag='pyecr_dct_peak_guide_matches')
                d.add_text('',tag='pyecr_dct_peak_guide_advice',wrap=TOKENS.dct.context_wrap,color=TOKENS.colors.warning)

    def _build_peak_locus_plot(self):
        d=self._require_dpg()
        with d.plot(label='Same tau1: possible D,k pairs',tag='pyecr_dct_peak_locus_plot',height=260,width=-1):
            d.add_plot_legend(location=d.mvPlot_Location_East)
            d.add_plot_axis(d.mvXAxis,label='log10(D / cm^2 s^-1)',tag='pyecr_dct_peak_locus_x')
            with d.plot_axis(d.mvYAxis,label='log10(k / cm s^-1)',tag='pyecr_dct_peak_locus_y'):
                for name,label in (('surface','Bi_ref < 0.03'),('mixed','0.03 <= Bi_ref <= 30'),('diffusion','Bi_ref > 30')):
                    d.add_line_series([],[],label=label,tag='pyecr_dct_peak_locus_'+name)
                d.add_scatter_series([],[],label='Current model parameters',tag='pyecr_dct_peak_locus_fit')
        d.add_text('Bi_ref uses the shortest active half-dimension. Regime thresholds and the 0.3-decade proximity count are heuristics, not confidence or assignment tests.',wrap=TOKENS.dct.context_wrap)

    def _guide_binding(self):
        s = self.controller.analysis_session
        d = self._require_dpg()
        choice = d.get_value('pyecr_dct_context_source')
        run = self.controller.simulation.selected
        request = None if run is None or not str(choice).startswith('Given') else run.request.content_sha256
        return (s.current_revision,s.fit_generation,id(s.fit_result),id(s.diagnostics),s.fit_stale,s.diagnostics_stale,
                choice,d.get_value('pyecr_dct_context_budget'),request,
                d.get_value('pyecr_dct_context_declared'),d.get_value('pyecr_dct_context_basis'),
                d.get_value('pyecr_component_dct_view'),d.get_value('pyecr_component_dct_scale'))

    def _refresh_peak_component_selector(self):
        d=self._require_dpg();component=self._dct_component_mode()
        d.configure_item('pyecr_dct_peak_guide_component',show=component)
        if not component:return
        job=self._selected_component_job()
        key=(id(job),d.get_value('pyecr_component_dct_view'))
        if key==getattr(self,'_dct_peak_component_selector_key',None):return
        self._dct_peak_component_selector_key=key
        labels=['Whole signal / global peaks','component_1','component_2']
        if job and job.bundle:
            from pyecr.gui.dct_component_readout import component_readout
            labels[1:]=[component_readout(job.bundle,i)['label']+' / its own shape peaks' for i in range(2)]
        self._dct_peak_component_choices=dict(zip(labels,('Overall','component_1','component_2')))
        d.configure_item('pyecr_dct_peak_guide_component',items=labels)
        current=d.get_value('pyecr_component_dct_view')
        d.set_value('pyecr_dct_peak_guide_component',labels[1] if current=='component_1' else labels[2] if current=='component_2' else labels[0])

    def _select_peak_guide_component(self,sender,value,user_data):
        d=self._require_dpg();choice=self._dct_peak_component_choices.get(value)
        if choice is None:return
        d.set_value('pyecr_component_dct_view',choice)
        self._change_component_view()
        self._preview_peak_transport()

    def _preview_peak_transport(self,*args):
        d = self._require_dpg()
        try:
            source,key,kind = self._physical_context_target()
            if str(d.get_value('pyecr_dct_context_source')).startswith('Given'):
                record = self._dct_context_review or self._dct_context_store.selected.get(key)
                if record is None or isinstance(record,tuple) or not record.declared_given:
                    raise ValueError('Review the explicitly declared Given background first')
                from pyecr.services.dct_context_service import assert_target
                assert_target(record,source,key,kind)
                run=self.controller.simulation.selected
                if run is None or record.transport.simulation_request_sha256!=run.request.content_sha256:
                    raise ValueError('Given request changed; Review the selected background again')
                summary = record.transport
            else:
                summary = summarize_session(self.controller.analysis_session)
                a,b = source.identity,summary.source
                if (a.dataset_sha256,a.recipe_sha256)!=(b.dataset_sha256,b.recipe_sha256):
                    raise ValueError('DCT and current Fit use different data/preprocessing. Select a matching Fit or rerun DCT explicitly.')
                if a.config_sha256 is not None and a.config_sha256!=b.config_sha256:
                    raise ValueError('DCT source and current Fit model configuration differ')
                if a.fit_id is not None:a.assert_matches(b)
            process_keys=None;scope='Whole spectrum: peak candidates are not assigned to H/O.'
            areas=[];scope_flags=[]
            if self._dct_signed_mode():
                job = self._selected_component_job() if self._dct_component_mode() else self._selected_signed_job()
                analysis = job.bundle.overall_analysis if self._dct_component_mode() else job.analysis
                peaks = [p.candidate.position_s for p in analysis.peak_diagnostics]
                areas=[abs(p.candidate.area) for p in analysis.peak_diagnostics]
                scope_flags=list(analysis.flags)
                if self._dct_component_mode() and d.get_value('pyecr_component_dct_view') in ('component_1','component_2'):
                    from pyecr.gui.dct_component_readout import component_readout
                    index=int(d.get_value('pyecr_component_dct_view')[-1])-1
                    if job.bundle.independent_results[index] is None:raise ValueError('Selected component spectrum is unavailable')
                    info=component_readout(job.bundle,index)
                    peaks=[p.position_s for p in info['peaks']]
                    areas=[p.area for p in info['peaks']]
                    scope_flags=list(info['warnings'])
                    process_keys=(info['process_key'],)
                    lam=job.bundle.independent_results[index].config.grid.regularization_lambda
                    scope=f"{info['label']}: own remaining-shape peaks; Manual lambda={lam:.5g}. A Fit role only. Component lambda stability is unassessed."
            else:
                peaks = list(self.controller.session.dct_result.result.peak_positions_s)
                areas=list(self.controller.session.dct_result.result.peak_areas)
            budget = int(d.get_value('pyecr_dct_context_budget'))
            guide = transport_guide(summary,peaks,per_axis=budget,time_s=source.time_s,process_keys=process_keys)
            guide.update(target_sha256=key,source_sha256=source.content_sha256,scope=scope,spectrum_flags=scope_flags,derived_component=process_keys is not None)
            self._dct_peak_guide = guide
            self._dct_peak_guide_binding = (key,self._guide_binding())
            self._dct_peak_guide_geometry = summary.geometry
            self._dct_peak_guide_peaks = peaks
            process_rows=[]
            f=lambda v:'N/A' if v is None else f'{v:.5g}'
            for p in guide['processes']:
                label=p['process']+' slot' if p['process'] in ('H','O') else p['label']
                process_rows.append((label,f(p['D_cm2_s']),f(p['k_cm_s']),f(p['tau1_s'])))
            from pyecr.model_display import display_model_text
            d.set_value('pyecr_dct_peak_guide_status',f"{display_model_text(summary.model_id)}; parent inference: {summary.parent_reliability}. "+
                        ('Exploratory reference; Apply/CI gates unchanged.' if summary.parent_reliability not in ('supported','not_applicable') else 'Conditional consistency comparison.'))
            d.set_value('pyecr_dct_peak_guide_scope',scope)
            self._dct_review_table('pyecr_dct_peak_guide_processes',('Fit role','D (cm^2/s)','k (cm/s)','tau1 (s)'),process_rows)
            self._dct_review_table('pyecr_dct_peak_guide_support',('Peak','tau (s)','Samples in 0.1-3 tau','Time-window support (heuristic)'),
                [(r['peak'],f(r['tau_s']),r['samples_in_0_1_to_3_tau'],r['note']) for r in guide['peak_support']])
            self._dct_review_table('pyecr_dct_peak_guide_matches',('Peak','tau peak (s)','Model role','Nearest mode indices','tau mode (s)','tau ratio','Delta decade','Modes +/-0.3 decade'),
                [(m['peak'],f(m['tau_peak_s']),m['process'],str(m['mode']),f(m['tau_mode_s']),f(m['ratio']),f(m['delta_decades']),m['modes_within_0_3_decade']) for m in guide['peak_comparisons']])
            d.set_value('pyecr_dct_peak_guide_advice','\n'.join([*guide['cautions'],*[p['label']+': '+advice for p in guide['processes'] for advice in p['advice']]]))
            d.set_value('pyecr_dct_peak_alignment_observed',[[log10(t) for t in peaks],[0.]*len(peaks)])
            ticks=[('DCT peaks',0.)];alignment_x=[log10(t) for t in peaks]
            for i in range(2):
                p=guide['processes'][i] if i<len(guide['processes']) else None
                modes=[] if p is None else sorted(p['modes'],key=lambda m:m['mass'],reverse=True)[:5]
                d.set_value(f'pyecr_dct_peak_alignment_{i}',[[log10(m['tau_s']) for m in modes],[float(i+1)]*len(modes)])
                alignment_x.extend(log10(m['tau_s']) for m in modes)
                if p is not None:
                    d.configure_item(f'pyecr_dct_peak_alignment_{i}',label=p['process']+' model modes')
                    ticks.append(('Model '+p['process'],float(i+1)))
            d.set_axis_ticks('pyecr_dct_peak_alignment_y',tuple(ticks))
            self._dct_peak_guide_peak_labels=[f'Peak {i+1}: {t:.5g} s' for i,t in enumerate(peaks)]
            self._dct_peak_guide_process_labels=[p['label'] for p in guide['processes']]
            for tag,labels in (('peak',self._dct_peak_guide_peak_labels),('process',self._dct_peak_guide_process_labels)):
                d.configure_item('pyecr_dct_peak_guide_'+tag,items=labels)
                d.set_value('pyecr_dct_peak_guide_'+tag,labels[0] if labels else '')
            if peaks:
                supported=[i for i,row in enumerate(guide['peak_support']) if row['samples_in_0_1_to_3_tau']>=3 and row['tau_s']<=max(source.time_s)]
                candidates=supported or list(range(len(peaks)))
                selected=max(candidates,key=lambda i:areas[i] if i<len(areas) else 0)
                d.set_value('pyecr_dct_peak_guide_peak',self._dct_peak_guide_peak_labels[selected])
            self._change_peak_locus()
            self._queue_peak_reference_limits('alignment',alignment_x,[0.,float(len(guide['processes']))])
        except (ValueError,TypeError,RuntimeError) as exc:
            self._clear_peak_transport('Guide unavailable: '+str(exc))

    def _clear_peak_transport(self,message):
        d = self._require_dpg()
        self._dct_peak_guide = None
        self._dct_peak_pending_limits={}
        d.set_value('pyecr_dct_peak_guide_status',message)
        d.set_value('pyecr_dct_peak_guide_advice','')
        d.set_value('pyecr_dct_peak_guide_takeaway','')
        d.set_value('pyecr_dct_peak_guide_scope','')
        for tag in ('processes','support','matches'):d.delete_item('pyecr_dct_peak_guide_'+tag,children_only=True)
        for tag in ('observed','0','1'):d.set_value('pyecr_dct_peak_alignment_'+tag,[[],[]])
        for tag in ('surface','mixed','diffusion','fit'):d.set_value('pyecr_dct_peak_locus_'+tag,[[],[]])

    def _refresh_peak_transport_binding(self):
        self._refresh_peak_component_selector()
        self._refresh_peak_reference_limits()
        if getattr(self,'_dct_peak_guide',None) is None:return
        try:key=self._physical_context_target()[1]
        except (ValueError,TypeError,RuntimeError):key=None
        if (key,self._guide_binding())!=self._dct_peak_guide_binding:
            self._clear_peak_transport('Spectrum or parent changed. Preview the guide again for the selected source.')

    def _queue_peak_reference_limits(self,name,x,y):
        if not x or not y:return
        def bounds(values):
            lo,hi=min(values),max(values);pad=max(.12,(hi-lo)*.08)
            return (lo-pad,hi+pad)
        pending=getattr(self,'_dct_peak_pending_limits',{})
        pending[name]=((bounds(x),bounds(y)),2)
        self._dct_peak_pending_limits=pending

    def _refresh_peak_reference_limits(self):
        """Hidden plots ignore ImPlot's Fit request. Apply data bounds on reveal."""
        d=self._require_dpg()
        for axis in getattr(self,'_dct_peak_release_axes',()):d.set_axis_limits_auto(axis)
        self._dct_peak_release_axes=[]
        pending=getattr(self,'_dct_peak_pending_limits',{})
        for name,(pairs,count) in tuple(pending.items()):
            plot='pyecr_dct_peak_'+name+'_plot'
            if not d.get_item_state(plot).get('visible'):continue
            axes=tuple('pyecr_dct_peak_'+name+'_'+a for a in ('x','y'))
            for axis,limits in zip(axes,pairs):d.set_axis_limits(axis,*limits)
            manager=getattr(self,'_plot_views',None)
            if manager is not None:
                manager.range_applied.add(plot)
                manager.layout_ranges.pop(plot,None)
            if count>1:pending[name]=(pairs,count-1)
            else:
                pending.pop(name);self._dct_peak_release_axes.extend(axes)

    def _change_peak_locus(self,*args):
        d = self._require_dpg()
        guide = getattr(self,'_dct_peak_guide',None)
        if guide is None:return
        getattr(self,'_dct_peak_pending_limits',{}).pop('locus',None)
        for tag in ('surface','mixed','diffusion','fit'):d.set_value('pyecr_dct_peak_locus_'+tag,[[],[]])
        if not self._dct_peak_guide_peaks:
            d.set_value('pyecr_dct_peak_guide_takeaway','No reportable peak in this selected spectrum. Parent Bi and parameters are references only.')
            return
        choice=d.get_value('pyecr_dct_peak_guide_peak')
        if choice not in self._dct_peak_guide_peak_labels:return
        index=self._dct_peak_guide_peak_labels.index(choice)
        p=self._dct_peak_guide_process_labels.index(d.get_value('pyecr_dct_peak_guide_process'))
        process=guide['processes'][p]
        f=lambda v:'N/A' if v is None else f'{v:.5g}'
        tau=self._dct_peak_guide_peaks[index];fundamental=process['tau1_s']
        lines=[f"Selected peak {tau:.5g} s; {process['label']} fundamental {f(fundamental)} s"+
               (f"; ratio peak/model={tau/fundamental:.3g}." if fundamental else '. No ideal-step material-mode comparison.')]
        lines.append('Bi: '+', '.join(a['axis']+'='+f(a['Bi']) for a in process['axes'])+
                     '; |d ln(tau1)/d ln(D)|='+f(process['sensitivity_D'])+
                     ', |d ln(tau1)/d ln(k)|='+f(process['sensitivity_k'])+'.')
        support=next((row for row in guide['peak_support'] if row['peak']==index+1),None)
        if support:lines.append(f"Sampling: {support['samples_in_0_1_to_3_tau']} points in 0.1-3 times this peak; {support['note']}.")
        lines+=process['advice']
        if guide.get('spectrum_flags'):lines.append('Spectrum caution: '+', '.join(guide['spectrum_flags'])+'.')
        lines.append(('This component is derived from the Fit; agreement is expected. ' if guide.get('derived_component') else 'The same dataset is used; agreement is a consistency check. ')+
                     'This is not independent validation. One peak cannot determine both D and k.')
        d.set_value('pyecr_dct_peak_guide_takeaway','\n'.join(lines))
        from pyecr.gui.components.compact_panels import content_width
        for suffix in ('status','scope','takeaway','advice'):
            tag='pyecr_dct_peak_guide_'+suffix;d.configure_item(tag,wrap=content_width(d,tag))
        # A surface-only/instant-surface or nonideal model does not support a
        # finite D,k fundamental-mode locus; do not invent its missing input.
        if process['D_cm2_s'] is None or process['k_cm_s'] is None or process['tau1_s'] is None:return
        locus=constant_tau_locus(self._dct_peak_guide_geometry,self._dct_peak_guide_peaks[index])
        for name,accept in (('surface',lambda b:b<=.03),('mixed',lambda b:.03<=b<=30),('diffusion',lambda b:b>=30)):
            part=[r for r in locus if accept(r['Bi_reference'])]
            d.set_value('pyecr_dct_peak_locus_'+name,[[r['log10_D'] for r in part],[r['log10_k'] for r in part]])
        d.set_value('pyecr_dct_peak_locus_fit',[[log10(process['D_cm2_s'])],[log10(process['k_cm_s'])]])
        self._queue_peak_reference_limits('locus',[r['log10_D'] for r in locus]+[log10(process['D_cm2_s'])],
            [r['log10_k'] for r in locus]+[log10(process['k_cm_s'])])
