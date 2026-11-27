"""Build component DCT controls and display results from the selected fit."""
from dataclasses import replace
from pyecr.contracts import ComponentDctConfig
from pyecr.gui.theme import TOKENS
from pyecr.gui.component_dct import display_component_job,complete_component_evidence,NOTICE
from pyecr.gui.components.action_state import set_action_availability


def component_parent_notice(eligibility, evaluation=None):
    if eligibility is None or not eligibility.can_run:return ''
    if evaluation is not None:
        dependence=evaluation.residual.dependence
        if dependence=='warning':return 'Residual dependence warning. Component DCT is available.'
        if dependence=='unassessed':return 'Residual independence unassessed. Component DCT is available.'
        if evaluation.interval_status!='conditional_iid':return 'Parent CI needs review. Component DCT is available.'
        return ''
    if not eligibility.original_reliable:
        from pyecr.services.component_eligibility import SHORT_WARNING
        return SHORT_WARNING
    return ''


class ComponentDctViewMixin:
    def _dct_component_mode(self):
        return self._require_dpg().get_value('pyecr_dct_analysis_mode')=='Components'

    def _build_component_controls(self):
        dpg=self._require_dpg()
        with dpg.group(tag='pyecr_component_dct_controls',show=False):
            dpg.add_text('Parent Fit and Diagnostics required.',tag='pyecr_component_parent_status',wrap=TOKENS.wraps.rail)
            with dpg.collapsing_header(label='Interpreting component spectra',default_open=False):
                dpg.add_text(NOTICE,wrap=TOKENS.wraps.rail,tag='pyecr_component_dct_notice')

    def _build_component_numeric_controls(self):
        d=self._require_dpg()
        with d.group(tag='pyecr_component_dct_settings',show=False):
            for i in (1,2):
                d.add_text(f'Component {i} λ (Manual)')
                d.add_input_double(tag=f'pyecr_component_dct_lambda_{i}',default_value=1e-6,format='%.3e',width=-1)

    def _build_component_view_controls(self):
        d=self._require_dpg()
        with d.group(tag='pyecr_component_dct_view_controls',show=False):
            d.add_text('Component view')
            d.add_combo(('Overall','component_1','component_2','Comparison'),default_value='Comparison',tag='pyecr_component_dct_view',callback=self._change_component_view,width=-1)
            d.add_combo(('Independent shape','Common signed scale'),default_value='Common signed scale',tag='pyecr_component_dct_scale',callback=self._change_component_view,width=-1)
            d.add_text('',tag='pyecr_component_role_readout',wrap=TOKENS.dct.context_wrap)
            d.add_button(label='Explain component signs',
                         tag='pyecr_component_sign_help',callback=self._show_component_sign)

    def _build_component_plot_series(self):
        dpg=self._require_dpg()
        for key,kind,label in (('component_1','spectrum','component_1 · B1/S'),('component_2','spectrum','component_2 · B2/S'),
            ('sum','spectrum','Σ weighted χ_j'),('parent_fit','reconstruction','Parent Y_fit'),
            ('delta','reconstruction','Explicit δ'),('overall','reconstruction','Whole signed reconstruction'),
            ('model_error','residual','Y_fit − component reconstruction')):
            dpg.add_line_series([],[],label=label,tag='pyecr_component_dct_'+key,parent='pyecr_dct_'+kind+'_y_axis',show=False)

    def _component_dct_draft(self):
        config,spec,ds,recipe=self._signed_dct_draft()
        dpg=self._require_dpg()
        components=tuple(ComponentDctConfig(grid=replace(config.grid,lambda_mode='manual',
            regularization_lambda=float(dpg.get_value(f'pyecr_component_dct_lambda_{i}')))) for i in (1,2))
        return config,components,spec,ds,recipe

    def _selected_component_job(self):
        self.controller.current_component_dct
        key=getattr(self,'_component_selected_id',None)
        return next((j for j in self.controller.component_dct_service.history if j.job_id==key),None)

    def _select_component_history(self,sender,value,user_data):
        self._component_selected_id=getattr(self,'_component_history_labels',{}).get(str(value),str(value).split(' | ')[0]) or None
        self._refresh_widgets()

    def _change_component_view(self,*args):
        self._component_render_key=None
        self._dct_fitted_axes.clear();self._dct_pending_axes.clear();self._dct_release_axes.clear()
        self._refresh_widgets()

    def _run_component_dct_widgets(self):
        cfg,components,spec,_,_=self._component_dct_draft()
        self._component_selected_id=self.controller.run_component_dct(cfg,components,normalization_spec=spec)
        self._signed_dct_callback_error=None

    def _component_complete_evidence(self):
        try:
            cfg,components,spec,_,_=self._component_dct_draft()
            draft=dict(config=cfg.to_dict(),component_configs=[c.to_dict() for c in components],
                normalization_spec=None if spec is None else spec.to_dict())
        except Exception as exc:draft=dict(error=str(exc))
        import json
        capture=self._component_parent_capture()
        e=capture.eligibility
        evidence=(e.to_dict() if e is not None else dict(blocking_reasons=capture.state.blocking_reasons))
        # Explicit Details/copy only. Preserve the stored-job marker and JSON
        # intact; current-parent evidence is separate from frozen history.
        current='Current parent eligibility (not a submitted task):\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\n\n'
        return current+complete_component_evidence(self._selected_component_job(),draft)

    def _component_parent_capture(self):
        """Read-only gate cache, invalidated by immutable owner/revision changes."""
        s=self.controller.analysis_session
        key=(id(s.dataset),id(s.preprocessing),id(s.preprocessing.config) if s.preprocessing else None,id(s.current_analysis_config),id(s.fit_result),id(s.diagnostics),
             s.current_revision,s.fit_revision,s.fit_generation,s.fit_stale,s.diagnostics_stale,
             s.diagnostics_fit_revision,s.diagnostics_fit_generation,s.diagnostics_identity)
        cached=getattr(self,'_component_parent_cache',None)
        if cached is None or cached[0]!=key:
            cached=(key,self.controller.workflow.capture_component_sources())
            self._component_parent_cache=cached
        return cached[1]

    def _refresh_component_dct_layout(self,source):
        dpg=self._require_dpg();job=self._selected_component_job();s=self.controller.analysis_session
        self._component_history_labels=self._dct_history_choices(self.controller.component_dct_service.history)
        choices=list(self._component_history_labels)
        dpg.configure_item('pyecr_component_dct_history',items=choices,show=bool(choices))
        dpg.set_value('pyecr_component_dct_history',next(v for v in choices if self._component_history_labels[v]==job.job_id) if job else 'Select stored run' if choices else '')
        reason='';matches=False;parent_notice='';parent_ready=False
        try:
            cfg,components,spec,ds,recipe=self._component_dct_draft()
            if s.fit_result is None or s.fit_stale:reason='A current joint Fit is required. Run Fit explicitly.'
            elif s.diagnostics is None or s.diagnostics_stale or s.diagnostics_identity is None:reason='Run current basic Diagnostics explicitly.'
            else:
                capture=self._component_parent_capture()
                if not capture.state.ready:reason='; '.join(capture.state.blocking_reasons)
                else:
                    parent_ready=True
                    parent_notice=component_parent_notice(capture.eligibility,getattr(s.diagnostics,'evaluation',None))
            if not reason and spec is None and not ds.metadata.get('stability_windows',{}).get('final'):reason='Accept a final window or explicitly declare an external terminal.'
            matches=bool(job and cfg==job.request.config and components==job.request.component_configs
                and spec==job.request.normalization_spec and ds is job.request.dataset and recipe==job.request.recipe
                and s.diagnostics_identity==job.request.capture.identity and not job.stale_reasons)
        except Exception as exc:reason=str(exc)
        running=any(j.status in {'queued','running'} for j in self.controller.component_dct_service.history)
        self._component_eligibility_warning=parent_notice
        dpg.set_value('pyecr_component_parent_status',('Parent eligible · review cautions' if parent_notice else 'Parent eligible')
                      if parent_ready else 'Current parent Fit and Diagnostics required.')
        run_reason=reason or ('A component DCT job is running; wait or Cancel.' if running else
            parent_notice or 'Explicit Run only; component copies never modify Analysis.')
        set_action_availability(dpg,'pyecr_run_dct',enabled=not reason and not running,reason=run_reason)
        set_action_availability(dpg,'pyecr_cancel_dct',enabled=bool(job and job.status in {'queued','running'} and not job.cancellation_requested),
            reason='No cancellable component task; inspect its stored status.')
        dpg.set_value('pyecr_dct_run_reason',run_reason)
        dpg.set_value('pyecr_dct_source',source+'\nModel-derived components; complete observations.')
        dpg.set_value('pyecr_dct_mode_notice','Components · draft '+('matches stored run.' if matches else 'differs from stored run / has not run.'))
        dpg.configure_item('pyecr_signed_dct_physical',show=False)
        dpg.configure_item('pyecr_signed_dct_terminal',enabled=dpg.get_value('pyecr_signed_dct_reference')=='External known terminal')
        dpg.configure_item('pyecr_signed_dct_terminal_evidence',enabled=dpg.get_value('pyecr_signed_dct_reference')=='External known terminal')
        mode=str(dpg.get_value('pyecr_dct_lambda_mode')).lower()
        dpg.configure_item('pyecr_dct_lambda',show=mode=='manual',enabled=mode=='manual')
        dpg.configure_item('pyecr_dct_gcv_lambda',show=mode!='manual')
        dpg.configure_item('pyecr_signed_dct_lambdas',enabled=mode!='manual')
        dpg.set_value('pyecr_dct_gcv_lambda',
            (f'{job.bundle.overall_result.selected_lambda:.5g} · saved'+('' if matches else ' · draft differs'))
            if job and job.bundle else 'No whole λ stored')
        dpg.set_value('pyecr_dct_lambda_reason','Whole Manual/GCV; each component has its own explicit Manual λ. λ multiplies penalty; no component CI.')
        warning='\n'.join((reason,*((job.failure,) if job and job.failure else ()),*(job.stale_reasons if job else ()),
            *((str(getattr(self,'_signed_dct_callback_error','')),) if getattr(self,'_signed_dct_callback_error',None) else ())))
        if job and job.bundle:
            warning+='\nWhole DCT flags: '+', '.join(job.bundle.overall_analysis.flags)+f'; {len(job.bundle.state.warnings)} warnings — Summary / Full details.'
            warning+=''.join(f'\ncomponent_{i+1}: {r}' for i,r in enumerate(job.bundle.unavailable_reasons) if r)
        dpg.set_value('pyecr_dct_result_warning',warning);dpg.configure_item('pyecr_dct_result_warning',show=bool(warning))
        dpg.set_value('pyecr_page_status_dct','No component run.' if job is None else f'{job.job_id} · {job.status}\n'+
            (job.progress[-1][1] if job.progress else '')+(
                '\nCancellation requested; waiting for checkpoint.' if job.cancellation_requested and job.status=='running'
                else '\nCancelled; no result published.' if job.cancellation_requested else ''))
        if dpg.get_item_configuration('pyecr_dct_evidence_window')['show']:
            tags=('pyecr_dct_tau_min','pyecr_dct_tau_max','pyecr_dct_n_tau','pyecr_dct_lambda_mode','pyecr_dct_lambda',
                'pyecr_component_dct_lambda_1','pyecr_component_dct_lambda_2','pyecr_signed_dct_lambdas',
                'pyecr_signed_dct_reference','pyecr_signed_dct_terminal','pyecr_signed_dct_terminal_evidence')
            evidence_key=(job,getattr(self,'_component_parent_cache',(None,))[0],tuple(str(dpg.get_value(tag)) for tag in tags))
            if getattr(self,'_component_evidence_key',None)!=evidence_key:
                dpg.set_value('pyecr_dct_evidence_text',self._component_complete_evidence())
                self._component_evidence_key=evidence_key

    def _render_component_dct(self):
        dpg=self._require_dpg();job=self._selected_component_job()
        view_name=str(dpg.get_value('pyecr_component_dct_view'));scale=str(dpg.get_value('pyecr_component_dct_scale'))
        key=(job,view_name,scale)
        if getattr(self,'_component_render_key',None)==key:return
        view,extras=display_component_job(job,view_name,scale)
        from pyecr.gui.dct_component_readout import component_readout
        readouts=[] if not job or not job.bundle else [component_readout(job.bundle,i) for i in range(2)]
        selected=int(view_name[-1])-1 if view_name in ('component_1','component_2') else None
        visible=readouts if selected is None else readouts[selected:selected+1]
        dpg.set_value('pyecr_component_role_readout','\n'.join(r['explanation'] for r in visible))
        from pyecr.gui.components.compact_panels import content_width
        if dpg.does_item_exist('pyecr_component_role_readout') and hasattr(dpg,'get_item_parent'):
            dpg.configure_item('pyecr_component_role_readout',wrap=content_width(dpg,'pyecr_component_role_readout'))
        from pyecr.gui.signed_dct import SIGN_LEGEND
        dpg.set_value('pyecr_signed_dct_legend',
            'Independent shapes chi_j are nonnegative. The sign of B_j restores signal direction; the common-scale spectrum is (B_j/S_total) chi_j. Components are not independent experiments and do not identify H/O.'
            if view_name in {'component_1','component_2'} and scale=='Independent shape' else SIGN_LEGEND)
        kind=job.bundle.parent_source.source_kind.value if job and job.bundle else ''
        observed_label={'simulation_clean':'Simulated clean Y','simulation_observed':'Simulated observations Y'}.get(kind,'Experimental Y')
        dpg.configure_item('pyecr_dct_data_series',label=(visible[0]['label']+' remaining' if visible else 'Component remaining') if selected is not None else observed_label)
        for i,r in enumerate(readouts):
            dpg.configure_item('pyecr_component_dct_component_'+str(i+1),label=r['label']+' terminal spectrum')
        dpg.configure_item('pyecr_dct_reconstruction_series',label='δ + weighted component DCT' if view_name=='Comparison' else 'DCT reconstruction')
        for tag,ys in (('positive',tuple(max(v,0) for v in view.chi)),('negative',tuple(min(v,0) for v in view.chi)),('zero',(0.,)*len(view.x))):
            dpg.set_value('pyecr_signed_dct_'+tag,[list(view.x),list(ys)])
        for tag,ys in (('data',view.observed),('reconstruction',view.reconstruction),('residual',view.residual)):
            dpg.set_value('pyecr_dct_'+tag+'_series',[list(view.time[:len(ys)]),list(ys)])
        for name in ('component_1','component_2','sum','parent_fit','delta','overall','model_error'):
            x,y=extras.get(name,((),()))
            dpg.set_value('pyecr_component_dct_'+name,[list(x[:len(y)]),list(y)])
            dpg.configure_item('pyecr_component_dct_'+name,show=bool(y))
        for tag,x,y in (('lambda_series',view.lambda_x,view.lambda_score),('lambda_points',view.lambda_x,view.lambda_score),
                        ('stability_series',view.peak_x,view.peak_shift)):
            dpg.set_value('pyecr_signed_dct_'+tag,[list(x),list(y)])
        dpg.set_value('pyecr_dct_summary',view.summary);dpg.set_value('pyecr_dct_details',view.details)
        dpg.configure_item('pyecr_dct_spectrum_y_axis',label='Shape density / decade' if selected is not None and scale=='Independent shape' else 'Terminal density / decade')
        dpg.configure_item('pyecr_dct_reconstruction_y_axis',label='Remaining shape' if selected is not None and scale=='Independent shape' else 'Terminal deviation')
        dpg.configure_item('pyecr_dct_residual_y_axis',label='Full-point residual (1)')
        for kind in ('spectrum','reconstruction','residual'):
            for axis in ('x','y'):dpg.fit_axis_data('pyecr_dct_'+kind+'_'+axis+'_axis')
        self._component_display_view=view;self._component_render_key=key

    def _show_component_sign(self,*args):
        import numpy as np
        from pyecr.gui.dct_component_readout import component_readout
        d=self._require_dpg();job=self._selected_component_job()
        if job is None or job.bundle is None:return
        choice=d.get_value('pyecr_component_dct_view')
        index=int(choice[-1])-1 if choice in ('component_1','component_2') else 0
        b=job.bundle;r=b.independent_results[index]
        if r is None:return
        if not d.does_item_exist('pyecr_component_sign_window'):
            with d.window(label='Component sign: change versus terminal deviation',tag='pyecr_component_sign_window',show=False,width=1000,height=740):
                d.add_text('',tag='pyecr_component_sign_text',wrap=920)
                for name,title,xlabel,ylabel in (('time','Same component: two reference zeros','Time (s)','Conductivity contribution'),
                        ('spectrum','Opposite density signs use different kernels','log10(tau / s)','Density / decade')):
                    with d.plot(label=title,tag='pyecr_component_sign_'+name+'_plot',width=-1,height=250):
                        d.add_plot_legend(location=d.mvPlot_Location_East);d.add_plot_axis(d.mvXAxis,label=xlabel,tag='pyecr_component_sign_'+name+'_x')
                        with d.plot_axis(d.mvYAxis,label=ylabel,tag='pyecr_component_sign_'+name+'_y'):
                            for role,label in (('change','Change from initial'),('terminal','Deviation from terminal')):
                                d.add_line_series([],[],label=label,tag='pyecr_component_sign_'+name+'_'+role)
            self._register_owned_root('pyecr_component_sign_window')
        info=component_readout(b,index)
        d.set_value('pyecr_component_sign_text',info['explanation']+'\n'
            'c(t)=Delta_sigma*(1-r(t)); c(t)-c(infinity)=-Delta_sigma*r(t). '
            'The existing DCT uses exp(-t/tau) on the terminal deviation, so its sign is opposite to the change amplitude. '
            'The change-density below uses (1-exp(-t/tau)); it is a different convention, not a corrected fit. '
            'Neither sign identifies a carrier by itself.')
        physical=r.physical_source;values=np.asarray(physical.signal)
        d.set_value('pyecr_component_sign_time_change',[list(physical.time_s),values.tolist()])
        d.set_value('pyecr_component_sign_time_terminal',[list(physical.time_s),(values-physical.component_terminal_value).tolist()])
        d.configure_item('pyecr_component_sign_time_y',label='Contribution ('+physical.signal_unit+')')
        for role,factor in (('terminal',info['factor']),('change',-info['factor'])):
            d.set_value('pyecr_component_sign_spectrum_'+role,[list(r.log10_tau),[factor*v for v in r.chi]])
        width=max(620,min(1040,d.get_viewport_client_width()-40))
        d.configure_item('pyecr_component_sign_window',show=True,width=width,height=max(480,min(790,d.get_viewport_client_height()-40)))
        d.configure_item('pyecr_component_sign_text',wrap=width-36)
        for name in ('time','spectrum'):
            for axis in ('x','y'):d.fit_axis_data('pyecr_component_sign_'+name+'_'+axis)
