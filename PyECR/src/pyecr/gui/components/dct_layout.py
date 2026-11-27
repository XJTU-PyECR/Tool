"""Build the DCT settings panel and saved-spectrum plots."""
import json
from pyecr.gui.theme import TOKENS
from pyecr.gui.state import GuiPage
from pyecr.gui.dct_layout import geometry
from pyecr.gui.components.action_state import add_action_tooltip, set_action_availability
from pyecr.gui.dct_context import DctContextLayoutMixin
from pyecr.gui.components.step_panel import step_panel
from pyecr.gui.workflow_display import brief_text
from pyecr.gui.components.dct_search_controls import DctSearchControlsMixin
from pyecr.gui.components.dct_review import DctReviewMixin
from pyecr.gui.components.dct_lambda_scan import DctLambdaScanMixin


class DctLayoutMixin(DctLambdaScanMixin, DctReviewMixin, DctSearchControlsMixin, DctContextLayoutMixin):
    def _build_dct_layout(self):
        dpg=self._require_dpg();self._dct_route_controls=True
        self._dct_fitted_axes={};self._dct_pending_axes={};self._dct_release_axes={}
        self._dct_legacy_evidence_key=None
        with dpg.group(tag='pyecr_dct_layout'):
            dpg.add_text('',tag='pyecr_dct_layout_anchor',show=False)
            with dpg.group(horizontal=True,tag='pyecr_dct_columns'):
                with dpg.child_window(tag='pyecr_dct_left',width=TOKENS.dct.left_compact,height=TOKENS.heights.dialog,border=False,no_scrollbar=True):
                    with dpg.child_window(tag='pyecr_dct_parameters',width=-1,height=TOKENS.heights.results,border=False):
                        with step_panel(dpg,'dct_source','01 Source & representation',default_open=True):
                            with dpg.group(parent='pyecr_stage_dct_source_content'):
                                dpg.add_input_text(tag='pyecr_dct_source',readonly=True,multiline=True,width=-1,height=TOKENS.heights.notes)
                                self._build_signed_controls()
                        with step_panel(dpg,'dct_settings','02 Spectrum settings',default_open=True):
                            with dpg.group(parent='pyecr_stage_dct_settings_content'):
                                dpg.add_text('',tag='pyecr_dct_tau_advice',wrap=TOKENS.wraps.rail)
                                dpg.add_button(label='Apply data range',tag='pyecr_dct_apply_tau_advice',callback=self._apply_dct_tau_advice)
                                with dpg.tooltip('pyecr_dct_apply_tau_advice'):
                                    dpg.add_text('Suggests first positive processed time / 10 to last time × 10. Search guidance, not a resolved interval. Edits draft only.',wrap=TOKENS.wraps.rail)
                                for name,label,value in (('min','Tau minimum (s)',1e-3),('max','Tau maximum (s)',1e3)):
                                    dpg.add_text(label)
                                    dpg.add_input_double(tag='pyecr_dct_tau_'+name,default_value=value,format='%.3e',width=-1)
                                dpg.add_text('Tau grid points')
                                dpg.add_input_int(tag='pyecr_dct_n_tau',default_value=121,min_value=10,min_clamped=True,width=-1)
                                self._build_signed_mode_controls()
                                dpg.add_text('Regularization',tag='pyecr_dct_regularization_label')
                                dpg.add_combo(('GCV','Manual'),tag='pyecr_dct_lambda_mode',default_value='GCV',callback=self._select_dct_lambda_mode,width=-1)
                                dpg.add_input_text(tag='pyecr_dct_gcv_lambda',readonly=True,default_value='Selected after Run',width=-1)
                                dpg.add_button(label='View lambda scan',tag='pyecr_dct_view_scan',callback=self._show_lambda_scan,enabled=False)
                                dpg.add_input_double(tag='pyecr_dct_lambda',default_value=1e-3,format='%.3e',width=-1,enabled=False,show=False)
                                self._build_convex_noise_controls()
                                dpg.add_text('',tag='pyecr_dct_gcv_boundary',color=TOKENS.colors.warning,wrap=TOKENS.wraps.rail,show=False)
                                dpg.add_button(label='Expand GCV range',tag='pyecr_dct_expand_gcv',callback=self._expand_dct_gcv,show=False)
                                with dpg.tooltip('pyecr_dct_expand_gcv'):
                                    dpg.add_text('Adds four decades at the selected boundary; keeps tested candidates. Run DCT to evaluate. Component Manual λ values stay independent.',wrap=TOKENS.wraps.rail)
                                dpg.add_text('',tag='pyecr_dct_search_message',wrap=TOKENS.wraps.rail,show=False)
                                with dpg.collapsing_header(label='Advanced spectrum settings',tag='pyecr_dct_advanced_settings',default_open=False):
                                    with dpg.group(tag='pyecr_dct_legacy_gcv'):
                                        dpg.add_text('GCV lambda candidates')
                                        dpg.add_input_text(tag='pyecr_dct_lambdas',width=-1,hint='Blank: default 1e-8–1e2, 25 candidates')
                                    self._build_signed_numeric_controls()
                                    self._build_component_numeric_controls()
                                    dpg.add_text('',tag='pyecr_dct_lambda_reason',wrap=TOKENS.wraps.rail)
                                    dpg.add_checkbox(label='Peak-ratio heuristic (approximate)',tag='pyecr_dct_classical_bi',default_value=False)
                                    dpg.add_text('Peak ratios are historical heuristics. Quantitative interpretation needs explicitly reviewed Physical context. No species assignment or automatic model Apply.',tag='pyecr_dct_explanation',wrap=TOKENS.wraps.rail)
                    dpg.move_item('pyecr_component_dct_settings',parent='pyecr_stage_dct_settings_content',before='pyecr_dct_advanced_settings')
                    with step_panel(dpg,'dct_run','03 Run',default_open=True):
                        with dpg.group(parent='pyecr_stage_dct_run_content',tag='pyecr_dct_run_toolbar'):
                            with dpg.group(horizontal=True):
                                dpg.add_button(label='Run DCT',tag='pyecr_run_dct',callback=self._run_dct,enabled=False)
                                dpg.add_button(label='Cancel',tag='pyecr_cancel_dct',callback=self._cancel_dct,enabled=False)
                                dpg.add_button(label='Details',tag='pyecr_dct_full_details',callback=self._show_dct_evidence)
                                dpg.add_button(label='Fit axes',tag='pyecr_dct_fit_axes',callback=self._fit_dct_plot_axes,enabled=False)
                                with dpg.tooltip('pyecr_dct_fit_axes'):
                                    dpg.add_text('Fit DCT axes to the stored samples. Resets view ranges only; no DCT rerun.',wrap=TOKENS.wraps.rail)
                            dpg.add_text('',tag='pyecr_dct_run_status',wrap=TOKENS.wraps.rail,color=TOKENS.colors.warning)
                            dpg.add_text('',tag='pyecr_dct_run_reason',wrap=TOKENS.wraps.rail)
                            dpg.add_button(label='Use Signed representation',tag='pyecr_dct_use_signed_profile',
                                           callback=self._use_signed_dct_profile,show=False)
                with dpg.child_window(tag='pyecr_dct_right',width=TOKENS.widths.model_selector,height=TOKENS.heights.dialog,border=False):
                    with step_panel(dpg,'dct_results','04 Results & export',default_open=True):
                        with dpg.group(parent='pyecr_stage_dct_results_content'):
                            dpg.add_combo((),tag='pyecr_signed_dct_history',width=-1,show=False,callback=self._select_signed_history)
                            dpg.add_combo((),tag='pyecr_component_dct_history',width=-1,show=False,callback=self._select_component_history)
                            dpg.add_text('',tag='pyecr_dct_result_compact',wrap=TOKENS.wraps.page)
                            dpg.add_text('',tag='pyecr_dct_warning_compact',wrap=TOKENS.wraps.page,color=TOKENS.colors.warning)
                            self._build_component_view_controls()
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label='Review & interpretation',tag='pyecr_dct_more_plots',callback=self._toggle_dct_more_plots)
                                dpg.add_button(label='Physical interpretation',tag='pyecr_dct_context_action',callback=self._show_dct_context)
                            with dpg.tab_bar(tag='pyecr_dct_plot_tabs',tab_list_popup_button=True):
                                with dpg.tab(label='Spectrum',tag='pyecr_dct_spectrum_tab'):
                                    self._build_dct_saved_plot('reconstruction','Data and DCT reconstruction',TOKENS.dct.reconstruction_min)
                                    self._build_dct_saved_plot('spectrum','DCT spectrum',TOKENS.dct.spectrum_min)
                                with dpg.tab(label='Residuals',tag='pyecr_dct_reconstruction_tab'):
                                    self._build_dct_saved_plot('residual','Full observed − reconstructed residual',TOKENS.dct.residual_min)
                                self._build_dct_review_tabs()
                                self._build_physical_context()
                            self._build_dct_export_controls()
            with dpg.group(show=False):
                self._build_page_status('dct')
                dpg.configure_item('pyecr_page_context_dct',callback=self._dct_status_details)
                dpg.add_text('',tag='pyecr_dct_summary')
                dpg.add_text('',tag='pyecr_dct_result_warning')
                dpg.add_text('',tag='pyecr_dct_params_anchor')
        dpg.configure_item('pyecr_dct_context_tab',show=False)
        self._build_component_plot_series()
        self._build_lambda_scan_window()
        with dpg.window(label='DCT run details',tag='pyecr_dct_evidence_window',show=False,width=TOKENS.widths.report_path,height=TOKENS.heights.dialog):
            with dpg.group(horizontal=True):
                dpg.add_button(label='Close',callback=lambda:dpg.hide_item('pyecr_dct_evidence_window'))
                dpg.add_button(label='Copy run record',tag='pyecr_dct_copy_evidence',callback=self._copy_dct_evidence)
            dpg.add_input_text(tag='pyecr_dct_evidence_text',readonly=True,multiline=True,width=-1,height=-1)
        for key in ('source','settings','run','results'):
            for tag in ('pyecr_step_dct_'+key+'_status','pyecr_step_dct_'+key+'_warning','pyecr_stage_dct_'+key+'_reason'):dpg.hide_item(tag)
        self._register_owned_root('pyecr_dct_layout');self._register_owned_root('pyecr_dct_evidence_window')
        for tag in ('pyecr_run_dct','pyecr_cancel_dct'):add_action_tooltip(dpg,tag)

    def _toggle_dct_more_plots(self,*args):
        d=self._require_dpg();show=bool(d.get_value('pyecr_dct_more_plots'))
        for tag in ('pyecr_dct_evidence_tab','pyecr_dct_context_tab'):d.configure_item(tag,show=show)
        d.configure_item('pyecr_signed_dct_diagnostics_tab',show=show)
        selected=d.get_value('pyecr_dct_plot_tabs')
        if isinstance(selected,int):selected=d.get_item_alias(selected)
        if not show and selected not in ('pyecr_dct_spectrum_tab','pyecr_dct_reconstruction_tab'):
            d.set_value('pyecr_dct_plot_tabs','pyecr_dct_spectrum_tab')
            selected='pyecr_dct_spectrum_tab'
        reviewing=show and selected in ('pyecr_dct_evidence_tab','pyecr_signed_dct_diagnostics_tab','pyecr_dct_context_tab')
        d.configure_item('pyecr_component_dct_view_controls',show=self._dct_component_mode() and not reviewing)
        d.configure_item('pyecr_dct_result_compact',show=not reviewing)
        d.configure_item('pyecr_dct_context_action',show=not show)

    def _show_dct_context(self,*args):
        d=self._require_dpg();d.set_value('pyecr_dct_more_plots',True);self._toggle_dct_more_plots()
        d.set_value('pyecr_dct_plot_tabs','pyecr_dct_context_tab')

    def _build_dct_saved_plot(self,kind,title,height):
        dpg=self._require_dpg()
        with dpg.group(horizontal=True):
            dpg.add_spacer(tag='pyecr_dct_'+kind+'_spacer',width=TOKENS.spacing.spacer_small)
            with dpg.plot(label=title,tag='pyecr_dct_'+kind+'_plot',height=height,width=TOKENS.widths.model_selector):
                if kind in {'reconstruction','spectrum'}:dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis,label='log10(τ / s)' if kind=='spectrum' else 'Time (s)',tag='pyecr_dct_'+kind+'_x_axis')
                with dpg.plot_axis(dpg.mvYAxis,label='χ' if kind=='spectrum' else ('Normalized response' if kind=='reconstruction' else 'Observed − reconstructed'),tag='pyecr_dct_'+kind+'_y_axis'):
                    if kind=='reconstruction':
                        dpg.add_scatter_series([],[],label='Data',tag='pyecr_dct_data_series')
                        dpg.add_line_series([],[],label='Reconstruction',tag='pyecr_dct_reconstruction_series')
                    elif kind=='spectrum':
                        dpg.add_line_series([],[],label='Candidate time-scale distribution',tag='pyecr_dct_spectrum_series')
                        for name,label,color in (('positive','χ > 0: above terminal',(225,90,55,255)),('negative','χ < 0: below terminal',(60,145,235,255)),('zero','Zero',(160,160,160,255))):
                            dpg.add_line_series([],[],label=label,tag='pyecr_signed_dct_'+name,show=False)
                            with dpg.theme() as theme:
                                with dpg.theme_component(dpg.mvLineSeries):
                                    dpg.add_theme_color(dpg.mvPlotCol_Line,color,category=dpg.mvThemeCat_Plots)
                            dpg.bind_item_theme('pyecr_signed_dct_'+name,theme)
                    else:dpg.add_scatter_series([],[],label='Full residual',tag='pyecr_dct_residual_series')

    def _show_dct_evidence(self,*args):
        self._size_display_window('pyecr_dct_evidence_window');self._require_dpg().configure_item('pyecr_dct_evidence_window',show=True)
        self._refresh_dct_layout()

    def _dct_status_details(self,*args):
        if self._dct_signed_mode():self._show_dct_evidence()
        else:self._show_page_context(user_data='dct')

    def _copy_dct_evidence(self,*args):
        if not self._dct_signed_mode():self._ensure_legacy_dct_evidence()
        self._require_dpg().set_clipboard_text(self._signed_complete_evidence() if self._dct_signed_mode() else self._require_dpg().get_value('pyecr_dct_evidence_text'))

    def _refresh_dct_layout(self):
        dpg=self._require_dpg()
        if not getattr(dpg,'does_item_exist',lambda _:False)('pyecr_dct_layout'):return
        self._refresh_lambda_scan()
        dpg.set_value('pyecr_dct_export_status', self.controller.dct_export_status)
        dpg.configure_item('pyecr_dct_export_controls', show=self._dct_signed_mode())
        dpg.configure_item('pyecr_dct_export_run', enabled=self.controller._dct_export_task_id is None)
        dpg.configure_item('pyecr_dct_export_cancel', enabled=self.controller._dct_export_task_id is not None)
        session=self.controller.analysis_session;data=session.dataset;result=self.controller.session.dct_result
        signed_profile_mismatch = bool(data is not None and
            data.metadata.get('normalization_mode') == 'signed_terminal' and not self._dct_signed_mode())
        dpg.configure_item('pyecr_dct_use_signed_profile', show=signed_profile_mismatch)
        source='No imported dataset; units unconfirmed.' if data is None else (
            f"{data.metadata.get('source_path',data.metadata.get('dataset_name','in-memory data'))}\nRaw: {data.raw_time.size} rows · {data.metadata.get('raw_time_unit',data.metadata.get('time_unit','unknown'))}, {data.metadata.get('raw_signal_unit',data.metadata.get('signal_unit','unknown'))}\n"
            f"Processed: {data.processed_time.size} rows · {data.metadata.get('time_unit','unknown')}, {data.metadata.get('signal_unit','unknown')}")
        self._dct_full_source=source
        if self._dct_signed_mode():
            self._refresh_signed_dct_layout(source)
            self._refresh_dct_disclosure()
            self._refresh_dct_search_controls()
            view=getattr(self,'_component_display_view',None) if self._dct_component_mode() else getattr(self,'_signed_display_view',None)
            self._size_dct_panels(view if view is not None and view.x else None)
            self._refresh_physical_context()
            return
        dpg.set_value('pyecr_dct_source',source)
        mode=str(dpg.get_value('pyecr_dct_lambda_mode')).lower();manual=mode=='manual'
        dpg.configure_item('pyecr_dct_lambda',show=manual,enabled=manual)
        dpg.configure_item('pyecr_dct_gcv_lambda',show=not manual)
        dpg.set_value('pyecr_dct_lambda_reason','Manual λ: editable draft, used only by explicit Run.' if manual else 'GCV λ: read-only, selected by GCV for the recorded settings.')
        value='Automatic selection pending — Run GCV'
        if result is not None and result.config.lambda_mode=='gcv':
            try:matches=self._dct_config_from_widgets()==result.config
            except (ValueError,TypeError):matches=False
            value=f'{result.result.regularization_lambda:.5g} · saved'+('' if matches else ' · draft differs')
        dpg.set_value('pyecr_dct_gcv_lambda',value)
        config=session.current_analysis_config
        reason=self.controller.dct_unavailable_reason(use_classical_bi=bool(dpg.get_value('pyecr_dct_classical_bi')))
        if signed_profile_mismatch:
            reason = ('This source already uses Signed-DCT normalization. '
                      'Choose Signed representation using the button below; the accepted profile is retained.')
        dpg.set_value('pyecr_dct_run_reason',reason or 'Ready for explicit Run. Fit and geometry are optional physical interpretation only.')
        notice=('Model-independent normalized DCT; analytic single k has no D / Bi. Finite-D model suggestions are alternatives, not an automatic Apply.'
                if config is not None and config.model_id.value=='M7_SINGLE_K'
                else 'Candidate time scales only — no species / mechanism or automatic model Apply.')
        dpg.set_value('pyecr_dct_mode_notice',reason if reason else notice)
        warnings=() if result is None else tuple(dict.fromkeys((*result.result.resolution_warnings,*result.suggestion.warnings)))
        warning_text='' if not warnings else f'{len(warnings)} resolution / artifact warnings: '+warnings[0]+' — complete reasons in Summary / Full details.'
        dpg.set_value('pyecr_dct_result_warning',warning_text);dpg.configure_item('pyecr_dct_result_warning',show=bool(warnings))
        running=self.controller.state.tasks[GuiPage.DCT].status=='running'
        set_action_availability(dpg,'pyecr_run_dct',enabled=not reason and not running,
            reason=reason or 'A DCT task is running. Wait or Cancel.')
        if dpg.get_item_configuration('pyecr_dct_evidence_window')['show']:self._ensure_legacy_dct_evidence()
        self._refresh_dct_disclosure()
        self._refresh_dct_search_controls()
        self._size_dct_panels(result)
        self._refresh_physical_context()

    def _ensure_legacy_dct_evidence(self):
        d=self._require_dpg();result=self.controller.session.dct_result
        source=getattr(self,'_dct_full_source','');draft=tuple(str(d.get_value('pyecr_dct_'+key)) for key in ('tau_min','tau_max','n_tau','lambda_mode','lambda','lambdas'))
        old=self._dct_legacy_evidence_key
        if old is not None and old[0] is result and old[1:]==(source,draft):return
        self._dct_legacy_evidence_key=(result,source,draft)
        text=source+'\n\n'+d.get_value('pyecr_dct_summary')+'\n\n'+d.get_value('pyecr_dct_details')
        try:draft_evidence=self._dct_config_from_widgets().to_dict()
        except (ValueError,TypeError) as exc:draft_evidence={'error':str(exc)}
        text+='\n\nCurrent draft (not applied):\n'+json.dumps(draft_evidence,ensure_ascii=False,indent=2)
        if result is not None:text+='\n\nSaved result (read-only):\n'+json.dumps(result.to_dict(),ensure_ascii=False,indent=2,allow_nan=True)
        d.set_value('pyecr_dct_evidence_text',text)

    def _refresh_dct_disclosure(self):
        d=self._require_dpg();c=self.controller;data=c.analysis_session.dataset
        from pathlib import PureWindowsPath
        signed=self._dct_signed_mode();component=self._dct_component_mode()
        representation='Components / Signed' if component else 'Whole / Signed' if signed else 'Whole / Monotonic'
        d.set_value('pyecr_dct_regularization_label','Whole regularization' if component else 'Regularization')
        unit='unknown' if data is None else str(data.metadata.get('signal_unit','unknown'))
        source='Select an imported source.' if data is None else PureWindowsPath(str(data.metadata.get('source_path') or data.metadata.get('dataset_name') or 'Current dataset')).name+'\n'+representation+' · '+str(data.metadata.get('time_unit','unknown'))+' · '+unit
        d.set_value('pyecr_dct_source',source)
        d.set_value('pyecr_signed_dct_terminal_unit','Input signal unit: '+unit)
        d.set_value('pyecr_signed_dct_terminal_label','Terminal value ('+unit+')')
        external=signed and data is not None and data.metadata.get('normalization_mode')!='signed_terminal' and d.get_value('pyecr_signed_dct_reference')=='External known terminal'
        d.configure_item('pyecr_signed_dct_external',show=external)
        d.configure_item('pyecr_signed_dct_numeric',show=signed)
        d.configure_item('pyecr_component_dct_settings',show=component)
        d.configure_item('pyecr_component_dct_view_controls',show=component)
        if signed:
            job=self._selected_component_job() if component else self._selected_signed_job()
            available=job is not None and (job.bundle is not None if component else job.analysis is not None)
            saved=job.request.dataset if job else None
            label='' if saved is None else PureWindowsPath(str(saved.metadata.get('source_path') or saved.metadata.get('dataset_name') or 'Captured source')).name
            history=c.component_dct_service.history if component else c.signed_dct_service.history
            summary=('Select a stored run or Run for a new result.' if history else 'Run to create a stored result.') if job is None else ('History' if job.status=='stale' or job.stale_reasons else job.status.title())+' · '+label
            if available:
                result=job.bundle.overall_result if component else job.analysis.selected_result
                if component:
                    from pyecr.gui.component_dct import compact_component_metrics
                    summary+='\n'+compact_component_metrics(job.bundle,d.get_value('pyecr_component_dct_view'),d.get_value('pyecr_component_dct_scale'))
                elif result is not None and result.residuals:
                    from math import sqrt
                    summary+=f' · RMSE(Y) {sqrt(sum(v*v for v in result.residuals)/len(result.residuals)):.5g}'
            if job is not None and 'differs' in str(d.get_value('pyecr_dct_mode_notice')):summary+='\nShowing saved results; draft edits require Run.'
        else:
            result=c.session.dct_result
            summary=('Run to create a stored result.' if result is None else
                     f'Saved Monotonic result · RMSE {result.result.RMSE:.5g} · n={len(result.time_s)}\n'
                     'Plots show the saved run; current inputs/settings apply to the next Run.')
        d.set_value('pyecr_dct_result_compact',summary)
        view=(getattr(self,'_component_display_view',None) if component else getattr(self,'_signed_display_view',None)) if signed else None
        has_data=bool(view is not None and view.time_s) if signed else result is not None
        for kind in ('reconstruction','spectrum','residual'):d.configure_item('pyecr_dct_'+kind+'_plot',show=has_data)
        d.configure_item('pyecr_dct_fit_axes', enabled=has_data)
        for kind,values in (('lambda',() if view is None else view.lambda_x),('stability',() if view is None else view.peak_x)):
            d.configure_item('pyecr_dct_'+kind+'_plot',show=bool(values))
        warning=''
        if signed:
            if job is not None and (job.failure or job.stale_reasons):warning=brief_text(job.failure or '; '.join(job.stale_reasons),170)+' Details retain the full reason.'
            elif available:
                analysis=job.bundle.overall_analysis if component else job.analysis
                flags=analysis.flags if analysis else ()
                if flags:
                    from pyecr.gui.components.dct_review import primary_caution
                    warning=('Whole signal: ' if component else '')+brief_text(primary_caution(analysis),145)+' See Summary & peaks.'
                if component and any(job.bundle.unavailable_reasons):warning='Some components are unavailable. '+brief_text(next(x for x in job.bundle.unavailable_reasons if x),145)+' See Details.'
        elif result is not None:
            warnings=(*result.result.resolution_warnings,*result.suggestion.warnings)
            if warnings:warning='Resolution or interpretation caution: review Details before interpreting candidate peaks.'
        d.set_value('pyecr_dct_warning_compact',warning);d.configure_item('pyecr_dct_warning_compact',show=bool(warning))
        activity=c.dct_activity
        d.configure_item('pyecr_cancel_dct',show=bool(activity))
        if activity:
            cancelling=any(cancel for _,_,cancel in activity)
            status='Cancelling | waiting for worker exit' if cancelling else 'Running '+activity[0][0]+' DCT'
            set_action_availability(d,'pyecr_run_dct',enabled=False,reason='Wait for this task to exit, or Cancel before another Run.')
            set_action_availability(d,'pyecr_cancel_dct',enabled=not cancelling,reason='Cancellation already requested; waiting for actual worker exit.')
            d.set_value('pyecr_dct_run_reason','')
        else:
            task=c.state.tasks[GuiPage.DCT]
            status=(brief_text(job.failure or job.status,150) if signed and job is not None and job.status in ('cancelled','error') else
                    brief_text(task.message,150) if not signed and task.status in ('cancelled','error') else '')
            set_action_availability(d,'pyecr_cancel_dct',enabled=False,reason='No DCT computation is running.')
            ready=d.get_item_configuration('pyecr_run_dct')['enabled']
            reason=str(d.get_value('pyecr_dct_run_reason'))
            # Readiness removes ordinary instructional text, never the U01
            # purpose/reliability warning on an otherwise eligible parent.
            eligibility_warning=getattr(self,'_component_eligibility_warning','') if component else ''
            d.set_value('pyecr_dct_run_reason',eligibility_warning if ready else brief_text(reason,180))
        d.set_value('pyecr_dct_run_status',status);d.configure_item('pyecr_dct_run_status',show=bool(status))
        d.configure_item('pyecr_dct_run_reason',show=bool(d.get_value('pyecr_dct_run_reason')))
        self._refresh_dct_review()
        self._toggle_dct_more_plots()

    def _size_dct_panels(self,result):
        dpg=self._require_dpg()
        width,height=dpg.get_viewport_client_width(),dpg.get_viewport_client_height()
        top=max(0,dpg.get_item_pos('pyecr_dct_left')[1])
        g=geometry(width,height-top-TOKENS.spacing.spacer_medium);self._dct_geometry=g
        dpg.configure_item('pyecr_dct_left',width=g.left,height=g.panel_height)
        dpg.configure_item('pyecr_dct_right',width=g.right,height=g.panel_height)
        self._size_dct_review_tables(g.right)
        run_height=dpg.get_item_rect_size('pyecr_step_panel_dct_run')[1] or TOKENS.heights.history
        dpg.configure_item('pyecr_dct_parameters',height=max(TOKENS.heights.notes,g.panel_height-run_height-TOKENS.spacing.spacer_medium))
        pending=list(dpg.get_item_children('pyecr_dct_parameters',1) or [])+[dpg.get_alias_id('pyecr_dct_run_toolbar')]
        while pending:
            item=pending.pop()
            if dpg.get_item_info(item)['type'].endswith('mvText'):dpg.configure_item(item,wrap=max(TOKENS.widths.input_medium,g.left-2*TOKENS.dct.plot_gutter))
            pending.extend(dpg.get_item_children(item,1) or [])
        dpg.configure_item('pyecr_dct_result_compact',wrap=g.right-TOKENS.dct.plot_gutter)
        dpg.configure_item('pyecr_dct_warning_compact',wrap=g.right-TOKENS.dct.plot_gutter)
        for tag in ('pyecr_page_status_dct','pyecr_dct_lambda_reason','pyecr_dct_mode_notice','pyecr_dct_run_reason','pyecr_dct_explanation'):
            gutter=TOKENS.dct.plot_gutter if tag=='pyecr_page_status_dct' else 2*TOKENS.dct.plot_gutter
            dpg.configure_item(tag,wrap=max(TOKENS.widths.input_medium,g.left-gutter))
        for tag in ('pyecr_dct_summary','pyecr_dct_result_warning','pyecr_signed_dct_legend','pyecr_signed_dct_display_notice'):dpg.configure_item(tag,wrap=g.right-TOKENS.dct.plot_gutter)
        for tag in ('pyecr_dct_mode_recommendation','pyecr_signed_dct_terminal_label'):dpg.configure_item(tag,wrap=max(TOKENS.widths.input_medium,g.left-2*TOKENS.dct.plot_gutter))
        for kind in ('lambda','stability'):dpg.configure_item('pyecr_dct_'+kind+'_plot',width=-1,height=min(240,g.spectrum_height))
        for tag in ('review_scope','review_settings','review_resolution','review_peak_help','review_peak_empty','review_cautions',
                    'check_scope','check_lambda_note','check_stability','check_limits'):
            dpg.configure_item('pyecr_dct_'+tag,wrap=max(240,g.right-2*TOKENS.dct.plot_gutter))
        # Review tabs contain static captions as well as dynamic readouts.
        # Keep all ordinary text inside the actual results rail; table cells
        # retain their independently sized, horizontally scrollable columns.
        from pyecr.gui.components.compact_panels import content_width
        review_width=content_width(dpg,'pyecr_dct_plot_tabs')
        pending=list(dpg.get_item_children('pyecr_dct_right',1) or [])
        while pending:
            item=pending.pop();kind=dpg.get_item_info(item)['type'].split('::')[-1]
            if kind=='mvText':dpg.configure_item(item,wrap=review_width)
            if kind not in ('mvPlot','mvTable','mvChildWindow'):
                pending.extend(dpg.get_item_children(item,1) or [])
        # Measure real heading/tab/toolbar space after a rendered frame. This
        # accommodates fonts, warnings and outside legends without fitting axes.
        residual_state=dpg.get_item_state('pyecr_dct_residual_plot')
        right_state=dpg.get_item_state('pyecr_dct_right')
        residual_height=g.residual_height
        if residual_state.get('visible') and residual_state.get('rect_min') and right_state.get('rect_min'):
            inset=residual_state['rect_min'][1]-right_state['rect_min'][1]
            residual_height=max(TOKENS.dct.residual_min,min(TOKENS.dct.residual_max,
                int(g.panel_height-inset-TOKENS.dct.footer)))
        for kind,w,h in (('spectrum',g.spectrum_width,g.spectrum_height),('reconstruction',g.reconstruction_width,g.reconstruction_height),('residual',g.residual_width,residual_height)):
            # A scrollbar and the StepPanel's own padding reduce the real
            # content width. Bound the whole row using rendered coordinates;
            # viewport arithmetic alone clipped the last tick in narrow views.
            header=dpg.get_item_state('pyecr_stage_dct_results')
            plot=dpg.get_item_state('pyecr_dct_'+kind+'_plot')
            available=g.right-TOKENS.dct.plot_gutter
            if plot.get('visible') and header.get('rect_min') and header.get('rect_size') and plot.get('rect_min'):
                # mvSpacer has no rect state in DPG 2.3. Infer the stable row
                # origin from the rendered plot and its configured leading gap.
                leading=dpg.get_item_configuration('pyecr_dct_'+kind+'_spacer')['width']
                row_left=plot['rect_min'][0]-leading-TOKENS.spacing.item_spacing[0]
                available=min(available,int(header['rect_min'][0]+header['rect_size'][0]-row_left
                    -TOKENS.spacing.spacer_small-TOKENS.spacing.item_spacing[0]))
            w=min(w,max(1,available))
            dpg.configure_item('pyecr_dct_'+kind+'_plot',width=w,height=h)
            dpg.configure_item('pyecr_dct_'+kind+'_spacer',width=TOKENS.spacing.spacer_small+max(0,(available-w)//2))
            if result is not None and dpg.get_item_state('pyecr_dct_'+kind+'_plot').get('visible',False) and self._dct_fitted_axes.get(kind) is not result:
                if self._dct_pending_axes.get(kind) is result:
                    x=result.log10_tau if kind=='spectrum' else result.time_s
                    y=result.chi if kind=='spectrum' else result.reconstruction if kind=='reconstruction' else result.residuals
                    if not x or not y:continue
                    if kind=='spectrum' and self._dct_signed_mode():y=(*y,0.)
                    if kind=='reconstruction':y=(*y,*result.signal)
                    if self._dct_component_mode():
                        names={'spectrum':('component_1','component_2','sum'),
                            'reconstruction':('parent_fit','delta','overall'),'residual':('model_error',)}[kind]
                        for name in names:
                            extra_x,extra_y=dpg.get_value('pyecr_component_dct_'+name)
                            x=(*x,*extra_x);y=(*y,*extra_y)
                    lo,hi=min(y),max(y);pad=(hi-lo)*TOKENS.plot.fit_padding[1] or max(abs(lo),1.)*TOKENS.plot.fit_padding[1]
                    dpg.set_axis_limits('pyecr_dct_'+kind+'_x_axis',min(x),max(x))
                    dpg.set_axis_limits('pyecr_dct_'+kind+'_y_axis',lo-pad,hi+pad)
                    self._dct_fitted_axes[kind]=result
                    self._dct_release_axes[kind]=dpg.get_frame_count()+2
                else:self._dct_pending_axes[kind]=result
            if kind in self._dct_release_axes and dpg.get_frame_count()>=self._dct_release_axes[kind]:
                for axis in ('x','y'):dpg.set_axis_limits_auto('pyecr_dct_'+kind+'_'+axis+'_axis')
                del self._dct_release_axes[kind]
        if self._dct_signed_mode() and result is not None:
            for kind,x,y in (('lambda',result.lambda_x,result.lambda_score),('stability',result.peak_x,result.peak_shift)):
                if x and dpg.get_item_state('pyecr_dct_'+kind+'_plot').get('visible',False) and self._dct_fitted_axes.get(kind) is not result:
                    if self._dct_pending_axes.get(kind) is result:
                        lo,hi=min(x),max(x);pad=(hi-lo)*.05 or .1
                        dpg.set_axis_limits('pyecr_dct_'+kind+'_x_axis',lo-pad,hi+pad)
                        lo,hi=min(y),max(y);pad=(hi-lo)*.08 or max(abs(lo),1e-8)*.08
                        dpg.set_axis_limits('pyecr_dct_'+kind+'_y_axis',lo-pad,hi+pad)
                        self._dct_fitted_axes[kind]=result
                    else:self._dct_pending_axes[kind]=result
