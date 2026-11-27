"""Four Batch steps over the existing immutable Add and worker boundaries."""
import json
from pyecr.gui.theme import TOKENS
from pyecr.gui.state import GuiPage
from pyecr.gui.batch_display import BATCH_STEPS,MODE_TABS,input_rows,result_rows,result_counts,template_name,source_name,parameter_text,global_kinetic_text
from pyecr.gui.components.step_panel import step_panel
from pyecr.gui.components.action_state import add_action_tooltip
from pyecr.gui.workflow_display import brief_text
from pyecr.gui.components.tags import WidgetTags
from pyecr.services.batch_analysis import BatchAnalysisResult,GlobalAnalysisResult
from pyecr.gui.components.batch_figures import BatchFiguresMixin


class BatchLayoutMixin(BatchFiguresMixin):
    def _build_batch_layout(self):
        d=self._require_dpg();self._batch_mode='independent'
        self._batch_table_key=self._batch_result_key=self._batch_details_key=None
        self._batch_template_details_key=None;self._batch_card_texts=[];self._batch_input_texts=[]
        with d.group(tag='pyecr_batch_layout'):
            self._build_page_status('batch')
            with d.group(horizontal=True):
                with d.child_window(tag='pyecr_batch_workflow',width=TOKENS.analysis.left_compact,height=TOKENS.heights.dialog,border=False,no_scrollbar=True):
                    with d.child_window(tag='pyecr_batch_body',width=-1,height=TOKENS.heights.results,border=False):
                        with step_panel(d,'batch_inputs',dict(BATCH_STEPS)['inputs'],default_open=True):
                            with d.group(parent='pyecr_stage_batch_inputs_content'):
                                d.add_text('',tag='pyecr_batch_source',wrap=TOKENS.wraps.rail)
                                d.add_text('Name')
                                d.add_input_text(tag='pyecr_batch_record_id',width=-1,hint='New input name')
                                with d.group(horizontal=True):
                                    with d.group():
                                        d.add_text('Temperature')
                                        d.add_input_double(tag='pyecr_batch_temperature',default_value=800.,format='%.6g',width=TOKENS.widths.input_small)
                                    with d.group():
                                        d.add_text('Unit')
                                        d.add_combo(('K','degC'),tag='pyecr_batch_temperature_unit',default_value='K',width=TOKENS.widths.input_micro)
                                with d.group(tag='pyecr_batch_global_options',show=False):
                                    d.add_text('For the next Add only',wrap=TOKENS.wraps.rail)
                                    d.add_text('Residual weight (1 / noise SD)')
                                    d.add_input_double(tag='pyecr_batch_noise_weight',default_value=1.,min_value=0.,format='%.6g',width=-1)
                                    d.add_checkbox(label='Fit baseline',tag='pyecr_batch_fit_baseline',default_value=True)
                                    d.add_checkbox(label='Fit amplitude',tag='pyecr_batch_fit_amplitude',default_value=True)
                                    d.add_text('If unchecked: baseline = first signal; amplitude = last − first.',tag='pyecr_batch_fixed_observation_note',wrap=TOKENS.wraps.rail)
                                with d.group(horizontal=True):
                                    d.add_button(label='Add current',tag='pyecr_batch_add_current',callback=self._add_current_batch_dataset)
                                    d.add_button(label='Add project',tag='pyecr_batch_add_project',callback=self._show_batch_project_dialog)
                                d.add_text('Add captures this input. Later edits do not change existing rows.',wrap=TOKENS.wraps.rail)
                                d.add_group(tag='pyecr_batch_input_table_host')
                                d.add_text('Add a dataset or saved project.',tag='pyecr_batch_empty',wrap=TOKENS.wraps.rail)
                                d.add_button(label='Clear inputs',tag='pyecr_batch_clear_inputs',callback=self._clear_batch_inputs)
                        with step_panel(d,'batch_method',dict(BATCH_STEPS)['method'],default_open=True):
                            with d.group(parent='pyecr_stage_batch_method_content'):
                                with d.tab_bar(tag='pyecr_batch_mode_tabs',callback=self._select_batch_mode):
                                    with d.tab(label='Independent',tag='pyecr_batch_independent_tab'):pass
                                    with d.tab(label='Global',tag='pyecr_batch_global_tab'):pass
                                d.add_text('',tag='pyecr_batch_method_note',wrap=TOKENS.wraps.rail)
                                d.add_text('',tag='pyecr_batch_template_compact',wrap=TOKENS.wraps.rail)
                                d.add_button(label='Open Analysis Model',tag='pyecr_batch_open_model',callback=self._batch_open_model)
                                with d.collapsing_header(label='Template Details',tag='pyecr_batch_template_details',default_open=False):
                                    d.add_input_text(tag='pyecr_batch_template_text',readonly=True,multiline=True,width=-1,height=TOKENS.heights.history)
                                    d.add_group(tag='pyecr_batch_template_table_host')
                    # Same StepPanel surface as Analysis; outside the input scroller.
                    with step_panel(d,'batch_run',dict(BATCH_STEPS)['run'],default_open=True):
                        with d.group(parent='pyecr_stage_batch_run_content',tag='pyecr_batch_run_toolbar'):
                            with d.group(horizontal=True):
                                with d.group(tag='pyecr_batch_independent_actions'):
                                    d.add_button(label='Run independent',tag='pyecr_batch_run_independent',callback=self._run_batch_independent)
                                with d.group(tag='pyecr_batch_global_actions',show=False):
                                    d.add_button(label='Run Global',tag='pyecr_batch_run_global',callback=self._run_global_analysis)
                                d.add_button(label='Cancel',tag='pyecr_batch_cancel',callback=self._cancel_batch,enabled=False)
                            d.add_text('',tag='pyecr_batch_mode_reason',wrap=TOKENS.wraps.rail,color=TOKENS.colors.warning)
                with d.child_window(tag='pyecr_batch_results_scroll',width=-1,height=TOKENS.heights.dialog,border=False):
                    with step_panel(d,'batch_results',dict(BATCH_STEPS)['results'],default_open=True):
                        with d.group(parent='pyecr_stage_batch_results_content'):
                            d.add_text('',tag='pyecr_batch_result_summary',wrap=TOKENS.wraps.map)
                            d.add_text('',tag='pyecr_batch_cancel_summary',wrap=TOKENS.wraps.map,color=TOKENS.colors.warning)
                            d.add_text('',tag='pyecr_batch_global_summary',wrap=TOKENS.wraps.map)
                            self._build_batch_figure_controls()
                            d.add_group(tag='pyecr_batch_result_table_host')
                            d.add_text('',tag='pyecr_batch_arrhenius_summary',wrap=TOKENS.wraps.map)
                            d.add_text('',tag='pyecr_batch_result_warnings',wrap=TOKENS.wraps.map,color=TOKENS.colors.warning)
                            with d.group(horizontal=True):
                                d.add_button(label='Details',tag='pyecr_batch_details_action',callback=self._show_batch_details)
                                d.add_button(label='Open Reports',tag='pyecr_batch_export_options',callback=self._batch_open_reports)
            with d.group(tag='pyecr_batch_legacy',show=False):
                for tag in ('pyecr_batch_inputs','pyecr_batch_results','pyecr_batch_independent_reason','pyecr_batch_global_reason'):
                    d.add_text('',tag=tag)
        with d.window(label='Batch Details',tag='pyecr_batch_details_window',show=False,width=TOKENS.widths.report_path,height=TOKENS.heights.dialog):
            with d.group(horizontal=True):
                d.add_button(label='Close',callback=lambda:d.hide_item('pyecr_batch_details_window'))
                d.add_button(label='Copy complete details',tag='pyecr_batch_copy_details',callback=self._copy_batch_details)
            d.add_input_text(tag='pyecr_batch_details_text',readonly=True,multiline=True,width=-1,height=-1)
        for root in ('pyecr_batch_layout','pyecr_batch_details_window'):self._register_owned_root(root)
        for key,_ in BATCH_STEPS:
            for tag in ('pyecr_step_batch_'+key+'_status','pyecr_step_batch_'+key+'_warning','pyecr_stage_batch_'+key+'_reason'):
                d.configure_item(tag,show=False)
        self._batch_static_texts=[];pending=[d.get_alias_id('pyecr_batch_workflow')]
        while pending:
            item=pending.pop()
            if d.get_item_info(item)['type'].endswith('mvText'):self._batch_static_texts.append(item)
            pending.extend(d.get_item_children(item,1) or [])
        for tag in ('pyecr_batch_add_current','pyecr_batch_run_independent','pyecr_batch_run_global','pyecr_batch_cancel'):add_action_tooltip(d,tag)

    def _select_batch_mode(self,sender=None,app_data=None,user_data=None):
        d=self._require_dpg();selected=d.get_value('pyecr_batch_mode_tabs') if app_data is None else app_data
        if isinstance(selected,int):selected=d.get_item_alias(selected)
        self._batch_mode=MODE_TABS[selected];d.set_value('pyecr_batch_mode_tabs',selected)
        self._refresh_batch_layout()

    def _batch_open_model(self,*args):
        d=self._require_dpg();d.set_value(WidgetTags.PAGES,WidgetTags.page('analysis'))
        self._on_tab_change(None,WidgetTags.page('analysis'),None);self._navigate_workflow(user_data='model')

    def _batch_open_reports(self,*args):
        d=self._require_dpg();d.set_value(WidgetTags.PAGES,WidgetTags.page('reports'))
        self._on_tab_change(None,WidgetTags.page('reports'),None)

    def _remove_batch_row(self,sender=None,app_data=None,user_data=None):
        try:self.controller.remove_batch_input(user_data)
        except Exception as exc:
            import traceback
            self.controller.record_callback_exception(GuiPage.BATCH,exc,traceback.format_exc())
        self._refresh_widgets()

    def _show_batch_details(self,*args):
        self._size_display_window('pyecr_batch_details_window');self._require_dpg().show_item('pyecr_batch_details_window')
        self._refresh_batch_layout()

    def _copy_batch_details(self,*args):
        self._ensure_batch_details()
        self._require_dpg().set_clipboard_text(self._require_dpg().get_value('pyecr_batch_details_text'))

    def _ensure_batch_details(self):
        d=self._require_dpg();s=self.controller.session;mode=self._batch_mode
        result=s.batch_result if mode=='independent' else s.global_result;template=self.controller.current_analysis_config
        task=self.controller.state.tasks[GuiPage.BATCH]
        key=(mode,id(result),id(template),tuple(id(e) for e in s.batch_inputs),task.status,task.task_id,task.message)
        if key==self._batch_details_key:return
        from pyecr.report.snapshot import _json_safe
        def entry(e):
            ds=e.dataset
            return dict(record_id=e.record_id,temperature=e.temperature,temperature_unit=e.temperature_unit.value,temperature_K=e.temperature_K,
                source_project=e.source_project,noise_weight=e.noise_weight,fit_baseline=e.fit_baseline,fit_amplitude=e.fit_amplitude,
                baseline=e.baseline,amplitude=e.amplitude,analysis_config=None if e.analysis_config is None else e.analysis_config.to_dict(),
                dataset=dict(raw_time=ds.raw_time.tolist(),raw_signal=ds.raw_signal.tolist(),processed_time=ds.processed_time.tolist(),
                    processed_signal=ds.processed_signal.tolist(),mask=ds.mask.tolist(),metadata=ds.metadata,history=[h.to_dict() for h in ds.history]))
        context=self.controller.batch_result_context(result)
        value=dict(version='batch-ui/2',mode=mode,current_inputs=[entry(e) for e in s.batch_inputs],
            current_template=None if template is None else template.to_dict(),counts=result_counts(result,task.status),
            stored_result=None if result is None else result.to_dict(),
            published_source=None if context is None else dict(job_id=context[3],template=context[2].to_dict(),inputs=[entry(e) for e in context[1]]),
            publication_source_note='Original captured inputs are unavailable for restored historical results.' if result is not None and context is None else '',
            task=dict(status=task.status,message=task.message))
        d.set_value('pyecr_batch_details_text',json.dumps(_json_safe(value),ensure_ascii=False,indent=2,allow_nan=False))
        self._batch_details_key=key

    def _refresh_batch_layout(self):
        d=self._require_dpg()
        if not d.does_item_exist('pyecr_batch_layout'):return
        c=self.controller;s=c.session;t=c.state.tasks[GuiPage.BATCH];cfg=c.current_analysis_config;mode=self._batch_mode
        result=s.batch_result if mode=='independent' else s.global_result
        self._refresh_batch_figures(result,mode)
        for tag,shown in (('pyecr_batch_global_options',mode=='global'),('pyecr_batch_independent_actions',mode=='independent'),('pyecr_batch_global_actions',mode=='global')):d.configure_item(tag,show=shown)
        for tag in ('pyecr_batch_add_project','pyecr_batch_clear_inputs'):d.configure_item(tag,enabled=t.status!='running' and (tag.endswith('project') or bool(s.batch_inputs)))
        ready=c.can_run_batch_independent if mode=='independent' else c.can_run_global_analysis
        supported=cfg is None or (c.batch_analysis_service.supports_independent(cfg.model_id) if mode=='independent' else c.batch_analysis_service.supports_global(cfg.model_id))
        reason=('Open Analysis Model and apply a template.' if cfg is None else 'Choose a supported template in Analysis Model.' if not supported or 'fixed_d' in cfg.model_specific else
            ('Add at least one input.' if mode=='independent' else 'Add at least two compatible inputs.') if not ready else '')
        if mode=='global':reason=c.batch_analysis_service.global_unavailable_reason(s.batch_inputs,cfg)
        elif cfg is not None and not supported:reason='Independent supports models 01 through 06.'
        if t.status=='running':
            reason='Cancelling…' if t.cancellation_requested else 'Running…'
            d.set_value('pyecr_stage_batch_run',True)
        d.set_value('pyecr_batch_mode_reason',reason);d.configure_item('pyecr_batch_mode_reason',show=bool(reason))
        ds=c.analysis_session.dataset
        source='Import data in Analysis, or add a saved project.' if ds is None else 'Current: '+str(ds.metadata.get('dataset_name') or 'Analysis dataset')
        d.set_value('pyecr_batch_source',brief_text(source,110))
        d.set_value('pyecr_batch_method_note','Separate fits; Arrhenius includes every eligible D and k, in separate plots.' if mode=='independent' else 'Shared '+global_kinetic_text(cfg)+'. Same temperature and Ideal step. Set next-input options in 01 before Add.')
        dual_global=cfg is not None and cfg.model_id.value=='M3_DUAL_HO_DK'
        d.configure_item('pyecr_batch_fit_amplitude',label='Fit both amplitudes' if dual_global else 'Fit amplitude')
        d.set_value('pyecr_batch_fixed_observation_note','If unchecked: use captured Model y0 / A_H / A_O. These Global switches set local observation freedom.' if dual_global else 'If unchecked: baseline = first signal; amplitude = last − first.')
        d.set_value('pyecr_batch_template_compact',template_name(cfg))
        d.configure_item('pyecr_batch_empty',show=not s.batch_inputs)
        key=(tuple(id(e) for e in s.batch_inputs),mode,id(result),t.status)
        if key!=self._batch_table_key:
            self._batch_table_key=key;d.delete_item('pyecr_batch_input_table_host',children_only=True);self._batch_input_texts=[]
            rows=input_rows(s.batch_inputs,cfg,mode,result,t.status)
            for e,row in zip(s.batch_inputs,rows):
                with d.group(parent='pyecr_batch_input_table_host'):
                    status={'Joint / row N/A':'Joint result','Cancelled / row N/A':'No published result','Running / pending':'Pending'}.get(row['status'],row['status'])
                    for text in (e.record_id,source_name(e),f'{row["temperature"]} {row["unit"]} · {status}'):
                        self._batch_input_texts.append(d.add_text(text,wrap=TOKENS.wraps.rail))
                    if mode=='global':
                        self._batch_input_texts.append(d.add_text(f'Captured: weight {e.noise_weight:.6g}; baseline '+('Fit' if e.fit_baseline else 'Fixed')+'; amplitude '+('Fit' if e.fit_amplitude else 'Fixed'),wrap=TOKENS.wraps.rail))
                    d.add_button(label='Remove',callback=self._remove_batch_row,user_data=e.record_id,enabled=t.status!='running')
                    d.add_separator()
            d.set_value('pyecr_batch_inputs','\n'.join(e.record_id for e in s.batch_inputs))
        if d.get_value('pyecr_batch_template_details') and id(cfg)!=self._batch_template_details_key:
            d.set_value('pyecr_batch_template_text','No applied template.' if cfg is None else json.dumps(cfg.to_dict(),ensure_ascii=False,indent=2))
            self._batch_template_details_key=id(cfg)
        if (mode,id(result))!=self._batch_result_key:
            self._batch_result_key=(mode,id(result));d.delete_item('pyecr_batch_result_table_host',children_only=True);self._batch_card_texts=[]
            def text(value,warning=False):
                item=d.add_text(value,parent='pyecr_batch_result_table_host',wrap=TOKENS.wraps.map,color=TOKENS.colors.warning if warning else TOKENS.steps.text)
                self._batch_card_texts.append(item)
            arr=[];shared=[]
            if isinstance(result,BatchAnalysisResult):
                for r in result.records:
                    text(r.record_id+' · '+('Failed' if not r.fit_success else 'Reliable' if r.reliable else 'Reliability warning'),not r.reliable)
                    text(f'{r.temperature_K:.6g} K\n'+parameter_text(r.params))
                    if r.RMSE is not None:text(f'RMSE {r.RMSE:.6g}'+(' · R² '+f'{r.R2:.6g}' if r.R2 is not None else ''))
                    reason=r.error or next(iter(r.reliability_reasons),'')
                    if reason:text(brief_text(reason,150)+(' Review input and template.' if not r.fit_success else ''),True)
                    d.add_separator(parent='pyecr_batch_result_table_host')
                for v in result.arrhenius.values():
                    name=v.parameter.replace('_cm2_s','').replace('_cm_s','')
                    arr.append(f'{name}: Ea {v.activation_energy_J_mol/1000:.6g} kJ/mol\nPrefactor {v.pre_exponential_factor:.6g} · R² {v.R2:.6g}')
                if not arr:arr=['No eligible stored Arrhenius fit. Review record reliability in Details.']
            elif isinstance(result,GlobalAnalysisResult):
                shared=[parameter_text(result.shared_params), 'Local parameter separation supported' if result.reliable else 'Local parameter separation not established']
                for curve in result.curves:
                    text(curve.curve_id)
                    rmse=(sum(x*x for x in curve.residuals)/len(curve.residuals))**.5 if curve.residuals else None
                    text('Baseline '+f'{curve.baseline:.6g}'+'\n'+parameter_text(curve.component_amplitudes) if curve.component_amplitudes and curve.baseline is not None else f'Baseline {curve.baseline:.6g} · amplitude {curve.amplitude:.6g}' if curve.baseline is not None and curve.amplitude is not None else 'Observation parameters: Details')
                    text('Residual RMSE '+('unavailable' if rmse is None else f'{rmse:.6g}')+' · full curves and residuals: Details')
                    d.add_separator(parent='pyecr_batch_result_table_host')
            d.set_value('pyecr_batch_arrhenius_summary','Arrhenius\n'+'\n'.join(arr) if arr else '')
            d.set_value('pyecr_batch_global_summary','\n'.join(shared))
            warnings=() if result is None else result.warnings
            warning=''
            if warnings:
                first=warnings[0]
                warning=('Arrhenius unavailable: too few eligible results. Review record reliability.' if first.startswith('ARRHENIUS_SKIPPED') else
                    'Negative activation energy. Review the model and temperature inputs.' if first.startswith('NEGATIVE_ACTIVATION_ENERGY') else
                    'Arrhenius unavailable. Review temperatures and record reliability.' if first.startswith('ARRHENIUS_FAILED') else
                    'Result warnings: review record status and Details.')
            d.set_value('pyecr_batch_result_warnings',warning)
            summary='Run to create results.' if result is None else f'Saved Independent · {len(result.records)} records · {sum(not r.fit_success for r in result.records)} failed' if isinstance(result,BatchAnalysisResult) else f'Saved Global · {len(result.curves)} curves · shared '+global_kinetic_text(result)
            d.set_value('pyecr_batch_result_summary',summary);d.set_value('pyecr_batch_results',summary)
        retained=' Previous complete result retained.' if result is not None and (t.status in ('cancelled','error') or 'not published' in t.message) else ''
        message=('Cancelled.'+retained if t.status=='cancelled' else brief_text(t.message,160)+retained if t.status=='error' or (t.status=='warning' and (result is None or 'not published' in t.message)) else '')
        d.set_value('pyecr_batch_cancel_summary',message)
        d.configure_item('pyecr_batch_cancel',show=t.status=='running')
        for tag in ('pyecr_batch_cancel_summary','pyecr_batch_result_warnings','pyecr_batch_arrhenius_summary','pyecr_batch_global_summary'):d.configure_item(tag,show=bool(d.get_value(tag)))
        if d.get_item_configuration('pyecr_batch_details_window')['show']:self._ensure_batch_details()
        width=d.get_viewport_client_width();height=d.get_viewport_client_height()
        left=min(TOKENS.analysis.left_compact,max(TOKENS.analysis.left_min,width*.34))
        top=d.get_item_pos('pyecr_batch_workflow')[1]
        available=max(TOKENS.heights.results,height-max(0,top)-TOKENS.spacing.spacer_medium)
        d.configure_item('pyecr_batch_workflow',width=left,height=available);d.configure_item('pyecr_batch_results_scroll',height=available)
        run_height=d.get_item_rect_size('pyecr_step_panel_batch_run')[1] or 145
        d.configure_item('pyecr_batch_body',height=max(TOKENS.heights.notes,available-run_height-TOKENS.spacing.spacer_medium))
        for item in self._batch_input_texts+self._batch_static_texts:d.configure_item(item,wrap=max(TOKENS.widths.input_medium,left-65))
        for item in self._batch_card_texts:d.configure_item(item,wrap=max(TOKENS.widths.panel_field,width-left-100))
        for tag in ('pyecr_batch_result_summary','pyecr_batch_cancel_summary','pyecr_batch_global_summary','pyecr_batch_arrhenius_summary','pyecr_batch_result_warnings'):d.configure_item(tag,wrap=max(TOKENS.widths.panel_field,width-left-100))
