"""R05 draft controls and pinned actions. Existing backend dispatch only."""
import json

from pyecr.config import AnalysisConfig
from pyecr.contracts import FitControlBinding, SourceIdentity, TerminationPolicy
from pyecr.gui.fit_control_draft import VERSION, MONITORED_FIELDS, draft_identity, unavailable_reason
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS, SECTION_FONT_TAG, SECONDARY_FONT_TAG
from pyecr.gui.components.action_state import set_action_availability
from pyecr.gui.local_intervals import LocalIntervalLayoutMixin
from pyecr.gui.noise_reference import NoiseReferenceLayoutMixin
from pyecr.gui.transport_summary import TransportSummaryLayoutMixin
from pyecr.gui.components.fit_display import FitDisplayMixin
from pyecr.gui.components.analysis_results import AnalysisResultsMixin


class FitControlLayoutMixin(AnalysisResultsMixin,FitDisplayMixin,TransportSummaryLayoutMixin,NoiseReferenceLayoutMixin,LocalIntervalLayoutMixin):
    def _r05_heading(self,text):
        dpg=self._require_dpg();tag=dpg.add_text(text)
        if dpg.does_item_exist(SECTION_FONT_TAG):dpg.bind_item_font(tag,SECTION_FONT_TAG)
        return tag

    def _r05_note(self,text,**kwargs):
        dpg=self._require_dpg();tag=dpg.add_text(text,color=TOKENS.steps.secondary,wrap=TOKENS.wraps.stage,**kwargs)
        if dpg.does_item_exist(SECONDARY_FONT_TAG):dpg.bind_item_font(tag,SECONDARY_FONT_TAG)
        return tag

    def _build_r05_control_layout(self):
        dpg=self._require_dpg()
        self._r05_applied_controls=None;self._r05_run_bindings={};self._r05_draft_revision=0
        self._r05_actual_key=None;self._r05_actual_record=None
        with dpg.group(parent='pyecr_stage_model_content',before='pyecr_step_model_details',tag='pyecr_model_action_bar',horizontal=True):
            pass
        for tag in ('pyecr_model_apply','pyecr_parameter_table_expand','pyecr_parameter_copy_draft'):dpg.move_item(tag,parent='pyecr_model_action_bar')
        with dpg.child_window(parent='pyecr_stage_model_content',before='pyecr_step_model_details',
                              tag='pyecr_model_form_scroll',width=-1,height=TOKENS.steps.model_form_height,border=False):
            pass
        dpg.move_item('pyecr_model_schema_container',parent='pyecr_model_form_scroll')
        self._move_model_evidence()
        with dpg.group(parent='pyecr_stage_fit_content',before='pyecr_step_fit_details',tag='pyecr_fit_controls_review'):
            dpg.add_text('Current: no completed Fit.',tag='pyecr_fit_actual_controls',wrap=TOKENS.wraps.stage)
            with dpg.child_window(tag='pyecr_fit_controls_scroll',width=-1,height=TOKENS.steps.fit_controls_height,border=False):
                self._r05_heading('Uncertainty — draft')
                dpg.add_input_text(label='Confidence (%)',tag='pyecr_fit_confidence_percent',default_value='95',
                                   decimal=True,scientific=True,width=-1,callback=self._fit_control_draft_changed)
                self._r05_note('Blank = 95%. Recompute applies output C to this parent without a refit. Run retains the fixed 95% gate.')
                self._r05_heading('Stopping — draft')
                dpg.add_radio_button(('Standard','Monitored'),tag='pyecr_fit_stopping_mode',default_value='Standard',
                                     horizontal=True,callback=self._fit_control_draft_changed)
                with dpg.group(tag='pyecr_fit_monitored_form',show=False):
                    self._r05_note('TRF only. Accepted-step AND monitoring; native success and reliability remain separate. No automatic polish.')
                    with dpg.table(width=-1,header_row=False,policy=dpg.mvTable_SizingStretchProp):
                        dpg.add_table_column(init_width_or_weight=1.8);dpg.add_table_column(init_width_or_weight=1)
                        for name,label in (('objective_change_tolerance','Objective tolerance'),('parameter_change_tolerance','Parameter tolerance'),
                                           ('patience','Patience (accepted iter.)'),('max_iterations','Accepted step cap / call'),('max_total_nfev','nfev budget / job')):
                            with dpg.table_row():
                                dpg.add_text(label)
                                dpg.add_input_text(tag='pyecr_fit_control_'+name,default_value={
                                    'objective_change_tolerance':'1e-7','parameter_change_tolerance':'1e-6','patience':'7',
                                    'max_iterations':'1000','max_total_nfev':'20000'}[name],width=-1,
                                                   callback=self._fit_control_draft_changed)
                    with dpg.collapsing_header(label='Optimizer settings (Expert)',default_open=False):
                        for name,label,value in (('ftol','Optimizer ftol (blank = disabled)',''),('xtol','Optimizer xtol (blank = disabled)',''),('gtol','Optimizer gtol','1e-8'),
                                ('max_wall_seconds','Wall budget (s; blank = unset)',''),('max_trace_records','Trace record budget','10000')):
                            dpg.add_input_text(label=label,tag='pyecr_fit_control_'+name,default_value=value,width=-1,
                                               callback=self._fit_control_draft_changed)
                        self._r05_note('Optimizer tolerances do not edit production-kernel tolerances in Model Expert settings.')
                self._r05_note('Standard uses the applied Model configuration. Monitored fields apply only in Monitored mode.',tag='pyecr_fit_legacy_note')
            dpg.add_text('',tag='pyecr_fit_control_draft_status',wrap=TOKENS.wraps.stage)
            dpg.add_text('',tag='pyecr_fit_control_reason',wrap=TOKENS.wraps.stage,color=TOKENS.colors.warning)
            dpg.add_button(label='Apply controls',tag='pyecr_fit_controls_apply',callback=self._apply_fit_control_draft,enabled=False)
        dpg.add_input_text(parent='pyecr_step_fit_details',tag='pyecr_fit_controls_details',readonly=True,multiline=True,
                           width=-1,height=TOKENS.steps.details_height)
        # Run/Cancel/Apply and blocking reasons precede the scrollable forms.
        dpg.move_item('pyecr_fit_controls_apply',parent=dpg.get_item_parent('pyecr_step_action_pyecr_run_fit'))
        dpg.move_item('pyecr_fit_control_reason',parent='pyecr_fit_controls_review',before='pyecr_fit_actual_controls')
        dpg.move_item('pyecr_fit_control_draft_status',parent='pyecr_fit_controls_review',before='pyecr_fit_controls_scroll')
        self._prepare_owned_form('pyecr_fit_controls_review')
        self._r05_controls_ready=True
        self._build_r06_intervals()
        self._build_r10_noise()
        self._build_r11_transport()
        self._build_fit_display()
        self._build_result_summary()

    def _move_model_evidence(self):
        dpg=self._require_dpg()
        if not dpg.does_item_exist('pyecr_step_model_details'):return
        for tag in ('pyecr_model_assumptions','pyecr_model_limitations'):
            if dpg.does_item_exist(tag):
                item=dpg.get_alias_id(tag)
                wrapper=self._owned_form_fields.get(item,(item,))[0]
                dpg.move_item(wrapper,parent='pyecr_step_model_details')

    def _fit_control_values(self):
        dpg=self._require_dpg()
        selected=dpg.get_value('pyecr_fit_stopping_mode')
        values=dict(confidence_percent=dpg.get_value('pyecr_fit_confidence_percent'),mode='Legacy' if selected=='Standard' else selected)
        if values['mode']=='Monitored':
            values.update({k:dpg.get_value('pyecr_fit_control_'+k) for k in MONITORED_FIELDS})
        return values

    def _fit_control_draft_changed(self,*args):
        self._r05_draft_revision+=1
        self._refresh_widgets()

    def _r05_control_binding(self):
        s=self.controller.analysis_session
        values=self._fit_control_values();controls,reason=unavailable_reason(values,s.current_analysis_config)
        if reason:raise ValueError(reason)
        if s.dataset is None or s.current_analysis_config is None:raise ValueError('Import data and Apply a Model configuration first.')
        recipe=s.preprocessing.config
        return FitControlBinding.capture(SourceIdentity.capture(s.dataset,recipe,config=s.current_analysis_config),controls)

    def _apply_fit_control_draft(self,*args):
        # Applying controls stores the intention in UI memory; no Fit or data
        # calculation. Run is a separate explicit action.
        try:self._r05_applied_controls=self._r05_control_binding()
        except (ValueError,TypeError) as exc:
            self._require_dpg().set_value('pyecr_fit_control_reason','Reason: '+str(exc))
            return
        self._refresh_widgets()

    def _record_r05_legacy_run(self,task_id,binding,dataset,recipe):
        provenance=self.controller.analysis_session.fit_job_provenance
        if provenance.get('effective_config'):
            # This is the actual old prepare_fit request, including its seed,
            # not a new call to prepare_fit or a guessed optimizer result.
            config=AnalysisConfig.from_dict(provenance['effective_config'])
            from dataclasses import replace
            controls=binding.controls if binding.controls.termination.mode=='monitored' else replace(binding.controls,termination=TerminationPolicy.legacy_for(config))
            binding=FitControlBinding.capture(SourceIdentity.capture(dataset,recipe,config=config),controls)
        self._r05_run_bindings[task_id]=binding

    def _actual_fit_controls(self):
        s=self.controller.analysis_session;fit=s.fit_result
        key=(id(fit),id(s.current_analysis_config),s.fit_stale)
        if key==self._r05_actual_key:return self._r05_actual_record
        self._r05_actual_key=key;record=None
        if fit is not None:
            config=None
            payload=fit.diagnostics.get('fitting_randomness',{}).get('effective_config')
            if payload:
                try:config=AnalysisConfig.from_dict(payload)
                except (ValueError,TypeError):pass
            if config is None:
                config=next((e.analysis_config for e in s.fit_history if e.fit_result is fit),None)
            if config is None and not s.fit_stale:config=s.current_analysis_config
            record=dict(status='Stale / History' if s.fit_stale else 'Current',confidence_level=.95,
                        ci_source='Existing local-95 calculation path; no output-CI control sidecar',termination='Legacy',native_policy=None,
                        trace_status='Standard stopping uses the saved native optimizer settings; no monitored trace is collected.')
            if config is not None:
                try:record['native_policy']=TerminationPolicy.legacy_for(config).to_dict()
                except (ValueError,TypeError) as exc:record['native_policy_reason']=str(exc)
            if record['native_policy'] is None:record.setdefault('native_policy_reason','Original configuration not recoverable; optimizer tolerances unavailable for this history.')
            monitored=self.controller.monitored_fit_records.get(id(fit))
            if monitored is not None:
                record.update(termination='Monitored',native_policy=monitored['binding']['controls']['termination'],
                    trace_status=monitored['execution_outcome'],execution=monitored)
        self._r05_actual_record=record
        return record

    def _refresh_r05_control_layout(self):
        if not getattr(self,'_r05_controls_ready',False):return
        dpg=self._require_dpg();s=self.controller.analysis_session;task=self.controller.state.tasks[GuiPage.FIT]
        values=self._fit_control_values();controls,reason=unavailable_reason(values,s.current_analysis_config)
        monitored=values['mode']=='Monitored'
        dpg.configure_item('pyecr_fit_monitored_form',show=monitored)
        dpg.configure_item('pyecr_fit_legacy_note',show=not monitored)
        # Run availability intersects the existing readiness/reliability gate.
        # Cancel never depends on draft validity or unsupported intentions.
        if reason:
            set_action_availability(dpg,'pyecr_run_fit',enabled=False,reason=reason)
            dpg.configure_item('pyecr_step_action_pyecr_run_fit',enabled=False)
            dpg.set_value('pyecr_step_fit_status','Run unavailable for this draft.')
        can_apply=not reason and s.dataset is not None and s.current_analysis_config is not None and task.status!='running'
        dpg.configure_item('pyecr_fit_controls_apply',enabled=can_apply)
        prefix='Draft controls: '+('Standard' if values['mode']=='Legacy' else values['mode'])
        dpg.set_value('pyecr_fit_control_draft_status',prefix+' (not applied).')
        dpg.set_value('pyecr_fit_control_reason','Reason: '+reason if reason else '')
        dpg.configure_item('pyecr_fit_control_reason',show=bool(reason))
        record=self._actual_fit_controls()
        stopping_label=('Standard' if record is not None and str(record['termination']).lower()=='legacy' else '' if record is None else record['termination'])
        dpg.set_value('pyecr_fit_actual_controls','Current: no completed Fit.' if record is None else
                      f'{record["status"]}: basic local CI 95%; stopping {stopping_label}. Output draft is separate.')
        if task.status=='running':
            frozen=self._r05_run_bindings.get(task.task_id)
            dpg.set_value('pyecr_fit_actual_controls','Running '+str(task.task_id)+(f'; frozen {frozen.controls.termination.mode} / 95% gate. Output CI needs Recompute.' if frozen else '; existing task, UI control record unavailable.'))
        if dpg.does_item_exist('pyecr_fit_result_heading'):
            dpg.set_value('pyecr_fit_result_heading','Parameters · 95% CI · Reliability' if record else 'Parameters · Local intervals · Reliability')
        applied=self._r05_applied_controls
        applied_state='Not applied'
        if applied is not None:
            identity_key=(id(applied),id(s),id(s.dataset),id(s.preprocessing.config) if s.preprocessing else None,
                id(s.current_analysis_config),s.current_revision)
            if identity_key!=getattr(self,'_fit_applied_display_key',None):
                try:applied.assert_execution_source(SourceIdentity.capture(s.dataset,s.preprocessing.config,config=s.current_analysis_config));self._fit_applied_display_state='Current source'
                except (ValueError,TypeError,AttributeError):self._fit_applied_display_state='Stale source'
                self._fit_applied_display_key=identity_key
            applied_state=self._fit_applied_display_state
            if applied_state=='Current source' and controls==applied.controls:
                dpg.set_value('pyecr_fit_control_draft_status','Controls applied: '+('Standard' if values['mode']=='Legacy' else values['mode'])+'.')
            elif applied_state=='Stale source':dpg.set_value('pyecr_fit_control_draft_status',prefix+'; applied controls Stale.')
        # Completed evidence is detached and immutable in the worker/store flow.
        # Serialize its full arrays only when an identity/status/draft changes,
        # rather than doing O(N * starts * iterations) work every render frame.
        full=self._fit_details_requested()
        cache_key=(full,self._r05_draft_revision,draft_identity(values),reason,id(applied),applied_state,
            self._r05_actual_key,task.task_id,task.status,len(self._r05_run_bindings),
            tuple((k,id(v)) for k,v in self.controller.monitored_runs.items()))
        if cache_key!=getattr(self,'_r09_details_cache_key',None):
            details=dict(version=VERSION,draft_revision=self._r05_draft_revision,draft_id=draft_identity(values),draft_values=values,
                parsed_intention=None if controls is None else controls.to_dict(),reason=reason,
                applied_controls=None if applied is None else applied.to_dict(),applied_state=applied_state,
                actual_result=record if full or record is None else {k:v for k,v in record.items() if k!='execution'},
                running_task_id=task.task_id if task.status=='running' else None,
                frozen_runs={k:v.to_dict() for k,v in self._r05_run_bindings.items()} if full else {},
                monitored_history=self.controller.monitored_runs if full else {},
                disclosure='Complete execution records on Details or Copy complete details.' if not full else 'Complete evidence requested.',
                backend_scope='Standard uses native stopping; Monitored TRF also checks accepted steps. Output CI Recompute and fixed95 gates remain separate.')
            self._r09_details_cache_text=json.dumps(details,ensure_ascii=False,indent=2,allow_nan=False)
            self._r09_details_cache_key=cache_key
        text=self._r09_details_cache_text
        if dpg.get_value('pyecr_fit_controls_details')!=text:dpg.set_value('pyecr_fit_controls_details',text)
        # The original Copy complete details contract copies this exact text.
        # _refresh_step_panels has just replaced its base on every refresh.
        if full:
            base=dpg.get_value('pyecr_step_fit_details_text')
            dpg.set_value('pyecr_step_fit_details_text',base+'\n\n'+text)
        model_details=dpg.get_value('pyecr_step_model_details_text')
        dpg.set_value('pyecr_step_model_details_text',model_details+'\n\n'+dpg.get_value('pyecr_parameter_count_preview'))
        rail=dpg.get_item_state('pyecr_analysis_left').get('rect_size',(0,0))
        for step,tag,max_height in (('model','pyecr_model_form_scroll',TOKENS.steps.model_form_height),('fit','pyecr_fit_controls_scroll',TOKENS.steps.fit_controls_height)):
            form=dpg.get_item_state(tag)
            end=dpg.get_item_state('pyecr_step_'+step+'_details')
            height=int(rail[1])-TOKENS.steps.control_reserve
            if form.get('rect_size',(0,0))[1]>0 and end.get('rect_size',(0,0))[1]>0:
                after=end['pos'][1]+end['rect_size'][1]-form['pos'][1]-form['rect_size'][1]
                height=int(rail[1]-form['pos'][1]-after-2*TOKENS.steps.padding)
            dpg.configure_item(tag,height=max(TOKENS.steps.preprocess_form_min,min(max_height,height)))
        pending=getattr(self,'_r05_navigation_pending',None)
        if pending and dpg.get_frame_count()>=pending[1]:
            pos=dpg.get_item_state('pyecr_step_panel_'+pending[0]).get('pos')
            if pos:dpg.set_y_scroll('pyecr_analysis_left',max(0,pos[1]))
            self._r05_navigation_pending=None
        self._refresh_r06_intervals()
        self._refresh_r10_noise()
        self._refresh_r11_transport()
        self._refresh_fit_display(reason)
        self._refresh_result_summary()
