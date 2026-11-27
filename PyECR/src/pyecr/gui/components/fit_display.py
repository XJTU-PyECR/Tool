"""U04 Fit disclosure only. No request, solver, gate or interval computation."""
import json
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.gui.workflow_display import brief_text


def _brief_check_notice(message):
    return {'RESIDUAL_DEPENDENCE_UNASSESSED':'Residual independence has not been assessed.',
            'RESIDUAL_DEPENDENCE':'Residual dependence needs review.',
            'LOCAL_CI_ASSUMPTIONS_UNVERIFIED':'CI error assumptions need review.'}.get(message.partition(':')[0],message)


class FitDisplayMixin:
    def _fit_details_requested(self,step='fit'):
        return (getattr(self,'_fit_details_force',None)==step
                or bool(self._require_dpg().get_value('pyecr_step_'+step+'_details')))

    def _build_fit_display(self):
        d=self._require_dpg();review='pyecr_fit_controls_review';scroll='pyecr_fit_controls_scroll'
        # Actions are already proxies to original tags, outside this scroller.
        for tag in ('pyecr_step_fit_status','pyecr_step_fit_warning'):
            d.move_item(tag,parent='pyecr_stage_fit_content',before=review)
        with d.collapsing_header(label='Settings',tag='pyecr_fit_settings',parent=review,before=scroll,default_open=False):pass
        d.move_item(scroll,parent='pyecr_fit_settings')
        first=d.get_item_children(scroll,1)[0]
        with d.collapsing_header(label='Output intervals',tag='pyecr_fit_output_settings',parent=scroll,before=first,default_open=False):pass
        with d.collapsing_header(label='Stopping settings',tag='pyecr_fit_stop_settings',parent=scroll,before=first,default_open=False):pass
        item=d.get_alias_id('pyecr_fit_confidence_percent')
        wrapper=self._owned_form_fields.get(item,(item,))[0]
        d.move_item(wrapper,parent='pyecr_fit_output_settings')
        d.move_item('pyecr_local_interval_actions',parent='pyecr_fit_output_settings')
        d.add_text('',tag='pyecr_fit_interval_summary',parent='pyecr_fit_output_settings',wrap=TOKENS.wraps.stage,show=False)
        for tag in ('pyecr_fit_stopping_mode','pyecr_fit_control_draft_status','pyecr_fit_monitored_form'):
            d.move_item(tag,parent='pyecr_fit_stop_settings')
        d.move_item('pyecr_fit_control_reason',parent='pyecr_fit_settings',before=scroll)
        for tag in ('pyecr_fit_actual_controls','pyecr_noise_status_fit','pyecr_local_interval_status'):
            d.move_item(tag,parent='pyecr_step_fit_details')
        d.configure_item('pyecr_noise_controls',label='Noise reference')
        # Retain original text as evidence; remove it from the editable form.
        prefixes=('Uncertainty — draft','Stopping — draft','Blank = 95%.',
            'TRF only. Accepted-step','Optimizer tolerances do not edit',
            'Legacy uses the applied Model','Only accepted windows are eligible.',
            'Post-stop evidence only.')
        def move_notes(root):
            for children in list(d.get_item_children(root).values()):
                for child in list(children):
                    cfg=d.get_item_configuration(child)
                    if d.get_item_info(child)['type'].endswith('mvText') and str(d.get_value(child)).startswith(prefixes):
                        # The old Monitored paragraph predates the T14 routes.
                        if str(d.get_value(child)).startswith('TRF only.'):
                            d.set_value(child,'Supported TRF policies only. Native termination and original reliability are separate.')
                        d.move_item(child,parent='pyecr_step_fit_details')
                    else:move_notes(child)
        move_notes(scroll)
        for child in d.get_item_children('pyecr_fit_monitored_form',1):
            if d.get_item_info(child)['type'].endswith('mvCollapsingHeader'):d.configure_item(child,label='Advanced')
        for tag,text in (
            ('pyecr_fit_confidence_percent','Editing C changes the output draft only. Recompute uses the stored parent Jacobian without refitting. The original gate remains 95%.'),
            ('pyecr_fit_stopping_mode','Legacy uses the applied Model configuration. Monitored uses only its supported policies. Editing does not submit work.'),
            ('pyecr_fit_controls_apply','Store these controls for review. Run Fit separately freezes the current valid controls; Apply performs no calculation.'),
            ('pyecr_noise_mode','A declared noise reference is post-stop evidence. It never stops Fit or changes reliability.')):
            with d.tooltip(tag):d.add_text(text,wrap=TOKENS.wraps.stage)
        for tag in ('pyecr_fit_controls_details','pyecr_local_interval_details','pyecr_noise_details_fit'):
            d.configure_item(tag,show=False) # compatible data tags; complete content is in one combined reader
        self._fit_display_ready=True

    def _fit_display_warnings(self,reason):
        c=self.controller;s=c.analysis_session;t=c.state.tasks[GuiPage.FIT]
        quality=getattr(self,'_fit_quality_cache_report',None);items=[]
        error=c.state.current_errors.get(GuiPage.FIT)
        if error is not None:items.append((str(error.message),'Review Details.'))
        if self._fit_unpublished_stop():items.append((t.message,'Review Stopping settings or Details.'))
        if reason:items.append((reason,'Review Settings.' if t.status!='running' else 'Draft only; running request is frozen.'))
        if getattr(self,'_local_interval_error',''):items.append((self._local_interval_error,'Review Output intervals in Settings.'))
        if quality is not None and not quality.ready and s.dataset is not None:
            items.extend((r,'Review Model or Preprocess.') for r in quality.blocking_reasons)
        if s.fit_stale and s.fit_result is not None:items.append(('Fit source changed.','Run Fit to update; the previous result is retained.'))
        if s.diagnostics is not None and s.diagnostics_stale:items.append(('Diagnostics are out of date.','Run Diagnostics for the current Fit.'))
        if s.diagnostics is not None and not s.diagnostics_stale:
            if s.diagnostics.evaluation is not None:
                items.extend((message, 'Review Diagnostics.') for message in s.diagnostics.evaluation.warnings)
            for failure in s.diagnostics.gate_failures:
                if str(failure).startswith('RESIDUAL_AUTOCORRELATION_EXCEEDS_LIMIT'):
                    items.append(('Residual correlation warning.','Review Diagnostics.'))
                else:items.append((str(failure),'Review Diagnostics.'))
        if quality is not None:items.extend((r,'Review Preprocess.') for r in quality.warnings)
        p=c.state.fit_presentation
        if p is not None:
            items.extend((r,'Review Diagnostics.') for r in p.warnings)
            if not p.reliable and not items:items.append(('Fit reliability warning.','Review Diagnostics.'))
        if getattr(self,'_noise_error',''):items.append((self._noise_error,'Review Noise reference in Settings.'))
        unique={}
        for message,hint in items:unique.setdefault(' '.join(str(message).split()),hint)
        return tuple(unique.items())

    def _fit_unpublished_stop(self):
        c=self.controller;t=c.state.tasks[GuiPage.FIT]
        job=getattr(c,'monitored_task_jobs',{}).get(t.task_id)
        record=c.monitored_runs.get(job,{})
        outcome=record.get('execution_outcome') or {}
        return t.status=='warning' and outcome.get('qualified') is False

    def _fit_cached_component_eligibility(self):
        # Match the existing U01 cache without calling Capture/select_model or
        # recomputing a local interval merely to read Fit Details.
        s=self.controller.analysis_session
        key=(id(s.dataset),id(s.preprocessing),id(s.preprocessing.config) if s.preprocessing else None,id(s.current_analysis_config),id(s.fit_result),id(s.diagnostics),
            s.current_revision,s.fit_revision,s.fit_generation,s.fit_stale,s.diagnostics_stale,
            s.diagnostics_fit_revision,s.diagnostics_fit_generation,s.diagnostics_identity)
        cached=getattr(self,'_component_parent_cache',None)
        if cached is not None and cached[0]==key:
            cap=cached[1]
            return cap.eligibility.to_dict() if cap.eligibility is not None else dict(blocking_reasons=cap.state.blocking_reasons)
        return dict(status='NOT_RUN',reason='Component eligibility has not been reviewed for this source. Review components explicitly.')

    def _refresh_fit_display(self,reason):
        if not getattr(self,'_fit_display_ready',False):return
        d=self._require_dpg();c=self.controller;s=c.analysis_session;t=c.state.tasks[GuiPage.FIT]
        if t.status=='running':status='Cancelling Fit…' if t.cancellation_requested else 'Fitting…'
        elif t.status=='error' or GuiPage.FIT in c.state.current_errors:status='Fit failed.'
        elif t.status=='cancelled':status=('Fit cancelled; best candidate in Results.' if c.current_fit_candidate is not None else
                                        'Fit cancelled; previous result retained.' if s.fit_result is not None else 'Fit cancelled.')
        elif self._fit_unpublished_stop():status='Fit stopped; previous result retained.' if s.fit_result is not None else 'Fit stopped; no result published.'
        elif s.fit_result is not None:status='Fit out of date.' if s.fit_stale else 'Fit completed.'
        elif s.dataset is None:status='Import data and apply a model.'
        elif s.current_analysis_config is None:status='Apply a model before fitting.'
        else:status='Ready for Fit.'
        warnings=self._fit_display_warnings(reason);self._fit_display_warning_items=warnings
        text='' if not warnings else brief_text(_brief_check_notice(warnings[0][0]),120)+' '+warnings[0][1]+(f' (+{len(warnings)-1}; Details)' if len(warnings)>1 else ' — Details')
        d.set_value('pyecr_step_fit_status',status)
        wrap=max(TOKENS.analysis.field_min,int(d.get_item_configuration('pyecr_analysis_left')['width'])-2*TOKENS.steps.padding-TOKENS.analysis.form_gutter)
        d.set_value('pyecr_step_fit_warning',text);d.configure_item('pyecr_step_fit_warning',show=bool(text),wrap=wrap)
        d.configure_item('pyecr_stage_fit_reason',show=False)
        # The header content can show prerequisites with disabled actions even
        # before import; readiness and original callbacks remain authoritative.
        d.configure_item('pyecr_stage_fit_content',show=True)
        record=self._local_interval_record_for_display()
        summary=''
        if record is not None:
            summary=f'Computed {100*record.evidence["confidence_level"]:.12g}% · {record.evidence["status"]}'
            try:
                C=self._output_ci_policy().confidence_level
                if C!=record.evidence['confidence_level']:summary+=f' | Draft {100*C:.12g}%'
            except (ValueError,TypeError):summary+=' | Invalid draft'
        elif self._local_interval_store.latest is not None:summary='Previous intervals are out of date. Recompute for this Fit.'
        if self._local_interval_error:summary+='\n'+self._local_interval_error
        d.set_value('pyecr_fit_interval_summary',summary);d.configure_item('pyecr_fit_interval_summary',show=bool(summary),wrap=wrap)
        # Full readers remain available, but are composed only when requested.
        if self._fit_details_requested():
            source_key=(id(s.dataset),id(s.current_analysis_config),id(s.fit_result),id(s.diagnostics),s.current_revision,
                s.fit_generation,s.fit_stale,s.diagnostics_stale,id(getattr(self,'_component_parent_cache',None)))
            if source_key!=getattr(self,'_fit_display_source_key',None):
                from pyecr.report.snapshot import _diagnostics_mapping,_json_safe
                from pyecr.project.repository import _encode_special_floats
                eligibility=None
                if s.fit_result is not None and s.current_analysis_config is not None and s.current_analysis_config.model_id.value in {'M2_DUAL_HO_D','M3_DUAL_HO_DK','M8_DUAL_K'}:
                    eligibility=self._fit_cached_component_eligibility()
                self._fit_display_source_text=json.dumps(_encode_special_floats(_json_safe(dict(
                    version='fit-display-evidence/1',fit=None if s.fit_result is None else s.fit_result.to_dict(),
                    source_metadata=None if s.dataset is None else dict(s.dataset.metadata),
                    recipe=None if s.preprocessing is None else s.preprocessing.config.to_dict(),
                    applied_config=None if s.current_analysis_config is None else s.current_analysis_config.to_dict(),
                    diagnostics=None if s.diagnostics is None else _diagnostics_mapping(s.diagnostics),component_eligibility=eligibility,
                    limitations='Original reliability and 95% evidence are unchanged. Component permission does not validate correlated-error CI.'))),
                    ensure_ascii=False,indent=2,allow_nan=False)
                self._fit_display_source_key=source_key
            target='pyecr_step_fit_details_text';base=d.get_value(target)
            summary_tags=('pyecr_fit_actual_controls','pyecr_noise_fit_numerical','pyecr_noise_fit_comparison',
                'pyecr_noise_fit_reliability','pyecr_noise_fit_metrics','pyecr_local_interval_status','pyecr_fit_control_draft_status')
            summaries='\n'.join(str(d.get_value(tag)) for tag in summary_tags)
            d.set_value(target,base+'\n\nActual evidence / current draft:\n'+summaries+
                '\n\nWarnings / blockers:\n'+'\n'.join(a+' '+b for a,b in warnings)+'\n\n'+self._fit_display_source_text)
