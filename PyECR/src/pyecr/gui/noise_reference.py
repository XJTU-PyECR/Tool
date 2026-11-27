"""Manage noise reference settings and display diagnostics after a fit stops."""
import json

from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.services.noise_reference import (manual_reference,platform_reference,convert_normalized_reference,
    compare_current_fit,compare_stopped_candidate,save_noise_evidence)

ASSUMPTIONS={'Unknown':'unknown','IID, stationary, constant SD':'iid_stationary_constant_sd',
             'Correlated':'correlated','Varying SD':'heteroscedastic'}


class NoiseReferenceLayoutMixin:
    def _build_r10_noise(self):
        d=self._require_dpg();self._noise_reference=None;self._noise_parent=None;self._noise_error=''
        self._noise_reference_history=[];self._noise_comparison_history=[];self._noise_cache_key=None
        self._noise_current_evidence=None;self._noise_unit_edited=False
        for step,parent,before in (('fit','pyecr_fit_controls_review','pyecr_fit_actual_controls'),
                                   ('advanced','pyecr_stage_advanced_content','pyecr_step_advanced_details')):
            with d.group(parent=parent,before=before,tag='pyecr_noise_status_'+step):
                for name in ('numerical','comparison','reliability','metrics'):
                    d.add_text('',tag='pyecr_noise_'+step+'_'+name,wrap=TOKENS.wraps.compact_status)
        with d.collapsing_header(label='Noise reference (optional)',parent='pyecr_fit_controls_scroll',default_open=False,tag='pyecr_noise_controls'):
            d.add_radio_button(('Not set','Manual SD','Accepted platform'),tag='pyecr_noise_mode',default_value='Not set',callback=self._noise_draft_changed)
            with d.group(tag='pyecr_noise_manual',show=False):
                d.add_input_text(label='Noise SD',tag='pyecr_noise_sd',default_value='',width=-1,callback=self._noise_draft_changed)
                d.add_input_text(label='SD unit (current response)',tag='pyecr_noise_unit',default_value='S/cm',width=-1,callback=self._noise_draft_changed)
            with d.group(tag='pyecr_noise_platform',show=False):
                d.add_radio_button(('Initial','Final'),tag='pyecr_noise_window',default_value='Final',horizontal=True,callback=self._noise_draft_changed)
                d.add_button(label='Review Platforms',callback=self._noise_review_platforms)
                d.add_text('Only accepted windows are eligible. Preview candidates are ignored.',wrap=TOKENS.wraps.stage,color=TOKENS.steps.secondary)
            with d.group(tag='pyecr_noise_assumptions_form',show=False):
                d.add_radio_button(tuple(ASSUMPTIONS),tag='pyecr_noise_assumptions',default_value='Unknown',callback=self._noise_draft_changed)
                d.add_input_text(label='Source note / assumptions',tag='pyecr_noise_basis',width=-1,callback=self._noise_draft_changed)
            d.add_button(label='Apply noise reference',tag='pyecr_noise_apply',callback=self._apply_noise_reference)
            d.add_button(label='Convert SD using accepted normalization',tag='pyecr_noise_convert',callback=self._convert_noise_reference,enabled=False)
            d.add_text('',tag='pyecr_noise_action_reason',wrap=TOKENS.wraps.stage,color=TOKENS.colors.warning)
            d.add_text('Post-stop evidence only. A noise ratio never stops Fit or changes reliability.',wrap=TOKENS.wraps.stage,color=TOKENS.steps.secondary)
        for step in ('fit','advanced'):
            with d.group(parent='pyecr_step_'+step+'_details'):
                d.add_button(label='Copy noise evidence',callback=self._copy_noise_evidence)
                d.add_button(label='Save noise evidence JSON',tag='pyecr_noise_save_'+step,callback=lambda:d.show_item('pyecr_noise_save_dialog'),enabled=False)
                d.add_input_text(tag='pyecr_noise_details_'+step,multiline=True,readonly=True,width=-1,height=TOKENS.steps.details_height)
        with d.file_dialog(tag='pyecr_noise_save_dialog',show=False,default_filename='noise_evidence.json',callback=self._save_noise_evidence,
                           width=TOKENS.report.dialog_width,height=TOKENS.report.dialog_height):d.add_file_extension('.json')
        # Put the optional noise form first within the existing control scroller.
        # It remains collapsed by default and does not displace pinned Run/Cancel.
        children=d.get_item_children('pyecr_fit_controls_scroll',1)
        d.move_item('pyecr_noise_controls',parent='pyecr_fit_controls_scroll',before=children[0])
        self._prepare_owned_form('pyecr_noise_controls');self._noise_ready=True

    def _noise_draft_changed(self,sender=None,*args):
        if sender=='pyecr_noise_unit':self._noise_unit_edited=True
        self._refresh_widgets()

    def _noise_review_platforms(self,*args):
        self._navigate_workflow(user_data='preprocess');self._choose_preprocess_category(user_data='platforms');self._refresh_widgets()

    def _apply_noise_reference(self,*args):
        d=self._require_dpg();s=self.controller.analysis_session
        try:
            mode=d.get_value('pyecr_noise_mode');ref=parent=None
            if mode!='Not set':
                if s.dataset is None or s.preprocessing is None:raise ValueError('Import a source before declaring noise')
                assumptions=ASSUMPTIONS[d.get_value('pyecr_noise_assumptions')]
                basis=d.get_value('pyecr_noise_basis').strip() or ('User-declared external SD' if mode=='Manual SD' else 'Explicitly selected accepted platform statistics')
                if mode=='Manual SD':
                    try:sd=float(d.get_value('pyecr_noise_sd'))
                    except (ValueError,TypeError) as exc:raise ValueError('Noise SD requires a positive finite number') from exc
                    ref=manual_reference(s.dataset,s.preprocessing.config,sd=sd,unit=d.get_value('pyecr_noise_unit'),assumptions=assumptions,basis=basis)
                    parent=(s.dataset,s.preprocessing.config)
                else:
                    ref,ds,recipe=platform_reference(s.preprocessing,d.get_value('pyecr_noise_window').lower(),assumptions=assumptions,basis=basis)
                    parent=(ds,recipe)
            self._noise_reference=ref;self._noise_parent=parent;self._noise_error=''
            s.noise_reference=ref
            self._noise_reference_history.append(dict(action='explicit_apply',reference=None if ref is None else ref.to_dict()))
        except (ValueError,TypeError,KeyError) as exc:self._noise_error='Reason: '+str(exc)
        self._refresh_widgets()

    def _convert_noise_reference(self,*args):
        s=self.controller.analysis_session
        try:
            if self._noise_reference is None or self._noise_parent is None:raise ValueError('Apply a parent noise reference first')
            ref=convert_normalized_reference(self._noise_reference,*self._noise_parent,s.dataset,s.preprocessing.config)
            self._noise_reference=ref;self._noise_parent=(s.dataset,s.preprocessing.config);self._noise_error=''
            s.noise_reference=ref
            self._noise_reference_history.append(dict(action='explicit_scale_conversion',reference=ref.to_dict()))
        except (ValueError,TypeError) as exc:self._noise_error='Reason: '+str(exc)
        self._refresh_widgets()

    def _copy_noise_evidence(self,*args):
        self._require_dpg().set_clipboard_text(self._noise_details_text)

    def _save_noise_evidence(self,sender=None,app_data=None,user_data=None):
        try:
            payload=self._noise_current_evidence or self._noise_reference
            if payload is None:raise ValueError('No noise evidence to save')
            save_noise_evidence(payload,app_data['file_path_name']);self._noise_error=''
        except (ValueError,TypeError,OSError) as exc:self._noise_error='Reason: '+str(exc)
        self._refresh_widgets()

    def _noise_latest_job(self):
        c=self.controller;task=c.state.tasks[GuiPage.FIT]
        return c.monitored_runs.get(c.monitored_task_jobs.get(task.task_id))

    def _noise_compute_current(self):
        s=self.controller.analysis_session;job=self._noise_latest_job()
        if job is not None and job['state']=='failed':
            if not self._noise_job_current():raise ValueError('Stopped attempt is History / Stale')
            return compare_stopped_candidate(s.dataset,s.preprocessing.config,job,self._noise_reference)
        return compare_current_fit(s,self._noise_reference)

    def _noise_job_current(self):
        from pyecr.fitting.progress import FitProgressRevision
        task=self.controller.state.tasks[GuiPage.FIT];s=self.controller.analysis_session
        return task.fit_progress is not None and task.fit_progress.revision==FitProgressRevision.capture(s.current_revision)

    def _noise_numerical_status(self):
        c=self.controller;s=c.analysis_session;task=c.state.tasks[GuiPage.FIT];job=self._noise_latest_job()
        if task.status=='running':return 'Running; noise is not a stop condition.'
        if task.status=='cancelled':return 'Cancelled; prior Fit retained.'
        if job is not None and job['state']=='failed':
            if not self._noise_job_current():return 'History / Stale stopped attempt.'
            if job['contexts']:
                end=job['contexts'][-1]['accepted_trace']['records'][-1]
                return end['stop_reason']+'; native status '+str(end['native_status'])+'; no qualified Fit published.'
            return 'Failed / budget before a complete native endpoint.'
        if task.status=='error':return 'Failed: '+task.message[:120]
        if s.fit_result is None:return 'Not run.'
        if s.fit_stale:return 'History / Stale.'
        status=s.fit_result.exit_status
        return {1:'Native gtol.',2:'Native ftol.',3:'Native xtol.',4:'Native ftol and xtol.',
                0:'Native evaluation budget exhausted.',-2:'Callback stopped; native success=False.'}.get(status,'Native status '+str(status))

    def _noise_reliability_status(self):
        s=self.controller.analysis_session;job=self._noise_latest_job()
        if job is not None and job['state']=='failed':return 'Not assessed for stopped attempt; noise cannot waive gates.'
        if s.fit_result is None:return 'Not assessed.'
        if s.fit_stale:return 'History / Stale.'
        if s.diagnostics is not None and not s.diagnostics_stale and s.diagnostics.evaluation is not None:
            from pyecr.gui.components.analysis_results import diagnostics_summary
            summary=diagnostics_summary(s.diagnostics)
            return summary.title+'; '+summary.note
        if s.fit_presentation is not None and not s.fit_presentation.reliable:return 'Fit checks warn; basic 95% checks.'
        if s.diagnostics is None or s.diagnostics_stale:return 'Fit checks only; Diagnostics not current.'
        return 'Existing gates passed.' if getattr(s.diagnostics,'reliable',False) else 'Diagnostics warns; basic 95% checks.'

    def _refresh_r10_noise(self):
        if not getattr(self,'_noise_ready',False):return
        d=self._require_dpg();s=self.controller.analysis_session;task=self.controller.state.tasks[GuiPage.FIT]
        if d.get_value('pyecr_noise_controls'):
            # The existing narrow-rail minimum (80 px) clips a complete source
            # radio group. An explicitly opened noise form needs one full group;
            # the outer workflow scroller and pinned global actions remain.
            height=d.get_item_configuration('pyecr_fit_controls_scroll')['height']
            d.configure_item('pyecr_fit_controls_scroll',height=max(128,height))
        mode=d.get_value('pyecr_noise_mode')
        for group,show in (('manual',mode=='Manual SD'),('platform',mode=='Accepted platform'),('assumptions_form',mode!='Not set')):
            d.configure_item('pyecr_noise_'+group,show=show)
        if s.dataset is not None and not self._noise_unit_edited:d.set_value('pyecr_noise_unit',str(s.dataset.metadata.get('signal_unit','S/cm')))
        if getattr(self,'_noise_restored_session',None) is not s:
            self._noise_restored_session=s;self._noise_reference=s.noise_reference
            self._noise_parent=None if s.noise_reference is None else (s.dataset,s.preprocessing.config)
        parent_recipe=None if self._noise_parent is None else self._noise_parent[1]
        can_convert=self._noise_reference is not None and self._noise_reference.parent_reference_id is None and parent_recipe is not None and s.preprocessing is not None and (
            len(s.preprocessing.config.steps)==len(parent_recipe.steps)+1 and s.preprocessing.config.steps[-1].operation in ('normalize','normalize_signed_dct'))
        d.configure_item('pyecr_noise_convert',enabled=can_convert)
        job=self._noise_latest_job()
        key=(id(s),id(s.dataset),id(s.current_analysis_config),id(s.fit_result),s.current_revision,s.fit_stale,
             id(self._noise_reference),task.task_id,task.status,id(job))
        if key!=self._noise_cache_key:
            old=self._noise_current_evidence
            try:self._noise_current_evidence=self._noise_compute_current();self._noise_metric_reason=''
            except (ValueError,TypeError,KeyError) as exc:self._noise_current_evidence=None;self._noise_metric_reason=str(exc)
            if old is not None and old!=self._noise_current_evidence:self._noise_comparison_history.append(old)
            self._noise_cache_key=key
        evidence=self._noise_current_evidence;ref=self._noise_reference
        comparison=('Not set.' if ref is None else 'N/A: '+self._noise_metric_reason) if evidence is None else (
            f'Conditional ({evidence.RSE_over_sigma:.4g}).' if evidence.status=='conditional' else
            'Not set.' if evidence.status=='not_set' else 'Stale reference.' if evidence.status=='stale' else
            'N/A: '+evidence.reasons[0].split(':')[0].replace('_',' '))
        fmt=lambda x:'N/A' if x is None else f'{x:.6g}'
        metrics=('Raw RMSE / RSE: N/A.' if evidence is None else
            f'Raw RMSE={fmt(evidence.raw_RMSE)}; RSE={fmt(evidence.RSE)}\nNoise SD={fmt(evidence.sigma_noise)} {evidence.unit}; RSE/SD={fmt(evidence.RSE_over_sigma)}')
        for step in ('fit','advanced'):
            for name,value in (('numerical','Numerical stop: '+self._noise_numerical_status()),('comparison','Noise comparison: '+comparison),
                               ('reliability','Reliability: '+self._noise_reliability_status()),('metrics',metrics)):
                d.set_value('pyecr_noise_'+step+'_'+name,value)
                width=d.get_item_rect_size('pyecr_analysis_left')[0]
                d.configure_item('pyecr_noise_'+step+'_'+name,wrap=max(180,int(width)-48))
            d.configure_item('pyecr_noise_save_'+step,enabled=evidence is not None or ref is not None)
        d.set_value('pyecr_noise_action_reason',self._noise_error)
        draft={k:d.get_value('pyecr_noise_'+k) for k in ('mode','sd','unit','window','assumptions','basis')}
        details_key=(key,self._noise_error,tuple(draft.items()),len(self._noise_reference_history),len(self._noise_comparison_history),s.diagnostics_stale,id(s.diagnostics))
        if details_key!=getattr(self,'_noise_details_key',None):
            self._noise_details_text=json.dumps(dict(version='noise-reference-ui/1',draft=draft,
                applied_reference=None if ref is None else ref.to_dict(),comparison=None if evidence is None else evidence.to_dict(),
                reference_history=self._noise_reference_history,comparison_history=[x.to_dict() for x in self._noise_comparison_history],
                reason=self._noise_error or self._noise_metric_reason,normalization_conversion='Explicit accepted-scale SD conversion only; Analysis remains unchanged.',
                persistence='Standalone noise evidence includes its source, units and assumptions.'),ensure_ascii=False,indent=2,allow_nan=False)
            self._noise_details_key=details_key
        for step in ('fit','advanced'):
            tag='pyecr_noise_details_'+step
            if d.get_value(tag)!=self._noise_details_text:d.set_value(tag,self._noise_details_text)
            if step!='fit' or self._fit_details_requested():
                target='pyecr_step_'+step+'_details_text';d.set_value(target,d.get_value(target)+'\n\n'+self._noise_details_text)
