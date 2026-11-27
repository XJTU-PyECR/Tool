"""R06 explicit output-CI action and detached local interval evidence."""
import json

from pyecr.contracts import ConfidencePolicy, parse_confidence_percent
from pyecr.gui.theme import TOKENS
from pyecr.gui.state import GuiPage
from pyecr.services.local_intervals import LocalIntervalStore, save_local_intervals
from pyecr.gui.advanced_intervals import AdvancedIntervalLayoutMixin


class LocalIntervalLayoutMixin(AdvancedIntervalLayoutMixin):
    def _build_r06_intervals(self):
        dpg=self._require_dpg();self._local_interval_store=LocalIntervalStore();self._local_interval_error=''
        with dpg.group(parent='pyecr_fit_controls_review',before='pyecr_fit_controls_scroll',horizontal=True,tag='pyecr_local_interval_actions'):
            dpg.add_button(label='Recompute intervals',tag='pyecr_local_interval_recompute',enabled=False,callback=self._recompute_output_intervals)
            dpg.add_button(label='Copy intervals',tag='pyecr_local_interval_copy',enabled=False,callback=self._copy_output_intervals)
        dpg.add_text('',parent='pyecr_fit_controls_review',before='pyecr_fit_controls_scroll',tag='pyecr_local_interval_status',wrap=TOKENS.wraps.stage)
        dpg.add_input_text(parent='pyecr_step_fit_details',tag='pyecr_local_interval_details',multiline=True,readonly=True,width=-1,height=TOKENS.steps.details_height)
        dpg.add_button(parent='pyecr_step_fit_details',label='Save intervals JSON',tag='pyecr_local_interval_save',enabled=False,callback=lambda:self._show_interval_save())
        with dpg.file_dialog(tag='pyecr_local_interval_save_dialog',show=False,default_filename='selected_local_intervals.json',
                             callback=self._save_output_intervals,width=TOKENS.report.dialog_width,height=TOKENS.report.dialog_height):
            dpg.add_file_extension('.json')
        self._prepare_owned_form('pyecr_fit_controls_review')
        self._build_r07_advanced()

    def _output_ci_policy(self):
        return ConfidencePolicy(confidence_level=parse_confidence_percent(self._require_dpg().get_value('pyecr_fit_confidence_percent')))

    def _local_interval_record_for_display(self):
        s=self.controller.analysis_session;store=self._local_interval_store
        key=(id(s),id(s.dataset),id(s.preprocessing.config) if s.preprocessing else None,id(s.current_analysis_config),
            id(s.fit_result),s.fit_generation,s.fit_revision,s.current_revision,s.fit_stale,id(store.latest))
        if key!=getattr(self,'_local_interval_display_key',None):
            self._local_interval_display_record=store.current(s)
            self._local_interval_display_key=key
        return self._local_interval_display_record

    def _recompute_output_intervals(self,*args):
        store=self._local_interval_store;s=self.controller.analysis_session;job=None
        try:
            job=store.begin(s,self._output_ci_policy())
            # Small stored-J covariance is synchronous; freeze/compute/publish
            # remain separate and publication always rechecks parent + request.
            record=store.compute(job)
            if store.publish(job,record,s):s.selected_local_intervals=record
            self._local_interval_error=''
        except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc:
            self._local_interval_error='Reason: '+str(exc)
            if job is not None:store.fail(job,str(exc))
        self._refresh_widgets()

    def _copy_output_intervals(self,*args):
        record=self._local_interval_store.current(self.controller.analysis_session)
        if record is not None:self._require_dpg().set_clipboard_text(record.to_json())

    def _show_interval_save(self):
        if self._local_interval_store.current(self.controller.analysis_session) is not None:
            self._require_dpg().show_item('pyecr_local_interval_save_dialog')

    def _save_output_intervals(self,sender=None,app_data=None,user_data=None):
        try:
            record=self._local_interval_store.current(self.controller.analysis_session)
            if record is None:raise ValueError('CI source is Stale; select the bound parent before saving')
            save_local_intervals(record,app_data['file_path_name']);self._local_interval_error=''
        except (ValueError,TypeError,OSError,KeyError) as exc:self._local_interval_error='Reason: '+str(exc)
        self._refresh_widgets()

    def _refresh_r06_intervals(self):
        dpg=self._require_dpg();s=self.controller.analysis_session;store=self._local_interval_store
        # Project restore binds the original generation, never relevels endpoints.
        restored=s.selected_local_intervals
        if restored is not None and store.latest is not restored:
            try:
                from pyecr.services.local_intervals import _current_source
                restored.binding.assert_ci_source(_current_source(s))
                if restored.fit_generation==s.fit_generation:store.latest=restored
            except ValueError:pass
        record=self._local_interval_record_for_display();policy=None;reason=''
        try:policy=self._output_ci_policy()
        except (ValueError,TypeError) as exc:reason=str(exc)
        valid_parent=s.fit_result is not None and not s.fit_stale and s.fit_revision==s.current_revision and s.current_analysis_config is not None
        running=self.controller.state.tasks[GuiPage.FIT].status=='running'
        dpg.configure_item('pyecr_local_interval_recompute',enabled=policy is not None and valid_parent and not running)
        for tag in ('pyecr_local_interval_copy','pyecr_local_interval_save'):dpg.configure_item(tag,enabled=record is not None)
        status='Output CI: not computed. Recompute uses the stored parent Jacobian; no refit.'
        if not valid_parent:status='Output CI: N/A — current complete parent Fit required.'
        if record is not None:
            c=record.evidence['confidence_level'];state=record.evidence['status']
            status=f'Current output local CI {100*c:.12g}% · {state}. Gate stays 95%.'
            if record.evidence['reason']:
                why=record.evidence['reason'];status+=' Reason: '+(why if len(why)<=120 else why[:117]+'... — Details')
        elif store.latest is not None:status='Output CI: Stale / History — source changed. Recompute explicitly for the new parent.'
        if reason:status+=' Draft reason: '+reason
        if self._local_interval_error:
            why=self._local_interval_error;status+=' Last interval action: '+(why if len(why)<=120 else why[:117]+'... — Details')
        dpg.set_value('pyecr_local_interval_status',status)
        if record is not None:
            heading=f'Parameters · Output local {100*record.evidence["confidence_level"]:.12g}% CI · Gate 95% reliability'
        else:heading='Parameters · Gate 95% CI · Reliability'
        if dpg.does_item_exist('pyecr_fit_result_heading'):dpg.set_value('pyecr_fit_result_heading',heading)
        for name in (() if s.fit_result is None else s.fit_result.params):
            tag='pyecr_parameter_output_ci_'+name
            if not dpg.does_item_exist(tag):continue
            text='Selected output CI: not computed for this parent.'
            if record is not None:
                e=record.evidence;c=100*e['confidence_level'];row=e['intervals'].get(name)
                text=(f'Selected local {c:.12g}% CI: [{row["lower"]:.6g}, {row["upper"]:.6g}] {e["units"][name]}' if row else
                      f'Selected local {c:.12g}% CI: N/A — '+e['unavailable'].get(name,e['reason']))
            dpg.set_value(tag,text)
        # UI reads only immutable owners until full evidence is requested.
        key=(id(record),len(store.history),len(store.failures),store.active,self._local_interval_error)
        if self._fit_details_requested():
            if key!=getattr(self,'_local_interval_details_key',None):
                details=dict(version='selected-local-ci-ui/1',current=None if record is None else record.to_dict(),
                    history=[dict(job_id=h['job_id'],current_at_publication=h['current_at_publication'],record=h['record'].to_dict()) for h in store.history],
                    failed_requests=store.failures,
                    error=self._local_interval_error,execution='Explicit synchronous stored-J covariance only; no optimizer/forward call.',
                    persistence='Standalone JSON, project result evidence and frozen report snapshot; actual C is preserved.')
                self._local_interval_details_text=json.dumps(details,ensure_ascii=False,allow_nan=False,indent=2)
                self._local_interval_details_key=key
            text=self._local_interval_details_text
            dpg.set_value('pyecr_local_interval_details',text)
            dpg.set_value('pyecr_step_fit_details_text',dpg.get_value('pyecr_step_fit_details_text')+'\n\n'+text)
        self._refresh_r07_advanced()
