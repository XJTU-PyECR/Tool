"""R07 explicit advanced runs, saved-evidence recompute and standalone JSON."""
import json
from pathlib import Path

from pyecr.contracts import parse_confidence_percent
from pyecr.contracts.common import strict_json_loads
from pyecr.gui.theme import TOKENS
from pyecr.gui.state import GuiPage
from pyecr.services.advanced_intervals import advanced_to_dict, advanced_from_dict, assert_current, save_advanced
from pyecr.uncertainty import AdvancedUncertaintyConfig


class AdvancedIntervalLayoutMixin:
    def _build_r07_advanced(self):
        dpg=self._require_dpg();self._r07_error=''
        with dpg.collapsing_header(parent='pyecr_stage_advanced_content',before='pyecr_step_advanced_details',
                label='Advanced uncertainty',tag='pyecr_advanced_output_controls',default_open=False):
            dpg.add_text('',tag='pyecr_advanced_output_draft',wrap=TOKENS.wraps.stage)
            dpg.add_button(label='Confidence settings',callback=lambda:self._navigate_workflow(user_data='fit'))
            with dpg.group(tag='pyecr_r07_run_actions',horizontal=True):pass
            with dpg.table(header_row=False,policy=dpg.mvTable_SizingStretchProp,width=-1):
                dpg.add_table_column(init_width_or_weight=1.7);dpg.add_table_column(init_width_or_weight=1)
                for label,tag,value in (('Bootstrap seed','pyecr_advanced_seed',20260915),('Resamples (200–1000)','pyecr_advanced_budget',200),
                        ('Profile points (odd)','pyecr_advanced_points',41)):
                    with dpg.table_row():
                        dpg.add_text(label);dpg.add_input_int(tag=tag,default_value=value,width=-1,callback=self._advanced_draft_changed)
                with dpg.table_row():
                    dpg.add_text('Profile span (decades)');dpg.add_input_double(tag='pyecr_advanced_span',default_value=1.,width=-1,callback=self._advanced_draft_changed)
            dpg.add_text('Budgets/seed are draft. New runs require explicit action; saved-evidence recompute does not draw samples or refit.',wrap=TOKENS.wraps.stage)
            for method,label in (('bootstrap','Recompute from saved samples'),('profile','Recompute from saved profile'),('experimental_m1','Recompute saved M1 samples')):
                dpg.add_button(label=label,tag='pyecr_advanced_recompute_'+method,user_data=method,callback=self._recompute_saved_output,enabled=False)
            dpg.add_text('',tag='pyecr_advanced_output_status',wrap=TOKENS.wraps.stage)
            dpg.add_radio_button(('bootstrap','profile','experimental_m1'),tag='pyecr_advanced_record_method',default_value='bootstrap',horizontal=False)
            dpg.add_input_text(label='Independent advanced JSON path',tag='pyecr_advanced_record_path',default_value='advanced_output.json',width=-1)
            with dpg.group(horizontal=True):
                dpg.add_button(label='Copy record',callback=self._copy_advanced_output)
                dpg.add_button(label='Save JSON',callback=self._save_advanced_output)
                dpg.add_button(label='Reopen JSON',callback=self._open_advanced_output)
            dpg.add_input_text(tag='pyecr_advanced_output_details',multiline=True,readonly=True,width=-1,height=TOKENS.steps.details_height)
        for method,label in (('bootstrap','Run bootstrap'),('profile','Run profile')):
            tag='pyecr_diagnostics_'+method
            dpg.move_item(tag,parent='pyecr_r07_run_actions');dpg.configure_item(tag,label=label)
        self._prepare_owned_form('pyecr_advanced_output_controls');self._r07_ready=True

    def _r07_settings(self):
        d=self._require_dpg()
        return AdvancedUncertaintyConfig(enabled=True,bootstrap_resamples=d.get_value('pyecr_advanced_budget'),bootstrap_n_jobs=1,
            profile_points=d.get_value('pyecr_advanced_points'),profile_span_decades=d.get_value('pyecr_advanced_span'))

    def _advanced_draft_changed(self,*args):self._refresh_widgets()

    def _recompute_saved_output(self,sender=None,app_data=None,user_data=None):
        try:
            self.controller.recompute_saved_advanced(user_data,parse_confidence_percent(self._require_dpg().get_value('pyecr_fit_confidence_percent')))
            self._r07_error=''
        except (ValueError,TypeError,RuntimeError) as exc:self._r07_error=str(exc)
        self._refresh_widgets()

    def _selected_advanced_output(self):
        method=self._require_dpg().get_value('pyecr_advanced_record_method')
        p=self.controller.analysis_session.advanced_diagnostics.get(method)
        if p is None:raise ValueError('No saved method result; explicitly Run analysis first')
        assert_current(p,self.controller.analysis_session)
        return p

    def _copy_advanced_output(self,*args):
        try:self._require_dpg().set_clipboard_text(json.dumps(advanced_to_dict(self._selected_advanced_output()),ensure_ascii=False,allow_nan=False,indent=2));self._r07_error=''
        except (ValueError,TypeError) as exc:self._r07_error=str(exc)
        self._refresh_widgets()

    def _save_advanced_output(self,*args):
        try:save_advanced(self._selected_advanced_output(),self._require_dpg().get_value('pyecr_advanced_record_path'));self._r07_error=''
        except (ValueError,TypeError,OSError) as exc:self._r07_error=str(exc)
        self._refresh_widgets()

    def _open_advanced_output(self,*args):
        try:
            if self.controller.state.tasks[GuiPage.DIAGNOSTICS].status=='running':
                raise ValueError('Finish or explicitly Cancel running Diagnostics before reopening saved evidence')
            p=advanced_from_dict(strict_json_loads(Path(self._require_dpg().get_value('pyecr_advanced_record_path')).read_text(encoding='utf-8')))
            assert_current(p,self.controller.analysis_session)
            method='experimental_m1' if p.experimental_result is not None else 'bootstrap' if p.method.value=='residual_bootstrap' else 'profile'
            self.controller.analysis_session.advanced_diagnostics[method]=p;self.controller.state.advanced_diagnostics[method]=p
            if not hasattr(self.controller,'advanced_output_history'):self.controller.advanced_output_history=[]
            self.controller.advanced_output_history.append(dict(action='explicit_reopen_JSON',method=method,presentation=p,current_at_publication=True));self._r07_error=''
        except (ValueError,TypeError,OSError) as exc:self._r07_error=str(exc)
        self._refresh_widgets()

    def _refresh_r07_advanced(self):
        if not getattr(self,'_r07_ready',False):return
        d=self._require_dpg();c=self.controller;s=c.analysis_session;reason='';C=None
        # These existing actions now belong to this explicitly opened group,
        # rather than the old Expert-only controls hidden by information level.
        for method in ('bootstrap','profile'):
            d.configure_item('pyecr_diagnostics_'+method,show=True)
        try:C=parse_confidence_percent(d.get_value('pyecr_fit_confidence_percent'))
        except ValueError as exc:reason=str(exc)
        d.set_value('pyecr_advanced_output_draft',f'Output confidence: {100*C:.12g}%. Change in 04 Run fit; basic checks use 95%.' if C is not None else 'Reason: '+reason)
        task=c.state.tasks[GuiPage.DIAGNOSTICS];lines=[];records={}
        for method in ('bootstrap','profile','experimental_m1'):
            p=s.advanced_diagnostics.get(method);current=False
            if p is not None:
                try:assert_current(p,s);current=True
                except ValueError:pass
                result=p.experimental_result if p.experimental_result is not None else p.result
                label=f'{100*result.confidence_level:.12g}%' if result is not None else 'no computed C'
                lines.append(f'{method}: {"Current" if current else "Stale / History"}, actual {label}; '+('available' if p.available else 'N/A — '+str(p.reason)))
                records[method]=p
            d.configure_item('pyecr_advanced_recompute_'+method,enabled=current and C is not None and task.status!='running')
        if s.current_analysis_config is not None:
            supported=s.current_analysis_config.model_id.value in {'M0_SINGLE_D','M1_SINGLE_DK'}
            if not supported:
                d.configure_item('pyecr_diagnostics_profile',enabled=False)
                lines.append('Profile: Unsupported for this model; no validated constrained-refit adapter.')
        if reason:
            for method in ('bootstrap','profile','experimental_m1'):d.configure_item('pyecr_diagnostics_'+method,enabled=False)
        active=getattr(c,'advanced_request_bindings',{}).get(task.task_id)
        if task.status=='running' and active is not None:lines.append(f'Running: frozen output C={100*active.controls.output_ci.confidence_level:.12g}% / {active.controls.output_ci.method}')
        if self._r07_error:lines.append('Reason: '+self._r07_error)
        text='\n'.join(lines) if lines else 'No advanced output. Run explicitly; changing C does not submit a task.'
        d.set_value('pyecr_advanced_output_status',text)
        bindings=getattr(c,'advanced_request_bindings',{});history=getattr(c,'advanced_output_history',[])
        key=(tuple((k,id(v)) for k,v in records.items()),self._r07_error,
             tuple((k,id(v)) for k,v in bindings.items()),len(history))
        if key!=getattr(self,'_advanced_display_details_key',None):
            details=dict(version='advanced-output-ui/1',records={k:advanced_to_dict(v) for k,v in records.items()},error=self._r07_error,
                requests={k:v.to_dict() for k,v in bindings.items()},
                history=[{**{k:v for k,v in h.items() if k!='presentation'},'record':advanced_to_dict(h['presentation'])} for h in history])
            self._advanced_display_details_text=json.dumps(details,ensure_ascii=False,allow_nan=False,indent=2)
            self._advanced_display_details_key=key
            d.set_value('pyecr_advanced_output_details',self._advanced_display_details_text)
        if self._fit_details_requested('advanced'):
            d.set_value('pyecr_step_advanced_details_text',d.get_value('pyecr_step_advanced_details_text')+'\n\n'+self._advanced_display_details_text)
