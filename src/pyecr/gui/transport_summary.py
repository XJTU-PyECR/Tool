"""R11 read-only result region. Numbers always come from the shared service."""
import json
from pyecr.gui.theme import TOKENS,SECTION_FONT_TAG
from pyecr.services.transport_summary import summarize_session,save_transport_summary


class TransportSummaryLayoutMixin:
    def _build_r11_transport(self):
        d=self._require_dpg();self._transport_key=None;self._transport_current=None;self._transport_last=None;self._transport_history=[];self._transport_error=''
        with d.group(tag='pyecr_transport_region',parent='pyecr_fit_results_group',before='pyecr_fit_result_heading'):
            d.add_spacer(height=8);d.add_separator();d.add_spacer(height=4)
            heading=d.add_text('Transport')
            if d.does_item_exist(SECTION_FONT_TAG):d.bind_item_font(heading,SECTION_FONT_TAG)
            d.add_text('Not run.',tag='pyecr_transport_status',wrap=TOKENS.wraps.compact_status)
            d.add_text('',tag='pyecr_transport_reason',wrap=TOKENS.wraps.compact_status,color=TOKENS.colors.warning)
            d.add_group(tag='pyecr_transport_sizes')
            d.add_group(tag='pyecr_transport_axes')
            d.add_text('',tag='pyecr_transport_ci_status',wrap=TOKENS.wraps.compact_status)
            with d.collapsing_header(label='Transport Details',tag='pyecr_transport_details_header',default_open=False):
                d.add_group(tag='pyecr_transport_full_axes')
                d.add_button(label='Copy transport details',tag='pyecr_transport_copy',callback=self._copy_transport_details)
                d.add_button(label='Save transport JSON',tag='pyecr_transport_save',callback=lambda:d.show_item('pyecr_transport_save_dialog'),enabled=False)
                d.add_input_text(tag='pyecr_transport_details',readonly=True,multiline=True,width=-1,height=TOKENS.heights.transport_details)
        d.move_item('pyecr_transport_ci_status',parent='pyecr_transport_details_header',before='pyecr_transport_full_axes')
        with d.file_dialog(tag='pyecr_transport_save_dialog',show=False,default_filename='transport_summary.json',callback=self._save_transport,
                width=TOKENS.report.dialog_width,height=TOKENS.report.dialog_height):d.add_file_extension('.json')
        self._transport_ready=True

    def _copy_transport_details(self,*args):
        self._require_dpg().set_clipboard_text(self._transport_details_text)

    def _save_transport(self,sender=None,app_data=None,user_data=None):
        try:
            if self._transport_current is None:raise ValueError('Current source-bound summary required; historical details remain copyable')
            save_transport_summary(self._transport_current,app_data['file_path_name']);self._transport_error=''
        except (ValueError,TypeError,OSError) as exc:self._transport_error='Reason: '+str(exc)
        self._refresh_widgets()

    def _refresh_r11_transport(self):
        if not getattr(self,'_transport_ready',False):return
        d=self._require_dpg();s=self.controller.analysis_session
        record=self._local_interval_record_for_display()
        key=(id(s),id(s.dataset),id(s.current_analysis_config),id(s.fit_result),s.current_revision,s.fit_stale,s.diagnostics_stale,id(s.diagnostics),id(record))
        if key!=self._transport_key:
            previous=self._transport_current;self._transport_current=None;reason=''
            try:self._transport_current=summarize_session(s,local_record=record)
            except (ValueError,TypeError) as exc:reason=str(exc)
            if previous is not None and previous!=self._transport_current:self._transport_history.append(previous)
            if self._transport_current is not None:self._transport_last=self._transport_current
            self._transport_key=key;self._transport_source_reason=reason
            summary=self._transport_current or self._transport_last
            current=self._transport_current is not None
            for tag in ('pyecr_transport_sizes','pyecr_transport_axes','pyecr_transport_full_axes'):d.delete_item(tag,children_only=True)
            status='N/A: no current parent Fit.'
            ci='Derived CI: N/A.'
            if summary is not None:
                status=('Current' if current else 'History / Stale')+' · '+('Conditional on fixed D' if summary.source_kind=='conditional_fixed_d' else 'From fitted D and k')
                if summary.parent_reliability!='supported':status+='\n'+('Diagnostics pending' if summary.parent_reliability=='pending' else 'Parent checks need review')
                color=TOKENS.steps.text
                for p in summary.processes:
                    title=d.add_text(p.label.replace('Parent role ','Role '),parent='pyecr_transport_sizes',color=color)
                    if d.does_item_exist(SECTION_FONT_TAG):d.bind_item_font(title,SECTION_FONT_TAG)
                    value_text='N/A' if p.half_length_cm is None else f'{p.half_length_cm*10000:.5g} µm'
                    value=d.add_text('D/k  '+value_text,parent='pyecr_transport_sizes',color=TOKENS.colors.csv)
                    if d.does_item_exist(SECTION_FONT_TAG):d.bind_item_font(value,SECTION_FONT_TAG)
                    axes=[a for a in summary.axes if a.process==p.process]
                    if axes:
                        values=' · '.join(f'{a.axis} {a.value:.3g}' if a.value is not None else f'{a.axis} N/A' for a in axes)
                        d.add_text('Bi: '+values,parent='pyecr_transport_sizes',wrap=TOKENS.wraps.compact_status)
                        regimes=tuple(dict.fromkeys(a.regime.replace('_',' ') for a in axes))
                        d.add_text(' / '.join(regimes),parent='pyecr_transport_sizes',wrap=TOKENS.wraps.compact_status,color=TOKENS.steps.secondary)
                    if p.reason:d.add_text('Reason: '+p.reason,parent='pyecr_transport_sizes',wrap=TOKENS.wraps.compact_status,color=TOKENS.colors.warning)
                with d.table(parent='pyecr_transport_full_axes',tag='pyecr_transport_table',header_row=True,borders_innerH=True,
                    borders_innerV=True,borders_outerH=True,borders_outerV=True,scrollX=True,scrollY=True,
                    width=-1,height=min(190,56+28*len(summary.axes)),inner_width=620,policy=d.mvTable_SizingFixedFit):
                    for label,width in (('Process',85),('Axis',40),('Full size (cm)',110),('Bi',85),('Regime',170)):
                        d.add_table_column(label=label,width_fixed=True,init_width_or_weight=width)
                    labels={'single':'Single','H':'Role H','O':'Role O','component_1':'Mode 1','component_2':'Mode 2'}
                    for a in summary.axes:
                        with d.table_row():
                            for value in (labels[a.process],a.axis,f'{a.full_dimension_cm:.6g}','N/A' if a.value is None else f'{a.value:.6g}',a.regime):d.add_text(value,color=color)
                ci=(f'Derived local {100*summary.intervals[0].confidence_level:.12g}% CI: Details. Gate stays 95%.' if summary.intervals else 'Derived CI: N/A. '+summary.interval_reason)
            d.set_value('pyecr_transport_status',status);d.set_value('pyecr_transport_ci_status',ci)
            d.configure_item('pyecr_transport_status',color=TOKENS.steps.secondary if summary is None or summary.parent_reliability=='supported' else TOKENS.colors.warning)
        shown=self._transport_current or self._transport_last
        reasons=[self._transport_error or self._transport_source_reason]
        if shown is not None:
            if shown.parent_reliability!='supported':reasons.extend(shown.reliability_reasons)
        reason=next((r for r in reasons if r),'')
        d.set_value('pyecr_transport_reason',reason[:180]);d.configure_item('pyecr_transport_reason',show=bool(reason) and (self._transport_error!='' or self._transport_current is None))
        d.configure_item('pyecr_transport_region',show=shown is not None)
        d.configure_item('pyecr_transport_save',enabled=self._transport_current is not None)
        details_key=(key,self._transport_error,len(self._transport_history))
        if details_key!=getattr(self,'_transport_details_key',None):
            self._transport_details_text=json.dumps(dict(version='transport-ui/1',state='Current' if self._transport_current else 'History / Stale' if shown else 'Not run',
                point_id=None if shown is None else shown.point_id,summary_id=None if shown is None else shown.content_sha256,
                summary=None if shown is None else shown.to_dict(),reason=self._transport_error or self._transport_source_reason,
                history=[x.to_dict() for x in self._transport_history],interpretation='0.03/30 are heuristics; never a new gate. Full thickness is a symmetric two-face slab reference.',
                persistence='Standalone JSON; DCT and report consumers use this same summary identity via the shared service.'),ensure_ascii=False,allow_nan=False,indent=2)
            self._transport_details_key=details_key;d.set_value('pyecr_transport_details',self._transport_details_text)
        for step in ('fit','advanced'):
            if step=='fit' and not self._fit_details_requested():continue
            tag='pyecr_step_'+step+'_details_text';d.set_value(tag,d.get_value(tag)+'\n\n'+self._transport_details_text)
