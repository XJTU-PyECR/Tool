"""Explicit export/open/review/send controls for persisted Simulation evidence."""
from pathlib import Path
from textwrap import fill
import traceback
from pyecr.gui.theme import TOKENS

E='pyecr_sim_io_'
TARGETS={'Require empty Analysis':'require_empty','Replace current Analysis; clear config/results/project association':'replace_current'}


class SimulationExchangeViewMixin:
    def _build_simulation_exchange(self):
        dpg=self._require_dpg();self._sim_send_plan=None;self._sim_exchange_message='';self._sim_exchange_trace=''
        with dpg.window(label='Simulation — Export / Open / Send',tag=E+'window',show=False,width=TOKENS.widths.simulation_exchange,height=TOKENS.heights.simulation_exchange):
            dpg.add_button(label='Close',callback=lambda:dpg.configure_item(E+'window',show=False))
            dpg.add_text('',tag=E+'source',wrap=TOKENS.wraps.simulation_exchange)
            with dpg.collapsing_header(label='Source Details',default_open=False):
                dpg.add_input_text(tag=E+'source_details',readonly=True,multiline=True,width=-1,height=TOKENS.heights.notes)
            with dpg.tab_bar(tag=E+'tabs'):
                with dpg.tab(label='Export captured run',tag=E+'export_tab'):
                    dpg.add_text('Export this simulated result to a new directory.',wrap=TOKENS.wraps.simulation_exchange)
                    dpg.add_input_text(tag=E+'destination',hint='Absolute NEW destination directory',width=-1)
                    dpg.add_checkbox(label='XLSX (15 digits approximate; JSON/CSV are authoritative)',tag=E+'xlsx',default_value=False)
                    dpg.add_checkbox(label='Auto-place legends (export only)',tag=E+'legend_auto',default_value=False)
                    with dpg.group(horizontal=True):
                        for fmt in ('png','svg','pdf'):dpg.add_checkbox(label=fmt.upper(),tag=E+fmt,default_value=True)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='Export captured Simulation',tag=E+'export',callback=self._export_simulation_widgets)
                        dpg.add_button(label='Cancel export before commit',tag=E+'cancel',callback=lambda:self.controller.cancel_simulation_export())
                    dpg.add_text('',tag=E+'status',wrap=TOKENS.wraps.simulation_exchange)
                with dpg.tab(label='Open verified bundle',tag=E+'open_tab'):
                    dpg.add_text('Open a verified Simulation bundle to view its stored result.',wrap=TOKENS.wraps.simulation_exchange)
                    dpg.add_input_text(tag=E+'open_path',hint='Absolute Simulation bundle directory',width=-1)
                    dpg.add_button(label='Open verified Simulation bundle',tag=E+'open',callback=self._open_simulation_widgets)
                with dpg.tab(label='Send simulated data to Analysis',tag=E+'send_tab'):
                    dpg.add_text('Export or open this result first. Choose a channel and target policy, then Review before Send.',wrap=TOKENS.wraps.simulation_exchange)
                    dpg.add_text('Source channel');dpg.add_combo(('clean','observed'),tag=E+'channel',default_value='',width=-1,callback=self._invalidate_simulation_send_review)
                    dpg.add_text('Target effect');dpg.add_combo(tuple(TARGETS),tag=E+'target',default_value='',width=-1,callback=self._invalidate_simulation_send_review)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='Review Send impact',tag=E+'review',callback=self._review_simulation_send)
                        dpg.add_button(label='Send selected data to Analysis',tag=E+'send',callback=self._send_simulation_widgets,enabled=False)
                        dpg.add_button(label='Copy full review',tag=E+'copy_review',callback=lambda:dpg.set_clipboard_text(dpg.get_value(E+'review_text')))
                    dpg.add_input_text(tag=E+'review_text',readonly=True,multiline=True,width=-1,height=TOKENS.heights.simulation_send_review)
            dpg.add_text('',tag=E+'message',wrap=TOKENS.wraps.simulation_exchange,color=TOKENS.colors.warning)
        self._register_owned_root(E+'window')

    def _show_simulation_exchange(self,*args):
        dpg=self._require_dpg();self._size_display_window(E+'window');dpg.configure_item(E+'window',show=True)
        self._refresh_simulation_exchange()

    def _sim_exchange_error(self,exc):
        self._sim_exchange_message=f'{type(exc).__name__}: {exc}';self._sim_exchange_trace=traceback.format_exc()

    def _export_simulation_widgets(self,*args):
        dpg=self._require_dpg()
        try:
            value=str(dpg.get_value(E+'destination')).strip()
            if not value or not Path(value).is_absolute():raise ValueError('Enter an absolute NEW destination directory')
            self.controller.start_simulation_export(value,figure_formats=tuple(f for f in ('png','svg','pdf') if dpg.get_value(E+f)),
                include_xlsx=bool(dpg.get_value(E+'xlsx')),plot_views=self._plot_views.freeze(),
                legend_policy='auto' if dpg.get_value(E+'legend_auto') else 'follow')
            self._sim_exchange_message='';self._sim_exchange_trace=''
        except Exception as exc:self._sim_exchange_error(exc)
        self._refresh_simulation_exchange()

    def _open_simulation_widgets(self,*args):
        try:
            value=str(self._require_dpg().get_value(E+'open_path')).strip()
            if not value or not Path(value).is_absolute():raise ValueError('Enter an absolute Simulation bundle directory')
            self.controller.open_simulation_bundle(value)
            self._sim_exchange_message='Loaded verified Simulation result; Analysis and draft unchanged.';self._sim_exchange_trace=''
        except Exception as exc:self._sim_exchange_error(exc)
        self._refresh_simulation()

    def _invalidate_simulation_send_review(self,*args):
        self._sim_send_plan=None
        self._sim_exchange_message='Choose source/target and Review again. No data changed.'
        self._require_dpg().set_value(E+'review_text',self._sim_exchange_message)
        self._refresh_simulation_exchange()

    def _simulation_send_target_key(self):
        s=self.controller.analysis_session
        return (id(s),s.current_revision,s.fit_generation,id(s.dataset),id(s.current_analysis_config),
            id(s.diagnostics),s.fit_stale,s.diagnostics_stale)

    def _review_simulation_send(self,*args):
        dpg=self._require_dpg();self._sim_send_plan=None
        try:
            channel=str(dpg.get_value(E+'channel'));target=TARGETS.get(str(dpg.get_value(E+'target')))
            plan=self.controller.prepare_simulation_send(channel,target)
            self._sim_send_plan=plan;self._sim_send_review_job=self.controller.simulation.selected_id
            self._sim_send_review_target=self._simulation_send_target_key()
            d=plan.imported.dataset;origin=d.metadata['simulation_origin']
            dpg.set_value(E+'review_text',f'SOURCE: {d.metadata["dataset_name"]}\n{origin["source_kind"]}; {len(d.raw_time)} complete rows; s; {d.metadata["signal_unit"]}\n'
                f'Request SHA256: {origin["request_sha256"]}\nTARGET fingerprint: {plan.target_sha256}\nTARGET action: {plan.target_action}\n'+fill('IMPACT: '+plan.impact,90)+'\n'+
                fill('Given parameters stay provenance. User must separately select/apply an Analysis model and run Fit/Diagnostics.',90))
            self._sim_exchange_message='Review prepared. No data changed.';self._sim_exchange_trace=''
        except Exception as exc:self._sim_exchange_error(exc)
        self._refresh_simulation_exchange()

    def _send_simulation_widgets(self,*args):
        try:
            if self._sim_send_plan is None:raise ValueError('Review channel and target impact first')
            result=self.controller.send_simulated_data_to_analysis(self._sim_send_plan)
            self._sim_send_plan=None
            self._sim_exchange_message='Sent '+str(result.dataset.metadata['source_kind'])+' to Analysis; no Apply/Fit/Diagnostics. Save As requires a new project association.'
            self._sim_exchange_trace=''
            self._require_dpg().set_value(E+'review_text',self._sim_exchange_message)
        except Exception as exc:
            self._sim_send_plan=None
            self._sim_exchange_error(exc)
            self._require_dpg().set_value(E+'review_text','Send not completed. Review again.\n'+self._sim_exchange_message)
        self._refresh_simulation()

    def _refresh_simulation_exchange(self):
        dpg=self._require_dpg()
        if not dpg.does_item_exist(E+'window'):return
        c=self.controller;run=c.simulation.selected;bundle=c.selected_simulation_bundle()
        if self._sim_send_plan is not None:
            source_changed=c.simulation.selected_id!=getattr(self,'_sim_send_review_job',None)
            target_changed=self._simulation_send_target_key()!=getattr(self,'_sim_send_review_target',None)
            if source_changed or target_changed:
                self._sim_send_plan=None
                self._sim_exchange_message=('Selected source changed' if source_changed else 'Analysis target changed')+'; Review Send again.'
                dpg.set_value(E+'review_text',self._sim_exchange_message)
        dpg.set_value(E+'source','Select a stored result.' if run is None else f'Simulated data · {run.request.name} · {run.status}'+
            (' · Verified saved result' if bundle else '\nExport or Open this run before Send.'))
        dpg.set_value(E+'source_details','' if run is None else f'{run.job_id}\n'+(str(bundle.root) if bundle else 'No verified bundle attached.'))
        busy=c._simulation_export_task_id is not None
        ready=run is not None and run.result is not None and run.result.state.ready
        dpg.configure_item(E+'export',enabled=ready and not busy)
        dpg.configure_item(E+'cancel',enabled=busy,show=busy)
        dpg.configure_item(E+'review',enabled=bundle is not None and not busy
            and dpg.get_value(E+'channel') in {'clean','observed'} and dpg.get_value(E+'target') in TARGETS)
        dpg.configure_item(E+'send',enabled=self._sim_send_plan is not None and not busy)
        dpg.set_value(E+'status',c.simulation_export_status);dpg.set_value(E+'message',self._sim_exchange_message)
