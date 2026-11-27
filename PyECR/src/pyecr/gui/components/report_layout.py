"""Sequential Reports UI; original snapshot/export callbacks and service retained."""
from pathlib import Path
from threading import Thread
from queue import Queue, Empty
from pyecr.gui.report_directory import choose_parent_directory
from pyecr.gui.theme import TOKENS
from pyecr.gui.state import GuiPage
from pyecr.gui.report_display import (PARENT_MODE, MANUAL_MODE, CURRENT_SOURCE, HISTORY_SOURCE, compose_target, check_new_target)
from pyecr.gui.components.action_state import add_action_tooltip, set_action_availability
from pyecr.gui.components.scientific_input import _directory_dialog_path
from pyecr.gui.components.reports_workflow import ReportsWorkflowMixin


class ReportLayoutMixin(ReportsWorkflowMixin):
    def _build_legacy_report_layout(self):
        dpg = self._require_dpg()
        self._report_generated_destination = ''
        self._report_manual_draft = ''
        self._report_last_mode = PARENT_MODE
        self._report_preview_pending = None
        self._report_preview_fitted = None
        self._report_preview_release = 0
        self._report_picker_running = False
        self._report_picker_results = Queue()
        with dpg.group(tag='pyecr_report_layout'):
            dpg.add_text('Reports — select source → preset → new destination → export and verify', tag='pyecr_report_layout_anchor')
            with dpg.group(horizontal=True, tag='pyecr_report_columns'):
                with dpg.child_window(tag='pyecr_report_setup', width=TOKENS.report.left_compact, height=TOKENS.report.build_height):
                    self._report_heading('01  Source', 'source')
                    dpg.add_combo((CURRENT_SOURCE, HISTORY_SOURCE), tag='pyecr_report_source', default_value=CURRENT_SOURCE,
                                  width=-1, callback=self._report_layout_changed)
                    dpg.add_text('', tag='pyecr_report_source_summary', wrap=TOKENS.report.wrap_left)
                    self._report_heading('02  Capture or select', 'capture')
                    dpg.add_button(label='Capture Current Snapshot', tag='pyecr_report_capture', enabled=False,
                                   callback=self._capture_report_snapshot)
                    dpg.add_text('Historical selection uses the one captured snapshot in this session / reopened project. It never recaptures the current analysis.',
                                 tag='pyecr_report_history_help', wrap=TOKENS.report.wrap_left, show=False)
                    dpg.add_input_text(tag='pyecr_report_snapshot_status', default_value='No AnalysisSnapshot captured.',
                                       readonly=True, multiline=True, width=-1, height=TOKENS.report.snapshot_height)
                    self._report_heading('03  Preset', 'preset')
                    dpg.add_combo(('review', 'excel', 'publication'), tag='pyecr_export_preset', default_value='review',
                                  width=-1, callback=self._export_options_changed)
                    dpg.add_text('', tag='pyecr_report_preset_summary', wrap=TOKENS.report.wrap_left)
                    with dpg.collapsing_header(label='Complete file list / scientific precision', default_open=False):
                        dpg.add_text('', tag='pyecr_export_file_preview', wrap=TOKENS.report.wrap_left)
                        dpg.add_text('CSV / JSON retain scientific precision (17 digits). Excel numeric precision is 15 digits; exact replay uses CSV / JSON.',
                                     tag='pyecr_report_precision', wrap=TOKENS.report.wrap_left)
                    self._report_heading('04  Output location — new folder only', 'location')
                    dpg.add_combo((PARENT_MODE, MANUAL_MODE), default_value=PARENT_MODE, tag='pyecr_report_path_mode',
                                  callback=self._report_location_changed, width=-1)
                    with dpg.group(tag='pyecr_report_parent_inputs'):
                        dpg.add_text('Parent directory')
                        dpg.add_input_text(tag='pyecr_report_parent', width=-1, callback=self._report_location_changed)
                        dpg.add_button(label='Choose folder…', tag='pyecr_report_browse_parent', callback=self._show_report_parent_dialog)
                        dpg.add_text('New report folder name')
                        dpg.add_input_text(tag='pyecr_report_name', width=-1, hint='e.g. Sample A results', callback=self._report_location_changed)
                    with dpg.group(tag='pyecr_report_manual_inputs', show=False):
                        dpg.add_text('Full output path')
                        dpg.add_input_text(tag='pyecr_report_destination', width=-1, hint='Full path to a new folder',
                                           callback=self._report_full_path_changed)
                    dpg.add_text('', tag='pyecr_report_picker_notice', wrap=TOKENS.report.wrap_left)
                    with dpg.collapsing_header(label='Advanced export settings', tag='pyecr_report_advanced', default_open=False):
                        for label, tag, values, value in (
                            ('New CSV encoding', 'pyecr_export_encoding', ('utf-8', 'utf-8-sig'), 'utf-8'),
                            ('Delimiter', 'pyecr_export_delimiter', (',', ';', 'tab'), ','),
                            ('PNG DPI', 'pyecr_export_dpi', ('300', '600'), '300')):
                            dpg.add_text(label);dpg.add_combo(values, tag=tag, default_value=value, width=-1)
                        for label, tag, value in (('Width (inch)', 'pyecr_export_width', 7.2), ('Height (inch)', 'pyecr_export_height', 4.8)):
                            dpg.add_text(label);dpg.add_input_float(tag=tag, default_value=value, width=-1)
                        dpg.add_text('Figure font');dpg.add_input_text(tag='pyecr_export_font', default_value='DejaVu Sans', width=-1)
                        dpg.add_checkbox(label='Auto-place legends (export only)',tag='pyecr_export_legend_auto',default_value=False)
                        dpg.add_text('Publication includes curve / residual figure PDFs. A full-text report PDF is a separate optional renderer; it is not enabled by the publication preset.',
                                     tag='pyecr_report_pdf_explanation', wrap=TOKENS.report.wrap_left)
                        dpg.add_text('PDF: optional/not configured', tag='pyecr_report_pdf_status', wrap=TOKENS.report.wrap_left)
                with dpg.child_window(tag='pyecr_report_output', width=TOKENS.report.build_right, height=TOKENS.report.build_height, no_scrollbar=True):
                    self._report_heading('05  Export and verify', 'export')
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='Export Current Analysis', tag='pyecr_report_export_current', enabled=False, callback=self._export_current_report)
                        dpg.add_button(label='Export Historical Snapshot', tag='pyecr_report_export_captured', enabled=False, show=False, callback=self._export_captured_report)
                        dpg.add_button(label='Cancel Export', tag='pyecr_export_cancel', enabled=False, callback=self._cancel_export)
                    self._build_page_status('reports');dpg.configure_item('pyecr_page_status_row_reports', horizontal=False)
                    dpg.add_text('', tag='pyecr_report_disabled_reason', wrap=TOKENS.report.wrap_right)
                    dpg.add_input_text(tag='pyecr_report_target_preview', readonly=True, multiline=True, width=-1, height=TOKENS.report.preview_height)
                    dpg.add_text('', tag='pyecr_report_target_status', wrap=TOKENS.report.wrap_right, color=TOKENS.colors.warning)
                    dpg.add_button(label='Copy target path', tag='pyecr_report_copy_target', callback=self._copy_report_target, enabled=False)
                    dpg.add_text('', tag='pyecr_report_results_anchor')
                    with dpg.child_window(tag='pyecr_report_results', width=-1, height=TOKENS.report.results_height):
                        self._report_heading('06  Result files', 'files')
                        dpg.add_text('', tag='pyecr_report_receipt', wrap=TOKENS.report.wrap_right)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label='Open verified folder', tag='pyecr_report_open_output', enabled=False, callback=self._open_report_output)
                            dpg.add_button(label='Copy verified path', tag='pyecr_report_copy_output', enabled=False, callback=self._copy_report_output)
                        with dpg.tab_bar(tag='pyecr_report_result_tabs'):
                            with dpg.tab(label='Verified files', tag='pyecr_report_files_tab'):
                                dpg.add_input_text(tag='pyecr_report_artifacts', default_value='No report exported.', readonly=True,
                                                   multiline=True, width=-1, height=TOKENS.report.artifacts_height)
                            with dpg.tab(label='Captured curve', tag='pyecr_report_curve_tab'):
                                with dpg.plot(label='Captured joint-fit preview', tag='pyecr_export_preview_plot', height=TOKENS.report.plot_height, width=-1, show=False):
                                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest)
                                    dpg.add_plot_axis(dpg.mvXAxis, label='Time (s)', tag='pyecr_export_preview_x')
                                    axis=dpg.add_plot_axis(dpg.mvYAxis, label='Signal', tag='pyecr_export_preview_y')
                                    dpg.add_scatter_series([], [], label='Captured observed', parent=axis, tag='pyecr_export_preview_observed')
                                    dpg.add_line_series([], [], label='Captured fitted', parent=axis, tag='pyecr_export_preview_fit')
        self._register_owned_root('pyecr_report_layout')
        for tag in ('pyecr_report_capture', 'pyecr_report_export_current', 'pyecr_report_export_captured', 'pyecr_export_cancel'):
            add_action_tooltip(dpg, tag)
        self._export_options_changed()

    def _report_heading(self, label, key):
        dpg=self._require_dpg();dpg.add_separator();dpg.add_text(label, tag='pyecr_report_heading_'+key, color=TOKENS.colors.csv)

    def _report_layout_changed(self, *args):
        self._refresh_widgets()

    def _report_location_changed(self, sender=None, app_data=None, user_data=None):
        dpg=self._require_dpg();mode=dpg.get_value('pyecr_report_path_mode')
        if mode==MANUAL_MODE and self._report_last_mode!=MANUAL_MODE:
            dpg.set_value('pyecr_report_destination', self._report_manual_draft)
        elif mode==PARENT_MODE and self._report_last_mode==MANUAL_MODE:
            self._report_manual_draft=str(dpg.get_value('pyecr_report_destination'))
            dpg.set_value('pyecr_report_destination', self._report_generated_destination)
        self._report_last_mode=mode
        dpg.set_value('pyecr_report_picker_notice', '')
        self._refresh_widgets()

    def _report_full_path_changed(self, *args):
        dpg=self._require_dpg();self._report_manual_draft=str(dpg.get_value('pyecr_report_destination'))
        dpg.set_value('pyecr_report_path_mode', MANUAL_MODE);self._report_last_mode=MANUAL_MODE
        self._refresh_widgets()

    def _resolve_report_destination(self):
        dpg=self._require_dpg();mode=dpg.get_value('pyecr_report_path_mode')
        value=str(dpg.get_value('pyecr_report_destination'))
        # Existing callers may write the stable full-path tag without a callback.
        if mode==PARENT_MODE and value and value!=self._report_generated_destination:
            dpg.set_value('pyecr_report_path_mode', MANUAL_MODE);self._report_last_mode=MANUAL_MODE
            self._report_manual_draft=value;mode=MANUAL_MODE
        if mode==PARENT_MODE:
            try:target=compose_target(dpg.get_value('pyecr_report_parent'), dpg.get_value('pyecr_report_name'))
            except (ValueError, OSError):
                self._report_generated_destination='';dpg.set_value('pyecr_report_destination', '')
                raise
            self._report_generated_destination=str(target);dpg.set_value('pyecr_report_destination', str(target))
        else:target=value
        return str(check_new_target(target))

    def _show_report_parent_dialog(self, *args):
        dpg=self._require_dpg();parent=str(dpg.get_value('pyecr_report_parent')).strip()
        if not Path(parent).is_dir():parent=str(Path.cwd())
        if self._report_picker_running:return
        self._report_picker_running=True
        dpg.set_value('pyecr_report_picker_notice','Parent chooser open — choose an existing directory or Cancel.')
        def select():
            try:self._report_picker_results.put(('selected',choose_parent_directory(parent)))
            except Exception as exc:self._report_picker_results.put(('error',str(exc)))
        # DPG's Windows directory iterator can terminate on Unicode paths.
        # Windows uses its Unicode shell chooser; only the main DPG thread edits widgets.
        Thread(target=select,name='pyecr-report-directory',daemon=True).start()

    def _report_parent_selected(self, sender=None, app_data=None, user_data=None):
        dpg=self._require_dpg()
        self._report_picker_running=False
        try:
            value=str(Path(_directory_dialog_path(app_data)).resolve())
            if not Path(value).is_dir():raise ValueError('Select an existing parent directory; no folder was created.')
        except (ValueError, OSError) as exc:
            dpg.set_value('pyecr_report_picker_notice', str(exc));return
        dpg.set_value('pyecr_report_parent', value)
        dpg.set_value('pyecr_report_picker_notice', 'Parent selected.')
        self._refresh_widgets()

    def _report_parent_cancelled(self, *args):
        dpg=self._require_dpg();self._report_picker_running=False
        dpg.set_value('pyecr_report_picker_notice', 'Folder selection cancelled.')

    def _copy_report_target(self, *args):
        self._require_dpg().set_clipboard_text(self._resolve_report_destination())

    def _copy_report_output(self, *args):
        result=self.controller.session.report_export_result
        if result is not None and result.verified:self._require_dpg().set_clipboard_text(str(result.root))

    def _open_report_output(self, *args):
        result=self.controller.session.report_export_result
        if result is not None and result.verified and result.root.is_dir():
            import os
            os.startfile(str(result.root))

    def _refresh_report_layout(self):
        dpg=self._require_dpg()
        if getattr(dpg,'does_item_exist',lambda _:False)('pyecr_reports_workflow'):
            self._refresh_reports_workflow();return
        if not getattr(dpg, 'does_item_exist', lambda _:False)('pyecr_report_layout'):return
        try:
            kind,value=self._report_picker_results.get_nowait()
        except Empty:pass
        else:
            if kind=='error':
                self._report_picker_running=False
                dpg.set_value('pyecr_report_picker_notice','Directory chooser unavailable: '+value+' — enter the existing parent manually.')
            elif value:self._report_parent_selected(app_data={'current_path':value})
            else:self._report_parent_cancelled()
        running=self.controller.state.tasks[GuiPage.REPORT].status=='running'
        history=dpg.get_value('pyecr_report_source')==HISTORY_SOURCE
        for tag, show in (('pyecr_report_export_current', not history), ('pyecr_report_export_captured', history),
                          ('pyecr_report_capture', not history), ('pyecr_report_history_help', history)):
            dpg.configure_item(tag, show=show)
        session=self.controller.analysis_session;snapshot=self.controller.session.report_snapshot
        dpg.set_value('pyecr_report_snapshot_status', str(dpg.get_value('pyecr_report_snapshot_status')).replace(' | ', '\n'))
        dpg.configure_item('pyecr_report_disabled_reason', show=bool(dpg.get_value('pyecr_report_disabled_reason')))
        source=('Explicit historical snapshot; current analysis may differ.' if history else 'Current analysis — must have a current Fit; export captures an immutable snapshot.')
        from pyecr.diagnostics.interpretation import reproducibility_text
        source += '\n' + reproducibility_text(snapshot.fit_result if history and snapshot is not None else session.fit_result)
        if not history and session.fit_stale:source+=' Out of date: current export is refused.'
        if history and snapshot is None:source+=' No captured snapshot available; capture a current Fit or reopen a completed project.'
        dpg.set_value('pyecr_report_source_summary', source)
        preset=dpg.get_value('pyecr_export_preset')
        dpg.set_value('pyecr_report_preset_summary', {'review':'Review — complete scientific CSV / JSON, dictionary and manifest.',
            'excel':'Excel — review files plus analysis.xlsx with units / status.',
            'publication':'Publication — review files plus PNG / SVG / curve and residual PDF figures.'}[preset])
        try:
            target=self._resolve_report_destination();reason='New target does not exist. Export verifies files before atomic publication.';valid=True
        except (ValueError, OSError) as exc:
            target=str(dpg.get_value('pyecr_report_destination'));reason=str(exc);valid=False
        dpg.set_value('pyecr_report_target_preview', target or 'Choose a parent and a new name, or enter a full new directory.')
        dpg.set_value('pyecr_report_target_status', reason)
        manual=dpg.get_value('pyecr_report_path_mode')==MANUAL_MODE
        dpg.configure_item('pyecr_report_parent_inputs', show=not manual);dpg.configure_item('pyecr_report_manual_inputs', show=manual)
        for tag in ('pyecr_report_source','pyecr_report_path_mode','pyecr_report_parent','pyecr_report_name','pyecr_report_destination',
                    'pyecr_report_browse_parent','pyecr_export_preset','pyecr_export_encoding','pyecr_export_delimiter','pyecr_export_dpi',
                    'pyecr_export_width','pyecr_export_height','pyecr_export_font','pyecr_export_legend_auto'):
            dpg.configure_item(tag, enabled=not running and not self._report_picker_running)
        dpg.configure_item('pyecr_report_copy_target', enabled=valid)
        from pyecr.fitting.reproducibility import reproducibility_issue
        rng_issue=reproducibility_issue(session.fit_result)
        if rng_issue and not history and not running:
            dpg.set_value('pyecr_page_status_reports','Reproducible snapshot unavailable — rerun fit.')
            dpg.configure_item('pyecr_page_status_reports',color=TOKENS.colors.warning)
        for tag, ready in (('pyecr_report_export_current', self.controller.can_capture_snapshot and not rng_issue), ('pyecr_report_export_captured', snapshot is not None and snapshot.fully_reproducible)):
            set_action_availability(dpg, tag, enabled=ready and not running and valid and not self._report_picker_running,
                                    reason=reason if not valid else rng_issue or 'Requires a current reproducible Fit / captured snapshot.')
        result=self.controller.session.report_export_result
        verified=result is not None and result.verified
        for tag in ('pyecr_report_open_output','pyecr_report_copy_output'):dpg.configure_item(tag, enabled=verified)
        receipt='No report exported. Choose a source, preset and new destination.'
        if result is not None:
            scope=(result.bundle.export_metadata or {}).get('export_scope', 'captured')
            receipt=f'{scope} bundle · {result.verification_status} · {len(result.artifacts)} files\n{result.root}\nFull-text report {dpg.get_value("pyecr_report_pdf_status")} (separate from publication figure PDFs).'
            if self.controller.state.tasks[GuiPage.REPORT].status in ('error','cancelled'):receipt+='\nPrevious verified output retained; the latest attempt was not published.'
        dpg.set_value('pyecr_report_receipt', receipt)
        width=dpg.get_viewport_client_width();height=dpg.get_viewport_client_height()
        anchor=dpg.get_item_state('pyecr_report_layout_anchor');top=anchor.get('rect_min',(0,110))[1]+anchor.get('rect_size',(0,24))[1]+12
        panel=max(TOKENS.report.panel_min, int(height-top-TOKENS.report.footer));left=TOKENS.report.left_compact if width<1600 else TOKENS.report.left_medium if width<2200 else TOKENS.report.left_large
        right=max(TOKENS.report.right_min, width-TOKENS.report.margin-TOKENS.report.gap-left)
        dpg.configure_item('pyecr_report_setup', width=left, height=panel);dpg.configure_item('pyecr_report_output', width=right, height=panel)
        results_top=dpg.get_item_state('pyecr_report_results_anchor').get('rect_min',(0,top+250))[1]
        dpg.configure_item('pyecr_report_results', height=max(TOKENS.report.results_min, panel-(results_top-top)-TOKENS.report.margin))
        for tag in ('pyecr_report_source_summary','pyecr_report_history_help','pyecr_report_preset_summary','pyecr_export_file_preview',
                    'pyecr_report_precision','pyecr_report_picker_notice','pyecr_report_pdf_explanation','pyecr_report_pdf_status'):
            dpg.configure_item(tag, wrap=left-TOKENS.report.gutter)
        for tag in ('pyecr_page_status_reports','pyecr_report_disabled_reason','pyecr_report_target_status','pyecr_report_receipt'):
            dpg.configure_item(tag, wrap=right-TOKENS.report.gutter)
        # The original renderer may receive a snapshot while this tab is hidden.
        # Initialize visible axes once from the captured arrays, then release for user zoom.
        if snapshot is not None and dpg.get_item_state('pyecr_export_preview_plot').get('visible', False) and self._report_preview_fitted is not snapshot:
            if self._report_preview_pending is snapshot:
                data=snapshot.fit_plot_data;values=(*data.observed, *data.fitted)
                lo, hi=min(values), max(values);pad=max((hi-lo)*TOKENS.plot.fit_padding[1], 1e-12)
                dpg.set_axis_limits('pyecr_export_preview_x', min(data.time_s), max(data.time_s))
                dpg.set_axis_limits('pyecr_export_preview_y', lo-pad, hi+pad)
                self._report_preview_fitted=snapshot;self._report_preview_release=dpg.get_frame_count()+2
            else:self._report_preview_pending=snapshot
        if self._report_preview_release and dpg.get_frame_count()>=self._report_preview_release:
            dpg.set_axis_limits_auto('pyecr_export_preview_x');dpg.set_axis_limits_auto('pyecr_export_preview_y');self._report_preview_release=0
