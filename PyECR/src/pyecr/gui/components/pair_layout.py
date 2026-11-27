"""Pair-owned groups and explicit table drafts; no scientific service calls."""
import json
from pyecr.gui.theme import TOKENS,PRIMARY_THEME_TAG,CANCEL_THEME_TAG,EXPORT_THEME_TAG
from pyecr.gui.state import GuiPage
from pyecr.gui.components.widgets import table_style_kwargs
from pyecr.gui.components.step_panel import step_panel
from pyecr.gui.components.tags import WidgetTags
from pyecr.gui.pair_display import PAIR_STEPS,PLAN_STEPS,PAIR_KIND_LABELS,decode_inputs,encode_draft,member_text,source_caption,missing_conditions,short_pair_reason
from pyecr.services.paired_ecr import DIRECTIONS,PAIR_KINDS
from pyecr.report.snapshot import dataset_sha256,_sha256_json
from pyecr.services.guided_refit import fit_result_id


class PairLayoutMixin:
    def _build_pair_layout(self):
        dpg=self._require_dpg();self._rendered_pair_design_state=None
        self._pair_layout_key=self._pair_active_key=self._pair_active_hashes=None
        self._pair_details_key=None;self._pair_card_members={};self._pair_card_errors={}
        self._draft_geometry_tags=[];self._draft_duration_tags=[]
        self._pair_wrap_texts=[];self._pair_field_headings={}
        with dpg.group(tag='pyecr_pair_layout'):
            with dpg.group(tag='pyecr_pair_comparison_tab'):
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Experiment Design Assistant',tag='pyecr_pair_show_planning',callback=self._show_pair_planning)
                with dpg.group(show=False):self._build_page_status('pair')
                with dpg.group(horizontal=True):
                    with dpg.child_window(tag='pyecr_pair_workflow',width=TOKENS.analysis.left_compact,height=TOKENS.heights.dialog,border=False,no_scrollbar=True):
                        with dpg.child_window(tag='pyecr_pair_body',width=-1,height=TOKENS.heights.results,border=False):
                            with step_panel(dpg,'pair_sources',PAIR_STEPS[0],default_open=True):
                                with dpg.group(parent='pyecr_stage_pair_sources_content'):
                                    dpg.add_text('Capture / Load freezes settings. Later edits need Capture / Load again.',wrap=TOKENS.wraps.rail)
                                    for slot in ('a','b'):self._build_pair_source_card(slot)
                        with step_panel(dpg,'pair_check',PAIR_STEPS[1],default_open=True):
                            with dpg.group(parent='pyecr_stage_pair_check_content',tag='pyecr_pair_check_toolbar'):
                                dpg.add_combo(tuple(PAIR_KIND_LABELS),tag='pyecr_pair_kind',default_value='Oxidation / reduction',width=-1)
                                with dpg.group(horizontal=True):
                                    dpg.add_button(label='Compare',tag='pyecr_pair_compare',callback=self._pair_compare,enabled=False)
                                    dpg.add_button(label='Open Analysis',tag='pyecr_pair_open_analysis',callback=self._pair_open_analysis)
                                dpg.add_text('',tag='pyecr_pair_check_reason',wrap=TOKENS.wraps.rail,color=TOKENS.colors.warning)
                    with dpg.child_window(tag='pyecr_pair_results_scroll',width=-1,height=TOKENS.heights.dialog,border=False):
                        with step_panel(dpg,'pair_results',PAIR_STEPS[2],default_open=True):
                            with dpg.group(parent='pyecr_stage_pair_results_content'):
                                dpg.add_text('',tag='pyecr_pair_compact_result',wrap=TOKENS.wraps.map)
                                dpg.add_group(tag='pyecr_pair_parameters')
                                self._pair_saved_plot('Independent fitted curves','fit',TOKENS.heights.plot)
                                self._pair_saved_plot('Independent residuals','residual',TOKENS.heights.residual)
                                dpg.add_button(label='Details',callback=self._show_pair_details,tag='pyecr_pair_details_action')
                                self._build_pair_artifacts()
            with dpg.group(tag='pyecr_pair_planning_tab',show=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Back to comparison',tag='pyecr_pair_hide_planning',callback=self._hide_pair_planning)
                    dpg.add_button(label='Save / open / export',tag='pyecr_design_artifacts',callback=self._pair_show_artifacts)
                self._build_pair_planning()
            # Legacy callback adapters remain callable; no visible target-slot detour.
            with dpg.group(show=False):
                dpg.add_combo(('A','B'),tag='pyecr_pair_slot',default_value='A')
                dpg.add_input_text(tag='pyecr_pair_label')
                dpg.add_combo(DIRECTIONS,tag='pyecr_pair_direction',default_value='oxidation')
                dpg.add_input_text(tag='pyecr_pair_project_path')
                dpg.add_checkbox(tag='pyecr_pair_record_temperature')
                dpg.add_input_double(tag='pyecr_pair_temperature',default_value=800.)
                for phase in ('initial','final'):dpg.add_input_text(tag='pyecr_pair_'+phase)
                dpg.add_button(tag='pyecr_pair_capture',callback=self._pair_capture)
                dpg.add_button(tag='pyecr_pair_load',callback=self._pair_load)
                dpg.add_text('',tag='pyecr_pair_members')
                dpg.add_input_text(tag='pyecr_pair_result',multiline=True)
        with dpg.window(label='Pair Details',tag='pyecr_pair_details_window',show=False,width=TOKENS.widths.report_path,height=TOKENS.heights.dialog):
            with dpg.group(horizontal=True):
                dpg.add_button(label='Close',callback=lambda:dpg.hide_item('pyecr_pair_details_window'))
                dpg.add_button(label='Copy complete details',tag='pyecr_pair_copy_details',callback=self._copy_pair_details)
            dpg.add_input_text(tag='pyecr_pair_details_text',readonly=True,multiline=True,width=-1,height=-1)
        with dpg.window(label='Load completed project',tag='pyecr_pair_load_window',show=False,width=TOKENS.widths.report_path,height=TOKENS.heights.results):
            dpg.add_text('',tag='pyecr_pair_load_title')
            dpg.add_text('Full project directory')
            dpg.add_input_text(tag='pyecr_pair_load_path',width=-1,hint='Paste the complete project path',on_enter=True,callback=self._pair_load_selected)
            with dpg.group(horizontal=True):
                dpg.add_button(label='Load project',tag='pyecr_pair_load_confirm',callback=self._pair_load_selected)
                dpg.add_button(label='Cancel',callback=lambda:dpg.hide_item('pyecr_pair_load_window'))
            dpg.add_text('',tag='pyecr_pair_load_reason',wrap=TOKENS.wraps.map,color=TOKENS.colors.warning)
        for root in ('pyecr_pair_layout','pyecr_pair_details_window','pyecr_pair_load_window'):self._register_owned_root(root)
        for key in ('sources','check','results'):
            for tag in ('pyecr_step_pair_'+key+'_status','pyecr_step_pair_'+key+'_warning','pyecr_stage_pair_'+key+'_reason'):dpg.hide_item(tag)
        # Keep existing planning form labels above fields instead of clipping them.
        pending=list(dpg.get_item_children('pyecr_pair_planning_tab',1) or [])
        while pending:
            item=pending.pop();kind=dpg.get_item_info(item)['type']
            if kind.endswith(('mvTable','mvPlot')):continue
            if kind.endswith('mvText'):self._pair_wrap_texts.append(item)
            pending.extend(dpg.get_item_children(item,1) or [])
            if kind.endswith(('mvInputText','mvInputDouble','mvInputInt','mvCombo')):
                cfg=dpg.get_item_configuration(item);label=cfg.get('label','')
                if label:
                    parent=dpg.get_item_parent(item);wrapper=dpg.add_group(parent=parent,before=item)
                    heading=dpg.add_text(label,parent=wrapper,wrap=TOKENS.wraps.page)
                    dpg.move_item(item,parent=wrapper);dpg.configure_item(item,label='')
                    self._pair_field_headings[heading]=cfg.get('width',0)
        for tag in ('pyecr_pair_compare','pyecr_design_run'):dpg.bind_item_theme(tag,PRIMARY_THEME_TAG)
        dpg.bind_item_theme('pyecr_design_cancel',CANCEL_THEME_TAG)
        for tag in ('pyecr_pair_capture_a','pyecr_pair_capture_b','pyecr_pair_design_save','pyecr_pair_design_report'):dpg.bind_item_theme(tag,EXPORT_THEME_TAG)
        self._load_design_tables()

    def _build_pair_source_card(self,slot):
        d=self._require_dpg()
        with d.child_window(tag='pyecr_pair_card_'+slot,width=-1,auto_resize_y=True,always_auto_resize=True,border=True,no_scrollbar=True):
            d.add_text('Source '+slot.upper(),color=TOKENS.colors.csv)
            d.add_text('',tag='pyecr_pair_source_'+slot,wrap=TOKENS.wraps.rail)
            d.add_text('',tag='pyecr_pair_source_status_'+slot,wrap=TOKENS.wraps.rail)
            d.add_text('Next capture direction',wrap=TOKENS.wraps.rail)
            d.add_combo(DIRECTIONS,tag='pyecr_pair_direction_'+slot,default_value='oxidation' if slot=='a' else 'reduction',width=-1)
            with d.group(horizontal=True):
                d.add_button(label='Capture current',tag='pyecr_pair_capture_'+slot,callback=self._pair_capture,user_data=slot.upper(),enabled=False)
                d.add_button(label='Load project',tag='pyecr_pair_load_'+slot,callback=self._show_pair_load,user_data=slot.upper())
            with d.tooltip('pyecr_pair_capture_'+slot):
                d.add_text('Capture requires current Fit and matching Diagnostics. Load project uses a completed saved source.',wrap=TOKENS.wraps.rail)
            d.add_text('',tag='pyecr_pair_card_error_'+slot,wrap=TOKENS.wraps.rail,color=TOKENS.colors.warning,show=False)
            with d.collapsing_header(label='Name & conditions',tag='pyecr_pair_conditions_'+slot,default_open=False):
                d.add_text('',tag='pyecr_pair_missing_'+slot,wrap=TOKENS.wraps.rail)
                d.add_text('Name for next Capture / Load')
                d.add_input_text(tag='pyecr_pair_label_'+slot,width=-1,hint='Use source name')
                d.add_checkbox(label='Declare temperature',tag='pyecr_pair_record_temperature_'+slot)
                d.add_input_double(label='K',tag='pyecr_pair_temperature_'+slot,default_value=800.,width=TOKENS.widths.input_medium)
                for phase in ('initial','final'):
                    d.add_text(phase.title()+' condition')
                    d.add_input_text(tag='pyecr_pair_'+phase+'_'+slot,width=-1)
                d.add_text('Declarations apply on Capture / Load and must agree with source metadata.',wrap=TOKENS.wraps.rail)

    def _build_pair_artifacts(self):
        d=self._require_dpg()
        with d.collapsing_header(label='Save / open / export',default_open=False,tag='pyecr_pair_artifacts'):
            d.add_text('Snapshot or HTML destination')
            d.add_input_text(tag='pyecr_pair_design_path',width=-1,hint='Full snapshot path (.json)')
            with d.group(horizontal=True):
                d.add_button(label='Save snapshot',tag='pyecr_pair_design_save',callback=self._pair_design_save)
                d.add_button(label='Open snapshot',tag='pyecr_pair_design_open',callback=self._pair_design_open)
            with d.group(horizontal=True):
                d.add_button(label='Export HTML',tag='pyecr_pair_design_report',callback=self._pair_design_report)
                d.add_button(label='Clear pair / planning',tag='pyecr_pair_design_clear',callback=self._pair_design_clear)
            d.add_text('Snapshots include stored pair and planning results. HTML contains tables and explanations.',wrap=TOKENS.wraps.map)
            d.add_text('',tag='pyecr_pair_artifact_status',wrap=TOKENS.wraps.map,show=False)

    def _show_pair_planning(self,*args):
        d=self._require_dpg();d.hide_item('pyecr_pair_comparison_tab');d.show_item('pyecr_pair_planning_tab')

    def _hide_pair_planning(self,*args):
        d=self._require_dpg();d.show_item('pyecr_pair_comparison_tab');d.hide_item('pyecr_pair_planning_tab')

    def _pair_show_artifacts(self,*args):
        self._hide_pair_planning();d=self._require_dpg()
        d.set_value('pyecr_stage_pair_results',True);d.set_value('pyecr_pair_artifacts',True)
        self._pair_scroll_to_artifacts=3

    def _pair_open_analysis(self,*args):
        d=self._require_dpg();d.set_value(WidgetTags.PAGES,WidgetTags.page('analysis'))
        self._on_tab_change(None,WidgetTags.page('analysis'),None);self._navigate_workflow(user_data='advanced')

    def _build_pair_planning(self):
        dpg=self._require_dpg()
        with dpg.group(horizontal=True):
            dpg.add_text(PLAN_STEPS[4])
            dpg.add_button(label='Scan Candidate Regions',tag='pyecr_design_run',callback=self._design_run)
            dpg.add_button(label='Cancel Scan',tag='pyecr_design_cancel',callback=self._design_cancel,enabled=False)
            dpg.add_button(label='Status details / copy',tag='pyecr_page_context_pair_design',callback=self._show_page_context,user_data='pair')
        dpg.add_button(label='Sources / results',callback=self._show_pair_details)
        with dpg.group(show=False):dpg.add_text('',tag='pyecr_page_status_pair_design')
        dpg.add_text('',tag='pyecr_design_status',wrap=TOKENS.wraps.page,color=TOKENS.colors.warning)
        dpg.add_text('Single D+k planning. Suggestions are conditional; review assumptions before use.',wrap=TOKENS.wraps.page)
        with dpg.child_window(tag='pyecr_design_body',width=-1,height=TOKENS.heights.dialog):
            with dpg.collapsing_header(label='Planning assumptions & limits',default_open=False):
                dpg.add_text('Single D+k planner only. Suggestions depend on the assumed noise, boundary and parameter ranges; they are not measured confidence intervals.',wrap=TOKENS.wraps.page,color=TOKENS.colors.warning)
            dpg.add_text('Tables are editable drafts. Use tables as scan inputs, then Scan; viewing results never changes inputs.',wrap=TOKENS.wraps.page)
            with dpg.group(horizontal=True):
                dpg.add_button(label='Use tables as scan inputs',tag='pyecr_design_use_tables',callback=self._commit_design_tables)
                dpg.add_button(label='Load current JSON into tables',tag='pyecr_design_load_tables',callback=self._load_design_tables)
            dpg.add_text('',tag='pyecr_design_table_status',wrap=TOKENS.wraps.page)
            dpg.add_text(PLAN_STEPS[0],color=TOKENS.colors.csv)
            with dpg.table(**table_style_kwargs()):
                for title in ('Parameter','Lower','Upper','Unit'):dpg.add_table_column(label=title)
                for key,unit in (('d','cm²/s'),('k','cm/s')):
                    with dpg.table_row():
                        dpg.add_text(key.upper() if key=='d' else key)
                        for bound in ('lower','upper'):dpg.add_input_text(tag=f'pyecr_design_table_{key}_{bound}',width=-1)
                        dpg.add_text(unit)
            dpg.add_input_double(label='Noise SD (normalized)',tag='pyecr_design_noise',default_value=.003,format='%.3e',width=TOKENS.widths.input_medium)
            dpg.add_text(PLAN_STEPS[1],color=TOKENS.colors.csv)
            with dpg.group(horizontal=True):
                for axis in 'xyz':dpg.add_checkbox(label='Active '+axis+' opposing pair',tag='pyecr_design_active_'+axis,default_value=axis=='x')
            dpg.add_text('Full x/y/z dimensions in cm. Active pairs are shared by all candidate rows, as in the original input.',wrap=TOKENS.wraps.page)
            dpg.add_group(tag='pyecr_design_geometry_table_host')
            with dpg.group(horizontal=True):
                dpg.add_button(label='Add geometry row',callback=self._change_design_rows,user_data=('geometries',1))
                dpg.add_button(label='Remove last geometry draft',callback=self._change_design_rows,user_data=('geometries',-1))
            dpg.add_text(PLAN_STEPS[2],color=TOKENS.colors.csv)
            dpg.add_group(tag='pyecr_design_duration_table_host')
            with dpg.group(horizontal=True):
                dpg.add_button(label='Add duration row',callback=self._change_design_rows,user_data=('durations',1))
                dpg.add_button(label='Remove last duration draft',callback=self._change_design_rows,user_data=('durations',-1))
            with dpg.group(horizontal=True):
                dpg.add_input_int(label='Points per duration',tag='pyecr_design_points',default_value=80,width=TOKENS.widths.input_medium)
                dpg.add_combo(('linear','log'),label='Sampling',tag='pyecr_design_sampling',default_value='linear',width=TOKENS.widths.input_medium)
                dpg.add_input_int(label='Log range grid per parameter',tag='pyecr_design_range_samples',default_value=3,width=TOKENS.widths.input_medium)
            dpg.add_text('Linear includes both endpoints. Log includes t=0, then duration/(N−1)² through duration. Duplicate / invalid candidates and the 512-scenario budget retain original validation.',wrap=TOKENS.wraps.page)
            dpg.add_text(PLAN_STEPS[3],color=TOKENS.colors.csv)
            with dpg.group(horizontal=True):
                dpg.add_combo(('ideal_step','exponential_flush','measured_history'),label='Boundary',tag='pyecr_design_boundary',default_value='ideal_step',width=TOKENS.widths.input_regular)
                dpg.add_input_double(label='tau_flush (s)',tag='pyecr_design_flush',default_value=10.,width=TOKENS.widths.input_small)
            with dpg.collapsing_header(label='Measured boundary — used only by measured_history',default_open=False):
                for tag,label,value in (('time','Measured time (s)','[0,10000]'),('values','Measured gas values','[0.1,1.0]')):
                    dpg.add_input_text(label=label,tag='pyecr_design_boundary_'+tag,default_value=value,width=-1)
                dpg.add_combo(('pO2','pH2O'),label='Gas variable',tag='pyecr_design_gas',default_value='pO2',width=TOKENS.widths.input_medium)
                dpg.add_input_double(label='Pre-step gas value',tag='pyecr_design_initial_gas',default_value=.1,width=TOKENS.widths.input_medium)
                dpg.add_input_double(label='Final gas value',tag='pyecr_design_final_gas',default_value=1.,width=TOKENS.widths.input_medium)
                dpg.add_input_text(label='Gas value unit',tag='pyecr_design_gas_unit',default_value='atm',width=TOKENS.widths.input_medium)
                dpg.add_text('Measured history must cover the complete candidate interval; no extrapolation.',wrap=TOKENS.wraps.page)
            with dpg.collapsing_header(label='Advanced — import / review JSON inputs',default_open=False,tag='pyecr_design_legacy_header'):
                for key,label,value in (('d','D range (cm²/s)','[5e-7, 2e-6]'),('k','k range (cm/s)','[5e-5, 2e-4]'),
                                        ('geometries','Candidate full dimensions (cm)','[[0.10,0.20,0.30],[0.03,0.20,0.30]]'),('durations','Candidate durations (s)','[1000,10000]')):
                    dpg.add_input_text(label=label,tag='pyecr_design_'+key,default_value=value,width=-1)
                dpg.add_text('After editing JSON, select Load current JSON into tables. Review the visible tables before Scan.',wrap=TOKENS.wraps.page)
            dpg.add_text(PLAN_STEPS[5],color=TOKENS.colors.csv)
            dpg.add_text('',tag='pyecr_design_candidates_summary',wrap=TOKENS.wraps.page)
            dpg.add_group(tag='pyecr_design_candidate_table_host')
            with dpg.collapsing_header(label='Complete stored planning reasons / assumptions',default_open=False):
                dpg.add_input_text(tag='pyecr_design_result',readonly=True,multiline=True,width=-1,height=TOKENS.heights.results)
            with dpg.group(horizontal=True):
                dpg.add_combo((),label='Preview candidate',tag='pyecr_design_candidate',callback=self._design_preview,width=TOKENS.widths.input_medium)
                dpg.add_input_int(label='Scenario index (0-based)',tag='pyecr_design_scenario',default_value=0,callback=self._design_preview,width=TOKENS.widths.input_medium)
            dpg.add_text('No planning preview.',tag='pyecr_design_preview_summary',wrap=TOKENS.wraps.page)
            with dpg.plot(label='Stored planning sensitivity (signal/decade)',tag='pyecr_design_sensitivity_plot',height=TOKENS.heights.plot,width=-1):
                dpg.add_plot_legend();dpg.add_plot_axis(dpg.mvXAxis,label='Time (s)',tag='pyecr_design_sensitivity_x')
                with dpg.plot_axis(dpg.mvYAxis,label='∂y/∂log10(parameter)',tag='pyecr_design_sensitivity_y'):
                    for suffix,label in (('d','S_logD'),('k','S_logk')):dpg.add_line_series([],[],label=label,tag='pyecr_design_sensitivity_'+suffix)

    def _pair_saved_plot(self,title,kind,height):
        dpg=self._require_dpg()
        with dpg.plot(label=title,tag=f'pyecr_pair_{kind}_plot',height=height,width=-1):
            dpg.add_plot_legend();dpg.add_plot_axis(dpg.mvXAxis,label='Fit time (s)',tag=f'pyecr_pair_{kind}_x')
            with dpg.plot_axis(dpg.mvYAxis,label='Signal' if kind=='fit' else 'Observed − fitted',tag=f'pyecr_pair_{kind}_y'):
                for suffix in ('a','b'):dpg.add_line_series([],[],label=suffix.upper()+(' fit' if kind=='fit' else ' residual'),tag=f'pyecr_pair_{kind}_{suffix}')

    def _legacy_design_inputs(self):
        return {key:self._require_dpg().get_value('pyecr_design_'+key) for key in ('d','k','geometries','durations')}

    def _design_draft(self):
        dpg=self._require_dpg()
        return {'d':[dpg.get_value('pyecr_design_table_d_'+b) for b in ('lower','upper')],
                'k':[dpg.get_value('pyecr_design_table_k_'+b) for b in ('lower','upper')],
                'geometries':[[dpg.get_value(tag) for tag in row] for row in self._draft_geometry_tags],
                'durations':[dpg.get_value(tag) for tag in self._draft_duration_tags]}

    def _render_design_draft(self,draft):
        dpg=self._require_dpg()
        for key in ('d','k'):
            for bound,value in zip(('lower','upper'),draft[key]):dpg.set_value(f'pyecr_design_table_{key}_{bound}',str(value))
        for key in ('geometry','duration'):dpg.delete_item('pyecr_design_'+key+'_table_host',children_only=True)
        self._draft_geometry_tags=[];self._draft_duration_tags=[]
        with dpg.table(parent='pyecr_design_geometry_table_host',**table_style_kwargs()):
            for label in ('Candidate draft','Full x (cm)','Full y (cm)','Full z (cm)'):dpg.add_table_column(label=label)
            for index,row in enumerate(draft['geometries']):
                tags=[]
                with dpg.table_row():
                    dpg.add_text('G'+str(index+1))
                    for axis,value in zip('xyz',row):
                        tag=f'pyecr_design_table_geometry_{index}_{axis}';tags.append(tag)
                        dpg.add_input_text(tag=tag,default_value=str(value),width=-1)
                self._draft_geometry_tags.append(tags)
        with dpg.table(parent='pyecr_design_duration_table_host',**table_style_kwargs()):
            dpg.add_table_column(label='Duration draft');dpg.add_table_column(label='Duration (s)')
            for index,value in enumerate(draft['durations']):
                tag=f'pyecr_design_table_duration_{index}';self._draft_duration_tags.append(tag)
                with dpg.table_row():dpg.add_text('T'+str(index+1));dpg.add_input_text(tag=tag,default_value=str(value),width=-1)

    def _load_design_tables(self,*args):
        try:
            parsed=decode_inputs(self._legacy_design_inputs());self._render_design_draft(parsed)
            text='Loaded input text into drafts; scan inputs and Analysis unchanged.'
        except (ValueError,TypeError) as error:text='Draft load rejected: '+str(error)
        self._require_dpg().set_value('pyecr_design_table_status',text)

    def _commit_design_tables(self,*args):
        try:
            values=encode_draft(self._design_draft(),self._legacy_design_inputs())
            for key,value in values.items():self._require_dpg().set_value('pyecr_design_'+key,value)
            text='Tables set as scan inputs. Scan explicitly runs original validation; Analysis is unchanged.'
        except (ValueError,TypeError) as error:text='Table inputs rejected; original scan inputs retained: '+str(error)
        self._require_dpg().set_value('pyecr_design_table_status',text)

    def _design_draft_reason(self):
        """Never scan hidden old values while the visible table shows edits."""
        try:
            original=self._legacy_design_inputs()
            current=encode_draft(self._design_draft(),original)
            if decode_inputs(current)!=decode_inputs(original):
                return 'Tables changed. Select Use tables as scan inputs before Scan.'
        except (ValueError,TypeError,OverflowError) as error:
            return 'Review planning tables: '+str(error)
        return ''

    def _change_design_rows(self,sender=None,app_data=None,user_data=None):
        key,delta=user_data;draft=self._design_draft()
        if delta>0:draft[key].append(list(draft[key][-1]) if key=='geometries' and draft[key] else (['','',''] if key=='geometries' else ''))
        elif draft[key]:draft[key].pop()
        self._render_design_draft(draft)
        self._require_dpg().set_value('pyecr_design_table_status','Draft rows changed; use tables explicitly before Scan. Duplicate candidates are not silently removed.')

    def _show_pair_details(self,*args):
        self._size_display_window('pyecr_pair_details_window');self._require_dpg().configure_item('pyecr_pair_details_window',show=True)
        self._refresh_pair_layout()

    def _copy_pair_details(self,*args):
        self._ensure_pair_details()
        self._require_dpg().set_clipboard_text(self._require_dpg().get_value('pyecr_pair_details_text'))

    def _ensure_pair_details(self):
        d=self._require_dpg();state=self.controller.session.pair_design_state
        key=(state,self._pair_active_key)
        if self._pair_details_key is not None and self._pair_details_key[0] is state and self._pair_details_key[1]==self._pair_active_key:return
        self._pair_details_key=key
        text='Captured sources / comparison / planning. Read-only stored evidence; no fit or scan.\n\n'
        text+='\n\n'.join(member_text(slot.upper(),member,self._pair_active_relations.get(slot))
            for slot,member in (('a',state.member_a),('b',state.member_b)))
        text+='\n\n'+json.dumps(state.to_dict(),ensure_ascii=False,indent=2,allow_nan=True)
        d.set_value('pyecr_pair_details_text',text)

    def _refresh_pair_layout(self):
        dpg=self._require_dpg()
        if not hasattr(self,'_pair_wrap_texts') or not getattr(dpg,'does_item_exist',lambda _:False)('pyecr_pair_layout'):return
        session=self.controller.analysis_session;state=self.controller.session.pair_design_state
        key=(id(session),id(session.dataset),id(session.current_analysis_config),id(session.fit_result),id(session.diagnostics),
             session.fit_revision,session.diagnostics_fit_revision,session.fit_stale,session.diagnostics_stale)
        active_changed=key!=self._pair_active_key
        if active_changed:
            self._pair_active_key=key
            self._pair_active_hashes=None if session.fit_result is None or session.dataset is None or session.current_analysis_config is None else (
                dataset_sha256(session.dataset),_sha256_json(session.current_analysis_config.to_dict()),fit_result_id(session.fit_result))
        changed=state is not self._pair_layout_key
        if changed or active_changed:
            self._pair_active_relations={}
            for slot,member in (('a',state.member_a),('b',state.member_b)):
                active=None
                if member is not None and self._pair_active_hashes is not None:
                    snap=member.snapshot
                    active={'matches':self._pair_active_hashes==(snap.dataset_sha256,_sha256_json(snap.analysis_config.to_dict()),member.fit_id),
                            'stale':session.fit_stale or session.diagnostics_stale}
                self._pair_active_relations[slot]=active
                dpg.set_value('pyecr_pair_source_'+slot,source_caption(member))
                status='Capture needs current Fit and Diagnostics.' if member is None and not self.controller.can_capture_pair_member else ''
                if member is not None:
                    status=''
                    from pyecr.services.paired_ecr import point_comparison_supported
                    if not point_comparison_supported(member.snapshot):status='Point comparison blocked — review source Diagnostics.'
                    elif member.snapshot.diagnostics_result.get('reliable') is not True:status='Point comparison available · interval assumptions unverified.'
                    if active and active['matches'] and active['stale']:status='Historical snapshot · active Analysis is out of date.'+('\n'+status if status else '')
                dpg.set_value('pyecr_pair_source_status_'+slot,status)
                dpg.configure_item('pyecr_pair_source_status_'+slot,show=bool(status))
                if self._pair_card_members.get(slot) is not member:
                    self._pair_card_members[slot]=member
                    self._pair_card_errors.pop(slot,None)
                    missing=missing_conditions(member)
                    dpg.set_value('pyecr_pair_missing_'+slot,('Missing: '+', '.join(missing)+'. Declare if known, then capture / load again.') if missing else 'Recorded conditions are available in Details.')
                    dpg.set_value('pyecr_pair_conditions_'+slot,bool(missing))
                    if member is not None:
                        dpg.set_value('pyecr_pair_direction_'+slot,member.direction)
                        dpg.set_value('pyecr_pair_label_'+slot,member.label)
                        dpg.set_value('pyecr_pair_record_temperature_'+slot,'temperature_k' in member.context)
                        if 'temperature_k' in member.context:dpg.set_value('pyecr_pair_temperature_'+slot,float(member.context['temperature_k']))
                        for phase in ('initial','final'):dpg.set_value('pyecr_pair_'+phase+'_'+slot,str(member.context.get(phase+'_condition','')))
                    else:
                        dpg.set_value('pyecr_pair_label_'+slot,'')
                        dpg.set_value('pyecr_pair_record_temperature_'+slot,False)
                        for phase in ('initial','final'):dpg.set_value('pyecr_pair_'+phase+'_'+slot,'')
        result=state.comparison
        if result is None:
            missing=[slot for slot,member in (('A',state.member_a),('B',state.member_b)) if member is None]
            reason='Capture or load '+ ' and '.join(missing)+'.' if missing else 'Sources captured. Select Compare to check compatibility.'
            text='Compare the captured sources to create results.'
        elif not result.ready:
            reason='\n'.join(dict.fromkeys(short_pair_reason(r) for r in result.compatibility.blocking_reasons))
            text='Comparison blocked. Resolve the checks beside Compare; full reasons are in Details.'
        else:
            reason=''
            text='Comparison ready · '+next((label for label,value in PAIR_KIND_LABELS.items() if value==result.kind),result.kind)
            if result.compatibility.warnings:text+='\nCondition or reliability cautions: review Details before interpreting differences.'
        action_error=getattr(self,'_pair_action_error','')
        if action_error:reason=action_error+'\n'+reason
        dpg.set_value('pyecr_pair_check_reason',reason);dpg.configure_item('pyecr_pair_check_reason',show=bool(reason))
        dpg.set_value('pyecr_pair_compact_result',text)
        dpg.configure_item('pyecr_pair_compact_result',color=(TOKENS.colors.warning if result is not None and result.ready and result.compatibility.warnings else
            TOKENS.colors.success if result is not None and result.ready else TOKENS.steps.text))
        for slot in ('a','b'):
            error=self._pair_card_errors.get(slot,'');dpg.set_value('pyecr_pair_card_error_'+slot,error);dpg.configure_item('pyecr_pair_card_error_'+slot,show=bool(error))
        artifact=getattr(self,'_pair_artifact_message','') or getattr(self,'_pair_action_error','')
        dpg.set_value('pyecr_pair_artifact_status',artifact)
        dpg.configure_item('pyecr_pair_artifact_status',show=bool(artifact),color=TOKENS.colors.warning if getattr(self,'_pair_action_error','') else TOKENS.colors.success)
        for kind in ('fit','residual'):dpg.configure_item('pyecr_pair_'+kind+'_plot',show=result is not None and result.ready)
        if changed:
            dpg.delete_item('pyecr_pair_parameters',children_only=True);self._pair_parameter_texts=[]
            if result is not None and result.ready:
                for row in result.parameters:
                    name=result.parameter_label(row.parameter).replace('_cm2_s','').replace('_cm_s','').replace('_',' ')
                    text=f'{name} ({row.unit})\nA  {row.value_a:.6g}     B  {row.value_b:.6g}\nB − A  {row.difference_b_minus_a:.6g}   ({row.relative_difference:+.2%} relative to A)'
                    item=dpg.add_text(text,parent='pyecr_pair_parameters',wrap=TOKENS.wraps.map);self._pair_parameter_texts.append(item)
                    dpg.add_separator(parent='pyecr_pair_parameters')
        dpg.configure_item('pyecr_design_sensitivity_plot',show=bool(state.plans))
        if state is not self._pair_layout_key:
            self._pair_layout_key=state;dpg.delete_item('pyecr_design_candidate_table_host',children_only=True)
            plans=state.plans
            summary='No stored planning candidates. Review inputs and Scan.'
            if plans:
                plan=plans[-1];summary=f'{len(plan.candidates)} stored candidates; conditional suggested region: '+(', '.join(plan.recommended_candidate_ids) or 'none')
                with dpg.table(parent='pyecr_design_candidate_table_host',scrollY=True,height=TOKENS.heights.preview,**table_style_kwargs()):
                    for title in ('ID','Full cm / active pairs','Duration (s)','Conditional fraction','Region flag','Reasons'):dpg.add_table_column(label=title)
                    for c in plan.candidates:
                        with dpg.table_row():
                            for value in (c.candidate_id,str(c.geometry.full_dims_cm)+' / '+str(c.geometry.active_dimensions),str(c.duration_s),
                                          f'{c.supported_fraction:.1%}',str(c.recommended_region),'; '.join(c.reasons) or 'See sampled assumptions'):
                                item=dpg.add_text(value[:55]+('…' if len(value)>55 else ''))
                                with dpg.tooltip(item):dpg.add_text(value,wrap=TOKENS.wraps.map)
            dpg.set_value('pyecr_design_candidates_summary',summary)
        if dpg.get_item_configuration('pyecr_pair_details_window')['show']:self._ensure_pair_details()
        width=dpg.get_viewport_client_width();height=dpg.get_viewport_client_height()
        left=min(TOKENS.analysis.left_compact,max(TOKENS.analysis.left_min,width*.36))
        top=dpg.get_item_pos('pyecr_pair_workflow')[1]
        available=max(TOKENS.heights.results,height-max(0,top)-TOKENS.spacing.spacer_medium)
        dpg.configure_item('pyecr_pair_workflow',width=left,height=available)
        dpg.configure_item('pyecr_pair_results_scroll',height=available)
        if getattr(self,'_pair_scroll_to_artifacts',False):
            dpg.set_y_scroll('pyecr_pair_results_scroll',dpg.get_y_scroll_max('pyecr_pair_results_scroll'))
            self._pair_scroll_to_artifacts-=1
        # Reserve the actual check-card height so long blocker lists remain reachable.
        check_height=dpg.get_item_rect_size('pyecr_step_panel_pair_check')[1] or TOKENS.heights.history
        dpg.configure_item('pyecr_pair_body',height=max(TOKENS.heights.notes,available-check_height-TOKENS.spacing.spacer_medium))
        pending=list(dpg.get_item_children('pyecr_pair_body',1) or [])+[dpg.get_alias_id('pyecr_pair_check_toolbar')]
        while pending:
            item=pending.pop()
            if dpg.get_item_info(item)['type'].endswith('mvText'):dpg.configure_item(item,wrap=max(TOKENS.widths.input_medium,left-65))
            pending.extend(dpg.get_item_children(item,1) or [])
        for item in self._pair_parameter_texts:dpg.configure_item(item,wrap=max(TOKENS.widths.panel_field,width-left-100))
        dpg.configure_item('pyecr_pair_compact_result',wrap=max(TOKENS.widths.panel_field,width-left-100))
        dpg.configure_item('pyecr_pair_artifact_status',wrap=max(TOKENS.widths.panel_field,width-left-100))
        for item in self._pair_wrap_texts:dpg.configure_item(item,wrap=max(TOKENS.widths.panel_field,width-TOKENS.viewport.outer_margin))
        for item,preferred in self._pair_field_headings.items():
            dpg.configure_item(item,wrap=min(preferred,width-TOKENS.viewport.outer_margin) if preferred>0 else width-TOKENS.viewport.outer_margin)
        dpg.configure_item('pyecr_design_body',height=max(TOKENS.heights.history,height-max(0,dpg.get_item_pos('pyecr_design_body')[1])-TOKENS.spacing.spacer_medium))
        running=self.controller.state.tasks[GuiPage.PAIR].status=='running'
        task=self.controller.state.tasks[GuiPage.PAIR]
        message=(f'Scanning candidates… {task.progress:.0%}' if running else
                 'Scan cancelled. Stored results are retained.' if task.status=='cancelled' else
                 short_pair_reason(task.message) if task.status=='error' else '')
        if not running and not message:message=self._design_draft_reason()
        if getattr(self,'_pair_action_error',''):message=self._pair_action_error
        dpg.set_value('pyecr_design_status',message);dpg.configure_item('pyecr_design_status',show=bool(message))
        for tag in ('pyecr_design_use_tables','pyecr_design_load_tables'):dpg.configure_item(tag,enabled=not running)
