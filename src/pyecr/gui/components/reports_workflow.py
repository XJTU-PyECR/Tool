"""Three-step Reports page using the existing path and T09 transactions."""
from dataclasses import replace
from pathlib import Path
from pyecr.gui.theme import TOKENS
from pyecr.gui.state import GuiPage
from pyecr.gui.components.tags import WidgetTags
from pyecr.gui.components.step_panel import step_panel
from pyecr.gui.report_display import PARENT_MODE,MANUAL_MODE
from pyecr.gui.reports_workflow import result_choices,result_versions,figure_preview,freeze_report,short_reason,reviewed_warnings,report_notice
from pyecr.gui.components.reports_layout import ReportsGeometryMixin
from pyecr.report.figure_catalog import FigureSource,freeze,thaw

class ReportsWorkflowMixin(ReportsGeometryMixin):
    def _build_report_layout(self):
        d=self._require_dpg()
        # Stable old tags/callbacks remain available to API integrations. Their
        # review preset does not define the new page's image selection.
        with d.group(tag='pyecr_reports_legacy',show=False):self._build_legacy_report_layout()
        self._reports_sources=();self._reports_choices=();self._reports_guards=();self._reports_live=();self._reports_checks=[]
        self._reports_result_key=None;self._reports_error='';self._reports_last_result=None;self._reports_request=None
        self._reports_selection_epoch=0;self._reports_receipt_binding=None
        with d.group(tag='pyecr_reports_workflow'):
            d.add_spacer(height=0,tag='pyecr_reports_anchor')
            with d.group(horizontal=True,tag='pyecr_reports_columns'):
                with d.group(tag='pyecr_reports_setup'):
                    with step_panel(d,'reports_results','01 Results',default_open=True):
                        with d.group(parent='pyecr_stage_reports_results_content'):
                            d.add_button(label='Refresh results',tag='pyecr_reports_refresh',callback=self._reports_refresh_sources)
                            with d.tab_bar(tag='pyecr_reports_module_tabs',callback=self._reports_module_changed):
                                for module in ('Analysis','Batch','Pair','DCT','Simulation'):
                                    with d.tab(label=module,tag='pyecr_reports_module_'+module):pass
                            with d.child_window(tag='pyecr_reports_sources',width=-1,height=1,border=False):pass
                            d.add_text('',tag='pyecr_reports_source_count',wrap=TOKENS.report.wrap_left,show=False)
                            d.add_button(label='Review content',tag='pyecr_reports_review',callback=self._reports_load_content)
                            with d.collapsing_header(label='Open saved result',default_open=False):
                                d.add_input_text(tag='pyecr_reports_saved_path',hint='Path to a saved source.json',width=-1)
                                d.add_button(label='Load as history',callback=self._reports_load_history)
                    with step_panel(d,'reports_content','02 Content',default_open=True):
                        with d.group(parent='pyecr_stage_reports_content_content'):
                            d.add_checkbox(label='All available figures',tag='pyecr_reports_all',default_value=True,callback=self._reports_content_changed)
                            d.add_checkbox(label='Scientific CSV / JSON',default_value=True,enabled=False)
                            d.add_checkbox(label='PNG',tag='pyecr_reports_png',default_value=True,callback=self._reports_content_changed)
                            d.add_text('',tag='pyecr_reports_count',wrap=TOKENS.report.wrap_left,show=False)
                            d.add_text('',tag='pyecr_reports_source_warning',wrap=TOKENS.report.wrap_left,show=False,color=TOKENS.colors.warning)
                            with d.child_window(tag='pyecr_reports_figures',width=-1,height=1,border=False,show=False):pass
                            with d.collapsing_header(label='Unavailable figures',tag='pyecr_reports_unavailable',default_open=False,show=False):
                                d.add_group(tag='pyecr_reports_unavailable_rows')
                            with d.collapsing_header(label='Advanced formats',tag='pyecr_reports_formats',default_open=False):
                                d.add_checkbox(label='Auto-place legends (export only)',tag='pyecr_reports_legend_auto',default_value=False,
                                               callback=self._reports_content_changed)
                                for fmt,label in (('svg','SVG'),('pdf','Figure PDF'),('xlsx','XLSX arrays')):
                                    d.add_checkbox(label=label,tag='pyecr_reports_'+fmt,callback=self._reports_content_changed)
                                d.add_text('XLSX retains 15 digits. Figure PDF contains graphics.',wrap=TOKENS.report.wrap_left)
                with d.group(tag='pyecr_reports_output'):
                    with step_panel(d,'reports_destination','03 Destination & export',default_open=True):
                        with d.group(parent='pyecr_stage_reports_destination_content'):
                            with d.group(tag='pyecr_reports_destination'):
                                for tag in ('pyecr_report_parent_inputs','pyecr_report_manual_inputs','pyecr_report_picker_notice'):
                                    d.move_item(tag,parent='pyecr_reports_destination')
                                d.add_checkbox(label='Use full path',tag='pyecr_reports_use_full_path',callback=self._reports_path_mode_changed)
                            d.add_text('',tag='pyecr_reports_export_summary',wrap=TOKENS.report.wrap_right,show=False)
                            with d.group(horizontal=True):
                                d.add_button(label='Export',tag='pyecr_reports_export',callback=self._reports_export)
                                d.add_button(label='Cancel',tag='pyecr_reports_cancel',callback=self._reports_cancel,enabled=False)
                            d.add_text('',tag='pyecr_reports_destination_reason',wrap=TOKENS.report.wrap_right,color=TOKENS.colors.warning)
                            d.add_text('',tag='pyecr_reports_status',wrap=TOKENS.report.wrap_right)
                            d.add_text('',tag='pyecr_reports_receipt',wrap=TOKENS.report.wrap_right)
                            with d.group(horizontal=True):
                                d.add_button(label='Open folder',tag='pyecr_reports_open',callback=self._reports_open,show=False)
                                d.add_button(label='Copy path',tag='pyecr_reports_copy',callback=self._reports_copy,show=False)
                            with d.collapsing_header(label='Details',tag='pyecr_reports_details',default_open=False):
                                d.add_input_text(tag='pyecr_reports_details_text',readonly=True,multiline=True,width=-1,height=TOKENS.steps.details_height)
                                d.add_text('Figure PDF exports graphics. The Analysis report preset provides the full report.',wrap=TOKENS.report.wrap_right)
        d.bind_item_theme('pyecr_reports_export','pyecr_export_action_theme');d.bind_item_theme('pyecr_reports_cancel','pyecr_cancel_action_theme')
        for key in ('reports_results','reports_content','reports_destination'):
            for tag in ('pyecr_step_'+key+'_status','pyecr_stage_'+key+'_reason'):d.hide_item(tag)
        self._register_owned_root('pyecr_reports_workflow');self._reports_refresh_sources()

    def _open_reports_workflow(self,*args):
        d=self._require_dpg();owner=self.controller.state.active_page
        module=owner.value if owner in (GuiPage.BATCH,GuiPage.PAIR,GuiPage.DCT,GuiPage.SIMULATION) else 'Analysis'
        d.set_value('pyecr_reports_module_tabs','pyecr_reports_module_'+module);self._reports_module_changed()
        d.set_value(WidgetTags.PAGES,WidgetTags.page('reports'))
        self._on_tab_change(None,WidgetTags.page('reports'),None)
        # Navigation preserves this page's explicit content selection.
        self._refresh_report_layout()

    def _reports_sources_key(self):
        c=self.controller;s=c.analysis_session
        return (id(s),s.current_revision,id(s.fit_result),s.fit_stale,id(c.session.report_snapshot),id(c.session.batch_result),
            id(c.session.global_result),id(c.session.pair_design_state),id(c.session.dct_result),
            tuple((id(j),j.status) for service in (c.signed_dct_service,c.component_dct_service) for j in service.history),
            tuple((id(j),j.status) for j in c.simulation.history))

    def _reports_refresh_sources(self,*args,preserve=False):
        had_review=bool(self._reports_sources);old_key=self._reports_result_key
        d=self._require_dpg();previous={choice.key for tag,choice in getattr(self,'_reports_source_tags',()) if d.does_item_exist(tag) and d.get_value(tag)}
        self._reports_choices=result_choices(self.controller);self._reports_source_tags=[];self._reports_wrapped_titles=[];self._reports_empty_module_text={}
        d.delete_item('pyecr_reports_sources',children_only=True)
        for module in ('Analysis','Batch','Pair','DCT','Simulation'):
            parent='pyecr_reports_results_'+module;d.add_group(tag=parent,parent='pyecr_reports_sources')
            rows=[v for v in self._reports_choices if v.module==module]
            if module=='Analysis' and not any(v.key=='analysis' for v in rows):
                text='Current result is out of date. Select history explicitly.' if self.controller.analysis_session.fit_stale else 'Run Fit in Analysis to create a current result.'
                d.add_text(text,parent=parent,wrap=TOKENS.report.wrap_left)
                self._reports_empty_module_text[module]=text
            for choice in rows:
                tag='pyecr_reports_source_'+str(len(self._reports_source_tags))
                selected=choice.key in previous
                with d.group(horizontal=True,parent=parent):
                    d.add_checkbox(tag=tag,default_value=selected,callback=self._reports_source_changed)
                    text=d.add_text(choice.label,wrap=TOKENS.report.wrap_left)
                    with d.tooltip(text):d.add_text(choice.details or choice.label,wrap=TOKENS.report.wrap_right)
                self._reports_wrapped_titles.append(text)
                self._reports_source_tags.append((tag,choice))
            if not rows and module!='Analysis':
                text='Open '+module+' to create or load a result.';self._reports_empty_module_text[module]=text
                d.add_text(text,parent=parent,wrap=TOKENS.report.wrap_left)
        self._reports_module_changed();self._reports_result_key=self._reports_sources_key()
        if not preserve:
            reason='Current Analysis is out of date. Select history explicitly.' if 'analysis' in previous and self.controller.analysis_session.fit_stale else ''
            if not reason and had_review and old_key!=self._reports_result_key:
                reason=('Selected result changed. Review content again.' if previous.intersection(v.key for v in self._reports_choices)
                        else 'Selected result is no longer available. Select another result.')
            self._reports_invalidate(reason)
        elif getattr(self,'_reports_saved_history',False):
            d.set_value('pyecr_reports_source_count','Selected history · '+self._reports_sources[0].module)
        else:self._reports_selected_label()

    def _reports_module_changed(self,*args):
        d=self._require_dpg();selected=d.get_value('pyecr_reports_module_tabs')
        if not selected:selected='pyecr_reports_module_Analysis'
        for module in ('Analysis','Batch','Pair','DCT','Simulation'):
            tag='pyecr_reports_results_'+module
            if d.does_item_exist(tag):d.configure_item(tag,show=selected=='pyecr_reports_module_'+module)

    def _reports_selected_label(self):
        chosen=[('Global' if c.key=='global' else c.module) for tag,c in self._reports_source_tags if self._require_dpg().get_value(tag)]
        self._require_dpg().set_value('pyecr_reports_source_count',(str(len(chosen))+' selected · '+', '.join(chosen)) if chosen else '')
        self._require_dpg().configure_item('pyecr_reports_source_count',show=bool(chosen))
        self._require_dpg().configure_item('pyecr_stage_reports_results',label='01 Results'+(' · '+str(len(chosen))+' selected' if chosen else ''))

    def _reports_invalidate(self,reason):
        self._reports_selection_epoch+=1
        self._reports_sources=();self._reports_guards=();self._reports_live=();self._reports_checks=[];self._reports_error=''
        self._reports_preview_rows=();self._reports_figure_titles=[];self._reports_saved_history=False;self._reports_list_items=[];self._reports_warnings=();self._reports_unavailable_texts=[]
        d=self._require_dpg();d.delete_item('pyecr_reports_figures',children_only=True);d.delete_item('pyecr_reports_unavailable_rows',children_only=True)
        d.configure_item('pyecr_reports_unavailable',show=False);d.configure_item('pyecr_reports_source_warning',show=False)
        d.set_value('pyecr_reports_count',reason);d.configure_item('pyecr_reports_count',show=bool(reason));self._reports_selected_label()

    def _reports_source_changed(self,*args):
        self._reports_has_selection=True;self._reports_invalidate('')

    def _reports_layers(self):
        d=self._require_dpg();layers=dict(version=1)
        for tag,key in (('pyecr_signed_dct_physical','dct_physical'),('pyecr_sensitivity_normalized','sensitivity_normalized'),
            ('pyecr_component_dct_scale','component_scale'),('pyecr_component_dct_view','component_view')):
            if d.does_item_exist(tag):layers[key]=d.get_value(tag)
        return layers

    def _reports_load_content(self,*args):
        d=self._require_dpg()
        try:
            selected=[choice for tag,choice in self._reports_source_tags if d.get_value(tag)]
            if not selected:raise ValueError('Select at least one stored result.')
            if not all(c.guard() for c in selected):raise ValueError('Result changed. Refresh results and review content.')
            sources=[];layers=self._reports_layers()
            self.controller.analysis_session.project_metadata['figure_display_options']=layers
            for c in selected:
                s=c.capture()
                if c.live_view:s=replace(s,display_options=freeze(layers))
                elif c.key=='analysis_history':s=replace(s,display_options=freeze(thaw(s.payload)['result']['export_context'].get('project_metadata',{}).get('figure_display_options',{})))
                sources.append(replace(s,label=c.module+' · '+c.label))
            if len({s.source_id for s in sources})!=len(sources):raise ValueError('The same stored result is selected twice. Select it once.')
            self._reports_sources=tuple(sources);self._reports_guards=tuple(c.guard for c in selected);self._reports_live=tuple(c.live_view for c in selected)
            versions=result_versions(self.controller);self._reports_selected_versions={c.key:versions[c.key] for c in selected};self._reports_saved_history=False
            self._reports_result_key=self._reports_sources_key();self._reports_error='';self._reports_show_content()
            d.set_value('pyecr_stage_reports_results',False)
        except Exception as exc:
            self._reports_invalidate('Content was not loaded. Refresh results and review again.')
            self._reports_result_key=self._reports_sources_key();self._reports_error=str(exc)

    def _reports_load_history(self,*args):
        d=self._require_dpg()
        try:
            p=Path(str(d.get_value('pyecr_reports_saved_path')).strip())
            if not p.is_absolute():raise ValueError('Enter an absolute saved source.json path.')
            s=replace(FigureSource.from_dict(thaw(p.read_text(encoding='utf-8'))),scope='historical')
            self._reports_sources=(s,);self._reports_guards=();self._reports_live=(False,);self._reports_error='';self._reports_show_content()
            self._reports_saved_history=True
            for tag,_ in self._reports_source_tags:d.set_value(tag,False)
            d.set_value('pyecr_reports_source_count','Selected history · '+s.module)
            d.configure_item('pyecr_stage_reports_results',label='01 Results · Historical '+s.module)
            d.set_value('pyecr_stage_reports_results',False)
        except Exception as exc:
            self._reports_invalidate('Saved result was not loaded. Select a result or correct the path.');self._reports_error=str(exc)

    def _reports_show_content(self):
        d=self._require_dpg();d.delete_item('pyecr_reports_figures',children_only=True);d.delete_item('pyecr_reports_unavailable_rows',children_only=True)
        self._reports_checks=[];self._reports_figure_titles=[];self._reports_list_items=[];self._reports_unavailable_texts=[];missing=0
        self._reports_preview_rows=figure_preview(self._reports_sources)
        self._reports_warnings=reviewed_warnings(self._reports_sources)
        warning='Warnings in selected results. Details.' if self._reports_warnings else ''
        d.set_value('pyecr_reports_source_warning',warning);d.configure_item('pyecr_reports_source_warning',show=bool(warning))
        for s in self._reports_sources:
            rows=[r for r in self._reports_preview_rows if r['source_id']==s.source_id]
            if any(r['available'] for r in rows):
                label=s.label or s.module;short=label if len(label)<=TOKENS.report.label_chars else label[:TOKENS.report.label_chars-1]+'…'
                title=d.add_text(short,parent='pyecr_reports_figures',wrap=TOKENS.report.wrap_left)
                with d.tooltip(title):d.add_text(label,wrap=TOKENS.report.wrap_right)
                self._reports_list_items.append((short,False));self._reports_figure_titles.append((title,short))
            for row in rows:
                if row['available']:
                    tag='pyecr_reports_figure_'+str(len(self._reports_checks))
                    with d.group(horizontal=True,parent='pyecr_reports_figures'):
                        d.add_checkbox(tag=tag,default_value=True,callback=self._reports_content_changed)
                        text=d.add_text(row['title'],wrap=TOKENS.report.wrap_left)
                    self._reports_figure_titles.append((text,row['title']))
                    self._reports_checks.append((tag,s.source_id+'/'+row['figure_id']))
                    self._reports_list_items.append((row['title'],True))
                else:
                    missing+=1
                    text=d.add_text(s.module+' · '+row['title']+': '+short_reason(row['reason']),parent='pyecr_reports_unavailable_rows',wrap=TOKENS.report.wrap_left)
                    self._reports_unavailable_texts.append(text)
                    with d.tooltip(text):d.add_text(row['reason'],wrap=TOKENS.report.wrap_right)
        d.configure_item('pyecr_reports_unavailable',label=f'Unavailable figures ({missing})',show=bool(missing))
        self._reports_content_changed()

    def _reports_content_changed(self,*args):
        self._reports_selection_epoch+=1
        d=self._require_dpg();all_=bool(d.get_value('pyecr_reports_all'))
        for tag,_ in self._reports_checks:d.configure_item(tag,enabled=not all_)
        count=sum(all_ or d.get_value(tag) for tag,_ in self._reports_checks)
        keys={key for tag,key in self._reports_checks if all_ or d.get_value(tag)}
        self._reports_preview_rows=tuple({**row,'selected':row['available'] and row['source_id']+'/'+row['figure_id'] in keys} for row in getattr(self,'_reports_preview_rows',()))
        d.set_value('pyecr_reports_count',f'{count} '+('figure' if count==1 else 'figures')+' selected · Scientific CSV / JSON')
        d.configure_item('pyecr_reports_count',show=bool(self._reports_sources))
        self._reports_error=''

    def _reports_path_mode_changed(self,*args):
        d=self._require_dpg();d.set_value('pyecr_report_path_mode',MANUAL_MODE if d.get_value('pyecr_reports_use_full_path') else PARENT_MODE)
        self._report_location_changed()

    def _reports_export(self,*args):
        d=self._require_dpg()
        try:
            if not self._reports_sources:raise ValueError('Select results and review content first.')
            if not all(g() for g in self._reports_guards):raise ValueError('Current result changed. Refresh results and review content.')
            target=self._resolve_report_destination()
            selected=None if d.get_value('pyecr_reports_all') else tuple(key for tag,key in self._reports_checks if d.get_value(tag))
            from pyecr.report.export_options import ExportOptions
            style = ExportOptions(legend_policy='auto' if d.get_value('pyecr_reports_legend_auto') else 'follow')
            sources,options=freeze_report(self._reports_sources,live_views=self._reports_live,views=self._plot_views.freeze(),
                layers=self._reports_layers(),map_views=self.controller.analysis_session.project_metadata.get('error_map_view',{}),
                decomposition_view=self.controller.decomposition_view_options.to_dict(),selection=selected,
                formats=tuple(fmt for fmt in ('png','svg','pdf') if d.get_value('pyecr_reports_'+fmt)),xlsx=bool(d.get_value('pyecr_reports_xlsx')),style=style)
            preview=figure_preview(sources,options.figure_ids)
            signature=lambda rows:tuple((r['figure_id'],r['title'],r['available']) for r in rows)
            if signature(preview)!=signature(self._reports_preview_rows):raise ValueError('Figure names or availability changed. Review content again.')
            task=self.controller.start_figure_export(sources,target,options=options,source_guards=self._reports_guards)
            self._reports_request=dict(task_id=task,sources=sources,options=options,preview=preview,target=target,
                selection_epoch=self._reports_selection_epoch,target_declaration=self._reports_target_declaration())
            self._reports_error=''
        except Exception as exc:self._reports_error=str(exc)

    def _reports_cancel(self,*args):
        if getattr(self.controller,'_figure_export_task_id',None):self.controller.cancel_figure_export()
        else:self.controller.cancel_page(GuiPage.REPORT)

    def _reports_verified_root(self):
        result=getattr(self.controller,'figure_export_result',None)
        if result is None or not result.root.is_dir():raise ValueError('Verified export folder is unavailable.')
        result.verify();return result.root

    def _reports_target_declaration(self):
        d=self._require_dpg();mode=d.get_value('pyecr_report_path_mode')
        return (mode,str(d.get_value('pyecr_report_destination'))) if mode==MANUAL_MODE else (mode,str(d.get_value('pyecr_report_parent')),str(d.get_value('pyecr_report_name')))

    def _reports_copy(self,*args):
        try:self._require_dpg().set_clipboard_text(str(self._reports_verified_root()))
        except Exception as exc:self._reports_error=str(exc)

    def _reports_open(self,*args):
        try:
            import os
            os.startfile(str(self._reports_verified_root()))
        except Exception as exc:self._reports_error=str(exc)

    def _refresh_reports_workflow(self):
        d=self._require_dpg();c=self.controller;active=bool(getattr(c,'_figure_export_task_id',None));legacy=c.state.tasks[GuiPage.REPORT].status=='running'
        from queue import Empty
        try:kind,value=self._report_picker_results.get_nowait()
        except Empty:pass
        else:
            if kind=='error':
                self._report_picker_running=False;d.set_value('pyecr_report_picker_notice','Directory chooser unavailable. Enter the parent directory.');self._reports_error=value
            elif value:self._report_parent_selected(app_data={'current_path':value})
            else:self._report_parent_cancelled()
        if not active and self._reports_sources_key()!=self._reports_result_key:
            versions=result_versions(c)
            preserve=bool(self._reports_sources and (getattr(self,'_reports_saved_history',False) or
                (all(g() for g in self._reports_guards) and all(versions.get(k)==v for k,v in getattr(self,'_reports_selected_versions',{}).items()))))
            self._reports_refresh_sources(preserve=preserve)
        for tag in ('pyecr_reports_all','pyecr_reports_png','pyecr_reports_svg','pyecr_reports_pdf','pyecr_reports_xlsx','pyecr_reports_legend_auto',
                    'pyecr_report_parent','pyecr_report_name','pyecr_report_destination','pyecr_report_browse_parent','pyecr_reports_use_full_path'):
            d.configure_item(tag,enabled=not active and not self._report_picker_running)
        for tag,_ in self._reports_source_tags:d.configure_item(tag,enabled=not active)
        for tag,_ in self._reports_checks:d.configure_item(tag,enabled=not active and not d.get_value('pyecr_reports_all'))
        try:target=self._resolve_report_destination();target_reason=''
        except (ValueError,OSError) as exc:target='';target_reason=str(exc)
        manual=d.get_value('pyecr_report_path_mode')==MANUAL_MODE
        result=getattr(c,'figure_export_result',None)
        if target_reason and result is not None and str(d.get_value('pyecr_report_destination'))==str(result.root):target_reason='Choose a new folder name to export again.'
        d.set_value('pyecr_reports_use_full_path',manual)
        d.configure_item('pyecr_report_parent_inputs',show=not manual);d.configure_item('pyecr_report_manual_inputs',show=manual)
        status=report_notice(self._reports_error) if self._reports_error else ''
        if active:
            token=c._cancellation_tokens.get(c._figure_export_task_id)
            status='Cancelling…' if token is not None and token.is_cancelled else f'Exporting · {100*getattr(c,"figure_export_progress",0.):.0f}%'
        elif legacy:status=f'Exporting Analysis package · {100*c.state.tasks[GuiPage.REPORT].progress:.0f}%'
        elif not status:
            phase=getattr(c,'figure_export_phase','')
            if phase in ('cancelled','error'):status='Export cancelled. Previous output retained.' if phase=='cancelled' else 'Export failed. Previous output retained; see Details.'
        d.set_value('pyecr_reports_status',status);d.configure_item('pyecr_reports_status',show=bool(status))
        # One actionable reason, not duplicated blank-state coaching.
        touched=bool(self._reports_sources or self._reports_error or any(str(d.get_value(t)).strip() for t in ('pyecr_report_parent','pyecr_report_name','pyecr_report_destination')))
        d.set_value('pyecr_reports_destination_reason',target_reason)
        d.configure_item('pyecr_reports_destination_reason',show=bool(target_reason and touched and not status))
        count=sum(r['selected'] for r in getattr(self,'_reports_preview_rows',()))
        formats=' / '.join(f.upper() for f in ('png','svg','pdf','xlsx') if d.get_value('pyecr_reports_'+f))
        n=len(self._reports_sources)
        summary=(f'{n} '+('result' if n==1 else 'results')+f' · {count} '+('figure' if count==1 else 'figures')+(' · '+formats if formats else '')) if self._reports_sources else ''
        d.set_value('pyecr_reports_export_summary',summary);d.configure_item('pyecr_reports_export_summary',show=bool(summary))
        valid=bool(self._reports_sources and target and all(g() for g in self._reports_guards))
        for tag in ('pyecr_reports_export','pyecr_reports_review','pyecr_reports_refresh'):
            allowed=valid if tag=='pyecr_reports_export' else any(d.get_value(t) for t,_ in self._reports_source_tags) if tag=='pyecr_reports_review' else True
            d.configure_item(tag,enabled=not active and not legacy and not self._report_picker_running and allowed)
        d.configure_item('pyecr_reports_cancel',enabled=active or legacy,show=active or legacy)
        if result is not None and result is not self._reports_last_result:
            self._reports_last_result=result;index=thaw((result.root/'figures.json').read_text(encoding='utf-8'))
            count=sum(r['selected'] for r in index['figures']);self._reports_receipt_text=f'Verified · {count} figures · {len(index["sources"])} results'
            request=self._reports_request
            self._reports_receipt_binding=(request['selection_epoch'],request['target_declaration']) if request is not None and Path(request['target'])==result.root else None
        receipt_current=bool(result is not None and self._reports_receipt_binding==(self._reports_selection_epoch,self._reports_target_declaration()) and not active and not status)
        if receipt_current:d.configure_item('pyecr_reports_destination_reason',show=False)
        d.set_value('pyecr_reports_receipt',getattr(self,'_reports_receipt_text','') if receipt_current else '')
        d.configure_item('pyecr_reports_receipt',show=receipt_current)
        for tag in ('pyecr_reports_open','pyecr_reports_copy'):d.configure_item(tag,show=receipt_current,enabled=receipt_current and result.root.is_dir())
        if d.get_value('pyecr_reports_details'):
            key=(id(result),target,status,self._reports_error,id(getattr(self,'_reports_preview_rows',None)))
            if key!=getattr(self,'_reports_details_key',None):
                self._reports_details_key=key;lines=['Destination: '+target]
                if result is not None:lines.extend(('Verified folder: '+str(result.root),freeze(result.manifest)))
                lines.append('Figure list and complete reasons: '+freeze(getattr(self,'_reports_preview_rows',())))
                lines.append('Original source warnings: '+freeze(getattr(self,'_reports_warnings',())))
                if self._reports_error:lines.append('Action failure: '+self._reports_error)
                lines.append(getattr(c,'figure_export_error_trace',''));d.set_value('pyecr_reports_details_text','\n\n'.join(lines))
        self._refresh_reports_geometry()
