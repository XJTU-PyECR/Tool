"""One explicit per-figure selector for independent stored module results."""
from pathlib import Path
from dataclasses import replace
from pyecr.gui.theme import TOKENS

class FigureExportViewMixin:
    def _build_figure_export_dialog(self):
        d=self._require_dpg();self._figure_sources=();self._figure_guards=();self._figure_checks=[];self._figure_live_views=()
        with d.window(label='Export stored result figures',tag='pyecr_figure_export_window',show=False,
            width=TOKENS.figures.width,height=TOKENS.figures.height):
            with d.child_window(tag='pyecr_figure_body',width=-1,height=-TOKENS.figures.footer,border=False):
                d.add_text('Select stored results. Export never runs Fit, DCT, Simulation or Error map.')
                d.add_button(label='Refresh result list',callback=self._figure_refresh_sources)
                with d.child_window(tag='pyecr_figure_sources',height=TOKENS.figures.sources_height,width=-1):pass
                d.add_button(label='Load figure list from selected results',tag='pyecr_figure_load',callback=self._figure_load_selection)
                with d.collapsing_header(label='Open saved figure source (explicit history)',default_open=False):
                    d.add_input_text(tag='pyecr_figure_source_path',hint='Full path to a source.json from a figure bundle',width=-1)
                    d.add_button(label='Load saved source',callback=self._figure_load_saved)
                d.add_checkbox(label='All available figures',tag='pyecr_figure_all',default_value=True,callback=self._figure_all_changed)
                d.add_checkbox(label='Auto-place legends (export only)',tag='pyecr_figure_legend_auto',default_value=False)
                with d.child_window(tag='pyecr_figure_list',height=TOKENS.figures.list_height,width=-1):pass
                with d.collapsing_header(label='Advanced formats',default_open=False):
                    with d.group(horizontal=True):
                        for fmt in ('png','svg','pdf'):d.add_checkbox(label=fmt.upper(),tag='pyecr_figure_'+fmt,default_value=fmt=='png')
                        d.add_checkbox(label='XLSX arrays (15 digits)',tag='pyecr_figure_xlsx')
                    d.add_text('JSON / CSV keep complete samples and float64 precision. PDF here contains figures; full-text report PDF is not configured.')
                d.add_text('Existing parent directory')
                d.add_input_text(tag='pyecr_figure_parent',width=-1)
                d.add_button(label='Choose parent directory',callback=lambda:d.show_item('pyecr_figure_parent_dialog'))
                d.add_text('New output folder name');d.add_input_text(tag='pyecr_figure_name',hint='Figures',width=-1)
            with d.group(horizontal=True):
                d.add_button(label='Export selected figures',tag='pyecr_figure_run',callback=self._figure_run)
                d.add_button(label='Cancel export',tag='pyecr_figure_cancel',callback=lambda:self.controller.cancel_figure_export())
                d.add_button(label='Close',callback=lambda:d.hide_item('pyecr_figure_export_window'))
            d.add_text('',tag='pyecr_figure_status',wrap=TOKENS.figures.wrap)
        with d.file_dialog(directory_selector=True,show=False,tag='pyecr_figure_parent_dialog',width=TOKENS.figures.width,height=TOKENS.figures.picker_height,
            callback=self._figure_parent_selected):pass
        self._owned_layout_roots.add('pyecr_figure_export_window')

    def _open_figure_export(self,*args):
        self._figure_refresh_sources();d=self._require_dpg();d.show_item('pyecr_figure_export_window')
        d.set_item_pos('pyecr_figure_export_window',(TOKENS.figures.margin,TOKENS.figures.margin))

    def _figure_refresh_sources(self,*args):
        d=self._require_dpg();d.delete_item('pyecr_figure_sources',children_only=True)
        self._figure_choices=self.controller.figure_export_choices();self._figure_source_tags=[]
        page=self.controller.state.active_page.value.lower()
        for i,choice in enumerate(self._figure_choices):
            tag='pyecr_figure_source_'+str(i);key,label,_,_=choice
            selected=(key=='analysis' and page not in ('batch','pair','dct','simulation')) or (page=='batch' and key in ('independent','global')) or (page=='pair' and key=='pair')
            if page=='simulation':selected=key=='simulation_'+str(self.controller.simulation.selected_id)
            # DCT history must be explicitly chosen. No first-history heuristic.
            d.add_checkbox(label=label,tag=tag,default_value=selected,parent='pyecr_figure_sources');self._figure_source_tags.append(tag)
        if not self._figure_choices:d.add_text('No completed stored result. Run a module explicitly first.',parent='pyecr_figure_sources')

    def _figure_load_selection(self,*args):
        d=self._require_dpg()
        try:
            selected=[c for tag,c in zip(self._figure_source_tags,self._figure_choices) if d.get_value(tag)]
            if not selected:raise ValueError('Select a stored result source first')
            if not all(c[3]() for c in selected):raise ValueError('Current result changed. Refresh the result list before selecting it.')
            from pyecr.report.figure_catalog import freeze,thaw
            layers=dict(version=1)
            for tag,key in (('pyecr_signed_dct_physical','dct_physical'),('pyecr_sensitivity_normalized','sensitivity_normalized'),
                ('pyecr_component_dct_scale','component_scale'),('pyecr_component_dct_view','component_view')):
                if d.does_item_exist(tag):layers[key]=d.get_value(tag)
            self.controller.analysis_session.project_metadata['figure_display_options']=layers
            captured=[]
            for choice in selected:
                source=choice[2]()
                layer=layers
                if choice[0]=='analysis_history':layer=thaw(source.payload)['result']['export_context'].get('project_metadata',{}).get('figure_display_options',{})
                captured.append(replace(source,display_options=freeze(layer)))
            sources=tuple(captured)
            if len({s.source_id for s in sources})!=len(sources):raise ValueError('The same result was selected twice; choose it once')
            self._figure_sources=sources;self._figure_guards=tuple(c[3] for c in selected)
            self._figure_live_views=tuple(c[0]!='analysis_history' for c in selected);self._figure_show_catalog()
        except Exception as exc:d.set_value('pyecr_figure_status','Selection rejected: '+str(exc))

    def _figure_load_saved(self,*args):
        d=self._require_dpg()
        try:
            from pyecr.report.figure_catalog import FigureSource,thaw
            path=Path(str(d.get_value('pyecr_figure_source_path')).strip())
            if not path.is_absolute():raise ValueError('Enter an absolute source.json path')
            source=replace(FigureSource.from_dict(thaw(path.read_text(encoding='utf-8'))),scope='historical')
            self._figure_sources=(source,);self._figure_guards=();self._figure_live_views=(False,);self._figure_show_catalog()
        except Exception as exc:d.set_value('pyecr_figure_status','History rejected: '+str(exc))

    def _figure_show_catalog(self):
        from pyecr.report.figure_catalog import catalog
        d=self._require_dpg();d.delete_item('pyecr_figure_list',children_only=True);self._figure_checks=[]
        for source in self._figure_sources:
            d.add_text((source.label or source.module)+' — '+source.scope,parent='pyecr_figure_list')
            for f in catalog(source):
                if f.available:
                    tag='pyecr_figure_item_'+str(len(self._figure_checks))
                    d.add_checkbox(label=f.title,tag=tag,default_value=True,enabled=not d.get_value('pyecr_figure_all'),parent='pyecr_figure_list')
                    self._figure_checks.append((tag,source.source_id+'/'+f.figure_id))
                else:d.add_text(f.title+': '+f.unavailable_reason,wrap=TOKENS.figures.wrap,parent='pyecr_figure_list')
        d.set_value('pyecr_figure_status',str(len(self._figure_checks))+' available figures. Selection is captured; Export freezes it.')

    def _figure_all_changed(self,*args):
        d=self._require_dpg()
        for tag,_ in self._figure_checks:d.configure_item(tag,enabled=not d.get_value('pyecr_figure_all'))

    def _figure_parent_selected(self,sender,app_data,user_data=None):
        from pyecr.gui.components.scientific_input import _directory_dialog_path
        self._require_dpg().set_value('pyecr_figure_parent',_directory_dialog_path(app_data))

    def _figure_run(self,*args):
        d=self._require_dpg()
        try:
            from pyecr.report.figure_bundle import FigureExportOptions
            from pyecr.report.export_options import ExportOptions
            from pyecr.gui.report_display import compose_target
            target=compose_target(str(d.get_value('pyecr_figure_parent')),str(d.get_value('pyecr_figure_name')))
            ids=None if d.get_value('pyecr_figure_all') else tuple(key for tag,key in self._figure_checks if d.get_value(tag))
            options=FigureExportOptions(ids,tuple(fmt for fmt in ('png','svg','pdf') if d.get_value('pyecr_figure_'+fmt)),bool(d.get_value('pyecr_figure_xlsx')),
                style=ExportOptions(legend_policy='auto' if d.get_value('pyecr_figure_legend_auto') else 'follow'))
            views=self._plot_views.freeze()
            sources=tuple(replace(source,view_state=views) if live else source
                for source,live in zip(self._figure_sources,self._figure_live_views,strict=True))
            self.controller.start_figure_export(sources,target,options=options,source_guards=self._figure_guards)
        except Exception as exc:d.set_value('pyecr_figure_status','Export rejected: '+str(exc))

    def _refresh_figure_export_widgets(self):
        d=self._require_dpg();active=bool(getattr(self.controller,'_figure_export_task_id',None))
        size=(d.get_viewport_client_width(),d.get_viewport_client_height())
        if size!=getattr(self,'_figure_window_size',None) and min(size)>0:
            self._figure_window_size=size
            d.configure_item('pyecr_figure_export_window',width=min(TOKENS.figures.width,size[0]-2*TOKENS.figures.margin),height=min(TOKENS.figures.height,size[1]-2*TOKENS.figures.margin))
        d.configure_item('pyecr_figure_cancel',enabled=active);d.configure_item('pyecr_figure_run',enabled=not active)
        d.configure_item('pyecr_figure_cancel_active',show=active)
        status=getattr(self.controller,'figure_export_status',None)
        if status and status!=getattr(self,'_figure_seen_status',None):
            d.set_value('pyecr_figure_status',status);self._figure_seen_status=status
