"""Page-owned forms and one movable parameter table. Never apply or infer."""
import json
from pyecr.gui.theme import TOKENS
from pyecr.gui.layout import disclosure_columns
from pyecr.gui.components.tags import WidgetTags
from pyecr.gui.components.widgets import table_style_kwargs
from pyecr.gui.components.fit_control_layout import FitControlLayoutMixin

FIELD_TYPES=('mvInputText','mvInputDouble','mvInputFloat','mvInputInt','mvCombo','mvSliderInt')


class AnalysisFormsMixin(FitControlLayoutMixin):
    def _build_import_setup(self):
        from pyecr.gui.components.dependencies import (
            _HEADER_MODE_CHOICES,_TEXT_PARSE_MODE_CHOICES,_TIME_UNIT_CHOICES,_SIGNAL_TYPE_CHOICES,
            HeaderMode,DelimiterMode,_DATA_PREVIEW_CONTAINER_TAG)
        from pyecr.units import get_units
        from pyecr.gui.components.action_state import add_action_tooltip
        dpg=self._require_dpg()
        self._initialize_data_file_picker()
        with dpg.group(tag='pyecr_import_setup'):
            dpg.add_input_text(label='Import draft — selected file',tag='pyecr_data_source_path',readonly=True,width=-1)
            dpg.add_button(label='Select file',tag='pyecr_data_select_file',callback=self._show_data_file_dialog)
            dpg.add_text('',tag='pyecr_data_picker_notice',show=False,wrap=TOKENS.wraps.rail)
            with dpg.collapsing_header(label='Open by full path',default_open=False):
                dpg.add_input_text(label='Full CSV / TXT / XLSX path',tag='pyecr_data_path_draft',width=-1,
                                   on_enter=True,callback=self._preview_data_file_path)
                dpg.add_button(label='Preview',tag='pyecr_data_preview_path',callback=self._preview_data_file_path)
            dpg.add_combo(('No workbook sheets',),label='XLSX sheet',tag='pyecr_data_sheet',enabled=False,
                          width=-1,callback=self._select_data_sheet,show=False)
            with dpg.group(tag='pyecr_import_mapping'):
                dpg.add_text('Columns and scientific units — confirm before import')
                with dpg.group(tag='pyecr_import_column_fields'):
                    for role,label in (('time','Select time column'),('signal','Select signal column')):
                        dpg.add_combo((),label=label,tag='pyecr_data_'+role+'_column',enabled=False,width=-1,
                                      callback=self._select_data_mapping_column,user_data=role)
                with dpg.group(tag='pyecr_import_unit_fields'):
                    dpg.add_combo(_TIME_UNIT_CHOICES,label='Time unit',tag='pyecr_data_time_unit',default_value='s',width=-1)
                    dpg.add_combo(_SIGNAL_TYPE_CHOICES,label='Signal type',tag='pyecr_data_signal_type',default_value='arbitrary',
                                  width=-1,callback=self._select_signal_type)
                    dpg.add_combo(tuple(item.label for item in get_units('arbitrary')),label='Signal unit',tag='pyecr_data_signal_unit',
                                  default_value='arbitrary',width=-1,callback=self._select_signal_unit)
                    dpg.add_input_text(label='Custom unit',tag='pyecr_data_custom_unit',default_value='',width=-1,show=False)
                dpg.add_button(label='Import',tag='pyecr_data_import',callback=self._import_previewed_data,enabled=False)
                add_action_tooltip(dpg,'pyecr_data_import')
                dpg.add_text('Select a file and map two valid columns.',tag='pyecr_data_import_reason')
            with dpg.collapsing_header(label='File details', tag='pyecr_file_details', default_open=False):
                dpg.add_button(label='Copy complete file details', tag='pyecr_copy_file_details',
                               callback=self._copy_step_details, user_data='data')
                dpg.add_input_text(tag='pyecr_file_details_text', readonly=True, multiline=True,
                                   width=-1, height=TOKENS.steps.details_height)
                dpg.add_text('Select a CSV, TXT, or XLSX file to preview its columns.',tag='pyecr_data_preview_summary')
                dpg.add_group(tag='pyecr_file_parsing_host')
            with dpg.collapsing_header(label='Advanced parsing — AUTO defaults retained',tag='pyecr_import_advanced',default_open=False,
                                       parent='pyecr_file_parsing_host'):
                with dpg.group(tag='pyecr_import_parse_fields'):
                    dpg.add_combo(_HEADER_MODE_CHOICES,label='Header',tag='pyecr_data_header_mode',default_value=HeaderMode.AUTO.name,
                                  width=-1,callback=self._select_data_header_mode)
                    dpg.add_combo(_TEXT_PARSE_MODE_CHOICES,label='Text parsing mode',tag='pyecr_data_text_parse_mode',
                                  default_value=DelimiterMode.AUTO.name,width=-1,callback=self._select_data_text_parse_mode)
                    dpg.add_input_text(label='Custom delimiter',tag='pyecr_data_custom_delimiter',default_value='|',width=-1,
                                       show=False,callback=self._select_data_text_parse_mode)
                    dpg.add_input_int(label='Header row (1-based)',tag='pyecr_data_header_row_index',default_value=1,min_value=1,
                                      min_clamped=True,width=-1,enabled=False,callback=self._select_data_header_mode)
                for tag,text in (('pyecr_data_header_status','Header has not been detected yet.'),
                                 ('pyecr_data_text_parse_status','Text parser has not been detected yet.'),
                                 ('pyecr_data_time_suggestion','Time-column suggestion: unavailable until preview.')):
                    dpg.add_text(text,tag=tag)
            dpg.add_group(tag=_DATA_PREVIEW_CONTAINER_TAG)
        self._prepare_owned_form('pyecr_import_setup')
        for tag in ('pyecr_import_column_fields','pyecr_import_unit_fields','pyecr_import_parse_fields'):
            self._pair_owned_fields(tag)

    def _register_owned_root(self, tag):
        self._owned_layout_roots.add(tag)

    def _prepare_owned_form(self, root):
        """Build-time labels only; refresh/resize never wraps or recreates inputs."""
        dpg=self._require_dpg()
        self._register_owned_root(root)
        pending=list(dpg.get_item_children(root,1) or [])
        while pending:
            item=pending.pop(0)
            kind=dpg.get_item_info(item)['type'].split('::')[-1]
            if kind=='mvTable':continue
            if kind=='mvText':self._owned_form_texts.add(item)
            pending.extend(dpg.get_item_children(item,1) or [])
            if kind not in FIELD_TYPES:continue
            cfg=dpg.get_item_configuration(item);label=cfg.get('label','')
            if not label or item in self._owned_form_fields:continue
            parent=dpg.get_item_parent(item)
            wrapper=dpg.add_group(parent=parent,before=item)
            heading=dpg.add_text(label,parent=wrapper)
            dpg.move_item(item,parent=wrapper)
            dpg.configure_item(item,label='',width=-1)
            self._owned_form_fields[item]=(wrapper,heading,root)

    def _pair_owned_fields(self, tag):
        dpg=self._require_dpg()
        children=list(dpg.get_item_children(tag,1) or [])
        wrappers={info[0] for info in self._owned_form_fields.values()}
        fields=[child for child in children if child in wrappers]
        for index in range(0,len(fields),2):
            pair=fields[index:index+2]
            row=dpg.add_group(parent=tag,before=pair[0],horizontal=False)
            for child in pair:dpg.move_item(child,parent=row)
            self._owned_form_rows[row]=tuple(pair)

    def _build_parameter_table(self, descriptor, defaults):
        dpg=self._require_dpg()
        with dpg.group(tag='pyecr_model_parameters_host'):
            with dpg.group(horizontal=True):
                self._r05_heading('Parameters — draft')
                parent={'parent':'pyecr_model_action_bar'} if dpg.does_item_exist('pyecr_model_action_bar') else {}
                dpg.add_button(label='Expand table',tag='pyecr_parameter_table_expand',callback=self._open_parameter_table,**parent)
                dpg.add_button(label='Copy draft',tag='pyecr_parameter_copy_draft',callback=self._copy_parameter_draft,**parent)
            dpg.add_text('',tag='pyecr_parameter_count_preview',show=False)
            with dpg.table(tag='pyecr_model_parameters_table',width=-1,scrollX=True,scrollY=True,
                           height=TOKENS.steps.parameter_table_height,inner_width=TOKENS.steps.parameter_table_width,
                           policy=dpg.mvTable_SizingFixedFit,**table_style_kwargs()):
                for name,label,width in zip(('name','value','lower','upper','state','unit'),
                        ('Parameter','Initial','Lower','Upper','Fixed','Unit'),TOKENS.steps.parameter_columns):
                    dpg.add_table_column(label=label,tag='pyecr_parameter_column_'+name,width_fixed=True,init_width_or_weight=width)
                for parameter in descriptor.parameters:
                    control=defaults.parameters[parameter.name]
                    with dpg.table_row():
                        title=dpg.add_text(parameter.name,tag='pyecr_parameter_name_'+parameter.name)
                        with dpg.tooltip(parent=title):dpg.add_text(parameter.role+' | '+parameter.constraint)
                        for field,value in (('value',control.value),('lower',control.lower_bound),('upper',control.upper_bound)):
                            dpg.add_input_double(tag=WidgetTags.model_parameter(parameter.name,field),default_value=value,
                                                 width=-1,format='%.6e',step=0,step_fast=0)
                        dpg.add_checkbox(tag=WidgetTags.model_parameter(parameter.name,'fixed'),default_value=not control.free,
                                         callback=self._parameter_fixed_changed,user_data=parameter.name)
                        dataset=self.controller.analysis_session.dataset
                        unit=parameter.unit if parameter.unit!='signal' or dataset is None else str(dataset.metadata.get('signal_unit','signal'))
                        dpg.add_text(unit.replace('^2','²'),tag='pyecr_parameter_unit_'+parameter.name)
            # Compatibility tags carry current-model metadata, never a second
            # user-editable column. Apply reads the visible Fixed checkbox.
            with dpg.group(tag='pyecr_parameter_compatibility_state',show=False):
                for parameter in descriptor.parameters:
                    dpg.add_checkbox(tag=WidgetTags.model_parameter(parameter.name,'free'),default_value=defaults.parameters[parameter.name].free)
            self._parameter_names=tuple(p.name for p in descriptor.parameters)
            self._parameter_roles={p.name:p.role for p in descriptor.parameters}
            self._parameter_required_free={p.name for p in descriptor.parameters if
                (descriptor.model_id.value=='M7_SINGLE_K' and p.name=='k_cm_s') or
                (descriptor.model_id.value=='M8_DUAL_K' and p.name.startswith('k_'))}
            self._parameter_fixed_overrides={}
            self._parameter_table_wide=True
            self._parameter_home=dpg.get_item_parent('pyecr_model_parameters_host')

    def _parameter_fixed_changed(self,sender=None,app_data=None,user_data=None):
        dpg=self._require_dpg()
        if user_data not in self._parameter_names:return
        if user_data in self._parameter_required_free:
            dpg.set_value(WidgetTags.model_parameter(user_data,'fixed'),False)
        dpg.set_value(WidgetTags.model_parameter(user_data,'free'),not dpg.get_value(WidgetTags.model_parameter(user_data,'fixed')))
        self._refresh_widgets()

    def _parameter_is_free(self,name):
        dpg=self._require_dpg()
        if getattr(self,'_parameter_names',None) and dpg.does_item_exist(WidgetTags.model_parameter(name,'fixed')):
            return not bool(dpg.get_value(WidgetTags.model_parameter(name,'fixed')))
        return bool(dpg.get_value(WidgetTags.model_parameter(name,'free')))

    def _copy_parameter_draft(self,*args):
        dpg=self._require_dpg();s=self.controller.analysis_session
        descriptor=self.controller.model_descriptor(dpg.get_value('pyecr_model_id'))
        value=dict(version='model-table-draft/1',state='Draft; not an applied configuration or estimated parameters',
            model_id=descriptor.model_id.value,display=dpg.get_value('pyecr_model_display'),length_unit='cm',
            full_dims_cm=[dpg.get_value('pyecr_model_dim_'+axis) for axis in 'xyz'],
            active_dimensions=[dpg.get_value('pyecr_model_active_'+axis) for axis in 'xyz'],
            parameters={p.name:dict(initial=dpg.get_value(WidgetTags.model_parameter(p.name,'value')),
                lower=dpg.get_value(WidgetTags.model_parameter(p.name,'lower')),upper=dpg.get_value(WidgetTags.model_parameter(p.name,'upper')),
                fixed=not self._parameter_is_free(p.name),unit=dpg.get_value('pyecr_parameter_unit_'+p.name)) for p in descriptor.parameters},
            count_summary=dpg.get_value('pyecr_parameter_count_preview'),
            source_metadata=None if s.dataset is None else dict(s.dataset.metadata),
            applied_config=None if s.current_analysis_config is None else s.current_analysis_config.to_dict())
        from pyecr.report.snapshot import _json_safe
        dpg.set_clipboard_text(json.dumps(_json_safe(value),ensure_ascii=False,indent=2,allow_nan=False))

    def _build_parameter_window(self):
        dpg=self._require_dpg()
        self._parameter_window_open=False
        with dpg.window(label='03 Model — complete parameter draft',tag='pyecr_parameter_table_window',show=False,
                        width=TOKENS.analysis.parameter_window_width,height=TOKENS.analysis.parameter_window_height,
                        on_close=self._close_parameter_table):
            dpg.add_button(label='Return to workflow',callback=self._close_parameter_table)
            dpg.add_group(tag='pyecr_parameter_window_host')
        self._register_owned_root('pyecr_parameter_table_window')

    def _open_parameter_table(self,*args):
        dpg=self._require_dpg()
        if self._parameter_window_open:return
        self._parameter_window_open=True
        self._parameter_home=dpg.get_item_parent('pyecr_model_parameters_host')
        dpg.move_item('pyecr_model_parameters_host',parent='pyecr_parameter_window_host')
        spec=TOKENS.analysis
        dpg.configure_item('pyecr_parameter_table_window',
                           width=min(spec.parameter_window_width,dpg.get_viewport_width()-spec.form_gutter*2),
                           height=min(spec.parameter_window_height,dpg.get_viewport_height()-spec.field_min),
                           pos=(spec.margin,spec.form_gutter))
        dpg.configure_item('pyecr_parameter_table_window',show=True)
        self._refresh_owned_analysis_forms()

    def _build_geometry_table(self,descriptor,defaults):
        dpg=self._require_dpg()
        self._r05_heading('Geometry — full sizes')
        self._r05_note('Active means both opposing faces exposed. Enter full sizes, not half sizes.')
        with dpg.table(tag='pyecr_geometry_table',width=-1,policy=dpg.mvTable_SizingStretchProp,**table_style_kwargs()):
            for label,weight in (('Axis',.7),('Full size ('+descriptor.geometry_unit+')',2.6),('Active',1.8)):
                dpg.add_table_column(label=label,width_stretch=True,init_width_or_weight=weight)
            for index,axis in enumerate('xyz'):
                with dpg.table_row():
                    dpg.add_text(axis)
                    dpg.add_input_double(tag='pyecr_model_dim_'+axis,default_value=defaults.full_dims_cm[index],
                                         min_value=0.,min_clamped=False,format='%.6e',step=0,step_fast=0,width=-1)
                    dpg.add_checkbox(tag='pyecr_model_active_'+axis,default_value=defaults.active_dimensions[index])

    def _close_parameter_table(self,*args):
        dpg=self._require_dpg()
        if self._parameter_window_open and dpg.does_item_exist('pyecr_model_parameters_host'):
            dpg.move_item('pyecr_model_parameters_host',parent=self._parameter_home,before='pyecr_model_derived_header')
        self._parameter_window_open=False
        dpg.configure_item('pyecr_parameter_table_window',show=False)
        self._refresh_owned_analysis_forms()

    def _refresh_owned_analysis_forms(self):
        dpg=self._require_dpg()
        spec=TOKENS.analysis
        if not dpg.does_item_exist('pyecr_analysis_left'):return
        rail=dpg.get_item_configuration('pyecr_analysis_left').get('width',spec.left_compact)
        width=max(spec.field_min,int(rail)-spec.form_gutter-2*TOKENS.steps.padding)
        model_inner=dpg.get_item_state('pyecr_model_form_scroll').get('rect_size',(0,0))[0] if dpg.does_item_exist('pyecr_model_form_scroll') else 0
        preview=self.controller.session.import_preview
        if dpg.does_item_exist('pyecr_import_mapping'):
            dpg.configure_item('pyecr_import_mapping',show=preview is not None)
            dpg.configure_item('pyecr_data_sheet',show=preview is not None and str(preview.source.source_path).lower().endswith('.xlsx'))
        self._form_two_columns=disclosure_columns(width,getattr(self,'_form_two_columns',None))
        alive=set(dpg.get_all_items())
        self._owned_form_fields={item:info for item,info in self._owned_form_fields.items() if item in alive}
        self._owned_form_rows={row:fields for row,fields in self._owned_form_rows.items() if row in alive}
        self._owned_form_texts.intersection_update(alive)
        inner_widths={}
        for tag in ('pyecr_model_form_scroll','pyecr_fit_controls_scroll'):
            if dpg.does_item_exist(tag):
                inner=dpg.get_item_state(tag).get('rect_size',(0,0))[0]
                if inner:inner_widths[dpg.get_alias_id(tag)]=max(spec.field_min,int(inner)-2*TOKENS.steps.padding-TOKENS.steps.scrollbar_gutter)
        parent_widths={}
        def width_for_parent(parent):
            if not parent:return width
            if parent in parent_widths:return parent_widths[parent]
            if parent in inner_widths:result=min(width,inner_widths[parent])
            else:result=width_for_parent(dpg.get_item_parent(parent))
            parent_widths[parent]=result
            return result
        def local_width(item):
            parent=dpg.get_item_parent(item)
            return width_for_parent(parent)
        layout_cache=getattr(self,'_owned_form_layout_cache',{})
        for item in self._owned_form_texts:
            item_width=local_width(item)
            if layout_cache.get(('text',item))!=item_width:
                dpg.configure_item(item,wrap=item_width);layout_cache[('text',item)]=item_width
        paired={field:len(fields) for fields in self._owned_form_rows.values() for field in fields}
        for row,fields in self._owned_form_rows.items():
            if layout_cache.get(('row',row))!=self._form_two_columns:
                dpg.configure_item(row,horizontal=self._form_two_columns);layout_cache[('row',row)]=self._form_two_columns
        self._owned_field_widths={}
        for item,(wrapper,heading,root) in self._owned_form_fields.items():
            size=(width-spec.gap)//2 if self._form_two_columns and paired.get(wrapper,1)>1 else width
            size=min(size,local_width(item))
            if root=='pyecr_model_schema_container' and model_inner:
                size=min(size,max(spec.field_min,int(model_inner)-2*TOKENS.steps.padding-TOKENS.steps.scrollbar_gutter))
            if dpg.get_item_alias(item).startswith('pyecr_model_dim_'):size=max(spec.field_min,width-spec.slider_width)
            self._owned_field_widths[item]=size
            adopted=item in self._compact_fields.rows
            show=dpg.get_item_configuration(item).get('show',True)
            key=(size,show,adopted)
            if layout_cache.get(('field',item))!=key:
                dpg.configure_item(wrapper,width=0 if adopted else size,show=show)
                if not adopted:dpg.configure_item(heading,wrap=size)
                layout_cache[('field',item)]=key
        self._owned_form_layout_cache=layout_cache
        if not dpg.does_item_exist('pyecr_model_parameters_table'):return
        dpg.configure_item('pyecr_parameter_compatibility_state',show=False)
        model_height=dpg.get_item_state('pyecr_model_form_scroll').get('rect_size',(0,0))[1] if dpg.does_item_exist('pyecr_model_form_scroll') else 0
        table_height=TOKENS.steps.parameter_table_height
        if not self._parameter_window_open and model_height:
            table_height=max(TOKENS.steps.parameter_table_min_height,min(table_height,int(model_height)-TOKENS.steps.parameter_caption_reserve))
        dpg.configure_item('pyecr_model_parameters_table',height=table_height)
        total=kinetic=0
        fixed_d=dpg.get_value('pyecr_fixed_d_enabled') if dpg.does_item_exist('pyecr_fixed_d_enabled') else False
        for name in self._parameter_names:
            fixed_tag=WidgetTags.model_parameter(name,'fixed')
            fixed_on_apply=fixed_d and name.startswith('D_')
            if fixed_on_apply and name not in self._parameter_fixed_overrides:
                self._parameter_fixed_overrides[name]=bool(dpg.get_value(fixed_tag))
            if fixed_on_apply:dpg.set_value(fixed_tag,True)
            elif name in self._parameter_fixed_overrides:dpg.set_value(fixed_tag,self._parameter_fixed_overrides.pop(name))
            if name in self._parameter_required_free:dpg.set_value(fixed_tag,False)
            dpg.configure_item(fixed_tag,enabled=name not in self._parameter_required_free and not fixed_on_apply)
            free=not bool(dpg.get_value(fixed_tag))
            total+=free;kinetic+=free and self._parameter_roles[name]=='transport'
            dpg.configure_item(WidgetTags.model_parameter(name,'free'),show=False)
        dpg.set_value('pyecr_parameter_count_preview',f'Free: kinetic {kinetic}; total {total}. Draft until Apply.')
        if dpg.does_item_exist('pyecr_model_experiment_summary'):
            dpg.set_value('pyecr_model_experiment_summary','Conditions: '+' / '.join(str(dpg.get_value(tag)) for tag in
                          ('pyecr_model_surface_mode','pyecr_model_step_type','pyecr_model_boundary_mode')))
        if dpg.does_item_exist('pyecr_model_derived_values'):
            fit=self.controller.analysis_session.fit_result
            selected=dpg.get_value('pyecr_model_id')
            config=self.controller.analysis_session.current_analysis_config
            text='Derived values appear from a matching completed Fit. No calculation runs while changing layout.'
            if fit is not None and config is not None and selected==config.model_id.value:
                record=fit.diagnostics.get('surface_k',fit.diagnostics.get('dual_surface_k',{}))
                values=[f'{name} = {record[name]!r} {record.get("units",{}).get(name,"")}' for name in
                        ('g_cm_inv','lambda_s_inv','lambda_1_s_inv','lambda_2_s_inv','tau_surface_exchange_s','tau_1_surface_exchange_s','tau_2_surface_exchange_s') if name in record]
                if values:text=('Historical/out of date Fit\n' if self.controller.analysis_session.fit_stale else 'Captured applied Fit (not the unapplied form)\n')+'\n'.join(values)+'\nBi N/A without D; surface-exchange and diffusion-mode times differ.'
            dpg.set_value('pyecr_model_derived_values',text)
