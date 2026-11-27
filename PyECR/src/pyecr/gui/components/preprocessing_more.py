"""Compact More tools, with explicit input conditions and draft-only helpers."""
from collections.abc import Mapping
from pyecr.gui.theme import TOKENS
from pyecr.gui.preprocessing_workflow import FIELD_DOMAINS,reviewable_mask_indices

TOOL_GROUPS=(('Units & signal',('convert_time_unit','resistance_to_conductance','conductance_to_conductivity')),
    ('Rows & sampling',('mask_points','unmask_points','outlier_mask','adaptive_downsample')),
    ('Trend & timing',('drift_correction','suggest_window','suggest_t0')))
PURPOSE={
    'convert_time_unit':'Convert the current processed time axis to the selected unit.',
    'resistance_to_conductance':'G = 1/R. Requires positive resistance values.',
    'conductance_to_conductivity':'Conductivity = G × L/A. G is converted to S; use electrical path geometry.',
    'mask_points':'Exclude original rows; raw observations are retained.',
    'unmask_points':'Restore explicit masks only. Crop and downsample exclusions remain.',
    'drift_correction':'Estimate drift on a stable reference, then correct only the selected interval.',
    'outlier_mask':'Detect candidate spikes, then review which original rows to exclude.',
    'adaptive_downsample':'Keep all early points and endpoints. Later rows are selected without averaging.',
    'suggest_window':'Suggest an Initial or Final platform. Review its bounds before Accept.',
    'suggest_t0':'Suggest onset time. Accept shifts time; pre-onset rows remain until explicitly excluded.',
}


class MorePreprocessingMixin:
    def _more_final_window(self,ds):
        if ds is None:return None
        w=ds.metadata.get('stability_windows',{}).get('final')
        if not isinstance(w,Mapping):return None
        key=(id(ds),id(w))
        if key!=getattr(self,'_more_window_key',None):
            from pyecr.gui.preprocessing_plots import window_is_current
            # A calibration outside the fit interval remains valid for scaling,
            # but cannot supply observed points for estimating a new drift.
            complete=all(ds.mask[int(i)] for i in w.get('raw_indices',()))
            self._more_window_key=key;self._more_window_valid=complete and window_is_current(ds,'final',w)
        if not self._more_window_valid:return None
        start,end=ds.stability_window_bounds('final',w)
        return dict(w,start_time=start,end_time=end,time_unit=ds.metadata['time_unit'])

    def _build_more_picker(self,tools):
        d=self._require_dpg();labels=dict(tools);self._more_rows=[]
        with d.group(tag='pyecr_more_tool_list'):
            for title,operations in TOOL_GROUPS:
                d.add_text(title,color=TOKENS.steps.secondary)
                for start in range(0,len(operations),2):
                    pair=operations[start:start+2]
                    with d.group(horizontal=True) as row:
                        for op in pair:
                            d.add_button(label=labels[op],tag='pyecr_direct_more_'+op,
                                callback=self._choose_more_operation,user_data=op)
                    self._more_rows.append((row,pair))
        with d.group(horizontal=True,tag='pyecr_more_heading'):
            d.add_button(label='‹ Tools',tag='pyecr_direct_more_back',callback=self._show_more_tools)
            d.add_text('',tag='pyecr_more_title')
        d.add_text('',tag='pyecr_more_purpose',wrap=TOKENS.wraps.stage)
        d.add_text('',tag='pyecr_more_condition',wrap=TOKENS.wraps.stage,color=TOKENS.colors.warning)
        with d.group(tag='pyecr_more_defaults',horizontal=True):
            d.add_button(label='Current range',tag='pyecr_more_current_range',callback=self._more_use_data,user_data='scope')
            d.add_button(label='Final reference',tag='pyecr_more_final_reference',callback=self._more_use_data,user_data='reference')
            d.add_button(label='Suggest sampling settings',tag='pyecr_more_sampling_defaults',callback=self._more_use_data,user_data='sampling')

    def _prepare_more_labels(self):
        d=self._require_dpg()
        for field,label in (('to_unit','Target time unit'),('path_length_cm','Path length (cm)'),
            ('cross_section_area_cm2','Cross-section (cm²)'),('raw_indices','Original rows (0-based)')):
            tag='pyecr_typed_'+field;d.configure_item(tag,label=label)
            with d.tooltip(parent=tag):d.add_text(FIELD_DOMAINS[field],wrap=320)
            for child in d.get_item_children('pyecr_typed_group_'+field,1) or ():
                if d.get_item_info(child)['type'].endswith('mvText') and not d.get_item_alias(child):
                    d.configure_item(child,show=False)
        d.configure_item('pyecr_data_stability_min_points',label='Minimum points')
        d.configure_item('pyecr_optional_review_indices',label='Keep these suggested rows for masking')
        d.configure_item('pyecr_optional_evidence',label='Candidate evidence',height=90)

    def _more_condition(self,op):
        s=self.controller.session;ds=s.dataset
        if ds is None:return 'Import data first.'
        kind=str(ds.metadata.get('signal_type','unknown')).lower()
        if op=='resistance_to_conductance' and kind not in ('resistance','unknown'):
            return 'Requires resistance input; current signal is '+kind+'.'
        if op=='conductance_to_conductivity' and kind not in ('conductance','unknown'):
            return 'Requires conductance input. Convert resistance to G first if needed.'
        if op=='unmask_points' and not reviewable_mask_indices(s.preprocessing.config):
            return 'No explicit masks to restore. Crop/downsample exclusions cannot be unmasked.'
        if op=='suggest_t0' and ds.metadata.get('time_unit','s')!='s':
            return 'Suggest t0 requires seconds. Use Time unit to convert to s first.'
        return ''

    def _more_use_data(self,sender=None,app_data=None,user_data=None):
        d=self._require_dpg();ds=self.controller.session.dataset
        if ds is None or self._direct_category!='more':return
        t=ds.processed_time
        if user_data=='scope' and self._direct_more_operation=='drift_correction':
            for field,value in (('scope_start',t[0]),('scope_end',t[-1]),('anchor_time',t[0])):
                d.set_value('pyecr_optional_field_'+field,format(float(value),'.17g'))
        elif user_data=='reference' and self._direct_more_operation=='drift_correction':
            w=self._more_final_window(ds)
            if w is None:return
            for field,edge in (('reference_start','start_time'),('reference_end','end_time')):
                d.set_value('pyecr_optional_field_'+field,format(float(w[edge]),'.17g'))
        elif user_data=='sampling' and self._direct_more_operation=='adaptive_downsample' and len(t)>3:
            for field,value in (('target_points',max(3,len(t)//2)),('early_until',float(t[max(0,len(t)//10-1)])),('late_power',2)):
                d.set_value('pyecr_optional_field_'+field,str(value))
        else:return
        self._direct_inputs_changed()

    def _refresh_more_layout(self,tools):
        d=self._require_dpg();op=self._direct_more_operation
        selected=self._direct_category=='more' and not self._direct_more_list_open
        ds=self.controller.session.dataset
        for tag in ('title','purpose','condition'):
            d.configure_item('pyecr_more_'+tag,show=selected)
        d.set_value('pyecr_more_title',dict(tools)[op])
        d.configure_item('pyecr_more_heading',show=selected)
        d.configure_item('pyecr_typed_current_unit',show=self._direct_category!='more')
        d.set_value('pyecr_more_purpose',PURPOSE[op])
        reason=self._more_condition(op)
        if op=='unmask_points' and not reason:
            indices=reviewable_mask_indices(self.controller.session.preprocessing.config)
            reason='Masked original rows: '+', '.join(map(str,indices[:12]))+(' …' if len(indices)>12 else '')
        d.set_value('pyecr_more_condition',reason)
        d.configure_item('pyecr_more_condition',show=selected and bool(reason))
        d.configure_item('pyecr_more_defaults',show=selected and op in ('drift_correction','adaptive_downsample'))
        for tag in ('current_range','final_reference'):d.configure_item('pyecr_more_'+tag,show=op=='drift_correction',enabled=ds is not None)
        if ds is not None:
            d.configure_item('pyecr_more_final_reference',enabled=self._more_final_window(ds) is not None)
        d.configure_item('pyecr_more_sampling_defaults',show=op=='adaptive_downsample',enabled=ds is not None and len(ds.processed_time)>3)
        width=d.get_item_state('pyecr_preprocess_forms').get('rect_size',(0,0))[0]
        width=max(160,int(width or 340)-2*TOKENS.steps.padding-TOKENS.steps.scrollbar_gutter)
        for row,pair in self._more_rows:
            inline=width>=300;d.configure_item(row,horizontal=inline,width=0)
            for key in pair:d.configure_item('pyecr_direct_more_'+key,width=(width-TOKENS.spacing.item_spacing[0])//2 if inline and len(pair)==2 else width)
        if self._direct_category=='more':
            d.configure_item('pyecr_data_apply_processing',show=not self._direct_more_list_open,
                enabled=selected and not bool(self._more_condition(op)),label='Find candidates' if op in ('outlier_mask','suggest_window','suggest_t0') else 'Preview')
        else:d.configure_item('pyecr_data_apply_processing',show=True,label='Preview')
        pending=getattr(self,'_more_scroll_pending',None)
        if pending is not None and d.get_frame_count()>=pending:
            rail=d.get_item_state('pyecr_analysis_left');panel=d.get_item_state('pyecr_step_panel_preprocess')
            if rail.get('rect_min') and panel.get('rect_min'):
                target=d.get_y_scroll('pyecr_analysis_left')+panel['rect_min'][1]-rail['rect_min'][1]-TOKENS.steps.padding
                d.set_y_scroll('pyecr_analysis_left',max(0,target))
            elif panel.get('pos'):
                d.set_y_scroll('pyecr_analysis_left',max(0,panel['pos'][1]))
            self._more_scroll_pending=None
