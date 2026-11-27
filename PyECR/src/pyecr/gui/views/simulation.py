"""Independent Simulation page. Widgets edit drafts; Generate alone dispatches."""
import traceback
from pyecr.model_display import display_label
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS, status_color
from pyecr.gui.workflow_display import brief_text
from pyecr.gui.simulation import MATRIX, MAX_PENDING_RUNS, complete_evidence, request_signature
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.units import (SIGNAL_TYPES,get_units,get_default_unit,validate_pair,normalize_alias,unit_label,
    get_physical_units,physical_unit,parameter_dimension,PRESSURE_UNIT_CHOICES)
from pyecr.gui.views.simulation_exchange import SimulationExchangeViewMixin

P='pyecr_sim_'


from pyecr.gui.components.simulation_layout import (SimulationLayoutMixin, MODEL_LABELS, BOUNDARY_LABELS, SAMPLING_LABELS, PARAMETER_LABELS)
from pyecr.gui.components.widgets import table_style_kwargs


class SimulationViewMixin(SimulationLayoutMixin, SimulationExchangeViewMixin):
    def _build_simulation_area(self):
        self._build_simulation_layout()

    def _load_simulation_widgets(self):
        dpg=self._require_dpg();d=self.controller.simulation.draft;m=d.current;row=MATRIX[d.model_id]
        for key in ('signal_type','sampling','start','stop','count','first_positive','explicit_times','relative_tolerance','absolute_tolerance','max_terms','name','noise_sd','seed','sequence_number'):
            dpg.set_value(P+key,getattr(d,key))
        dpg.set_value(P+'length_unit',physical_unit('length',d.length_unit).label)
        dpg.set_value(P+'time_unit',physical_unit('time',d.time_unit).label)
        self._load_simulation_signal_units()
        for i,axis in enumerate('xyz'):
            dpg.set_value(P+'dim_'+axis,d.dims[i]);dpg.set_value(P+'active_'+axis,d.active[i])
        dpg.set_value(P+'model',MODEL_LABELS[d.model_id]);dpg.configure_item(P+'boundary',items=[BOUNDARY_LABELS[b] for b in row['boundary_modes']])
        dpg.set_value(P+'boundary',BOUNDARY_LABELS.get(m.boundary_mode,m.boundary_mode))
        desc=DEFAULT_MODEL_REGISTRY.get(d.model_id)
        dpg.configure_item(P+'step_type',items=list(desc.supported_step_types))
        for key in ('step_type','sign_H','sign_O','flush_time_s','history_time','history_value','gas_type','initial_gas','final_gas','gas_unit','internal_step','internal_budget'):
            dpg.set_value(P+key,getattr(m,key))
        for key in ('flush_time_unit','history_time_unit','internal_step_unit'):
            dpg.set_value(P+key,physical_unit('time',getattr(m,key)).label)
        dpg.configure_item(P+'gas_unit',items=list(PRESSURE_UNIT_CHOICES)+([] if m.gas_unit in PRESSURE_UNIT_CHOICES else [m.gas_unit]))
        dpg.delete_item(P+'parameters',children_only=True)
        with dpg.table(parent=P+'parameters',width=-1,policy=dpg.mvTable_SizingStretchProp,**table_style_kwargs()):
            dpg.add_table_column(label='Parameter',width_fixed=True,init_width_or_weight=TOKENS.simulation.parameter_name_width)
            dpg.add_table_column(label='Given')
            dpg.add_table_column(label='Unit',width_fixed=True,init_width_or_weight=TOKENS.simulation.unit_width)
            for p in desc.parameters:
                unit=d.signal_unit if p.unit=='signal' else p.unit
                with dpg.table_row():
                    dpg.add_text(PARAMETER_LABELS.get(p.name,p.name),tag=P+'parameter_label_'+p.name,
                        wrap=TOKENS.simulation.parameter_name_width)
                    dpg.add_input_text(tag=P+'parameter_'+p.name,default_value=m.parameters.get(p.name,''),width=-1,callback=self._simulation_edited)
                    dimension=parameter_dimension(p.unit)
                    if dimension:
                        dpg.add_combo(tuple(u.label for u in get_physical_units(dimension)),tag=P+'parameter_unit_'+p.name,
                            default_value=physical_unit(dimension,m.parameter_units.get(p.name,p.unit)).label,
                            width=-1,callback=self._simulation_edited)
                    else:dpg.add_text(unit,tag=P+'parameter_unit_'+p.name,wrap=TOKENS.simulation.unit_width)
        try:self._sim_noise_choice='Gaussian' if float(d.noise_sd)!=0 else 'None'
        except ValueError:self._sim_noise_choice='Gaussian'
        dpg.set_value(P+'model_notice',display_label(d.model_id)+'\n'+('H/O names are parent model roles only.' if 'HO' in d.model_id else 'Effective numbered modes; no H/O assignment.' if d.model_id=='M8_DUAL_K' else 'Single given process.'))
        unsupported=[b for b in ('exponential_flush','measured_history') if b not in row['boundary_modes']]
        dpg.set_value(P+'boundary_notice','Unsupported: '+', '.join(unsupported)+' — no registered gas composition for this model.' if unsupported else 'Existing production boundary composition; coverage and numerical warnings are retained.')
        self._simulation_visibility();self._sim_review_revision=None;self._sim_layout_size=None

    def _read_simulation_widgets(self):
        dpg=self._require_dpg();d=self.controller.simulation.draft;m=d.current
        before=repr(d)
        for key in ('signal_type','sampling','start','stop','count','first_positive','explicit_times','relative_tolerance','absolute_tolerance','max_terms','name','noise_sd','seed','sequence_number','signal_custom_unit'):
            setattr(d,key,str(dpg.get_value(P+key)))
        selected=str(dpg.get_value(P+'signal_unit'))
        if selected=='Custom…':d.signal_unit=d.signal_custom_unit
        else:
            try:same=normalize_alias(selected)==normalize_alias(d.signal_unit)
            except (ValueError,TypeError):same=False
            if not same:d.signal_unit=selected
        d.length_unit=physical_unit('length',str(dpg.get_value(P+'length_unit'))).key
        d.time_unit=physical_unit('time',str(dpg.get_value(P+'time_unit'))).key
        d.dims=tuple(str(dpg.get_value(P+'dim_'+a)) for a in 'xyz');d.active=tuple(bool(dpg.get_value(P+'active_'+a)) for a in 'xyz')
        boundary=str(dpg.get_value(P+'boundary'));m.boundary_mode=next((k for k,v in BOUNDARY_LABELS.items() if v==boundary),boundary)
        for key in ('step_type','sign_H','sign_O','flush_time_s','history_time','history_value','gas_type','initial_gas','final_gas','gas_unit','internal_step','internal_budget'):
            setattr(m,key,str(dpg.get_value(P+key)))
        m.parameters={p.name:str(dpg.get_value(P+'parameter_'+p.name)) for p in DEFAULT_MODEL_REGISTRY.get(d.model_id).parameters}
        for p in DEFAULT_MODEL_REGISTRY.get(d.model_id).parameters:
            if (dimension:=parameter_dimension(p.unit)):
                key=physical_unit(dimension,str(dpg.get_value(P+'parameter_unit_'+p.name))).key
                if key==p.unit:m.parameter_units.pop(p.name,None)
                else:m.parameter_units[p.name]=key
        for key in ('flush_time_unit','history_time_unit','internal_step_unit'):
            setattr(m,key,physical_unit('time',str(dpg.get_value(P+key))).key)
        if repr(d)!=before:d.edited()

    def _simulation_visibility(self):
        dpg=self._require_dpg();d=self.controller.simulation.draft;m=d.current
        for key,show in (('flush',m.boundary_mode=='exponential_flush'),('measured',m.boundary_mode=='measured_history'),
                ('custom_signs',m.step_type=='custom' and 'custom' in DEFAULT_MODEL_REGISTRY.get(d.model_id).supported_step_types),('sample_regular',d.sampling!='explicit'),('sample_explicit',d.sampling=='explicit')):
            dpg.configure_item(P+key,show=show)
        dpg.configure_item(P+'sample_start',show=d.sampling=='linear')
        dpg.configure_item(P+'sample_first_positive',show=d.sampling=='log_with_zero')
        dpg.set_value(P+'sampling_choice',SAMPLING_LABELS[d.sampling])
        for slot,num in (('H',1),('O',2)):
            dpg.set_value(P+'sign_'+slot+'_label',f'Process {num} sign' if d.model_id=='M8_DUAL_K' else slot+' role sign')
        dpg.configure_item(P+'start',enabled=d.sampling=='linear');dpg.configure_item(P+'first_positive',enabled=d.sampling=='log_with_zero')
        try:
            sd=float(d.noise_sd);noisy=sd>0
            if sd!=0:self._sim_noise_choice='Gaussian'
        except ValueError:noisy=False;self._sim_noise_choice='Gaussian'
        dpg.set_value(P+'noise_model',self._sim_noise_choice)
        dpg.configure_item(P+'noise_fields',show=self._sim_noise_choice=='Gaussian')
        dpg.configure_item(P+'noise_advanced',show=noisy)
        dpg.configure_item(P+'seed',enabled=noisy)
        dpg.configure_item(P+'sequence_number',enabled=noisy)
        dpg.configure_item(P+'signal_custom_unit',show=dpg.get_value(P+'signal_unit')=='Custom…')
        dpg.configure_item(P+'dimensions_column',label='Full L ('+d.length_unit+')')
        for key,unit in (('start',d.time_unit),('stop',d.time_unit),('first_positive',d.time_unit),('explicit_times',d.time_unit),
            ('flush_time_s',m.flush_time_unit),('history_time',m.history_time_unit),('internal_step',m.internal_step_unit)):
            label={'start':'Start','stop':'Stop','first_positive':'First positive time','explicit_times':'Times',
                   'flush_time_s':'Flush residence time','history_time':'Measured gas times','internal_step':'Time step'}[key]
            dpg.set_value(P+key+'_label',label+' ('+unit+')'+(', blank = Auto' if key=='internal_step' else ''))
        dpg.set_value(P+'noise_sd_label','Gaussian SD ('+d.signal_unit+')')

    def _simulation_sampling_changed(self,sender=None,app_data=None,user_data=None):
        mode=next(k for k,v in SAMPLING_LABELS.items() if v==app_data)
        self._require_dpg().set_value(P+'sampling',mode);self._simulation_edited()

    def _simulation_noise_changed(self,sender=None,app_data=None,user_data=None):
        d=self._require_dpg()
        if self._sim_noise_choice==str(app_data):self._simulation_edited();return
        self._sim_noise_choice=str(app_data)
        if app_data=='None':
            self._sim_gaussian_sd=str(d.get_value(P+'noise_sd'));d.set_value(P+'noise_sd','0')
        else:d.set_value(P+'noise_sd',self._sim_gaussian_sd)
        self._simulation_edited()

    def _simulation_edited(self,*args):
        self._read_simulation_widgets();self._simulation_visibility();self._refresh_simulation()

    def _simulation_boundary_changed(self,*args):self._simulation_edited()

    def _simulation_model_changed(self,sender=None,app_data=None,user_data=None):
        self._read_simulation_widgets()
        mid=next((k for k,v in MODEL_LABELS.items() if v==app_data),None)
        if mid is None:mid=next(k for k,v in MATRIX.items() if v['display_label']==app_data)
        self.controller.simulation.draft.select_model(mid);self._load_simulation_widgets();self._refresh_simulation()

    def _simulation_unit_changed(self,*args):
        self._read_simulation_widgets();d=self.controller.simulation.draft
        try:validate_pair(d.signal_type,d.signal_unit)
        except (ValueError,TypeError):
            d.signal_unit=get_default_unit(d.signal_type).label
            self._sim_unit_notice='Unit changed to '+d.signal_unit+'. Values are unchanged.'
            d.edited()
        self._load_simulation_widgets();self._refresh_simulation()

    def _load_simulation_signal_units(self):
        dpg=self._require_dpg();d=self.controller.simulation.draft
        types=list(SIGNAL_TYPES)
        if d.signal_type=='unknown':types.append('unknown')
        dpg.configure_item(P+'signal_type',items=types)
        options=get_units(d.signal_type);items=[u.label for u in options]
        try:
            key=validate_pair(d.signal_type,d.signal_unit)
            choice='Custom…' if key=='custom' else unit_label(d.signal_unit)
            if key=='custom' and d.signal_unit not in {'custom','Custom…'}:d.signal_custom_unit=d.signal_unit
        except (ValueError,TypeError):
            choice=d.signal_unit  # Do not relabel an invalid legacy declaration.
            if choice not in items:items.append(choice)
        dpg.configure_item(P+'signal_unit',items=items,enabled=d.signal_type!='normalized')
        dpg.set_value(P+'signal_unit',choice);dpg.set_value(P+'signal_custom_unit',d.signal_custom_unit)

    def _simulation_signal_unit_selected(self,*args):
        self._sim_unit_notice='Unit declaration changed. Values are unchanged.'
        self._simulation_edited()

    def _generate_simulation(self,*args):
        try:
            self._read_simulation_widgets();self.controller.run_simulation()
        except Exception as exc:
            self.controller.record_callback_exception(GuiPage.SIMULATION,exc,traceback.format_exc())
            self._sim_callback_error=str(exc)
        self._refresh_simulation()

    def _cancel_simulation(self,*args):self.controller.cancel_simulation();self._refresh_simulation()

    def _copy_analysis_simulation(self,*args):
        try:
            missing=self.controller.copy_analysis_to_simulation();self._sim_unit_notice='';self._load_simulation_widgets()
            self._sim_callback_error='Copied only applied initial/fixed values; enter missing: '+', '.join(missing) if missing else ''
        except Exception as exc:self._sim_callback_error=str(exc)
        self._refresh_simulation()

    def _simulation_history_selected(self,sender=None,app_data=None,user_data=None):
        key=getattr(self,'_sim_history_labels',{}).get(str(app_data),str(app_data).split(' · ',1)[0]);self.controller.simulation.select(key);self._refresh_simulation()

    def _show_simulation_details(self,*args):
        self._read_simulation_widgets();dpg=self._require_dpg();self._size_display_window(P+'details_window')
        dpg.configure_item(P+'details_window',show=True);self._refresh_simulation()

    def _copy_simulation_details(self,*args):
        self._require_dpg().set_clipboard_text(self._simulation_complete_details())

    def _simulation_complete_details(self):
        errors=[e.as_text() for e in self.controller.state.errors if e.page is GuiPage.SIMULATION]
        plan=getattr(self,'_sim_send_plan',None)
        review='' if plan is None else f'\nReviewed Send: {plan.channel}; {plan.target_action}; target SHA256={plan.target_sha256}\n'+plan.impact
        export_trace=getattr(self.controller,'simulation_export_error_trace','')
        return complete_evidence(self.controller.simulation)+'\n\nExport: '+self.controller.simulation_export_status+'\n'+getattr(self,'_sim_exchange_message','')+'\n'+getattr(self,'_sim_exchange_trace','')+review+('\n\nExport worker traceback:\n'+export_trace if export_trace else '')+('\n\nRetained Simulation callback errors:\n\n'+'\n\n'.join(errors) if errors else '')

    def _refresh_simulation(self):
        dpg=self._require_dpg()
        if not dpg.does_item_exist(P+'root'):return
        w=self.controller.simulation;d=w.draft;run=w.selected
        if self._sim_review_revision!=d.revision:
            try:self._sim_draft_request=d.build_request();self._sim_draft_signature=request_signature(self._sim_draft_request);self._sim_invalid=''
            except Exception as exc:self._sim_draft_request=None;self._sim_draft_signature=None;self._sim_invalid=str(exc)
            self._sim_review_revision=d.revision;self._sim_callback_error=''
        active=run is not None and run.job_id in w.active_ids
        message=run.message if run else ''
        compact=brief_text(message,250)
        if compact!=message:compact+=' — Full details contains every reason.'
        status=(f'{run.request.name} · {run.status}: '+compact if active else '')
        if w.active_ids:status+=(' | ' if status else '')+f'Pending {len(w.active_ids)}/{MAX_PENDING_RUNS}'
        dpg.set_value(P+'status',status);dpg.configure_item(P+'status',color=status_color(run.status if run else 'idle'))
        dpg.configure_item(P+'status',show=bool(status))
        reason=self._sim_invalid or getattr(self,'_sim_callback_error','')
        if run and self._sim_draft_request is not None:
            if run.draft_signature!=self._sim_draft_signature:
                reason+=('\n' if reason else '')+'Draft changed. Generate to update the saved curves.'
        dpg.set_value(P+'reason',brief_text(reason,190))
        dpg.configure_item(P+'reason',show=bool(reason))
        dpg.configure_item(P+'generate',enabled=len(w.active_ids)<MAX_PENDING_RUNS and not self._sim_invalid)
        dpg.configure_item(P+'cancel',enabled=active and run.status!='cancelling',show=active)
        dpg.configure_item(P+'copy_analysis',enabled=self.controller.analysis_session.current_analysis_config is not None)
        for p in DEFAULT_MODEL_REGISTRY.get(d.model_id).parameters:
            dpg.set_value(P+'parameter_label_'+p.name,PARAMETER_LABELS.get(p.name,p.name))
            dimension=parameter_dimension(p.unit)
            dpg.set_value(P+'parameter_unit_'+p.name,
                physical_unit(dimension,d.current.parameter_units.get(p.name,p.unit)).label if dimension else d.signal_unit)
        dpg.set_value(P+'noise_notice','SD unit: '+d.signal_unit)
        notice=getattr(self,'_sim_unit_notice','')
        dpg.set_value(P+'unit_notice',notice);dpg.configure_item(P+'unit_notice',show=bool(notice))
        self._sim_history_labels={f'Run {i} · {r.request.name} · {r.status}':r.job_id for i,r in enumerate(w.history,1)}
        choices=list(self._sim_history_labels)
        dpg.configure_item(P+'history',items=choices,show=bool(choices))
        if run:dpg.set_value(P+'history',next(x for x in choices if self._sim_history_labels[x]==run.job_id))
        render_key=(run.job_id,run.status,id(run.result),w.last_complete_id) if run else None
        if render_key!=self._sim_render_key:
            self._sim_render_key=render_key
            r=None if run is None else run.result
            ready=r is not None and r.state.ready
            times=list(r.time_s) if ready else []
            dpg.set_value(P+'total_series',[times,list(r.signal_clean) if ready else []])
            dpg.set_value(P+'observed_series',[times,list(r.signal_observed) if ready else []])
            dpg.set_value(P+'noise_series',[times,list(r.noise) if ready else []])
            for suffix in ('total_series','observed_series','noise_series','zero'):
                dpg.configure_item(P+suffix,show=ready and bool(times))
            parts=list(r.contributions.items()) if ready else []
            for i,slot in enumerate(('1','2')):
                key,values=parts[i] if i<len(parts) else ('No component',())
                dpg.set_value(P+'component_'+slot,[times,list(values)] if i<len(parts) else [[],[]])
                dpg.configure_item(P+'component_'+slot,label='Component '+slot,show=i<len(parts))
            dpg.set_value(P+'zero',[[times[0],times[-1]],[0.,0.]] if times else [[],[]])
            unit=r.request.signal_unit if r else 'signal'
            from pyecr.plotting.labels import signal_unit_axis_label
            for kind in ('total','components','noise'):
                label=signal_unit_axis_label(unit) if kind=='total' else ('Signed contribution' if kind=='components' else 'Noise')+' ('+unit+')'
                dpg.configure_item(P+kind+'_y',label=label)
            dpg.set_value(P+'stored_summary','Choose parameters, then Generate.' if run is None else f'{run.request.name} · {run.status.title()}')
            dpg.configure_item(P+'stored_summary',show=run is not None)
            lines=[] if run is None else [run.source_note,'Source: simulation_observed (simulated, not measured), simulation_clean, clean_model_contributions; given parameters, not estimates.',
                'Name: '+run.request.name,'Sampling: '+run.request.sampling_kind+'; complete grid stored in request.time_s; no display replacement.',
                f'Gaussian SD: {run.request.noise_sd:.17g} {run.request.signal_unit}; actual seed={run.request.seed}; RNG={run.request.rng_algorithm}; sequence={run.request.sequence_number}',
                'Model: '+display_label(run.request.model.model_id),
                'Request SHA256: '+run.request.content_sha256,'Full dimensions cm: '+str(run.request.model.geometry.full_dims_cm),
                'Active axes (both faces): '+str(run.request.model.geometry.active_dimensions),'Boundary: '+run.request.boundary.mode,
                'Parameter origin: '+run.request.parameter_origin,
                *[f'{p.name} = {run.request.given_parameters[p.name]:.17g} {run.request.signal_unit if p.unit=="signal" else p.unit} · given'
                    for p in DEFAULT_MODEL_REGISTRY.get(run.request.model.model_id).parameters],
                'Warnings / status:\n'+run.message,'No Fit, R² or CI; no automatic Apply to Analysis.']
            dpg.set_value(P+'result_parameters','\n'.join(lines) or 'No saved parameter list.')
            if run is not None and run.request.input_units is not None:
                inputs=run.request.input_units
                text='Input units: length='+inputs.length_unit+', time='+inputs.time_unit+'; '+inputs.conversion_version+'\n'+str(dict(inputs.parameter_units))
                dpg.set_value(P+'result_parameters','\n'.join(lines)+'\n'+text+'\nFull original input values and canonical request are retained in complete Details.')
        if dpg.get_item_configuration(P+'details_window')['show']:
            details_key=(d.revision,render_key,w.active_ids,None if run is None else (run.message,run.progress),len(self.controller.state.errors),self.controller.simulation_export_status,getattr(self,'_sim_exchange_message',''))
            if getattr(self,'_sim_details_key',None)!=details_key:
                self._sim_details_key=details_key
                dpg.set_value(P+'details_text',self._simulation_complete_details())
        # Small page context stays independent of Analysis source and fit gates;
        # the explicit Full details action copies the complete arrays.
        self._page_context_texts['simulation']=status+'\n\n'+d.source_note+'\n\n'+reason
        self._refresh_simulation_exchange()
        result=None if run is None else run.result
        ready=result is not None and result.state.ready
        for kind in ('total','components','noise'):dpg.configure_item(P+kind+'_plot',show=ready)
        warning='' if run is None or active or run.status=='completed' else brief_text(run.message,210)+' Details retain the full reason.'
        dpg.set_value(P+'result_warning',warning);dpg.configure_item(P+'result_warning',show=bool(warning))
        self._toggle_simulation_plots()
        self._resize_simulation()
        self._fit_simulation_axes(result)

    def _resize_simulation(self):
        dpg=self._require_dpg();t=TOKENS.simulation
        width=max(TOKENS.viewport.min_usable,int(dpg.get_viewport_client_width())-t.margin)
        y=dpg.get_item_rect_min(P+'columns')[1]
        height=max(TOKENS.heights.simulation_history,int(dpg.get_viewport_client_height())-int(y)-TOKENS.spacing.spacer_medium)
        left=t.left_narrow if width<t.compact_breakpoint else t.left_compact
        right=width-left-t.gap
        run_height=dpg.get_item_rect_size('pyecr_step_panel_sim_generate')[1] or t.run_fallback
        size=(width,height,left,y,run_height)
        if getattr(self,'_sim_layout_size',None)==size:return
        self._sim_layout_size=size
        for key,w in (('left',left),('results',right)):dpg.configure_item(P+key,width=w,height=height)
        dpg.configure_item(P+'inputs',height=max(TOKENS.heights.notes,height-run_height-TOKENS.spacing.spacer_medium))
        for kind in ('total','components','noise'):dpg.configure_item(P+kind+'_plot',height=max(t.plot_min,height-t.run_fallback),width=-1)
        for tag in ('status','reason'):dpg.configure_item(P+tag,wrap=left-t.gutter)
        dpg.configure_item(P+'title',wrap=width)
        def wrap_text(root,width):
            for children in dpg.get_item_children(root).values():
                for child in children:
                    if dpg.get_item_info(child)['type'].endswith('mvTable'):continue
                    if dpg.get_item_info(child)['type'].endswith('mvText'):dpg.configure_item(child,wrap=width)
                    wrap_text(child,width)
        wrap_text(P+'inputs',left-t.gutter);wrap_text(P+'results',right-t.gutter)
