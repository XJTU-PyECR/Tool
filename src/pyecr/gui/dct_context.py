"""Explicit DCT physical interpretation controls; viewing never computes."""
import json
from pyecr.gui.theme import TOKENS
from pyecr.contracts import canonical_sha256
from pyecr.services.dct_context_service import DctContextStore, review_parent, review_given, save_context
from pyecr.gui.components.dct_peak_transport import DctPeakTransportMixin


class DctContextLayoutMixin(DctPeakTransportMixin):
    def _build_physical_context(self):
        dpg=self._require_dpg();self._dct_context_store=self.controller.dct_context_store;self._dct_context_review=None
        self._dct_context_message='No physical context applied. Whole experimental DCT does not require a Fit or geometry.'
        with dpg.tab(label='Physics',tag='pyecr_dct_context_tab'):
            self._dct_review_heading('Optional physical interpretation')
            with dpg.group(horizontal=True,tag='pyecr_dct_context_columns'):
                with dpg.child_window(tag='pyecr_dct_context_source_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    self._dct_review_heading('01  Physical background')
                    dpg.add_combo(('None','Given (selected Simulation request)','Current fit','Parent component fit'),width=-1,
                        tag='pyecr_dct_context_source',default_value='None',callback=self._physical_context_draft)
                    with dpg.group(tag='pyecr_dct_context_given_form',show=False):
                        dpg.add_checkbox(label='I declare this background applicable',
                            tag='pyecr_dct_context_declared',callback=self._physical_context_draft)
                        dpg.add_text('Given basis')
                        dpg.add_input_text(tag='pyecr_dct_context_basis',width=-1,callback=self._physical_context_draft)
                    with dpg.collapsing_header(label='Mode approximation settings',default_open=False):
                        dpg.add_text('Modes per active axis (1–8). More modes retain more theoretical response.',wrap=TOKENS.dct.context_wrap)
                        dpg.add_input_int(tag='pyecr_dct_context_budget',default_value=4,
                            min_value=1,max_value=8,min_clamped=True,max_clamped=True,width=150,callback=self._physical_context_draft)
                with dpg.child_window(tag='pyecr_dct_context_action_panel',width=360,auto_resize_y=True,always_auto_resize=True,border=False,no_scrollbar=True):
                    self._dct_review_heading('02  Review and apply')
                    with dpg.group(horizontal=True):
                        dpg.add_button(label='Review background',tag='pyecr_dct_context_review',callback=self._review_physical_context)
                        dpg.add_button(label='Apply interpretation',tag='pyecr_dct_context_apply',callback=self._apply_physical_context,enabled=False)
                    with dpg.collapsing_header(label='Help / assumptions',default_open=False):
                        dpg.add_text('Compare time scales with a declared model; this optional step does not change the spectrum.',
                            tag='pyecr_dct_context_intro',wrap=TOKENS.dct.context_wrap)
                        dpg.add_text('',tag='pyecr_dct_context_source_hint',wrap=TOKENS.dct.context_wrap,color=TOKENS.steps.secondary)
            dpg.add_text('',tag='pyecr_dct_context_status',wrap=TOKENS.dct.context_wrap)
            dpg.add_separator()
            self._dct_review_heading('No physical background applied',tag='pyecr_dct_context_result_title')
            dpg.add_text('',tag='pyecr_dct_context_geometry',wrap=TOKENS.dct.context_wrap)
            dpg.add_group(tag='pyecr_dct_context_process_table')
            dpg.add_group(tag='pyecr_dct_context_axis_table')
            dpg.add_input_text(tag='pyecr_dct_context_summary',readonly=True,multiline=True,width=-1,height=TOKENS.dct.context_summary_height,show=False)
            dpg.add_text('',tag='pyecr_dct_context_mode_status',wrap=TOKENS.dct.context_wrap)
            self._build_peak_transport_guide()
            with dpg.collapsing_header(label='Theoretical mode comparison',default_open=False):
                dpg.add_text('Discrete masses are not DCT peak heights. The forward comparison shows truncation error, not an independent validation of the fitted model.',
                             tag='pyecr_dct_context_plot_notice',wrap=TOKENS.dct.context_wrap)
                with dpg.plot(label='Conditional mode times and retained mass',tag='pyecr_dct_context_modes_plot',width=-1,height=TOKENS.dct.context_plot_height):
                    dpg.add_plot_legend();dpg.add_plot_axis(dpg.mvXAxis,label='log10(tau / s)',tag='pyecr_dct_context_modes_x')
                    with dpg.plot_axis(dpg.mvYAxis,label='Discrete mass',tag='pyecr_dct_context_modes_y'):
                        for i in range(2):dpg.add_scatter_series([],[],tag=f'pyecr_dct_context_modes_{i}',label=f'Process {i+1}')
                with dpg.plot(label='Full model and retained modes',tag='pyecr_dct_context_forward_plot',width=-1,height=TOKENS.dct.context_plot_height):
                    dpg.add_plot_legend();dpg.add_plot_axis(dpg.mvXAxis,label='Time (s)',tag='pyecr_dct_context_forward_x')
                    with dpg.plot_axis(dpg.mvYAxis,label='Remaining fraction',tag='pyecr_dct_context_forward_y'):
                        for i in range(2):
                            dpg.add_line_series([],[],tag=f'pyecr_dct_context_full_{i}',label=f'Full {i+1}')
                            dpg.add_line_series([],[],tag=f'pyecr_dct_context_partial_{i}',label=f'Retained {i+1}')
            with dpg.collapsing_header(label='Full background details / history / save',default_open=False):
                dpg.add_text('All numbers are conditional on the declared geometry, boundary and parameters. 0.03/30 regime labels are heuristics, not gates. D/k is a half-length; 2D/k is the symmetric slab full-thickness Bi=1 reference.',wrap=TOKENS.dct.context_wrap)
                dpg.add_button(label='Copy background details',tag='pyecr_dct_context_copy',callback=self._copy_physical_context)
                dpg.add_input_text(tag='pyecr_dct_context_details',readonly=True,multiline=True,width=-1,height=TOKENS.dct.context_details_height)
                dpg.add_text('New context JSON path')
                dpg.add_input_text(tag='pyecr_dct_context_path',width=-1)
                dpg.add_button(label='Save applied context JSON',tag='pyecr_dct_context_save',callback=self._save_physical_context,enabled=False)

    def _physical_context_target(self):
        if self._dct_component_mode():
            job=self._selected_component_job()
            if job is None or job.bundle is None:raise ValueError('No stored component bundle')
            if job.status not in ('completed','warning') or job.stale_reasons:
                raise ValueError('Component parent is stale/unavailable; select a run bound to the current Fit and Diagnostics. Saved spectra remain viewable')
            job.bundle.component_capture.require_usable()
            return job.bundle.component_capture.parent_source,job.bundle.content_sha256,'components'
        if self._dct_signed_mode():
            job=self._selected_signed_job()
            if job is None or job.analysis is None or job.prepared is None:raise ValueError('No stored whole signed spectrum')
            return job.prepared.processed,job.analysis.content_sha256,'whole'
        result=self.controller.session.dct_result
        if result is None:raise ValueError('No stored monotonic spectrum. Run DCT in Whole / Monotonic, or select Signed to view its saved spectrum. Choosing a Fit alone does not create a DCT result.')
        entry=self.controller._legacy_dct_sources.get(id(result))
        if entry is None or entry[0] is not result:
            raise ValueError('Legacy spectrum has no frozen source identity. Spectrum remains readable; quantitative context Apply unavailable.')
        return entry[1],canonical_sha256(result.to_dict()),'monotonic'

    def _physical_context_draft_key(self):
        dpg=self._require_dpg();source,key,kind=self._physical_context_target()
        choice=dpg.get_value('pyecr_dct_context_source')
        run=self.controller.simulation.selected
        req=None if run is None else run.request
        # Current Fit identity is checked again by the service at Apply.
        return (source.content_sha256,key,kind,choice,dpg.get_value('pyecr_dct_context_declared'),
            dpg.get_value('pyecr_dct_context_basis'),dpg.get_value('pyecr_dct_context_budget'),
            None if req is None or not choice.startswith('Given') else req.content_sha256)

    def _physical_context_draft(self,*args):
        self._dct_context_review=None;self._dct_context_message='Draft changed. Review and Apply explicitly; stored spectrum is unchanged.'
        self._refresh_physical_context()

    def _review_physical_context(self,*args):
        dpg=self._require_dpg();self._dct_context_review=None
        try:
            source,key,kind=self._physical_context_target();choice=dpg.get_value('pyecr_dct_context_source')
            if choice=='None':
                self._dct_context_review=('clear',key);self._dct_context_message='Reviewed None. Apply context removes only this spectrum interpretation.'
            else:
                seq=len(self._dct_context_store.history)+1;budget=dpg.get_value('pyecr_dct_context_budget')
                if choice.startswith('Given'):
                    run=self.controller.simulation.selected
                    if run is None:raise ValueError('Explicitly select a saved Simulation request first')
                    record=review_given(run.request,source,key,kind,declared=dpg.get_value('pyecr_dct_context_declared'),
                        basis=dpg.get_value('pyecr_dct_context_basis'),per_axis=budget,sequence=seq)
                else:
                    if (kind=='components') != (choice=='Parent component fit'):
                        raise ValueError('Select Parent component fit for components; Current fit for whole observations')
                    if kind=='components':
                        eligibility=self._selected_component_job().bundle.component_capture.eligibility
                        if eligibility is not None and not eligibility.original_reliable:
                            raise ValueError('Physical interpretation requires an originally reliable parent Fit. Component DCT remains available for exploratory review.')
                    record=review_parent(self.controller.analysis_session,source,key,kind,per_axis=budget,sequence=seq,
                        transport_snapshot=getattr(self,'_transport_current',None))
                self._dct_context_review=record;self._dct_context_message='Reviewed candidate. Apply context explicitly; inversion is unchanged.'
            self._dct_context_review_key=self._physical_context_draft_key()
        except (ValueError,TypeError,RuntimeError) as exc:self._dct_context_message='Reason: '+str(exc)
        self._refresh_physical_context()

    def _apply_physical_context(self,*args):
        try:
            record=self._dct_context_review
            if record is None:raise ValueError('Review context first')
            if self._dct_context_review_key!=self._physical_context_draft_key():raise ValueError('Draft or target changed; Review context again')
            source,key,kind=self._physical_context_target()
            if isinstance(record,tuple):self._dct_context_store.clear(key)
            else:
                run=self.controller.simulation.selected
                self._dct_context_store.apply(record,source,key,kind,session=self.controller.analysis_session,
                    request=None if run is None else run.request)
                if kind=='monotonic':self.controller._legacy_dct_context_results[key]=(self.controller.session.dct_result,source)
            self._dct_context_review=None;self._dct_context_message='Applied interpretation only. Analysis, spectrum and jobs unchanged.'
        except (ValueError,TypeError,RuntimeError) as exc:
            self._dct_context_review=None;self._dct_context_message='Reason: '+str(exc)
        self._refresh_physical_context()

    def _physical_context_evidence(self):
        return self._require_dpg().get_value('pyecr_dct_context_details')

    def _copy_physical_context(self,*args):self._require_dpg().set_clipboard_text(self._physical_context_evidence())

    def _save_physical_context(self,*args):
        try:
            _,key,_=self._physical_context_target();record=self._dct_context_store.selected.get(key)
            save_context(record,self._require_dpg().get_value('pyecr_dct_context_path'))
            self._dct_context_message='Saved applied context JSON after staging readback.'
        except (ValueError,TypeError,OSError) as exc:self._dct_context_message='Reason: '+str(exc)
        self._refresh_physical_context()

    def _refresh_physical_context(self):
        from pyecr.gui.components.compact_panels import resize_pair,content_width
        d=self._require_dpg()
        resize_pair(d,'pyecr_dct_context_columns',
                    ('pyecr_dct_context_source_panel','pyecr_dct_context_action_panel'),minimum=370)
        for suffix in ('status','geometry','mode_status'):
            tag='pyecr_dct_context_'+suffix
            d.configure_item(tag,wrap=content_width(d,tag))
        dpg=self._require_dpg()
        if not dpg.does_item_exist('pyecr_dct_context_tab'):return
        component=self._dct_component_mode()
        if component!=getattr(self,'_dct_context_choices_component',None):
            self._dct_context_choices_component=component
            items=('None','Parent component fit') if component else ('None','Given (selected Simulation request)','Current fit')
            choice=dpg.get_value('pyecr_dct_context_source')
            if choice=='Current fit' and component:choice='Parent component fit'
            elif choice=='Parent component fit' and not component:choice='Current fit'
            elif choice not in items:choice='None'
            dpg.configure_item('pyecr_dct_context_source',items=items)
            dpg.set_value('pyecr_dct_context_source',choice)
        self._refresh_peak_transport_binding()
        dpg.configure_item('pyecr_dct_context_given_form',show=str(dpg.get_value('pyecr_dct_context_source')).startswith('Given'))
        dpg.set_value('pyecr_dct_context_source_hint',
            'Components use their captured parent Fit. The guide below remains available when inference Apply is blocked.'
            if self._dct_component_mode() else 'Current fit uses the matching Analysis result; Given uses the selected Simulation request. None clears interpretation after Review + Apply.')
        eligibility=None
        if self._dct_component_mode():
            job=self._selected_component_job()
            if job is not None and job.bundle is not None:eligibility=job.bundle.component_capture.eligibility
        blocked=eligibility is not None and not eligibility.original_reliable
        if blocked:
            dpg.set_value('pyecr_dct_context_source_hint','Exploratory component DCT is available. Physical interpretation requires the original reliable parent Fit; its reliability warnings currently prevent Apply.')
        dpg.configure_item('pyecr_dct_context_source_hint',color=TOKENS.colors.warning if blocked else TOKENS.steps.secondary)
        reason='';record=None;key=None
        try:
            source,key,kind=self._physical_context_target()
            record=self._dct_context_store.selected.get(key)
        except (ValueError,TypeError,RuntimeError) as exc:reason=str(exc)
        review=self._dct_context_review
        if review is not None:
            try:valid=self._dct_context_review_key==self._physical_context_draft_key()
            except (ValueError,RuntimeError):valid=False
            if not valid:self._dct_context_review=None;review=None
        displayed=review if review is not None and not isinstance(review,tuple) else record
        dpg.configure_item('pyecr_dct_context_apply',enabled=review is not None and not reason)
        dpg.configure_item('pyecr_dct_context_save',enabled=record is not None and not reason)
        dpg.configure_item('pyecr_dct_context_status',color=TOKENS.colors.warning if reason else TOKENS.colors.text)
        # Keep the blocking cause visible once; complete evidence stays below.
        status = ('Reason: '+reason.split('. ')[0]+'. Details below.' if reason else self._dct_context_message)
        if not reason and status.startswith('No physical context applied.'):
            status = 'Optional background; review before Apply.'
        dpg.set_value('pyecr_dct_context_status',status)
        lines=[reason, self._dct_context_message] if reason else ['No context. Valid whole spectra and reconstruction remain available.']
        display_state='unavailable' if reason else 'none';display_parent_reason=''
        if displayed is not None:
            s=displayed.transport
            lines=[('Review candidate' if displayed is review else 'Applied snapshot')+' | '+s.source_kind+' | '+s.model_id,
                   'Parent reliability: '+s.parent_reliability+' (original gate; distinct from calculable values)',
                   'Full sizes (cm): '+str(s.geometry.full_dims_cm)+' | Active: '+str(s.geometry.active_dimensions)]
            for p in s.processes:
                fmt=lambda v:'N/A' if v is None else f'{v:.8g}'
                lines.append(f'{p.label}: D={fmt(p.D_cm2_s)} cm^2/s; k={fmt(p.k_cm_s)} cm/s; D/k={fmt(p.half_length_cm)} cm; 2D/k={fmt(p.slab_full_thickness_cm)} cm')
            lines+= [f'{a.process} / {a.axis}: full L={a.full_dimension_cm:g} cm; Bi={a.value if a.value is not None else "N/A"}; {a.regime}' for a in s.axes]
            lines+=list(displayed.modal_reference['reasons'])
            if s.source is not None:
                session=self.controller.analysis_session
                binding=(s.source.content_sha256,id(session),session.current_revision,session.fit_generation,id(session.fit_result),id(session.diagnostics),session.fit_stale,session.diagnostics_stale)
                if getattr(self,'_dct_context_parent_binding',None)!=binding:
                    from pyecr.services.transport_summary import summarize_session
                    try:
                        now=summarize_session(session);s.source.assert_matches(now.source);why=''
                    except (ValueError,TypeError,RuntimeError) as exc:why=str(exc)
                    self._dct_context_parent_binding=binding;self._dct_context_parent_reason=why
                if self._dct_context_parent_reason:lines.insert(1,'History / Stale parent: '+self._dct_context_parent_reason)
                display_parent_reason=self._dct_context_parent_reason
                display_state='history_stale_parent' if display_parent_reason else 'current_bound_parent'
            else:display_state='given_conditional_background'
            caution=s.parent_reliability not in ('supported','not_applicable') or (s.source is not None and bool(self._dct_context_parent_reason))
            if caution:dpg.configure_item('pyecr_dct_context_status',color=TOKENS.colors.warning)
        dpg.set_value('pyecr_dct_context_summary','\n'.join(lines))
        self._refresh_physical_context_tables(displayed,review,reason,display_state,display_parent_reason)
        evidence=dict(interpretation_version='dct-physical-context/1',reason=reason,status=self._dct_context_message,
            applied_state='none' if record is None else 'frozen_applied_snapshot',display_parent_state=display_state,display_parent_reason=display_parent_reason,
            applied=None if record is None else record.to_dict(),review=None if displayed is not review or review is None else review.to_dict(),
            history=[r.to_dict() for r in self._dct_context_store.history],
            legacy_heuristic='possible_bi_values and reciprocals remain historical heuristics; never quantitative Bi.',
            limits='No peak assignment, automatic initial values, spectrum rerun, material deconvolution or new CI. Nearest-mode rows are conditional time-scale comparisons only.',
            peak_transport_guide=getattr(self,'_dct_peak_guide',None))
        dpg.set_value('pyecr_dct_context_details',json.dumps(evidence,ensure_ascii=False,allow_nan=False,indent=2))
        processes=() if displayed is None else displayed.modal_reference['processes']
        for plot in ('modes','forward'):dpg.configure_item('pyecr_dct_context_'+plot+'_plot',show=bool(processes))
        times=[] if displayed is None else list(displayed.target_source.time_s)
        for i in range(2):
            p=processes[i] if i<len(processes) else None
            for tag in ('modes','full','partial'):dpg.configure_item(f'pyecr_dct_context_{tag}_{i}',show=p is not None)
            if p is not None:
                dpg.configure_item(f'pyecr_dct_context_modes_{i}',label=p['label'])
                dpg.configure_item(f'pyecr_dct_context_full_{i}',label='Production '+p['label'])
                dpg.configure_item(f'pyecr_dct_context_partial_{i}',label='Truncated '+p['label'])
            dpg.set_value(f'pyecr_dct_context_modes_{i}',[[] if p is None else [m['log10_tau'] for m in p['modes']],[] if p is None else [m['mass'] for m in p['modes']]])
            for tag,field in (('full','production_remaining'),('partial','truncated_remaining')):
                dpg.set_value(f'pyecr_dct_context_{tag}_{i}',[[] if p is None else times,[] if p is None else list(p[field])])
        if displayed is not None:
            if not hasattr(self,'_dct_context_plotted'):self._dct_context_plotted={}
            for kind in ('modes','forward'):
                visible=dpg.get_item_state('pyecr_dct_context_'+kind+'_plot').get('visible',False)
                if visible and self._dct_context_plotted.get(kind)!=displayed.content_sha256:
                    for axis in ('x','y'):dpg.fit_axis_data('pyecr_dct_context_'+kind+'_'+axis)
                    self._dct_context_plotted[kind]=displayed.content_sha256

    def _refresh_physical_context_tables(self, displayed, review, reason, state, parent_reason):
        """Structured view of the already-created context; never computes modes."""
        d=self._require_dpg()
        key=(None if displayed is None else displayed.content_sha256,displayed is review,reason,state,parent_reason)
        if key==getattr(self,'_dct_context_table_key',None):return
        self._dct_context_table_key=key
        d.configure_item('pyecr_dct_context_geometry',show=displayed is not None)
        d.configure_item('pyecr_dct_context_mode_status',show=displayed is not None)
        from pyecr.gui.components.dct_review import number
        from pyecr.model_display import display_label
        if displayed is None:
            d.set_value('pyecr_dct_context_result_title','No physical background applied')
            d.set_value('pyecr_dct_context_geometry',reason or 'Choose a background and Review to see conditional transport values.')
            d.set_value('pyecr_dct_context_mode_status','Time-scale inversion and its reconstruction remain available independently.')
            processes=axes=()
        else:
            s=displayed.transport
            reliability={'supported':'Supported','unreliable':'Review required',
                         'not_applicable':'Not applicable'}.get(s.parent_reliability,s.parent_reliability.replace('_',' '))
            title='Review candidate · not applied' if displayed is review else 'Applied background'
            if parent_reason:title+=' · historical parent'
            d.set_value('pyecr_dct_context_result_title',title+' · '+display_label(s.model_id))
            d.set_value('pyecr_dct_context_geometry','Full dimensions (cm): '+ ' × '.join(f'{v:g}' for v in s.geometry.full_dims_cm)+
                        ' · Active: '+', '.join(a for a,v in zip('xyz',s.geometry.active_dimensions) if v)+
                        '\nParent reliability: '+reliability+('. '+parent_reason if parent_reason else '')+
                        (' · values below are conditional' if s.parent_reliability not in ('supported','not_applicable') else ''))
            processes=tuple((p.label,number(p.D_cm2_s),number(p.k_cm_s),number(p.half_length_cm),number(p.slab_full_thickness_cm)) for p in s.processes)
            axes=tuple((a.process,a.axis,number(a.value),a.regime.replace('_',' ')) for a in s.axes)
            d.set_value('pyecr_dct_context_mode_status','; '.join(displayed.modal_reference['reasons']) or
                        'Conditional mode comparison available below. Mode times and masses do not assign DCT peaks to H/O.')
        caution=displayed is not None and displayed.transport.parent_reliability not in ('supported','not_applicable')
        d.configure_item('pyecr_dct_context_result_title',color=TOKENS.colors.warning if reason or parent_reason or displayed is review or caution else TOKENS.colors.csv)
        self._dct_review_table('pyecr_dct_context_process_table',('Process','D (cm²/s)','k (cm/s)','D/k (cm)','2D/k (cm)'),processes)
        self._dct_review_table('pyecr_dct_context_axis_table',('Process','Axis','Bi','Regime'),axes)
