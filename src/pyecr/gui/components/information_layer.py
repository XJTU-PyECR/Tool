"""Independent top disclosure, local navigation and copyable page-owned status."""
from pyecr.gui.workflow_display import (
    InformationLevel, LEVEL_LABELS, WORKFLOW_STEPS, step_labels, step_key,
    flow_help, disclosure_visible, DETAIL_TAGS, EXPERT_TAGS, EMPTY_TEXT_TAGS,
    issue_lines, compact_fit_summary, text_has_content, PAGE_TASKS, page_owner,
    page_status, task_status_text, brief_text, recovery_hint,
)
from pyecr.gui.components.tags import WidgetTags
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS, status_color


class InformationLayerMixin:
    def _model_form_signature(self):
        """Only widget drafts; no config validation, factory or science calls."""
        dpg = self._require_dpg()
        values = []
        def visit(item):
            alias = dpg.get_item_alias(item)
            kind = dpg.get_item_info(item)['type']
            compatibility_free=alias.startswith('pyecr_model_parameter_') and alias.endswith('_free')
            if not compatibility_free and alias.startswith(('pyecr_model_', 'pyecr_boundary_', 'pyecr_fixed_d_')) and any(
                    word in kind for word in ('Input', 'Checkbox', 'Combo', 'Slider')) and not dpg.get_item_configuration(item).get('readonly', False):
                values.append((alias, repr(dpg.get_value(item))))
            for children in dpg.get_item_children(item).values():
                for child in children: visit(child)
        visit('pyecr_stage_model_content')
        if getattr(self,'_parameter_window_open',False):visit('pyecr_model_parameters_host')
        return tuple(sorted(values))

    def _remember_applied_model_form(self):
        self._applied_form_config = self.controller.analysis_session.current_analysis_config
        self._applied_form_signature = self._model_form_signature()

    def _build_information_bar(self):
        dpg = self._require_dpg()
        self._information_level = InformationLevel.DEFAULT
        self._top_details_visible = False
        self._page_context_texts = {}
        self._page_context_owner = None
        with dpg.group(horizontal=True, tag='pyecr_information_controls'):
            dpg.add_text('Information')
            dpg.add_combo(tuple(LEVEL_LABELS.values()), tag='pyecr_information_level',
                          default_value=LEVEL_LABELS[self._information_level], width=TOKENS.widths.input_medium, callback=self._set_information_level)
            dpg.add_checkbox(label='Top details', tag='pyecr_top_details_toggle', default_value=False,
                             callback=self._set_top_details_visible)
        with dpg.group(tag='pyecr_top_details', show=False):
            dpg.add_input_text(tag='pyecr_global_raw_source', readonly=True, default_value='Raw source: no imported data.', width=-1)
            dpg.add_text('', tag='pyecr_global_units', wrap=TOKENS.wraps.page)
            dpg.add_input_text(tag='pyecr_global_issues', readonly=True, multiline=True, default_value='',
                              height=TOKENS.heights.status_small, width=-1, show=False)
            dpg.add_button(label='Copy top details', tag='pyecr_copy_top_details', callback=self._copy_top_details)
        with dpg.group(horizontal=True, tag='pyecr_background_tasks', show=False):
            for page in GuiPage:
                dpg.add_button(label='', tag='pyecr_background_task_'+page.value.lower(), show=False,
                               callback=self._navigate_task_page, user_data=page.value)
        with dpg.window(label='Page status and context', tag='pyecr_page_context_window', show=False,
                        width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            with dpg.group(horizontal=True):
                dpg.add_button(label='Close', tag='pyecr_page_context_close', callback=self._close_page_context)
                dpg.add_button(label='Copy complete context', tag='pyecr_page_context_copy', callback=self._copy_page_context)
            dpg.add_input_text(tag='pyecr_page_context_text', readonly=True, multiline=True, width=-1, height=-1)
        with dpg.window(label='Six-step workflow', tag='pyecr_workflow_help_window', show=False, width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            dpg.add_button(label='Close', callback=lambda: dpg.configure_item('pyecr_workflow_help_window', show=False))
            dpg.add_input_text(tag='pyecr_workflow_help_text', default_value=flow_help(), multiline=True,
                               readonly=True, width=-1, height=-1)

    def _build_analysis_navigation(self):
        dpg = self._require_dpg()
        with dpg.group(horizontal=True, tag='pyecr_analysis_navigation'):
            dpg.add_text('Workflow')
            dpg.add_combo(step_labels(), tag='pyecr_workflow_step',
                          default_value=step_labels()[0], width=TOKENS.widths.field_wide, callback=self._navigate_workflow)
            dpg.add_button(label='Workflow help', callback=self._show_workflow_help, tag='pyecr_workflow_help_action')

    def _build_page_status(self, owner):
        dpg = self._require_dpg()
        with dpg.group(horizontal=True, tag='pyecr_page_status_row_'+owner):
            dpg.add_text('', tag='pyecr_page_status_'+owner, wrap=TOKENS.wraps.page)
            dpg.add_button(label='Status details / copy', tag='pyecr_page_context_'+owner,
                           callback=self._show_page_context, user_data=owner)

    def _set_top_details_visible(self, sender=None, app_data=None, user_data=None):
        dpg = self._require_dpg()
        self._top_details_visible = bool(dpg.get_value('pyecr_top_details_toggle') if app_data is None else app_data)
        dpg.set_value('pyecr_top_details_toggle', self._top_details_visible)
        dpg.configure_item('pyecr_top_details', show=self._top_details_visible)
        self._resize_analysis_columns()

    def _copy_top_details(self, *args):
        dpg = self._require_dpg()
        dpg.set_clipboard_text('\n\n'.join(dpg.get_value(tag) for tag in
                              ('pyecr_global_raw_source','pyecr_global_units','pyecr_global_issues')))

    def _show_page_context(self, sender=None, app_data=None, user_data=None):
        owner = user_data or page_owner(self.controller.state.active_page)
        self._page_context_owner = owner
        self._refresh_page_context_window()
        self._size_display_window('pyecr_page_context_window')
        self._require_dpg().configure_item('pyecr_page_context_window', show=True, label=owner.title()+' — status and context')

    def _close_page_context(self, *args):
        self._page_context_owner = None
        self._require_dpg().configure_item('pyecr_page_context_window', show=False)

    def _copy_page_context(self, *args):
        # Captured display text only; no Fit, diagnostics, capture or gate call.
        self._require_dpg().set_clipboard_text(self._page_context_texts.get(self._page_context_owner, ''))

    def _refresh_page_context_window(self):
        if self._page_context_owner is not None:
            self._require_dpg().set_value('pyecr_page_context_text', self._page_context_texts.get(self._page_context_owner,''))

    def _navigate_task_page(self, sender=None, app_data=None, user_data=None):
        page = GuiPage(user_data)
        owner = page_owner(page)
        if owner == 'analysis':
            self._navigate_workflow(None, None, {GuiPage.FIT:'fit', GuiPage.DIAGNOSTICS:'advanced', GuiPage.MODEL:'model'}.get(page,'data'))
        else:
            tag = WidgetTags.page(owner)
            self._require_dpg().set_value(WidgetTags.PAGES, tag)
            self._on_tab_change(None, tag, None)

    def _show_workflow_help(self, *args):
        self._require_dpg().configure_item('pyecr_workflow_help_window', show=True)

    def _restore_opened_model_display(self):
        """New AnalysisSession means an opened/new project, not a tier change or save."""
        session=self.controller.analysis_session
        if getattr(self,'_information_seen_session',None) is session:return
        self._information_seen_session=session
        config=session.current_analysis_config
        if config is None:return
        dpg=self._require_dpg()
        dpg.set_value('pyecr_model_id',config.model_id.value)
        self._render_model_controls(config.model_id.value)
        for axis,size,active in zip('xyz',config.model.geometry.full_dims_cm,config.model.geometry.active_dimensions):
            dpg.set_value('pyecr_model_dim_'+axis,size);dpg.set_value('pyecr_model_active_'+axis,active)
        model=config.model
        for tag,value in (('pyecr_model_surface_mode',model.surface_mode),('pyecr_model_step_type',model.step_type),
                          ('pyecr_model_boundary_mode',model.boundary_mode),('pyecr_model_relative_tolerance',config.solver.relative_tolerance),
                          ('pyecr_model_absolute_tolerance',config.solver.absolute_tolerance),('pyecr_model_max_terms',config.solver.max_terms),
                          ('pyecr_model_max_nfev',config.fit.max_nfev),('pyecr_model_loss',config.fit.loss)):
            if dpg.does_item_exist(tag):dpg.set_value(tag,value)
        from pyecr.gui.components.scientific_input import _model_parameter_tag
        descriptor=self.controller.model_descriptor(config.model_id)
        defaults=self.controller.default_model_form(config.model_id)
        for parameter in descriptor.parameters:
            name=parameter.name
            free=name in model.free_parameters if model.free_parameters else defaults.parameters[name].free and name not in model.fixed_parameters
            dpg.set_value(_model_parameter_tag(name,'free'),free)
            if dpg.does_item_exist(_model_parameter_tag(name,'fixed')):dpg.set_value(_model_parameter_tag(name,'fixed'),not free)
            value=model.fixed_parameters.get(name,model.initial_values.get(name))
            if value is not None:dpg.set_value(_model_parameter_tag(name,'value'),value)
            if name in model.parameter_bounds:
                lo,hi=model.parameter_bounds[name]
                dpg.set_value(_model_parameter_tag(name,'lower'),lo);dpg.set_value(_model_parameter_tag(name,'upper'),hi)
        for name,value in (('h',model.sign_H),('o',model.sign_O)):
            if value is not None:dpg.set_value('pyecr_model_sign_'+name,value)
        if config.boundary is not None:
            boundary=config.boundary
            if boundary.flush_time_s is not None:dpg.set_value('pyecr_boundary_flush_time_s',boundary.flush_time_s)
            if boundary.mode=='measured_history':
                for name in ('measured_time_s','measured_values'):
                    dpg.set_value('pyecr_boundary_'+name,' '.join(map(str,getattr(boundary,name))))
                for name,tag in (('measured_signal_type','measured_signal_type'),('measured_value_unit','measured_unit'),
                                 ('measured_initial_value','measured_initial'),('measured_final_value','measured_final'),
                                 ('resample_time_step_s','resample_step_s'),('max_internal_points','max_internal_points')):
                    value=getattr(boundary,name)
                    dpg.set_value('pyecr_boundary_'+tag,0. if value is None else value)
        self._select_model_step_type(None,model.step_type,None)
        if dpg.does_item_exist('pyecr_model_random_seed'):
            from pyecr.fitting.reproducibility import SECTIONS
            value=config.model_specific.get('fit_random_seed',config.model_specific.get(SECTIONS.get(config.model_id.value,''),{}).get('random_seed'))
            record={} if self.controller.analysis_session.fit_result is None else self.controller.analysis_session.fit_result.diagnostics.get('fitting_randomness',{})
            if record.get('seed_source')=='automatic_versioned_engine_default':value=None
            dpg.set_value('pyecr_model_random_seed','' if value is None else str(value))
        self._select_boundary_mode(None,model.boundary_mode,None)
        dpg.set_value('pyecr_model_apply_status','Applied configuration restored for review. Unspecified controls retain model defaults.')
        self._analysis_seen_config_revision=None;self._analysis_seen_import_revision=None

    def _set_information_level(self, sender=None, app_data=None, user_data=None):
        """No refresh_widgets/model rebuild/service call: unsaved input survives."""
        dpg = self._require_dpg()
        choice = user_data if user_data is not None else app_data if app_data is not None else dpg.get_value('pyecr_information_level')
        level = InformationLevel(choice) if isinstance(choice, (int, InformationLevel)) else next(k for k,v in LEVEL_LABELS.items() if v == choice)
        self._information_level = level
        dpg.set_value('pyecr_information_level', LEVEL_LABELS[level])
        self._refresh_information_layer()
        self._resize_analysis_columns()
        self._refresh_display_layout(force=True)
        self._refresh_information_layer()

    def _navigate_workflow(self, sender=None, app_data=None, user_data=None):
        """Navigation only; old page values and callbacks stay addressable."""
        dpg = self._require_dpg()
        choice = user_data if user_data is not None else app_data if app_data is not None else dpg.get_value('pyecr_workflow_step')
        labels = step_labels()
        key = WORKFLOW_STEPS[labels.index(choice)].key if choice in labels else step_key(choice)
        dpg.set_value('pyecr_workflow_step', labels[next(i for i,s in enumerate(WORKFLOW_STEPS) if s.key == key)])
        page = WidgetTags.page('reports' if key == 'export' else 'analysis')
        dpg.set_value(WidgetTags.PAGES, page)
        self._on_tab_change(WidgetTags.PAGES, page, None)
        for step in WORKFLOW_STEPS:
            dpg.set_value('pyecr_stage_'+step.key, step.key == key)
        if key != 'export':
            if str(choice).lower() == 'boundary': dpg.set_value('pyecr_stage_boundary', True)
            dpg.set_y_scroll('pyecr_analysis_left', 0)
            if key in ('model','fit') and getattr(self,'_r05_controls_ready',False):
                self._r05_navigation_pending=(key,dpg.get_frame_count()+2)
            plot = {'data':'raw', 'preprocess':'processed', 'fit':'fit', 'advanced':'components'}.get(key)
            if plot:
                tag='pyecr_analysis_'+plot+'_tab'
                if dpg.get_item_configuration(tag).get('show',False):dpg.set_value('pyecr_analysis_plot_tabs',tag)
        self._refresh_information_layer()

    def _select_analysis_step(self, sender=None, app_data=None, user_data=None):
        """An explicitly opened header changes viewing selection, not readiness."""
        if app_data:
            index = next(i for i, step in enumerate(WORKFLOW_STEPS) if step.key == user_data)
            self._require_dpg().set_value('pyecr_workflow_step', step_labels()[index])

    def _set_disclosed_item(self, tag, show):
        dpg = self._require_dpg()
        if not dpg.does_item_exist(tag): return
        dpg.configure_item(tag, show=show)
        # P62 labels live in display-only wrappers; keep conditional labels in sync.
        item=dpg.get_alias_id(tag) if hasattr(dpg,'get_alias_id') else tag
        fields=getattr(self,'_display_fields',{})
        if item in fields:dpg.configure_item(fields[item][0],show=show)
        owned=getattr(self,'_owned_form_fields',{})
        if item in owned:dpg.configure_item(owned[item][0],show=show)

    def _refresh_information_layer(self):
        dpg = self._require_dpg()
        if not dpg.does_item_exist('pyecr_information_level'):return
        level=getattr(self,'_information_level',InformationLevel.DEFAULT)
        opened = {step.key: bool(dpg.get_value('pyecr_stage_'+step.key)) for step in WORKFLOW_STEPS}
        previous = getattr(self, '_analysis_opened_headers', opened)
        newly_opened = [key for key,value in opened.items() if value and not previous.get(key,False)]
        if newly_opened: self._select_analysis_step(None, True, newly_opened[-1])
        self._analysis_opened_headers = opened
        state=self.controller.state;session=self.controller.analysis_session
        dataset=session.dataset
        self._set_disclosed_item('pyecr_analysis_empty_hint',dataset is None and self.controller.session.import_preview is None)
        source=None if session.preprocessing is None else session.preprocessing.source
        raw=source if source is not None else dataset
        if raw is not None:
            path=str(raw.metadata.get('source_path') or raw.metadata.get('dataset_name') or state.dataset_name or 'in-memory raw data')
            source_text=f'Raw source: {path} | {raw.raw_time.size} original rows'
            raw_units=f"raw time={raw.metadata.get('time_unit','unknown')}, signal={raw.metadata.get('signal_unit','unknown')}"
        else:
            preview=self.controller.session.import_preview
            source_text='Raw source: no imported data.' if preview is None else f'Raw preview: {preview.source.source_path} (not imported)'
            raw_units='raw units unconfirmed'
        config=session.current_analysis_config
        current_time=state.time_unit if dataset is None else dataset.metadata.get('time_unit',state.time_unit)
        current_signal=state.signal_unit if dataset is None else dataset.metadata.get('signal_unit',state.signal_unit)
        selected=str(dpg.get_value('pyecr_model_id'))
        from pyecr.gui.model_display import display_label,display_name,BY_ID
        if selected in BY_ID:dpg.set_value('pyecr_model_display',display_label(selected))
        model_text=f'Applied model: {display_name(config.model_id)}' if config is not None else 'No applied model configuration'
        if config is None or selected!=config.model_id.value:model_text+=f'; selected {display_name(selected)} (not applied)'
        if config is not None:
            if getattr(self, '_applied_form_config', None) is not config:
                self._remember_applied_model_form()
            if self._model_form_signature() != self._applied_form_signature:
                model_text += '; unapplied form edits — Run uses the applied configuration'
        dpg.set_value('pyecr_global_raw_source',source_text)
        dpg.set_value('pyecr_global_units',f"{raw_units} | current time={current_time}, signal={current_signal or 'unconfirmed'}, length={state.length_unit}, D={state.diffusion_unit}, k={state.surface_exchange_unit} | {model_text}")
        try:descriptor=self.controller.model_descriptor(selected)
        except (KeyError,ValueError):descriptor=None
        lines=list(issue_lines(state,session,descriptor,getattr(self,'_fit_quality_cache_report',None)))
        analysis_extra_lines = []
        if 'unapplied form edits' in model_text:
            analysis_extra_lines.append('Unapplied form edits: review step 3 and explicitly Apply; existing Run uses the applied configuration.')
        # Preserve feature-specific Unsupported/failed/stale warnings even when their expert tool is collapsed.
        critical_tags=('pyecr_guided_refit_eligibility','pyecr_diagnostics_experimental_m1_status',
                       'pyecr_diagnostics_error_map_warnings','pyecr_diagnostics_dual_map_warnings',
                       'pyecr_diagnostics_bootstrap_status','pyecr_diagnostics_profile_status','pyecr_reliability_bi_guidance')
        for tag in critical_tags:
            if not dpg.does_item_exist(tag):continue
            text=dpg.get_value(tag)
            if isinstance(text,str) and any(word in text.lower() for word in ('out of date','failed','cancelled','warning:')):
                analysis_extra_lines.append(text)
        if dataset is not None and (not current_signal or current_signal=='unknown'):analysis_extra_lines.append('Unit warning: confirm the signal unit before scientific interpretation.')
        lines.extend(analysis_extra_lines)
        from pyecr.model_display import display_model_text
        issues=display_model_text('\n'.join(dict.fromkeys(lines)))
        dpg.set_value('pyecr_global_issues',issues)
        cap=TOKENS.heights.status_short if level==InformationLevel.DEFAULT else TOKENS.heights.status_small
        dpg.configure_item('pyecr_global_issues',show=bool(issues),height=min(cap,max(TOKENS.heights.status_short,20*len(lines))))
        dpg.configure_item('pyecr_top_details', show=self._top_details_visible)
        self._refresh_page_statuses(descriptor, source_text, dpg.get_value('pyecr_global_units'), analysis_extra_lines)
        for page in ('analysis','batch','pair','dct','report','reports'):
            self._set_disclosed_item(WidgetTags.summary(page),False)
        if dpg.does_item_exist('pyecr_fit_summary'):
            dpg.set_value('pyecr_fit_summary',compact_fit_summary(state.fit_presentation,session.fit_result,session.fit_stale,config,state.diagnostics_presentation))
        for tag in DETAIL_TAGS | EXPERT_TAGS | EMPTY_TEXT_TAGS:
            if dpg.does_item_exist(tag):self._set_disclosed_item(tag,disclosure_visible(tag,level,dpg.get_value(tag)))
        for name in ('fit','diagnostics','batch','pair','dct','report'):
            page=next(p for p in GuiPage if p.value.lower()==name)
            task=state.tasks[page]
            # Short page status owns terminal state; legacy telemetry remains optional.
            self._set_disclosed_item(WidgetTags.task_status(name),level>=InformationLevel.DETAILS and task.status!='idle')
            if name!='fit':self._set_disclosed_item(WidgetTags.progress(name),task.status=='running')
        self._set_disclosed_item('pyecr_fit_ready_status',False)
        self._set_disclosed_item('pyecr_fit_disabled_reason',False)
        self._set_disclosed_item('pyecr_fit_blocking_reasons',False)
        self._set_disclosed_item('pyecr_fit_quality_warnings',False)
        self._set_disclosed_item('pyecr_error_current_group',False)  # complete errors live in page context
        self._set_disclosed_item('pyecr_warning_current_group',False)
        self._set_disclosed_item('pyecr_history_group',level>=InformationLevel.DETAILS and bool(state.notices or state.errors))
        self._set_disclosed_item('pyecr_cancel_fit',state.tasks[GuiPage.FIT].status=='running')
        self._set_disclosed_item('pyecr_cancel_diagnostics',state.tasks[GuiPage.DIAGNOSTICS].status=='running')
        self._set_disclosed_item('pyecr_fit_results_group',state.fit_presentation is not None)
        self._set_disclosed_item('pyecr_diagnostic_results_group',state.fit_presentation is not None or state.diagnostics_presentation is not None or level==InformationLevel.EXPERT)
        self._set_disclosed_item('pyecr_developer_traceback_header',level==InformationLevel.EXPERT and bool(state.errors))
        for row in (() if state.fit_presentation is None else state.fit_presentation.parameters):
            for prefix in ('pyecr_parameter_bound_','pyecr_parameter_reliable_'):
                alias=prefix+row.name
                if dpg.does_item_exist(alias):self._set_disclosed_item(alias,disclosure_visible(alias,level,dpg.get_value(alias)))
        self._refresh_owned_analysis_forms()
        self._refresh_preprocessing_layout()
        self._refresh_dct_layout()
        self._refresh_report_layout()
        self._refresh_step_panels()
        self._refresh_r05_control_layout()

    def _refresh_page_statuses(self, descriptor, source_text, units_text, analysis_extra_lines):
        dpg = self._require_dpg()
        state, session = self.controller.state, self.controller.analysis_session
        quality = getattr(self, '_fit_quality_cache_report', None)
        context_tags = {
            'analysis': ('pyecr_fit_blocking_reasons','pyecr_fit_quality_warnings','pyecr_fit_context_summary','pyecr_fit_usage_help','pyecr_fit_reliability','pyecr_diagnostics_reliability'),
            'batch': ('pyecr_batch_inputs','pyecr_batch_independent_reason','pyecr_batch_global_reason'),
            'pair': ('pyecr_pair_members','pyecr_pair_result','pyecr_design_preview_summary'),
            'dct': ('pyecr_dct_run_reason','pyecr_dct_mode_notice','pyecr_dct_result_warning','pyecr_dct_summary','pyecr_dct_details'),
            'reports': ('pyecr_report_snapshot_status','pyecr_report_disabled_reason','pyecr_report_pdf_status'),
        }
        for owner, pages in PAGE_TASKS.items():
            if owner == 'simulation':
                # Simulation owns given-parameter context, independent of the
                # current Analysis source, Fit gates and model aliases.
                continue
            owned_lines = list(issue_lines(state, session, descriptor, quality, pages=pages))
            if owner == 'analysis':
                owned_lines.extend(analysis_extra_lines)
                from pyecr.diagnostics.interpretation import diagnostic_summary, BOUND_LEGEND
                if session.fit_result is not None:
                    owned_lines.append(('Out of date / historical Fit' if session.fit_stale else 'Current captured Fit') + '\n' +
                                       diagnostic_summary(session.fit_result, None if session.diagnostics_stale else state.diagnostics_presentation))
                    owned_lines.append(BOUND_LEGEND)
            values = []
            for tag in context_tags[owner]:
                if dpg.does_item_exist(tag):
                    value = dpg.get_value(tag)
                    if isinstance(value, str) and value.strip():
                        values.append(value)
                        if value.startswith(('Unsupported', 'Out of date', 'Failed', 'Legacy fit')):
                            owned_lines.append(value)
            # Page prerequisites stay page-owned; unrelated Analysis gate text is absent.
            if owner == 'reports' and session.fit_stale:
                owned_lines.append('Out of date current Fit: current export is disabled; explicitly captured historical snapshots may be exported.')
            snapshot = self.controller.session.report_snapshot if owner == 'reports' else None
            if snapshot is not None:
                from pyecr.diagnostics.interpretation import diagnostic_summary, BOUND_LEGEND
                values.extend((diagnostic_summary(snapshot.fit_result, snapshot.diagnostics_result), BOUND_LEGEND))
                owned_lines.extend('Captured snapshot warning: '+warning for warning in snapshot.warnings)
                owned_lines.extend('Reliability warning (captured Diagnostics): '+str(reason)
                                   for reason in snapshot.diagnostics_result.get('gate_failures', ()))
                metadata = snapshot.dataset.metadata
                values.append(f"Captured snapshot source: {metadata.get('source_path',metadata.get('dataset_name','in-memory data'))} | "
                              f"{snapshot.dataset.raw_time.size} original rows | raw time={metadata.get('raw_time_unit',metadata.get('time_unit','unknown'))}, "
                              f"raw signal={metadata.get('raw_signal_unit',metadata.get('signal_unit','unknown'))} | "
                              f"processed time={metadata.get('time_unit','unknown')}, signal={metadata.get('signal_unit','unknown')} | "
                              f"model={snapshot.analysis_config.model_id.value} | dataset SHA256={snapshot.dataset_sha256}")
            if owner == 'dct' and session.dataset is not None and session.dataset.metadata.get('signal_unit',state.signal_unit) in {None,'','unknown'}:
                owned_lines.append('Unit warning: confirm the signal unit before interpretation.')
            text, severity = page_status(state, session, owner, owned_lines, quality)
            if owner == 'analysis' and text.startswith('Data · completed:'):
                # Statistics remain in the unabridged task record and File details.
                suffix = '\n'+text.split('\n', 1)[1] if '\n' in text else ''
                text = ('Data imported.' if session.dataset is not None else 'Import draft ready.')+suffix
            if owner == 'dct' and state.tasks[GuiPage.DCT].status == 'idle' and not self._dct_signed_mode():
                # Generic action refresh runs before the DCT layout's stricter
                # existing support check. Do not derive readiness from that
                # transient widget enabled value for unsupported modes.
                reason = self.controller.dct_unavailable_reason(
                    use_classical_bi=bool(dpg.get_value('pyecr_dct_classical_bi')))
                suffix='\n'+text.split('\n',1)[1] if '\n' in text else ''
                if reason:text='DCT unavailable: '+reason+suffix
                elif dpg.get_item_configuration('pyecr_run_dct')['enabled']:text='Ready for DCT.'+suffix
            if owner == 'reports' and state.tasks[GuiPage.REPORT].status == 'idle' and not session.fit_stale and self.controller.session.report_snapshot is not None:
                text = 'Snapshot captured; choose current or explicit historical export.' + ('\n'+text.split('\n',1)[1] if '\n' in text else '')
            # Context contains complete sources/units, owned errors, notices and all owned task records.
            details = [owner.title()+' — complete status and context', source_text, units_text,
                       *dict.fromkeys(owned_lines), recovery_hint(owner, owned_lines), *values]
            details.extend(task_status_text(page, state.tasks[page]) for page in pages if state.tasks[page].status != 'idle')
            details.extend(error.as_text() for page,error in state.current_errors.items() if page in pages)
            details.extend(notice.as_text() for notice in state.notices if notice.page in pages)
            from pyecr.model_display import display_model_text
            self._page_context_texts[owner] = display_model_text('\n\n'.join(dict.fromkeys(value for value in details if value)))
            dpg.set_value('pyecr_page_status_'+owner, text)
            dpg.configure_item('pyecr_page_status_'+owner, color=status_color(severity),
                               wrap=max(280, min(900, dpg.get_viewport_width()-280)))
            if owner == 'pair':
                dpg.set_value('pyecr_page_status_pair_design', text)
                dpg.configure_item('pyecr_page_status_pair_design', color=status_color(severity))
        active_owner = page_owner(state.active_page)
        any_running = False
        for page, task in state.tasks.items():
            show = task.status == 'running' and page_owner(page) != active_owner
            tag = 'pyecr_background_task_'+page.value.lower()
            dpg.configure_item(tag, show=show, label=f'Go to {page_owner(page).title()} · '+brief_text(task_status_text(page, task),75))
            any_running |= show
        dpg.configure_item('pyecr_background_tasks', show=any_running)
        self._refresh_page_context_window()
