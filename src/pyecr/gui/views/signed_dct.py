"""Run whole-trace Signed DCT and display current or archived spectra."""
from dataclasses import replace
import traceback

from pyecr.contracts import SignedDctConfig, SignedDctDiagnosticConfig, SignedNormalizationSpec, SourceIdentity
from pyecr.gui.signed_dct import display_signed_job, complete_signed_evidence
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.preprocessing import PreprocessConfig
from pyecr.preprocessing.signed_normalization import cumulative_t0_s
from pyecr.services.dct_sources import prepare_signed_experiment
from pyecr.gui.components.action_state import set_action_availability


class SignedDctViewMixin:
    def _dct_signed_mode(self):
        dpg = self._require_dpg()
        return getattr(dpg, 'does_item_exist', lambda _: False)('pyecr_dct_analysis_mode') and dpg.get_value('pyecr_dct_analysis_mode') in {'Signed','Components'}

    def _build_signed_controls(self):
        dpg=self._require_dpg();self._dct_whole_representation='Monotonic'
        dpg.add_text('Source')
        dpg.add_radio_button(('Whole','Components'),tag='pyecr_dct_source_mode',default_value='Whole',horizontal=True,callback=self._select_dct_route)
        dpg.add_text('Representation')
        dpg.add_combo(('Monotonic','Signed'),tag='pyecr_dct_representation',default_value='Monotonic',width=-1,callback=self._select_dct_route)
        with dpg.group(show=False):
            dpg.add_combo(('Monotonic','Signed','Components'),tag='pyecr_dct_analysis_mode',default_value='Monotonic',callback=self._select_dct_analysis_mode)
            dpg.add_text('',tag='pyecr_dct_mode_recommendation')
        with dpg.group(tag='pyecr_signed_dct_controls',show=False):
            dpg.add_text('Terminal source')
            dpg.add_combo(('Accepted final window','External known terminal'),default_value='Accepted final window',tag='pyecr_signed_dct_reference',width=-1)
            dpg.add_text('',tag='pyecr_signed_dct_terminal_unit',wrap=TOKENS.wraps.rail)
            with dpg.group(tag='pyecr_signed_dct_external'):
                dpg.add_text('Terminal value',tag='pyecr_signed_dct_terminal_label',wrap=TOKENS.wraps.rail)
                dpg.add_input_double(tag='pyecr_signed_dct_terminal',default_value=0.,format='%.12g',width=-1)
                dpg.add_input_text(tag='pyecr_signed_dct_terminal_evidence',hint='Required terminal source / evidence',width=-1)
            dpg.add_button(label='Review DCT copy',tag='pyecr_signed_dct_review',callback=self._review_signed_dct)
            with dpg.collapsing_header(label='Normalization review Details',tag='pyecr_signed_dct_review_header',default_open=False):
                dpg.add_input_text(tag='pyecr_signed_dct_review_text',readonly=True,multiline=True,width=-1,height=TOKENS.dct.review_height,default_value='')
        self._build_component_controls()

    def _build_signed_numeric_controls(self):
        dpg=self._require_dpg()
        with dpg.group(tag='pyecr_signed_dct_numeric',show=False):
            dpg.add_text('GCV lambda candidates')
            dpg.add_input_text(tag='pyecr_signed_dct_lambdas',width=-1,default_value='1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1')
            dpg.add_text('Sensitivity checks are not confidence intervals; full settings and failures are in Details.',wrap=TOKENS.wraps.rail)
            with dpg.collapsing_header(label='Saved Signed DCT results (experimental)',default_open=False):
                dpg.add_text('Open a prior_archive.json file or its report folder. To calculate a new spectrum, select Run DCT.',wrap=TOKENS.wraps.rail)
                dpg.add_input_text(tag='pyecr_prior_archive_path',hint='Archive file or report folder',width=-1)
                dpg.add_button(label='Open saved result',callback=self._open_prior_evidence)
                dpg.add_checkbox(label='Show archived spectrum',tag='pyecr_prior_view',default_value=False,callback=lambda *a:self._refresh_widgets())
                dpg.add_input_text(tag='pyecr_prior_export_path',hint='New report directory',width=-1)
                dpg.add_button(label='Export archived report',callback=self._export_prior_evidence)
                dpg.add_text('',tag='pyecr_prior_message',wrap=TOKENS.wraps.rail)

    def _build_signed_mode_controls(self):
        d=self._require_dpg()
        with d.group(tag='pyecr_signed_mode_controls',show=False):
            d.add_text('Whole Signed mode')
            d.add_combo(('Legacy','Cached (Manual)','Balanced','Suppression'),default_value='Legacy',tag='pyecr_signed_linear_solver',width=-1,callback=self._select_signed_solver_mode)
            d.add_text('',tag='pyecr_signed_mode_hint',wrap=TOKENS.wraps.rail)

    def _build_convex_noise_controls(self):
        d=self._require_dpg()
        with d.group(tag='pyecr_convex_noise_controls',show=False):
            d.add_text('Known physical noise SD (original input unit)',tag='pyecr_convex_noise_label',wrap=TOKENS.wraps.rail)
            d.add_input_double(tag='pyecr_convex_noise_sd',default_value=0.,format='%.9g',width=-1)
            d.add_input_text(tag='pyecr_convex_noise_evidence',hint='Independent noise source / evidence',width=-1)
            with d.collapsing_header(label='Correlated noise (optional)',default_open=False):
                d.add_text('Known OU correlation time (s); 0 = independent',wrap=TOKENS.wraps.rail)
                d.add_input_double(tag='pyecr_convex_ou_seconds',default_value=0.,format='%.6g',width=-1)

    def _signed_solver_mode(self):
        d=self._require_dpg()
        return str(d.get_value('pyecr_signed_linear_solver') or 'Legacy') if not self._dct_component_mode() else 'Legacy'

    def _convex_options_from_widgets(self):
        from pyecr.contracts.convex_dct import ConvexDctOptions
        mode=self._signed_solver_mode()
        if mode not in ('Balanced','Suppression'):return None
        d=self._require_dpg();selection=str(d.get_value('pyecr_dct_lambda_mode'))
        if selection not in ('Manual','Known-noise auto'):raise ValueError('Balanced/Suppression support Manual or Known-noise auto; GCV is not valid for the absolute-mass penalty')
        return ConvexDctOptions(mode=mode,selection='manual' if selection=='Manual' else 'discrepancy',
            noise_sd_physical=float(d.get_value('pyecr_convex_noise_sd') or 0.),noise_evidence=str(d.get_value('pyecr_convex_noise_evidence') or ''),ou_seconds=float(d.get_value('pyecr_convex_ou_seconds') or 0.))

    def _configure_signed_mode_choices(self):
        d=self._require_dpg();whole=self._dct_signed_mode() and not self._dct_component_mode();mode=self._signed_solver_mode();convex=whole and mode in ('Balanced','Suppression')
        if d.does_item_exist('pyecr_signed_mode_controls'):d.configure_item('pyecr_signed_mode_controls',show=whole)
        choices=('Manual','Known-noise auto') if convex else ('GCV','Manual')
        d.configure_item('pyecr_dct_lambda_mode',items=choices)
        choice=d.get_value('pyecr_dct_lambda_mode')
        if choice not in choices:d.set_value('pyecr_dct_lambda_mode','Known-noise auto' if convex else 'GCV')
        if d.does_item_exist('pyecr_convex_noise_controls'):
            d.configure_item('pyecr_convex_noise_controls',show=convex)
            auto=d.get_value('pyecr_dct_lambda_mode')=='Known-noise auto'
            # The declaration remains editable for explicit Manual correlated-noise hypotheses.
            d.configure_item('pyecr_convex_noise_sd',enabled=auto)
            ds=self.controller.analysis_session.dataset
            if ds is not None:
                audit=ds.metadata.get('signed_dct_normalization',{})
                unit=audit.get('normalization',{}).get('spec',{}).get('signal_unit',ds.metadata.get('signal_unit','original input unit'))
                d.set_value('pyecr_convex_noise_label','Known physical noise SD ('+str(unit)+')')
        if d.does_item_exist('pyecr_signed_mode_hint'):
            hints={'Legacy':'Original Signed algorithm and GCV workflow.', 'Cached (Manual)':'Same curvature objective, reused solver; choose Manual lambda.',
                'Balanced':'Experimental: reduce spurious signed mass while retaining more negative contributions. Change is draft only.',
                'Suppression':'Experimental stronger suppression; may lose weak, broad or slow negative contributions. Change is draft only.'}
            d.set_value('pyecr_signed_mode_hint',hints.get(mode,''))

    def _select_signed_solver_mode(self,*args):
        d=self._require_dpg()
        if self._signed_solver_mode()=='Cached (Manual)':d.set_value('pyecr_dct_lambda_mode','Manual')
        self._configure_signed_mode_choices();self._refresh_widgets()

    def _select_dct_route(self,*args):
        d=self._require_dpg();source=d.get_value('pyecr_dct_source_mode')
        representation=d.get_value('pyecr_dct_representation')
        if source=='Components':
            if d.get_value('pyecr_dct_analysis_mode')!='Components':self._dct_whole_representation=representation
            selected='Components'
        else:
            selected=self._dct_whole_representation if d.get_value('pyecr_dct_analysis_mode')=='Components' else representation
            self._dct_whole_representation=selected
        d.set_value('pyecr_dct_analysis_mode',selected);self._select_dct_analysis_mode()

    def _use_signed_dct_profile(self, *args):
        """View-only recovery for an explicitly accepted signed source profile."""
        ds = self.controller.analysis_session.dataset
        if ds is None or ds.metadata.get('normalization_mode') != 'signed_terminal':
            return
        d = self._require_dpg()
        self._dct_whole_representation = 'Signed'
        d.set_value('pyecr_dct_analysis_mode', 'Signed')
        self._select_dct_analysis_mode()

    def _select_dct_analysis_mode(self, *args):
        dpg = self._require_dpg()
        self._configure_signed_mode_choices()
        signed = self._dct_signed_mode()
        component = self._dct_component_mode()
        if hasattr(self,'_dct_route_controls'):
            dpg.set_value('pyecr_dct_source_mode','Components' if component else 'Whole')
            dpg.configure_item('pyecr_dct_representation',items=('Signed',) if component else ('Monotonic','Signed'))
            dpg.set_value('pyecr_dct_representation','Signed' if signed else 'Monotonic')
            dpg.configure_item('pyecr_signed_dct_numeric',show=signed)
            dpg.configure_item('pyecr_component_dct_settings',show=component)
            dpg.configure_item('pyecr_component_dct_view_controls',show=component)
        dpg.configure_item('pyecr_component_dct_controls',show=component)
        dpg.configure_item('pyecr_component_dct_history',show=component)
        dpg.configure_item('pyecr_signed_dct_controls', show=signed)
        dpg.configure_item('pyecr_signed_dct_history', show=signed and not component)
        dpg.configure_item('pyecr_signed_dct_diagnostics_tab', show=not hasattr(self,'_dct_route_controls') or bool(dpg.get_value('pyecr_dct_more_plots')))
        dpg.configure_item('pyecr_signed_dct_legend', show=signed)
        dpg.configure_item('pyecr_signed_dct_display_notice', show=signed)
        dpg.configure_item('pyecr_dct_classical_bi', enabled=not signed)
        self._rendered_dct_presentation = None
        self._signed_dct_render_key = None
        self._component_render_key = None
        for name in ('component_1','component_2','sum','parent_fit','delta','overall','model_error'):
            dpg.set_value('pyecr_component_dct_'+name,[[],[]])
            dpg.configure_item('pyecr_component_dct_'+name,show=False)
        self._dct_fitted_axes.clear()
        self._dct_pending_axes.clear()
        self._dct_release_axes.clear()
        if not signed:
            dpg.configure_item('pyecr_dct_data_series',label='Data')
            dpg.configure_item('pyecr_dct_reconstruction_series',label='Reconstruction')
            for kind,label in (('spectrum','χ'),('reconstruction','Normalized response'),('residual','Observed − reconstructed')):
                dpg.configure_item('pyecr_dct_'+kind+'_y_axis',label=label)
        for tag in ('pyecr_dct_spectrum_series', 'pyecr_dct_data_series', 'pyecr_dct_reconstruction_series',
                    'pyecr_dct_residual_series', 'pyecr_signed_dct_positive', 'pyecr_signed_dct_negative', 'pyecr_signed_dct_zero'):
            dpg.set_value(tag, [[], []])
        for tag in ('positive', 'negative', 'zero'):
            dpg.configure_item('pyecr_signed_dct_'+tag, show=signed)
        dpg.configure_item('pyecr_dct_spectrum_series', show=not signed)
        self._refresh_widgets()

    def _signed_dct_draft(self):
        dpg = self._require_dpg()
        cfg = self._dct_config_from_widgets()
        cfg = replace(cfg, use_classical_bi=False)
        lambdas = tuple(float(v.strip()) for v in str(dpg.get_value('pyecr_signed_dct_lambdas')).split(',') if v.strip()) if cfg.lambda_mode == 'gcv' else ()
        config = SignedDctConfig(grid=cfg, gcv_lambdas=lambdas)
        if self._signed_solver_mode() in ('Balanced','Suppression'):
            config=replace(config,schema_version=2,algorithm_version='signed-tikhonov/2',solver='reused_svd')
        elif not self._dct_component_mode() and dpg.does_item_exist('pyecr_signed_linear_solver') and dpg.get_value('pyecr_signed_linear_solver') in ('Equivalent cached L2 Manual only','Cached (Manual)'):
            if cfg.lambda_mode!='manual':raise ValueError('Cached L2 UI option requires an explicit Manual lambda; automatic selection/defaults remain Legacy.')
            config=replace(config,schema_version=2,algorithm_version='signed-tikhonov/2',solver='projected_svd')
        session = self.controller.analysis_session
        ds = session.dataset
        if ds is None:
            raise ValueError('Import an experimental dataset first; Fit / Model configuration is optional.')
        recipe = session.preprocessing.config if session.preprocessing else PreprocessConfig()
        spec = None
        if ds.metadata.get('normalization_mode') != 'signed_terminal' and dpg.get_value('pyecr_signed_dct_reference') == 'External known terminal':
            import numpy as np
            binding = (id(ds), id(recipe), dpg.get_value('pyecr_signed_dct_terminal'), dpg.get_value('pyecr_signed_dct_terminal_evidence'))
            cached = getattr(self, '_signed_spec_cache', None)
            if cached and cached[0] == binding:
                return config, cached[1], ds, recipe
            spec = SignedNormalizationSpec(identity=SourceIdentity.capture(ds, recipe), raw_count=len(ds.raw_time),
                raw_indices=tuple(int(i) for i in np.flatnonzero(ds.mask)), signal_type=ds.metadata['signal_type'],
                signal_unit=ds.metadata['signal_unit'], reference_kind='external_known_terminal',
                reference_value=float(dpg.get_value('pyecr_signed_dct_terminal')),
                reference_evidence=str(dpg.get_value('pyecr_signed_dct_terminal_evidence')), t0_s=cumulative_t0_s(ds))
            self._signed_spec_cache = (binding, spec)
        return config, spec, ds, recipe

    def _review_signed_dct(self, *args):
        dpg = self._require_dpg()
        try:
            _, spec, ds, recipe = self._signed_dct_draft()
            if self._dct_component_mode():
                capture=self.controller.workflow.capture_component_sources()
                capture.require_usable()
            p = prepare_signed_experiment(ds, recipe, normalization_spec=spec,
                diagnostic_config=SignedDctDiagnosticConfig(lambda_factors=(), tau_ranges_s=(), grid_sizes=(), terminal_offsets_in_S=()))
            n = p.normalization
            text = (f'profile={n.spec.profile_id}\nσ∞={n.reference_value!r} {n.spec.signal_unit}; S={n.scale_value!r} {n.spec.signal_unit}\n'
                    f'terminal={n.spec.reference_kind}; rows={n.spec.reference_rows}\nmax raw index (0-based)={n.scale_raw_index}; valid={len(p.processed.signal)}/{p.processed.raw_count}\n'
                    f't0={p.processed.t0_s}s\n'+ '\n'.join((*n.state.blocking_reasons, *n.state.warnings)))
            self._signed_dct_review = (ds, recipe, spec, text)
            if self._dct_component_mode():
                text += '\nParent identity: '+str(capture.identity.to_dict())+'\nB='+str(capture.B_j)+'\n'+ '\n'.join(capture.state.warnings)
                if capture.eligibility is not None:
                    e=capture.eligibility
                    text+=f'\nEligibility: {e.policy_version}; can_run={e.can_run}; original reliable={e.original_reliable}; best_reliable={e.original_best_reliable}\nCaptured r1={e.residual_lag1}; '+(f'legacy limit={e.residual_limit}; ' if e.residual_limit is not None else 'v2 multi-lag/time-spacing evidence; ')+f'local C={e.gate_confidence_level}\n'+ '\n'.join(e.original_gate_failures)
            dpg.set_value('pyecr_signed_dct_review_text', text)
        except Exception as exc:
            self._signed_dct_review = None
            dpg.set_value('pyecr_signed_dct_review_text', f'Review rejected: {exc}')
        if hasattr(self,'_dct_route_controls'):dpg.set_value('pyecr_signed_dct_review_header',True)

    def _selected_signed_job(self):
        service = self.controller.signed_dct_service
        self.controller.current_signed_dct  # Synchronize owner inputs before reading history.
        key = getattr(self, '_signed_selected_id', None)
        return next((j for j in service.history if j.job_id == key), None)

    def _select_signed_history(self, sender, value, user_data):
        if self._require_dpg().does_item_exist('pyecr_prior_view'):self._require_dpg().set_value('pyecr_prior_view',False)
        self._signed_selected_id = getattr(self,'_signed_history_labels',{}).get(str(value),str(value).split(' | ')[0]) or None
        self._refresh_widgets()

    def _dct_history_choices(self,history):
        from pathlib import PureWindowsPath
        return {f'Run {i} · {job.status.title()} · '+PureWindowsPath(str(job.request.dataset.metadata.get('source_path') or job.request.dataset.metadata.get('dataset_name') or 'Captured source')).name+
                (' · '+job.request.convex_options.mode if getattr(job.request,'convex_options',None) is not None else ''):job.job_id
                for i,job in enumerate(history,1)}

    def _run_signed_dct_widgets(self):
        if self._dct_component_mode():
            self._run_component_dct_widgets();return
        if self._require_dpg().does_item_exist('pyecr_prior_view'):self._require_dpg().set_value('pyecr_prior_view',False)
        self._configure_signed_mode_choices()
        cfg, spec, _, _ = self._signed_dct_draft()
        self._signed_selected_id = self.controller.run_signed_dct(cfg, normalization_spec=spec,convex_options=self._convex_options_from_widgets())
        self._signed_dct_callback_error = None

    def _signed_complete_evidence(self):
        if self._dct_component_mode():return self._component_complete_evidence()
        if self._prior_evidence_active():
            import json
            return 'Archived experimental result (read-only). Run DCT uses the settings in the panel.\n'+json.dumps(self._prior_archive.to_dict(),ensure_ascii=False,indent=2,allow_nan=False)
        try:
            cfg, spec, _, _ = self._signed_dct_draft()
            draft = dict(config=cfg.to_dict(), normalization_spec=None if spec is None else spec.to_dict())
            options=self._convex_options_from_widgets()
            if options is not None:draft['convex_options']=options.to_dict()
        except Exception as exc:
            draft = dict(error=str(exc))
        return complete_signed_evidence(self._selected_signed_job(), draft)

    def _refresh_signed_dct_layout(self, source):
        if self._dct_component_mode():
            self._refresh_component_dct_layout(source);return
        dpg = self._require_dpg()
        self._configure_signed_mode_choices()
        dpg.configure_item('pyecr_signed_dct_physical',enabled=True,show=True)
        job = self._selected_signed_job()
        history = self.controller.signed_dct_service.history
        self._signed_history_labels=self._dct_history_choices(history)
        choices = list(self._signed_history_labels)
        dpg.configure_item('pyecr_signed_dct_history', items=choices,show=bool(choices))
        dpg.set_value('pyecr_signed_dct_history', next(v for v in choices if self._signed_history_labels[v]==job.job_id) if job else 'Select stored run' if choices else '')
        reason = ''
        try:
            cfg, spec, ds, recipe = self._signed_dct_draft()
            if ds.metadata.get('time_unit') != 's':
                reason = 'Convert processed time to seconds explicitly before Run.'
            elif ds.metadata.get('normalization_mode') != 'signed_terminal' and spec is None and not ds.metadata.get('stability_windows', {}).get('final'):
                reason = 'Accept a final window in preprocessing, or explicitly declare an external known terminal.'
            review = getattr(self, '_signed_dct_review', None)
            if review and (review[0] is not ds or review[1] != recipe or review[2] != spec):
                dpg.set_value('pyecr_signed_dct_review_text', 'Previous review is out of date. Click Review DCT copy explicitly.\n'+review[3])
            matches = bool(job and cfg == job.request.config and spec == job.request.normalization_spec
                           and ds is job.request.dataset and recipe == job.request.recipe
                           and self._convex_options_from_widgets()==job.request.convex_options)
            self._convex_options_from_widgets() # validate noise before enabling Run, including when no job exists
        except Exception as exc:
            reason, matches = str(exc), False
        running = any(j.status in {'queued', 'running'} for j in history)
        set_action_availability(dpg,'pyecr_run_dct',enabled=not reason and not running,
            reason=reason or 'A Signed DCT task is running. Wait or Cancel.')
        dpg.configure_item('pyecr_cancel_dct', enabled=bool(job and job.status in {'queued', 'running'} and not job.cancellation_requested))
        dpg.set_value('pyecr_dct_run_reason', reason or ('A signed job is running; cancel/wait before another explicit Run.' if running else 'Ready for explicit Run; this never modifies Analysis.'))
        dpg.set_value('pyecr_dct_mode_notice', 'Whole Signed DCT: Fit is optional.'+
            (' Draft differs from the saved run; saved curves remain unchanged.' if job and not matches else ''))
        data=self.controller.analysis_session.dataset
        if data is not None:
            from pathlib import Path
            source=Path(str(data.metadata.get('source_path','in-memory'))).name+'\n'+source
        dpg.set_value('pyecr_dct_source', source+'\nprofile='+str(data.metadata.get('normalization_mode', 'physical / unqualified')) if data is not None else source)
        mode = str(dpg.get_value('pyecr_dct_lambda_mode')).lower()
        applied=bool(data is not None and data.metadata.get('normalization_mode')=='signed_terminal')
        external=not applied and dpg.get_value('pyecr_signed_dct_reference')=='External known terminal'
        dpg.configure_item('pyecr_signed_dct_reference',enabled=not applied)
        dpg.configure_item('pyecr_signed_dct_terminal',enabled=external)
        dpg.configure_item('pyecr_signed_dct_terminal_evidence',enabled=external)
        dpg.configure_item('pyecr_signed_dct_lambdas',enabled=mode=='gcv')
        dpg.configure_item('pyecr_dct_lambda', show=mode == 'manual', enabled=mode == 'manual')
        dpg.configure_item('pyecr_dct_gcv_lambda', show=mode != 'manual')
        convex=self._signed_solver_mode() in ('Balanced','Suppression')
        dpg.set_value('pyecr_dct_lambda_reason', 'Known-noise auto uses discrepancy; Manual keeps the entered lambda. No nonlinear GCV or confidence interval.' if convex else 'Signed λ multiplies ||Lχ||²; editing is draft only. GCV is not a CI.')
        selected = job.analysis.selected_result if job and job.analysis else None
        dpg.set_value('pyecr_dct_gcv_lambda', f'{selected.selected_lambda:.5g} · saved' if selected else 'No selected stored λ')
        warnings = () if job is None else (*job.stale_reasons, *((job.failure,) if job.failure else ()), *job.warnings)
        flags=job.analysis.flags if job and job.analysis else ()
        text=(' / '.join(flags[:2])+'; ' if flags else '')+(f'{len(warnings)} reasons — Summary / Full details.' if warnings else '')
        if job and (job.failure or job.stale_reasons):
            text='\n'.join((*job.stale_reasons,*((job.failure,) if job.failure else ())))+'\n'+text
        callback_error=getattr(self,'_signed_dct_callback_error',None)
        if callback_error:
            text='Draft Run error: '+callback_error+'\n'+text
        elif reason:
            text='Draft not ready: '+reason+'\n'+text
        dpg.set_value('pyecr_dct_result_warning', text)
        dpg.configure_item('pyecr_dct_result_warning', show=bool(text))
        status = 'No signed job. Draft changes never run automatically.' if job is None else f'{job.job_id} · {job.status}\n'+(job.progress[-1].message if job.progress else '')
        if job and job.cancellation_requested:
            status+='\nCancellation requested; waiting for a solver checkpoint.' if job.status=='running' else '\nCancellation recorded; no result published.'
        dpg.set_value('pyecr_page_status_dct', status)
        if dpg.get_item_configuration('pyecr_dct_evidence_window')['show']:
            key = (job, tuple(str(dpg.get_value(t)) for t in ('pyecr_dct_tau_min','pyecr_dct_tau_max','pyecr_dct_n_tau','pyecr_dct_lambda_mode','pyecr_dct_lambda','pyecr_signed_dct_lambdas','pyecr_signed_dct_reference','pyecr_signed_dct_terminal','pyecr_signed_dct_terminal_evidence','pyecr_signed_linear_solver','pyecr_convex_noise_sd','pyecr_convex_noise_evidence','pyecr_convex_ou_seconds')))
            if getattr(self, '_signed_evidence_key', None) != key:
                dpg.set_value('pyecr_dct_evidence_text', self._signed_complete_evidence())
                self._signed_evidence_key = key
        if self._prior_evidence_active():
            dpg.set_value('pyecr_dct_mode_notice','Viewing an archived experimental result. Run DCT uses the settings in the panel.')
            dpg.set_value('pyecr_dct_source','Archived data source:\n'+str(self._prior_archive.report.source.identity.to_dict()))

    def _render_signed_dct(self):
        if self._dct_component_mode():
            self._render_component_dct();return
        dpg = self._require_dpg()
        job = self._selected_signed_job()
        physical = bool(dpg.get_value('pyecr_signed_dct_physical'))
        imported=self._prior_evidence_active()
        key = (job, physical,id(self._prior_archive) if imported else None)
        if getattr(self, '_signed_dct_render_key', None) == key:
            return
        if imported:
            from pyecr.gui.prior_evidence import prior_display,prior_scan
            view=prior_display(self._prior_archive,physical=physical)
            dpg.set_value('pyecr_dct_source','Archived data source:\n'+str(self._prior_archive.report.source.identity.to_dict()))
            dpg.set_value('pyecr_dct_mode_notice','Viewing an archived experimental result. Run DCT uses the settings in the panel.')
            dpg.configure_item('pyecr_dct_lambda_y_axis',label=prior_scan(self._prior_archive).score_name)
        else:
            view = display_signed_job(job, physical=physical)
            if job and job.request.convex_options is not None:
                dpg.configure_item('pyecr_dct_lambda_y_axis',label=job.analysis.score_name if job.analysis else 'Selection score')
            elif job and job.analysis and job.analysis.selected_result and job.analysis.selected_result.config.schema_version==2:
                r=job.analysis.selected_result
                view=replace(view,summary='Explicit equivalent fixed L2 solver | '+view.summary,
                    details=f'algorithm={r.config.algorithm_version}; solver={r.config.solver}; same L2 objective; conditional numeric equivalence, not physical reliability.\n'+view.details)
            if not job or job.request.convex_options is None:dpg.configure_item('pyecr_dct_lambda_y_axis',label='GCV')
        from pyecr.gui.signed_dct import SIGN_LEGEND
        dpg.set_value('pyecr_signed_dct_legend',SIGN_LEGEND)
        dpg.configure_item('pyecr_dct_data_series',label='Data')
        dpg.configure_item('pyecr_dct_reconstruction_series',label='Reconstruction')
        for tag, x, y in (('positive',view.x,tuple(max(v,0) for v in view.chi)), ('negative',view.x,tuple(min(v,0) for v in view.chi)),
                ('zero',view.x,tuple(0. for _ in view.x))):
            dpg.set_value('pyecr_signed_dct_'+tag, [list(x),list(y)])
        for tag, x, y in (('data',view.time,view.observed), ('reconstruction',view.time,view.reconstruction), ('residual',view.time,view.residual)):
            dpg.set_value('pyecr_dct_'+tag+'_series', [list(x),list(y)])
        dpg.set_value('pyecr_signed_dct_lambda_series', [list(view.lambda_x),list(view.lambda_score)])
        dpg.set_value('pyecr_signed_dct_lambda_points', [list(view.lambda_x),list(view.lambda_score)])
        dpg.set_value('pyecr_signed_dct_stability_series', [list(view.peak_x),list(view.peak_shift)])
        dpg.set_value('pyecr_dct_summary', view.summary)
        dpg.set_value('pyecr_dct_details', view.details)
        dpg.configure_item('pyecr_dct_spectrum_y_axis', label=f'χ ({view.unit}/decade)')
        from pyecr.plotting.labels import signal_unit_axis_label
        dpg.configure_item('pyecr_dct_reconstruction_y_axis', label=signal_unit_axis_label(view.unit))
        dpg.configure_item('pyecr_dct_residual_y_axis', label=f'Observed − reconstructed ({view.unit})')
        for kind in ('spectrum','reconstruction','residual','lambda','stability'):
            for axis in ('x','y'):
                dpg.fit_axis_data('pyecr_dct_'+kind+'_'+axis+'_axis')
        self._signed_dct_render_key = key
        self._signed_display_view = view

    def _prior_evidence_active(self):
        d=self._require_dpg()
        return bool(not self._dct_component_mode() and getattr(self,'_prior_archive',None) is not None and d.does_item_exist('pyecr_prior_view') and d.get_value('pyecr_prior_view'))

    def _open_prior_evidence(self,*args):
        from pathlib import Path
        from pyecr.project.prior_archive import SignedPriorArchive
        from pyecr.report.prior_bundle import read_prior_bundle
        d=self._require_dpg()
        try:
            path=Path(d.get_value('pyecr_prior_archive_path'))
            archive=read_prior_bundle(path) if path.is_dir() else SignedPriorArchive.from_json(path.read_text(encoding='utf-8'))
            if not isinstance(archive,SignedPriorArchive):raise ValueError('; '.join(archive.state.blocking_reasons))
            self._prior_archive=archive;d.set_value('pyecr_prior_view',True)
            d.set_value('pyecr_prior_message','Opened archived result: '+archive.state.status)
        except Exception as exc:
            self._prior_archive=None;d.set_value('pyecr_prior_view',False);d.set_value('pyecr_prior_message','Could not open the archived result: '+str(exc))
        self._signed_dct_render_key=None;self._refresh_widgets()

    def _export_prior_evidence(self,*args):
        from pyecr.report.prior_bundle import export_prior_bundle
        d=self._require_dpg()
        try:
            if getattr(self,'_prior_archive',None) is None:raise ValueError('Open a saved result first.')
            path=str(d.get_value('pyecr_prior_export_path')).strip()
            if not path:raise ValueError('Declare a new export directory')
            export_prior_bundle(self._prior_archive,path);d.set_value('pyecr_prior_message','Archived report exported to '+path)
        except Exception as exc:d.set_value('pyecr_prior_message','Export refused: '+str(exc))
