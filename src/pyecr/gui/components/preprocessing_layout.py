"""R03 direct operation layout; original history/provenance services retained."""
import numpy as np
from pyecr.gui.preprocessing_plots import plateau_reference_summary
from pyecr.gui.theme import TOKENS
from pyecr.gui.components.preprocessing_direct import DirectPreprocessingMixin

VERSION='preprocessing-layout/2.0.0'


class PreprocessingLayoutMixin(DirectPreprocessingMixin):
    def _build_preprocessing_layout(self):
        self._build_direct_preprocessing_layout()

    def _move_long_tool_help(self,root):
        dpg=self._require_dpg()
        queue=list(dpg.get_item_children(root,1) or [])
        while queue:
            item=queue.pop(0);queue.extend(dpg.get_item_children(item,1) or [])
            info=dpg.get_item_info(item)
            if info['type'].endswith('mvText') and not dpg.get_item_alias(item):
                value=dpg.get_value(item)
                if isinstance(value,str) and len(value)>90:
                    self._preprocessing_help.append(value)
                    dpg.configure_item(item,show=False)

    def _show_preprocessing_help(self,*args):
        self._size_display_window('pyecr_preprocess_help_window')
        self._require_dpg().configure_item('pyecr_preprocess_help_window',show=True)

    def _show_preprocessing_evidence(self,*args):
        self._size_display_window('pyecr_preprocess_evidence_window')
        self._require_dpg().configure_item('pyecr_preprocess_evidence_window',show=True)
        self._refresh_preprocessing_layout()

    def _copy_preprocessing_evidence(self,*args):
        dpg=self._require_dpg()
        dpg.set_clipboard_text('\n\n'.join(dpg.get_value(tag) for tag in
                              ('pyecr_preprocess_indices_evidence','pyecr_data_fit_quality','pyecr_data_processing_history')))

    def _refresh_preprocessing_layout(self):
        dpg=self._require_dpg()
        if not dpg.does_item_exist('pyecr_preprocessing_layout'):return
        self._refresh_direct_preprocessing()
        pipeline=self.controller.session.preprocessing
        dataset=None if pipeline is None else pipeline.dataset
        if dataset is None:
            state='Import a dataset first; units unconfirmed.';time='t0 / fit interval: unavailable before import.';summary='No accepted processing steps.'
            negative=0
        else:
            source=pipeline.source;meta=dataset.metadata
            state=(f"Raw: {source.raw_time.size} rows · {source.metadata.get('time_unit','unknown')}, {source.metadata.get('signal_unit','unknown')}\n"
                   f"Processed: {dataset.processed_time.size} rows · {meta.get('time_unit','unknown')}, {meta.get('signal_unit','unknown')}")
            t=dataset.processed_time
            span='empty' if not t.size else f'{float(t.min()):.6g} … {float(t.max()):.6g} {meta.get("time_unit","unknown")}'
            negative=int(np.count_nonzero(t<0))
            time=(f"Current interval: {span}; accepted t0={meta.get('manual_t0','not set')}.\n"
                  + (f'{negative} negative-time rows: existing Fit gate applies; review explicit crop/mask.' if negative else 'No negative processed times; other Fit checks remain in force.'))
            reference_note = plateau_reference_summary(dataset)
            if reference_note:
                time += '\n' + reference_note
            operations=[step.operation for step in pipeline.config.steps]
            summary=f'{len(operations)} accepted steps · revision {pipeline.revision}'
            if operations:summary+='\n'+' → '.join(operations[-4:])
        for tag,value in (('pyecr_preprocess_state_compact',state),('pyecr_preprocess_time_compact',time),('pyecr_preprocess_summary_compact',summary)):
            dpg.set_value(tag,value)
            dpg.configure_item(tag,wrap=max(TOKENS.analysis.field_min,dpg.get_item_configuration('pyecr_analysis_left')['width']-TOKENS.analysis.form_gutter))
        dpg.set_value('pyecr_preprocess_critical',f'Negative times: {negative} rows — explicit crop/mask review; Fit gate unchanged.' if negative else '')
        dpg.configure_item('pyecr_preprocess_critical',show=bool(negative),color=TOKENS.colors.warning,
                           wrap=max(TOKENS.analysis.field_min,dpg.get_item_configuration('pyecr_analysis_left')['width']-TOKENS.analysis.form_gutter))
        pending=[name for name,attribute in (('Manual','_pending_typed_preview'),('Stability','_pending_stability_suggestion'),
                                             ('t0','_pending_t0_suggestion'),('Optional','_pending_optional_preview')) if getattr(self,attribute,None) is not None]
        dpg.set_value('pyecr_preprocessing_pending_summary','Pending preview: '+', '.join(pending)+' — review before Accept.' if pending else '')
        dpg.configure_item('pyecr_preprocessing_pending_summary',show=bool(pending))
        # This is disclosure only. Do not call replay, suggest, Preview, Accept,
        # unit conversion or gate services from this method.
        if dpg.get_item_configuration('pyecr_preprocess_evidence_window')['show']:
            if dataset is None:evidence='No dataset / original row provenance.'
            else:
                kept=np.flatnonzero(dataset.mask).tolist();removed=np.flatnonzero(~dataset.mask).tolist()
                evidence=(f'Current original indices ({len(kept)}): {kept}\nExcluded original indices: {removed}\n'
                          'Reasons and per-step original row destinations are in the unchanged processing history and raw-overlay reason series below.\n'
                          +state+'\n'+time)
            dpg.set_value('pyecr_preprocess_indices_evidence',evidence)
        self._set_disclosed_item('pyecr_data_processing_history',True)
