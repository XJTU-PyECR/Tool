"""R03 direct forms and one action bar over the existing P53/P03 services.

No scientific equation is implemented here. Category/mode edits never preview
or commit. A current preview binds only the active operation's visible inputs.
"""
from __future__ import annotations

import json
from hashlib import sha256
from math import isfinite

from pyecr.contracts import SourceIdentity
from pyecr.gui.preprocessing_workflow import FORM_FIELDS
from pyecr.gui.signed_normalization import SIGNED_DISPLAY, SIGNED_OPERATION
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS
from pyecr.preprocessing import ProcessingStep, raw_observations_sha256
from pyecr.gui.preprocessing_plots import preview_uses_separate_axes
from pyecr.gui.components.preprocessing_regions import PreprocessingRegionsMixin
from pyecr.gui.components.preprocessing_more import MorePreprocessingMixin

VERSION = 'preprocess-direct/2'
CATEGORIES = (('normalize', 'Normalize'), ('time', 'Time & range'), ('platforms', 'Platforms'), ('more', 'More'))
NORMALIZATION_MODES = ('Progress', 'Remaining', 'Signed-DCT')
MORE_TOOLS = (
    ('convert_time_unit', 'Time unit'),
    ('resistance_to_conductance', 'Resistance to G'),
    ('conductance_to_conductivity', 'Conductivity'),
    ('mask_points', 'Mask rows'), ('unmask_points', 'Unmask rows'),
    ('drift_correction', 'Drift correction'), ('outlier_mask', 'Outlier review'),
    ('adaptive_downsample', 'Downsample'),
    ('suggest_window', 'Suggest platform'), ('suggest_t0', 'Suggest t0'),
)
OPTIONAL = ('drift_correction', 'outlier_mask', 'adaptive_downsample')


class DirectPreprocessingMixin(MorePreprocessingMixin, PreprocessingRegionsMixin):
    def _build_direct_preprocessing_layout(self):
        dpg = self._require_dpg()
        self._direct_category = 'normalize'
        self._direct_mode = 'Progress'
        self._direct_time_operation = 'set_manual_t0'
        self._direct_platform = 'initial'
        self._direct_more_operation = 'convert_time_unit'
        self._direct_more_list_open = True
        self._direct_preview_binding = None
        self._direct_review_id = None
        self._direct_preview_sequence = 0
        self._direct_ignored_events = []
        self._outlier_review_origin = None
        self._outlier_review_user_edited = False
        self._direct_more_drafts = {}
        self._platform_plot_selection = None
        self._direct_message = ''
        self._direct_details_key = None
        self._preprocessing_help = []
        with dpg.group(parent='pyecr_stage_preprocess_content', tag='pyecr_preprocessing_layout'):
            # Two stable rows fit a 350px rail without hiding any category.
            for pair in (CATEGORIES[:2], CATEGORIES[2:]):
                with dpg.group(horizontal=True):
                    for key, label in pair:
                        dpg.add_button(label=label, tag='pyecr_preprocess_category_'+key,
                                       callback=self._choose_preprocess_category, user_data=key)
            with dpg.child_window(tag='pyecr_preprocess_forms', width=-1,
                                  height=TOKENS.steps.preprocess_form_max, border=False):
                with dpg.group(tag='pyecr_preprocess_form_normalize'):
                    dpg.add_radio_button(NORMALIZATION_MODES, tag='pyecr_preprocess_normalization_mode',
                        default_value='Progress', callback=self._choose_normalization_mode)
                    dpg.add_text('Uses accepted Initial / Final platforms.', tag='pyecr_normalize_platform_note', wrap=TOKENS.wraps.stage)
                    dpg.add_button(label='Review Platforms', tag='pyecr_normalize_open_platforms', callback=self._choose_preprocess_category, user_data='platforms')
                    dpg.add_group(tag='pyecr_direct_signed_host')
                with dpg.group(tag='pyecr_preprocess_form_time', show=False):
                    dpg.add_text('Current time axis; select an operation, then Preview.', wrap=TOKENS.wraps.stage)
                    with dpg.table(header_row=False, width=-1, policy=dpg.mvTable_SizingStretchProp):
                        dpg.add_table_column(init_width_or_weight=1.2)
                        dpg.add_table_column(init_width_or_weight=2.1)
                        with dpg.table_row():
                            dpg.add_button(label='Shift t0', tag='pyecr_direct_time_shift', callback=self._choose_time_operation, user_data='set_manual_t0')
                            dpg.add_group(tag='pyecr_direct_t0_host')
                        with dpg.table_row():
                            dpg.add_button(label='Crop interval', tag='pyecr_direct_time_crop', callback=self._choose_time_operation, user_data='crop')
                            with dpg.table(header_row=False, width=-1, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column();dpg.add_table_column()
                                with dpg.table_row():
                                    dpg.add_group(tag='pyecr_direct_range_start_host')
                                    dpg.add_group(tag='pyecr_direct_range_end_host')
                with dpg.group(tag='pyecr_preprocess_form_platforms', show=False):
                    dpg.add_text('', tag='pyecr_platform_reference_status', wrap=TOKENS.wraps.stage)
                    dpg.add_text('New selection bounds use the current time axis. These fields are drafts, not the recorded references.', wrap=TOKENS.wraps.stage)
                    with dpg.table(header_row=True, width=-1, policy=dpg.mvTable_SizingStretchProp):
                        dpg.add_table_column(label='Platform',init_width_or_weight=1.6)
                        dpg.add_table_column(label='Bounds (draft)',init_width_or_weight=2.2)
                        dpg.add_table_column(label='Review',init_width_or_weight=1.8)
                        for kind, start, end in (('initial', '0', '10'), ('final', '90', '100')):
                            with dpg.table_row():
                                with dpg.group():
                                    dpg.add_button(label=kind.title(), tag='pyecr_direct_platform_'+kind,
                                                   callback=self._choose_platform, user_data=kind)
                                    dpg.add_button(label='Select on plot', tag='pyecr_platform_plot_'+kind,
                                                   callback=self._select_platform_on_plot, user_data=kind, width=-1)
                                with dpg.group():
                                    for edge, value in (('start',start),('end',end)):
                                        with dpg.group(horizontal=True,tag=f'pyecr_platform_{kind}_{edge}_row'):
                                            dpg.add_text(edge.title())
                                            dpg.add_input_text(tag=f'pyecr_platform_{kind}_{edge}',default_value=value,
                                                              width=-1,callback=self._direct_inputs_changed)
                                with dpg.group():
                                    dpg.add_button(label='Preview window', tag='pyecr_platform_preview_'+kind,
                                                   callback=self._preview_platform_window, user_data=kind, width=-1)
                                    dpg.add_button(label='Accept window', tag='pyecr_platform_accept_'+kind,
                                                   callback=self._accept_platform_window, show=False, width=-1)
                    dpg.add_text('', tag='pyecr_platform_plot_notice', wrap=TOKENS.wraps.stage)
                with dpg.group(tag='pyecr_preprocess_form_more', show=False):
                    self._build_more_picker(MORE_TOOLS)
                    dpg.add_group(tag='pyecr_direct_generic_host')
                    dpg.add_group(tag='pyecr_direct_optional_host')
                    dpg.add_group(tag='pyecr_direct_stability_host')
                    dpg.add_group(tag='pyecr_direct_t0_suggestion_host')
                    dpg.move_item('pyecr_more_defaults',parent='pyecr_preprocess_form_more')
                    with dpg.collapsing_header(label='Full evidence / help', tag='pyecr_preprocess_more_details'):
                        dpg.add_group(tag='pyecr_preprocess_manual_details_host')
                        dpg.add_group(tag='pyecr_preprocess_optional_details_host')
                        with dpg.group(tag='pyecr_preprocess_overview'):
                            for name in ('state', 'time', 'summary'):
                                dpg.add_text('', tag=f'pyecr_preprocess_{name}_compact')
                        dpg.add_text('', tag='pyecr_preprocessing_pending_summary', show=False)
                        dpg.add_button(label='Help', tag='pyecr_preprocess_help_action', callback=self._show_preprocessing_help)
            # Keep the commit action ahead of potentially long preview warnings.
            dpg.add_group(horizontal=True, tag='pyecr_preprocessing_action_bar')
            dpg.add_text('Preview first, then Accept to apply.', tag='pyecr_preprocess_accept_status', wrap=TOKENS.wraps.stage)
            dpg.add_text('', tag='pyecr_preprocess_critical', wrap=TOKENS.wraps.stage)
            dpg.add_text('', tag='pyecr_preprocess_change_summary', wrap=TOKENS.wraps.stage, show=False)
            with dpg.group(horizontal=True):
                dpg.add_button(label='Preview details', tag='pyecr_preprocess_preview_details',
                               callback=self._show_direct_preview_details, show=False)
                dpg.add_button(label='View preview plots', tag='pyecr_preprocess_preview_plots',
                               callback=self._show_direct_preview_plots, show=False)
            dpg.add_group(horizontal=True, tag='pyecr_preprocessing_undo_reset_actions')
        # Move original controls once. Preserve all numerical callbacks/values.
        for tag in ('pyecr_signed_controls', 'pyecr_signed_review_group'):
            dpg.move_item(tag, parent='pyecr_direct_signed_host')
        for tag, parent in (('pyecr_typed_group_t0', 'pyecr_direct_t0_host'),
                            ('pyecr_typed_group_start_time', 'pyecr_direct_range_start_host'),
                            ('pyecr_typed_group_end_time', 'pyecr_direct_range_end_host'),
                            ('pyecr_typed_processing_controls', 'pyecr_direct_generic_host'),
                            ('pyecr_optional_controls', 'pyecr_direct_optional_host'),
                            ('pyecr_data_stability_controls', 'pyecr_direct_stability_host'),
                            ('pyecr_data_t0_controls', 'pyecr_direct_t0_suggestion_host')):
            dpg.move_item(tag, parent=parent)
        for field, label in (('t0','t0'),('start_time','Start'),('end_time','End')):
            tag='pyecr_typed_'+field
            dpg.add_text(label,parent='pyecr_typed_group_'+field,before=tag,show=field!='t0')
            dpg.configure_item(tag,label='',width=-1)
            for child in dpg.get_item_children('pyecr_typed_group_'+field,1) or []:
                if dpg.get_item_info(child)['type'].endswith('mvText') and not dpg.get_item_alias(child) and child!=tag:
                    value=dpg.get_value(child)
                    if value!=label:
                        self._preprocessing_help.append(value);dpg.configure_item(child,show=False)
        for tag in ('pyecr_data_apply_processing', 'pyecr_data_accept_processing', 'pyecr_data_reject_processing'):
            dpg.move_item(tag, parent='pyecr_preprocessing_action_bar')
        dpg.set_item_callback('pyecr_data_apply_processing', self._preview_direct_preprocessing)
        dpg.set_item_callback('pyecr_data_accept_processing', self._accept_direct_preprocessing)
        dpg.set_item_callback('pyecr_data_reject_processing', self._reject_direct_preprocessing)
        dpg.set_item_callback('pyecr_signed_accept', self._accept_signed_direct)
        dpg.set_item_callback('pyecr_signed_reject', self._reject_signed_direct)
        dpg.set_item_callback('pyecr_optional_review_indices', self._outlier_review_changed)
        for tag in ('pyecr_data_undo_processing', 'pyecr_data_reset_processing'):
            dpg.move_item(tag, parent='pyecr_preprocessing_undo_reset_actions')
        dpg.add_button(label='History / Details', tag='pyecr_preprocess_evidence_action',
                       parent='pyecr_preprocessing_undo_reset_actions', callback=self._show_preprocessing_evidence)
        # Existing detailed evidence/recipe is retained, with one visible footer.
        for tag in ('pyecr_data_processing_parameters', 'pyecr_typed_preview_evidence', 'pyecr_typed_error',
                    'pyecr_data_processing_reason', 'pyecr_data_processing_buttons'):
            dpg.move_item(tag, parent='pyecr_preprocess_manual_details_host')
        for tag in ('pyecr_optional_options', 'pyecr_optional_comparison'):
            dpg.move_item(tag, parent='pyecr_preprocess_optional_details_host')
        residual = dpg.get_item_parent('pyecr_optional_residual_x_axis')
        dpg.move_item(residual, parent='pyecr_preprocess_optional_details_host')
        self._direct_hidden_actions = (
            'pyecr_data_stability_suggest', 'pyecr_data_stability_accept', 'pyecr_data_stability_reject',
            'pyecr_data_t0_suggest', 'pyecr_data_t0_accept', 'pyecr_data_t0_reject',
            'pyecr_optional_preview', 'pyecr_optional_accept', 'pyecr_optional_reject',
            'pyecr_optional_unmask', 'pyecr_optional_unmask_indices')
        for tag in self._direct_hidden_actions:dpg.configure_item(tag, show=False)
        for root in ('pyecr_data_stability_controls','pyecr_data_t0_controls','pyecr_optional_controls'):
            self._move_long_tool_help(root)
        self._build_direct_evidence_windows()
        self._build_direct_preview_review()
        self._build_preprocess_regions()
        # Drag lines only edit the armed platform draft, never source arrays.
        for edge in ('start','end'):
            dpg.add_drag_line(tag='pyecr_platform_drag_'+edge, label=edge.title(),
                parent='pyecr_data_processed_plot', show=False, color=TOKENS.colors.primary,
                callback=self._platform_dragged, user_data=edge)
        self._direct_preprocess_ready = True
        self._set_direct_dispatch()
        self._prepare_more_labels()
        self._prepare_owned_form('pyecr_preprocessing_layout')
        self._register_owned_root('pyecr_preprocess_evidence_window')
        self._register_owned_root('pyecr_preprocess_help_window')

    def _build_direct_evidence_windows(self):
        dpg=self._require_dpg()
        with dpg.window(label='Processing history / original row provenance', tag='pyecr_preprocess_evidence_window',
                        show=False, width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            with dpg.group(horizontal=True):
                dpg.add_button(label='Close', callback=lambda:dpg.configure_item('pyecr_preprocess_evidence_window',show=False))
                dpg.add_button(label='Copy preprocessing details', tag='pyecr_preprocess_copy_evidence', callback=self._copy_preprocessing_evidence)
            dpg.add_input_text(tag='pyecr_preprocess_indices_evidence', readonly=True, multiline=True, width=-1, height=TOKENS.heights.history)
            dpg.add_group(tag='pyecr_preprocess_quality_host');dpg.add_group(tag='pyecr_preprocess_history_host')
        for tag,host in (('pyecr_data_fit_quality','pyecr_preprocess_quality_host'),('pyecr_data_processing_history','pyecr_preprocess_history_host')):
            dpg.move_item(tag,parent=host)
        with dpg.window(label='Preprocessing help — explicit actions only', tag='pyecr_preprocess_help_window',
                        show=False, width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            dpg.add_button(label='Close', callback=lambda:dpg.configure_item('pyecr_preprocess_help_window',show=False))
            dpg.add_group(tag='pyecr_preprocess_help_host')
            dpg.add_input_text(tag='pyecr_preprocess_help_text', readonly=True, multiline=True, width=-1, height=-1,
                default_value='Choose a category and a visible mode. Preview changes no data. Accept alone records one existing step.\n'
                'Select on plot edits draft bounds only. Suggestions remain candidates.\n'
                'Undo/Reset replay the original service. No automatic Fit, normalization, crop or smoothing.\n\n'+'\n'.join(self._preprocessing_help))
        for tag in ('pyecr_data_processing_separator','pyecr_data_processing_header','pyecr_data_processing_description'):
            dpg.move_item(tag,parent='pyecr_preprocess_help_host')

    def _direct_route(self):
        return (self._direct_category, {'normalize': self._direct_mode, 'time': self._direct_time_operation,
                'platforms': self._direct_platform, 'more': self._direct_more_operation}[self._direct_category])

    def _build_direct_preview_review(self):
        dpg = self._require_dpg()
        with dpg.window(label='Preprocessing preview details', tag='pyecr_preprocess_preview_window',
                        show=False, width=TOKENS.widths.report_path, height=TOKENS.heights.dialog):
            with dpg.group(horizontal=True):
                self._build_preview_commit_actions('preview')
                dpg.add_button(label='Close', callback=lambda: dpg.configure_item('pyecr_preprocess_preview_window', show=False))
                dpg.add_button(label='Copy complete details', tag='pyecr_preprocess_copy_preview', callback=self._copy_direct_preview_details)
            dpg.add_input_text(tag='pyecr_preprocess_preview_text', readonly=True, multiline=True, width=-1, height=-1)
        with dpg.window(label='Preprocessing preview — separate axes', tag='pyecr_preprocess_plots_window',
                        show=False, width=TOKENS.steps.preprocess_plot_width, height=TOKENS.steps.preprocess_plot_height):
            with dpg.group(horizontal=True):
                self._build_preview_commit_actions('plots')
                dpg.add_button(label='Close', callback=lambda: dpg.configure_item('pyecr_preprocess_plots_window', show=False))
            dpg.add_text('', tag='pyecr_preprocess_plots_status', wrap=TOKENS.wraps.stage)
            with dpg.child_window(width=-1, height=-1):
                for role, title in (('input','Current processed source'),('candidate','Candidate only — not accepted')):
                    with dpg.plot(label=title, tag=f'pyecr_preview_{role}_plot', width=-1, height=TOKENS.heights.plot):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label='Time', tag=f'pyecr_preview_{role}_x')
                        dpg.add_plot_axis(dpg.mvYAxis, label='Signal', tag=f'pyecr_preview_{role}_y')
                        dpg.add_scatter_series([], [], label=title, parent=f'pyecr_preview_{role}_y', tag=f'pyecr_preview_{role}_series')
                        dpg.add_scatter_series([], [], label='Window preview', parent=f'pyecr_preview_{role}_y', tag=f'pyecr_preview_{role}_window',show=False)
        for tag in ('pyecr_preprocess_preview_window', 'pyecr_preprocess_plots_window'):
            self._register_owned_root(tag)

    def _build_preview_commit_actions(self, window):
        d = self._require_dpg()
        d.add_button(label='Accept', tag=f'pyecr_preprocess_{window}_accept', enabled=False,
                     callback=self._accept_direct_preprocessing)
        d.add_button(label='Reject', tag=f'pyecr_preprocess_{window}_reject', enabled=False,
                     callback=self._reject_direct_preprocessing)

    def _preview_review_evidence(self):
        p = self.controller.session.preprocessing
        binding = self._direct_preview_binding
        if self._direct_binding_current():
            current, proposed = binding['dataset'], binding['proposed']
            payload = {k: binding[k] for k in ('version','preview_id','signature','parameters_sha256','operation_sha256','step','summary')}
            payload.update(source=binding['source'].to_dict(), recipe=binding['recipe'].to_dict(), state='current_preview')
        elif getattr(self, '_signed_review', None) is not None and p is not None:
            current, proposed = self._signed_review['dataset'], None
            payload = dict(version=VERSION, preview_id=self._direct_review_id, state='blocked_review',
                           source=SourceIdentity.capture(current,p.config).to_dict(), recipe=p.config.to_dict())
        else:
            return dict(version=VERSION, state='no_current_preview', reason='Preview again before Accept.')
        payload.update(raw_observations_sha256=raw_observations_sha256(current),
                       raw_hash_scope='Original raw time and signal only; mask and processed arrays have a separate dataset identity.',
                       mask=current.mask.tolist(), raw_indices=[i for i, valid in enumerate(current.mask) if valid],
                       input_units={k:current.metadata.get(k) for k in ('time_unit','signal_type','signal_unit')})
        if proposed is not None:
            payload.update(proposed_mask=proposed.mask.tolist(),
                proposed_raw_indices=[i for i, valid in enumerate(proposed.mask) if valid],
                proposed_units={k:proposed.metadata.get(k) for k in ('time_unit','signal_type','signal_unit')})
        review = getattr(self, '_signed_review', None)
        if review is not None:
            result = review['result']
            payload.update(signed=result.to_dict(), declared_form=json.loads(self._direct_signature())[-1],
                inverse=dict(formula='sigma_i = S * Y_i + sigma_inf', signal_unit=result.spec.signal_unit,
                             reference_value=result.reference_value, scale_value=result.scale_value))
        return payload

    def _show_direct_preview_details(self, *args):
        dpg = self._require_dpg()
        self._direct_details_key = None
        self._sync_direct_preview_details()
        dpg.configure_item('pyecr_preprocess_preview_window', show=True)

    def _sync_direct_preview_details(self):
        current = self._direct_binding_current()
        review = getattr(self, '_signed_review', None)
        key = (self._direct_review_id if current else None, current, id(review))
        if key != self._direct_details_key:
            self._require_dpg().set_value('pyecr_preprocess_preview_text',
                json.dumps(self._preview_review_evidence(), ensure_ascii=False, indent=2, allow_nan=False))
            self._direct_details_key = key

    def _copy_direct_preview_details(self, *args):
        self._require_dpg().set_clipboard_text(self._require_dpg().get_value('pyecr_preprocess_preview_text'))

    def _show_direct_preview_plots(self, *args):
        if not self._direct_binding_current(): return
        b = self._direct_preview_binding
        dpg = self._require_dpg()
        for role, dataset in (('input', b['dataset']), ('candidate', b['proposed'])):
            dpg.set_value(f'pyecr_preview_{role}_series', [dataset.processed_time.tolist(), dataset.processed_signal.tolist()])
            dpg.configure_item(f'pyecr_preview_{role}_x', label=f"Time ({dataset.metadata['time_unit']})")
            if b['step']['operation']=='set_manual_t0':
                dpg.configure_item(f'pyecr_preview_{role}_x',label=f"Time ({dataset.metadata['time_unit']}) · "+('current origin' if role=='input' else 'proposed origin'))
            from pyecr.plotting.labels import signal_axis_label
            dpg.configure_item(f'pyecr_preview_{role}_y', label=signal_axis_label(dataset.metadata))
            for axis in ('x','y'): dpg.fit_axis_data(f'pyecr_preview_{role}_{axis}')
        self._preview_region_on_panels(b)
        self._direct_plots_id = b['preview_id']
        dpg.set_value('pyecr_preprocess_plots_status', b['preview_id']+' | Current preview; all observations, independent labeled axes. No display grid.')
        dpg.configure_item('pyecr_preprocess_plots_window', show=True)

    def _preview_platform_window(self, sender=None, app_data=None, user_data=None):
        if user_data not in ('initial', 'final'): raise ValueError('Unknown platform role')
        if self._direct_category != 'platforms':
            self._direct_ignored_events.append(dict(event='window_preview', role=user_data)); return
        self._choose_platform(user_data=user_data)
        self._preview_direct_preprocessing()

    def _accept_platform_window(self, sender=None, app_data=None, user_data=None):
        # A row event carries both role and the immutable preview token. It
        # cannot select a row or retarget a newer review as a side effect.
        if (not isinstance(user_data, tuple) or len(user_data) != 2
                or self._direct_route() != ('platforms', user_data[0])
                or user_data[1] is None or user_data[1] != self._direct_review_id):
            self._direct_ignored_events.append(dict(event='window_accept', token=user_data)); return
        self._accept_direct_preprocessing(user_data=user_data[1])

    def _set_direct_dispatch(self):
        dpg=self._require_dpg();category,choice=self._direct_route()
        op = ({'Progress':'normalize','Remaining':'normalize','Signed-DCT':SIGNED_OPERATION}[choice] if category=='normalize'
              else self._direct_time_operation if category=='time' else f'choose_{choice}_stability_window' if category=='platforms' else choice)
        dpg.set_value('pyecr_data_processing_operation', SIGNED_DISPLAY if op==SIGNED_OPERATION else op)
        if category=='normalize':dpg.set_value('pyecr_typed_mode', choice.lower() if choice!='Signed-DCT' else 'progress')
        if op in OPTIONAL:dpg.set_value('pyecr_optional_kind', op)
        self._show_typed_processing_fields(op)
        self._show_optional_fields(op)

    def _clear_direct_preview(self, message=''):
        self._direct_preview_binding=None
        self._direct_review_id=None
        for name in ('_pending_typed_preview','_pending_optional_preview','_pending_stability_suggestion','_pending_t0_suggestion','_signed_review'):
            setattr(self,name,None)
        self._direct_message=message
        self._signed_message=message or 'Preview before Accept. Data and recipe unchanged.'
        self._clear_optional_input_error()
        d=self._require_dpg()
        from pyecr.gui.preprocessing_workflow import FIELD_DOMAINS
        for field in FIELD_DOMAINS:
            tag='pyecr_optional_error_'+field
            if d.does_item_exist(tag):d.set_value(tag,'');d.configure_item(tag,show=False)

    def _choose_preprocess_category(self, sender=None, app_data=None, user_data=None):
        if user_data not in dict(CATEGORIES):raise ValueError('Unknown preprocessing category')
        self._clear_direct_preview()
        self._disarm_platform_plot()
        self._direct_category=user_data
        if user_data=='more':self._direct_more_list_open=True
        self._set_direct_dispatch();self._refresh_widgets()
        # Presentation-only alignment leaves room for the shared footer.
        dpg=self._require_dpg();pos=dpg.get_item_state('pyecr_step_panel_preprocess').get('pos')
        if pos:dpg.set_y_scroll('pyecr_analysis_left',max(0,float(pos[1])))
        dpg.set_y_scroll('pyecr_preprocess_forms',0)
        if user_data=='more':self._more_scroll_pending=dpg.get_frame_count()+2

    def _choose_normalization_mode(self, sender=None, app_data=None, user_data=None):
        mode=app_data if app_data is not None else self._require_dpg().get_value('pyecr_preprocess_normalization_mode')
        if mode not in NORMALIZATION_MODES:raise ValueError('Unknown normalization mode')
        self._clear_direct_preview();self._direct_mode=mode
        self._require_dpg().set_value('pyecr_preprocess_normalization_mode',mode)
        self._set_direct_dispatch();self._refresh_widgets()

    def _choose_time_operation(self, sender=None, app_data=None, user_data=None):
        if user_data not in ('set_manual_t0','crop'):raise ValueError('Unknown time operation')
        self._clear_direct_preview();self._direct_time_operation=user_data
        self._set_direct_dispatch();self._refresh_widgets()

    def _choose_platform(self, sender=None, app_data=None, user_data=None):
        if user_data not in ('initial','final'):raise ValueError('Unknown platform role')
        self._clear_direct_preview();self._disarm_platform_plot();self._direct_platform=user_data
        self._set_direct_dispatch();self._refresh_widgets()

    def _choose_more_operation(self, sender=None, app_data=None, user_data=None):
        if user_data not in dict(MORE_TOOLS):raise ValueError('Unknown More operation')
        dpg=self._require_dpg();old=self._direct_more_operation
        if old in FORM_FIELDS:
            self._direct_more_drafts[old]={name:dpg.get_value('pyecr_typed_'+name) for name in FORM_FIELDS[old]}
        self._clear_direct_preview();self._direct_more_operation=user_data
        self._direct_more_list_open=False
        for name,value in self._direct_more_drafts.get(user_data,{}).items():dpg.set_value('pyecr_typed_'+name,value)
        self._set_direct_dispatch();self._refresh_widgets()
        dpg.set_y_scroll('pyecr_preprocess_forms',0)
        self._more_scroll_pending=dpg.get_frame_count()+2

    def _show_more_tools(self,sender=None,app_data=None,user_data=None):
        self._clear_direct_preview();self._direct_more_list_open=True
        self._refresh_widgets();self._require_dpg().set_y_scroll('pyecr_preprocess_forms',0)
        self._more_scroll_pending=self._require_dpg().get_frame_count()+2

    def _direct_typed_values(self, operation):
        dpg=self._require_dpg()
        if self._direct_category=='platforms':
            return {edge+'_time':dpg.get_value(f'pyecr_platform_{self._direct_platform}_{edge}') for edge in ('start','end')}
        return {name:dpg.get_value('pyecr_typed_'+name) for name in FORM_FIELDS[operation]}

    def _direct_signature(self):
        dpg=self._require_dpg();category,choice=self._direct_route()
        if category=='normalize':
            values=self._signed_values() if choice=='Signed-DCT' else {'mode':choice.lower()}
        elif category=='time':values=self._direct_typed_values(choice)
        elif category=='platforms':values=self._direct_typed_values('choose_'+choice+'_stability_window')
        elif choice in OPTIONAL:
            names={'drift_correction':('drift_method','reference_start','reference_end','scope_start','scope_end','anchor_time'),
                   'outlier_mask':('threshold','half_window','noise_floor'),
                   'adaptive_downsample':('target_points','early_until','late_power')}[choice]
            values={name:dpg.get_value('pyecr_optional_field_'+name) for name in names}
            if choice=='outlier_mask':values['review_indices']=dpg.get_value('pyecr_optional_review_indices')
        elif choice=='suggest_window':
            values={name:dpg.get_value('pyecr_data_stability_'+name) for name in ('kind','min_points')}
        elif choice=='suggest_t0':values={}
        else:values=self._direct_typed_values(choice)
        return json.dumps([VERSION,category,choice,values],sort_keys=True,ensure_ascii=False,allow_nan=False)

    def _direct_pending_kind(self):
        category,choice=self._direct_route()
        return 'optional' if category=='more' and choice in OPTIONAL else 'stability' if choice=='suggest_window' else 't0' if choice=='suggest_t0' else 'typed'

    def _pending_for_kind(self, kind):
        return getattr(self, {'typed':'_pending_typed_preview','optional':'_pending_optional_preview',
                              'stability':'_pending_stability_suggestion','t0':'_pending_t0_suggestion'}[kind],None)

    def _direct_binding_current(self):
        binding=self._direct_preview_binding;p=self.controller.session.preprocessing
        if binding is None or p is None:return False
        try:signature=self._direct_signature()
        except (ValueError,TypeError,KeyError):return False
        return (binding['signature']==signature and binding['dataset'] is p.dataset and binding['recipe'] is p.config
                and binding['data_revision']==self.controller.analysis_session.data_revision
                and binding['workflow_revision']==self.controller.analysis_session.current_revision
                and binding['revision']==p.revision and self._pending_for_kind(binding['kind']) is binding['pending'])

    def _validate_direct_preview(self, kind):
        if not self._direct_binding_current() or self._direct_preview_binding['kind']!=kind:
            self._clear_direct_preview('Preview expired or inputs changed; Preview again before Accept.')
            raise ValueError(self._direct_message)
        # Reuse the existing source contract at commit, without regenerating a step.
        p=self.controller.session.preprocessing
        self._direct_preview_binding['source'].assert_matches(SourceIdentity.capture(p.dataset,p.config))
        step, _ = self._direct_candidate(kind, self._pending_for_kind(kind))
        if self._operation_fingerprint(step) != self._direct_preview_binding['operation_sha256']:
            self._clear_direct_preview('Preview operation changed; Preview again before Accept.')
            raise ValueError(self._direct_message)

    @staticmethod
    def _operation_fingerprint(step):
        # Acceptance time for old suggestion callbacks is intentionally not a
        # parameter. The scientific operation and all parameters are frozen.
        data = dict(operation=step.operation, parameters=step.to_dict()['parameters'])
        return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()

    def _preview_direct_preprocessing(self, sender=None, app_data=None, user_data=None):
        dpg=self._require_dpg();kind=self._direct_pending_kind()
        if self._direct_category=='more' and self._direct_more_list_open:
            self._clear_direct_preview('Choose a More tool before Preview.');self._refresh_widgets();return
        keep_reviewed=(self._direct_route()==('more','outlier_mask') and self._outlier_review_user_edited
                       and self._outlier_review_origin==self._plot_source_key())
        reviewed=dpg.get_value('pyecr_optional_review_indices') if keep_reviewed else ''
        self._clear_direct_preview();self._set_direct_dispatch()
        self._direct_preview_sequence+=1
        self._direct_review_id=f'r04-preview-{self._direct_preview_sequence}'
        callbacks={'typed':self._preview_typed_preprocessing,'optional':self._preview_optional,
                   'stability':self._suggest_stability_window,'t0':self._suggest_t0}
        errors_before=len(self.controller.state.errors)
        callbacks[kind](None,None,None)
        if kind=='optional' and keep_reviewed and self._pending_optional_preview is not None:
            dpg.set_value('pyecr_optional_review_indices',reviewed)
        if self._direct_route()==('more','outlier_mask'):
            self._outlier_review_origin=self._plot_source_key()
        pending=self._pending_for_kind(kind);p=self.controller.session.preprocessing
        if pending is not None and p is not None:
            try:
                if (kind=='optional' and pending[0].step.operation=='outlier_mask'
                        and not self._optional_step_for_accept(pending[0]).parameters['raw_indices']):
                    self._direct_message='No reviewed outlier rows; nothing to Accept. Adjust detection settings or retain all rows.'
                    self._refresh_widgets();return
                step,proposed=self._direct_candidate(kind,pending)
                summary=self._direct_preview_summary(step, p.dataset, proposed)
                self._direct_preview_binding=dict(version=VERSION,signature=self._direct_signature(),kind=kind,
                    preview_id=self._direct_review_id,
                    parameters_sha256=sha256(self._direct_signature().encode('utf-8')).hexdigest(),
                    operation_sha256=self._operation_fingerprint(step), pending=pending, proposed=proposed,
                    dataset=p.dataset,recipe=p.config,revision=p.revision,data_revision=self.controller.analysis_session.data_revision,
                    workflow_revision=self.controller.analysis_session.current_revision,
                    source=SourceIdentity.capture(p.dataset,p.config),step=step.to_dict(),summary=summary)
                self._direct_message=summary
                if self._different_preview_units(p.dataset, proposed) and step.operation != SIGNED_OPERATION:
                    self._show_direct_preview_plots()
            except (ValueError,TypeError,KeyError) as exc:
                self._clear_direct_preview('Preview unavailable: '+str(exc))
        else:
            self._direct_message=(dpg.get_value('pyecr_typed_error') if kind=='typed' else
                'No acceptable candidate; review the tool evidence and errors.')
            errors=[error for error in self.controller.state.errors[errors_before:] if error.page==GuiPage.DATA]
            if errors:self._direct_message='Preview blocked: '+errors[-1].message
            elif kind in ('stability','t0'):
                self._direct_message=getattr(self,'_stability_preview_message' if kind=='stability' else '_t0_preview_message',self._direct_message)
            review = getattr(self, '_signed_review', None)
            if review is not None and not review['result'].state.ready:
                self._direct_message = 'Preview blocked: '+'; '.join(review['result'].state.blocking_reasons)
        self._refresh_widgets()

    def _direct_preview_summary(self, step, current, proposed):
        display=step.operation
        if self._direct_category=='more':
            display=dict(MORE_TOOLS).get(step.operation,{'choose_initial_stability_window':'Initial platform',
                'choose_final_stability_window':'Final platform','set_manual_t0':'Set t0'}.get(step.operation,step.operation))
        lines = [f'Preview: {display}']
        n, m = len(current.processed_time), len(proposed.processed_time)
        if n != m:
            lines.append(f'{n} → {m} observations; {abs(n-m)} '+('excluded' if m < n else 'restored')+'.')
        elif step.operation == 'normalize':
            lines.append(f'{n} → {m} observations; complete valid set retained.')
        for key, label in (('time_unit','Time'), ('signal_unit','Signal')):
            before, after = current.metadata.get(key), proposed.metadata.get(key)
            if before != after: lines.append(f'{label} unit: {before} → {after}.')
        if step.operation.startswith('choose_') and step.operation.endswith('_stability_window'):
            lines.append(f"Window [{step.parameters['start_time']:.9g}, {step.parameters['end_time']:.9g}] {current.metadata['time_unit']}; candidate until Accept.")
        if step.operation == SIGNED_OPERATION:
            result = self._signed_review['result']
            lines.append(f'sigma_inf (σ∞)={result.reference_value:.9g}; S={result.scale_value:.9g} {result.spec.signal_unit}.')
            # The scientific warning and all metrics remain verbatim in Details.
            short = {'external_known_terminal':'External known terminal; conditional on your declaration.',
                     'terminal_unverified_observed':'Observed final plateau not accepted; tail check only.',
                     'terminal_short':'Short final plateau; confirm stability.',
                     'low_information':'Low information relative to local noise.',
                     'noise_unassessed':'Too few observations to assess noise.',
                     'terminal_unstable_drift':'Final plateau drift detected.',
                     'terminal_unstable_scatter':'Final plateau scatter detected.',
                     'observed_terminal_mismatch':'Observed tail differs from the declared terminal.',
                     'terminal_window_stale':'Recorded final window is stale.',
                     'scale_single_point_dominance':'Scale dominated by one observation; inspect outliers. Max retained.'}
            for warning in result.state.warnings:
                lines.append(short.get(warning.split(':',1)[0], warning))
        return '\n'.join(lines)

    @staticmethod
    def _different_preview_units(current, proposed):
        return preview_uses_separate_axes(current,proposed)

    def _direct_candidate(self,kind,pending):
        p=self.controller.session.preprocessing
        if kind=='typed':return pending[0],pending[1]
        if kind=='optional':
            if pending[0].step.operation=='outlier_mask':
                from pyecr.gui.preprocessing_workflow import parse_indices
                if not parse_indices(self._require_dpg().get_value('pyecr_optional_review_indices')):
                    raise ValueError('No reviewed outlier rows; nothing to Accept.')
            step=self._optional_step_for_accept(pending[0])
            if step.operation=='outlier_mask' and not step.parameters['raw_indices']:
                raise ValueError('No reviewed outlier rows; nothing to Accept.')
        elif kind=='stability':
            suggestion=pending[0]
            step=ProcessingStep.create('choose_'+suggestion.kind+'_stability_window',
                {'start_time':float(suggestion.start_time),'end_time':float(suggestion.end_time)})
        else:step=ProcessingStep.create('set_manual_t0',{'t0':float(pending[0].t0_s)})
        return step,p.config.append(step).replay(p.source)

    def _accept_direct_preprocessing(self,sender=None,app_data=None,user_data=None):
        if user_data is not None and user_data!=self._direct_review_id:
            self._direct_ignored_events.append(dict(event='accept',preview_id=user_data,current=self._direct_review_id));return
        kind=self._direct_pending_kind()
        p=self.controller.session.preprocessing;revision=None if p is None else p.revision
        b=self._direct_preview_binding
        count_changed=bool(b and len(b['dataset'].processed_time)!=len(b['proposed'].processed_time))
        summary=self._direct_message if count_changed else ''
        # Original handlers validate again and remain the only commit routes.
        {'typed':self._accept_typed_preprocessing,'optional':self._accept_optional,
         'stability':self._accept_stability_window,'t0':self._accept_t0}[kind](None,None,None)
        if not self._direct_binding_current():
            accepted=p is not None and p.revision!=revision
            self._clear_direct_preview(('Accepted one step.'+('\n'+summary.removeprefix('Preview: ') if summary else ' See History / Details.')) if accepted else 'No current preview; Preview again before Accept.')
        self._refresh_widgets()

    def _reject_direct_preprocessing(self,sender=None,app_data=None,user_data=None):
        if user_data is not None and user_data!=self._direct_review_id:
            self._direct_ignored_events.append(dict(event='reject',preview_id=user_data,current=self._direct_review_id));return
        self._clear_direct_preview('Rejected. Data and recipe unchanged.');self._refresh_widgets()

    def _accept_signed_direct(self,sender=None,app_data=None,user_data=None):
        if self._direct_route()!=('normalize','Signed-DCT'):
            self._direct_ignored_events.append(dict(event='signed_accept',preview_id=user_data,current=self._direct_review_id));return
        self._accept_direct_preprocessing(sender,app_data,user_data)

    def _reject_signed_direct(self,sender=None,app_data=None,user_data=None):
        if self._direct_route()!=('normalize','Signed-DCT'):
            self._direct_ignored_events.append(dict(event='signed_reject',preview_id=user_data,current=self._direct_review_id));return
        self._reject_direct_preprocessing(sender,app_data,user_data)

    def _direct_inputs_changed(self,sender=None,app_data=None,user_data=None):
        self._clear_direct_preview('Inputs changed; Preview again before Accept.');self._refresh_widgets()

    def _outlier_review_changed(self,sender=None,app_data=None,user_data=None):
        self._outlier_review_origin=self._plot_source_key();self._outlier_review_user_edited=True
        self._direct_inputs_changed(sender,app_data,user_data)

    def _plot_source_key(self):
        p=self.controller.session.preprocessing
        return None if p is None else (id(p.dataset),p.revision,self.controller.analysis_session.data_revision)

    def _disarm_platform_plot(self):
        self._platform_plot_selection=None
        dpg=self._require_dpg()
        for edge in ('start','end'):
            if dpg.does_item_exist('pyecr_platform_drag_'+edge):dpg.configure_item('pyecr_platform_drag_'+edge,show=False)

    def _select_platform_on_plot(self,sender=None,app_data=None,user_data=None):
        dpg=self._require_dpg();self._choose_platform(user_data=user_data)
        try:
            if self._plot_source_key() is None:raise ValueError('Import data first.')
            for edge in ('start','end'):
                value=float(dpg.get_value(f'pyecr_platform_{user_data}_{edge}'))
                if not isfinite(value):raise ValueError('Enter finite draft bounds before plot selection.')
                dpg.set_value('pyecr_platform_drag_'+edge,value);dpg.configure_item('pyecr_platform_drag_'+edge,show=True)
            self._platform_plot_selection=(user_data,self._plot_source_key())
            dpg.set_value('pyecr_analysis_plot_tabs','pyecr_analysis_processed_tab')
            dpg.set_value('pyecr_platform_plot_notice','Drag Start / End on the Processed plot. Draft only; then Preview.')
        except (ValueError,TypeError) as exc:
            self._disarm_platform_plot();dpg.set_value('pyecr_platform_plot_notice',str(exc))

    def _platform_dragged(self,sender=None,app_data=None,user_data=None):
        armed=self._platform_plot_selection
        if armed is None or armed[1]!=self._plot_source_key() or self._direct_category!='platforms':
            self._disarm_platform_plot();return
        value=float(self._require_dpg().get_value(sender) if app_data is None else app_data)
        if not isfinite(value):return
        self._require_dpg().set_value(f'pyecr_platform_{armed[0]}_{user_data}',format(value,'.17g'))
        self._direct_inputs_changed()

    def _refresh_direct_preprocessing(self):
        dpg=self._require_dpg();category,choice=self._direct_route()
        if self._outlier_review_origin is not None and self._outlier_review_origin!=self._plot_source_key():
            dpg.set_value('pyecr_optional_review_indices','')
            self._outlier_review_origin=None;self._outlier_review_user_edited=False
        if self._direct_preview_binding is not None and not self._direct_binding_current():
            self._clear_direct_preview('Inputs or source changed; Preview again before Accept.')
        if self._platform_plot_selection is not None and self._platform_plot_selection[1]!=self._plot_source_key():self._disarm_platform_plot()
        for key,_ in CATEGORIES:
            dpg.configure_item('pyecr_preprocess_form_'+key,show=category==key)
            dpg.bind_item_theme('pyecr_preprocess_category_'+key,'pyecr_preprocess_selected_theme' if category==key else 0)
        for op,tag in (('set_manual_t0','pyecr_direct_time_shift'),('crop','pyecr_direct_time_crop')):
            dpg.bind_item_theme(tag,'pyecr_preprocess_selected_theme' if op==self._direct_time_operation else 0)
        for kind in ('initial','final'):
            dpg.bind_item_theme('pyecr_direct_platform_'+kind,'pyecr_preprocess_selected_theme' if kind==self._direct_platform else 0)
            row_current = self._direct_binding_current() and self._direct_route() == ('platforms', kind)
            dpg.configure_item('pyecr_platform_accept_'+kind, show=row_current, enabled=row_current)
            dpg.set_item_user_data('pyecr_platform_accept_'+kind, (kind,self._direct_review_id))
            dpg.configure_item('pyecr_platform_preview_'+kind, enabled=self.controller.session.dataset is not None)
        for op,_ in MORE_TOOLS:
            dpg.bind_item_theme('pyecr_direct_more_'+op,'pyecr_preprocess_selected_theme' if op==self._direct_more_operation else 0)
        more_form=category=='more' and not self._direct_more_list_open
        dpg.configure_item('pyecr_more_tool_list',show=self._direct_more_list_open)
        dpg.configure_item('pyecr_direct_more_back',show=not self._direct_more_list_open)
        for kind,tag in (('optional','pyecr_direct_optional_host'),('stability','pyecr_direct_stability_host'),('t0','pyecr_direct_t0_suggestion_host')):
            dpg.configure_item(tag,show=more_form and self._direct_pending_kind()==kind)
        dpg.configure_item('pyecr_direct_generic_host',show=more_form and choice in FORM_FIELDS)
        for field in ('t0','start_time','end_time'):
            dpg.configure_item('pyecr_typed_group_'+field,show=True)
            error='pyecr_typed_error_'+field
            dpg.configure_item(error,show=bool(dpg.get_value(error)))
        dpg.configure_item('pyecr_normalize_platform_note',show=self._direct_mode!='Signed-DCT')
        dpg.configure_item('pyecr_normalize_open_platforms',show=self._direct_mode!='Signed-DCT')
        if category == 'normalize' and choice != 'Signed-DCT':
            ds = self.controller.session.dataset
            windows = {} if ds is None else ds.metadata.get('stability_windows', {})
            missing = [kind.title() for kind in ('initial', 'final') if not windows.get(kind)]
            already = ds is not None and ds.metadata.get('normalization_source') == 'stability_windows' and ds.metadata.get('signal_type') == 'normalized'
            dpg.set_value('pyecr_normalize_platform_note',
                'Already normalized. Shift t0 and crop without reselecting platforms. Undo normalization before changing its references.' if already else
                'Requires accepted Initial and Final platforms.\n' +
                ('Missing: ' + ', '.join(missing) + '. Review Platforms before Preview.' if missing
                 else 'Initial and Final recorded. Preview reviews the normalization before Accept.'))
        from pyecr.gui.preprocessing_plots import plateau_reference_summary
        dpg.set_value('pyecr_platform_reference_status', plateau_reference_summary(self.controller.session.dataset))
        dpg.configure_item('pyecr_platform_reference_status', wrap=max(TOKENS.analysis.field_min,
            dpg.get_item_configuration('pyecr_analysis_left')['width'] - TOKENS.analysis.form_gutter))
        for tag in self._direct_hidden_actions:dpg.configure_item(tag,show=False)
        for tag in ('pyecr_optional_review_indices','pyecr_optional_evidence'):
            dpg.configure_item(tag,show=category=='more' and choice=='outlier_mask')
        current=self._direct_binding_current()
        for tag in ('pyecr_data_accept_processing','pyecr_data_reject_processing','pyecr_signed_accept','pyecr_signed_reject'):
            dpg.set_item_user_data(tag,self._direct_review_id)
        dpg.configure_item('pyecr_data_apply_processing',enabled=self.controller.session.dataset is not None and not (category=='more' and self._direct_more_list_open))
        self._refresh_more_layout(MORE_TOOLS)
        dpg.configure_item('pyecr_data_accept_processing',show=current,enabled=current)
        review=getattr(self,'_signed_review',None)
        rejectable = current or review is not None
        for window in ('preview', 'plots'):
            for action, enabled in (('accept', current), ('reject', rejectable)):
                tag = f'pyecr_preprocess_{window}_{action}'
                dpg.configure_item(tag, enabled=enabled)
                dpg.set_item_user_data(tag, self._direct_review_id)
        blocked = (not current and (review is not None or self._direct_message.startswith(
            ('Input error:', 'Preview blocked:', 'Preview unavailable:'))))
        dpg.set_value('pyecr_preprocess_accept_status',
            'Preview ready. Accept to apply; data unchanged until Accept.' if current else
            'Preview blocked. Resolve the reason below, then Preview again.' if blocked else
            'Preview first; Accept appears when a candidate is ready.')
        dpg.configure_item('pyecr_data_reject_processing',show=current or review is not None,enabled=current or review is not None)
        dpg.configure_item('pyecr_preprocess_preview_details', show=current or review is not None)
        dpg.configure_item('pyecr_preprocess_preview_plots', show=current and self._direct_route()!=('normalize','Signed-DCT'))
        if dpg.get_item_configuration('pyecr_preprocess_plots_window')['show'] and (not current or self._direct_plots_id != self._direct_review_id):
            dpg.set_value('pyecr_preprocess_plots_status', 'Historical preview '+str(self._direct_plots_id)+'; expired, rejected or accepted. It cannot be accepted again.')
        if dpg.get_item_configuration('pyecr_preprocess_preview_window')['show']:
            # Open evidence follows the actual current state. No stale record
            # can be mistaken for a new candidate after a source/input change.
            self._sync_direct_preview_details()
        dpg.set_value('pyecr_preprocess_change_summary',self._direct_message)
        dpg.configure_item('pyecr_preprocess_change_summary',show=bool(self._direct_message))
        rail=dpg.get_item_state('pyecr_analysis_left').get('rect_size',(0,0))
        form_height=max(TOKENS.steps.preprocess_form_min,min(TOKENS.steps.preprocess_form_max,int(rail[1])-TOKENS.steps.preprocess_footer_reserve))
        available=form_height
        form_state=dpg.get_item_state('pyecr_preprocess_forms')
        end_state=dpg.get_item_state('pyecr_preprocessing_undo_reset_actions')
        header_state=dpg.get_item_state('pyecr_stage_preprocess')
        if all(s.get('rect_size',(0,0))[1]>0 for s in (form_state,end_state,header_state)):
            prefix=form_state['pos'][1]-header_state['pos'][1]+TOKENS.steps.padding
            footer=end_state['pos'][1]+end_state['rect_size'][1]-form_state['pos'][1]-form_state['rect_size'][1]
            form_height=max(TOKENS.steps.preprocess_form_min,min(TOKENS.steps.preprocess_form_max,
                int(rail[1]-prefix-footer-2*TOKENS.steps.padding)))
            available=max(TOKENS.steps.preprocess_form_min,int(rail[1]-prefix-footer-2*TOKENS.steps.padding))
        if category in ('more', 'time', 'normalize'):
            content=dpg.get_item_state('pyecr_preprocess_form_'+category).get('rect_size',(0,0))[1]
            if content>0:form_height=max(TOKENS.steps.preprocess_form_min,min(available,int(content)+2*TOKENS.steps.padding))
        dpg.configure_item('pyecr_preprocess_forms',height=form_height)
        width=max(TOKENS.analysis.field_min,dpg.get_item_configuration('pyecr_analysis_left')['width']-TOKENS.analysis.form_gutter-2*TOKENS.steps.padding)
        for key,_ in CATEGORIES:dpg.configure_item('pyecr_preprocess_category_'+key,width=(width-TOKENS.spacing.item_spacing[0])//2)
        inner=dpg.get_item_state('pyecr_preprocess_forms').get('rect_size',(0,0))[0]
        narrow=0<inner<TOKENS.steps.preprocess_table_wrap_width
        for role in ('initial','final'):
            for edge in ('start','end'):
                dpg.configure_item(f'pyecr_platform_{role}_{edge}_row',horizontal=not narrow)
            for action,normal,small in (('plot','Select on plot','Select\non plot'),
                                        ('preview','Preview window','Preview\nwindow'),
                                        ('accept','Accept window','Accept\nwindow')):
                dpg.configure_item(f'pyecr_platform_{action}_{role}',label=small if narrow else normal)
        wrap=max(TOKENS.analysis.field_min,int(inner)-2*TOKENS.steps.padding-TOKENS.steps.scrollbar_gutter)
        # Traverse the actual nested container. DPG fill-width fields do not
        # inherit group widths; table cells retain their native column sizing.
        queue=list(dpg.get_item_children('pyecr_preprocess_forms',1) or [])
        while queue:
            item=queue.pop();kind=dpg.get_item_info(item)['type'].split('::')[-1]
            if kind=='mvTable':continue
            queue.extend(dpg.get_item_children(item,1) or [])
            if kind in ('mvInputText','mvInputDouble','mvInputFloat','mvInputInt','mvCombo'):
                # CompactFields owns both control and wrapper dimensions. A
                # positive group width is broadcast to its input AND spacer by
                # DPG; overwriting a cached inline row pushed inputs off-screen.
                if item in self._compact_fields.rows:continue
                dpg.configure_item(item,width=wrap)
                parent=dpg.get_item_parent(item)
                if dpg.get_item_info(parent)['type'].endswith('mvGroup'):
                    dpg.configure_item(parent,width=wrap)
            elif kind=='mvText' and item not in self._compact_fields.labels:dpg.configure_item(item,wrap=wrap)
