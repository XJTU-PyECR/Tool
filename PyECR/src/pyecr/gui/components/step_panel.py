"""R02 presentation containers. No workflow/service mutation or gate evaluation."""
from contextlib import contextmanager
import json

from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS, FONT_TAG, SECTION_FONT_TAG
from pyecr.gui.workflow_display import WORKFLOW_STEPS, brief_text
from pyecr.model_display import display_fit_worker_text

VERSION = 'analysis-step-panel/1'
STEP_PAGES = {'data': GuiPage.DATA, 'preprocess': GuiPage.DATA,
              'model': GuiPage.MODEL, 'fit': GuiPage.FIT,
              'advanced': GuiPage.DIAGNOSTICS, 'export': GuiPage.REPORT}


@contextmanager
def step_panel(dpg, key, label, *, default_open=None):
    """A real auto-height child paints both header and every descendant body.

    Keep the old collapsing-header/content aliases (including persisted open
    state). Nested children inherit ChildBg, so scrolling evidence has no hole.
    """
    with dpg.child_window(tag=f'pyecr_step_panel_{key}', width=-1,
                          auto_resize_y=True, always_auto_resize=True,
                          border=False, no_scrollbar=True,
                          always_use_window_padding=True):
        dpg.bind_item_theme(f'pyecr_step_panel_{key}', 'pyecr_step_theme_normal')
        with dpg.collapsing_header(label=label, tag=f'pyecr_stage_{key}', default_open=key == 'data' if default_open is None else default_open):
            if dpg.does_item_exist(SECTION_FONT_TAG):
                dpg.bind_item_font(f'pyecr_stage_{key}', SECTION_FONT_TAG)
            dpg.add_text('', tag=f'pyecr_step_{key}_status', wrap=TOKENS.wraps.stage)
            dpg.add_text('', tag=f'pyecr_step_{key}_warning', wrap=TOKENS.wraps.stage, show=False)
            dpg.add_group(tag=f'pyecr_stage_{key}_content')
            dpg.add_text('', tag=f'pyecr_stage_{key}_reason', wrap=TOKENS.wraps.stage)
            # The header is larger/regular, but its body keeps normal control
            # typography even on runtimes which inherit a container's font.
            if dpg.does_item_exist(FONT_TAG):
                for tag in (f'pyecr_step_{key}_status',f'pyecr_step_{key}_warning',
                            f'pyecr_stage_{key}_content',f'pyecr_stage_{key}_reason'):
                    dpg.bind_item_font(tag,FONT_TAG)
            yield


def step_status(key, session, state):
    """One short phrase; unabridged evidence is kept in each Details block."""
    page = STEP_PAGES[key]
    task = state.tasks[page]
    error = state.current_errors.get(page)
    if error is not None:
        return 'Error: '+brief_text(error.message), 'error'
    if task.status in {'running', 'error', 'cancelled'}:
        message = display_fit_worker_text(task.message) if page is GuiPage.FIT else task.message
        return task.status.title()+': '+brief_text(message), 'error' if task.status == 'error' else 'normal'
    if key == 'data':
        return ('Data imported.' if session.dataset is not None else 'Select a file.'), 'normal'
    if key == 'preprocess':
        return 'Review processing; changes require Accept.', 'normal'
    if key == 'model':
        return ('Configuration applied.' if session.current_analysis_config is not None else 'Review and Apply configuration.'), 'normal'
    if key == 'fit':
        return ('Historical Fit — out of date.' if session.fit_stale and session.fit_result is not None else
                'Fit completed.' if session.fit_result is not None else 'Ready for explicit Run Fit.'), 'normal'
    if key == 'advanced':
        return ('Diagnostics out of date.' if session.diagnostics is not None and session.diagnostics_stale else
                'Diagnostics available; review reliability.' if session.diagnostics is not None else 'Run Diagnostics explicitly.'), 'normal'
    return 'Capture or export a current result.', 'normal'


def file_details(preview, raw, dataset):
    """Full draft metadata and current source are different identities.

    No shortened UI path/hash is used as input. Original row provenance stays
    verbatim in metadata; large raw arrays remain in the original dataset.
    """
    draft = None if preview is None else preview.to_dict()
    current = None if raw is None else {
        'metadata': dict(raw.metadata), 'raw_rows': int(raw.raw_time.size),
        'processed_rows': int(dataset.processed_time.size),
        'raw_original_indices': list(range(int(raw.raw_time.size))),
        'current_original_mask': dataset.mask.tolist(),
        'raw_index_convention': 'zero-based; full original row mappings in metadata',
    }
    return json.dumps({'schema_version': VERSION, 'import_draft_not_current_source': draft,
                       'current_analysis_source': current}, ensure_ascii=False, indent=2, default=str)


def details_text(key, dpg):
    tags = {
        'data': ('pyecr_data_header_status', 'pyecr_data_text_parse_status', 'pyecr_data_time_suggestion'),
        'preprocess': ('pyecr_preprocess_state_compact', 'pyecr_preprocess_time_compact', 'pyecr_preprocess_summary_compact'),
        'model': ('pyecr_model_apply_status', 'pyecr_model_assumptions', 'pyecr_model_limitations', 'pyecr_model_derived_values'),
        'fit': ('pyecr_fit_ready_status', 'pyecr_fit_blocking_reasons', 'pyecr_fit_quality_warnings',
                'pyecr_fit_context_summary', 'pyecr_fit_usage_help', 'pyecr_fit_start_note', 'pyecr_task_status_fit',
                'pyecr_fit_live_status', 'pyecr_fit_progress_details', 'pyecr_solver_details'),
        'advanced': ('pyecr_task_status_diagnostics', 'pyecr_diagnostics_reliability', 'pyecr_diagnostics_warnings',
                     'pyecr_diagnostics_disabled_reason', 'pyecr_diagnostics_cancel_reason'),
        'export': ('pyecr_analysis_report_reason',),
    }
    texts = [next(s.help_text for s in WORKFLOW_STEPS if s.key == key)]
    for tag in tags[key]:
        if dpg.does_item_exist(tag):
            value = dpg.get_value(tag)
            if isinstance(value, str) and value.strip(): texts.append(value)
    return '\n\n'.join(dict.fromkeys(texts))
