"""Compact, read-only Analysis results. Scientific evidence stays in Details."""
from dataclasses import dataclass

from pyecr.gui.theme import TOKENS, SECTION_FONT_TAG
from pyecr.gui.state import GuiPage
from pyecr.gui.workflow_display import brief_text
from pyecr.model_display import display_label


@dataclass(frozen=True)
class ParameterCheck:
    name: str
    status: str
    reason: str
    severity: str


@dataclass(frozen=True)
class DiagnosticsSummary:
    title: str
    severity: str
    checks: tuple[str, ...] = ()
    parameters: tuple[ParameterCheck, ...] = ()
    note: str = ''


def parameter_label(name):
    return name.removesuffix('_cm2_s').removesuffix('_cm_s')


def diagnostics_summary(presentation, *, stale=False):
    """Describe each parameter's own checks, separately from CI assumptions."""
    if stale:
        return DiagnosticsSummary('Out of date', 'warning', note='Run Diagnostics for the current Fit.')
    if presentation is None:
        return DiagnosticsSummary('Not run', 'muted', note='Run Diagnostics to check this Fit.')
    e = getattr(presentation, 'evaluation', None)
    if e is None:
        rows = tuple(ParameterCheck(r.name, 'Pass' if r.reliable else 'Review',
            '' if r.reliable else brief_text(r.reliability_reason, 72),
            'success' if r.reliable else 'warning') for r in presentation.parameters)
        return DiagnosticsSummary('Checks passed' if presentation.reliable else 'Review required',
            'success' if presentation.reliable else 'warning', parameters=rows,
            note='Local 95% checks; see Details for assumptions.')
    rows = []
    for p in e.parameters:
        if p.precision == 'unavailable':
            status, reason, severity = 'N/A', 'Precision unavailable', 'warning'
        elif p.precision == 'poor':
            status, reason, severity = 'Fail', 'Weak precision', 'error'
        else:
            status, reason, severity = 'Pass', '', 'success'
        if p.boundary == 'outside':
            status, reason, severity = 'Fail', 'Outside bounds', 'error'
        elif p.boundary in {'contact', 'interval_crosses'} and status == 'Pass':
            status, reason, severity = 'Review', ('At bound' if p.boundary == 'contact' else 'CI reaches bound'), 'warning'
        elif p.boundary == 'unconfigured' and status == 'Pass':
            reason = 'Precision only; bounds unset'
        rows.append(ParameterCheck(p.name, status, reason, severity))
    failed = not e.point_estimates_usable or bool(e.blocking_reasons)
    qualified = e.status == 'supported'
    title = ('Fit not supported' if not e.point_estimates_usable else 'Parameter checks failed' if failed
             else 'Checks passed' if qualified else 'Usable · review cautions')
    severity = 'error' if failed else 'success' if qualified else 'warning'
    residual = {'not_detected': 'No dependence detected', 'warning': 'Dependence warning',
                'unassessed': 'Independence unassessed'}[e.residual.dependence]
    curve = {'supported': 'Pass', 'caution': 'Review', 'poor': 'Fail', 'unavailable': 'N/A'}[e.adequacy_status]
    precision = {'supported': 'Pass', 'caution': 'Review', 'weak': 'Weak', 'unavailable': 'N/A'}[e.identifiability_status]
    note = {'conditional_iid': 'Local 95% CI: conditional on independent errors.',
            'assumptions_unverified': 'Local 95% CI: error assumptions unverified.',
            'boundary_limited': 'Local 95% CI: limited by bounds.',
            'unavailable': 'Local 95% CI: unavailable.'}[e.interval_status]
    return DiagnosticsSummary(title, severity, (f'Curve: {curve} · Precision: {precision}', residual), tuple(rows), note)


def _color(severity):
    return TOKENS.steps.secondary if severity == 'muted' else getattr(TOKENS.colors, severity)


class AnalysisResultsMixin:
    def _result_heading(self, text, **kwargs):
        d = self._require_dpg()
        item = d.add_text(text, **kwargs)
        if d.does_item_exist(SECTION_FONT_TAG):
            d.bind_item_font(item, SECTION_FONT_TAG)
        return item

    def _build_result_summary(self):
        d = self._require_dpg()
        with d.theme(tag='pyecr_result_table_theme'):
            with d.theme_component(d.mvAll):
                d.add_theme_style(d.mvStyleVar_CellPadding,4,1)
                d.add_theme_style(d.mvStyleVar_ItemSpacing,4,2)
        original_fit_items = list(d.get_item_children('pyecr_fit_results_group', 1))
        first = original_fit_items[0]
        with d.group(parent='pyecr_fit_results_group', before=first, tag='pyecr_result_fit'):
            self._result_heading('Fit', tag='pyecr_result_fit_title')
            self._result_heading('', tag='pyecr_result_fit_metric', color=TOKENS.colors.csv)
            d.add_text('', tag='pyecr_result_fit_status', wrap=280)
            d.add_text('', tag='pyecr_result_fit_note', wrap=280, color=TOKENS.steps.secondary, show=False)
            d.add_group(tag='pyecr_result_fit_parameters')
        with d.collapsing_header(label='Fit details', parent='pyecr_fit_results_group',
                                 before=first, tag='pyecr_result_fit_details', default_open=False):
            pass
        for tag in ('pyecr_fit_result_heading', 'pyecr_fit_summary', 'pyecr_fit_context_summary',
                    'pyecr_fit_usage_help', 'pyecr_fit_reliability', 'pyecr_fit_warnings',
                    'pyecr_fit_parameter_container'):
            d.move_item(tag, parent='pyecr_result_fit_details')
        for item in original_fit_items:
            if d.get_item_alias(item) != 'pyecr_transport_region' and d.get_item_parent(item) in (
                    'pyecr_fit_results_group', d.get_alias_id('pyecr_fit_results_group')):
                d.move_item(item, parent='pyecr_result_fit_details')
        # Transport follows the fit rather than displacing the primary result.
        d.move_item('pyecr_transport_region', parent='pyecr_analysis_right')
        children = list(d.get_item_children('pyecr_diagnostic_results_group', 1))
        with d.group(parent='pyecr_diagnostic_results_group', before=children[0], tag='pyecr_result_diagnostics'):
            d.add_spacer(height=8); d.add_separator(); d.add_spacer(height=4)
            self._result_heading('Diagnostics')
            self._result_heading('', tag='pyecr_result_diagnostic_conclusion', wrap=280)
            d.add_text('', tag='pyecr_result_diagnostic_checks', wrap=280)
            d.add_text('', tag='pyecr_result_diagnostic_count')
            d.add_group(tag='pyecr_result_diagnostic_parameters')
            d.add_text('', tag='pyecr_result_diagnostic_note', wrap=280, color=TOKENS.steps.secondary)
        with d.collapsing_header(label='Diagnostic details', parent='pyecr_diagnostic_results_group',
                                 tag='pyecr_result_diagnostic_details', default_open=False):
            pass
        for item in children:
            d.move_item(item, parent='pyecr_result_diagnostic_details')
        # Full post-stop metrics remain copyable without repeating the result rail.
        d.move_item('pyecr_noise_status_advanced', parent='pyecr_step_advanced_details')
        d.move_item('pyecr_step_advanced_warning', parent='pyecr_step_advanced_details')
        d.add_text('', tag='pyecr_diagnostic_step_note', parent='pyecr_stage_advanced_content',
                   before='pyecr_step_advanced_details', wrap=280, color=TOKENS.steps.secondary)
        self._result_summary_ready = True

    def _result_parameter_table(self, params, unit):
        d = self._require_dpg()
        with d.table(parent='pyecr_result_fit_parameters', header_row=False,
                     policy=d.mvTable_SizingStretchProp, width=-1, pad_outerX=False) as table:
            d.add_table_column(init_width_or_weight=.8); d.add_table_column(init_width_or_weight=1.6)
            for name, value in params.items():
                units = 'cm²/s' if name.startswith('D_') else 'cm/s' if name.startswith('k_') else unit
                with d.table_row():
                    d.add_text(parameter_label(name), color=TOKENS.steps.secondary)
                    d.add_text(f'{value:.5g} {units}', color=TOKENS.steps.text)
        d.bind_item_theme(table, 'pyecr_result_table_theme')

    def _refresh_result_summary(self):
        if not getattr(self, '_result_summary_ready', False):
            return
        d = self._require_dpg(); s = self.controller.analysis_session
        fit = s.fit_result
        presentation = s.diagnostics if not s.diagnostics_stale else None
        stale = bool(s.fit_stale or (s.diagnostics is not None and s.diagnostics_stale))
        view = diagnostics_summary(presentation, stale=stale)
        fit_task=self.controller.state.tasks[GuiPage.FIT]
        candidate = self.controller.current_fit_candidate
        live = self.controller.live_fit_is_current and (fit_task.status in {'running', 'cancelled'}
            or (candidate is not None and fit_task.status in {'cancelled', 'error', 'warning'}))
        snapshot = fit_task.fit_progress
        counts = None if snapshot is None else (snapshot.phase, snapshot.start_index,
                                                snapshot.total_starts, snapshot.completed_starts)
        key = (id(s), id(fit), id(s.diagnostics), s.fit_stale, s.diagnostics_stale,
               fit_task.status, fit_task.task_id, live, id(candidate), counts,
               fit_task.cancellation_requested)
        if key != getattr(self, '_result_summary_key', None):
            d.delete_item('pyecr_result_fit_parameters', children_only=True)
            unit = '' if s.dataset is None else str(s.dataset.metadata.get('signal_unit', ''))
            d.set_value('pyecr_result_fit_title', 'Fit · current best' if live else 'Fit')
            d.configure_item('pyecr_result_fit_metric', color=TOKENS.colors.warning if live else TOKENS.colors.csv)
            d.configure_item('pyecr_result_fit_note', show=live)
            d.configure_item('pyecr_result_fit_details', show=fit is not None,
                             label='Previous completed Fit' if live else 'Fit details')
            if live:
                state = ('Cancelling…' if fit_task.cancellation_requested else 'Running…') if fit_task.status == 'running' else (
                    'Cancelled · best completed candidate' if fit_task.status == 'cancelled' else 'Stopped · best completed candidate')
                if candidate is None:
                    d.set_value('pyecr_result_fit_metric', 'Waiting for first candidate' if fit_task.status == 'running' else 'No completed candidate')
                    d.set_value('pyecr_result_fit_status', state)
                else:
                    d.set_value('pyecr_result_fit_metric', f'RMSE  {candidate.rmse:.4g} {unit}')
                    model = display_label(candidate.model_id)
                    starts = (f' · {snapshot.completed_starts or 0}/{snapshot.total_starts} starts'
                              if snapshot is not None and snapshot.total_starts else '')
                    if s.current_analysis_config.model_id.value != candidate.model_id:
                        model += ' · preparation stage'
                    d.set_value('pyecr_result_fit_status',
                                f'{state}{starts}\n{model}\nSSE {candidate.raw_sse:.4g} · best start #{candidate.start_index}')
                    self._result_parameter_table(dict(candidate.params), unit)
                d.set_value('pyecr_result_fit_note', 'Updates after each completed start. Candidate only; Diagnostics not run.')
            elif fit is not None:
                d.set_value('pyecr_result_fit_metric', f'RMSE  {fit.RMSE:.4g} {unit}')
                status = 'Out of date' if s.fit_stale else 'Completed' if fit.exit_status > 0 else 'Stopped · review required'
                if not s.fit_stale and fit_task.status in {'running','error','cancelled'}:
                    status={'running':'Retained Fit · new fit running','error':'Retained Fit · last attempt failed',
                            'cancelled':'Retained Fit · last attempt cancelled'}[fit_task.status]
                d.set_value('pyecr_result_fit_status', f'{status}\nR² {fit.R2:.9g} · n={len(fit.residuals)}')
                self._result_parameter_table(fit.params, unit)
            else:
                d.set_value('pyecr_result_fit_metric', '')
                d.set_value('pyecr_result_fit_status', 'No Fit result.')
            d.set_value('pyecr_result_diagnostic_conclusion', view.title)
            d.configure_item('pyecr_result_diagnostic_conclusion', color=_color(view.severity))
            d.set_value('pyecr_result_diagnostic_checks', '\n'.join(view.checks))
            passed = sum(p.status == 'Pass' for p in view.parameters)
            d.set_value('pyecr_result_diagnostic_count', f'Parameter checks: {passed}/{len(view.parameters)} passed' if view.parameters else '')
            d.delete_item('pyecr_result_diagnostic_parameters', children_only=True)
            if view.parameters:
                with d.table(parent='pyecr_result_diagnostic_parameters', header_row=False,
                             policy=d.mvTable_SizingStretchProp, width=-1, pad_outerX=False) as table:
                    d.add_table_column(init_width_or_weight=.8); d.add_table_column(init_width_or_weight=1.6)
                    for row in view.parameters:
                        with d.table_row():
                            d.add_text(parameter_label(row.name))
                            item = d.add_text(row.status + (' · ' + row.reason if row.reason else ''),
                                              color=_color(row.severity), wrap=190)
                            with d.tooltip(item):
                                d.add_text(row.name + ': ' + (row.reason or 'Local precision and configured bounds passed.'), wrap=300)
                d.bind_item_theme(table,'pyecr_result_table_theme')
            d.set_value('pyecr_result_diagnostic_note', view.note)
            self._result_summary_key = key
        # Hide previous scientific checks while the rail displays another
        # candidate. Published Fit/Diagnostics remain available in their own
        # state and cannot be replaced by this display-only endpoint.
        d.configure_item('pyecr_fit_results_group', show=live or fit is not None)
        if live:
            d.configure_item('pyecr_diagnostic_results_group', show=False)
            d.configure_item('pyecr_transport_region', show=False)
        task = self.controller.state.tasks[GuiPage.DIAGNOSTICS]
        running = task.status == 'running'
        status = 'Checking…' if running else view.title if presentation is not None or stale else 'Check fit reliability.'
        if task.status in {'error', 'failed', 'cancelled'}:status = 'Cancelled' if task.status=='cancelled' else 'Diagnostics failed'
        d.set_value('pyecr_step_advanced_status', status)
        d.configure_item('pyecr_step_advanced_status', color=_color('error' if task.status in {'error','failed'} else view.severity))
        note = 'Parameter checks are in Results.' if view.parameters else view.note
        if task.status in {'error','failed','cancelled'}:note = brief_text(task.message, 100)
        d.set_value('pyecr_diagnostic_step_note', note)
        for tag in ('pyecr_stage_advanced_reason', 'pyecr_diagnostics_cancel_reason'):
            d.configure_item(tag, show=False)
        if d.get_value('pyecr_result_diagnostic_details'):
            for tag in ('pyecr_diagnostics_jacobian','pyecr_diagnostics_bi_container',
                        'pyecr_diagnostics_parameter_container','pyecr_diagnostics_correlation_status',
                        'pyecr_diagnostics_correlation_container','pyecr_diagnostics_multistart_status',
                        'pyecr_diagnostics_multistart_container','pyecr_reliability_evidence'):
                self._set_disclosed_item(tag,True)
        width = max(150, int(d.get_item_rect_size('pyecr_analysis_right')[0] or 320) - 30)
        for tag in ('pyecr_result_fit_metric', 'pyecr_result_fit_status', 'pyecr_result_fit_note', 'pyecr_result_diagnostic_conclusion',
                    'pyecr_result_diagnostic_checks', 'pyecr_result_diagnostic_note', 'pyecr_transport_status'):
            d.configure_item(tag, wrap=width)
