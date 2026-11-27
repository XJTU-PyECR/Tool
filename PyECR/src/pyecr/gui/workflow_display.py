"""One presentation flow and read-only disclosure policy; no scientific actions."""
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from re import fullmatch
from pyecr.gui.state import GuiPage
from pyecr.services.workflow import WorkflowState
from pyecr.diagnostics.interpretation import field, reproducibility_text
from pyecr.model_display import display_model_text, display_fit_worker_text

VERSION = 'workflow-display/1.0.0'
PAGE_STATUS_VERSION = 'page-status/1.1.0'
PAGE_TASKS = MappingProxyType({
    'analysis': (GuiPage.PROJECT, GuiPage.DATA, GuiPage.MODEL, GuiPage.FIT, GuiPage.DIAGNOSTICS),
    'batch': (GuiPage.BATCH,), 'pair': (GuiPage.PAIR,), 'dct': (GuiPage.DCT,), 'reports': (GuiPage.REPORT,),
    'simulation': (GuiPage.SIMULATION,),
})


def page_owner(page):
    return next(name for name, pages in PAGE_TASKS.items() if page in pages)


def task_status_text(page, task):
    """Terminal state wins over a retained cancellation request flag."""
    if task.status == 'running' and task.cancellation_requested:
        return f'{page.value}: cancellation requested; waiting for the worker to stop.'
    message = display_fit_worker_text(task.message) if page is GuiPage.FIT else display_model_text(task.message)
    return f'{page.value} · {task.status}: {message}'


def brief_text(text, limit=110):
    value = ' '.join(display_model_text(text).split())
    return value if len(value) <= limit else value[:limit-1] + '…'


class InformationLevel(IntEnum):
    DEFAULT = 0
    DETAILS = 1
    EXPERT = 2


LEVEL_LABELS = MappingProxyType({InformationLevel.DEFAULT: 'Default', InformationLevel.DETAILS: 'Details', InformationLevel.EXPERT: 'Advanced'})


@dataclass(frozen=True)
class WorkflowStep:
    key: str
    title: str
    aliases: tuple[str, ...]
    next_states: tuple[WorkflowState, ...]
    prerequisite: str
    help_text: str


WORKFLOW_STEPS = (
    WorkflowStep('data', 'Import', ('import', GuiPage.DATA.value, GuiPage.PROJECT.value), (WorkflowState.EMPTY,), 'none',
                 'Choose a raw file, review the column mapping and confirm units.'),
    WorkflowStep('preprocess', 'Preprocess', ('preprocessing',), (WorkflowState.IMPORTED,), 'imported',
                 'Review raw data; preview and accept only intentional processing. Raw remains immutable.'),
    WorkflowStep('model', 'Model & experiment', ('boundary', GuiPage.MODEL.value), (WorkflowState.PREPROCESSED,), 'imported',
                 'Select the model, complete dimensions, active faces, free/fixed parameters and boundary conditions; apply configuration.'),
    WorkflowStep('fit', 'Run fit', (GuiPage.FIT.value,), (WorkflowState.CONFIGURED,), 'configured',
                 'Fit every eligible observation using the applied configuration. Cancel remains available while running.'),
    WorkflowStep('advanced', 'Diagnostics', ('diagnostics', 'decomposition', GuiPage.DIAGNOSTICS.value), (WorkflowState.FITTED,), 'fitted',
                 'Review reliability and residuals; inspect continuous signed contributions for a supported joint dual fit.'),
    WorkflowStep('export', 'Export', ('reports', GuiPage.REPORT.value), (WorkflowState.DIAGNOSED,), 'fitted',
                 'Choose a preset in Reports; export a current captured result, or explicitly select a historical snapshot.'),
)
COMPATIBILITY = MappingProxyType({name.lower(): step.key for step in WORKFLOW_STEPS for name in (step.key, *step.aliases)})


def step_key(name):
    return COMPATIBILITY[str(getattr(name, 'value', name)).lower()]


def step_labels():
    return tuple(f'{n:02d}  {step.title}' for n, step in enumerate(WORKFLOW_STEPS, 1))


def step_number(name):
    return next(n for n,s in enumerate(WORKFLOW_STEPS,1) if s.key==step_key(name))


def flow_help():
    return '\n'.join(f'{label}: {step.help_text}' for label, step in zip(step_labels(), WORKFLOW_STEPS))


def next_step_text(state):
    step = next(s for s in WORKFLOW_STEPS if state in s.next_states)
    number = WORKFLOW_STEPS.index(step) + 1
    return f'Next · {number} {step.title}: {step.help_text}'


def step_available(step, state):
    imported = state is not WorkflowState.EMPTY
    configured = state in {WorkflowState.CONFIGURED, WorkflowState.FITTED, WorkflowState.DIAGNOSED}
    fitted = state in {WorkflowState.FITTED, WorkflowState.DIAGNOSED}
    return {'none': True, 'imported': imported, 'configured': configured, 'fitted': fitted}[step.prerequisite]


def legacy_stage_visibility(state, dual_model=False):
    """Old P30 caller contract; numbered UI uses WORKFLOW_STEPS instead."""
    result = {s.key: step_available(s, state) for s in WORKFLOW_STEPS if s.key != 'export'}
    result['boundary'] = state is not WorkflowState.EMPTY and not dual_model
    return {key: result[key] for key in ('data', 'preprocess', 'model', 'boundary', 'fit', 'advanced')}


def boundary_support_message(descriptor):
    if not descriptor.factory_supported:
        return display_model_text(f'Unsupported: {descriptor.model_id.value}: {descriptor.factory_unavailable_reason}')
    unsupported = [mode for mode in ('ideal_step', 'exponential_flush', 'measured_history') if mode not in descriptor.supported_boundary_modes]
    return display_model_text(f"{descriptor.model_id.value} boundary: {', '.join(descriptor.supported_boundary_modes)}. Unsupported: {', '.join(unsupported)}."
                              if unsupported else '')


def compact_fit_summary(presentation, fit, stale=False, config=None, diagnostics=None):
    if stale:
        return 'Out of date Fit: retained historical result; refit before current diagnostics/export. ' + reproducibility_text(fit)
    if presentation is None or fit is None:
        return ''
    evaluation=field(diagnostics,'evaluation') if diagnostics is not None else None
    text = 'Optimizer: ' + ('completed' if field(fit,'exit_status',0)>0 else 'success not reported')
    text += (' | Parameter precision: '+field(evaluation,'identifiability_status','unavailable') if evaluation is not None
             else f' | Local checks: {"passed" if getattr(presentation,"reliable",False) else "warning"}')
    if diagnostics is None:
        text += '\nDiagnostics: not run; review step 5'
    else:
        advisor=field(diagnostics,'reliability_advisor',{})
        text += f'\nObserved fit adequacy: {field(evaluation,"adequacy_status",field(advisor,"model_adequacy_status","unavailable"))}'
        text += f' | Identifiability: {field(evaluation,"identifiability_status",field(advisor,"identifiability_status","unavailable"))}'
        if evaluation is not None:
            text += '\nPoint estimates: '+('usable' if field(evaluation,'point_estimates_usable',False) else 'not supported')
            text += ' | Local CI: '+field(evaluation,'interval_status','unavailable')
        else:
            text += '\nCredibility gates: '+('passed' if field(diagnostics,'reliable',False) else 'warning; review step 5')
    text += '\nCurrent Fit | '+reproducibility_text(fit)
    text += f'\nRMSE={presentation.RMSE:.4e} | R²={presentation.R2:.4e}'
    record = fit.diagnostics.get('fixed_d', fit.diagnostics.get('dual_surface_k', fit.diagnostics.get('surface_k', {})))
    text += f" | free={record.get('free_parameter_count', len(fit.params))}"
    fixed={**({} if config is None else dict(config.model.fixed_parameters)),**dict(record.get('fixed_parameters',{}))}
    if fixed:text+=' | fixed: '+', '.join(f'{n}={v:.4e}' for n,v in fixed.items())
    text += '\nCI: conditional on fixed inputs; method / assumptions in Status details'
    if 'fixed_d' in fit.diagnostics:
        text += '; fixed D source uncertainty is not propagated'
    return text


def issue_lines(state, session, descriptor=None, quality=None, *, pages=None):
    """Read-only, optionally page-owned context. None preserves the legacy API."""
    pages = frozenset(state.tasks) if pages is None else frozenset(pages)
    analysis = GuiPage.FIT in pages
    lines = []
    for page, error in state.current_errors.items():
        if page in pages: lines.append(f'Error · {page.value}: {error.exception_type}: {error.message}')
    if analysis and session.fit_result is not None and session.fit_stale:
        lines.append('Out of date Fit: current diagnostics/export are disabled; retained result is historical.')
    if analysis and session.diagnostics is not None and session.diagnostics_stale:
        lines.append('Out of date Diagnostics: rerun for the current Fit.')
    for page, task in state.tasks.items():
        if page in pages and task.status in {'running', 'error', 'cancelled'}:
            lines.append(task_status_text(page, task))
    d = state.diagnostics_presentation if analysis else None
    if d is not None:
        lines.extend(f'Reliability warning (Diagnostics): {r}' for r in d.gate_failures)
    p = state.fit_presentation if analysis else None
    current_evaluation = (getattr(d, 'evaluation', None) if d is session.diagnostics
                          and not session.diagnostics_stale and not session.fit_stale else None)
    if p is not None:
        if current_evaluation is None and not p.reliable: lines.append(f'Fit reliability warning: {p.reliability_label}')
        lines.extend(f'Fit warning: {w}' for w in p.warnings)
        for row in (() if current_evaluation is not None else p.parameters):
            if not row.reliability_flag: lines.append(f'{row.name} ({row.unit}) warning: {row.reliability_reason}')
    if analysis and descriptor is not None:
        config=session.current_analysis_config
        if not descriptor.factory_supported:
            lines.append(f'Unsupported model: {descriptor.factory_unavailable_reason}')
        elif config is not None and config.model_id == descriptor.model_id and config.model.boundary_mode not in descriptor.supported_boundary_modes:
            lines.append(f'Unsupported boundary: {config.model.boundary_mode}')
    if analysis and quality is not None:
        lines.extend(f'Fit blocked: {r}' for r in quality.blocking_reasons)
        lines.extend(f'Data warning: {r}' for r in quality.warnings)
    if d is not None:
        shared=set(() if p is None else p.warnings) | set(d.gate_failures)
        lines.extend(f'Diagnostics warning: {w}' for w in d.warnings if w not in shared)
    if analysis and session.fit_result is not None:
        from pyecr.fitting.reproducibility import reproducibility_issue
        if hasattr(session.fit_result, 'diagnostics'):
            problem = reproducibility_issue(session.fit_result)
            if problem: lines.append(problem)
    for page,task in state.tasks.items():
        if page in pages and task.status=='warning' and not (task.message.startswith('Restored ') and d is not None and d.gate_failures):
            lines.append(f'{page.value} · warning: {task.message}')
    return tuple(dict.fromkeys(line for line in lines if line))


def recovery_hint(owner, lines):
    """Existing page/step destinations, without changing any gate or action."""
    text = '\n'.join(lines).lower()
    if owner == 'analysis':
        hints = []
        if any(word in text for word in ('column mapping','unit warning','import a dataset')):
            hints.append('Review step 1 Import: mapping and units.')
        if any(word in text for word in ('negative time','negative pre','fit interval','preprocessing')):
            hints.append('Review step 2 Preprocess: t0 / eligible interval and row history.')
        if any(word in text for word in ('unsupported model','unsupported boundary','configuration','fixed d','unapplied')):
            hints.append('Review step 3 Model & experiment: apply intended inputs and check supported boundary / fixed sources.')
        if any(word in text for word in ('credibility','reliability','identifiability','parameter_at_bound')):
            hints.append('Review step 5 Diagnostics: contact, proximity, residuals and parameter evidence; a warning is not an optimizer crash.')
        return ' '.join(hints)
    return {
        'batch': 'Review inputs, mode and applied template; inspect every excluded / failed record before Arrhenius interpretation.',
        'pair': 'Review A/B captured sources and compatibility; planning remains a separate conditional scan.',
        'dct': 'Review source, tau grid and GCV / Manual; peaks are candidate time scales, not assigned mechanisms.',
        'reports': 'Review Current / Historical source and output location; current stale fits require rerun, historical export uses captured evidence.',
        'simulation': 'Review the independent given-parameter draft; Generate freezes it. Cancel blocks publication; no Analysis Apply or uncertainty claim.',
    }[owner]


def page_status(state, session, owner, lines=(), quality=None):
    """Short status next to actions; complete reasons remain in page details."""
    # Among equally terminal tasks, prefer the downstream action/result.
    tasks = [(page, state.tasks[page]) for page in reversed(PAGE_TASKS[owner])]
    priority = {'running': 0, 'error': 1, 'cancelled': 2, 'warning': 3, 'completed': 4, 'idle': 5}
    page, task = min(tasks, key=lambda pair: priority[pair[1].status])
    if task.status != 'idle':
        text, severity = brief_text(task_status_text(page, task)), task.status
        if owner == 'analysis' and session.fit_result is not None and task.status in {'running','error','cancelled'}:
            text += '\nRetained previous Fit; latest attempt has not published a new result.'
    else:
        defaults = {'analysis': 'Import data to begin.', 'batch': 'Add inputs and choose a Batch mode.',
                    'pair': 'Capture or load A/B; choose comparison or planning.', 'dct': 'Import a source for whole DCT. Fit is optional.',
                    'reports': 'Capture a current result or choose an explicit historical snapshot.',
                    'simulation': 'Review given parameters and explicitly Generate.'}
        text, severity = defaults[owner], 'idle'
        if owner == 'analysis' and session.dataset is not None:
            if session.current_analysis_config is None: text = 'Apply model configuration before Fit.'
            elif session.fit_result is not None: text = 'Fit result loaded; review diagnostics and export.'
            elif quality is not None and not quality.ready: text = 'Fit blocked: ' + brief_text('; '.join(quality.blocking_reasons), 90)
            else: text = 'Ready for Fit.'
    flags = []
    all_text = '\n'.join(lines).lower()
    if any(page in state.current_errors or state.tasks[page].status == 'error' for page in PAGE_TASKS[owner]): flags.append('Error')
    # Static instructions such as "Out of date export is disabled" are not a
    # declaration that the current result is stale.
    if any(line.lower().startswith('out of date') for line in lines):
        flags.append('Out of date' if session.fit_stale or session.diagnostics_stale else 'Out of date feature (details)')
    for word, label in (('reliability', 'Reliability warning'), ('unit warning', 'Units unconfirmed')):
        if word in all_text: flags.append(label)
    if any(line.lower().startswith(('unsupported', 'applied unsupported')) for line in lines):
        flags.append('Unsupported selection (details)')
    if 'legacy fit lacks random-seed' in all_text or 'reproducible snapshot unavailable' in all_text:
        flags.append('Reproducibility unavailable; rerun Fit')
    for word, label in (('failed', 'Failed (details)'), ('cancelled', 'Cancelled (details)')):
        # A failed scientific reliability gate is already a reliability warning,
        # not a failed task. Identify explicit feature outcomes only.
        if any(line.lower().startswith((word, 'profile '+word, 'bootstrap '+word)) for line in lines) and word not in text.lower(): flags.append(label)
    if any('warning' in line.lower() for line in lines) and 'Reliability warning' not in flags: flags.append('Warning')
    if flags:
        text += '\n' + ' · '.join(flags) + ' — open Status details'
        severity = 'error' if 'Error' in flags else 'warning' if severity != 'error' else severity
    return text, severity


DETAIL_TAGS = frozenset({
    'pyecr_fit_reliability', 'pyecr_fit_context_summary', 'pyecr_fit_usage_help', 'pyecr_activity_history', 'pyecr_history_group',
    'pyecr_fit_progress_details_header', 'pyecr_fit_start_note', 'pyecr_data_processing_history',
    'pyecr_diagnostics_jacobian', 'pyecr_diagnostics_bi_container', 'pyecr_diagnostics_parameter_container',
    'pyecr_diagnostics_correlation_status', 'pyecr_diagnostics_correlation_container',
    'pyecr_diagnostics_multistart_status', 'pyecr_diagnostics_multistart_container',
    'pyecr_reliability_evidence', 'pyecr_model_assumptions', 'pyecr_model_limitations',
})
EXPERT_TAGS = frozenset({
    'pyecr_model_numerical_controls', 'pyecr_solver_details_header', 'pyecr_developer_traceback_header',
    'pyecr_experimental_m1_controls', 'pyecr_diagnostics_bootstrap', 'pyecr_diagnostics_profile',
    'pyecr_diagnostics_experimental_m1', 'pyecr_diagnostics_bootstrap_status', 'pyecr_diagnostics_profile_status',
    'pyecr_diagnostics_experimental_m1_status', 'pyecr_diagnostics_experimental_m1_evidence', 'pyecr_guided_refit_controls',
    'pyecr_robustness_controls',
})
EMPTY_TEXT_TAGS = frozenset({
    'pyecr_fit_reliability', 'pyecr_fit_summary', 'pyecr_fit_warnings', 'pyecr_fit_quality_warnings',
    'pyecr_fit_blocking_reasons', 'pyecr_fit_disabled_reason', 'pyecr_fit_live_status', 'pyecr_current_task_error',
    'pyecr_current_warning', 'pyecr_activity_history', 'pyecr_diagnostics_reliability', 'pyecr_diagnostics_warnings',
    'pyecr_diagnostics_residual_summary', 'pyecr_diagnostics_information_criteria', 'pyecr_diagnostics_jacobian',
    'pyecr_diagnostics_bi_status', 'pyecr_diagnostics_parameter_status', 'pyecr_diagnostics_correlation_status',
    'pyecr_diagnostics_multistart_status', 'pyecr_diagnostics_error_map_status', 'pyecr_diagnostics_dual_map_status',
    'pyecr_diagnostics_error_map_warnings', 'pyecr_diagnostics_dual_map_warnings',
    'pyecr_reliability_identifiability', 'pyecr_reliability_adequacy', 'pyecr_reliability_bi_guidance',
    'pyecr_reliability_evidence', 'pyecr_guided_refit_summary', 'pyecr_diagnostics_experimental_m1_evidence',
    'pyecr_analysis_report_reason',
    'pyecr_dct_summary', 'pyecr_batch_result_summary', 'pyecr_report_snapshot_status',
    'pyecr_report_artifacts',
})
EMPTY_MESSAGES = ('warnings: none', 'blocking reasons: none', 'no fit result', 'no fit warnings', 'no diagnostics result',
                  'no diagnostics warnings', 'no current task error', 'no current warning', 'no warnings or errors',
                  'no error map warnings', 'no h/o error map warnings', 'no reliability advisor evidence',
                  'no experimental result', 'no live telemetry: idle', 'no guided refit comparison',
                  'no dct result', 'no current dct result', 'no analysissnapshot captured', 'no report exported',
                  'run the selected mode to produce a result; no published result')


def text_has_content(text):
    if not isinstance(text, str): return True
    # Inspect each line: an empty-state prefix must never conceal a later failure.
    for line in text.splitlines():
        value=line.strip().lower()
        if not value:continue
        benign=(any(value.startswith(empty) for empty in EMPTY_MESSAGES)
                or value.endswith('diagnostics have not run.')
                or fullmatch(r'[^:]+: not run\.?',value) is not None)
        if not benign:return True
        if any(word in value for word in ('failed','unsupported','out of date','stale','invalid','blocked','cancelled','warning:')):
            return True
    return False


def disclosure_visible(tag, level, value=None):
    if tag.startswith('pyecr_parameter_') and level==InformationLevel.DEFAULT and value in {'Bound hit: no','Reliability: reliable','Numerical contact: no','Local parameter checks: passed; see Diagnostics gates'}:
        return False
    minimum = InformationLevel.EXPERT if tag in EXPERT_TAGS else InformationLevel.DETAILS if tag in DETAIL_TAGS else InformationLevel.DEFAULT
    if level < minimum: return False
    return tag not in EMPTY_TEXT_TAGS or text_has_content(value)
