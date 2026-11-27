"""Read-only Batch projections. Counts refer to published scientific objects."""
from pyecr.services.batch_analysis import BatchAnalysisResult, GlobalAnalysisResult
from pyecr.model_display import display_name

VERSION = 'batch-display/1.0.0'
BATCH_STEPS = (
    ('inputs', '01 Inputs'),
    ('method', '02 Method & template'),
    ('run', '03 Run'),
    ('results', '04 Results & export'),
)
MODE_TABS = {'pyecr_batch_independent_tab': 'independent', 'pyecr_batch_global_tab': 'global'}


def global_kinetic_text(value):
    dual = ('k_H_cm_s' in value.shared_params if isinstance(value, GlobalAnalysisResult)
            else value is not None and value.model_id.value == 'M3_DUAL_HO_DK')
    return 'D_H, D_O, k_H, k_O' if dual else 'D + k'


def template_name(template):
    if template is None:return 'No applied template'
    return {'M0_SINGLE_D':'Single diffusion (D)','M7_SINGLE_K':'Single surface exchange (k)',
        'M1_SINGLE_DK':'Single diffusion + exchange (D + k)','M2_DUAL_HO_D':'Dual diffusion (D)',
        'M3_DUAL_HO_DK':'Dual diffusion + exchange (D + k)','M8_DUAL_K':'Dual surface exchange (k)'}.get(template.model_id.value,'Unsupported template')


def source_name(entry):
    from pathlib import PureWindowsPath
    source=entry.source_project or entry.dataset.metadata.get('source_path') or entry.dataset.metadata.get('dataset_name') or 'Current dataset'
    return PureWindowsPath(str(source)).name or str(source)


def parameter_text(params):
    values=[]
    for name,value in params.items():
        label=name.replace('_cm2_s','').replace('_cm_s','')
        unit=' cm²/s' if name.endswith('_cm2_s') else ' cm/s' if name.endswith('_cm_s') else ''
        values.append(f'{label} = {value:.6g}{unit}')
    return '\n'.join(values)


def result_counts(result, task_status):
    """Never convert an aborted joint job into invented per-record outcomes."""
    if isinstance(result, BatchAnalysisResult):
        success = sum(record.fit_success for record in result.records)
        total = len(result.records)
        scope = 'Published independent records'
    elif isinstance(result, GlobalAnalysisResult):
        success = int(result.fit_result.exit_status > 0)
        total = 1
        scope = 'Published Global joint fit (not independent curve outcomes)'
    else:
        success = total = 0
        scope = 'No published result'
    return {'scope': scope, 'success': success, 'failed': total-success,
            'denominator': total, 'published_cancelled': 0,
            'last_task_failed': int(task_status == 'error'),
            'last_task_cancelled': int(task_status == 'cancelled'),
            'per_record_cancelled': None}


def input_rows(entries, template, mode, result=None, task_status='idle'):
    independent = {r.record_id: r for r in result.records} if isinstance(result, BatchAnalysisResult) else {}
    global_ids = {c.curve_id for c in result.curves} if isinstance(result, GlobalAnalysisResult) else set()
    rows = []
    for entry in entries:
        meta = entry.dataset.metadata
        source = entry.source_project or meta.get('source_path') or meta.get('dataset_name') or 'current dataset'
        model = ('Global / '+display_name(template.model_id.value) if mode=='global' and template is not None
                 else 'Not configured' if template is None else template.model_id.value)
        record = independent.get(entry.record_id)
        if record is not None:
            state = 'Success' if record.fit_success else 'FAILED'
            if record.fit_success and not record.reliable:state += ' / credibility warning'
        elif entry.record_id in global_ids:
            state = 'Joint / row N/A'
        elif task_status == 'cancelled':
            state = 'Cancelled / row N/A'
        elif task_status == 'running':
            state = 'Running / pending'
        else:
            state = 'Awaiting published result'
        unit = getattr(entry.temperature_unit, 'value', entry.temperature_unit)
        source_model = None if entry.analysis_config is None else entry.analysis_config.model_id.value
        geometry_config = (entry.analysis_config or template) if mode=='global' else template
        geometry = ('unavailable' if geometry_config is None else
                    f'{geometry_config.model.geometry.full_dims_cm} cm; active pairs={geometry_config.model.geometry.active_dimensions}')
        context = (f'{source}\nRecord: {entry.record_id}\nTemperature: {entry.temperature:.17g} {unit}; '
                   f'{entry.temperature_K:.17g} K\nWeight: {entry.noise_weight:.17g}\n'
                   f'Execution model: {model}; stored source model: {source_model or "none"}\n'
                   f'Processed units: {meta.get("time_unit","s")}, {meta.get("signal_unit","unknown")}\n'
                   f'Captured Global flags: baseline={entry.fit_baseline}, amplitude={entry.fit_amplitude}\n'
                   f'Execution geometry: {geometry}\nStored curve baseline={entry.baseline}, amplitude={entry.amplitude}\n'
                   'Independent uses the shared template and log10-coordinate proximity gates for log parameters. '
                   'Diagnostics uses physical-coordinate proximity; different coordinate policies can differ. '
                   'Global uses each captured geometry and weight; its joint reliability is a separate test.')
        rows.append({'source': str(source), 'record_id': entry.record_id,
                     'temperature': f'{entry.temperature:.10g}', 'unit': str(unit),
                     'weight': f'{entry.noise_weight:.3e}', 'model': display_name(model), 'status': state,
                     'context': context})
    return tuple(rows)


def template_rows(template, descriptor, mode):
    if template is None or (mode=='global' and (template.model_id.value not in {'M1_SINGLE_DK','M3_DUAL_HO_DK'} or 'fixed_d' in template.model_specific)):return ()
    rows=[]
    for parameter in descriptor.parameters:
        name=parameter.name
        if mode=='global' and not name.startswith(('D_', 'k_')):continue
        if mode=='global':
            value=template.model.initial_values.get(name)
            state='joint free'
        elif name in template.model.fixed_parameters:
            value=template.model.fixed_parameters[name];state='fixed'
        else:
            value=template.model.initial_values.get(name)
            state='free' if name in template.model.free_parameters else 'model rule / unspecified'
        bounds=template.parameter_bounds.get(name)
        rows.append((name,'auto / not recorded' if value is None else f'{value:.10g}',parameter.unit,state,
                     'N/A' if bounds is None else f'{bounds[0]:.10g}', 'N/A' if bounds is None else f'{bounds[1]:.10g}'))
    return tuple(rows)


def result_rows(result):
    rows = []
    if isinstance(result, BatchAnalysisResult):
        for record in result.records:
            status = 'Success' if record.fit_success else 'FAILED'
            if record.fit_success and not record.reliable:status += ' / credibility warning'
            reason = record.error or '; '.join(record.reliability_reasons)
            if not record.fit_success and not reason:
                reason = ('Optimizer exit_status='+str(record.fit_result.exit_status)
                          if record.fit_result is not None else 'No FitResult; see stored evidence')
            params = ', '.join(f'{name}={value:.6g}' for name,value in record.params.items())
            rows.append((record.record_id, f'{record.temperature_K:.10g} K', status, params,
                         'N/A' if record.SSE is None else f'{record.SSE:.6g}', reason))
    elif isinstance(result, GlobalAnalysisResult):
        for curve in result.curves:
            amps = ', '.join(f'{name}={value:.6g}' for name,value in curve.component_amplitudes.items()) if curve.component_amplitudes else f'A={curve.amplitude}'
            params = f'y0={curve.baseline}; {amps}; weight={curve.noise_weight:.6g}'
            rows.append((curve.curve_id, 'Joint '+global_kinetic_text(result), 'Joint result', params, 'See joint SSE',
                         'Reliability established' if result.reliable else 'Reliability NOT established'))
    return tuple(rows)
