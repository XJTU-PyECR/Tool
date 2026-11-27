"""R13 adapters around existing results; no optimizer, CI estimator or new exporter.

An absent extension stays absent in legacy serialization. The read-only legacy
view explicitly labels its defaults and never recomputes historical endpoints.
"""
from collections.abc import Mapping
from dataclasses import replace
from dataclasses import asdict,fields
from html import escape
import json
from pyecr.config import _freeze_json
from pyecr.contracts import SourceIdentity,ReliabilityPolicy,TerminationPolicy,NoiseReference,NoiseComparison,TransportSummary
from pyecr.contracts.common import require,exact_fields,canonical_sha256,strict_json_loads
from pyecr.preprocessing import PreprocessConfig

VERSION='scientific-result-evidence/1'
HISTORY_KEY='r13_execution_history'


def plain(value):
    from pyecr.contracts.common import encode
    return encode(value)


def execution_history(records):
    """Independent attempts retain their own sources, including failed candidates."""
    value=dict(version='execution-history/1',records=plain(records))
    value['sha256']=canonical_sha256(value)
    return validate_execution_history(value)


def validate_execution_history(value):
    value=plain(value)
    exact_fields(value,('version','records','sha256'),'execution history')
    require(value['version']=='execution-history/1','Unknown execution history version')
    require(value['sha256']==canonical_sha256({k:v for k,v in value.items() if k!='sha256'}),'Execution history hash conflict')
    require(isinstance(value['records'],Mapping),'Execution history records must be a mapping')
    from pyecr.fitting.iteration_observation import read_observation
    from pyecr.fitting.monitored import read_monitored
    for job,record in value['records'].items():
        text=json.dumps(record,ensure_ascii=False,allow_nan=False)
        parsed=read_monitored(text) if record.get('mode')=='monitored' else read_observation(text)
        require(parsed['job_id']==job,'Execution history job identity conflict')
        require(parsed['state']!='running','Cannot persist a running attempt as a frozen result')
    return _freeze_json(value,path='execution_history')


def _source(snapshot,recipe):
    diag=snapshot.diagnostics_result
    return SourceIdentity.capture(snapshot.dataset,recipe,config=snapshot.analysis_config,fit=snapshot.fit_result,
        diagnostics=None if diag.get('available') is False else diag)


def _legacy_output(snapshot):
    from pyecr.report.bundle import _diagnostic_parameter_rows
    intervals={};reasons={};rows=_diagnostic_parameter_rows(snapshot)
    for n in snapshot.fit_result.params:
        row=rows.get(n,{})
        if row.get('ci_lower') is not None and row.get('ci_upper') is not None:
            intervals[n]=dict(lower=row['ci_lower'],upper=row['ci_upper'],confidence_level=.95)
        else:reasons[n]=row.get('ci_reason') or 'Historical interval not captured; no recomputation on read/export.'
    evaluation=snapshot.diagnostics_result.get('evaluation')
    if evaluation is not None:
        return dict(confidence_level=evaluation['policy']['confidence_level'],
            method='diagnostic_scaled_svd_student_t',df=evaluation['degrees_of_freedom'],
            N=len(snapshot.fit_result.y_fit),p_free=len(snapshot.fit_result.params),intervals=intervals,unavailable=reasons,
            level_origin='captured_diagnostics_v2',status='diagnostic_approximation',record=None,
            conditions=('Captured Diagnostics v2; local CI '+evaluation['interval_status']+
                '. Conditional on fixed inputs and local linearization; serial/robust coverage is not calibrated. No recalculation.'))
    return dict(confidence_level=.95,method='legacy_captured_method_not_recorded',df=None,
        N=len(snapshot.fit_result.y_fit),p_free=len(snapshot.fit_result.params),intervals=intervals,unavailable=reasons,
        level_origin='defaulted_legacy',status='captured_historical',record=None,
        conditions='Existing 95% endpoints and original Diagnostics preserved. Historical df/method detail is not asserted; no recalculation.')


def output_view(snapshot):
    return plain(snapshot.scientific_evidence['selected_output']) if snapshot.scientific_evidence else _legacy_output(snapshot)


def historical_output_notice(value):
    """Identify the exact earlier R13 declaration, without endorsing its method."""
    out=value.get('selected_output',{})
    if (out.get('record') is None and out.get('method')=='captured_local_covariance_student_t'
        and out.get('conditions')=='Existing 95% endpoints preserved. Historical df/method detail may be unavailable; no recalculation.'):
        return 'Historical interval method label is unverified; stored endpoints and source evidence were preserved without recomputation.'
    return ''


def _local_output(record):
    e=record.evidence
    return dict(confidence_level=e['confidence_level'],method=e['method'],df=e['df'],N=e['N'],p_free=e['p_free'],
        intervals=plain(e['intervals']),unavailable=plain(e['unavailable']),level_origin='explicit_computed_result',
        status=e['status'],record=record.to_dict(),conditions=plain(e['conditions']))


def capture_scientific(session,snapshot,*,execution=None,control_request=None):
    """Explicit snapshot action, reading committed results, never widget drafts."""
    from pyecr.services.local_intervals import LocalIntervalRecord
    from pyecr.services.transport_summary import summarize_session
    from pyecr.services.noise_reference import compare_current_fit
    from pyecr.services.advanced_intervals import advanced_to_dict,assert_current
    recipe=session.preprocessing.config;source=_source(snapshot,recipe)
    record=session.selected_local_intervals
    if record is not None:
        require(isinstance(record,LocalIntervalRecord),'Unknown selected output record')
        # Old records remain retained history, never silently re-labelled current.
        try:
            record.binding.assert_ci_source(source)
            require(record.fit_generation==session.fit_generation,'Stored CI generation is stale')
        except ValueError:record=None
    selected=_legacy_output(snapshot) if record is None else _local_output(record)
    ref=session.noise_reference
    noise=compare_current_fit(session,ref)
    transport=summarize_session(session,local_record=record)
    advanced={}
    for name,p in session.advanced_diagnostics.items():
        try:assert_current(p,session)
        except ValueError:continue
        advanced[name]=advanced_to_dict(p)
    actual_execution=execution if execution is not None else session.execution_evidence
    result=dict(version=VERSION,source=source.to_dict(),recipe=recipe.to_dict(),parameter_order=list(snapshot.fit_result.params),fit_generation=session.fit_generation,
        gate_fit_presentation=None if session.fit_presentation is None else plain(asdict(session.fit_presentation)),
        selected_output=selected,gate_policy=ReliabilityPolicy().to_dict(),
        termination=dict(mode='legacy',policy=TerminationPolicy.legacy_for(snapshot.analysis_config).to_dict(),
            policy_origin='captured_configuration; per-call overrides unavailable without trace',
            native_status=snapshot.fit_result.exit_status,native_success=snapshot.fit_result.exit_status>0,
            success_origin='native_exit_status_convention',final_polish='not_recorded',trace_summary=None,execution=None,
            control_request=plain(control_request)),
        noise=noise.to_dict(),transport=transport.to_dict(),advanced=advanced,
        defaults=dict(defaulted_legacy=record is None,selected_level_origin=selected['level_origin']),
        source_kind=str(snapshot.dataset.metadata.get('source_kind','experiment_or_user_source')))
    if actual_execution is not None:
        result['termination']=_execution_view(actual_execution,snapshot,source,control_request)
    validate_scientific(result,snapshot)
    return _freeze_json(result,path='scientific_evidence')


def _execution_view(value,snapshot,source,control_request):
    from pyecr.fitting.iteration_observation import read_observation
    from pyecr.fitting.monitored import read_monitored
    text=json.dumps(plain(value),ensure_ascii=False,allow_nan=False)
    value=read_monitored(text) if value.get('mode')=='monitored' else read_observation(text)
    require(value['state']=='completed' and value['final_source'] is not None,'A stopped candidate cannot substitute for this published Fit')
    final=SourceIdentity.from_dict(value['final_source'])
    require(replace(source,diagnostics_sha256=None)==final,'Execution final Fit identity conflict')
    require(tuple(value['final_residuals'])==tuple(snapshot.fit_result.residuals),'Execution full residual/Fit conflict')
    from pyecr.contracts import FitControlBinding
    binding=FitControlBinding.from_dict(value['binding']);binding.assert_execution_source(source)
    contexts=value['contexts'];ends=[c['accepted_trace']['records'][-1] for c in contexts if c['accepted_trace']['records']]
    return dict(mode='monitored' if value['mode']=='monitored' else 'legacy',policy=binding.controls.termination.to_dict(),
        policy_origin='actual_frozen_execution',native_status=snapshot.fit_result.exit_status,native_success=snapshot.fit_result.exit_status>0,
        success_origin='native_exit_status_and_validated_execution',final_polish=value.get('polish_used','not_requested'),
        trace_summary=dict(job_id=value['job_id'],counters=value['counters'],scope=value['counter_scope'],
            contexts=len(contexts),stored_accepted_records=sum(len(c['accepted_trace']['records']) for c in contexts),
            terminal_records=ends,warnings=value['warnings'],execution_outcome=value.get('execution_outcome')),
        execution=value,control_request=plain(control_request))


def validate_scientific(value,snapshot):
    """Strict extension reader. Existing method readers validate endpoints and traces."""
    value=plain(value)
    exact_fields(value,('version','source','recipe','parameter_order','fit_generation','gate_fit_presentation','selected_output','gate_policy','termination','noise','transport','advanced','defaults','source_kind'),'scientific evidence')
    require(value['version']==VERSION,'Unknown scientific evidence version')
    recipe=PreprocessConfig.from_dict(value['recipe']);source=_source(snapshot,recipe)
    source.assert_matches(SourceIdentity.from_dict(value['source']))
    require(tuple(value['parameter_order'])==tuple(snapshot.fit_result.params),'Jacobian parameter column identity conflict')
    from pyecr.contracts.common import integer
    integer(value['fit_generation'],'fit_generation')
    gate=ReliabilityPolicy.from_dict(value['gate_policy']);require(gate.confidence_level==.95,'Output CI cannot change gate')
    if value['gate_fit_presentation'] is not None:restore_gate_presentation(snapshot.dataset,snapshot.analysis_config,snapshot.fit_result,snapshot.diagnostics_result,value['gate_fit_presentation'])
    out=value['selected_output']
    exact_fields(out,('confidence_level','method','df','N','p_free','intervals','unavailable','level_origin','status','record','conditions'),'selected output')
    if out['record'] is not None:
        from pyecr.services.local_intervals import LocalIntervalRecord
        record=LocalIntervalRecord.from_dict(out['record']);record.binding.assert_ci_source(source)
        require(record.fit_generation==value['fit_generation'],'Selected CI Fit generation conflict')
        require(tuple(record.evidence['parameter_order'])==tuple(value['parameter_order']),'Stored CI/Jacobian coordinate order conflict')
        require(out==_local_output(record),'Selected output label/endpoints/method conflict')
        require(tuple(record.evidence['raw_indices'])==tuple(i for i,on in enumerate(snapshot.dataset.mask) if on),'CI all-row source conflict')
    else:
        expected=_legacy_output(snapshot)
        if historical_output_notice(value):
            # Compatibility is limited to the two exact historical declaration
            # strings. Every endpoint, C, N, order, source and generation still
            # passes its original check. Return the original frozen evidence.
            expected=dict(expected,method=out['method'],conditions=out['conditions'])
        require(out==expected,'Legacy endpoints/95% default conflict; explicit recomputation required')
    term=value['termination'];exact_fields(term,('mode','policy','policy_origin','native_status','native_success','success_origin','final_polish','trace_summary','execution','control_request'),'termination evidence')
    policy=TerminationPolicy.from_dict(term['policy'])
    require(policy.mode==term['mode'] and term['native_status']==snapshot.fit_result.exit_status and term['native_success']==(snapshot.fit_result.exit_status>0),'Native termination identity conflict')
    if term['control_request'] is not None:
        from pyecr.contracts import FitControlBinding
        FitControlBinding.from_dict(term['control_request']).assert_execution_source(source)
    if term['execution'] is not None:
        expected=_execution_view(term['execution'],snapshot,source,term['control_request'])
        require(term==expected,'Trace summary/actual execution conflict')
    else:
        require(term['mode']=='legacy' and policy==TerminationPolicy.legacy_for(snapshot.analysis_config) and term['trace_summary'] is None and term['final_polish']=='not_recorded','Missing actual execution evidence')
    noise=NoiseComparison.from_dict(value['noise']);replace(source,diagnostics_sha256=None).assert_matches(noise.source)
    require(tuple(noise.time)==tuple(snapshot.dataset.processed_time) and tuple(noise.raw_indices)==tuple(i for i,on in enumerate(snapshot.dataset.mask) if on),'Noise observation identity conflict')
    require(tuple(noise.residuals)==tuple(snapshot.fit_result.residuals) or tuple(noise.residuals)==tuple(-snapshot.fit_result.residuals),'Noise residual identity conflict')
    transport=TransportSummary.from_dict(value['transport']);source.assert_matches(transport.source)
    from pyecr.services.advanced_intervals import advanced_from_dict
    for p in value['advanced'].values():
        advanced=advanced_from_dict(p);require(advanced.binding is not None,'Advanced uncertainty lacks parent source')
        advanced.binding.assert_ci_source(source)
    canonical_sha256(value)
    return _freeze_json(value,path='scientific_evidence')


def parameter_output_rows(snapshot):
    """Overlay selected output only; keep gate credibility and its old endpoints separate."""
    from pyecr.report.bundle import _diagnostic_parameter_rows
    rows={n:dict(r) for n,r in _diagnostic_parameter_rows(snapshot).items()};out=output_view(snapshot)
    for n in snapshot.fit_result.params:
        row=rows.setdefault(n,{});interval=out['intervals'].get(n)
        row.update(ci_lower=None if interval is None else interval['lower'],ci_upper=None if interval is None else interval['upper'],
            ci_reason=out['unavailable'].get(n),output_confidence_level=out['confidence_level'],output_method=out['method'])
    return rows


def evidence_html(snapshot):
    history=snapshot.export_context.get('project_metadata',{}).get(HISTORY_KEY)
    history_html='' if history is None else '<details><summary>Independent execution history (not the current Fit)</summary><pre>'+escape(json.dumps(plain(history),ensure_ascii=False,indent=2,allow_nan=False))+'</pre></details>'
    if not snapshot.scientific_evidence:
        return '<h2>Numerical termination</h2><p>Legacy record; effective per-call trace not captured.</p>'+history_html+'<h2>Noise evidence</h2><p>Not captured.</p><h2>Transport regime</h2><p>Not captured; new interpretation requires an explicit action. CI defaults: legacy + 95% (defaulted_legacy); historical endpoints preserved.</p>'
    e=plain(snapshot.scientific_evidence);t=e['termination'];noise=e['noise'];tr=TransportSummary.from_dict(e['transport'])
    pretty=lambda v:escape(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2))
    fmt=lambda v:'N/A' if v is None else format(v,'.17g')
    rows=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in (a.process,a.axis,fmt(a.full_dimension_cm),fmt(a.full_dimension_cm/2),fmt(a.value),a.regime,a.reason))+'</tr>' for a in tr.axes)
    sizes=''.join('<p>'+escape(p.label)+': D/k (half-length)='+fmt(p.half_length_cm)+' cm; 2D/k (symmetric slab full thickness at Bi=1)='+fmt(p.slab_full_thickness_cm)+' cm. '+escape(p.reason)+'</p>' for p in tr.processes)
    out=output_view(snapshot)
    return (f'<h2>Numerical termination</h2><p>Mode={escape(t["mode"])}; native status={t["native_status"]}; native success={t["native_success"]}; final polish={escape(str(t["final_polish"]))}. Numerical success is separate from reliability.</p><details><summary>Thresholds, budgets and trace summary</summary><pre>{pretty({k:v for k,v in t.items() if k!="execution"})}</pre></details>'
        f'<h2>Noise evidence</h2><p>raw RMSE={fmt(noise["raw_RMSE"])}; RSE={fmt(noise["RSE"])}; sigma_noise={fmt(noise["sigma_noise"])} {escape(noise["unit"])}; conditional RSE/sigma={fmt(noise["RSE_over_sigma"])}. Status={escape(noise["status"])}.</p><pre>{pretty(noise["reasons"])}</pre><details><summary>Noise source and conditions</summary><pre>{pretty(noise["reference"])}</pre></details>'
        f'<h2>Transport regime</h2><p>Source={escape(tr.source_kind)}; parent reliability={escape(tr.parent_reliability)}; summary SHA={tr.content_sha256}. 0.03/30 are heuristics, not gates.</p>{sizes}<table><thead><tr><th>Process</th><th>Axis</th><th>Full size (cm)</th><th>Half size (cm)</th><th>Bi</th><th>Regime</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table><p>Derived intervals: {escape(tr.interval_reason or "available under recorded joint covariance conditions")}</p><details><summary>Transport intervals, parameters, units and conditions</summary><pre>{pretty(tr.to_dict())}</pre></details>'
        f'<h2>Selected uncertainty output</h2><p>Actual C={100*out["confidence_level"]:.12g}%; method={escape(out["method"])}; df={out["df"]}; origin={escape(out["level_origin"])}. Fit-stage policy: 95%, {escape(e["gate_policy"]["policy_version"])}; captured Diagnostics retains its own policy and error-assumption status.</p><pre>{pretty(out["conditions"])}</pre><pre>{pretty(out["unavailable"])}</pre><details><summary>Advanced methods: actual levels, samples and conditions</summary><pre>{pretty(e["advanced"])}</pre></details>'+history_html)


def verify_root_csv(snapshot,root):
    """Check the existing compatibility CSVs against the captured full arrays."""
    import csv
    from pathlib import Path
    root=Path(root);d=snapshot.dataset;f=snapshot.fit_result
    indices=[i for i,on in enumerate(d.mask) if on]
    for file,fields,arrays in (
        ('processed_data.csv',('time_s','processed_signal'),(d.processed_time,d.processed_signal)),
        ('fit_curve.csv',('time_s','observed','fitted','residual_observed_minus_fitted'),(d.processed_time,d.processed_signal,f.y_fit,f.residuals))):
        with (root/file).open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream))
        require(len(rows)==len(indices),'Compatibility CSV full observation count conflict: '+file)
        for j,row in enumerate(rows):
            require(int(row['original_raw_index'])==indices[j],'Compatibility raw row identity conflict')
            for name,array in zip(fields,arrays,strict=True):require(float(row[name])==float(array[j]),'Compatibility CSV float64 parity conflict: '+file+'/'+name)
    if snapshot.scientific_evidence:
        expected=parameter_output_rows(snapshot);out=output_view(snapshot)
        with (root/'parameters.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        rows={r['parameter']:r for r in rows}
        for name in snapshot.fit_result.params:
            row=rows[name];want=expected[name]
            require(float(row['confidence_level'])==out['confidence_level'] and row['ci_method']==out['method'],'Compatibility CI actual level/method conflict')
            for field in ('ci_lower','ci_upper'):
                require((row[field]=='' if want[field] is None else float(row[field])==want[field]),'Compatibility CI endpoint conflict')


def restore_gate_presentation(dataset,config,fit,diagnostics,saved=None):
    """Restore existing UI types, never estimate intervals on project open."""
    from pyecr.services.analysis import FitPresentation,FitParameterPresentation,_persisted_artifacts_from_fit_result,_component_tuple,_parameter_units
    from pyecr.report.snapshot import _diagnostics_mapping
    if saved is not None:
        value=plain(saved);exact_fields(value,[f.name for f in fields(FitPresentation)],'captured Fit presentation')
        value['parameters']=tuple(FitParameterPresentation(**r) for r in value['parameters'])
        for name in ('time_s','data','fit','component_H','component_O','residuals','warnings','component_1','component_2','component_labels'):value[name]=tuple(value[name])
        p=FitPresentation(**value)
        require(p.time_s==tuple(dataset.processed_time) and p.data==tuple(dataset.processed_signal) and p.fit==tuple(fit.y_fit) and p.residuals==tuple(fit.residuals),'Captured Fit presentation/full source conflict')
        require(p.SSE==fit.SSE and p.exit_status==fit.exit_status and p.nfev==fit.nfev,'Captured Fit presentation/native result conflict')
        require({r.name:r.estimate for r in p.parameters}==dict(fit.params),'Captured gate parameter identity conflict')
        require(tuple(r.name for r in p.parameters)==tuple(fit.params),'Captured gate/Jacobian coordinate order conflict')
        return p
    diag={} if diagnostics is None else _diagnostics_mapping(diagnostics)
    captured={r['name']:r for r in diag.get('parameter_diagnostics',{}).get('rows',())}
    units=_parameter_units(config);rows=[]
    for name,estimate in fit.params.items():
        r=captured.get(name,{})
        rows.append(FitParameterPresentation(name=name,estimate=estimate,unit=r.get('unit',units.get(name,'signal')),
            ci_lower=r.get('ci_lower'),ci_upper=r.get('ci_upper'),bound_hit=r.get('bound_hit'),bound_side=r.get('bound_side'),
            reliability_flag=bool(r.get('reliable',False)),reliability_reason=r.get('reliability_reason') or 'Legacy evidence unavailable; no interval recalculation on open'))
    artifacts=_persisted_artifacts_from_fit_result(fit);m8=fit.diagnostics.get('dual_surface_k',{})
    return FitPresentation(time_s=tuple(dataset.processed_time),data=tuple(dataset.processed_signal),fit=tuple(fit.y_fit),residuals=tuple(fit.residuals),
        component_H=_component_tuple(artifacts.component_H),component_O=_component_tuple(artifacts.component_O),parameters=tuple(rows),model_name=config.model_id.value,
        SSE=fit.SSE,RMSE=fit.RMSE,R2=fit.R2,exit_status=fit.exit_status,nfev=fit.nfev,
        reliable=bool(diag.get('reliable',False)) and all(r.reliability_flag for r in rows),reliability_label='Captured legacy Diagnostics, 95% defaulted_legacy; not recomputed',
        warnings=tuple(fit.warnings)+('DEFAULTED_LEGACY: original 95% evidence only; no inference on project open.',),
        component_1=tuple(m8.get('component_1',())),component_2=tuple(m8.get('component_2',())),component_labels=tuple(m8.get('component_labels',())),component_baseline=float(m8.get('component_baseline',0.)))
