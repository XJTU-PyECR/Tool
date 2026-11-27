"""Given/estimated comparison only after an actual user-run physical Fit."""
from pyecr.contracts import canonical_sha256
from pyecr.report.export_tables import ExportTable
from pyecr.report.export_v2 import _write_json,_write_table
from pyecr.report.export_options import ExportOptions


def simulation_comparison(snapshot):
    from pyecr.services.simulation_transfer import validate_simulation_origin
    y=validate_simulation_origin(snapshot.dataset)
    if y is None:return None
    r=y.request;c=snapshot.analysis_config;f=snapshot.fit_result;d=snapshot.dataset
    differences=[]
    if r.model.model_id!=c.model_id:differences.append('scientific model differs; parameters are not remapped')
    if r.model.geometry!=c.model.geometry:differences.append('geometry/active axes differ')
    if r.model.surface_mode!=c.model.surface_mode:differences.append('surface mode differs')
    if r.boundary!=c.boundary:differences.append('boundary differs')
    if (r.model.step_type,r.model.sign_H,r.model.sign_O)!=(c.model.step_type,c.model.sign_H,c.model.sign_O):differences.append('direction/sign convention differs')
    if d.metadata.get('signal_type')!=r.signal_type or d.metadata.get('signal_unit')!=r.signal_unit:differences.append('signal representation/units differ')
    import numpy as np
    if not (np.array_equal(d.processed_time,d.raw_time) and np.array_equal(d.processed_signal,d.raw_signal) and np.all(d.mask)):
        differences.append('preprocessing changed observations/time/representation; direct given-estimated differences disabled')
    estimates=dict(f.params)
    rows=[]
    for name,given in r.given_parameters.items():
        estimated=estimates.get(name);comparable=estimated is not None and not differences
        rows.append(dict(parameter=name,given_value=given,estimated_value=estimated,
            difference=estimated-given if comparable else None,comparable=comparable,
            reason='same scientific inputs; slot-based comparison, no mode/species reassignment' if comparable else
                '; '.join(differences) or 'not an estimated parameter in this Fit (may be fixed or absent)'))
    return dict(kind='simulation_fit_comparison',schema_version=1,source_kind=d.metadata['source_kind'],
        simulation_request_sha256=r.content_sha256,simulation_result_sha256=y.content_sha256,
        physical_fit_sha256=canonical_sha256(f.to_dict()),given_model_id=r.model.model_id.value,
        fitted_model_id=c.model_id.value,given_parameters=dict(r.given_parameters),estimated_parameters=estimates,
        parameter_comparison=rows,direct_comparison_reasons=differences,
        warnings=list(snapshot.warnings),diagnostics=snapshot.diagnostics_result,
        interpretation='Given truth and actual Fit estimates are separate. Fixed inputs are not estimated parameters. No independent process/species proof or new uncertainty is generated. Fit curve, observed-minus-fit residuals and captured Diagnostics remain in the original report.')


def simulation_fit_tables(snapshot,comparison=None):
    comparison=simulation_comparison(snapshot) if comparison is None else comparison
    if comparison is None:return ()
    origin=snapshot.dataset.metadata['simulation_origin'];r=snapshot.dataset.metadata['simulation_origin']['result']['request']
    from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
    given_units={p.name:r['signal_unit'] if p.unit=='signal' else p.unit for p in DEFAULT_MODEL_REGISTRY.get(comparison['given_model_id']).parameters}
    fit_units={p.name:snapshot.dataset.metadata['signal_unit'] if p.unit=='signal' else p.unit
        for p in DEFAULT_MODEL_REGISTRY.get(snapshot.analysis_config.model_id).parameters}
    given=comparison['given_parameters'];estimated=comparison['estimated_parameters'];rows=comparison['parameter_comparison']
    return (ExportTable('GivenParameters',('parameter','given_value','unit','role'),('text','float64','text','text'),{},len(given),
            lambda:((k,v,given_units[k],'given') for k,v in given.items())),
        ExportTable('EstimatedParameters',('parameter','estimated_value','unit','role'),('text','float64','text','text'),{},len(estimated),
            lambda:((k,v,fit_units.get(k,'see AnalysisConfig'),'estimated_in_actual_Fit') for k,v in estimated.items())),
        ExportTable('SimulationCompare',('parameter','given_value','estimated_value','difference','comparable','reason'),
            ('text','float64','float64?','float64?','boolean','text'),{},len(rows),
            lambda:((p['parameter'],p['given_value'],p['estimated_value'],p['difference'],p['comparable'],p['reason']) for p in rows)))


def write_simulation_fit_evidence(snapshot,root,*,check=lambda:None,write_tables=False):
    comparison=simulation_comparison(snapshot)
    if comparison is None:return
    check();origin=snapshot.dataset.metadata['simulation_origin']
    _write_json(root/'simulation_request.json',origin['result']['request'])
    _write_json(root/'simulation_comparison.json',comparison)
    _write_json(root/'given_parameters.json',dict(role='given_simulation_truth',model_id=comparison['given_model_id'],
        simulation_request_sha256=comparison['simulation_request_sha256'],parameters=comparison['given_parameters']))
    _write_json(root/'estimated_parameters.json',dict(role='estimated_in_actual_physical_Fit',model_id=comparison['fitted_model_id'],
        physical_fit_sha256=comparison['physical_fit_sha256'],parameters=comparison['estimated_parameters']))
    if write_tables:
        for table in simulation_fit_tables(snapshot,comparison):check();_write_table(root,table,ExportOptions(),check)


def verify_simulation_fit_evidence(snapshot,root,*,verify_tables=False):
    from pyecr.contracts.common import strict_json_loads,require
    expected=simulation_comparison(snapshot)
    names={'simulation_request.json','simulation_comparison.json','given_parameters.json','estimated_parameters.json'}
    if expected is None:
        require(not any((root/n).exists() for n in names),'non-simulation report contains simulation comparison evidence')
        return
    expected_files={'simulation_request.json':snapshot.dataset.metadata['simulation_origin']['result']['request'],
        'simulation_comparison.json':expected,
        'given_parameters.json':dict(role='given_simulation_truth',model_id=expected['given_model_id'],simulation_request_sha256=expected['simulation_request_sha256'],parameters=expected['given_parameters']),
        'estimated_parameters.json':dict(role='estimated_in_actual_physical_Fit',model_id=expected['fitted_model_id'],physical_fit_sha256=expected['physical_fit_sha256'],parameters=expected['estimated_parameters'])}
    for name,value in expected_files.items():
        require(canonical_sha256(strict_json_loads((root/name).read_text(encoding='utf-8')))==canonical_sha256(value),'simulation Fit comparison binding conflict: '+name)
    if verify_tables:
        import csv
        from pyecr.report.export_readback import _decode
        for table in simulation_fit_tables(snapshot,expected):
            with (root/table.info()['file']).open(encoding='utf-8',newline='') as stream:
                rows=csv.reader(stream);require(next(rows,None)==list(table.columns),'simulation Fit comparison CSV header conflict')
                for actual,row in zip(rows,table.rows(),strict=True):
                    require(len(actual)==len(row) and all(_decode(a,k)==v for a,v,k in zip(actual,row,table.types,strict=True)),
                        'simulation Fit comparison CSV value conflict')
