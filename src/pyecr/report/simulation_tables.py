"""Full scientific tables from a captured SimulationResult; no recalculation."""
from pyecr.contracts import SimulationResult
from pyecr.contracts.common import require
from pyecr.model_display import BY_ID, VERSION as DISPLAY_VERSION
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.report.export_tables import ExportTable


def simulation_metadata(result):
    r=result.request;item=BY_ID[r.model.model_id.value];n=len(r.time_s)
    metadata=dict(kind='simulation_bundle_metadata',schema_version=1,
        model=dict(internal_id=r.model.model_id.value,display_version=DISPLAY_VERSION,
            menu_number=item.number,menu_alias=item.title,display_name=item.label),
        units=dict(time_s='s',signal=r.signal_unit,length='cm',parameters={p.name:r.signal_unit if p.unit=='signal' else p.unit
            for p in DEFAULT_MODEL_REGISTRY.get(r.model.model_id).parameters}),
        channels=dict(signal_clean='simulation_clean',signal_observed='simulation_observed',
            noise='total_observation_noise',baseline='common_given_baseline',components='clean_model_contributions'),
        raw_rows=dict(count=n,indices=list(range(n)),mask=[True]*n,index_base=0,csv_header_lines=1,
            meaning='Simulation scientific samples, not original experimental observations; no dropped rows'),
        request_sha256=r.content_sha256,result_sha256=result.content_sha256,
        sampling_kind=r.sampling_kind,display_grid_used=False,given_parameters=dict(r.given_parameters),
        boundary=r.boundary.to_dict(),noise_sd=r.noise_sd,seed=r.seed,rng_algorithm=r.rng_algorithm,
        sequence_number=r.sequence_number,observation_evidence=None if result.observation_evidence is None else dict(result.observation_evidence),
        warnings=list(result.state.warnings),interpretation='Given-parameter simulated truth and observations; no Fit, statistical reliability, R2 or CI. Components have no independent noise.')
    if r.input_units is not None:
        metadata['input_units']=r.input_units.to_dict()
        metadata['canonical_request_sha256']=r.as_canonical().content_sha256
    return metadata


def build_simulation_tables(result):
    require(isinstance(result,SimulationResult) and result.state.ready,'complete SimulationResult required')
    r=result.request;n=len(r.time_s);unit=r.signal_unit
    baseline=r.given_parameters.get('baseline',r.given_parameters.get('y0'))
    parts=tuple(result.contributions)
    tables=[ExportTable('simulation_truth',('time_s','signal_clean','baseline',*parts,'simulation_row_index'),
        ('float64',)*(3+len(parts))+('integer',),
        {'time_s':'s',**{k:unit for k in ('signal_clean','baseline',*parts)}},n,
        lambda:((t,result.signal_clean[i],baseline,*(result.contributions[k][i] for k in parts),i) for i,t in enumerate(r.time_s))),
        ExportTable('simulation_observations',('time_s','signal_observed','noise','simulation_row_index'),
        ('float64','float64','float64','integer'),{'time_s':'s','signal_observed':unit,'noise':unit},n,
        lambda:((t,result.signal_observed[i],result.noise[i],i) for i,t in enumerate(r.time_s))),
        ExportTable('simulation_rows',('simulation_row_index','raw_index','csv_line','valid','time_s'),
        ('integer','integer','integer','boolean','float64'),{'time_s':'s'},n,
        lambda:((i,i,i+2,True,t) for i,t in enumerate(r.time_s)))]
    desc=DEFAULT_MODEL_REGISTRY.get(r.model.model_id)
    tables.append(ExportTable('given_parameters',('parameter','given_value','unit','role'),
        ('text','float64','text','text'),{},len(desc.parameters),
        lambda:((p.name,r.given_parameters[p.name],unit if p.unit=='signal' else p.unit,'given') for p in desc.parameters)))
    if r.boundary.mode=='measured_history':
        b=r.boundary
        tables.append(ExportTable('boundary_history',('time_s','gas_value','source_row_index'),
            ('float64','float64','integer'),{'time_s':'s','gas_value':b.measured_value_unit},len(b.measured_time_s),
            lambda:((t,v,i) for i,(t,v) in enumerate(zip(b.measured_time_s,b.measured_values)))))
    return tuple(tables)
