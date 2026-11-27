"""Explicit simulation-source import and immutable provenance; no Fit/config Apply."""
from dataclasses import dataclass,replace,field
from uuid import uuid4
from pathlib import Path
import csv
import numpy as np
from pyecr.contracts import SimulationResult,canonical_sha256
from pyecr.contracts.common import require,exact_fields,strict_json_loads,digest
from pyecr.contracts.simulation_parent import SimulationSendReceipt,SimulationParentEvidence,ORIGIN_ID_FIELDS
from pyecr.io.importer import ImportMapping,import_tabular_file
from pyecr.io.dataset import EcrDataset
from pyecr.report.export_readback import file_sha256
from pyecr.report.snapshot import dataset_sha256
from pyecr.report.simulation_bundle import read_simulation_bundle

CHANNELS={'clean':('simulation_clean','simulation_truth.csv'),
          'observed':('simulation_observed','simulation_observations.csv')}
ORIGIN_VERSION='simulation-analysis-source/1'


def validate_simulation_origin(dataset):
    origin=dataset.metadata.get('simulation_origin')
    if origin is None:
        require(not str(dataset.metadata.get('source_kind','')).startswith('simulation_'),'Simulation source label has no provenance')
        return None
    exact_fields(origin,('version','source_kind','request_sha256','result_sha256','result','raw_indices',
        'bundle_manifest_sha256','bundle_table','bundle_path_hint'),'simulation Analysis provenance')
    require(origin['version']==ORIGIN_VERSION,'unknown simulation Analysis provenance version')
    y=SimulationResult.from_dict(origin['result']);r=y.request
    require(y.state.ready and origin['request_sha256']==r.content_sha256 and origin['result_sha256']==y.content_sha256,'Simulation Analysis identity conflict')
    kinds={v[0]:v[1] for v in CHANNELS.values()};kind=origin['source_kind']
    require(kind in kinds and dataset.metadata.get('source_kind')==kind and origin['bundle_table']=='tables/'+kinds[kind],
        'Simulation Analysis source/channel conflict')
    require(all(type(i) is int for i in origin['raw_indices']) and tuple(origin['raw_indices'])==tuple(range(len(r.time_s))),'Simulation raw row mapping conflict')
    digest(origin['bundle_manifest_sha256'],'Simulation bundle manifest identity')
    require(dataset.metadata.get('raw_time_unit')=='s' and dataset.metadata.get('raw_signal_unit')==r.signal_unit,
        'Simulation raw units conflict')
    expected=y.signal_clean if kind=='simulation_clean' else y.signal_observed
    require(np.array_equal(dataset.raw_time,r.time_s) and np.array_equal(dataset.raw_signal,expected),
        'Simulation original rows/time/signal conflict')
    return y


def attach_simulation_origin(imported,bundle,channel):
    require(channel in CHANNELS,'choose clean or observed explicitly')
    y=bundle.result;r=y.request;kind,filename=CHANNELS[channel]
    d=imported.dataset
    require(d.metadata.get('time_unit')=='s' and d.metadata.get('signal_unit')==r.signal_unit
        and d.metadata.get('signal_type')==r.signal_type,'Simulation import must use recorded time/signal units and type')
    path=bundle.root/'tables'/filename
    require(Path(imported.preview.source.source_path).resolve()==path.resolve()
        and file_sha256(path)==imported.preview.source.sha256,'Simulation CSV source changed or mismatched during import')
    # Generic pandas numeric conversion does not guarantee a 17-digit decimal
    # roundtrip. Parse the verified CSV bytes with Python float in this explicit
    # bundle mode; never substitute production result arrays for source cells.
    with path.open(encoding='utf-8',newline='') as stream:
        reader=csv.reader(stream);next(reader)
        samples=tuple((float(row[0]),float(row[1])) for row in reader)
    require(file_sha256(path)==imported.preview.source.sha256,'Simulation CSV changed while parsing')
    times,signals=zip(*samples)
    changed=int(np.count_nonzero(d.raw_time!=times)+np.count_nonzero(d.raw_signal!=signals))
    d=EcrDataset.from_raw(times,signals,time_unit='s',signal_type=r.signal_type,signal_unit=r.signal_unit,
        metadata={**d.metadata_as_dict(),'numeric_parse_policy':'verified_simulation_csv_python_float_17g/1',
            'generic_numeric_parser_differing_cells':changed})
    origin=dict(version=ORIGIN_VERSION,source_kind=kind,request_sha256=r.content_sha256,result_sha256=y.content_sha256,
        result=y.to_dict(),raw_indices=list(range(len(r.time_s))),bundle_manifest_sha256=bundle.manifest['manifest_sha256'],
        bundle_table='tables/'+filename,bundle_path_hint=str(bundle.root))
    d=replace(d,metadata={**d.metadata_as_dict(), 'source_kind':kind,'simulation_origin':origin,
        'dataset_name':r.name+' · '+kind})
    validate_simulation_origin(d)
    return replace(imported,dataset=d)


def import_simulation_channel(root,channel):
    require(channel in CHANNELS,'choose clean or observed explicitly')
    bundle=read_simulation_bundle(root);r=bundle.result.request
    imported=import_tabular_file(bundle.root/'tables'/CHANNELS[channel][1],
        ImportMapping(time_column=0,signal_column=1,time_unit='s',signal_type=r.signal_type,signal_unit=r.signal_unit))
    return attach_simulation_origin(imported,bundle,channel)


def recognize_simulation_csv(imported):
    """Restore labels for an explicitly imported exported channel; values unchanged."""
    path=Path(imported.preview.source.source_path)
    channel=next((k for k,v in CHANNELS.items() if v[1]==path.name),None)
    if channel is None:return imported
    root=path.parent.parent;manifest=root/'manifest.json'
    require(path.parent.name=='tables' and manifest.is_file(),
        'Simulation channel CSV needs its complete companion bundle; source metadata must not be silently discarded')
    header=strict_json_loads(manifest.read_text(encoding='utf-8'))
    require(header.get('kind')=='simulation_bundle','Simulation channel CSV has no matching Simulation manifest')
    bundle=read_simulation_bundle(root)
    require(imported.mapping.time_column_index==0 and imported.mapping.signal_column_index==1,
        'Simulation channel import requires time column 0 and clean/observed column 1; choose the proper channel file')
    return attach_simulation_origin(imported,bundle,channel)


def target_identity(session):
    from pyecr.report.snapshot import _diagnostics_mapping
    return canonical_sha256(dict(dataset=None if session.dataset is None else dataset_sha256(session.dataset),
        config=None if session.current_analysis_config is None else session.current_analysis_config.to_dict(),
        recipe=None if session.preprocessing is None else session.preprocessing.config.to_dict(),
        fit=None if session.fit_result is None else session.fit_result.to_dict(),
        revision=repr(session.current_revision),generation=session.fit_generation,
        diagnostics=None if session.diagnostics is None else _diagnostics_mapping(session.diagnostics),diagnostics_stale=session.diagnostics_stale,fit_stale=session.fit_stale,
        diagnostics_identity=None if session.diagnostics_identity is None else session.diagnostics_identity.to_dict(),
        advanced={k:id(v) for k,v in session.advanced_diagnostics.items()},
        project_metadata=dict(session.project_metadata)))


@dataclass(frozen=True,slots=True)
class SimulationSendPlan:
    imported: object
    channel: str
    target_action: str
    target_session: object
    target_sha256: str
    impact: str
    review_id: str = field(default_factory=lambda:uuid4().hex)


def prepare_simulation_send(root,channel,target_action,session):
    require(target_action in {'require_empty','replace_current'},'choose target impact explicitly')
    if target_action=='require_empty':
        require(session.dataset is None and session.current_analysis_config is None and session.fit_result is None,
            'Analysis is not empty; explicitly choose replace_current and review the impact')
    imported=import_simulation_channel(root,channel)
    impact=('Import the selected channel into empty Analysis and start an unsaved analysis.' if target_action=='require_empty' else
        'Replace current Analysis data/recipe/config/Fit/Diagnostics and clear project association.')
    impact+=' Previous on-disk files stay unchanged. Captured reports/DCT remain historical; no Fit/Diagnostics/Apply is executed.'
    return SimulationSendPlan(imported,channel,target_action,session,target_identity(session),impact)


def simulation_origin_identity(dataset):
    """Stable origin binding excludes only a relocatable path hint."""
    validate_simulation_origin(dataset)
    origin=dataset.metadata['simulation_origin']
    return {name:origin[name] for name in ORIGIN_ID_FIELDS}


def commit_simulation_send(plan):
    """Called only after current source/target review checks, before publication."""
    from pyecr.preprocessing import raw_observations_sha256
    d=plan.imported.dataset;y=validate_simulation_origin(d)
    require(y is not None and d.metadata['source_kind']=='simulation_'+plan.channel,'Send channel changed after review')
    fields=dict(version='simulation-send/1',review_id=plan.review_id,channel=plan.channel,
        target_action=plan.target_action,target_before_sha256=plan.target_sha256,
        origin_sha256=canonical_sha256(simulation_origin_identity(d)),request_sha256=y.request_sha256,
        result_sha256=y.content_sha256,raw_sha256=raw_observations_sha256(d))
    receipt=SimulationSendReceipt(**{k:v for k,v in fields.items() if k!='version'},send_identity=canonical_sha256(fields))
    sent=replace(d,metadata={**d.metadata_as_dict(),'simulation_send':receipt.to_dict()})
    return replace(plan.imported,dataset=sent)


def capture_simulation_parent(dataset):
    """Frozen proof for estimated components; no Generate/Fit/Diagnostics."""
    y=validate_simulation_origin(dataset)
    if y is None:
        require(dataset.metadata.get('simulation_send') is None,'Experiment cannot carry a Simulation Send receipt')
        return None
    from pyecr.preprocessing import raw_observations_sha256
    value=dataset.metadata.get('simulation_send')
    require(value is not None,'SIMULATION_SEND_UNRECORDED: explicitly review and Send the verified Simulation result before capturing estimated components')
    send=SimulationSendReceipt.from_dict(value)
    require(send.raw_sha256==raw_observations_sha256(dataset),'Simulation Send/raw array conflict')
    return SimulationParentEvidence(result=y,send=send,origin_identity=simulation_origin_identity(dataset))


def validate_simulation_capture(capture,dataset):
    require(capture.simulation_parent==capture_simulation_parent(dataset),
        'Simulation capture Generate/Send/channel binding conflict')
