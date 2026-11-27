"""R13 DCT interpretation archive referencing actual captured spectra.

No spectrum/source is inferred from the current Analysis or from equal-looking
arrays. This is a provenance extension to the existing archive/export chain.
"""
from collections.abc import Mapping
import json
from pyecr.config import _freeze_json
from pyecr.contracts import DctContextReview,DctSourceSnapshot,canonical_sha256
from pyecr.contracts.common import require,exact_fields

VERSION='dct-interpretation-archive/1'


def plain(v):
    from pyecr.contracts.common import encode
    return encode(v)


def targets(archive,legacy_result=None,legacy_source=None):
    result={}
    if archive is not None:
        for j in archive.whole_jobs:
            if j.analysis is not None and j.prepared is not None:
                result[j.analysis.content_sha256]=(j.prepared.processed,'whole')
        for j in archive.component_jobs:
            if j.bundle is not None and j.bundle.component_capture is not None:
                result[j.bundle.content_sha256]=(j.bundle.component_capture.parent_source,'components')
    if legacy_source is not None:
        require(legacy_result is not None,'Legacy DCT source missing its actual result')
        require(tuple(legacy_result.time_s)==legacy_source.time_s and tuple(legacy_result.signal)==legacy_source.signal,'Legacy full array/source conflict')
        result[canonical_sha256(legacy_result.to_dict())]=(legacy_source,'monotonic')
    return result


def capture_interpretations(store,archive,legacy_result=None,legacy_source=None,*,legacy_targets=None):
    history=() if store is None else tuple(store.history)
    selected={} if store is None else {k:v.content_sha256 for k,v in store.selected.items()}
    legacy=dict(legacy_targets or {})
    if legacy_source is not None:legacy[canonical_sha256(legacy_result.to_dict())]=(legacy_result,legacy_source)
    value=dict(version=VERSION,history=[r.to_dict() for r in history],selected=selected,
        legacy_targets=[dict(result=r.to_dict(),source=s.to_dict()) for r,s in legacy.values()],
        restored_state='explicit_applied_snapshots; no inversion or Analysis mutation')
    value['sha256']=canonical_sha256(value)
    return validate_interpretations(value,archive,legacy_result)


def validate_interpretations(value,archive,legacy_result=None):
    value=plain(value)
    exact_fields(value,('version','history','selected','legacy_targets','restored_state','sha256'),'DCT interpretation archive')
    require(value['version']==VERSION,'Unknown DCT interpretation archive version')
    require(value['sha256']==canonical_sha256({k:v for k,v in value.items() if k!='sha256'}),'DCT interpretation archive hash conflict')
    available=targets(archive);records={}
    for item in value['legacy_targets']:
        exact_fields(item,('result','source'),'legacy DCT target')
        from pyecr.services.dct_service import DctPresentation
        r=DctPresentation.from_dict(item['result']);s=DctSourceSnapshot.from_dict(item['source'])
        target=targets(None,r,s)
        require(not set(target)&set(available),'Duplicate legacy DCT target identity')
        available.update(target)
    for row in value['history']:
        r=DctContextReview.from_dict(row)
        require(r.target_sha256 in available,'Missing frozen spectrum/source for DCT context')
        expected,kind=available[r.target_sha256]
        require(r.target_source==expected and r.target_kind==kind,'DCT context/spectrum dataset/Fit identity conflict')
        records[r.content_sha256]=r
    require(isinstance(value['selected'],Mapping),'Malformed selected DCT context history')
    for key,ref in value['selected'].items():
        require(ref in records and records[ref].target_sha256==key,'Selected DCT interpretation identity conflict')
    return _freeze_json(value,path='dct_interpretation_archive')


def restore_interpretations(value,store):
    """Explicit project open. Retain identity/history, do not apply background to Analysis."""
    store.history=[DctContextReview.from_dict(r) for r in value['history']]
    by_id={r.content_sha256:r for r in store.history}
    store.selected={key:by_id[ref] for key,ref in value['selected'].items()}


def legacy_targets(value):
    from pyecr.services.dct_service import DctPresentation
    return {canonical_sha256(item['result']):(DctPresentation.from_dict(item['result']),DctSourceSnapshot.from_dict(item['source'])) for item in value['legacy_targets']}


def dct_evidence(archive,interpretations):
    value=dict(version='r13-dct-evidence/1',archive=None if archive is None else archive.to_dict(),interpretations=plain(interpretations))
    return validate_dct_evidence(value)


def validate_dct_evidence(value,source=None):
    value=plain(value);exact_fields(value,('version','archive','interpretations'),'captured DCT evidence')
    require(value['version']=='r13-dct-evidence/1','Unknown captured DCT evidence version')
    from pyecr.project.dct_archive import DctArchive
    archive=None if value['archive'] is None else DctArchive.from_dict(value['archive'])
    validate_interpretations(value['interpretations'],archive)
    if source is not None:
        available=targets(archive)
        for key,(r,s) in legacy_targets(value['interpretations']).items():available[key]=(s,'monotonic')
        for s,kind in available.values():
            require((s.identity.dataset_sha256,s.identity.recipe_sha256)==(source.dataset_sha256,source.recipe_sha256),'DCT/physical report dataset identity conflict')
            if kind=='components':source.assert_matches(s.identity)
    return _freeze_json(value,path='dct_evidence')


def write_dct_evidence(snapshot,root,options,check):
    """Use existing DCT tables/figures and the same scientific CSV writer."""
    if not snapshot.dct_evidence:return
    from pyecr.report.export_v2 import _write_json,_write_table
    from pyecr.project.dct_archive import DctArchive
    from pyecr.report.dct_tables import build_dct_tables
    value=plain(snapshot.dct_evidence);check();root.mkdir(parents=True,exist_ok=True)
    _write_json(root/'captured_dct_evidence.json',value)
    if value['archive'] is not None:
        archive=DctArchive.from_dict(value['archive'])
        for table in build_dct_tables(archive):check();_write_table(root,table,options,check)
        from pyecr.report.export_options import ExportPreset
        if options.preset is ExportPreset.PUBLICATION:
            from pyecr.report.dct_figures import write_dct_figures
            write_dct_figures(archive,root/'figures',('png','svg','pdf'),options,check)


def verify_dct_evidence(snapshot,root,options,check):
    """Snapshot binding plus exact cells/row counts, using original table builders."""
    if not snapshot.dct_evidence:
        require(not root.exists(),'Unbound DCT evidence in physical Fit export');return
    import csv
    from pyecr.contracts.common import strict_json_loads
    from pyecr.report.export_readback import _decode
    value=strict_json_loads((root/'captured_dct_evidence.json').read_text(encoding='utf-8'))
    validate_dct_evidence(value);require(canonical_sha256(value)==canonical_sha256(snapshot.dct_evidence),'DCT evidence/snapshot identity conflict')
    if value['archive'] is not None:
        from pyecr.project.dct_archive import DctArchive
        from pyecr.report.dct_tables import build_dct_tables
        for table in build_dct_tables(DctArchive.from_dict(value['archive'])):
            check()
            with (root/table.info()['file']).open(encoding=options.csv_encoding,newline='') as f:
                rows=csv.reader(f,delimiter=options.csv_delimiter)
                require(next(rows)==list(table.columns),'Captured DCT CSV header conflict')
                n=0
                for actual,expected in zip(rows,table.rows(),strict=True):
                    require(len(actual)==len(expected),'Captured DCT column count conflict')
                    for cell,v,kind in zip(actual,expected,table.types,strict=True):
                        require(_decode(cell,kind)==(json.loads(v) if v is not None and kind.startswith('json') else v),'Captured DCT full-array CSV conflict')
                    n+=1
                require(n==table.row_count,'Captured DCT row count conflict')


def context_table(value):
    from pyecr.report.export_tables import ExportTable,_flatten
    rows=tuple(_flatten(plain(value)))
    return ExportTable('PhysicalContext',('field','value_json','status','reason'),('text','json','text','text?'),{},len(rows),lambda:iter(rows))
