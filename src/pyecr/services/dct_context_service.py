"""Read-only Review, explicit Apply, independently versioned interpretation.

No optimizer, DCT solver, preprocessing or cross-page write occurs here.
Review may evaluate the existing production forward kernel for modal evidence.
"""
from pathlib import Path
from uuid import uuid4
from pyecr.contracts import DctContextReview, canonical_sha256
from pyecr.contracts.common import require, strict_json_loads, exact_fields
from pyecr.services.transport_summary import summarize_session, summarize_simulation
from pyecr.diagnostics.transport_modes import modal_reference


def assert_target(record, source, spectrum_sha256, kind):
    require(record.target_sha256==spectrum_sha256 and record.target_source==source and record.target_kind==kind,
            'Spectrum/source changed after Review; Review this target explicitly')


def review_parent(session, source, spectrum_sha256, kind='whole', *, per_axis=4, sequence=1, transport_snapshot=None):
    summary=summarize_session(session)
    if transport_snapshot is not None:
        from pyecr.contracts import TransportSummary
        require(isinstance(transport_snapshot,TransportSummary),'Expected R11 TransportSummary snapshot')
        summary.source.assert_matches(transport_snapshot.source)
        require(summary.point_id==transport_snapshot.point_id,'R11 transport snapshot/input conflict')
        summary=transport_snapshot  # retain selected C and the exact R11 full summary identity
    if kind=='components':
        from pyecr.services.component_source import capture_component_sources
        capture=capture_component_sources(session)
        capture.require_usable();source.identity.assert_matches(capture.identity)
    cfg=session.current_analysis_config
    # Contract validates identities before any expensive forward call.
    require((source.identity.dataset_sha256,source.identity.recipe_sha256)==
            (summary.source.dataset_sha256,summary.source.recipe_sha256),'Physical context dataset/recipe identity conflict')
    require(source.identity.config_sha256 is None or (source.identity.config_sha256,source.identity.model_id)==
            (summary.source.config_sha256,summary.source.model_id),'Physical context config identity conflict')
    if source.identity.fit_id is not None:source.identity.assert_matches(summary.source)
    modes=modal_reference(summary,cfg.model,cfg.solver,cfg.boundary,source.time_s,per_axis=per_axis)
    return DctContextReview(target_sha256=spectrum_sha256,target_source=source,target_kind=kind,
        transport=summary,model=cfg.model,solver=cfg.solver,boundary=cfg.boundary,basis='Current bound parent Fit; original 95% gate retained.',
        declared_given=False,sequence=sequence,modal_reference=modes)


def review_given(request, source, spectrum_sha256, kind='whole', *, declared=False, basis='', per_axis=4, sequence=1):
    require(kind!='components','Given context cannot replace the reliable component parent')
    require(declared and bool(basis.strip()),'Explicit given background declaration and basis required')
    copied=request.copied_from
    if copied is not None:
        require((source.identity.dataset_sha256,source.identity.recipe_sha256)==(copied.dataset_sha256,copied.recipe_sha256),
                'Given request copied from a different dataset/recipe')
        if source.identity.fit_id is not None:source.identity.assert_matches(copied)
    summary=summarize_simulation(request)
    if source.simulation_request_sha256 is not None:
        require(source.simulation_request_sha256==request.content_sha256,'Simulation request identity conflict')
    modes=modal_reference(summary,request.model,request.solver_config,request.boundary,source.time_s,per_axis=per_axis)
    return DctContextReview(target_sha256=spectrum_sha256,target_source=source,target_kind=kind,transport=summary,
        model=request.model,solver=request.solver_config,boundary=request.boundary,basis=basis,declared_given=True,
        sequence=sequence,modal_reference=modes)


class DctContextStore:
    """Session-only independent history. No context is automatically current."""
    def __init__(self):
        self.history=[];self.selected={}

    def apply(self,record,source,spectrum_sha256,kind,*,session=None,request=None):
        require(isinstance(record,DctContextReview),'Review context first')
        assert_target(record,source,spectrum_sha256,kind)
        if record.transport.source is not None:
            require(session is not None,'Current bound Fit session required at Apply')
            current=summarize_session(session)
            record.transport.source.assert_matches(current.source)
            require(current.point_id==record.transport.point_id,'Physical parameters changed after Review')
            if kind=='components':
                from pyecr.services.component_source import capture_component_sources
                capture=capture_component_sources(session);capture.require_usable()
                source.identity.assert_matches(capture.identity)
        else:
            require(request is not None and request.content_sha256==record.transport.simulation_request_sha256,
                    'Given request changed after Review')
        self.history.append(record);self.selected[spectrum_sha256]=record
        return record

    def clear(self,spectrum_sha256):
        self.selected.pop(spectrum_sha256,None)


def save_context(record,path):
    require(isinstance(record,DctContextReview),'Expected reviewed context')
    path=Path(path);require(not path.exists(),'Context destination exists; choose a new file')
    value=dict(version='dct-context-sidecar/1',payload=record.to_dict(),sha256=record.content_sha256)
    import json
    stage=path.with_name(path.name+'.'+uuid4().hex+'.staging')
    try:
        stage.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        require(read_context(stage.read_text(encoding='utf-8'))==record,'Context staging roundtrip conflict')
        stage.rename(path)
    finally:
        if stage.exists():stage.unlink()
    return path


def read_context(text):
    data=strict_json_loads(text);exact_fields(data,('version','payload','sha256'),'DCT context sidecar')
    require(data['version']=='dct-context-sidecar/1','Unknown DCT context sidecar version')
    record=DctContextReview.from_dict(data['payload'])
    require(record.content_sha256==data['sha256'],'DCT context checksum conflict')
    return record
