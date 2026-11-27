"""Versioned Simulation export/readback; commit only a verified new directory."""
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import csv,os,shutil

from pyecr.contracts import SimulationRequest,SimulationResult,canonical_sha256
from pyecr.contracts.common import require,exact_fields,strict_json_loads
from pyecr.report.export_options import ExportOptions
from pyecr.report.export_readback import file_sha256,_decode
from pyecr.report.export_v2 import _write_json,_write_table
from pyecr.report.simulation_tables import build_simulation_tables,simulation_metadata

VERSION='simulation-bundle/1'


@dataclass(frozen=True,slots=True)
class SimulationBundle:
    root: Path
    result: SimulationResult
    manifest: dict

    def verify(self):return read_simulation_bundle(self.root)


def read_simulation_bundle(root,*,check=lambda:None):
    root=Path(root).expanduser()
    require(not root.is_symlink(),'simulation bundle root must not be a link')
    root=root.resolve()
    manifest=strict_json_loads((root/'manifest.json').read_text(encoding='utf-8'))
    exact_fields(manifest,('kind','schema_version','version','exported_at','request_sha256','result_sha256',
        'artifacts','tables','options','figure_formats','xlsx','precision','manifest_sha256'),'Simulation manifest')
    require(manifest['kind']=='simulation_bundle' and type(manifest['schema_version']) is int
        and manifest['schema_version']==1 and manifest['version']==VERSION,'unknown Simulation bundle kind/version')
    require(canonical_sha256({k:v for k,v in manifest.items() if k!='manifest_sha256'})==manifest['manifest_sha256'],
        'Simulation manifest hash mismatch')
    require(isinstance(manifest['artifacts'],dict),'Simulation artifact inventory required')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==set(manifest['artifacts'])|{'manifest.json'},'Simulation file inventory mismatch')
    for name,info in manifest['artifacts'].items():
        check();p=PurePosixPath(name)
        require(name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,'unsafe Simulation artifact path')
        path=root/name
        require(not any(part.is_symlink() for part in (path,*path.parents)) and path.resolve().is_relative_to(root),'Simulation artifact link/outside root')
        exact_fields(info,('sha256','bytes'),'Simulation artifact evidence')
        require(path.stat().st_size==info['bytes'] and file_sha256(path)==info['sha256'],'Simulation artifact hash mismatch: '+name)
    request=SimulationRequest.from_json((root/'request.json').read_text(encoding='utf-8'))
    result=SimulationResult.from_json((root/'result.json').read_text(encoding='utf-8'))
    require(result.state.ready and result.request==request,'Simulation request/result binding conflict or unavailable result')
    require(manifest['request_sha256']==request.content_sha256 and manifest['result_sha256']==result.content_sha256,'Simulation content identity conflict')
    meta=strict_json_loads((root/'metadata.json').read_text(encoding='utf-8'))
    require(canonical_sha256(meta)==canonical_sha256(simulation_metadata(result)),'Simulation units/model/rows metadata conflict')
    options=ExportOptions(**manifest['options'])
    require(options.csv_encoding=='utf-8' and options.csv_delimiter==',','Simulation bundle requires UTF-8 comma CSV')
    tables=build_simulation_tables(result)
    formats=manifest['figure_formats']
    require(isinstance(formats,list) and len(set(formats))==len(formats) and set(formats)<={'png','svg','pdf'},'invalid Simulation figure formats')
    expected_files={'request.json','result.json','metadata.json','README.md',*(t.info()['file'] for t in tables),
        *(f'figures/{name}.{fmt}' for name in ('simulation_truth','simulation_observations') for fmt in formats)}
    if manifest['xlsx'] is not None:expected_files.add('simulation.xlsx')
    require(set(manifest['artifacts'])==expected_files,'Simulation artifact roles/inventory conflict')
    require(manifest['tables']=={t.name:t.info() for t in tables},'Simulation table counts/units/columns conflict')
    for table in tables:
        check()
        with (root/table.info()['file']).open(encoding='utf-8',newline='') as stream:
            rows=csv.reader(stream)
            require(next(rows,None)==list(table.columns),'Simulation CSV header conflict: '+table.name)
            for i,(actual,expected) in enumerate(zip(rows,table.rows(),strict=True)):
                if i%1024==0:check()
                require(len(actual)==len(expected),'Simulation CSV column count conflict')
                require(all(_decode(cell,kind)==value for cell,value,kind in zip(actual,expected,table.types,strict=True)),
                    f'Simulation CSV full-array parity conflict: {table.name} row {i}')
    if manifest['xlsx'] is not None:
        from pyecr.report.export_xlsx import verify_xlsx_tables
        verify_xlsx_tables(tables,root/'simulation.xlsx',manifest['xlsx'],check)
    for fmt in formats:
        for name in ('simulation_truth','simulation_observations'):
            check();path=root/'figures'/f'{name}.{fmt}'
            if fmt=='png':
                from PIL import Image
                with Image.open(path) as img:require(img.format=='PNG','invalid Simulation PNG');img.verify()
            elif fmt=='svg':
                from xml.etree import ElementTree
                require(ElementTree.parse(path).getroot().tag.endswith('}svg'),'invalid Simulation SVG')
            else:
                raw=path.read_bytes();require(raw.startswith(b'%PDF-') and raw.rstrip().endswith(b'%%EOF'),'invalid Simulation PDF signature/trailer')
    check()
    return SimulationBundle(root,result,manifest)


def export_simulation_bundle(result,destination,*,figure_formats=('png','svg','pdf'),include_xlsx=False,
        options=None,cancel_check=None,before_publish=None,progress_callback=None):
    require(isinstance(result,SimulationResult) and result.state.ready,'complete SimulationResult required')
    result=SimulationResult.from_json(result.to_json()) # strict frozen snapshot; no forward/RNG
    options=options or ExportOptions()
    require(isinstance(options,ExportOptions) and options.csv_encoding=='utf-8' and options.csv_delimiter==',',
        'Simulation export requires UTF-8 comma CSV')
    require(type(include_xlsx) is bool,'include_xlsx must be boolean')
    formats=tuple(figure_formats)
    require(len(set(formats))==len(formats) and set(formats)<={'png','svg','pdf'},'unsupported/duplicate Simulation figure format')
    target=Path(destination).expanduser().absolute()
    require(target.name and not target.is_symlink(),'Simulation destination must be a new directory')
    target=target.resolve()
    if target.exists():raise FileExistsError('Simulation destination already exists; choose a new directory')
    target.parent.mkdir(parents=True,exist_ok=True)
    lock=target.parent/f'.{target.name}.simulation-export.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    staging=None
    def check():
        if cancel_check:cancel_check()
    def progress(f,m):
        if progress_callback:progress_callback(f,m)
    try:
        check()
        if target.exists():raise FileExistsError('Simulation destination appeared before staging')
        from pyecr.report.staging import create_export_staging
        staging=create_export_staging(target.parent,prefix=f'.{target.name}.simulation-tmp-')
        _write_json(staging/'request.json',result.request.to_dict());_write_json(staging/'result.json',result.to_dict())
        _write_json(staging/'metadata.json',simulation_metadata(result))
        tables=build_simulation_tables(result)
        for i,table in enumerate(tables):
            check();_write_table(staging,table,options,check)
            progress(.1+.4*(i+1)/len(tables),'Wrote '+table.name)
        xlsx=None
        if include_xlsx:
            from pyecr.report.export_xlsx import write_xlsx
            xlsx=write_xlsx(tables,staging/'simulation.xlsx',options,dict(kind='Simulation, not measured experiment',
                model_id=result.request.model.model_id.value,name=result.request.name,signal_unit=result.request.signal_unit,
                seed=result.request.seed,rng_algorithm=result.request.rng_algorithm,sequence=result.request.sequence_number,
                request_sha256=result.request_sha256,precision='XLSX numeric cells: 15 digits. Exact replay: request/result JSON and 17g UTF-8 CSV.'),check)
            xlsx['scientific_authority']='request.json, result.json and 17g CSV'
        if formats:
            from pyecr.report.simulation_figures import write_simulation_figures
            write_simulation_figures(result,staging/'figures',formats,options,check)
        (staging/'README.md').write_text('# PyECR Simulation bundle\n\n'
            'Given-parameter truth: tables/simulation_truth.csv; simulated observations and actual total noise: tables/simulation_observations.csv.\n'
            'Every scientific sample is retained. Components are clean model contributions, not independent noisy experiments.\n'
            'Units, model ID/menu name, full dimensions/both active faces, seed, sequence, boundaries and original simulation row mapping are retained in request.json, result.json, metadata.json and manifest.json.\n'
            'UTF-8 comma CSV uses 17 significant digits; JSON roundtrips float64. XLSX numeric cells are approximate 15 digits.\n'
            'No Fit, confidence intervals, R2 or statistical reliability is generated. Warnings in result.json remain applicable.\n'
            'Send to Analysis is explicit: choose clean/observed and review target replacement. Apply model, Fit and Diagnostics remain separate user actions.\n',encoding='utf-8')
        artifacts={p.relative_to(staging).as_posix():dict(sha256=file_sha256(p),bytes=p.stat().st_size)
            for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest=dict(kind='simulation_bundle',schema_version=1,version=VERSION,exported_at=datetime.now(timezone.utc).isoformat(),
            request_sha256=result.request_sha256,result_sha256=result.content_sha256,artifacts=artifacts,
            tables={t.name:t.info() for t in tables},options=options.to_dict(),figure_formats=list(formats),xlsx=xlsx,
            precision=dict(csv='17 significant digits; exact float64 parity',json='float64 roundtrip',xlsx='15 significant digits; approximate'))
        manifest['manifest_sha256']=canonical_sha256(manifest);_write_json(staging/'manifest.json',manifest)
        progress(.85,'Reading back full arrays, units, rows, XLSX and file hashes');read_simulation_bundle(staging,check=check)
        check()
        if before_publish:before_publish()
        check()
        if target.exists():raise FileExistsError('Simulation destination appeared during staging; preserved')
        # New directories only. On the supported Windows host rename never
        # overwrites an existing target. No fallible callbacks after commit.
        staging.rename(target);staging=None
        return SimulationBundle(target,result,manifest)
    finally:
        if staging is not None and staging.exists():
            require(staging.resolve().parent==target.parent and staging.name.startswith(f'.{target.name}.simulation-tmp-'),
                'refusing cleanup outside owned Simulation staging')
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)
