"""P10 versioned DCT reports with staged full-file and scientific readback."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import json
import os
import shutil

from pyecr.contracts.common import require, exact_fields, strict_json_loads, canonical_sha256
from pyecr.project.dct_archive import DctArchive
from pyecr.report.dct_tables import build_dct_tables, records
from pyecr.report.export_options import ExportOptions, decode_text
from pyecr.report.export_v2 import _write_table, _write_json
from pyecr.report.export_readback import file_sha256

LIMITATIONS = (
    'Observed effective response; no material-response deconvolution or automatic species assignment.',
    'Positive terminal-deviation spectrum means above terminal (decay downward); negative means below terminal (rise upward).',
    'Signed whole spectra are unconstrained; mass = chi * quadrature weight in log10(tau). No signed area renormalization.',
    'lambda multiplies the penalty, not lambda squared. GCV and sensitivity are not confidence intervals.',
    'Model components depend on the captured reliable parent Fit and Diagnostics. They are not independent experiments or independent noise samples.',
    'Component shape spectra are nonnegative unit-mass shapes; common spectra retain signed B/S. Delta is explicit and is not subtracted from experiment.',
    'No calibrated component CI or bootstrap is implemented. Unresolved/unstable/boundary peaks and unavailable components remain evidence.',
    'P09 component spectra contain no extracted peak candidates. Component peak lists are unavailable, not inferred during export; whole signed peak evidence is retained.',
    'All valid observations enter residuals. Display grids never enter errors. Historical evidence is not a current analysis.',
)


@dataclass(frozen=True, slots=True)
class DctReportBundle:
    root: Path
    archive: DctArchive
    manifest: dict

    def verify(self):
        return read_dct_bundle(self.root)


def read_dct_bundle(root, *, check=lambda: None):
    root = Path(root).resolve()
    csv.field_size_limit(64*1024*1024)
    manifest = strict_json_loads((root/'manifest.json').read_text(encoding='utf-8'))
    new=manifest.get('schema_version')==2
    exact_fields(manifest, ('kind', 'schema_version', 'archive_sha256', 'exported_at', 'artifacts',
        'tables', 'options', 'figure_formats', 'xlsx', 'precision', 'manifest_sha256',*(('interpretation_sha256',) if new else ()),
        *(('capability_evidence',) if 'capability_evidence' in manifest else ())), 'DCT manifest')
    require(manifest['kind'] == 'dct_report_bundle' and type(manifest['schema_version']) is int and manifest['schema_version'] in (1,2),
        'unknown DCT report kind/version')
    content = {k: v for k,v in manifest.items() if k != 'manifest_sha256'}
    require(canonical_sha256(content) == manifest['manifest_sha256'], 'DCT manifest hash mismatch')
    expected_files = set(manifest['artifacts']) | {'manifest.json'}
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual_files == expected_files, 'DCT report file inventory mismatch')
    for name, info in manifest['artifacts'].items():
        check()
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name,
            'unsafe DCT artifact path')
        file = root/name
        require(not any(p.is_symlink() for p in (file, *file.parents) if p != root.parent)
            and file.resolve().is_relative_to(root), 'DCT artifact link/outside root')
        require(file.stat().st_size == info['bytes'] and file_sha256(file) == info['sha256'], f'DCT artifact hash mismatch: {name}')
    archive = DctArchive.from_json((root/'dct_archive.json').read_text(encoding='utf-8'))
    require(canonical_sha256(archive.to_dict()) == manifest['archive_sha256'], 'DCT archive hash mismatch')
    from .capability_evidence import component_records
    evidence=component_records(archive)
    require(manifest.get('capability_evidence',[])==evidence,'DCT manifest eligibility conflict')
    options = ExportOptions(**manifest['options'])
    tables = build_dct_tables(archive)
    if new:
        from pyecr.project.interpretation_archive import validate_interpretations,context_table
        interpretation=validate_interpretations(strict_json_loads((root/'interpretations.json').read_text(encoding='utf-8')),archive)
        require(canonical_sha256(interpretation)==manifest['interpretation_sha256'],'DCT interpretation manifest identity conflict')
        tables=(*tables,context_table(interpretation))
    require({t.name:t.info() for t in tables} == manifest['tables'], 'DCT table schema/units/count conflict')
    for table in tables:
        check()
        with (root/table.info()['file']).open(encoding=options.csv_encoding, newline='') as stream:
            rows = csv.reader(stream, delimiter=options.csv_delimiter)
            require(next(rows, None) == list(table.columns), f'DCT CSV header conflict: {table.name}')
            for index, expected in enumerate(table.rows()):
                if index % 1024 == 0:
                    check()
                row = next(rows, None)
                require(row is not None and len(row) == len(expected), f'DCT CSV missing row: {table.name}')
                for cell, value, kind in zip(row, expected, table.types, strict=True):
                    if value is None:
                        same = cell == ''
                    elif kind.startswith('float'):
                        same = float(cell) == float(value)  # exact float64 parity, not tolerance
                    elif kind.startswith('integer'):
                        same = int(cell) == value
                    elif kind.startswith('boolean'):
                        same = cell == str(bool(value)).lower()
                    elif kind.startswith('json'):
                        same = strict_json_loads(cell) == strict_json_loads(value)
                    else:
                        same = decode_text(cell) == value
                    require(same, f'DCT CSV full-array parity mismatch: {table.name} row {index}')
            require(next(rows, None) is None, f'DCT CSV extra rows: {table.name}')
    return DctReportBundle(root, archive, manifest)


def export_dct_bundle(archive, destination, *, options=None, figure_formats=('png','svg','pdf'),
        include_xlsx=False, cancel_check=None, before_publish=None, progress_callback=None,interpretations=None):
    """Export captured history; replace only a verified DCT report directory.

    Reuses the existing 17g CSV writer, literal-safe XLSX writer and directory-swap
    implementation. No physical AnalysisSnapshot is fabricated for whole DCT.
    """
    options = options or ExportOptions()
    require(isinstance(archive, DctArchive) and isinstance(options, ExportOptions), 'DCT archive/options required')
    require(isinstance(include_xlsx, bool), 'include_xlsx must be boolean')
    formats = tuple(figure_formats)
    require(len(set(formats)) == len(formats) and set(formats) <= {'png','svg','pdf'}, 'unsupported/duplicate DCT figure format')
    archive = archive.validated()
    if interpretations is not None:
        from pyecr.project.interpretation_archive import validate_interpretations
        interpretations=validate_interpretations(interpretations,archive)
    target = Path(destination).expanduser().resolve()
    require(target.name and target != target.parent, 'DCT destination must name a directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent/f'.{target.name}.dct-export.lock'
    descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    os.close(descriptor)
    staging = None
    def check():
        if cancel_check:
            cancel_check()
    def progress(f, message):
        if progress_callback:
            progress_callback(f, message)
    try:
        check()
        original = None
        if target.exists():
            if interpretations is not None:raise FileExistsError('R13 DCT export destination exists; choose a new directory')
            original = read_dct_bundle(target, check=check).manifest['manifest_sha256']
        from pyecr.report.staging import create_export_staging
        staging = create_export_staging(target.parent,prefix=f'.{target.name}.dct-tmp-')
        _write_json(staging/'dct_archive.json', archive.to_dict())
        tables = build_dct_tables(archive)
        if interpretations is not None:
            from pyecr.project.interpretation_archive import context_table
            _write_json(staging/'interpretations.json',interpretations)
            tables=(*tables,context_table(interpretations))
        for i, table in enumerate(tables):
            check(); _write_table(staging, table, options, check)
            progress(.1+.4*(i+1)/max(len(tables),1), f'Wrote {table.name}')
        xlsx = None
        if include_xlsx:
            from pyecr.report.export_xlsx import write_xlsx
            xlsx = write_xlsx(tables, staging/'dct_tables.xlsx', options,
                dict(kind='DCT captured evidence', precision='15 digits in XLSX; exact replay uses dct_archive.json and 17g CSV'), check)
            xlsx['scientific_authority'] = 'dct_archive.json and 17g CSV'
        if formats:
            from pyecr.report.dct_figures import write_dct_figures
            write_dct_figures(archive, staging/'figures', formats, options, check)
        sections = ['# PyECR DCT captured evidence', *LIMITATIONS]
        if interpretations is not None:
            sections+=['## Physical context (explicit applied/history snapshots)',
                'Transport regime and conditional modes are independent interpretations; they never change the stored inversion. Complete values, units, conditions and parent identities: interpretations.json and tables/PhysicalContext.csv.',
                'Bi=kL/(2D) uses full L in cm. D/k is half-length; 2D/k is symmetric slab full thickness at Bi=1. Legacy peak-ratio numbers are not quantitative Bi.']
            for row in interpretations['history']:
                tr=row['transport'];sections+=[f'Spectrum {row["target_sha256"]}; source={tr["source_kind"]}; model={tr["model_id"]}; parent reliability={tr["parent_reliability"]}.']
        for prefix, j in records(archive):
            sections += [f'## {prefix}: {j.job_id}', f'Captured job status: {j.status}',
                'Whole source kind: '+(j.prepared.processed.source_kind.value if prefix.startswith('w') and j.prepared is not None else
                    j.request.capture.parent_source.source_kind.value if not prefix.startswith('w') and j.request.capture.parent_source is not None else 'unavailable')+
                    '; component curves: fit_component; no simulated curve is exported as experiment.',
                'Full source identity, units, mask, raw mapping, recipe, normalization and solver versions: dct_archive.json.',
                'Stale reasons: '+json.dumps(j.stale_reasons, ensure_ascii=False), 'Failure: '+str(j.failure)]
            if prefix.startswith('w'):
                sections += list(j.warnings)
                a = j.analysis
            else:
                sections += ['Parent Fit and Diagnostics identity: '+json.dumps(j.request.capture.identity.to_dict(), ensure_ascii=False),
                    'Parent gate: original parent reliability is retained; purpose-specific component eligibility is separate.',
                    'Captured parent Diagnostics reliable='+str(j.request.parent_snapshot.diagnostics_result.get('reliable')),
                    *j.request.parent_snapshot.warnings, *j.request.capture.state.warnings]
                if j.request.capture.eligibility is not None:
                    e=j.request.capture.eligibility
                    sections += [f'Component eligibility: {e.policy_version}; can_run={e.can_run}; original parent={e.parent_reliability}; r1={e.residual_lag1}; abs limit={e.residual_limit}',
                        'Correlated-error CI calibration remains unsupported. Components are parent-model derived, not independent experiments.']
                a = j.bundle.overall_analysis if j.bundle else None
                if j.bundle:
                    sections += [f'S_total={j.bundle.S_total:.17g}; delta={j.bundle.delta:.17g}',
                        'Unavailable components: '+json.dumps(j.bundle.unavailable_reasons, ensure_ascii=False), *j.bundle.state.warnings]
            if a is not None:
                n = a.source.normalization
                if getattr(a,'options',None) is not None:
                    sections += [f'Whole Signed mode: {a.options.mode}; selection: {a.options.selection}; score: {a.score_name}. No nonlinear GCV.',
                        'Convex objective: weighted SSE + lambda*curvature + mass_ratio*sqrt(lambda)*absolute nodal mass; open edges.',
                        'Frozen mode/noise declaration: '+json.dumps(a.options.to_dict(),ensure_ascii=False)]
                sections += [f'sigma_infinity={n.reference_value:.17g}; S={n.scale_value:.17g} {n.spec.signal_unit}',
                    f'Terminal source: {n.spec.reference_kind}; evidence: {n.spec.reference_evidence}',
                    'Resolution: '+a.resolution_status, 'Flags: '+', '.join(a.flags), *a.state.warnings, *a.interpretation]
        sections += ['## Precision and naming', 'JSON roundtrips float64; CSV uses 17 significant digits. XLSX numeric cells use 15 significant digits.',
            'source_raw and experiment/simulation_clean/simulation_observed retain their declared source kind; physical_fit: estimated parent Fit; component: model-derived contribution; dct: time-scale reconstruction.',
            'All indices are zero based. chi is density per decade; mass includes quadrature weight. CSV units and full counts are in manifest.json.',
            'Stored lambda candidates, failures, peak signed/absolute areas and sensitivity controls are retained in JSON; no new CI is generated.']
        (staging/'report.md').write_text('\n\n'.join(sections)+'\n', encoding='utf-8')
        artifacts = {p.relative_to(staging).as_posix():dict(sha256=file_sha256(p), bytes=p.stat().st_size)
            for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest = dict(kind='dct_report_bundle', schema_version=2 if interpretations is not None else 1, archive_sha256=canonical_sha256(archive.to_dict()),
            exported_at=datetime.now(timezone.utc).isoformat(), artifacts=artifacts,
            tables={t.name:t.info() for t in tables}, options=options.to_dict(), figure_formats=list(formats), xlsx=xlsx,
            precision=dict(csv='17 significant digits; exact float64 readback', json='float64 roundtrip', xlsx='15 significant digits; approximate'))
        if interpretations is not None:manifest['interpretation_sha256']=canonical_sha256(interpretations)
        from .capability_evidence import component_records
        evidence=component_records(archive)
        if evidence:manifest['capability_evidence']=evidence
        manifest['manifest_sha256'] = canonical_sha256(manifest)
        _write_json(staging/'manifest.json', manifest)
        progress(.85, 'Validating staged hashes, units, counts and every scientific CSV value')
        read_dct_bundle(staging, check=check)
        check()
        if before_publish:
            before_publish()
        check()
        if target.exists():
            require(original is not None and read_dct_bundle(target).manifest['manifest_sha256'] == original,
                'DCT destination changed during staging')
        else:
            require(original is None, 'DCT destination removed during staging')
        from pyecr.project.repository import _atomic_directory_replace
        _atomic_directory_replace(staging, target)
        staging = None
        # No fallible callbacks after commit: an exception must not imply rollback after success.
        return DctReportBundle(target, archive, manifest)
    finally:
        if staging is not None and staging.exists():
            require(staging.resolve().parent == target.parent and staging.name.startswith(f'.{target.name}.dct-tmp-'),
                'refusing cleanup outside owned DCT staging directory')
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)
