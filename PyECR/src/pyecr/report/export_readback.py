"""Real-file integrity and scientific roundtrip checks for export v2."""
from pathlib import Path, PurePosixPath
import csv
import json
import hashlib
from pyecr.report.snapshot import AnalysisSnapshot, _sha256_json
from pyecr.report.export_options import VERSION, decode_text
from pyecr.report.export_tables import build_tables


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode(value, kind):
    if value == "" and kind.endswith("?"):
        return None
    if kind.startswith("float"):
        return float(value)
    if kind.startswith("integer"):
        return int(value)
    if kind.startswith("boolean"):
        if value not in {"true", "false"}:
            raise ValueError("CSV boolean must be true/false")
        return value == "true"
    if kind.startswith("json"):
        return json.loads(value)
    return decode_text(value)


def verify_export_v2(root, *, check=lambda: None):
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("version") != VERSION:
        raise ValueError("unsupported export manifest version")
    payload = {k: v for k,v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha256_json(payload):
        raise ValueError("export manifest payload hash mismatch")
    for relative, info in manifest["artifacts"].items():
        check()
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
            raise ValueError("unsafe export manifest path")
        path = root/relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("unsafe export artifact path")
        if not path.is_file() or path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
            raise ValueError(f"export artifact hash/size mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["artifacts"]):
        raise ValueError("export manifest file inventory mismatch")
    snapshot = AnalysisSnapshot.from_dict(json.loads((root/"analysis.json").read_text(encoding="utf-8")))
    from dataclasses import replace
    order = manifest["source"].get("free_parameter_order")
    if order is None or len(order) != len(snapshot.fit_result.params) or set(order) != set(snapshot.fit_result.params):
        raise ValueError("export free parameter column order missing/mismatch")
    f = snapshot.fit_result
    snapshot = replace(snapshot, fit_result=replace(f, params={n:f.params[n] for n in order}, param_space={n:f.param_space[n] for n in order}))
    if snapshot.scientific_evidence:
        from pyecr.report.scientific_evidence import verify_root_csv
        verify_root_csv(snapshot,root)
    from pyecr.report.simulation_fit import verify_simulation_fit_evidence
    verify_simulation_fit_evidence(snapshot, root)
    from pyecr.project.interpretation_archive import verify_dct_evidence
    from pyecr.report.export_options import ExportOptions
    verify_dct_evidence(snapshot,root/'dct',ExportOptions(**manifest['options']),check)
    if (root/"experimental_m1_uncertainty.json").exists():
        from pyecr.services.m1_deep_uncertainty import ExperimentalM1UncertaintyResult
        ExperimentalM1UncertaintyResult.reopen(root/"experimental_m1_uncertainty.json", snapshot.dataset, snapshot.analysis_config, snapshot.fit_result)
    expected_source = manifest["source"]
    for key,value in (('scientific_evidence_sha256',snapshot.scientific_evidence),('dct_evidence_sha256',snapshot.dct_evidence)):
        if value:
            if expected_source.get(key)!=_sha256_json(value):raise ValueError('Export frozen scientific source hash conflict: '+key)
        elif key in expected_source:raise ValueError('Export claims missing result evidence: '+key)
    if 'seed' in expected_source:
        if expected_source['seed']!=snapshot.seed or expected_source.get('stochastic')!=snapshot.stochastic:
            raise ValueError('Report seed/stochastic provenance differs from AnalysisSnapshot')
        if expected_source.get('fitting_randomness')!=snapshot.fit_result.to_dict()['diagnostics'].get('fitting_randomness'):
            raise ValueError('Report fitting randomness provenance differs from FitResult')
    for key, value in (("dataset_sha256", snapshot.dataset_sha256), ("config_sha256", _sha256_json(snapshot.analysis_config.to_dict())),
                       ("fit_result_sha256", _sha256_json(snapshot.fit_result.to_dict())), ("snapshot_sha256", _sha256_json(snapshot.to_dict()))):
        if expected_source.get(key) != value:
            raise ValueError(f"export source binding mismatch: {key}")
    batch = json.loads((root/"batch.json").read_text(encoding="utf-8")) if (root/"batch.json").exists() else None
    if (None if batch is None else _sha256_json(batch)) != expected_source.get("batch_sha256"):
        raise ValueError("export batch source binding mismatch")
    tables, _, _, _ = build_tables(snapshot, batch)
    csv.field_size_limit(64*1024*1024)
    for table in tables:
        info = manifest["tables"].get(table.name)
        if info is None or info != table.info():
            raise ValueError(f"export table dictionary mismatch: {table.name}")
        count = 0
        with (root/info["file"]).open(encoding=manifest["options"]["csv_encoding"], newline="") as stream:
            reader = csv.reader(stream, delimiter=manifest["options"]["csv_delimiter"])
            if next(reader) != list(table.columns):
                raise ValueError(f"export CSV header mismatch: {table.name}")
            for actual_row, expected in zip(reader, table.rows(), strict=True):
                if count % 1024 == 0:
                    check()
                if len(actual_row) != len(table.columns):
                    raise ValueError(f"export CSV column count mismatch: {table.name}")
                for actual_cell, original, kind in zip(actual_row, expected, table.types, strict=True):
                    a = _decode(actual_cell, kind)
                    b = json.loads(original) if kind.startswith("json") and original is not None else original
                    if a != b:
                        raise ValueError(f"scientific CSV roundtrip mismatch: {table.name} row {count}")
                count += 1
        if count != table.row_count:
            raise ValueError(f"export CSV count mismatch: {table.name}")
    if manifest.get("xlsx") is not None:
        from pyecr.report.export_xlsx import verify_xlsx_tables
        verify_xlsx_tables(tables, root/"analysis.xlsx", manifest["xlsx"], check)
    check()
    return {"status":"verified", "scientific_csv_exact":True, "fit_eligible_rows":len(snapshot.fit_result.y_fit),
            "raw_rows":len(snapshot.dataset.raw_time), "SSE":snapshot.fit_result.SSE, "manifest":manifest}
