"""Atomic complete v2 packages; old root CSV/JSON remain compatible."""
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import csv
import json
import os
import shutil
from pyecr.report.staging import create_export_staging
from pyecr.report.snapshot import _sha256_json, _json_safe
from pyecr.report.export_options import ExportOptions, ExportPreset, VERSION, encode_text, field_dictionary
from pyecr.report.export_tables import build_tables, json_value
from pyecr.report.export_readback import verify_export_v2, file_sha256


def _write_json(path, value):
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def _write_readable_report(snapshot,root,experimental_uncertainty):
    """Existing report template, with honest preset-dependent figure availability."""
    import re
    from pyecr.report.bundle import _report_html
    html=_report_html(snapshot,experimental_uncertainty)
    # Only the renderer's local image tags are adapted. Source text is escaped
    # by the existing template; no labels, numbers or result data are rewritten.
    html=re.sub(r'<img\b[^>]*\bsrc="(figures/[^"]+)"[^>]*>',
        lambda m:m.group(0) if (root/m.group(1)).is_file() else '<p>Figure not included in this preset. Use the publication preset for figures; scientific arrays remain in CSV/JSON.</p>',html)
    (root/'report.html').write_text(html,encoding='utf-8')


def _write_table(root, table, options, check):
    path = root/table.info()["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding=options.csv_encoding, newline="") as stream:
        writer = csv.writer(stream, delimiter=options.csv_delimiter, lineterminator="\n")
        writer.writerow(table.columns)
        for row in table.rows():
            if count % 1024 == 0:
                check()
            fields = []
            for value, kind in zip(row, table.types, strict=True):
                if value is None:
                    fields.append("")
                elif kind.startswith("float"):
                    fields.append(format(float(value), ".17g"))
                elif kind.startswith("boolean"):
                    fields.append(str(bool(value)).lower())
                elif kind.startswith("text"):
                    fields.append(encode_text(str(value)))
                else:
                    fields.append(str(value))
            writer.writerow(fields)
            count += 1
    if count != table.row_count:
        raise ValueError(f"table {table.name} row count mismatch")


def export_v2(snapshot, destination, options, *, export_scope="captured", batch_result=None,
              cancel_check=None, before_publish=None, progress_callback=None, experimental_uncertainty=None):
    from pyecr.report.bundle import ReportBundle, _write_analysis_json, _write_parameters_csv, _write_processed_data_csv, _write_fit_curve_csv
    if not isinstance(options, ExportOptions):
        raise TypeError("ExportOptions required")
    if export_scope not in {"captured", "current", "historical"}:
        raise ValueError("export scope must be captured/current/historical")
    if export_scope == "current" and not callable(before_publish):
        raise ValueError("current export requires a revision guard")
    if experimental_uncertainty is not None:
        experimental_uncertainty.validate_provenance(snapshot.dataset, snapshot.analysis_config, snapshot.fit_result)
    from pyecr.services.batch_analysis import BatchAnalysisResult
    if batch_result is not None and not isinstance(batch_result, BatchAnalysisResult):
        raise TypeError("captured BatchAnalysisResult required")
    batch = None if batch_result is None else batch_result.to_dict()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError("export destination already exists; select a new directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent/f".{target.name}.export.lock"
    descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    os.close(descriptor)
    temporary = None

    def check():
        if cancel_check:
            cancel_check()

    def progress(value, message):
        if progress_callback:
            progress_callback(value, message)

    try:
        check()
        temporary = create_export_staging(target.parent,prefix=f".{target.name}.tmp-")
        tables, decomposition, diagnostics, metadata = build_tables(snapshot, batch)
        _write_analysis_json(snapshot, temporary/"analysis.json")
        _write_parameters_csv(snapshot, temporary/"parameters.csv")
        _write_processed_data_csv(snapshot, temporary/"processed_data.csv")
        _write_fit_curve_csv(snapshot, temporary/"fit_curve.csv")
        from pyecr.project.interpretation_archive import write_dct_evidence
        write_dct_evidence(snapshot,temporary/'dct',options,check)
        from pyecr.report.simulation_fit import write_simulation_fit_evidence
        write_simulation_fit_evidence(snapshot, temporary, check=check)
        from pyecr.report.bundle import (_sensitivity_mapping, _write_sensitivity_csv, _write_error_map_csv,
            _write_error_map_json, _write_dual_map_csv, _write_dual_map_json, _write_decomposition)
        sensitivity = _sensitivity_mapping(snapshot.diagnostics_result)
        if (snapshot.analysis_config.model_id.value in {"M1_SINGLE_DK","M3_DUAL_HO_DK"}
                and sensitivity and sensitivity.get("available") is True and sensitivity.get("curves")):
            _write_sensitivity_csv(snapshot, temporary/"sensitivity.csv")
        comparison=snapshot.optional_preprocessing_comparison
        if comparison is not None and comparison['before'].get('residual_curve') is not None:
            from pyecr.report.bundle import _write_preprocessing_residual_tables
            _write_preprocessing_residual_tables(comparison,temporary/'preprocessing_residual_comparison.csv')
        if snapshot.error_map_result is not None:
            _write_error_map_csv(snapshot,temporary/"error_map.csv")
            _write_error_map_json(snapshot,temporary/"error_map.json")
        if snapshot.dual_error_map_result is not None:
            for process,slice_ in (("H",snapshot.dual_error_map_result.h_map),("O",snapshot.dual_error_map_result.o_map)):
                _write_dual_map_csv(slice_,temporary/f"error_map_{process}.csv")
                _write_dual_map_json(snapshot,slice_,temporary/f"error_map_{process}.json")
        if snapshot.decomposition_view:
            _write_decomposition(snapshot,temporary,temporary/"figures")
        for i, table in enumerate(tables):
            check()
            _write_table(temporary, table, options, check)
            progress(.1+.35*(i+1)/len(tables), f"Wrote {table.name}")
        for file, value in (("diagnostics.json",diagnostics),("history.json",[r.to_dict() for r in snapshot.dataset.history]),
                            ("metadata.json",metadata),("data_dictionary.json",field_dictionary(scientific_extensions=bool(snapshot.scientific_evidence)))):
            check()
            _write_json(temporary/file,value)
        if batch is not None:
            _write_json(temporary/"batch.json", batch)
        if experimental_uncertainty is not None:
            _write_json(temporary/"experimental_m1_uncertainty.json", experimental_uncertainty.to_dict())
        source = {"seed":snapshot.seed,"stochastic":snapshot.stochastic,
                  "fitting_randomness":snapshot.fit_result.to_dict()['diagnostics'].get('fitting_randomness'),
                  "dataset_sha256":snapshot.dataset_sha256, "source_sha256":snapshot.source_sha256,
                  "config_sha256":_sha256_json(snapshot.analysis_config.to_dict()),"fit_result_sha256":_sha256_json(snapshot.fit_result.to_dict()),
                  "snapshot_sha256":_sha256_json(snapshot.to_dict()),"batch_sha256":None if batch is None else _sha256_json(batch)}
        source["free_parameter_order"] = list(snapshot.fit_result.params)
        source["requested_model_id"] = snapshot.analysis_config.model_id.value
        source["selected_model_id"] = metadata["selected_model_id"]
        if snapshot.scientific_evidence:
            source['scientific_evidence_sha256']=_sha256_json(snapshot.scientific_evidence)
        if snapshot.dct_evidence:source['dct_evidence_sha256']=_sha256_json(snapshot.dct_evidence)
        precision = {"machine_float":"17 significant digits, float64 roundtrip", "xlsx":"15 significant digits; use JSON/CSV for exact replay",
                     "text":"CSV excel_literal_escape_v1; XLSX literal strings; JSON original text"}
        xlsx = None
        if options.preset is ExportPreset.EXCEL:
            from pyecr.report.export_xlsx import write_xlsx
            from pyecr.report.scientific_evidence import output_view
            uncertainty_metadata = (output_view(snapshot) if snapshot.diagnostics_result.get('evaluation')
                                    else snapshot.scientific_evidence.get('selected_output') if snapshot.scientific_evidence else None)
            xlsx = write_xlsx(tables, temporary/"analysis.xlsx", options,
                {"requested_model":snapshot.analysis_config.model_id.value,"selected_model":source["selected_model_id"],"scope":export_scope,"fit_result_sha256":source["fit_result_sha256"],
                 "SSE":snapshot.fit_result.SSE,"fit_eligible_N":len(snapshot.fit_result.y_fit),"raw_N":len(snapshot.dataset.raw_time),
                 "free_parameters":len(snapshot.fit_result.params),"precision":precision["xlsx"],"captured_at":snapshot.export_context.get("captured_at","not recorded"),
                 **({k:uncertainty_metadata[k] for k in ('confidence_level','method','df')} if uncertainty_metadata else {'confidence_level':.95,'level_origin':'defaulted_legacy'}),
                 'gate_confidence_level':.95}, check)
        if options.preset is ExportPreset.PUBLICATION:
            from pyecr.report.export_figures import write_figures
            write_figures(snapshot, decomposition, temporary/"figures", options, check)
        _write_readable_report(snapshot,temporary,experimental_uncertainty)
        progress(.7,"Building manifest")
        missing = [{"table": t.name, "status":t.status, "reason":t.reason} for t in tables if t.status!="captured"]
        if snapshot.export_context.get("captured_at") is None:
            missing.append({"field":"captured_at","reason":"legacy_snapshot_capture_time_not_recorded"})
        if snapshot.source_sha256 is None:
            missing.append({"field":"source_sha256","reason":"source_file_hash_not_captured"})
        exported_at = datetime.now(timezone.utc).isoformat()
        from pyecr.diagnostics.interpretation import diagnostic_summary, BOUND_LEGEND
        (temporary/"README.md").write_text(
            f"# PyECR export\n\nPreset: {options.preset.value}\nScope: {export_scope}\nExported at: {exported_at}\n"
            f"\nCaptured diagnostic interpretation:\n{diagnostic_summary(snapshot.fit_result, snapshot.diagnostics_result)}\n\n{BOUND_LEGEND}\n"
            "Historical diagnostics remain captured decisions; export does not recalculate gates or refit.\n"
            "Readable frozen results: report.html. ScientificEvidence and TransportRegime tables preserve actual CI, termination, noise and transport sources. DCT reconstructions, if captured, are separate under dct/ and are not physical fitted curves.\n"
            "\nScientific authority: analysis.json and tables/*.csv (17g float64). Indexing is zero based. "
            "Residual = observed - fitted. FitCurve includes every fit-eligible observation. "
            "XLSX has Excel's 15 digit precision; it cannot replace scientific replay. "
            "CSV text columns use reversible leading-apostrophe escaping; numeric negatives stay numeric. "
            "Components are signed effective contributions, not independent carrier conductivities. "
            "No component CI. Display grid never enters residual/SSE/Jacobian/CI. "
            "Missing capture time or row reasons remain explicitly unavailable.\n",encoding="utf-8")
        artifacts = {p.relative_to(temporary).as_posix():{"sha256":file_sha256(p),"bytes":p.stat().st_size} for p in sorted(temporary.rglob("*")) if p.is_file()}
        manifest = {"schema_version":2,"version":VERSION,"preset":options.preset.value,"options":options.to_dict(),
                    "export_scope":export_scope,"exported_at":exported_at,"captured_at":snapshot.export_context.get("captured_at"),
                    "source":source,"artifacts":artifacts,"tables":{t.name:t.info() for t in tables},"xlsx":xlsx,"missing":missing,"precision":precision}
        manifest["manifest_sha256"] = _sha256_json(manifest)
        _write_json(temporary/"manifest.json",manifest)
        progress(.8,"Reading back scientific files")
        verify_export_v2(temporary, check=check)
        check()
        if before_publish:
            before_publish()
        check()
        if target.exists():
            raise FileExistsError("export destination appeared during build")
        temporary.rename(target)
        temporary = None
        progress(1.,"Export committed and verified")
        return ReportBundle(root=target, artifacts={k:v["sha256"] for k,v in artifacts.items()},dataset_sha256=snapshot.dataset_sha256,
                            seed=snapshot.seed,manifest_expected=True,schema_version=2,export_metadata=manifest)
    finally:
        if temporary is not None and temporary.exists():
            if temporary.resolve().parent != target.parent or not temporary.name.startswith(f".{target.name}.tmp-"):
                raise RuntimeError("refusing cleanup outside owned export staging directory")
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)
