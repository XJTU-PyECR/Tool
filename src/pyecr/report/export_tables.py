"""Versioned flat tables made only from captured analysis objects."""
from dataclasses import dataclass
from typing import Callable, Mapping
import json
import numpy as np
from pyecr.report.export_options import field_dictionary
from pyecr.report.snapshot import _json_safe, _sha256_json


def json_value(value):
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ExportTable:
    name: str
    columns: tuple[str, ...]
    types: tuple[str, ...]
    units: Mapping[str, str]
    row_count: int
    rows: Callable
    status: str = "captured"
    reason: str | None = None

    def info(self):
        return {"columns": list(self.columns), "types": list(self.types), "units": dict(self.units),
                "rows": self.row_count, "status": self.status, "reason": self.reason,
                "file": f"tables/{self.name}.csv"}


def _table(name, rows, *, units=None, count=None, status="captured", reason=None):
    spec = field_dictionary(scientific_extensions=name in ('ScientificEvidence','TransportRegime'))["tables"][name]
    if not callable(rows):
        values = tuple(rows)
        count = len(values)
        rows = lambda: iter(values)
    return ExportTable(name, tuple(spec["columns"]), tuple(spec["types"]), units or {}, count, rows, status, reason)


def _flatten(value, prefix=""):
    if isinstance(value, Mapping) and value:
        for key in sorted(value):
            item = value[key]
            yield from _flatten(item, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield (prefix, json_value(value), "unavailable" if value is None else "captured",
               "null_in_captured_result" if value is None else None)


def parameter_rows(snapshot, decomposition=None):
    from pyecr.report.bundle import _diagnostic_parameter_rows, _parameter_unit
    c, f = snapshot.analysis_config, snapshot.fit_result
    name = c.model_id.value
    record = f.diagnostics.get("fixed_d", f.diagnostics.get("dual_surface_k", f.diagnostics.get("surface_k", {})))
    fixed = dict(record.get("fixed_parameters", c.model.fixed_parameters))
    bounds = record.get("parameter_bounds", c.model.parameter_bounds)
    if name in {"M0_SINGLE_D", "M1_SINGLE_DK"}:
        single = c.model_specific.get("single_ecr", {})
        defaults = {"amplitude": 1., "baseline": 0.}
        fixed.update({n: v for n, v in defaults.items() if n not in f.params and n not in fixed})
    if decomposition is not None:
        if "y0" not in f.params:
            fixed["y0"] = decomposition.y0
    from pyecr.report.scientific_evidence import parameter_output_rows,output_view
    diagnostic = parameter_output_rows(snapshot) if snapshot.scientific_evidence else _diagnostic_parameter_rows(snapshot)
    selected=output_view(snapshot)
    conditions = record.get("ci_conditions", {
        "geometry": c.model.geometry.to_dict(), "fixed_parameters": fixed,
        "step_type": c.model.step_type, "sign_H": c.model.sign_H, "sign_O": c.model.sign_O,
    })
    scope = str(record.get("ci_scope", "conditional_on_fixed_inputs"))
    evaluation = snapshot.diagnostics_result.get('evaluation')
    derived = {}
    for key in ("g_cm_inv", "lambda_s_inv", "tau_surface_exchange_s", "lambda_1_s_inv", "lambda_2_s_inv",
                "tau_1_surface_exchange_s", "tau_2_surface_exchange_s"):
        if key in record:
            derived[key] = record[key]
    if decomposition is not None:
        derived.update(signed_B=decomposition.signed_amplitudes[0], signed_C=decomposition.signed_amplitudes[1])
    roles = [(n, v, "free") for n, v in f.params.items()]
    roles += [(n, fixed[n], "fixed") for n in sorted(fixed) if n not in f.params]
    roles += [(n, v, "derived") for n, v in derived.items()]
    for n, value, role in roles:
        row = diagnostic.get(n, {})
        lower, upper = row.get("ci_lower"), row.get("ci_upper")
        reason = row.get("ci_reason")
        method = row.get('output_method',"captured_local_covariance_student_t") if lower is not None and upper is not None else "not_captured"
        status = ("available" if row.get("reliable") else "unreliable") if method != "not_captured" else "unavailable"
        if evaluation is not None and method != 'not_captured':
            status = 'diagnostic_' + evaluation['interval_status']
            reason = ('Local parameter precision=' + str(row.get('reliable'))
                      + '; local CI=' + evaluation['interval_status'] + '; coverage is conditional, not calibrated for unsupported errors.')
            if not snapshot.scientific_evidence:
                method = selected['method']
                conditions = selected['conditions']
        if role == "fixed":
            lower = upper = None
            status, reason, method = "not_applicable", "fixed_input_no_free_CI", "none"
        if role == "derived":
            interval_key = n.replace("_s_inv", "_ci_s_inv") if n.startswith("lambda") else n.replace("_exchange_s", "_exchange_ci_s") if n.startswith("tau_") else None
            if interval_key in record:
                lower, upper = record[interval_key]
                method, status = "captured_transformed_local_interval", "available"
            else:
                lower = upper = None
                status, reason, method = "unavailable", "derived_interval_not_captured", "none"
        reason = reason or ("diagnostics_or_interval_not_captured" if status == "unavailable" else None)
        unit = record.get("units", {}).get(n, _parameter_unit(snapshot, n))
        unit = "cm^-1" if n == "g_cm_inv" else "s^-1" if n.startswith("lambda") else "s" if n.startswith("tau_") else unit
        b = bounds.get(n, (None, None)) if role == "free" else (None, None)
        fixed_source = record.get("sources", {}).get(n)
        if role == "fixed" and fixed_source is None:
            fixed_source = {"kind": "configured" if n in c.model.fixed_parameters else "recorded_model_default", "value": value}
        yield (n, value, unit, role, f.param_space.get(n), *b, lower, upper, method,
               selected['confidence_level'] if role=='free' and snapshot.scientific_evidence else .95 if lower is not None and upper is not None else None, scope, json_value(selected['conditions'] if role=='free' and snapshot.scientific_evidence else conditions), status, reason,
               json_value(fixed_source) if fixed_source is not None else None)


def build_tables(snapshot, batch=None):
    from pyecr.plotting.decomposition import DUAL_IDS, build_decomposition, DecompositionViewOptions
    d, f, c = snapshot.dataset, snapshot.fit_result, snapshot.analysis_config
    raw_indices = np.flatnonzero(d.mask)
    unit = str(d.metadata.get("signal_unit", "unknown"))
    raw_unit = str(d.metadata.get("raw_signal_unit", "unknown"))
    raw_time_unit = str(d.metadata.get("raw_time_unit", "unknown"))
    if d.metadata.get("time_unit", "s") != "s":
        raise ValueError("v2 fit export requires captured fit time in seconds")
    if not np.array_equal(f.residuals, d.processed_signal - f.y_fit):
        raise ValueError("captured full-data residual binding mismatch")
    if not np.isclose(f.SSE, float(f.residuals @ f.residuals), rtol=1e-10, atol=1e-14):
        raise ValueError("captured full-data SSE binding mismatch")
    context = snapshot.export_context
    reasons = context.get("row_reasons", {})
    status = context.get("row_audit_status", "incomplete_no_captured_recipe")
    positions = {int(raw): i for i, raw in enumerate(raw_indices)}
    source_rows = d.metadata.get("source_row_numbers")
    if source_rows is not None and len(source_rows) != len(d.raw_time):
        raise ValueError("source_row_numbers must align with raw observations")
    tables = [
        _table("RawData", lambda: ((i, None if source_rows is None else source_rows[i],
            "physical_source_row_mapping_not_recorded" if source_rows is None else "recorded_mapping",
            float(t), float(y), raw_time_unit, raw_unit, snapshot.source_sha256, bool(d.mask[i]))
            for i, (t, y) in enumerate(zip(d.raw_time, d.raw_signal, strict=True))),
            count=len(d.raw_time), units={"raw_time": raw_time_unit, "raw_signal": raw_unit}),
        _table("RowAudit", lambda: ((i, positions.get(i), bool(d.mask[i]), "retained" if d.mask[i] else "excluded",
            None if d.mask[i] else reasons.get(str(i), {}).get("reason", "unknown_missing_captured_recipe"),
            json_value(reasons.get(str(i), {}).get("step_ids", [])), status) for i in range(len(d.raw_time))), count=len(d.raw_time), status=status),
        _table("ProcessedData", lambda: ((i, int(raw), float(t), float(y), "s", unit, True)
            for i, (raw, t, y) in enumerate(zip(raw_indices, d.processed_time, d.processed_signal, strict=True))),
            count=len(raw_indices), units={"time": "s", "observed": unit}),
        _table("FitCurve", lambda: ((i, int(raw), float(t), float(y), float(p), float(r), unit, True)
            for i, (raw, t, y, p, r) in enumerate(zip(raw_indices, d.processed_time, d.processed_signal, f.y_fit, f.residuals, strict=True))),
            count=len(raw_indices), units={"time_s": "s", "observed": unit, "fitted": unit, "residual_observed_minus_fitted": unit}),
    ]
    decomposition = None
    if c.model_id.value in DUAL_IDS:
        decomposition = build_decomposition(d, c, f, DecompositionViewOptions.from_dict(snapshot.decomposition_view) if snapshot.decomposition_view else None)
        s = decomposition
        g = s.observation
        a, b = g.contributions(s.y0, "shifted")
        B, C = s.signed_amplitudes
        tables.append(_table("Components", lambda: ((i, int(raw_indices[i]), float(t), s.y0, g.component_1[i], g.component_2[i], a[i], b[i],
            None if B == 0 else g.component_1[i]/B, None if C == 0 else g.component_2[i]/C,
            "available" if B != 0 and C != 0 else "unavailable_for_zero_amplitude", *s.labels, s.label_semantics, unit) for i,t in enumerate(g.time_s)),
            count=len(raw_indices), units={**{n: unit for n in ("y0","component_1","component_2","shifted_1","shifted_2")},"time_s":"s","X1":"1","X2":"1"}))
    else:
        tables.append(_table("Components", [], status="not_applicable", reason="single_process_model"))
    tables.append(_table("Parameters", parameter_rows(snapshot, decomposition)))
    if snapshot.scientific_evidence:
        from pyecr.report.scientific_evidence import plain
        e=plain(snapshot.scientific_evidence)
        tables.append(_table('ScientificEvidence',_flatten(e)))
        tr=e['transport'];processes={p['process']:p for p in tr['processes']}
        from pyecr.contracts import TransportSummary
        summary=TransportSummary.from_dict(tr)
        rows=[]
        for axis in tr['axes']:
            p=processes[axis['process']]
            rows.append((axis['process'],axis['axis'],p['D_cm2_s'],p['k_cm_s'],axis['full_dimension_cm'],axis['full_dimension_cm']/2,axis['value'],
                p['half_length_cm'],p['slab_full_thickness_cm'],axis['regime'],tr['parent_reliability'],tr['source_kind'],
                summary.point_id,summary.content_sha256,axis['reason'] or None))
        tables.append(_table('TransportRegime',rows,units={n:u for n,u in [('D_cm2_s','cm^2/s'),('k_cm_s','cm/s'),('full_size_cm','cm'),('half_size_cm','cm'),('Bi','1'),('half_length_D_over_k_cm','cm'),('slab_full_thickness_2D_over_k_cm','cm')]}))
    diagnostics = {"captured_diagnostics": snapshot.diagnostics_result, "fit_diagnostics": f.diagnostics,
                   "statistics": {n: getattr(f,n) for n in ("SSE","RMSE","R2","AIC","AICc","BIC","exit_status","nfev")},
                   "fit_eligible_count": len(raw_indices), "free_parameter_count": len(f.params), "warnings": snapshot.warnings}
    tables.append(_table("Diagnostics", _flatten(diagnostics)))
    tables.append(_table("History", ((i, r.operation, json_value(r.parameters), r.timestamp) for i,r in enumerate(d.history, 1))))
    metadata = {"dataset": d.metadata, "project": context.get("project_metadata", {}),
                "geometry": c.model.geometry.to_dict(), "model_id": c.model_id.value, "model_version": snapshot.model_version,
                "selected_model_id": decomposition.model_id if decomposition is not None else f.diagnostics.get("model_id",c.model_id.value),
                "software_version": snapshot.software_version, "seed": snapshot.seed, "stochastic":snapshot.stochastic,
                "fully_reproducible":snapshot.fully_reproducible,"captured_at": context.get("captured_at"),
                "dataset_sha256": snapshot.dataset_sha256, "source_sha256": snapshot.source_sha256,
                "config_sha256": _sha256_json(c.to_dict()), "fit_result_sha256": _sha256_json(f.to_dict())}
    from pyecr.report.simulation_fit import simulation_comparison, simulation_fit_tables
    comparison = simulation_comparison(snapshot)
    if comparison is not None:
        metadata['simulation_comparison'] = comparison
        tables.extend(simulation_fit_tables(snapshot, comparison))
    tables.append(_table("Metadata", _flatten(metadata)))
    records = [] if batch is None else batch["records"]
    batch_rows = ((r["record_id"], r.get("temperature_K"), r["model_id"], "failed" if not r["fit_success"] else "completed" if r["reliable"] else "unreliable",
                   r["reliable"], r.get("error"), json_value(r["params"]), json_value(r["parameter_ci"]), r.get("SSE"), r.get("RMSE"), r.get("R2"),
                   None if r.get("fit_result") is None else _sha256_json(r["fit_result"])) for r in records)
    tables.append(_table("BatchSummary", batch_rows, status="captured" if batch is not None else "not_captured", reason=None if batch is not None else "batch_result_not_captured"))
    return tuple(tables), decomposition, diagnostics, metadata
