"""Deterministic directory report bundles generated from AnalysisSnapshot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from pyecr.report.staging import create_export_staging
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Protocol

from pyecr.plotting import render_fit_figure, render_residual_figure
from pyecr.plotting.matplotlib import render_dual_error_map_figure, render_error_map_figure
from pyecr.report.snapshot import AnalysisSnapshot
from pyecr.diagnostics.interpretation import diagnostic_summary, BOUND_LEGEND

if TYPE_CHECKING:
    from pyecr.services.m1_deep_uncertainty import ExperimentalM1UncertaintyResult

__all__ = ["PdfRenderer", "ReportBundle"]


class PdfRenderer(Protocol):
    """Optional adapter supplied by clients that need a PDF artifact."""

    def render(self, html_path: Path, pdf_path: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class ReportBundle:
    """Exported report root and SHA256 manifest held in memory."""

    root: Path
    artifacts: Mapping[str, str]
    dataset_sha256: str
    seed: int | None
    manifest_expected: bool = False
    error_maps_metadata: Mapping[str, object] | None = None
    schema_version: int = 1
    export_metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise ValueError("ReportBundle root must be an existing directory")
        object.__setattr__(self, "root", root)
        object.__setattr__(
            self, "artifacts", MappingProxyType(dict(self.artifacts))
        )
        if self.error_maps_metadata is not None:
            object.__setattr__(
                self, "error_maps_metadata",
                MappingProxyType(dict(_plain_json(self.error_maps_metadata))),
            )

    @classmethod
    def export(
        cls,
        snapshot: AnalysisSnapshot,
        destination: str | Path,
        *,
        pdf_renderer: PdfRenderer | None = None,
        experimental_uncertainty: ExperimentalM1UncertaintyResult | None = None,
        export_options=None,
        export_scope: str = "captured",
        batch_result=None,
        cancel_check=None,
        before_publish=None,
        progress_callback=None,
    ) -> ReportBundle:
        """Atomically create a report from one immutable analysis snapshot."""
        from pyecr.fitting.reproducibility import require_reproducibility
        if require_reproducibility(snapshot.fit_result)!=snapshot.seed:
            raise ValueError('Report snapshot seed differs from actual fit seed')

        if not isinstance(snapshot, AnalysisSnapshot):
            raise TypeError("snapshot must be an AnalysisSnapshot")
        if export_options is not None:
            from pyecr.report.export_v2 import export_v2
            return export_v2(snapshot, destination, export_options, export_scope=export_scope,
                batch_result=batch_result, cancel_check=cancel_check, before_publish=before_publish,
                progress_callback=progress_callback, experimental_uncertainty=experimental_uncertainty)
        if experimental_uncertainty is not None:
            experimental_uncertainty.validate_provenance(
                snapshot.dataset, snapshot.analysis_config, snapshot.fit_result,
            )
        target = Path(destination).resolve()
        if target.exists():
            raise FileExistsError(
                f"report destination already exists; choose a new directory: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        def check():
            if cancel_check is not None:cancel_check()
        check()
        temporary = create_export_staging(target.parent,prefix=f".{target.name}.tmp-")
        try:
            check()
            _write_analysis_json(snapshot, temporary / "analysis.json")
            if experimental_uncertainty is not None:
                _write_text(
                    temporary / "experimental_m1_uncertainty.json",
                    json.dumps(experimental_uncertainty.to_dict(), ensure_ascii=False,
                               sort_keys=True, indent=2, allow_nan=False) + "\n",
                )
            _write_parameters_csv(snapshot, temporary / "parameters.csv")
            _write_processed_data_csv(snapshot, temporary / "processed_data.csv")
            _write_fit_curve_csv(snapshot, temporary / "fit_curve.csv")
            from pyecr.project.interpretation_archive import write_dct_evidence,verify_dct_evidence
            from pyecr.report.export_options import ExportOptions
            write_dct_evidence(snapshot,temporary/'dct',ExportOptions(),check)
            verify_dct_evidence(snapshot,temporary/'dct',ExportOptions(),check)
            check()
            from pyecr.report.simulation_fit import write_simulation_fit_evidence
            write_simulation_fit_evidence(snapshot, temporary, write_tables=True)
            sensitivity = _sensitivity_mapping(snapshot.diagnostics_result)
            if (
                snapshot.analysis_config.model_id.value in
                {"M1_SINGLE_DK", "M3_DUAL_HO_DK"}
                and sensitivity is not None
                and sensitivity.get("available") is True
                and sensitivity.get("curves")
            ):
                _write_sensitivity_csv(snapshot, temporary / "sensitivity.csv")
            figure_root = temporary / "figures"
            plot_views=snapshot.export_context.get('project_metadata',{}).get('plot_views')
            render_fit_figure(snapshot.fit_plot_data, figure_root / "fit.png", plot_views=plot_views)
            render_residual_figure(
                snapshot.fit_plot_data, figure_root / "residuals.png", plot_views=plot_views
            )
            if snapshot.decomposition_view:
                _write_decomposition(snapshot, temporary, figure_root)
            comparison = snapshot.optional_preprocessing_comparison
            if comparison is not None and comparison["before"].get("residual_curve") is not None:
                _write_preprocessing_residual_comparison(comparison, temporary / "preprocessing_residual_comparison.csv", figure_root / "preprocessing_residuals.png")
            if snapshot.error_map_result is not None:
                _write_error_map_csv(snapshot, temporary / "error_map.csv")
                _write_error_map_json(snapshot, temporary / "error_map.json")
            if snapshot.dual_error_map_result is not None:
                for process, slice_ in (
                    ("H", snapshot.dual_error_map_result.h_map),
                    ("O", snapshot.dual_error_map_result.o_map),
                ):
                    _write_dual_map_csv(slice_, temporary / f"error_map_{process}.csv")
                    _write_dual_map_json(snapshot, slice_, temporary / f"error_map_{process}.json")
            from pyecr.report.map_figures import write_map_figures
            write_map_figures(snapshot,figure_root)
            html_path = temporary / "report.html"
            check()
            _write_text(html_path, _report_html(snapshot, experimental_uncertainty))
            if pdf_renderer is not None:
                render = getattr(pdf_renderer, "render", None)
                if not callable(render):
                    raise TypeError("pdf_renderer must provide render(html_path, pdf_path)")
                render(html_path, temporary / "report.pdf")
            artifacts = _artifact_hashes(temporary)
            error_maps_metadata = (
                _dual_map_manifest_metadata(snapshot, artifacts)
                if snapshot.dual_error_map_result is not None else None
            )
            manifest_payload: dict[str, object] = {
                "schema_version": 1, "artifacts": artifacts,
            }
            if error_maps_metadata is not None:
                manifest_payload["error_maps"] = error_maps_metadata
            _write_text(
                temporary / "manifest.json",
                json.dumps(manifest_payload,
                           ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            )
            if snapshot.scientific_evidence:
                restored=AnalysisSnapshot.from_dict(json.loads((temporary/'analysis.json').read_text(encoding='utf-8')))
            else:restored=snapshot
            from pyecr.report.scientific_evidence import verify_root_csv
            verify_root_csv(restored,temporary)
            cls(root=temporary,artifacts=artifacts,dataset_sha256=snapshot.dataset_sha256,seed=snapshot.seed,
                manifest_expected=True,error_maps_metadata=error_maps_metadata).verify()
            check()
            if before_publish is not None:before_publish()
            check()
            temporary.rename(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return cls(
            root=target,
            artifacts=artifacts,
            dataset_sha256=snapshot.dataset_sha256,
            seed=snapshot.seed,
            manifest_expected=True,
            error_maps_metadata=error_maps_metadata,
        )

    def verify(self) -> None:
        """Raise if a report artifact is missing or no longer matches its hash."""

        if self.schema_version == 2:
            from pyecr.report.export_readback import verify_export_v2
            record = verify_export_v2(self.root)
            if record["manifest"] != self.export_metadata:
                raise ValueError("v2 export manifest differs from captured output")
            if {p: i["sha256"] for p,i in record["manifest"]["artifacts"].items()} != dict(self.artifacts):
                raise ValueError("v2 export artifact hashes differ from captured output")
            return
        manifest = self.root / "manifest.json"
        if self.manifest_expected and not manifest.is_file():
            raise FileNotFoundError("report manifest.json is missing")
        if manifest.is_file():
            declared = json.loads(manifest.read_text(encoding="utf-8"))
            if declared.get("schema_version") != 1 or declared.get("artifacts") != dict(self.artifacts):
                raise ValueError("report manifest does not match artifact hashes")
            if declared.get("error_maps") != (
                None if self.error_maps_metadata is None else _plain_json(self.error_maps_metadata)
            ):
                raise ValueError("report Error Map manifest metadata does not match captured result")

        for relative, expected in self.artifacts.items():
            path = self.root / relative
            if not path.is_file():
                raise FileNotFoundError(f"report artifact is missing: {relative}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"report artifact SHA256 mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )
        analysis_payload = json.loads((self.root/'analysis.json').read_text(encoding='utf-8'))
        if analysis_payload.get('scientific_evidence') is not None:
            restored=AnalysisSnapshot.from_dict(analysis_payload)
            from pyecr.report.scientific_evidence import verify_root_csv
            verify_root_csv(restored,self.root)
            from pyecr.project.interpretation_archive import verify_dct_evidence
            from pyecr.report.export_options import ExportOptions
            verify_dct_evidence(restored,self.root/'dct',ExportOptions(),lambda:None)
        if analysis_payload.get('dataset',{}).get('metadata',{}).get('simulation_origin') is not None:
            from pyecr.report.simulation_fit import verify_simulation_fit_evidence
            verify_simulation_fit_evidence(AnalysisSnapshot.from_dict(analysis_payload),self.root,verify_tables=True)


def _write_analysis_json(snapshot: AnalysisSnapshot, path: Path) -> None:
    content = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    _write_text(path, content + "\n")


def _write_error_map_json(snapshot: AnalysisSnapshot, path: Path) -> None:
    result = snapshot.error_map_result
    config = snapshot.error_map_config
    assert result is not None and config is not None
    from dataclasses import asdict
    _write_text(path, json.dumps({
        "schema_version": 1,
        "dataset_sha256": snapshot.dataset_sha256,
        "config": _plain_json(asdict(config)),
        "result": result.to_dict(),
    }, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _write_error_map_csv(snapshot: AnalysisSnapshot, path: Path) -> None:
    result = snapshot.error_map_result
    assert result is not None
    rows = (
        (d_index, k_index, _number(10.0 ** d_value), _number(10.0 ** k_value),
         _number(d_value), _number(k_value),
         _number(result.objective_grid[d_index, k_index]) if result.valid_mask[d_index, k_index] else "",
         _number(result.relative_grid[d_index, k_index]) if result.valid_mask[d_index, k_index] else "",
         _number(result.log_ratio_grid[d_index, k_index]) if result.valid_mask[d_index, k_index] else "",
         str(bool(result.valid_mask[d_index, k_index])).lower())
        for d_index, d_value in enumerate(result.log10_D)
        for k_index, k_value in enumerate(result.log10_k)
    )
    _write_csv(path, ("d_index", "k_index", "D_cm2_s", "k_cm_s", "log10_D", "log10_k",
                      "SSE", "relative_SSE", "log10_relative_SSE", "valid"), rows)


def _write_dual_map_json(snapshot: AnalysisSnapshot, slice_: object, path: Path) -> None:
    from pyecr.diagnostics.dual_error_map import DualErrorMapSlice

    if not isinstance(slice_, DualErrorMapSlice) or snapshot.dual_error_map_config is None:
        raise TypeError("dual map report requires a captured H/O slice and config")
    payload = {
        "schema_version": 1,
        "dataset_sha256": snapshot.dataset_sha256,
        "selected_model": snapshot.analysis_config.model_id.value,
        "process": slice_.process,
        "units": _dual_map_units(snapshot),
        "config": snapshot.dual_error_map_config.to_dict(),
        "result": slice_.to_dict(),
    }
    _write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n")


def _write_dual_map_csv(slice_: object, path: Path) -> None:
    from pyecr.diagnostics.dual_error_map import DualErrorMapSlice

    if not isinstance(slice_, DualErrorMapSlice):
        raise TypeError("dual map CSV requires a captured H/O slice")
    rows = (
        (slice_.process, d_index, k_index,
         _number(10.0 ** d_value), _number(10.0 ** k_value),
         _number(d_value), _number(k_value),
         _number(slice_.raw_sse_grid[d_index, k_index]) if slice_.valid_mask[d_index, k_index] else "",
         str(bool(slice_.valid_mask[d_index, k_index])).lower())
        for d_index, d_value in enumerate(slice_.log10_D)
        for k_index, k_value in enumerate(slice_.log10_k)
    )
    _write_csv(path, ("process", "d_index", "k_index", "D_cm2_s", "k_cm_s",
                      "log10_D", "log10_k", "raw_SSE", "valid"), rows)


def _dual_map_units(snapshot: AnalysisSnapshot) -> dict[str, str]:
    signal = str(snapshot.dataset.metadata.get("signal_unit", "signal"))
    return {"D": "cm2/s", "k": "cm/s", "SSE": f"({signal})²"}


def _dual_map_manifest_metadata(
    snapshot: AnalysisSnapshot, artifacts: Mapping[str, str],
) -> dict[str, object]:
    result = snapshot.dual_error_map_result
    config = snapshot.dual_error_map_config
    assert result is not None and config is not None
    metadata: dict[str, object] = {}
    for process, slice_ in (("H", result.h_map), ("O", result.o_map)):
        metadata[process] = {
            "objective_kind": slice_.objective_kind,
            "fixed_nuisance": dict(slice_.fixed_nuisance),
            "selected_model": snapshot.analysis_config.model_id.value,
            "revision": asdict(slice_.revision),
            "units": _dual_map_units(snapshot),
            "warnings": list(slice_.warnings),
            "quality": asdict(slice_.quality),
            "range_resolution": config.to_dict(),
            "artifacts": {
                "csv": artifacts[f"error_map_{process}.csv"],
                "json": artifacts[f"error_map_{process}.json"],
                "png": artifacts[f"figures/error_map_{process}.png"],
            },
        }
    return metadata


def _write_parameters_csv(snapshot: AnalysisSnapshot, path: Path) -> None:
    diagnostic_rows = _diagnostic_parameter_rows(snapshot)
    if snapshot.scientific_evidence:
        from pyecr.report.scientific_evidence import parameter_output_rows,output_view
        diagnostic_rows=parameter_output_rows(snapshot)
    guided = snapshot.fit_result.diagnostics.get("guided_refit")
    fixed_parameter = (
        str(guided.get("fixed_parameter")) if isinstance(guided, Mapping) else None
    )
    rows: list[list[object]] = []
    for name, estimate in snapshot.fit_result.params.items():
        diagnostic = diagnostic_rows.get(name, {})
        is_fixed = name == fixed_parameter
        rows.append(
            [
                name,
                _number(estimate),
                _parameter_unit(snapshot, name),
                snapshot.fit_result.param_space[name],
                "" if is_fixed else _optional_number(diagnostic.get("ci_lower")),
                "" if is_fixed else _optional_number(diagnostic.get("ci_upper")),
                "Fixed parameter — no free CI" if is_fixed else diagnostic.get("ci_reason") or "",
                _optional_bool(diagnostic.get("bound_hit")),
                diagnostic.get("bound_side") or "",
                _optional_bool(diagnostic.get("reliable")),
                diagnostic.get("reliability_reason") or "",
            ]
        )
    if fixed_parameter is not None and fixed_parameter not in snapshot.fit_result.params:
        rows.append([
            fixed_parameter,
            _optional_number(guided.get("fixed_value")),  # type: ignore[union-attr]
            _parameter_unit(snapshot, fixed_parameter),
            "fixed", "", "", "Fixed parameter — no free CI",
            "", "", "", "Fixed for Guided Refit",
        ])
    extra=()
    if snapshot.scientific_evidence:
        output=output_view(snapshot);extra=('confidence_level','ci_method','ci_df','ci_level_origin')
        rows=[[*r,_number(output['confidence_level']),output['method'],output['df'],output['level_origin']] for r in rows]
    _write_csv(
        path,
        (
            "parameter",
            "estimate",
            "unit",
            "parameter_space",
            "ci_lower",
            "ci_upper",
            "ci_reason",
            "bound_hit",
            "bound_side",
            "reliable",
            "reliability_reason",*extra,
        ),
        rows,
    )


def _write_preprocessing_residual_tables(comparison: Mapping[str, object], csv_path: Path) -> None:
    rows = []
    for side in ('before','after'):
        curve = comparison[side].get("residual_curve") or {}
        rows.extend((side, str(int(index)), _number(time), _number(residual), str(curve.get("time_unit", "s")), str(curve.get("signal_unit", "")))
            for index, time, residual in zip(curve.get("original_indices", ()), curve.get("time", ()), curve.get("residual", ()), strict=True))
    _write_csv(csv_path, ("stage", "original_raw_index", "processed_time", "residual_observed_minus_fitted", "time_unit", "signal_unit"), rows)
    # Keep the original machine-readable row format; offer side-by-side curves
    # with each stage's own original indices, times and units for spreadsheet use.
    columns=[];values=[]
    for side in ('before','after'):
        curve=comparison[side].get('residual_curve') or {}
        columns.extend((f'{side}_original_raw_index',
            f'{side}_time ({curve.get("time_unit","s")})',
            f'{side}_residual ({curve.get("signal_unit","")})'))
        values.extend((tuple(str(int(v)) for v in curve.get('original_indices',())),
            tuple(_number(v) for v in curve.get('time',())),
            tuple(_number(v) for v in curve.get('residual',()))))
    _write_csv(csv_path.with_name(csv_path.stem+'_columns.csv'),tuple(columns),
        (tuple(a[i] if i<len(a) else '' for a in values) for i in range(max(map(len,values),default=0))))


def _write_preprocessing_residual_comparison(comparison: Mapping[str, object], csv_path: Path, figure_path: Path) -> None:
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 6), constrained_layout=True)
    axes = figure.subplots(2, 1)
    for axis, side in zip(axes, ('before','after'), strict=True):
        curve = comparison[side].get('residual_curve') or {}
        axis.plot(curve.get('time', ()), curve.get('residual', ()), marker='.', linewidth=1)
        axis.axhline(0, color='black', linewidth=0.5)
        axis.set_title(f'{side.title()} optional preprocessing / manual refit')
        axis.set_xlabel(f"Processed time ({curve.get('time_unit', 's')})")
        axis.set_ylabel(f"Observed - fitted ({curve.get('signal_unit', '')})")
    _write_preprocessing_residual_tables(comparison,csv_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)


def _write_processed_data_csv(snapshot: AnalysisSnapshot, path: Path) -> None:
    import numpy as np
    rows = (
        (_number(time), _number(signal), str(int(raw_index)))
        for time, signal, raw_index in zip(
            snapshot.dataset.processed_time,
            snapshot.dataset.processed_signal,
            np.flatnonzero(snapshot.dataset.mask),
            strict=True,
        )
    )
    _write_csv(path, ("time_s", "processed_signal", "original_raw_index"), rows)


def _write_decomposition(snapshot, root, figures):
    from pyecr.plotting.decomposition import build_decomposition, DecompositionViewOptions
    from pyecr.plotting.matplotlib import render_decomposition_figure
    data = build_decomposition(snapshot.dataset, snapshot.analysis_config, snapshot.fit_result,
                               DecompositionViewOptions.from_dict(snapshot.decomposition_view))
    _write_text(root / "decomposition.json", json.dumps(data.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    g = data.observation
    a, b = g.contributions(data.y0, "shifted")
    rows = ((str(i), str(raw), _number(t), _number(data.observed[i]), _number(g.total[i]), _number(data.residuals[i]),
             _number(g.component_1[i]), _number(g.component_2[i]), _number(data.y0), _number(a[i]), _number(b[i]), data.signal_unit)
            for i, (t, raw) in enumerate(zip(g.time_s, data.original_raw_indices, strict=True)))
    _write_csv(root / "decomposition.csv", ("processed_index", "original_raw_index", "time_s", "observed", "fitted",
        "residual_observed_minus_fitted", "component_1", "component_2", "y0", "shifted_1", "shifted_2", "signal_unit"), rows)
    # This separate file deliberately has no observed, residual, SSE or CI columns.
    g = data.display
    a, b = g.contributions(data.y0, "shifted")
    _write_csv(root / "decomposition_display.csv", ("display_time_s", "fitted", "component_1", "component_2", "y0", "shifted_1", "shifted_2"),
        ((_number(t), _number(g.total[i]), _number(g.component_1[i]), _number(g.component_2[i]), _number(data.y0), _number(a[i]), _number(b[i]))
         for i, t in enumerate(g.time_s)))
    render_decomposition_figure(data, figures / "decomposition.png",plot_views=snapshot.export_context.get('project_metadata',{}).get('plot_views'))


def _decomposition_html(snapshot):
    if not snapshot.decomposition_view:
        return ""
    return ('<h3>Continuous signed decomposition</h3><p>yfit = y0 + component_1 + component_2; '
            'shifted_j = y0 + component_j; yfit = shifted_1 + shifted_2 - y0. '
            'Signed effective contributions are not independent carrier conductivities. '
            'No component CI. Display grid does not enter SSE/Jacobian/CI.</p>'
            '<img src="figures/decomposition.png" alt="Continuous signed joint-model decomposition">'
            '<p><a href="decomposition.csv">Observation-grid CSV</a> · '
            '<a href="decomposition_display.csv">Display-only CSV</a> · '
            '<a href="decomposition.json">Versioned contract, labels, units and bindings</a></p>')


def _write_fit_curve_csv(snapshot: AnalysisSnapshot, path: Path) -> None:
    import numpy as np
    plot = snapshot.fit_plot_data
    raw_indices = np.flatnonzero(snapshot.dataset.mask)
    if plot.model_name=="M8_DUAL_K":
        _write_csv(path,("time_s","observed","fitted","residual_observed_minus_fitted","component_1","component_2","y0","original_raw_index"),
            [(_number(t),_number(y),_number(f),_number(r),_number(plot.component_1[i]),_number(plot.component_2[i]),
              _number(plot.component_baseline),str(int(raw_indices[i]))) for i,(t,y,f,r) in enumerate(zip(plot.time_s,plot.observed,plot.fitted,plot.residuals,strict=True))])
        return
    rows: list[tuple[str, ...]] = []
    for index, values in enumerate(
        zip(
            plot.time_s,
            plot.observed,
            plot.fitted,
            plot.residuals,
            strict=True,
        )
    ):
        time, observed, fitted, residual = values
        rows.append(
            (
                _number(time),
                _number(observed),
                _number(fitted),
                _number(residual),
                "" if not plot.component_H else _number(plot.component_H[index]),
                "" if not plot.component_O else _number(plot.component_O[index]),
                str(int(raw_indices[index])),
            )
        )
    _write_csv(
        path,
        (
            "time_s",
            "observed",
            "fitted",
            "residual_observed_minus_fitted",
            "component_H",
            "component_O",
            "original_raw_index",
        ),
        rows,
    )


def _sensitivity_mapping(diagnostics: Mapping[str, object]) -> Mapping[str, object] | None:
    value = diagnostics.get("sensitivity")
    return value if isinstance(value, Mapping) else None


def _write_sensitivity_csv(snapshot: AnalysisSnapshot, path: Path) -> None:
    sensitivity = _sensitivity_mapping(snapshot.diagnostics_result)
    if sensitivity is None:
        raise ValueError("captured sensitivity result is missing")
    times = snapshot.diagnostics_result.get("time_s")
    curves = sensitivity.get("curves")
    if not isinstance(times, (tuple, list)) or not isinstance(curves, (tuple, list)):
        raise ValueError("captured sensitivity curves require a time axis")
    rows: list[tuple[str, ...]] = []
    for curve in curves:
        if not isinstance(curve, Mapping):
            raise ValueError("captured sensitivity curve must be an object")
        values = curve.get("values")
        normalized = curve.get("normalized_values")
        if (
            not isinstance(values, (tuple, list))
            or not isinstance(normalized, (tuple, list))
            or len(values) != len(times)
            or len(normalized) != len(times)
        ):
            raise ValueError("captured sensitivity curve length must match time axis")
        for time, value, unit_value in zip(times, values, normalized, strict=True):
            rows.append((
                str(curve.get("parameter", "")),
                str(curve.get("label", "")),
                _number(time), _number(value), _number(unit_value),
            ))
    _write_csv(
        path,
        ("parameter", "label", "time_s", "dy_dlog10_parameter", "normalized_sensitivity"),
        rows,
    )
    # Legacy consumers retain sensitivity.csv. Human-facing reports link to
    # independent curve columns, made from exactly the same captured arrays.
    from pyecr.report.figure_catalog import FigureDescriptor,FigureSeries
    from pyecr.report.series_tables import build_series_table,TABLE_LAYOUT_VERSION
    units=str(snapshot.dataset.metadata.get('signal_unit','1'))
    info={}
    for field,stem,unit in (('values','sensitivity_columns',units+'/decade'),
                             ('normalized_values','sensitivity_normalized_columns','1')):
        series=tuple(FigureSeries(str(c.get('label') or c.get('parameter','')),
            times,c[field],str(c.get('parameter',''))) for c in curves)
        figure=FigureDescriptor('analysis.'+stem,'Analysis','AnalysisSnapshot','captured',
            stem,xlabel='Time (s)',x_unit='s',y_unit=unit,series=series)
        table=build_series_table(figure,stem)
        dest=path.with_name(stem+'.csv')
        _write_csv(dest,table.columns,(tuple('' if v is None else _number(v) for v in r) for r in table.rows()))
        info[stem]={**table.info(),'file':dest.name}
    _write_text(path.with_name('sensitivity_columns.json'),json.dumps(
        {'layout':TABLE_LAYOUT_VERSION,'tables':info},ensure_ascii=False,indent=2)+'\n')


def _reliability_advisor_html(diagnostics: Mapping[str, object]) -> str:
    advisor = diagnostics.get("reliability_advisor")
    if not isinstance(advisor, Mapping):
        return ""
    evidence = advisor.get("evidence")
    if isinstance(evidence, Mapping):
        items = [f"{key}: {value}" for key, value in evidence.items()]
    elif isinstance(evidence, (tuple, list)):
        items = [str(item) for item in evidence]
    elif evidence is None:
        items = []
    else:
        items = [str(evidence)]
    evidence_html = (
        "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"
        if items else "<p>Evidence: N/A.</p>"
    )
    guidance = advisor.get("bi_guidance")
    if isinstance(guidance, (tuple, list)):
        guidance_text = "; ".join(str(item) for item in guidance)
    else:
        guidance_text = str(guidance or "N/A")
    return (
        "\n  <h3>Reliability Advisor</h3>"
        "\n  <p><strong>Parameter identifiability: "
        f"{escape(str(advisor.get('identifiability_status', 'N/A')))}</strong> — "
        f"{escape(str(advisor.get('identifiability_explanation', 'N/A')))}</p>"
        "\n  <p><strong>Model adequacy: "
        f"{escape(str(advisor.get('model_adequacy_status', 'N/A')))}</strong> — "
        f"{escape(str(advisor.get('model_adequacy_explanation', 'N/A')))}</p>"
        "\n  <p>Bi guidance (heuristic): "
        f"{escape(guidance_text)}</p>"
        f"\n  <h4>Evidence</h4>{evidence_html}"
    )


def _sensitivity_html(diagnostics: Mapping[str, object]) -> str:
    sensitivity = _sensitivity_mapping(diagnostics)
    if sensitivity is None:
        return ""
    if sensitivity.get("available") is not True:
        return (
            "\n  <h2>Sensitivity</h2><p>N/A — "
            f"{escape(str(sensitivity.get('reason') or 'not available'))}</p>"
        )
    curves = sensitivity.get("curves")
    if not isinstance(curves, (tuple, list)) or not curves:
        return "\n  <h2>Sensitivity</h2><p>No curves were captured.</p>"
    rows: list[str] = []
    for curve in curves:
        if not isinstance(curve, Mapping):
            continue
        relative_error = curve.get("convergence_relative_error")
        rows.append(
            "<tr>"
            f"<td>{escape(str(curve.get('label', curve.get('parameter', 'N/A'))))}</td>"
            f"<td>{escape(_display(curve.get('peak_absolute'), 'N/A'))}</td>"
            f"<td>{escape(_display(curve.get('peak_time_s'), 'N/A'))}</td>"
            f"<td>{escape(_display(curve.get('early_information_fraction'), 'N/A'))}</td>"
            f"<td>{escape(_display(curve.get('mid_information_fraction'), 'N/A'))}</td>"
            f"<td>{escape(_display(curve.get('late_information_fraction'), 'N/A'))}</td>"
            f"<td>{escape(_display(relative_error, 'N/A'))}</td>"
            "</tr>"
        )
    return (
        "\n  <h2>Sensitivity</h2>"
        "<p>S_logD(t) = ∂y/∂log10(D); S_logk(t) = ∂y/∂log10(k). "
        "M3 H/O parameters follow the same definition. These are not q∂σ/∂q. "
        "Early, mid, and late information values are fractions of squared sensitivity. "
        "Bi thresholds are heuristic.</p>"
        f"<p>Method: {escape(str(sensitivity.get('method', 'N/A')))}; "
        f"step: {escape(_display(sensitivity.get('step_decades'), 'N/A'))} decades.</p>"
        "<table><thead><tr><th>Curve</th><th>Peak |S|</th><th>Peak time (s)</th>"
        "<th>Early info</th><th>Mid info</th><th>Late info</th>"
        "<th>Step convergence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        '<p>Captured curves: <a href="sensitivity_columns.csv">Curve columns CSV</a>; '
        '<a href="sensitivity_normalized_columns.csv">Normalized columns CSV</a>; '
        '<a href="sensitivity.csv">Row-oriented CSV</a></p>'
    )


def _guided_refit_html(guided: Mapping[str, object] | None) -> str:
    """Render captured refit evidence; report export never re-runs a fit."""

    if guided is None:
        return ""
    comparison = guided.get("comparison")
    comparison = comparison if isinstance(comparison, Mapping) else {}
    before = comparison.get("before")
    after = comparison.get("after")
    before = before if isinstance(before, Mapping) else {}
    after = after if isinstance(after, Mapping) else {}
    rows = []
    for label, key in (
        ("SSE", "SSE"), ("RMSE", "RMSE"), ("R²", "R2"),
        ("Residual mean", "residual_mean"),
        ("Residual SD", "residual_standard_deviation"),
        ("Max |residual|", "residual_max_abs"),
    ):
        rows.append(
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td>{escape(_display(before.get(key), 'N/A'))}</td>"
            f"<td>{escape(_display(after.get(key), 'N/A'))}</td>"
            "</tr>"
        )
    for label, key in (
        ("Confidence intervals", "confidence_intervals"),
        ("Identifiability", "identifiability"),
    ):
        rows.append(
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td><pre>{escape(_pretty_json(before.get(key, {})))}</pre></td>"
            f"<td><pre>{escape(_pretty_json(after.get(key, {})))}</pre></td>"
            "</tr>"
        )
    return (
        "\n  <h2>Guided Refit provenance</h2>"
        f"<p>Parent FitResult ID: <code>{escape(str(guided.get('parent_fit_id', 'N/A')))}</code>; "
        f"fixed parameter: {escape(str(guided.get('fixed_parameter', 'N/A')))} = "
        f"{escape(_display(guided.get('fixed_value'), 'N/A'))}; "
        f"value source: {escape(str(guided.get('value_source', 'N/A')))}. "
        "The fixed parameter has no free-parameter confidence interval.</p>"
        "<table><thead><tr><th>Measure</th><th>Before</th><th>After</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _report_html(
    snapshot: AnalysisSnapshot,
    experimental_uncertainty: ExperimentalM1UncertaintyResult | None = None,
) -> str:
    diagnostics = snapshot.diagnostics_result
    guided_refit = snapshot.fit_result.diagnostics.get("guided_refit")
    guided_refit = guided_refit if isinstance(guided_refit, Mapping) else None
    fixed_parameter = (
        str(guided_refit.get("fixed_parameter")) if guided_refit is not None else None
    )
    guided_refit_html = _guided_refit_html(guided_refit)
    supplementary_html = ""
    experimental_html = ""
    if experimental_uncertainty is not None:
        record = experimental_uncertainty
        interval_items = "".join(
            f"<li>{escape(name)}: [{escape(_number(values['lower']))}, "
            f"{escape(_number(values['upper']))}]</li>"
            for name, values in record.intervals.items()
        )
        evidence_items = "".join(
            f"<tr><th>{escape(str(name))}</th><td><pre>{escape(_pretty_json(value))}</pre></td></tr>"
            for name, value in record.evidence.items()
        )
        experimental_html = (
            "<h2>Experimental M1 deep uncertainty</h2>"
            f"<p>Method {escape(record.method_version)}; status {escape(record.status)}; "
            f"{record.n_successful}/{record.n_requested} successful; seed {record.seed}; "
            f"source FitResult {escape(record.source_fit_id)}.</p>"
            "<p>Centered residual bootstrap assumes exchangeable residuals; time correlation "
            "is not preserved. This does not replace local CI. Bi guidance is heuristic. "
            "Identifiability and model adequacy are separate.</p>"
            + ("<p>Guided Refit interval is conditional on the fixed value. Fixed-value "
               "uncertainty is excluded; the fixed parameter has no free CI.</p>"
               if record.guided_refit_fixed else "")
            + (f"<ul>{interval_items}</ul>" if interval_items else "<p>No complete interval.</p>")
            + "<h3>Separated evidence</h3><table><tbody>"
            + evidence_items + "</tbody></table>"
            + "<p>Full separated evidence and samples: experimental_m1_uncertainty.json</p>"
        )
    if snapshot.supplementary_results:
        from pyecr.services.pair_design_state import PairDesignState
        supplementary_html = PairDesignState.from_dict(snapshot.supplementary_results).report_html()
    parameter_rows = _diagnostic_parameter_rows(snapshot)
    from pyecr.report.scientific_evidence import parameter_output_rows,output_view,evidence_html
    parameter_rows=parameter_output_rows(snapshot)
    output=output_view(snapshot)
    ci_heading=f'{100*output["confidence_level"]:.12g}% CI - {output["method"]}'
    credibility_heading = ("Captured local precision (CI assumptions separate)" if snapshot.diagnostics_result.get("evaluation")
                           else "Captured credibility (original 95% gate)")
    parameter_html = []
    for name, estimate in snapshot.fit_result.params.items():
        row = parameter_rows.get(name, {})
        ci = "Fixed — no free CI" if name == fixed_parameter else _ci_text(row)
        parameter_html.append(
            "<tr>"
            f"<td>{escape(name)}</td>"
            f"<td>{escape(_number(estimate))}</td>"
            f"<td>{escape(_parameter_unit(snapshot, name))}</td>"
            f"<td>{escape(ci)}</td>"
            f"<td>{escape(_display(row.get('bound_hit'), 'N/A'))}</td>"
            f"<td>{escape(_display(row.get('reliable'), 'N/A'))}</td>"
            f"<td>{escape(str(row.get('reliability_reason') or 'N/A'))}</td>"
            "</tr>"
        )
    if fixed_parameter is not None and fixed_parameter not in snapshot.fit_result.params:
        parameter_html.append(
            "<tr>"
            f"<td>{escape(fixed_parameter)}</td>"
            f"<td>{escape(_display(guided_refit.get('fixed_value'), 'N/A'))}</td>"  # type: ignore[union-attr]
            f"<td>{escape(_parameter_unit(snapshot, fixed_parameter))}</td>"
            "<td>Fixed — no free CI</td><td>N/A</td><td>N/A</td>"
            "<td>Fixed for Guided Refit</td>"
            "</tr>"
        )
    warning_items = (
        "<li>None reported.</li>"
        if not snapshot.warnings
        else "".join(f"<li>{escape(item)}</li>" for item in snapshot.warnings)
    )
    history_rows = (
        "<tr><td colspan='4'>No preprocessing operations recorded.</td></tr>"
        if not snapshot.dataset.history
        else "".join(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{escape(record.operation)}</td>"
            f"<td><code>{escape(_compact_json(record.parameters))}</code></td>"
            f"<td>{escape(record.timestamp)}</td>"
            "</tr>"
            for index, record in enumerate(snapshot.dataset.history, start=1)
        )
    )
    comparison = snapshot.optional_preprocessing_comparison
    optional_comparison_html = ""
    if comparison is not None:
        before, after = comparison["before"], comparison["after"]
        parameter_names = sorted(set(before["params"]) | set(after["params"]))
        rows = [(name, before["params"].get(name), after["params"].get(name)) for name in parameter_names]
        rows.extend((name, before.get(name), after.get(name)) for name in ("SSE", "points", "residual_rms", "residual_max_abs"))
        optional_comparison_html = (
            "<h2>Optional preprocessing: before / after manual refit</h2>"
            f"<p>{escape(str(comparison['note']))}</p>"
            "<table><thead><tr><th>Quantity</th><th>Before</th><th>After</th></tr></thead><tbody>"
            + "".join(f"<tr><td>{escape(name)}</td><td>{escape(_display(a, 'N/A'))}</td><td>{escape(_display(b, 'N/A'))}</td></tr>" for name, a, b in rows)
            + "</tbody></table>"
        )
        if before.get("residual_curve") is not None:
            optional_comparison_html += '<img src="figures/preprocessing_residuals.png" alt="Before and after residual curves"><p><a href="preprocessing_residual_comparison_columns.csv">Before / after residual columns</a>; <a href="preprocessing_residual_comparison.csv">Row-oriented residuals with original indices</a></p>'
    criteria = diagnostics.get("information_criteria", {})
    if not isinstance(criteria, Mapping):
        criteria = {}
    reliable = diagnostics.get("reliable", "N/A")
    reliability_label = diagnostics.get("reliability_label", "N/A")
    reliability_advisor_html = _reliability_advisor_html(diagnostics)
    sensitivity_html = _sensitivity_html(diagnostics)
    source_hash = snapshot.source_sha256 or "Not recorded"
    map_quality_html = (
        '<p><strong>Grid quality: '
        f'{escape(snapshot.error_map_result.quality.status)}</strong> | '
        f'optimum in range={snapshot.error_map_result.quality.optimum_in_range} | '
        f'grid minimum on edge={snapshot.error_map_result.quality.grid_min_on_edge} | '
        f'distance to fit={escape(_display(snapshot.error_map_result.quality.distance_to_fit_in_cells, "N/A"))} cells | '
        f'invalid fraction={snapshot.error_map_result.quality.invalid_fraction:.3%} | '
        f'objective parity={snapshot.error_map_result.quality.objective_parity_ok}</p>'
        if snapshot.error_map_result is not None and snapshot.error_map_result.quality is not None else
        '<p>Grid quality: N/A — legacy Error Map without a P35 quality assessment.</p>'
    )
    map_warning_html = (
        "<ul class=\"warning\">" + "".join(
            f"<li>{escape(item)}</li>"
            for item in snapshot.error_map_result.diagnostics.warnings
        ) + "</ul>"
        if snapshot.error_map_result is not None else ""
    )
    error_map_html = (
        '<h2>Error Map (M1 fixed-nuisance objective)</h2>'
        '<p>Colors show log10(SSE/SSE_min), not a confidence interval. '
        'The white point is the continuous fitted optimum; the cross is the grid minimum.</p>'
        f'{map_quality_html}'
        '<img src="figures/error_map.png" alt="Captured D-k objective map">'
        f'{map_warning_html}'
        '<p>Numerical grid: <a href="error_map.csv">CSV</a> · '
        '<a href="error_map.json">JSON</a></p>'
        if snapshot.error_map_result is not None else ""
    )
    dual_map_html = ""
    if snapshot.dual_error_map_result is not None:
        sections = []
        for process, slice_ in (
            ("H", snapshot.dual_error_map_result.h_map),
            ("O", snapshot.dual_error_map_result.o_map),
        ):
            warnings_html = (
                "<ul class='warning'>" + "".join(
                    f"<li>{escape(item)}</li>" for item in slice_.warnings
                ) + "</ul>"
                if slice_.warnings else "<p>No map-specific warnings.</p>"
            )
            sections.append(
                f"<h3>{process} process D–k map</h3>"
                f"<p>Objective: {escape(slice_.objective_kind)}; "
                "fixed-nuisance full-signal SSE, not a confidence interval.</p>"
                f"<p>Fit-point SSE: {escape(_number(slice_.fitted_point.raw_sse))}; "
                f"invalid fraction: {slice_.quality.invalid_fraction:.3%}.</p>"
                f"{warnings_html}"
                f'<img src="figures/error_map_{process}.png" alt="Captured {process} D-k objective map">'
                f'<p>Numerical grid: <a href="error_map_{process}.csv">CSV</a> · '
                f'<a href="error_map_{process}.json">JSON</a></p>'
            )
        dual_map_html = "<h2>M3 H/O fixed-nuisance Error Maps</h2>" + "".join(sections)
    # Keep the legacy M1-only HTML byte-for-byte stable when no H/O section is
    # present; an empty extra line would change its frozen P41 artifact hash.
    dual_map_section = f"\n  {dual_map_html}" if dual_map_html else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PyECR analysis report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1100px; color: #20252b; }}
    h1, h2 {{ color: #183a5a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.5rem; }}
    th, td {{ border: 1px solid #c7cdd3; padding: 0.45rem; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f7; }}
    code, pre {{ background: #f4f6f8; }}
    pre {{ padding: 0.8rem; overflow-wrap: anywhere; white-space: pre-wrap; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d8dde2; }}
    .warning {{ color: #8b3e00; }}
  </style>
</head>
<body>
  <h1>PyECR analysis report</h1>
  <p>This report was generated from an immutable AnalysisSnapshot, not a GUI screenshot.</p>
  <h2>Provenance</h2>
  <table>
    <tr><th>Dataset SHA256</th><td><code>{snapshot.dataset_sha256}</code></td></tr>
    <tr><th>Source SHA256</th><td><code>{escape(source_hash)}</code></td></tr>
    <tr><th>Source signal</th><td>{escape(str(snapshot.dataset.metadata.get('raw_signal_type', 'unknown')))} ({escape(str(snapshot.dataset.metadata.get('raw_signal_unit', 'arbitrary')))})</td></tr>
    <tr><th>Processed signal</th><td>{escape(str(snapshot.dataset.metadata.get('signal_type', 'unknown')))} ({escape(str(snapshot.dataset.metadata.get('signal_unit', 'arbitrary')))})</td></tr>
    <tr><th>Fit randomness</th><td>stochastic={snapshot.stochastic}; seed={snapshot.seed if snapshot.stochastic else 'not used (deterministic)'}</td></tr>
    <tr><th>Software version</th><td>{escape(snapshot.software_version)}</td></tr>
    <tr><th>Model version</th><td>{escape(snapshot.model_version)}</td></tr>
    <tr><th>Model ID</th><td>{escape(snapshot.analysis_config.model_id.value)}</td></tr>
  </table>{guided_refit_html}
  <h2>Reliability</h2>
  <p>{escape(diagnostic_summary(snapshot.fit_result, diagnostics)).replace(chr(10), '<br>')}</p>
  <p><strong>{escape(str(reliability_label))}</strong> (reliable={escape(str(reliable))})</p>
  <p>SSE={escape(_display(criteria.get('SSE'), 'N/A'))}, RMSE={escape(_display(criteria.get('RMSE'), 'N/A'))}, R²={escape(_display(criteria.get('R2'), 'N/A'))}, AIC={escape(_display(criteria.get('AIC'), 'N/A'))}, AICc={escape(_display(criteria.get('AICc'), 'N/A'))}, BIC={escape(_display(criteria.get('BIC'), 'N/A'))}</p>{reliability_advisor_html}
  <h2>Warnings and credibility gates</h2>
  <ul class="warning">{warning_items}</ul>
  <h2>Parameters</h2>
  <p>{escape(BOUND_LEGEND)} Historical diagnostic decisions are captured records; this report does not recalculate them.</p>
  <table>
    <thead><tr><th>Parameter</th><th>Estimate</th><th>Unit</th><th>{escape(ci_heading)}</th><th>Numerical contact (bound_hit)</th><th>{escape(credibility_heading)}</th><th>Reason</th></tr></thead>
    <tbody>{''.join(parameter_html)}</tbody>
  </table>
  {evidence_html(snapshot)}
  {('<h2>Captured DCT evidence</h2><p>Separate time-scale inversion and reconstruction; not a physical Fit or new chemical evidence. Saved background interpretations retain their own parent identities and conditions.</p><p><a href="dct/captured_dct_evidence.json">Complete DCT arrays and interpretation sources</a></p>') if snapshot.dct_evidence else ''}
  <h2>Fit curves</h2>
  <img src="figures/fit.png" alt="Observed data and fitted ECR curve">{_decomposition_html(snapshot)}
  <img src="figures/residuals.png" alt="Observed-minus-fitted residuals">{sensitivity_html}
  {error_map_html}{dual_map_section}
  <h2>Preprocessing history</h2>
  <table><thead><tr><th>#</th><th>Operation</th><th>Parameters</th><th>Timestamp</th></tr></thead><tbody>{history_rows}</tbody></table>
  {optional_comparison_html}
  <p>Processed data and fit curve CSV rows include original_raw_index for raw-point provenance. Masked points remain in the snapshot raw arrays.</p>
  <h2>Configuration</h2>
  <h3>AnalysisConfig</h3><pre>{escape(_pretty_json(snapshot.analysis_config.to_dict()))}</pre>
  <h3>SolverConfig</h3><pre>{escape(_pretty_json(snapshot.solver_config))}</pre>
  <h3>FitConfig</h3><pre>{escape(_pretty_json(snapshot.fit_config))}</pre>
  {experimental_html}
  {supplementary_html}
</body>
</html>
"""


def _diagnostic_parameter_rows(
    snapshot: AnalysisSnapshot,
) -> dict[str, Mapping[str, object]]:
    raw = snapshot.diagnostics_result.get("parameters", ())
    if not isinstance(raw, (list, tuple)):
        return {}
    rows: dict[str, Mapping[str, object]] = {}
    for item in raw:
        if isinstance(item, Mapping) and isinstance(item.get("name"), str):
            rows[str(item["name"])] = item
    return rows


def _parameter_unit(snapshot: AnalysisSnapshot, name: str) -> str:
    unit = snapshot.analysis_config.parameter_units.get(name, "signal")
    if unit == "signal":
        return str(snapshot.dataset.metadata.get("signal_unit", "signal"))
    return unit


def _ci_text(row: Mapping[str, object]) -> str:
    lower = row.get("ci_lower")
    upper = row.get("ci_upper")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        return f"[{_number(lower)}, {_number(upper)}]"
    reason = row.get("ci_reason")
    return f"N/A — {reason}" if reason else "N/A — not available"


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: object,
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)  # type: ignore[arg-type]
    _write_text(path, stream.getvalue())


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> str:
    return format(float(value), ".17g")


def _optional_number(value: object) -> str:
    return "" if value is None else _number(value)


def _optional_bool(value: object) -> str:
    return "" if value is None else str(bool(value)).lower()


def _display(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float):
        return _number(value)
    return str(value)


def _compact_json(value: object) -> str:
    return json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json(value: object) -> str:
    return json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value
