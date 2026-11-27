"""Immutable, GUI-independent analysis snapshots for reproducible reports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from math import isclose, isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

from pyecr.config import AnalysisConfig
from pyecr.models.ids import ModelID
from pyecr.diagnostics.error_map import (
    ErrorMapConfig, ErrorMapMode, ErrorMapResolution, ErrorMapResult,
    assess_error_map_quality,
)
from pyecr.diagnostics.dual_error_map import DualErrorMapConfig, DualErrorMapResult
from pyecr.fitting.base import FitResult
from pyecr.io.dataset import EcrDataset, ProcessingRecord
from pyecr.plotting import FitPlotData

__all__ = ["AnalysisSnapshot", "dataset_sha256"]

SNAPSHOT_SCHEMA_VERSION = 1
SURFACE_K_SNAPSHOT_SCHEMA_VERSION = 2
DEFAULT_MODEL_VERSION = "1"

def _schema_for_config(config):
    return 2 if config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} or "fixed_d" in config.model_specific else 1

def _contract_key(config):
    return "fixed_d" if "fixed_d" in config.model_specific else "dual_surface_k" if config.model_id is ModelID.M8_DUAL_K else "surface_k"


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Complete, recoverable inputs and outputs for one fitted analysis."""

    dataset: EcrDataset
    dataset_sha256: str
    source_sha256: str | None
    analysis_config: AnalysisConfig
    solver_config: Mapping[str, object]
    fit_config: Mapping[str, object]
    seed: int | None
    fit_result: FitResult
    diagnostics_result: Mapping[str, object]
    software_version: str
    model_version: str
    component_H: tuple[float, ...] = ()
    component_O: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()
    error_map_result: ErrorMapResult | None = None
    error_map_config: ErrorMapConfig | None = None
    dual_error_map_result: DualErrorMapResult | None = None
    dual_error_map_config: DualErrorMapConfig | None = None
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    supplementary_results: Mapping[str, object] = field(default_factory=dict)
    decomposition_view: Mapping[str, object] = field(default_factory=dict)
    export_context: Mapping[str, object] = field(default_factory=dict)
    scientific_evidence: Mapping[str, object] = field(default_factory=dict)
    dct_evidence: Mapping[str,object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        if not isinstance(self.analysis_config, AnalysisConfig):
            raise TypeError("analysis_config must be an AnalysisConfig")
        self.analysis_config.validate()
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult")
        if (self.error_map_result is None) != (self.error_map_config is None):
            raise ValueError("error_map_result and error_map_config must be supplied together")
        if self.error_map_result is not None:
            if self.analysis_config.model_id is not ModelID.M1_SINGLE_DK:
                raise ValueError("Error Map snapshot requires M1_SINGLE_DK")
            assert self.error_map_config is not None
            if self.error_map_result.quality is not None:
                expected_quality = assess_error_map_quality(
                    self.error_map_result, self.error_map_config, self.fit_result.SSE,
                )
                if self.error_map_result.quality != expected_quality:
                    raise ValueError("Error Map quality disagrees with captured grid/config/fit")
        if (self.dual_error_map_result is None) != (self.dual_error_map_config is None):
            raise ValueError("dual_error_map_result and dual_error_map_config must be supplied together")
        if self.dual_error_map_result is not None:
            result = self.dual_error_map_result
            config = self.dual_error_map_config
            assert config is not None
            if self.analysis_config.model_id is not ModelID.M3_DUAL_HO_DK:
                raise ValueError("H/O Error Map snapshot requires M3_DUAL_HO_DK")
            dispatch = self.fit_result.diagnostics.get("analysis_dispatch")
            if (not isinstance(dispatch, Mapping)
                    or dispatch.get("selected_model_id") != ModelID.M3_DUAL_HO_DK.value
                    or dispatch.get("m3_ran") is not True):
                raise ValueError("H/O Error Map snapshot requires a final selected M3 fit")
            if not result.quality.fitted_point_parity_ok or not all(
                isclose(point.raw_sse, self.fit_result.SSE, rel_tol=1e-8, abs_tol=1e-12)
                for point in (result.h_map.fitted_point, result.o_map.fitted_point)
            ):
                raise ValueError("H/O Error Map fitted-point SSE disagrees with FitResult")
            if any(len(axis) != config.grid_size for axis in (
                result.h_map.log10_D, result.h_map.log10_k,
                result.o_map.log10_D, result.o_map.log10_k,
            )):
                raise ValueError("H/O Error Map resolution disagrees with captured config")
        if self.fit_result.y_fit.size != self.dataset.processed_time.size:
            raise ValueError("fit_result length must match processed dataset")
        model_schema = _schema_for_config(self.analysis_config)
        expected_schema = 3 if self.scientific_evidence else model_schema
        if self.schema_version != expected_schema:
            raise ValueError("unsupported AnalysisSnapshot schema_version")
        if model_schema == 2:
            from pyecr.models.surface_k import MODEL_VERSION
            expected_version="p69b-fixed-d-conditional-v1" if "fixed_d" in self.analysis_config.model_specific else MODEL_VERSION
            if self.model_version != expected_version:
                raise ValueError("M7 snapshot requires its surface-k model version")
            _validate_surface_k_binding(self.dataset, self.analysis_config, self.fit_result)
        if not _is_sha256(self.dataset_sha256):
            raise ValueError("dataset_sha256 must be a lowercase SHA256 hex digest")
        actual_hash = dataset_sha256(self.dataset)
        if actual_hash != self.dataset_sha256:
            raise ValueError(
                "dataset_sha256 does not match the serialized EcrDataset content"
            )
        if self.source_sha256 is not None and not _is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a SHA256 hex digest or None")
        from pyecr.fitting.reproducibility import require_reproducibility
        if self.seed is None:
            if self.fit_result.stochastic is not False:raise ValueError('Missing snapshot seed requires explicit deterministic fit provenance')
            require_reproducibility(self.fit_result)
        elif isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if 'fitting_randomness' in self.fit_result.diagnostics:
            if self.seed!=require_reproducibility(self.fit_result):raise ValueError('Snapshot seed differs from actual FitResult seed')
        if not self.software_version.strip() or not self.model_version.strip():
            raise ValueError("software_version and model_version must not be empty")
        solver = _frozen_mapping(self.solver_config, "solver_config")
        fit = _frozen_mapping(self.fit_config, "fit_config")
        if _json_safe(solver) != self.analysis_config.solver.to_dict():
            raise ValueError("solver_config must match AnalysisConfig.solver")
        if _json_safe(fit) != self.analysis_config.fit.to_dict():
            raise ValueError("fit_config must match AnalysisConfig.fit")
        diagnostics = _frozen_mapping(
            self.diagnostics_result, "diagnostics_result"
        )
        supplementary = _frozen_mapping(self.supplementary_results, "supplementary_results")
        if supplementary:
            from pyecr.services.pair_design_state import PairDesignState
            PairDesignState.from_dict(_json_safe(supplementary))
        object.__setattr__(self, "supplementary_results", supplementary)
        context = _frozen_mapping(self.export_context, "export_context")
        from pyecr.report.export_capture import validate_export_context
        validate_export_context(context, self)
        object.__setattr__(self, "export_context", context)
        view = _frozen_mapping(self.decomposition_view, "decomposition_view")
        object.__setattr__(self, "decomposition_view", view)
        if view:
            from pyecr.plotting.decomposition import DecompositionViewOptions, DUAL_IDS
            if self.analysis_config.model_id.value not in DUAL_IDS:
                raise ValueError("decomposition view requires a dual model")
            options = DecompositionViewOptions.from_dict(self.decomposition_view)
            object.__setattr__(self, "decomposition_view", _frozen_mapping(options.to_dict(), "decomposition_view"))
        component_h = _component("component_H", self.component_H, self.fit_result)
        component_o = _component("component_O", self.component_O, self.fit_result)
        warnings = tuple(str(item) for item in self.warnings)
        object.__setattr__(self, "solver_config", solver)
        object.__setattr__(self, "fit_config", fit)
        object.__setattr__(self, "diagnostics_result", diagnostics)
        object.__setattr__(self, "component_H", component_h)
        object.__setattr__(self, "component_O", component_o)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        if self.schema_version==3:
            from pyecr.report.scientific_evidence import validate_scientific
            object.__setattr__(self,'scientific_evidence',validate_scientific(self.scientific_evidence,self))
        if self.dct_evidence:
            if self.schema_version!=3:raise ValueError('Captured DCT context requires R13 scientific snapshot')
            from pyecr.project.interpretation_archive import validate_dct_evidence
            from pyecr.contracts import SourceIdentity
            object.__setattr__(self,'dct_evidence',validate_dct_evidence(self.dct_evidence,SourceIdentity.from_dict(self.scientific_evidence['source'])))

    @property
    def stochastic(self):
        return self.fit_result.stochastic

    @property
    def fully_reproducible(self):
        from pyecr.fitting.reproducibility import require_reproducibility
        try:return require_reproducibility(self.fit_result)==self.seed
        except (ValueError,RuntimeError):return False

    @property
    def optional_preprocessing_comparison(self) -> Mapping[str, object] | None:
        from pyecr.preprocessing.optional import preprocessing_comparison
        return preprocessing_comparison(self.dataset, self.fit_result)

    @property
    def preprocessing_history(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            MappingProxyType(record.to_dict()) for record in self.dataset.history
        )

    @property
    def fit_plot_data(self) -> FitPlotData:
        from pyecr.plotting.labels import signal_axis_label
        return FitPlotData(
            time_s=tuple(float(value) for value in self.dataset.processed_time),
            observed=tuple(
                float(value) for value in self.dataset.processed_signal
            ),
            fitted=tuple(float(value) for value in self.fit_result.y_fit),
            residuals=tuple(float(value) for value in self.fit_result.residuals),
            component_H=self.component_H,
            component_O=self.component_O,
            model_name=self.analysis_config.model_id.value,
            signal_label=signal_axis_label(self.dataset.metadata),
            warnings=self.warnings,
            component_1=tuple(self.fit_result.diagnostics.get("dual_surface_k",{}).get("component_1",())),
            component_2=tuple(self.fit_result.diagnostics.get("dual_surface_k",{}).get("component_2",())),
            component_labels=tuple(self.fit_result.diagnostics.get("dual_surface_k",{}).get("component_labels",())),
            component_baseline=float(self.fit_result.diagnostics.get("dual_surface_k",{}).get("component_baseline",0.)),
        )

    @classmethod
    def capture(
        cls,
        dataset: EcrDataset,
        analysis_config: AnalysisConfig,
        fit_result: FitResult,
        diagnostics_result: object,
        *,
        seed: int | None = None,
        software_version: str | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        error_map_result: ErrorMapResult | None = None,
        error_map_config: ErrorMapConfig | None = None,
        dual_error_map_result: DualErrorMapResult | None = None,
        dual_error_map_config: DualErrorMapConfig | None = None,
        decomposition_view: Mapping[str, object] | None = None,
    ) -> AnalysisSnapshot:
        """Freeze explicit analysis objects without consulting GUI session state."""

        from pyecr.fitting.reproducibility import require_reproducibility,effective_config
        seed=require_reproducibility(fit_result,seed)
        analysis_config=effective_config(analysis_config,fit_result)
        diagnostics = _diagnostics_mapping(diagnostics_result)
        component_h, component_o, artifact_warnings = _persisted_artifacts(
            fit_result
        )
        diagnostic_warnings = diagnostics.get("warnings", ())
        gate_failures = diagnostics.get("gate_failures", ())
        warnings = tuple(
            dict.fromkeys(
                (
                    *fit_result.warnings,
                    *artifact_warnings,
                    *_strings(diagnostic_warnings, "diagnostics warnings"),
                    *_strings(gate_failures, "diagnostics gate_failures"),
                    *(dual_error_map_result.warnings if dual_error_map_result is not None else ()),
                )
            )
        )
        source_hash = dataset.metadata.get("source_sha256")
        if source_hash is not None and not isinstance(source_hash, str):
            raise TypeError("dataset source_sha256 metadata must be a string")
        return cls(
            dataset=dataset,
            dataset_sha256=dataset_sha256(dataset),
            source_sha256=source_hash,
            analysis_config=analysis_config,
            solver_config=analysis_config.solver.to_dict(),
            fit_config=analysis_config.fit.to_dict(),
            seed=seed,
            fit_result=fit_result,
            diagnostics_result=diagnostics,
            software_version=software_version or _software_version(),
            model_version=("p69b-fixed-d-conditional-v1" if "fixed_d" in analysis_config.model_specific else "surface-k/1.0.0" if analysis_config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} else model_version),
            schema_version=_schema_for_config(analysis_config),
            component_H=component_h,
            component_O=component_o,
            warnings=warnings,
            error_map_result=error_map_result,
            error_map_config=error_map_config,
            dual_error_map_result=dual_error_map_result,
            dual_error_map_config=dual_error_map_config,
            decomposition_view=decomposition_view or {},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **({"model_contract": {_contract_key(self.analysis_config):
                                  _json_safe(self.fit_result.diagnostics[_contract_key(self.analysis_config)]),
                                  "fit_result_sha256": _sha256_json(self.fit_result.to_dict())}}
               if _schema_for_config(self.analysis_config) == 2 else {}),
            "schema_version": self.schema_version,
            **({'scientific_evidence':_json_safe(self.scientific_evidence)} if self.schema_version==3 else {}),
            **({'dct_evidence':_json_safe(self.dct_evidence)} if self.dct_evidence else {}),
            **({"export_context": _json_safe(self.export_context)} if self.export_context else {}),
            **({"decomposition_view": _json_safe(self.decomposition_view)} if self.decomposition_view else {}),
            "dataset_sha256": self.dataset_sha256,
            "source_sha256": self.source_sha256,
            "dataset": _dataset_to_dict(self.dataset),
            "preprocessing_history": [
                _json_safe(item) for item in self.preprocessing_history
            ],
            "analysis_config": self.analysis_config.to_dict(),
            "solver_config": _json_safe(self.solver_config),
            "fit_config": _json_safe(self.fit_config),
            "seed": self.seed,
            "fit_result": self.fit_result.to_dict(),
            "diagnostics_result": _json_safe(self.diagnostics_result),
            "software_version": self.software_version,
            "model_version": self.model_version,
            "component_H": list(self.component_H),
            "component_O": list(self.component_O),
            "warnings": list(self.warnings),
            **({"supplementary_results": _json_safe(self.supplementary_results)} if self.supplementary_results else {}),
            **({"error_map": {
                "config": _json_safe(asdict(self.error_map_config)),
                "result": self.error_map_result.to_dict(),
            }} if self.error_map_result is not None and self.error_map_config is not None else {}),
            **({"dual_error_map": _dual_map_payload(self)
               } if self.dual_error_map_result is not None and self.dual_error_map_config is not None else {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AnalysisSnapshot:
        if value.get("schema_version") not in {1, 2, 3}:
            raise ValueError("unsupported AnalysisSnapshot schema_version")
        if value.get('schema_version')==3:
            _required_mapping(value,'scientific_evidence')
            if not value['scientific_evidence']:raise ValueError('R13 snapshot missing scientific evidence')
        elif 'scientific_evidence' in value:raise ValueError('Scientific evidence requires snapshot schema 3')
        if 'dct_evidence' in value and not value['dct_evidence']:raise ValueError('Present DCT evidence cannot be empty/null')
        if value.get("schema_version") == 2 or (value.get('schema_version')==3 and _schema_for_config(AnalysisConfig.from_dict(_required_mapping(value,'analysis_config')))==2):
            contract = _required_mapping(value, "model_contract")
            fit_payload = _required_mapping(value, "fit_result")
            key="fixed_d" if "fixed_d" in contract else "dual_surface_k" if "dual_surface_k" in contract else "surface_k"
            if (contract.get("fit_result_sha256") != _sha256_json(fit_payload)
                    or contract.get(key) != fit_payload.get("diagnostics", {}).get(key)):
                raise ValueError("M7 snapshot model contract/FitResult binding mismatch")
        dataset_payload = _required_mapping(value, "dataset")
        dataset = _dataset_from_dict(dataset_payload)
        history = value.get("preprocessing_history")
        if not isinstance(history, (list, tuple)):
            raise TypeError("preprocessing_history must be an array")
        if [record.to_dict() for record in dataset.history] != list(history):
            raise ValueError("preprocessing_history disagrees with dataset history")
        error_map = value.get("error_map")
        if error_map is not None and not isinstance(error_map, Mapping):
            raise TypeError("error_map must be an object")
        map_config = None
        map_result = None
        if error_map is not None:
            settings = _required_mapping(error_map, "config")
            map_config = ErrorMapConfig(
                mode=ErrorMapMode(settings["mode"]),
                resolution=ErrorMapResolution(settings["resolution"]),
                n_points=settings.get("n_points"),
                log10_D_range=settings.get("log10_D_range"),
                log10_k_range=settings.get("log10_k_range"),
                auto_half_width_decades=float(settings["auto_half_width_decades"]),
                narrow_range_decades=float(settings["narrow_range_decades"]),
            )
            map_result = ErrorMapResult.from_dict(dict(_required_mapping(error_map, "result")))
        dual_payload = value.get("dual_error_map")
        dual_result = None
        dual_config = None
        if dual_payload is not None:
            dual_record = _mapping(dual_payload, "dual_error_map")
            if dual_record.get("schema_version") != 1:
                raise ValueError("unsupported dual_error_map snapshot schema_version")
            signed = {key: item for key, item in dual_record.items() if key != "sha256"}
            if dual_record.get("sha256") != _sha256_json(signed):
                raise ValueError("dual_error_map snapshot SHA256 digest mismatch")
            if dual_record.get("dataset_sha256") != value.get("dataset_sha256"):
                raise ValueError("dual_error_map dataset SHA256 mismatch")
            if dual_record.get("fit_result_sha256") != _sha256_json(value.get("fit_result")):
                raise ValueError("dual_error_map FitResult SHA256 mismatch")
            dual_config = DualErrorMapConfig.from_dict(_required_mapping(dual_record, "config"))
            dual_result = DualErrorMapResult.from_dict(_required_mapping(dual_record, "result"))
        restored_fit = FitResult.from_dict(_required_mapping(value, "fit_result"))
        order = value.get('scientific_evidence',{}).get('parameter_order',value.get("export_context", {}).get("free_parameter_order"))
        if order is None:
            # Component snapshots may omit export_context. Their saved
            # Diagnostics still carry the original named column/row
            # order as arrays; recover it without recomputing any evidence.
            diagnostics = _required_mapping(value, "diagnostics_result")
            correlation = diagnostics.get('correlation', {})
            rows = diagnostics.get('parameters', ())
            candidates = [correlation.get('parameter_names') if isinstance(correlation, Mapping) else None,
                          [r.get('name') for r in rows] if isinstance(rows, (list, tuple))
                          and all(isinstance(r, Mapping) for r in rows) else None]
            for names in candidates:
                if (isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names)
                        and len(names) == len(restored_fit.params) and set(names) == set(restored_fit.params)):
                    order = names
                    break
        if order is not None:
            if (not isinstance(order, (list, tuple)) or not all(isinstance(n, str) for n in order)
                    or len(order) != len(restored_fit.params) or set(order) != set(restored_fit.params)):
                raise ValueError("export parameter column order mismatch")
            restored_fit = replace(restored_fit, params={n:restored_fit.params[n] for n in order},
                                   param_space={n:restored_fit.param_space[n] for n in order},
                                   candidates=tuple(replace(c, params={n:c.params[n] for n in order})
                                       if set(c.params) == set(order) else c for c in restored_fit.candidates))
        return cls(
            dataset=dataset,
            dataset_sha256=str(value["dataset_sha256"]),
            source_sha256=(
                None
                if value.get("source_sha256") is None
                else str(value["source_sha256"])
            ),
            analysis_config=AnalysisConfig.from_dict(
                _required_mapping(value, "analysis_config")
            ),
            solver_config=_required_mapping(value, "solver_config"),
            fit_config=_required_mapping(value, "fit_config"),
            seed=None if value["seed"] is None else int(value["seed"]),
            fit_result=restored_fit,
            diagnostics_result=_required_mapping(value, "diagnostics_result"),
            software_version=str(value["software_version"]),
            model_version=str(value["model_version"]),
            schema_version=int(value["schema_version"]),
            component_H=_float_tuple(value.get("component_H", ()), "component_H"),
            component_O=_float_tuple(value.get("component_O", ()), "component_O"),
            warnings=_strings(value.get("warnings", ()), "warnings"),
            supplementary_results=value.get("supplementary_results", {}),
            decomposition_view=value.get("decomposition_view", {}),
            export_context=value.get("export_context", {}),
            scientific_evidence=value.get('scientific_evidence',{}),
            dct_evidence=value.get('dct_evidence',{}),
            error_map_result=map_result,
            error_map_config=map_config,
            dual_error_map_result=dual_result,
            dual_error_map_config=dual_config,
        )


def _validate_surface_k_binding(dataset, config, fit):
    if "fixed_d" in config.model_specific:
        from pyecr.services.fixed_d import validate_fixed_diffusion,VERSION
        sources=validate_fixed_diffusion(config)
        record=fit.diagnostics.get("fixed_d",{})
        if (record.get("version")!=VERSION or record.get("dataset_sha256")!=dataset_sha256(dataset)
                or record.get("config_sha256")!=_sha256_json(config.to_dict())
                or record.get("fit_eligible_count")!=dataset.processed_time.size
                or record.get("free_parameter_count")!=len(fit.params)
                or record.get("sources")!=sources
                or any(record.get("fixed_parameters",{}).get(n)!=s["value_cm2_s"] for n,s in sources.items())
                or set(sources)&set(fit.params) or record.get("ci_scope")!="conditional_on_D"):
            raise ValueError("Conditional fixed-D provenance/CI binding mismatch")
        return
    from pyecr.models.surface_k import MODEL_VERSION, CONTRACT_VERSION
    record = fit.diagnostics.get("dual_surface_k" if config.model_id is ModelID.M8_DUAL_K else "surface_k", {})
    if (record.get("model_version") != MODEL_VERSION or record.get("contract_version") != CONTRACT_VERSION
            or record.get("dataset_sha256") != dataset_sha256(dataset)
            or record.get("config_sha256") != _sha256_json(config.to_dict())
            or record.get("free_parameter_count") != len(fit.params)
            or record.get("fit_eligible_count") != dataset.processed_time.size):
        raise ValueError("M7 provenance does not match dataset/config/free parameters")
    if config.model_id is ModelID.M8_DUAL_K:
        import numpy as np
        reconstructed=record["component_baseline"]+np.asarray(record["component_1"])+np.asarray(record["component_2"])
        if not np.allclose(reconstructed,fit.y_fit,rtol=0,atol=1e-12*max(1.,float(np.max(np.abs(fit.y_fit))))):
            raise ValueError("M8 component reconstruction binding mismatch")


def _dual_map_payload(snapshot: AnalysisSnapshot) -> dict[str, object]:
    assert snapshot.dual_error_map_result is not None
    assert snapshot.dual_error_map_config is not None
    record: dict[str, object] = {
        "schema_version": 1,
        "dataset_sha256": snapshot.dataset_sha256,
        "fit_result_sha256": _sha256_json(snapshot.fit_result.to_dict()),
        "config": snapshot.dual_error_map_config.to_dict(),
        "result": snapshot.dual_error_map_result.to_dict(),
    }
    return {**record, "sha256": _sha256_json(record)}


def _sha256_json(value: object) -> str:
    canonical = json.dumps(
        _json_safe(value), sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def dataset_sha256(dataset: EcrDataset) -> str:
    """Hash every dataset array, metadata item, and preprocessing record."""

    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    canonical = json.dumps(
        _dataset_to_dict(dataset),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _dataset_to_dict(dataset: EcrDataset) -> dict[str, object]:
    return {
        "raw_time": dataset.raw_time.tolist(),
        "raw_signal": dataset.raw_signal.tolist(),
        "processed_time": dataset.processed_time.tolist(),
        "processed_signal": dataset.processed_signal.tolist(),
        "mask": dataset.mask.tolist(),
        "metadata": dataset.metadata_as_dict(),
        "history": [record.to_dict() for record in dataset.history],
    }


def _dataset_from_dict(value: Mapping[str, object]) -> EcrDataset:
    history = value.get("history")
    if not isinstance(history, (list, tuple)):
        raise TypeError("dataset history must be an array")
    return EcrDataset(
        raw_time=value["raw_time"],
        raw_signal=value["raw_signal"],
        processed_time=value["processed_time"],
        processed_signal=value["processed_signal"],
        mask=value["mask"],
        metadata=_required_mapping(value, "metadata"),
        history=tuple(
            ProcessingRecord.from_dict(_mapping(item, "history item"))
            for item in history
        ),
    )


def _diagnostics_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return _frozen_mapping(value, "diagnostics_result")
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        if converted.get("evaluation", "absent") is None:
            converted.pop("evaluation", None)
        return _frozen_mapping(converted, "diagnostics_result")
    raise TypeError("diagnostics_result must be a dataclass or mapping")


def _persisted_artifacts(
    fit_result: FitResult,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[str, ...]]:
    artifacts = fit_result.diagnostics.get("analysis_artifacts")
    if artifacts is None:
        return (), (), ()
    if not isinstance(artifacts, Mapping):
        raise TypeError("analysis_artifacts must be an object")
    return (
        _float_tuple(artifacts.get("component_H", ()), "component_H"),
        _float_tuple(artifacts.get("component_O", ()), "component_O"),
        _strings(artifacts.get("warnings", ()), "analysis_artifacts warnings"),
    )


def _component(
    name: str,
    value: tuple[float, ...],
    fit_result: FitResult,
) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if result and len(result) != fit_result.y_fit.size:
        raise ValueError(f"{name} must align with fit_result")
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _frozen_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        raise TypeError(f"{name} must be an object")
    frozen = _deep_freeze(safe)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{name} must be an object")
    return frozen


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if isfinite(numeric):
            return numeric
        if numeric != numeric:
            return "NaN"
        return "Infinity" if numeric > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _required_mapping(
    value: Mapping[str, object], name: str
) -> Mapping[str, object]:
    return _mapping(value.get(name), name)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return tuple(float(item) for item in value)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return tuple(str(item) for item in value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _software_version() -> str:
    try:
        return version("pyecr")
    except PackageNotFoundError:
        return "0.1.0"
