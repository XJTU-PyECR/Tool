"""Atomic, integrity-checked persistence for PyECR directory projects."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from math import isclose, isfinite
from pathlib import Path
from uuid import uuid4

import numpy as np

from pyecr.config import AnalysisConfig
from pyecr.models.ids import ModelID
from pyecr.diagnostics.dual_error_map import DualErrorMapConfig, DualErrorMapResult
from pyecr.diagnostics.sensitivity import SensitivityCurve, SensitivityPresentation
from pyecr.diagnostics.evaluation import evaluation_from_dict
from pyecr.fitting.base import FitResult
from pyecr.io.dataset import EcrDataset, ProcessingRecord
from pyecr.preprocessing import PreprocessConfig
from pyecr.report.snapshot import AnalysisSnapshot, dataset_sha256, _schema_for_config
from pyecr.services.diagnostics_service import (
    BiDimensionPresentation,
    BiPresentation,
    CorrelationPresentation,
    DiagnosticsPresentation,
    InformationCriteriaPresentation,
    JacobianPresentation,
    MultiStartPresentation,
    MultiStartSpreadPresentation,
    ParameterDiagnosticsPresentation,
    ParameterPresentation,
    ResidualSummaryPresentation,
    ReliabilityAdvisorPresentation,
)
from pyecr.services.guided_refit import FitHistoryEntry
from pyecr.services.pair_design_state import PairDesignState
from pyecr.services.batch_analysis import BatchAnalysisResult, GlobalAnalysisResult
from pyecr.services.dct_service import DctPresentation

__all__ = [
    "CURRENT_PROJECT_SCHEMA_VERSION",
    "LinkedRawDataMissingError",
    "ProjectHashMismatchError",
    "ProjectIntegrityError",
    "ProjectRepository",
    "ProjectRepositoryError",
    "ProjectSchemaError",
    "ProjectSnapshot",
    "RawDataMode",
]


CURRENT_PROJECT_SCHEMA_VERSION = 4
MANIFEST_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
DIAGNOSTICS_SCHEMA_VERSION = 1
MODEL_VERSION = "1"


class RawDataMode(str, Enum):
    """How the imported source file is referenced by a project."""

    LINKED = "linked"
    PORTABLE = "portable"


class ProjectRepositoryError(RuntimeError):
    """Base error for project persistence failures."""


class ProjectIntegrityError(ProjectRepositoryError):
    """A project payload is incomplete or internally inconsistent."""


class ProjectHashMismatchError(ProjectIntegrityError):
    """A managed file or raw source no longer matches its recorded SHA256."""


class LinkedRawDataMissingError(ProjectIntegrityError):
    """A linked project cannot locate its external raw source file."""


class ProjectSchemaError(ProjectRepositoryError):
    """The project schema is unsupported and cannot be migrated."""


MigrationHook = Callable[[dict[str, object]], dict[str, object]]


@dataclass(frozen=True, slots=True)
class ProjectSnapshot:
    """Fully restored project state independent of GUI widgets."""

    project_path: Path
    schema_version: int
    dataset: EcrDataset
    source_dataset: EcrDataset
    preprocess_config: PreprocessConfig
    analysis_config: AnalysisConfig | None
    fit_result: FitResult | None
    fit_history: tuple[FitHistoryEntry, ...]
    diagnostics_result: DiagnosticsPresentation | Mapping[str, object] | None
    batch_result: BatchAnalysisResult | None
    global_result: GlobalAnalysisResult | None
    dct_result: DctPresentation | None
    dual_error_map_result: DualErrorMapResult | None
    dual_error_map_config: DualErrorMapConfig | None
    dual_error_map_stale: bool
    report_snapshot: AnalysisSnapshot | None
    raw_data_mode: RawDataMode
    raw_data_path: Path
    raw_data_sha256: str
    seed: int | None
    pyecr_version: str
    model_version: str
    git_commit: str | None
    project_metadata: Mapping[str, object]
    created_at: str
    saved_at: str
    pair_design_state: PairDesignState | None = None
    dct_archive: object | None = None
    scientific_evidence: Mapping[str,object] | None = None
    dct_interpretations: Mapping[str,object] | None = None


class ProjectRepository:
    """Save and open one self-describing PyECR directory project."""

    def __init__(
        self,
        *,
        migrations: Mapping[int, MigrationHook] | None = None,
    ) -> None:
        supplied = dict(migrations or {})
        if any(
            isinstance(schema, bool)
            or not isinstance(schema, int)
            or schema < 0
            or not callable(hook)
            for schema, hook in supplied.items()
        ):
            raise TypeError("migrations must map nonnegative schema integers to callables")
        self._migrations = supplied

    def register_migration(self, schema_version: int, hook: MigrationHook) -> None:
        """Register an in-memory migration from one schema to its successor."""

        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 0
        ):
            raise TypeError("schema_version must be a nonnegative integer")
        if not callable(hook):
            raise TypeError("migration hook must be callable")
        self._migrations[schema_version] = hook

    def save(
        self,
        project_path: str | Path,
        *,
        dataset: EcrDataset,
        raw_data_mode: RawDataMode | str,
        raw_source_path: str | Path,
        source_dataset: EcrDataset | None = None,
        preprocess_config: PreprocessConfig | None = None,
        analysis_config: AnalysisConfig | None = None,
        fit_result: FitResult | None = None,
        fit_history: tuple[FitHistoryEntry, ...] | list[FitHistoryEntry] = (),
        pair_design_state: PairDesignState | None = None,
        diagnostics_result: DiagnosticsPresentation | Mapping[str, object] | None = None,
        batch_result: BatchAnalysisResult | None = None,
        global_result: GlobalAnalysisResult | None = None,
        dct_result: DctPresentation | None = None,
        dct_archive=None,
        scientific_evidence=None,
        dct_interpretations=None,
        dual_error_map_result: DualErrorMapResult | None = None,
        dual_error_map_config: DualErrorMapConfig | None = None,
        dual_error_map_stale: bool = False,
        report_snapshot: AnalysisSnapshot | None = None,
        seed: int | None = None,
        project_metadata: Mapping[str, object] | None = None,
        cancel_check=None,
        before_publish=None,
    ) -> ProjectSnapshot:
        """Write a complete project into a sibling temp directory, then swap it."""

        target = _project_target(project_path)
        _validate_save_target(target)
        if not isinstance(dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        source_dataset = dataset if source_dataset is None else source_dataset
        if not isinstance(source_dataset, EcrDataset):
            raise TypeError("source_dataset must be an EcrDataset")
        recipe = preprocess_config or PreprocessConfig()
        if not isinstance(recipe, PreprocessConfig):
            raise TypeError("preprocess_config must be a PreprocessConfig")
        _validate_replay(source_dataset, recipe, dataset)
        if analysis_config is not None:
            if not isinstance(analysis_config, AnalysisConfig):
                raise TypeError("analysis_config must be an AnalysisConfig or None")
            analysis_config.validate()
        if fit_result is not None and not isinstance(fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult or None")
        if fit_result is not None and analysis_config is None:
            raise ValueError("fit_result requires analysis_config")
        if fit_result is not None:
            from pyecr.fitting.reproducibility import effective_config
            analysis_config=effective_config(analysis_config,fit_result)
        if fit_result is not None and fit_result.y_fit.size != dataset.processed_signal.size:
            raise ValueError("fit_result length must match the processed dataset")
        history = tuple(fit_history)
        if pair_design_state is not None and not isinstance(pair_design_state, PairDesignState):
            raise TypeError("pair_design_state must be a PairDesignState or None")
        if any(not isinstance(entry, FitHistoryEntry) for entry in history):
            raise TypeError("fit_history must contain FitHistoryEntry values")
        if diagnostics_result is not None and fit_result is None:
            raise ValueError("diagnostics_result requires fit_result")
        if batch_result is not None and not isinstance(
            batch_result, BatchAnalysisResult
        ):
            raise TypeError("batch_result must be a BatchAnalysisResult or None")
        if global_result is not None and not isinstance(
            global_result, GlobalAnalysisResult
        ):
            raise TypeError("global_result must be a GlobalAnalysisResult or None")
        if dct_result is not None and not isinstance(dct_result, DctPresentation):
            raise TypeError("dct_result must be a DctPresentation or None")
        if dct_result is not None:
            if len(dct_result.time_s) != dataset.processed_time.size:
                raise ValueError("dct_result length must match the processed dataset")
            if not np.array_equal(dct_result.time_s, dataset.processed_time):
                raise ValueError("dct_result time axis must match the processed dataset")
            if not np.array_equal(dct_result.signal, dataset.processed_signal):
                raise ValueError("dct_result signal must match the processed dataset")
        if (dual_error_map_result is None) != (dual_error_map_config is None):
            raise ValueError("dual Error Map result and config must be supplied together")
        if not isinstance(dual_error_map_stale, bool):
            raise TypeError("dual_error_map_stale must be bool")
        if dual_error_map_result is None and dual_error_map_stale:
            raise ValueError("cannot mark a missing dual Error Map stale")
        if dual_error_map_result is not None and not dual_error_map_stale:
            _validate_current_dual_map(
                dual_error_map_result, dual_error_map_config, analysis_config, fit_result,
            )
        if report_snapshot is not None and not isinstance(report_snapshot, AnalysisSnapshot):
            raise TypeError("report_snapshot must be AnalysisSnapshot or None")
        mode = RawDataMode(raw_data_mode)
        raw_source = _existing_file(raw_source_path, "raw source")
        raw_hash = _sha256_file(raw_source)
        imported_hash = source_dataset.metadata.get("source_sha256")
        if isinstance(imported_hash, str) and imported_hash and imported_hash != raw_hash:
            raise ProjectHashMismatchError(
                "raw source SHA256 no longer matches the imported dataset: "
                f"expected {imported_hash}, got {raw_hash} for {raw_source}"
            )
        metadata = _json_mapping(project_metadata or {}, "project_metadata")
        if dct_archive is not None:
            from pyecr.project.dct_archive import DctArchive
            if not isinstance(dct_archive, DctArchive):
                raise TypeError('dct_archive must be a DctArchive')
            dct_archive = dct_archive.validated()
        if scientific_evidence is not None:
            if analysis_config is None or fit_result is None:raise ProjectIntegrityError('Scientific evidence requires a real bound Fit')
            from pyecr.report.scientific_evidence import validate_scientific
            captured=AnalysisSnapshot.capture(dataset,analysis_config,fit_result,diagnostics_result or {'available':False})
            scientific_evidence=validate_scientific(scientific_evidence,captured)
        if dct_interpretations is not None:
            from pyecr.project.interpretation_archive import validate_interpretations
            dct_interpretations=validate_interpretations(dct_interpretations,dct_archive,dct_result)
        _validate_decomposition_view(metadata)
        normalized_seed = _seed(seed, analysis_config, fit_result)
        created_at = _existing_created_at(target) or _timestamp()
        saved_at = _timestamp()

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        try:
            self._write_project(
                temporary,
                final_target=target,
                dataset=dataset,
                source_dataset=source_dataset,
                preprocess_config=recipe,
                analysis_config=analysis_config,
                fit_result=fit_result,
                fit_history=history,
                pair_design_state=pair_design_state,
                diagnostics_result=diagnostics_result,
                batch_result=batch_result,
                global_result=global_result,
                dct_result=dct_result,
                dct_archive=dct_archive,
                scientific_evidence=scientific_evidence,
                dct_interpretations=dct_interpretations,
                dual_error_map_result=dual_error_map_result,
                dual_error_map_config=dual_error_map_config,
                dual_error_map_stale=dual_error_map_stale,
                report_snapshot=report_snapshot,
                raw_data_mode=mode,
                raw_source_path=raw_source,
                raw_data_sha256=raw_hash,
                seed=normalized_seed,
                project_metadata=metadata,
                created_at=created_at,
                saved_at=saved_at,
            )
            if cancel_check:
                cancel_check()
            if dct_archive is not None or scientific_evidence is not None or dct_interpretations is not None or 'r13_execution_history' in metadata:
                self.open(temporary)  # validate full staged project before replacing an existing directory
            if before_publish:
                before_publish()
            if cancel_check:
                cancel_check()
            _atomic_directory_replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return self.open(target)

    def open(self, project_path: str | Path) -> ProjectSnapshot:
        """Verify hashes and restore all persisted project state."""

        root = _existing_directory(project_path)
        _verify_layout(root)
        project_file = root / "project.json"
        manifest_file = root / "manifest.json"
        raw_project = _read_json_object(project_file, "project.json")
        schema = _schema_version(raw_project)
        if schema > CURRENT_PROJECT_SCHEMA_VERSION:
            raise ProjectSchemaError(
                f"newer project schema {schema} is not supported; "
                f"current={CURRENT_PROJECT_SCHEMA_VERSION}"
            )
        manifest = _read_json_object(manifest_file, "manifest.json")
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ProjectSchemaError("unsupported manifest schema_version")
        _verify_manifest(root, manifest)
        project = self._migrate(raw_project)
        schema = _schema_version(project)

        paths = _required_mapping(project, "paths")
        dataset_payload = _read_managed_json(root, paths, "dataset")
        source_dataset, recipe, dataset = _datasets_from_dict(dataset_payload)
        _validate_replay(source_dataset, recipe, dataset)
        config_payload = project.get("analysis_config")
        analysis_config = (
            None
            if config_payload is None
            else AnalysisConfig.from_dict(_mapping(config_payload, "analysis_config"))
        )
        if analysis_config is not None:
            analysis_config.validate()
        if schema == 2 and (analysis_config is None or _schema_for_config(analysis_config)!=2):
            raise ProjectSchemaError("project schema 2 is reserved for M7 surface-k")
        if analysis_config is not None and _schema_for_config(analysis_config)==2 and schema not in {2, 3, 4}:
            raise ProjectSchemaError("M7 requires project schema 2")
        fit_payload = _read_optional_managed_json(root, paths, "fit_result")
        fit_result = (
            None
            if fit_payload is None
            else FitResult.from_dict(_mapping(fit_payload, "fit_result"))
        )
        if analysis_config is not None and _schema_for_config(analysis_config)==2 and fit_result is not None:
            from pyecr.report.snapshot import _validate_surface_k_binding
            _validate_surface_k_binding(dataset, analysis_config, fit_result)
        history_payload = _read_optional_managed_json(root, paths, "fit_history")
        if history_payload is None:
            fit_history: tuple[FitHistoryEntry, ...] = ()
        else:
            if history_payload.get("schema_version") != 1:
                raise ProjectSchemaError("unsupported fit history schema_version")
            entries = history_payload.get("entries")
            if not isinstance(entries, list):
                raise ProjectIntegrityError("fit_history.entries must be an array")
            fit_history = tuple(FitHistoryEntry.from_dict(item) for item in entries)
        diagnostics_payload = _read_optional_managed_json(root, paths, "diagnostics")
        diagnostics_result = (
            None
            if diagnostics_payload is None
            else _diagnostics_from_dict(diagnostics_payload)
        )
        batch_payload = _read_optional_managed_json(root, paths, "batch_result")
        batch_result = (
            None
            if batch_payload is None
            else BatchAnalysisResult.from_dict(batch_payload)
        )
        global_payload = _read_optional_managed_json(root, paths, "global_result")
        global_result = (
            None
            if global_payload is None
            else GlobalAnalysisResult.from_dict(global_payload)
        )
        dct_payload = _read_optional_managed_json(root, paths, "dct_result")
        dct_result = (
            None
            if dct_payload is None
            else DctPresentation.from_dict(dct_payload)
        )
        archive_payload = _read_optional_managed_json(root, paths, 'dct_archive')
        dct_archive = None
        if schema == 3 and archive_payload is None:
            raise ProjectSchemaError('project schema 3 requires DCT archive provenance')
        if archive_payload is not None:
            if schema not in {3,4}:
                raise ProjectSchemaError('DCT archive requires project schema 3')
            from pyecr.project.dct_archive import DctArchive
            dct_archive = DctArchive.from_json(_managed_path(root, paths['dct_archive']).read_text(encoding='utf-8'))
        scientific_evidence=_read_optional_managed_json(root,paths,'scientific_evidence')
        dct_interpretations=_read_optional_managed_json(root,paths,'dct_interpretations')
        if schema==4:
            if scientific_evidence is None and dct_interpretations is None:raise ProjectSchemaError('R13 project requires recorded result evidence or DCT interpretation provenance')
            if fit_result is not None and scientific_evidence is None:raise ProjectSchemaError('R13 fitted project missing scientific evidence')
        elif scientific_evidence is not None or dct_interpretations is not None:raise ProjectSchemaError('R13 evidence requires project schema 4')
        if scientific_evidence is not None:
            if analysis_config is None or fit_result is None:raise ProjectIntegrityError('Scientific source missing actual Fit')
            order=scientific_evidence.get('parameter_order')
            if not isinstance(order,list) or len(order)!=len(fit_result.params) or set(order)!=set(fit_result.params):raise ProjectIntegrityError('Missing/conflicting Jacobian column order')
            # JSON key sorting also affects candidate dictionaries. Their order
            # names multistart evidence rows; restore it with the same metadata.
            fit_result=replace(fit_result,params={n:fit_result.params[n] for n in order},param_space={n:fit_result.param_space[n] for n in order},
                candidates=tuple(replace(c,params={n:c.params[n] for n in order})
                    if set(c.params)==set(order) else c for c in fit_result.candidates))
            captured=AnalysisSnapshot.capture(dataset,analysis_config,fit_result,diagnostics_result or {'available':False})
            from pyecr.report.scientific_evidence import validate_scientific
            scientific_evidence=validate_scientific(scientific_evidence,captured)
        if dct_interpretations is not None:
            from pyecr.project.interpretation_archive import validate_interpretations
            dct_interpretations=validate_interpretations(dct_interpretations,dct_archive,dct_result)
        dual_payload = _read_optional_managed_json(root, paths, "dual_error_map")
        dual_result = None
        dual_config = None
        dual_stale = False
        if dual_payload is not None:
            if dual_payload.get("schema_version") != 1:
                raise ProjectSchemaError("unsupported dual Error Map project schema_version")
            dual_config = DualErrorMapConfig.from_dict(
                _required_mapping(dual_payload, "config")
            )
            dual_result = DualErrorMapResult.from_dict(
                _required_mapping(dual_payload, "result")
            )
            status = dual_payload.get("status")
            if status not in {"current", "stale"}:
                raise ProjectIntegrityError("dual Error Map status must be current or stale")
            dual_stale = status == "stale"
            if not dual_stale:
                _validate_current_dual_map(
                    dual_result, dual_config, analysis_config, fit_result,
                )
                binding = _required_mapping(dual_payload, "binding")
                expected = _dual_map_binding(dataset, analysis_config, fit_result)
                if dict(binding) != expected:
                    raise ProjectHashMismatchError(
                        "dual Error Map input binding SHA256 mismatch"
                    )
        snapshot_payload = _read_optional_managed_json(root, paths, "report_snapshot")
        report_snapshot = (
            None if snapshot_payload is None else AnalysisSnapshot.from_dict(snapshot_payload)
        )
        if fit_result is not None and report_snapshot is not None and report_snapshot.export_context:
            from pyecr.report.snapshot import _sha256_json
            context = report_snapshot.export_context
            if context["fit_result_sha256"] == _sha256_json(fit_result.to_dict()):
                order = context["free_parameter_order"]
                if scientific_evidence is not None and tuple(order)!=tuple(scientific_evidence['parameter_order']):
                    raise ProjectIntegrityError('Report/selected CI Jacobian parameter column identity conflict')
                fit_result = replace(fit_result, params={n:fit_result.params[n] for n in order},
                                     param_space={n:fit_result.param_space[n] for n in order},
                                     candidates=tuple(replace(c,params={n:c.params[n] for n in order})
                                         if set(c.params)==set(order) else c for c in fit_result.candidates))
        raw_data = _required_mapping(project, "raw_data")
        raw_mode = RawDataMode(raw_data.get("mode"))
        raw_path = _resolve_raw_path(root, raw_data, raw_mode)
        expected_raw_hash = _required_string(raw_data, "sha256")
        if not raw_path.is_file():
            if raw_mode is RawDataMode.LINKED:
                raise LinkedRawDataMissingError(
                    f"linked raw data file is missing: {raw_path}"
                )
            raise ProjectIntegrityError(f"portable raw data file is missing: {raw_path}")
        actual_raw_hash = _sha256_file(raw_path)
        if actual_raw_hash != expected_raw_hash:
            raise ProjectHashMismatchError(
                "raw data SHA256 mismatch: "
                f"expected {expected_raw_hash}, got {actual_raw_hash} for {raw_path}"
            )

        reproducibility = _required_mapping(project, "reproducibility")
        seed_value = reproducibility.get("seed")
        from pyecr.fitting.reproducibility import reproducibility_issue,require_reproducibility
        if fit_result is not None and not reproducibility_issue(fit_result):
            if seed_value!=require_reproducibility(fit_result):
                raise ProjectIntegrityError('Project seed differs from actual FitResult seed')
        metadata = _required_mapping(project, "project_metadata")
        _validate_decomposition_view(metadata)
        pair_payload = _read_optional_managed_json(root, paths, "pair_design")
        pair_state = None if pair_payload is None else PairDesignState.from_dict(pair_payload)
        return ProjectSnapshot(
            project_path=root,
            schema_version=_schema_version(project),
            dataset=dataset,
            source_dataset=source_dataset,
            preprocess_config=recipe,
            analysis_config=analysis_config,
            fit_result=fit_result,
            fit_history=fit_history,
            diagnostics_result=diagnostics_result,
            batch_result=batch_result,
            global_result=global_result,
            dct_result=dct_result,
            dct_archive=dct_archive,
            scientific_evidence=scientific_evidence,
            dct_interpretations=dct_interpretations,
            dual_error_map_result=dual_result,
            dual_error_map_config=dual_config,
            dual_error_map_stale=dual_stale,
            report_snapshot=report_snapshot,
            raw_data_mode=raw_mode,
            raw_data_path=raw_path,
            raw_data_sha256=expected_raw_hash,
            seed=None if seed_value is None else int(seed_value),
            pyecr_version=_required_string(reproducibility, "pyecr_version"),
            model_version=_required_string(reproducibility, "model_version"),
            git_commit=(
                None
                if reproducibility.get("git_commit") is None
                else str(reproducibility["git_commit"])
            ),
            project_metadata=metadata,
            created_at=_required_string(project, "created_at"),
            saved_at=_required_string(project, "saved_at"),
            pair_design_state=pair_state,
        )

    def make_portable(self, project_path: str | Path) -> ProjectSnapshot:
        """Copy the currently verified raw source into the project atomically."""

        snapshot = self.open(project_path)
        return self.save(
            snapshot.project_path,
            dataset=snapshot.dataset,
            source_dataset=snapshot.source_dataset,
            preprocess_config=snapshot.preprocess_config,
            analysis_config=snapshot.analysis_config,
            fit_result=snapshot.fit_result,
            fit_history=snapshot.fit_history,
            pair_design_state=snapshot.pair_design_state,
            diagnostics_result=snapshot.diagnostics_result,
            batch_result=snapshot.batch_result,
            global_result=snapshot.global_result,
            dct_result=snapshot.dct_result,
            dct_archive=snapshot.dct_archive,
            scientific_evidence=snapshot.scientific_evidence,
            dct_interpretations=snapshot.dct_interpretations,
            dual_error_map_result=snapshot.dual_error_map_result,
            dual_error_map_config=snapshot.dual_error_map_config,
            dual_error_map_stale=snapshot.dual_error_map_stale,
            report_snapshot=snapshot.report_snapshot,
            raw_data_mode=RawDataMode.PORTABLE,
            raw_source_path=snapshot.raw_data_path,
            seed=snapshot.seed,
            project_metadata=snapshot.project_metadata,
        )

    def _migrate(self, payload: dict[str, object]) -> dict[str, object]:
        current = dict(payload)
        schema = _schema_version(current)
        if schema in {1, 2, 3, 4}:
            return current
        while schema < 1:
            hook = self._migrations.get(schema)
            if hook is None:
                raise ProjectSchemaError(
                    f"project schema {schema} requires a migration hook"
                )
            migrated = hook(dict(current))
            if not isinstance(migrated, dict):
                raise ProjectSchemaError("migration hook must return a dict")
            next_schema = _schema_version(migrated)
            if next_schema <= schema:
                raise ProjectSchemaError(
                    "migration hook must increase schema_version"
                )
            current = migrated
            schema = next_schema
        if schema not in {1, 2, 3, 4}:
            raise ProjectSchemaError(f"unsupported migrated schema_version {schema}")
        return current

    def _write_project(
        self,
        root: Path,
        *,
        final_target: Path,
        dataset: EcrDataset,
        source_dataset: EcrDataset,
        preprocess_config: PreprocessConfig,
        analysis_config: AnalysisConfig | None,
        fit_result: FitResult | None,
        fit_history: tuple[FitHistoryEntry, ...],
        pair_design_state: PairDesignState | None,
        diagnostics_result: DiagnosticsPresentation | Mapping[str, object] | None,
        batch_result: BatchAnalysisResult | None,
        global_result: GlobalAnalysisResult | None,
        dct_result: DctPresentation | None,
        dct_archive,
        scientific_evidence,
        dct_interpretations,
        dual_error_map_result: DualErrorMapResult | None,
        dual_error_map_config: DualErrorMapConfig | None,
        dual_error_map_stale: bool,
        report_snapshot: AnalysisSnapshot | None,
        raw_data_mode: RawDataMode,
        raw_source_path: Path,
        raw_data_sha256: str,
        seed: int | None,
        project_metadata: Mapping[str, object],
        created_at: str,
        saved_at: str,
    ) -> None:
        for directory in ("data", "results", "plots", "logs"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        dataset_path = root / "data" / "dataset.json"
        _write_json(
            dataset_path,
            _datasets_to_dict(source_dataset, preprocess_config, dataset),
        )
        fit_relative: str | None = None
        if fit_result is not None:
            fit_relative = "results/fit_result.json"
            _write_json(root / fit_relative, fit_result.to_dict())
        history_relative: str | None = None
        if fit_history:
            history_relative = "results/fit_history.json"
            _write_json(root / history_relative, {
                "schema_version": 1,
                "entries": [entry.to_dict() for entry in fit_history],
            })
        diagnostics_relative: str | None = None
        pair_relative: str | None = None
        if pair_design_state is not None and not pair_design_state.empty:
            pair_relative = "results/pair_design.json"
            _write_json(root / pair_relative, pair_design_state.to_dict())
        if diagnostics_result is not None:
            diagnostics_relative = "results/diagnostics.json"
            _write_json(
                root / diagnostics_relative,
                _diagnostics_to_dict(diagnostics_result),
            )
        batch_relative: str | None = None
        if batch_result is not None:
            batch_relative = "results/batch_result.json"
            _write_json(root / batch_relative, batch_result.to_dict())
        global_relative: str | None = None
        if global_result is not None:
            global_relative = "results/global_result.json"
            _write_json(root / global_relative, global_result.to_dict())
        dct_relative: str | None = None
        if dct_result is not None:
            dct_relative = "results/dct_result.json"
            _write_json(root / dct_relative, dct_result.to_dict())
        dual_relative: str | None = None
        archive_relative = None
        if dct_archive is not None:
            archive_relative = 'results/dct_archive.json'
            _write_json(root / archive_relative, dct_archive.to_dict())
        scientific_relative=interpretation_relative=None
        if scientific_evidence is not None:
            scientific_relative='results/scientific_evidence.json';_write_json(root/scientific_relative,scientific_evidence)
        if dct_interpretations is not None:
            interpretation_relative='results/dct_interpretations.json';_write_json(root/interpretation_relative,dct_interpretations)
        if dual_error_map_result is not None:
            assert dual_error_map_config is not None
            dual_relative = "results/dual_error_map.json"
            payload: dict[str, object] = {
                "schema_version": 1,
                "status": "stale" if dual_error_map_stale else "current",
                "config": dual_error_map_config.to_dict(),
                "result": dual_error_map_result.to_dict(),
            }
            if not dual_error_map_stale:
                payload["binding"] = _dual_map_binding(dataset, analysis_config, fit_result)
            _write_json(root / dual_relative, payload)
        snapshot_relative: str | None = None
        if report_snapshot is not None:
            snapshot_relative = "results/analysis_snapshot.json"
            _write_json(root / snapshot_relative, report_snapshot.to_dict())

        if raw_data_mode is RawDataMode.PORTABLE:
            portable_name = raw_source_path.name or "raw_data"
            raw_relative = Path("data") / "raw" / portable_name
            raw_destination = root / raw_relative
            raw_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_source_path, raw_destination)
            raw_path_value = raw_relative.as_posix()
            path_type = "relative"
        else:
            raw_path_value, path_type = _linked_path(raw_source_path, final_target)

        payload: dict[str, object] = {
            "schema_version": 4 if scientific_evidence is not None or dct_interpretations is not None else 3 if dct_archive is not None else (_schema_for_config(analysis_config) if analysis_config is not None else 1),
            "project_type": "PyECR",
            "created_at": created_at,
            "saved_at": saved_at,
            "project_metadata": dict(project_metadata),
            "raw_data": {
                "mode": raw_data_mode.value,
                "path": raw_path_value,
                "path_type": path_type,
                "sha256": raw_data_sha256,
                "size_bytes": raw_source_path.stat().st_size,
            },
            "paths": {
                "dataset": "data/dataset.json",
                "fit_result": fit_relative,
                **({"fit_history": history_relative} if history_relative is not None else {}),
                **({"pair_design": pair_relative} if pair_relative is not None else {}),
                "diagnostics": diagnostics_relative,
                "batch_result": batch_relative,
                "global_result": global_relative,
                "dct_result": dct_relative,
                **({'dct_archive': archive_relative} if archive_relative is not None else {}),
                **({'scientific_evidence':scientific_relative} if scientific_relative is not None else {}),
                **({'dct_interpretations':interpretation_relative} if interpretation_relative is not None else {}),
                **({"dual_error_map": dual_relative} if dual_relative is not None else {}),
                **({"report_snapshot": snapshot_relative} if snapshot_relative is not None else {}),
                "plots": "plots",
                "logs": "logs",
            },
            "analysis_config": (
                None if analysis_config is None else analysis_config.to_dict()
            ),
            "reproducibility": {
                "seed": seed,
                "pyecr_version": _pyecr_version(),
                "model_version": MODEL_VERSION,
                "git_commit": _git_commit(),
            },
        }
        _write_json(root / "project.json", payload)
        managed_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "algorithm": "sha256",
            "generated_at": saved_at,
            "files": {
                path.relative_to(root).as_posix(): {
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in managed_files
            },
        }
        _write_json(root / "manifest.json", manifest)


def _validate_decomposition_view(metadata):
    from pyecr.report.scientific_evidence import HISTORY_KEY,validate_execution_history
    if HISTORY_KEY in metadata:validate_execution_history(metadata[HISTORY_KEY])
    if "decomposition_view" in metadata:
        from pyecr.plotting.decomposition import DecompositionViewOptions
        DecompositionViewOptions.from_dict(metadata["decomposition_view"])


def _datasets_to_dict(
    source: EcrDataset,
    recipe: PreprocessConfig,
    current: EcrDataset,
) -> dict[str, object]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset": _dataset_to_dict(source),
        "preprocess_config": recipe.to_dict(),
        "current_dataset": _dataset_to_dict(current),
    }


def _datasets_from_dict(
    payload: Mapping[str, object],
) -> tuple[EcrDataset, PreprocessConfig, EcrDataset]:
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ProjectSchemaError("unsupported dataset schema_version")
    source = _dataset_from_dict(
        _required_mapping(payload, "source_dataset")
    )
    recipe = PreprocessConfig.from_dict(
        _required_mapping(payload, "preprocess_config")
    )
    current = _dataset_from_dict(
        _required_mapping(payload, "current_dataset")
    )
    return source, recipe, current


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


def _dataset_from_dict(payload: Mapping[str, object]) -> EcrDataset:
    required = (
        "raw_time",
        "raw_signal",
        "processed_time",
        "processed_signal",
        "mask",
        "metadata",
        "history",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ProjectIntegrityError(
            "dataset payload is missing: " + ", ".join(missing)
        )
    history = payload["history"]
    if not isinstance(history, list):
        raise ProjectIntegrityError("dataset history must be an array")
    return EcrDataset(
        raw_time=payload["raw_time"],
        raw_signal=payload["raw_signal"],
        processed_time=payload["processed_time"],
        processed_signal=payload["processed_signal"],
        mask=payload["mask"],
        metadata=_mapping(payload["metadata"], "dataset metadata"),
        history=tuple(ProcessingRecord.from_dict(item) for item in history),
    )


def _diagnostics_to_dict(
    value: DiagnosticsPresentation | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(value, DiagnosticsPresentation):
        payload = asdict(value)
        if value.evaluation is None:
            payload.pop("evaluation", None)
        return {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "kind": "DiagnosticsPresentation",
            "payload": payload,
        }
    if isinstance(value, Mapping):
        return {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "kind": "mapping",
            "payload": _json_mapping(value, "diagnostics_result"),
        }
    raise TypeError(
        "diagnostics_result must be DiagnosticsPresentation, mapping, or None"
    )


def _diagnostics_from_dict(
    value: Mapping[str, object],
) -> DiagnosticsPresentation | Mapping[str, object]:
    if value.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
        raise ProjectSchemaError("unsupported diagnostics schema_version")
    payload = _required_mapping(value, "payload")
    if value.get("kind") == "mapping":
        return payload
    if value.get("kind") != "DiagnosticsPresentation":
        raise ProjectIntegrityError("unknown diagnostics result kind")
    parameters = tuple(
        ParameterPresentation(**dict(item))
        for item in _mapping_sequence(payload.get("parameters"), "parameters")
    )
    uncertainty_payload = _required_mapping(payload, "parameter_uncertainty")
    uncertainty_rows = tuple(
        ParameterPresentation(**dict(item))
        for item in _mapping_sequence(
            uncertainty_payload.get("rows"),
            "parameter_uncertainty.rows",
        )
    )
    bi_payload = _required_mapping(payload, "bi")
    correlation_payload = _required_mapping(payload, "correlation")
    jacobian_payload = _required_mapping(payload, "jacobian")
    multistart_payload = _required_mapping(payload, "multistart")
    return DiagnosticsPresentation(
        model_id=str(payload["model_id"]),
        time_s=tuple(float(item) for item in _sequence(payload["time_s"], "time_s")),
        residuals=tuple(
            float(item) for item in _sequence(payload["residuals"], "residuals")
        ),
        residual_summary=ResidualSummaryPresentation(
            **dict(_required_mapping(payload, "residual_summary"))
        ),
        bi=BiPresentation(
            available=bool(bi_payload["available"]),
            rows=tuple(
                BiDimensionPresentation(**dict(item))
                for item in _mapping_sequence(bi_payload.get("rows"), "bi.rows")
            ),
            reason=_optional_string(bi_payload.get("reason")),
        ),
        parameters=parameters,
        parameter_uncertainty=ParameterDiagnosticsPresentation(
            available=bool(uncertainty_payload["available"]),
            rows=uncertainty_rows,
            reason=_optional_string(uncertainty_payload.get("reason")),
        ),
        correlation=CorrelationPresentation(
            available=bool(correlation_payload["available"]),
            parameter_names=tuple(
                str(item)
                for item in _sequence(
                    correlation_payload.get("parameter_names", ()),
                    "correlation.parameter_names",
                )
            ),
            matrix=tuple(
                tuple(None if item is None else float(item) for item in row)
                for row in _sequence(
                    correlation_payload.get("matrix", ()),
                    "correlation.matrix",
                )
            ),
            reason=_optional_string(correlation_payload.get("reason")),
        ),
        jacobian=JacobianPresentation(**dict(jacobian_payload)),
        multistart=MultiStartPresentation(
            available=bool(multistart_payload["available"]),
            total_candidates=int(multistart_payload.get("total_candidates", 0)),
            comparable_candidates=int(
                multistart_payload.get("comparable_candidates", 0)
            ),
            rows=tuple(
                MultiStartSpreadPresentation(**dict(item))
                for item in _mapping_sequence(
                    multistart_payload.get("rows"),
                    "multistart.rows",
                )
            ),
            reason=_optional_string(multistart_payload.get("reason")),
        ),
        information_criteria=InformationCriteriaPresentation(
            **dict(_required_mapping(payload, "information_criteria"))
        ),
        optimizer_success=bool(payload["optimizer_success"]),
        reliable=bool(payload["reliable"]),
        reliability_label=str(payload["reliability_label"]),
        best_statistical=str(payload["best_statistical"]),
        best_reliable=_optional_string(payload.get("best_reliable")),
        gate_failures=tuple(
            str(item)
            for item in _sequence(payload.get("gate_failures", ()), "gate_failures")
        ),
        reasoning=tuple(
            str(item) for item in _sequence(payload.get("reasoning", ()), "reasoning")
        ),
        warnings=tuple(
            str(item) for item in _sequence(payload.get("warnings", ()), "warnings")
        ),
        sensitivity=_sensitivity_from_payload(payload.get("sensitivity")),
        reliability_advisor=_advisor_from_payload(payload.get("reliability_advisor")),
        evaluation=evaluation_from_dict(payload.get("evaluation")),
    )


def _sensitivity_from_payload(value: object) -> SensitivityPresentation:
    """Read P50 diagnostics while retaining schema-1 project compatibility."""

    if value is None:
        return SensitivityPresentation(
            available=False, reason="Sensitivity was not recorded in this project."
        )
    payload = _mapping(value, "sensitivity")
    curves = tuple(
        SensitivityCurve(
            parameter=str(item["parameter"]),
            label=str(item["label"]),
            values=tuple(float(v) for v in _sequence(item["values"], "sensitivity.values")),
            normalized_values=tuple(
                float(v) for v in _sequence(item["normalized_values"], "sensitivity.normalized_values")
            ),
            peak_absolute=float(item["peak_absolute"]),
            peak_time_s=float(item["peak_time_s"]),
            early_information_fraction=float(item["early_information_fraction"]),
            mid_information_fraction=float(item["mid_information_fraction"]),
            late_information_fraction=float(item["late_information_fraction"]),
            convergence_relative_error=float(item["convergence_relative_error"]),
        )
        for item in _mapping_sequence(payload.get("curves"), "sensitivity.curves")
    )
    return SensitivityPresentation(
        available=bool(payload.get("available", False)),
        curves=curves,
        reason=_optional_string(payload.get("reason")),
        method=str(payload.get("method", "central_finite_difference")),
        step_decades=float(payload.get("step_decades", 1.0e-3)),
    )


def _advisor_from_payload(value: object) -> ReliabilityAdvisorPresentation:
    if value is None:
        return ReliabilityAdvisorPresentation()
    payload = _mapping(value, "reliability_advisor")
    return ReliabilityAdvisorPresentation(
        identifiability_status=str(payload.get("identifiability_status", "unavailable")),
        identifiability_explanation=str(payload.get("identifiability_explanation", "No current identifiability evidence.")),
        model_adequacy_status=str(payload.get("model_adequacy_status", "unavailable")),
        model_adequacy_explanation=str(payload.get("model_adequacy_explanation", "No current fit-adequacy evidence.")),
        bi_guidance=str(payload.get("bi_guidance", "Bi is unavailable for this model.")),
        evidence=tuple(
            str(item) for item in _sequence(payload.get("evidence", ()), "reliability_advisor.evidence")
        ),
    )


def _validate_replay(
    source: EcrDataset,
    recipe: PreprocessConfig,
    expected: EcrDataset,
) -> None:
    try:
        replayed = recipe.replay(source)
    except Exception as exc:
        raise ProjectIntegrityError(
            f"preprocessing replay failed: {type(exc).__name__}: {exc}"
        ) from exc
    arrays = (
        ("raw_time", replayed.raw_time, expected.raw_time),
        ("raw_signal", replayed.raw_signal, expected.raw_signal),
        ("processed_time", replayed.processed_time, expected.processed_time),
        ("processed_signal", replayed.processed_signal, expected.processed_signal),
        ("mask", replayed.mask, expected.mask),
    )
    for name, actual, wanted in arrays:
        if not np.array_equal(actual, wanted):
            raise ProjectIntegrityError(
                f"saved current dataset does not match replayed {name}"
            )
    if replayed.metadata_as_dict() != expected.metadata_as_dict():
        raise ProjectIntegrityError(
            "saved current dataset metadata does not match preprocessing replay"
        )
    if tuple(item.to_dict() for item in replayed.history) != tuple(
        item.to_dict() for item in expected.history
    ):
        raise ProjectIntegrityError(
            "saved current dataset history does not match preprocessing replay"
        )


def _verify_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    files = _required_mapping(manifest, "files")
    for relative, entry in files.items():
        if not isinstance(relative, str):
            raise ProjectIntegrityError("manifest paths must be strings")
        managed = _managed_path(root, relative)
        if not managed.is_file():
            raise ProjectIntegrityError(f"managed project file is missing: {relative}")
        details = _mapping(entry, f"manifest entry {relative}")
        expected_hash = _required_string(details, "sha256")
        actual_hash = _sha256_file(managed)
        if actual_hash != expected_hash:
            raise ProjectHashMismatchError(
                f"managed file SHA256 mismatch for {relative}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        if int(details.get("size_bytes", -1)) != managed.stat().st_size:
            raise ProjectIntegrityError(f"managed file size mismatch for {relative}")


def _verify_layout(root: Path) -> None:
    for name in ("data", "results", "plots", "logs"):
        if not (root / name).is_dir():
            raise ProjectIntegrityError(
                f"required project directory is missing: {name}/"
            )


def _read_managed_json(
    root: Path,
    paths: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    relative = _required_string(paths, name)
    return _read_json_object(_managed_path(root, relative), relative)


def _read_optional_managed_json(
    root: Path,
    paths: Mapping[str, object],
    name: str,
) -> dict[str, object] | None:
    value = paths.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ProjectIntegrityError(f"paths.{name} must be a path or null")
    return _read_json_object(_managed_path(root, value), value)


def _managed_path(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProjectIntegrityError(
            f"managed path escapes the project directory: {relative}"
        ) from exc
    return candidate


def _resolve_raw_path(
    root: Path,
    raw_data: Mapping[str, object],
    mode: RawDataMode,
) -> Path:
    value = _required_string(raw_data, "path")
    path_type = raw_data.get("path_type")
    if path_type == "absolute":
        candidate = Path(value).expanduser().resolve()
    elif path_type == "relative":
        candidate = (root / Path(value)).resolve()
    else:
        raise ProjectIntegrityError("raw_data.path_type must be relative or absolute")
    if mode is RawDataMode.PORTABLE:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ProjectIntegrityError(
                "portable raw data path escapes the project directory"
            ) from exc
    return candidate


def _linked_path(source: Path, project_target: Path) -> tuple[str, str]:
    try:
        relative = os.path.relpath(source, start=project_target)
    except ValueError:
        return str(source), "absolute"
    return Path(relative).as_posix(), "relative"


def _atomic_directory_replace(temporary: Path, target: Path) -> None:
    backup: Path | None = None
    if target.exists():
        backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _validate_save_target(target: Path) -> None:
    if target.exists() and not target.is_dir():
        raise ProjectRepositoryError(f"project path is not a directory: {target}")
    if target.is_dir() and any(target.iterdir()) and not (target / "project.json").is_file():
        raise ProjectRepositoryError(
            "refusing to replace a nonempty directory that is not a PyECR project"
        )


def _project_target(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.name:
        raise ValueError("project path must name a directory")
    return path.resolve()


def _existing_directory(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ProjectRepositoryError(f"project directory is not accessible: {path}")
    return path


def _existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ProjectRepositoryError(f"{label} file is not accessible: {path}")
    return path


def _existing_created_at(target: Path) -> str | None:
    project_file = target / "project.json"
    if not project_file.is_file():
        return None
    try:
        payload = _read_json_object(project_file, "project.json")
    except ProjectRepositoryError:
        return None
    value = payload.get("created_at")
    return value if isinstance(value, str) and value else None


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProjectIntegrityError(f"required project file is missing: {label}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectIntegrityError(f"cannot read {label}: {exc}") from exc
    decoded = _decode_special_floats(value)
    if not isinstance(decoded, dict):
        raise ProjectIntegrityError(f"{label} root must be an object")
    return decoded


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_special_floats(payload)
    text = json.dumps(
        encoded,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def _encode_special_floats(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _encode_special_floats(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_special_floats(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        marker = "nan" if np.isnan(value) else "inf" if value > 0 else "-inf"
        return {"__pyecr_float__": marker}
    if isinstance(value, np.generic):
        return _encode_special_floats(value.item())
    return value


def _decode_special_floats(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {"__pyecr_float__"}:
            marker = value["__pyecr_float__"]
            if marker == "nan":
                return float("nan")
            if marker == "inf":
                return float("inf")
            if marker == "-inf":
                return float("-inf")
            raise ProjectIntegrityError("invalid special-float marker")
        return {str(key): _decode_special_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_special_floats(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        _encode_special_floats(value), sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dual_map_binding(
    dataset: EcrDataset,
    config: AnalysisConfig | None,
    fit: FitResult | None,
) -> dict[str, str]:
    if config is None or fit is None:
        raise ProjectIntegrityError("current dual Error Map requires config and FitResult")
    return {
        "dataset_sha256": dataset_sha256(dataset),
        "analysis_config_sha256": _canonical_hash(config.to_dict()),
        "fit_result_sha256": _canonical_hash(fit.to_dict()),
    }


def _validate_current_dual_map(
    result: DualErrorMapResult,
    config: DualErrorMapConfig | None,
    analysis_config: AnalysisConfig | None,
    fit: FitResult | None,
) -> None:
    from pyecr.models.ids import ModelID

    if (config is None or analysis_config is None or fit is None
            or analysis_config.model_id is not ModelID.M3_DUAL_HO_DK):
        raise ProjectIntegrityError("current dual Error Map requires an M3 FitResult and config")
    dispatch = fit.diagnostics.get("analysis_dispatch")
    if (not isinstance(dispatch, Mapping)
            or dispatch.get("selected_model_id") != ModelID.M3_DUAL_HO_DK.value
            or dispatch.get("m3_ran") is not True):
        raise ProjectIntegrityError("current dual Error Map requires final selected M3")
    revision = result.revision
    if (revision.fit_revision.data_revision != revision.dataset_revision
            or revision.fit_revision.config_revision != revision.model_revision):
        raise ProjectIntegrityError("dual Error Map revision is internally inconsistent")
    if not result.quality.fitted_point_parity_ok or any(
        not isclose(point.raw_sse, fit.SSE, rel_tol=1e-8, abs_tol=1e-12)
        for point in (result.h_map.fitted_point, result.o_map.fitted_point)
    ):
        raise ProjectIntegrityError("dual Error Map fit-point SSE parity failed")
    if any(len(axis) != config.grid_size for axis in (
        result.h_map.log10_D, result.h_map.log10_k,
        result.o_map.log10_D, result.o_map.log10_k,
    )):
        raise ProjectIntegrityError("dual Error Map resolution differs from config")


def _schema_version(payload: Mapping[str, object]) -> int:
    value = payload.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectSchemaError("project schema_version must be a nonnegative integer")
    return value


def _seed(value: int | None, config: AnalysisConfig | None, fit_result=None) -> int | None:
    from pyecr.fitting.reproducibility import require_reproducibility,reproducibility_issue
    if fit_result is not None:
        if reproducibility_issue(fit_result):return None
        return require_reproducibility(fit_result,value)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("seed must be a nonnegative integer or None")
        return value
    if config is None:
        return None
    for section in config.model_specific.values():
        if isinstance(section, Mapping):
            candidate = section.get("random_seed")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
    return None


def _pyecr_version() -> str:
    # Preserve the project schema field while identifying this distribution
    # by its application name, without a product release number.
    return "PyECR"


def _git_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProjectIntegrityError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _required_mapping(
    payload: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    if name not in payload:
        raise ProjectIntegrityError(f"project payload is missing {name}")
    return _mapping(payload[name], name)


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ProjectIntegrityError(f"{name} must be a nonempty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _sequence(value: object, label: str) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProjectIntegrityError(f"{label} must be an array")
    return value


def _mapping_sequence(
    value: object,
    label: str,
) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item, label) for item in _sequence(value or (), label))


def _json_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    try:
        encoded = _encode_special_floats(value)
        json.dumps(encoded, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-compatible") from exc
    return dict(value)
