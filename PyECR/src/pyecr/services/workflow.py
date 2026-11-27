"""Single-dataset workflow state, revisions, and fit orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from pyecr.contracts import SourceIdentity

from pyecr.config import AnalysisConfig
from pyecr.diagnostics.error_map import ErrorMapConfig, ErrorMapResult
from pyecr.diagnostics.dual_error_map import DualErrorMapConfig, DualErrorMapResult
from pyecr.fitting.base import FitResult
from pyecr.fitting.progress import fit_stage
from pyecr.fitting.dual_objective import DualObjectiveRevision
from pyecr.io.dataset import EcrDataset
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.models.ids import ModelID
from pyecr.preprocessing import (
    DownstreamStaleEvent,
    PreprocessConfig,
    PreprocessingPipeline,
    ProcessingStep,
)
from pyecr.preprocessing.quality import FitQualityReport, evaluate_fit_quality
from pyecr.report.snapshot import AnalysisSnapshot, dataset_sha256
from pyecr.services.analysis import (
    AnalysisService,
    CancellationToken,
    FitPresentation,
    ProgressReporter,
)
from pyecr.services.diagnostics_service import (
    AdvancedDiagnosticsPresentation,
    DiagnosticsPresentation,
)
from pyecr.services.error_map_service import ErrorMapComputation, ErrorMapContext
from pyecr.services.dual_error_map_service import (
    DualErrorMapComputation, DualErrorMapContext, DualErrorMapService,
    DualMapProgress,
)
from pyecr.services.guided_refit import (
    FitHistoryEntry, GuidedRefitEligibility, assess_guided_refit,
    build_guided_refit_config, fit_result_id, refit_comparison,
)

__all__ = [
    "AnalysisSession",
    "DiagnosticsRequest",
    "FitRequest",
    "GuidedRefitRequest",
    "GuidedRefitOutcome",
    "WorkflowController",
    "WorkflowDiagnosticsOutcome",
    "WorkflowFitOutcome",
    "WorkflowRevision",
    "WorkflowState",
]


class WorkflowState(str, Enum):
    """Highest currently valid milestone in a single-dataset analysis."""

    EMPTY = "EMPTY"
    IMPORTED = "IMPORTED"
    PREPROCESSED = "PREPROCESSED"
    CONFIGURED = "CONFIGURED"
    FITTED = "FITTED"
    DIAGNOSED = "DIAGNOSED"


@dataclass(frozen=True, slots=True)
class WorkflowRevision:
    """Physical-input versions captured when a fit is submitted."""

    data_revision: int
    config_revision: int
    physical_input_revision: int


@dataclass(frozen=True, slots=True)
class FitRequest:
    """Immutable dataset/config pair safe to execute in a worker thread."""

    dataset: EcrDataset
    config: AnalysisConfig
    revision: WorkflowRevision

    random_job: Mapping[str,object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowFitOutcome:
    """A fit and presentation tied to the revisions used to compute them."""

    request: FitRequest
    fit_result: FitResult
    presentation: FitPresentation


@dataclass(frozen=True, slots=True)
class GuidedRefitRequest:
    """Side-effect-free proposal bound to the exact current parent fit."""

    dataset: EcrDataset
    parent_config: AnalysisConfig
    parent_fit: FitResult
    parent_fit_id: str
    parent_revision: WorkflowRevision
    parent_fit_generation: int
    config: AnalysisConfig
    fixed_parameter: str
    fixed_value: float
    value_source: str


@dataclass(frozen=True, slots=True)
class GuidedRefitOutcome:
    request: GuidedRefitRequest
    fit_result: FitResult
    presentation: FitPresentation


@dataclass(frozen=True, slots=True)
class DiagnosticsRequest:
    """Immutable current-fit snapshot safe to diagnose in a worker."""

    dataset: EcrDataset
    config: AnalysisConfig
    fit_result: FitResult
    revision: WorkflowRevision


@dataclass(frozen=True, slots=True)
class WorkflowDiagnosticsOutcome:
    """Basic or advanced diagnostics tied to the fit's input revision."""

    request: DiagnosticsRequest
    kind: str
    presentation: DiagnosticsPresentation | AdvancedDiagnosticsPresentation


@dataclass(slots=True)
class AnalysisSession:
    """Authoritative state for one imported dataset and its derived analysis."""

    dataset: EcrDataset | None = None
    preprocessing: PreprocessingPipeline | None = None
    current_analysis_config: AnalysisConfig | None = None
    fit_result: FitResult | None = None
    fit_presentation: FitPresentation | None = None
    diagnostics: object | None = None
    advanced_diagnostics: dict[str, AdvancedDiagnosticsPresentation] = field(
        default_factory=dict
    )
    error_map_result: ErrorMapResult | None = None
    error_map_config: ErrorMapConfig | None = None
    error_map_context: ErrorMapContext | None = None
    error_map_stale: bool = False
    dual_error_map_result: DualErrorMapResult | None = None
    dual_error_map_config: DualErrorMapConfig | None = None
    dual_error_map_context: DualErrorMapContext | None = None
    dual_error_map_stale: bool = False
    dual_error_map_running: bool = False
    dual_error_map_last_status: str = "idle"
    dual_error_map_progress: DualMapProgress | None = None
    fit_stale: bool = False
    diagnostics_stale: bool = False
    data_revision: int = 0
    import_revision: int = 0
    preprocessing_revision: int = 0
    config_revision: int = 0
    physical_input_revision: int = 0
    project_metadata_revision: int = 0
    fit_revision: WorkflowRevision | None = None
    fit_generation: int = 0
    fit_job_provenance: Mapping[str,object] = field(default_factory=dict)
    fit_history: list[FitHistoryEntry] = field(default_factory=list)
    diagnostics_fit_revision: WorkflowRevision | None = None
    project_metadata: dict[str, object] = field(default_factory=dict)
    downstream_stale_events: list[DownstreamStaleEvent] = field(
        default_factory=list
    )
    # A fresh, explicitly recorded Diagnostics binds all P08 parent identities.
    diagnostics_identity: SourceIdentity | None = None
    diagnostics_fit_generation: int | None = None
    # Committed result metadata only; UI drafts never write these fields.
    selected_local_intervals: object | None = None
    noise_reference: object | None = None
    execution_evidence: object | None = None

    @property
    def state(self) -> WorkflowState:
        if (
            self.diagnostics is not None
            and not self.diagnostics_stale
            and self.fit_result is not None
            and not self.fit_stale
        ):
            return WorkflowState.DIAGNOSED
        if self.fit_result is not None and not self.fit_stale:
            return WorkflowState.FITTED
        if self.dataset is not None and self.current_analysis_config is not None:
            return WorkflowState.CONFIGURED
        if self.dataset is not None and self.preprocessing_revision > 0:
            return WorkflowState.PREPROCESSED
        if self.dataset is not None:
            return WorkflowState.IMPORTED
        return WorkflowState.EMPTY

    @property
    def current_revision(self) -> WorkflowRevision:
        return WorkflowRevision(
            data_revision=self.data_revision,
            config_revision=self.config_revision,
            physical_input_revision=self.physical_input_revision,
        )


class WorkflowController:
    """Coordinate import, preprocessing, configuration, and fitting revisions."""

    def __init__(
        self,
        *,
        session: AnalysisSession | None = None,
        analysis_service: AnalysisService | None = None,
    ) -> None:
        if session is not None and not isinstance(session, AnalysisSession):
            raise TypeError("session must be an AnalysisSession")
        if analysis_service is not None and not isinstance(
            analysis_service, AnalysisService
        ):
            raise TypeError("analysis_service must be an AnalysisService")
        self.session = session or AnalysisSession()
        self.analysis_service = analysis_service or AnalysisService()

    @property
    def can_fit(self) -> bool:
        return self.fit_quality.ready

    @property
    def fit_quality(self) -> FitQualityReport:
        """Current data/config quality, shared by GUI availability and Run Fit."""
        return evaluate_fit_quality(
            self.session.dataset, self.session.current_analysis_config
        )

    def import_dataset(self, dataset: EcrDataset) -> EcrDataset:
        if not isinstance(dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        old_pipeline = self.session.preprocessing
        if old_pipeline is not None:
            old_pipeline.unsubscribe(self._on_preprocessing_change)
        pipeline = PreprocessingPipeline(dataset)
        pipeline.subscribe(self._on_preprocessing_change)
        self.session.dataset = dataset
        self.session.preprocessing = pipeline
        self.session.import_revision += 1
        self.session.preprocessing_revision = 0
        self.session.data_revision += 1
        self.session.fit_history.clear()
        self.session.downstream_stale_events.clear()
        self._invalidate_fit_and_diagnostics()
        return dataset

    def restore_dataset(
        self,
        source_dataset: EcrDataset,
        preprocess_config: PreprocessConfig,
    ) -> EcrDataset:
        """Restore an imported source and deterministically replay its recipe."""

        if not isinstance(source_dataset, EcrDataset):
            raise TypeError("source_dataset must be an EcrDataset")
        if not isinstance(preprocess_config, PreprocessConfig):
            raise TypeError("preprocess_config must be a PreprocessConfig")
        old_pipeline = self.session.preprocessing
        if old_pipeline is not None:
            old_pipeline.unsubscribe(self._on_preprocessing_change)
        pipeline = PreprocessingPipeline(source_dataset, preprocess_config)
        pipeline.subscribe(self._on_preprocessing_change)
        self.session.dataset = pipeline.dataset
        self.session.preprocessing = pipeline
        self.session.import_revision += 1
        self.session.preprocessing_revision = pipeline.revision
        self.session.data_revision += 1
        self.session.fit_history.clear()
        self.session.downstream_stale_events.clear()
        self._invalidate_fit_and_diagnostics()
        return pipeline.dataset

    def apply_preprocessing(self, step: ProcessingStep) -> EcrDataset:
        from pyecr.preprocessing.optional import OPTIONAL_OPERATIONS, fit_summary
        session = self.session
        parent = None
        if (step.operation in OPTIONAL_OPERATIONS and session.fit_result is not None
                and not session.fit_stale and session.dataset is not None
                and session.current_analysis_config is not None):
            params = step.to_dict()["parameters"]
            params["before_fit"] = fit_summary(session.fit_result, data_revision=session.data_revision, dataset=session.dataset)
            params["before_fit_id"] = fit_result_id(session.fit_result)
            step = ProcessingStep.create(step.operation, params, timestamp=step.timestamp)
            config = session.current_analysis_config
            from pyecr.fitting.reproducibility import reproducibility_issue
            snapshot=None if reproducibility_issue(session.fit_result) else AnalysisSnapshot.capture(session.dataset, config, session.fit_result,
                session.diagnostics if session.diagnostics is not None and not session.diagnostics_stale else {})
            parent = FitHistoryEntry(fit_result_id(session.fit_result), config if snapshot is None else snapshot.analysis_config, session.fit_result, snapshot)
        result = self._require_pipeline().apply(step)
        if parent is not None and not any(entry.fit_id == parent.fit_id and (parent.report_snapshot is None or entry.report_snapshot is not None and entry.report_snapshot.dataset_sha256 == parent.report_snapshot.dataset_sha256) for entry in session.fit_history):
            session.fit_history.append(parent)
        return result

    def undo_preprocessing(self) -> EcrDataset:
        return self._require_pipeline().undo()

    def reset_preprocessing(self) -> EcrDataset:
        return self._require_pipeline().reset()

    def configure(self, config: AnalysisConfig) -> AnalysisConfig:
        if not isinstance(config, AnalysisConfig):
            raise TypeError("config must be an AnalysisConfig")
        DEFAULT_MODEL_REGISTRY.validate_config(config.model)
        config.validate()
        changed = config != self.session.current_analysis_config
        self.session.current_analysis_config = config
        if changed:
            self.session.config_revision += 1
            self._invalidate_fit_and_diagnostics()
        return config

    def prepare_fit(self) -> FitRequest:
        dataset = self.session.dataset
        config = self.session.current_analysis_config
        if dataset is None:
            raise RuntimeError("Fit requires an imported EcrDataset")
        if config is None:
            raise RuntimeError("Fit requires an applied AnalysisConfig")
        quality = self.fit_quality
        if not quality.ready:
            raise RuntimeError("Fit quality gate blocked Run Fit: " + "; ".join(quality.blocking_reasons))
        if self.session.diagnostics is not None or self.session.advanced_diagnostics:
            self.session.diagnostics_stale = True
        from pyecr.fitting.reproducibility import bind_config
        effective,job=bind_config(config)
        self.session.fit_job_provenance=job
        return FitRequest(dataset, effective, self.session.current_revision, job)

    def execute_fit(
        self,
        request: FitRequest,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
        iteration_observer=None,
    ) -> WorkflowFitOutcome:
        if not isinstance(request, FitRequest):
            raise TypeError("request must be a FitRequest")
        from pyecr.fitting.reproducibility import random_job_scope
        with random_job_scope(request.random_job):
            fit_result = self.analysis_service.fit(
                request.dataset, request.config, cancel_token=cancel_token, progress_callback=progress_callback,
                **({} if iteration_observer is None else {'iteration_observer':iteration_observer}),
            )
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if progress_callback is not None:
            progress_callback(0.90, "Preparing Fit plots and parameter table")
        fit_stage("Diagnostics", "Preparing local uncertainty and display data")
        presentation = self.analysis_service.describe_fit(
            request.dataset,
            request.config,
            fit_result,
        )
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        if progress_callback is not None:
            progress_callback(1.0, "Fit presentation ready")
        fit_stage("Finalizing", "Fit presentation ready; awaiting main-thread acceptance")
        return WorkflowFitOutcome(request, fit_result, presentation)

    def accept_fit(self, outcome: WorkflowFitOutcome) -> bool:
        if not isinstance(outcome, WorkflowFitOutcome):
            raise TypeError("outcome must be a WorkflowFitOutcome")
        if outcome.request.revision != self.session.current_revision:
            return False
        from pyecr.fitting.reproducibility import effective_config
        self.session.current_analysis_config=effective_config(outcome.request.config,outcome.fit_result)
        self.session.fit_result = outcome.fit_result
        self.session.fit_generation += 1
        self.session.fit_presentation = outcome.presentation
        self.session.fit_revision = outcome.request.revision
        self.session.fit_stale = False
        if self.session.diagnostics is not None or self.session.advanced_diagnostics:
            self.session.diagnostics_stale = True
        if self.session.error_map_result is not None:
            self.session.error_map_stale = True
        if self.session.dual_error_map_result is not None:
            self.session.dual_error_map_stale = True
        return True

    def run_fit(
        self,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
    ) -> WorkflowFitOutcome:
        request = self.prepare_fit()
        outcome = self.execute_fit(
            request,
            cancel_token=cancel_token,
            progress_callback=progress_callback,
        )
        if not self.accept_fit(outcome):  # pragma: no cover - synchronous path
            raise RuntimeError("Fit result became Out of date before it could be accepted")
        return outcome

    def assess_guided_refit(self) -> GuidedRefitEligibility:
        """Expose only the M1 case with one localized and one weak parameter."""

        session = self.session
        if session.fit_stale or session.diagnostics_stale:
            return GuidedRefitEligibility(False, reason="Current Fit and Diagnostics are required.")
        diagnostics = session.diagnostics
        return assess_guided_refit(
            session.current_analysis_config,
            session.fit_result,
            diagnostics if isinstance(diagnostics, DiagnosticsPresentation) else None,
        )

    def prepare_guided_refit(
        self,
        *,
        fixed_parameter: str,
        value_source: str,
        fixed_value: float | None = None,
    ) -> GuidedRefitRequest:
        """Build a proposed config without changing fit, config, or revisions."""

        eligibility = self.assess_guided_refit()
        if not eligibility.available:
            raise RuntimeError(eligibility.reason)
        if fixed_parameter != eligibility.reliable_parameter:
            raise ValueError("fix the localized parameter and refit the weak parameter")
        session = self.session
        dataset = session.dataset
        parent_config = session.current_analysis_config
        parent_fit = session.fit_result
        assert dataset is not None and parent_config is not None and parent_fit is not None
        if value_source == "parent_fit":
            selected_value = parent_fit.params[fixed_parameter]
            if fixed_value is not None and fixed_value != selected_value:
                raise ValueError("parent_fit source must use the exact parent estimate")
        elif value_source == "manual":
            if fixed_value is None:
                raise ValueError("manual source requires an explicit fixed_value")
            selected_value = fixed_value
        else:
            raise ValueError("value_source must be parent_fit or manual")
        parent_id = fit_result_id(parent_fit)
        config = build_guided_refit_config(
            parent_config, parent_fit,
            fixed_parameter=fixed_parameter,
            fixed_value=float(selected_value),
            value_source=value_source,
            parent_fit_id=parent_id,
        )
        quality = evaluate_fit_quality(dataset, config)
        if not quality.ready:
            raise RuntimeError("Fit quality gate blocked Guided Refit: " + "; ".join(quality.blocking_reasons))
        return GuidedRefitRequest(
            dataset, parent_config, parent_fit, parent_id,
            session.current_revision, session.fit_generation, config,
            fixed_parameter, float(selected_value), value_source,
        )

    def execute_guided_refit(
        self,
        request: GuidedRefitRequest,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
    ) -> GuidedRefitOutcome:
        """Run the production M1 fitter; cancellation leaves session untouched."""

        if not isinstance(request, GuidedRefitRequest):
            raise TypeError("request must be a GuidedRefitRequest")
        fit = self.analysis_service.fit(
            request.dataset, request.config,
            cancel_token=cancel_token, progress_callback=progress_callback,
        )
        if request.fixed_parameter in fit.params:
            raise RuntimeError("fixed parameter incorrectly appeared as a free FitResult parameter")
        provenance = {
            "parent_fit_id": request.parent_fit_id,
            "fixed_parameter": request.fixed_parameter,
            "fixed_value": request.fixed_value,
            "value_source": request.value_source,
            "parent_dataset_sha256": dataset_sha256(request.dataset),
            "comparison": refit_comparison(request.parent_fit, fit),
        }
        fit = replace(fit, diagnostics={**fit.diagnostics, "guided_refit": provenance})
        fit_stage("Diagnostics", "Preparing Guided Refit uncertainty and presentation")
        presentation = self.analysis_service.describe_fit(request.dataset, request.config, fit)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        fit_stage("Finalizing", "Guided Refit presentation ready")
        return GuidedRefitOutcome(request, fit, presentation)

    def accept_guided_refit(
        self,
        outcome: GuidedRefitOutcome,
        *,
        parent_snapshot: AnalysisSnapshot | None = None,
    ) -> bool:
        """Publish child fit atomically when the captured parent is still current."""

        if not isinstance(outcome, GuidedRefitOutcome):
            raise TypeError("outcome must be a GuidedRefitOutcome")
        request = outcome.request
        session = self.session
        if (
            request.parent_revision != session.current_revision
            or request.parent_fit_generation != session.fit_generation
            or request.dataset is not session.dataset
            or request.parent_config is not session.current_analysis_config
            or request.parent_fit is not session.fit_result
            or session.fit_stale
        ):
            return False
        if parent_snapshot is None:
            parent_snapshot = AnalysisSnapshot.capture(
                request.dataset, request.parent_config, request.parent_fit,
                session.diagnostics if session.diagnostics is not None and not session.diagnostics_stale else {},
            )
        parent_entry = FitHistoryEntry(
            request.parent_fit_id, request.parent_config, request.parent_fit,
            parent_snapshot,
        )
        session.fit_history.append(parent_entry)
        session.current_analysis_config = request.config
        session.config_revision += 1
        session.fit_result = outcome.fit_result
        session.fit_presentation = outcome.presentation
        session.fit_revision = session.current_revision
        session.fit_generation += 1
        session.fit_stale = False
        if session.diagnostics is not None or session.advanced_diagnostics:
            session.diagnostics_stale = True
        if session.error_map_result is not None:
            session.error_map_stale = True
        if session.dual_error_map_result is not None:
            session.dual_error_map_stale = True
        return True

    def rollback_guided_refit(
        self, *, current_snapshot: AnalysisSnapshot | None = None
    ) -> AnalysisSnapshot | None:
        """Restore the parent only while this child remains the current fit."""

        session = self.session
        child = session.fit_result
        config = session.current_analysis_config
        if (
            child is None or config is None or session.fit_stale
            or session.fit_revision != session.current_revision
        ):
            raise RuntimeError("No current Guided Refit can be rolled back")
        provenance = child.diagnostics.get("guided_refit")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("Current FitResult is not a Guided Refit")
        parent_id = provenance.get("parent_fit_id")
        matches = [entry for entry in session.fit_history if entry.fit_id == parent_id]
        if not matches:
            raise RuntimeError("Guided Refit parent is unavailable")
        parent = matches[-1]
        dataset = session.dataset
        if dataset is None or dataset_sha256(dataset) != provenance.get("parent_dataset_sha256"):
            raise RuntimeError("Dataset changed since Guided Refit")
        if current_snapshot is not None and (
            current_snapshot.fit_result.to_dict() != child.to_dict()
            or current_snapshot.analysis_config != config
        ):
            raise ValueError("current_snapshot does not capture the current Guided Refit")
        presentation = self.analysis_service.describe_fit(dataset, parent.analysis_config, parent.fit_result)
        session.fit_history.append(
            FitHistoryEntry(fit_result_id(child), config, child, current_snapshot)
        )
        session.current_analysis_config = parent.analysis_config
        session.config_revision += 1
        session.fit_result = parent.fit_result
        session.fit_presentation = presentation
        session.fit_revision = session.current_revision
        session.fit_generation += 1
        session.fit_stale = False
        if session.diagnostics is not None or session.advanced_diagnostics:
            session.diagnostics_stale = True
        if session.error_map_result is not None:
            session.error_map_stale = True
        if session.dual_error_map_result is not None:
            session.dual_error_map_stale = True
        return parent.report_snapshot

    @property
    def can_diagnose(self) -> bool:
        return self.session.fit_result is not None and not self.session.fit_stale

    @property
    def can_experimental_m1_uncertainty(self) -> bool:
        """UI availability only; the experimental service checks assumptions again."""

        config = self.session.current_analysis_config
        return bool(
            self.can_diagnose
            and self.session.fit_revision == self.session.current_revision
            and config is not None
            and config.model_id is ModelID.M1_SINGLE_DK
            and config.fit.loss == "linear"
            and "fixed_d" not in config.model_specific
            and config.boundary.mode == "ideal_step"
        )

    @property
    def can_error_map(self) -> bool:
        config = self.session.current_analysis_config
        return (
            self.can_diagnose
            and self.session.fit_revision == self.session.current_revision
            and config is not None
            and config.model_id is ModelID.M1_SINGLE_DK
            and not ({"D_cm2_s", "k_cm_s"} & set(config.model.fixed_parameters))
        )

    @property
    def can_dual_error_map(self) -> bool:
        config = self.session.current_analysis_config
        fit = self.session.fit_result
        dispatch = None if fit is None else fit.diagnostics.get("analysis_dispatch")
        return bool(
            self.can_diagnose
            and self.session.fit_revision == self.session.current_revision
            and config is not None
            and config.model_id is ModelID.M3_DUAL_HO_DK
            and "fixed_d" not in config.model_specific
            and isinstance(dispatch, Mapping)
            and dispatch.get("selected_model_id") == ModelID.M3_DUAL_HO_DK.value
            and dispatch.get("m3_ran") is True
        )

    @property
    def is_dual_error_map_model(self) -> bool:
        """Present the M3-only action without a second GUI model-ID list."""

        config = self.session.current_analysis_config
        return config is not None and config.model_id is ModelID.M3_DUAL_HO_DK

    def prepare_diagnostics(self) -> DiagnosticsRequest:
        dataset = self.session.dataset
        config = self.session.current_analysis_config
        fit_result = self.session.fit_result
        if dataset is None or config is None or fit_result is None:
            raise RuntimeError("Diagnostics requires a current dataset, config, and fit")
        if self.session.fit_stale:
            raise RuntimeError("Diagnostics cannot use an Out of date Fit")
        return DiagnosticsRequest(
            dataset=dataset,
            config=config,
            fit_result=fit_result,
            revision=self.session.current_revision,
        )

    def prepare_error_map(self) -> ErrorMapContext:
        """Capture a current M1 fit for execution outside the GUI main thread."""

        request = self.prepare_diagnostics()
        if request.config.model_id is not ModelID.M1_SINGLE_DK:
            raise ValueError("Error Map currently requires M1_SINGLE_DK")
        if {"D_cm2_s", "k_cm_s"} & set(request.config.model.fixed_parameters):
            raise ValueError("Error Map requires both D and k to be free")
        revision = request.revision
        return ErrorMapContext.capture(
            request.dataset,
            request.config,
            request.fit_result,
            revision=(
                revision.data_revision,
                revision.config_revision,
                revision.physical_input_revision,
            ),
        )

    def accept_error_map(self, outcome: ErrorMapComputation) -> bool:
        """Reject work completed for any superseded dataset, model, or fit."""

        if not isinstance(outcome, ErrorMapComputation):
            raise TypeError("outcome must be ErrorMapComputation")
        revision = self.session.current_revision
        if (
            outcome.context.revision != (
                revision.data_revision,
                revision.config_revision,
                revision.physical_input_revision,
            )
            or outcome.context.dataset is not self.session.dataset
            or outcome.context.analysis_config is not self.session.current_analysis_config
            or outcome.context.fit_result is not self.session.fit_result
            or self.session.fit_stale
        ):
            return False
        self.session.error_map_result = outcome.result
        self.session.error_map_config = outcome.config
        self.session.error_map_context = outcome.context
        self.session.error_map_stale = False
        return True

    def prepare_dual_error_map(self) -> DualErrorMapContext:
        """Capture a current selected M3 fit for a worker-private P44 run."""

        return DualErrorMapContext.capture(self.session)

    def accept_dual_error_map(self, outcome: DualErrorMapComputation) -> bool:
        """Publish H and O together only if every source revision is current."""

        if not isinstance(outcome, DualErrorMapComputation):
            raise TypeError("outcome must be DualErrorMapComputation")
        context = outcome.context
        fit_revision = self.session.fit_revision
        if (
            fit_revision is None
            or fit_revision != self.session.current_revision
            or context.revision != DualObjectiveRevision(
                self.session.data_revision,
                self.session.preprocessing_revision,
                self.session.config_revision,
                fit_revision,
            )
            or context.fit_generation != self.session.fit_generation
            or context.dataset is not self.session.dataset
            or context.analysis_config is not self.session.current_analysis_config
            or context.fit_result is not self.session.fit_result
            or self.session.fit_stale
            or context.selected_model_id is not ModelID.M3_DUAL_HO_DK
            or outcome.result.revision != context.revision
        ):
            return False
        self.session.dual_error_map_result = outcome.result
        self.session.dual_error_map_config = outcome.config
        self.session.dual_error_map_context = context
        self.session.dual_error_map_stale = False
        return True

    @property
    def current_dual_error_map_result(self) -> DualErrorMapResult | None:
        return DualErrorMapService.current_result(self.session)

    def accept_diagnostics(self, outcome: WorkflowDiagnosticsOutcome) -> bool:
        if not isinstance(outcome, WorkflowDiagnosticsOutcome):
            raise TypeError("outcome must be a WorkflowDiagnosticsOutcome")
        if (
            outcome.request.revision != self.session.current_revision
            or outcome.request.fit_result is not self.session.fit_result
            or self.session.fit_stale
        ):
            return False
        if outcome.kind == "basic":
            self.record_diagnostics(outcome.presentation)
        else:
            if not isinstance(outcome.presentation, AdvancedDiagnosticsPresentation):
                raise TypeError("advanced diagnostics require an advanced presentation")
            self.session.advanced_diagnostics[outcome.kind] = outcome.presentation
        return True

    def record_diagnostics(self, diagnostics: object) -> None:
        if self.session.fit_result is None or self.session.fit_stale:
            raise RuntimeError("current diagnostics require a current fit")
        self.session.diagnostics = diagnostics
        self.session.diagnostics_fit_revision = self.session.fit_revision
        self.session.diagnostics_stale = False
        self.session.diagnostics_identity = None
        self.session.diagnostics_fit_generation = None
        if (isinstance(diagnostics, DiagnosticsPresentation) and self.session.dataset is not None
                and self.session.current_analysis_config is not None and self.session.preprocessing is not None):
            from pyecr.contracts import SourceIdentity
            from pyecr.report.snapshot import _diagnostics_mapping
            self.session.diagnostics_identity = SourceIdentity.capture(self.session.dataset,
                self.session.preprocessing.config, config=self.session.current_analysis_config,
                fit=self.session.fit_result, diagnostics=_diagnostics_mapping(diagnostics))
            self.session.diagnostics_fit_generation = self.session.fit_generation

    def capture_component_sources(self, *, decomposition=None):
        """Explicit read-only P08 action; never runs normalization or DCT."""
        from pyecr.services.component_source import capture_component_sources
        return capture_component_sources(self.session, decomposition=decomposition)

    def update_project_metadata(
        self,
        updates: Mapping[str, object],
        *,
        affects_physical_inputs: bool = False,
    ) -> None:
        if not isinstance(updates, Mapping):
            raise TypeError("updates must be a mapping")
        self.session.project_metadata.update(dict(updates))
        self.session.project_metadata_revision += 1
        if affects_physical_inputs:
            self.session.physical_input_revision += 1
            self._invalidate_fit_and_diagnostics()

    def _require_pipeline(self) -> PreprocessingPipeline:
        pipeline = self.session.preprocessing
        if pipeline is None:
            raise RuntimeError("preprocessing requires an imported EcrDataset")
        return pipeline

    def _on_preprocessing_change(self, event: DownstreamStaleEvent) -> None:
        pipeline = self._require_pipeline()
        self.session.dataset = pipeline.dataset
        self.session.preprocessing_revision = event.revision
        self.session.data_revision += 1
        self.session.downstream_stale_events.append(event)
        self._invalidate_fit_and_diagnostics()

    def _invalidate_fit_and_diagnostics(self) -> None:
        if self.session.fit_result is not None:
            self.session.fit_stale = True
        if self.session.diagnostics is not None or self.session.advanced_diagnostics:
            self.session.diagnostics_stale = True
        if self.session.error_map_result is not None:
            self.session.error_map_stale = True
        if self.session.dual_error_map_result is not None:
            self.session.dual_error_map_stale = True
