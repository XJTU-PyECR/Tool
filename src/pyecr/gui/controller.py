"""Event interface joining GUI state, mock services, and workers."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace, asdict
from time import monotonic
from pathlib import Path
from typing import Literal, Mapping, cast

from pyecr.config import AnalysisConfig
from pyecr.diagnostics.error_map import ErrorMapConfig, ErrorMapResult
from pyecr.diagnostics.dual_error_map import DualErrorMapConfig, DualErrorMapResult
from pyecr.gui.model_builder import ModelFormValues, ModelPageConfigBuilder, ModelParameterValue
from pyecr.gui.services import GuiService, MockGuiService
from pyecr.gui.session import GuiSession
from pyecr.gui.state import GuiError, GuiNotice, GuiPage, GuiState
from pyecr.gui.worker import BackgroundTaskRunner, TaskEvent
from pyecr.gui.simulation_io import SimulationIoControllerMixin
from pyecr.gui.figure_exports import FigureExportControllerMixin
from pyecr.fitting.progress import FitProgressRevision
from pyecr.io.dataset import EcrDataset
from pyecr.io.importer import (
    HeaderMode,
    ImportMapping,
    ImportPreview,
    ImportResult,
    TextParseConfig,
)
from pyecr.models.ids import ModelID
from pyecr.models.registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelDescriptor,
    ModelRegistry,
)
from pyecr.preprocessing import PreprocessConfig, ProcessingStep
from pyecr.contracts import SignedDctConfig, SignedDctDiagnosticConfig, SignedNormalizationSpec
from pyecr.services.signed_dct_service import DctPhysicalContext, WholeSignedDctService
from pyecr.project import ProjectRepository, ProjectSnapshot, RawDataMode
from pyecr.report import AnalysisSnapshot
from pyecr.plotting import FitPlotData
from pyecr.services.analysis import (
    AnalysisService,
    CancellationToken,
)
from pyecr.services.batch_analysis import (
    BatchAnalysisResult,
    BatchAnalysisService,
    BatchDatasetInput,
    GlobalAnalysisResult,
)
from pyecr.services.diagnostics_service import (
    DiagnosticsPresentation,
    DiagnosticsService,
)
from pyecr.services.error_map_service import ErrorMapComputation, ErrorMapService
from pyecr.services.dual_error_map_service import (
    DualErrorMapComputation, DualErrorMapContext, DualErrorMapService, DualMapProgress,
)
from pyecr.services.dct_service import (
    DctAnalysisConfig,
    DctPresentation,
    DctService,
)
from pyecr.services.import_service import ImportService
from pyecr.services.report_service import ReportExportResult, ReportService
from pyecr.services.plot_preview_service import (
    BoundaryPlotComputation,
    ExplorerPlotComputation,
    PlotPreviewContext,
    PlotPreviewService,
)
from pyecr.services.workflow import (
    AnalysisSession,
    GuidedRefitOutcome,
    WorkflowDiagnosticsOutcome,
    WorkflowController,
    WorkflowFitOutcome,
    WorkflowRevision,
)
from pyecr.units import validate_pair
from pyecr.services.paired_ecr import PairMember, PairedComparison, compare_paired_ecr
from pyecr.services.experiment_design import ExperimentDesignRequest, PlanningSnapshot, scan_experiment_design
from pyecr.services.pair_design_state import PairDesignState
from pyecr.services.guided_refit import fit_result_id
from pyecr.services.m1_deep_uncertainty import ExperimentalM1UncertaintyResult
from pyecr.services.preprocessing_robustness import (
    RobustnessAssessment, RobustnessRequest, RobustnessVariation,
    prepare_robustness, run_robustness,
)

__all__ = ["GuiController", "UnitConfirmationRequired"]


class UnitConfirmationRequired(ValueError):
    """A legacy project has a signal type/unit pair that needs user approval."""


class GuiController(FigureExportControllerMixin, SimulationIoControllerMixin):
    """UI-facing commands with no equations or direct widget dependencies."""

    def __init__(
        self,
        *,
        state: GuiState | None = None,
        service: GuiService | None = None,
        runner: BackgroundTaskRunner | None = None,
        analysis_service: AnalysisService | None = None,
        import_service: ImportService | None = None,
        session: GuiSession | None = None,
        model_registry: ModelRegistry | None = None,
        workflow_controller: WorkflowController | None = None,
        diagnostics_service: DiagnosticsService | None = None,
        error_map_service: ErrorMapService | None = None,
        dual_error_map_service: DualErrorMapService | None = None,
        project_repository: ProjectRepository | None = None,
        batch_analysis_service: BatchAnalysisService | None = None,
        dct_service: DctService | None = None,
        report_service: ReportService | None = None,
        plot_preview_service: PlotPreviewService | None = None,
        simulation_service=None,
    ) -> None:
        self.state = state or GuiState()
        self.session = session or GuiSession()
        self.service = service or MockGuiService()
        self.runner = runner or BackgroundTaskRunner()
        if workflow_controller is not None:
            if workflow_controller.session is not self.session.analysis_session:
                raise ValueError(
                    "workflow_controller must own the GuiSession AnalysisSession"
                )
            if (
                analysis_service is not None
                and workflow_controller.analysis_service is not analysis_service
            ):
                raise ValueError(
                    "analysis_service must match workflow_controller.analysis_service"
                )
            self.workflow = workflow_controller
            self.analysis_service = workflow_controller.analysis_service
        else:
            self.analysis_service = analysis_service or AnalysisService()
            self.workflow = WorkflowController(
                session=self.session.analysis_session,
                analysis_service=self.analysis_service,
            )
        self.import_service = import_service or ImportService()
        self.diagnostics_service = diagnostics_service or DiagnosticsService(
            analysis_service=self.analysis_service
        )
        self.error_map_service = error_map_service or ErrorMapService()
        self.dual_error_map_service = dual_error_map_service or DualErrorMapService()
        self.project_repository = project_repository or ProjectRepository()
        self.batch_analysis_service = batch_analysis_service or BatchAnalysisService(
            analysis_service=self.analysis_service
        )
        self.dct_service = dct_service or DctService(isolate_numeric=True)
        self.signed_dct_service = WholeSignedDctService(isolate_numeric=True)
        from pyecr.services.component_dct_service import ComponentDctService
        self.component_dct_service = ComponentDctService(isolate_numeric=True)
        self._component_dct_tasks: dict[str, str] = {}
        self._signed_dct_tasks: dict[str, str] = {}
        self._signed_dct_binding = None
        self._dct_export_task_id = None
        self.dct_export_status = 'No DCT export requested'
        self.dct_export_result = None
        self.report_service = report_service or ReportService()
        self.plot_preview_service = plot_preview_service or PlotPreviewService()
        self.model_builder = ModelPageConfigBuilder(
            DEFAULT_MODEL_REGISTRY if model_registry is None else model_registry
        )
        self._cancellation_tokens: dict[str, CancellationToken] = {}
        self.monitored_runs = {}  # frozen completed evidence, including failed/cancelled candidates
        self.monitored_fit_records = {}  # successful real Fit object identity -> execution evidence
        self._fit_control_records = {}  # actual frozen Run request, separate from UI draft and computed output CI
        self._fit_live_binding = None  # latest worker's exact session/source; transient only
        self._fit_published_job = None
        self.monitored_task_jobs = {}  # existing worker task ID -> independent execution job ID
        self._dct_revisions: dict[str, WorkflowRevision] = {}
        self._dct_bindings: dict[str, tuple[object, object, object]] = {}
        self._dct_input_sources = {}
        self._legacy_dct_sources = {}  # actual newly-run result object + frozen source; never inferred on old read
        self._batch_requests = {}
        self._batch_result_contexts = {}  # At most the latest published source for each mode.
        from pyecr.services.dct_context_service import DctContextStore
        self.dct_context_store=DctContextStore()
        self._legacy_dct_context_results={}
        self._report_contexts: dict[str, tuple[AnalysisSnapshot, AnalysisSession, WorkflowRevision, bool]] = {}
        self._error_map_task_id: str | None = None
        self._dual_error_map_task_id: str | None = None
        self._guided_refit_task_id: str | None = None
        self._robustness_task_id: str | None = None
        self._robustness_request: RobustnessRequest | None = None
        self.robustness_assessment: RobustnessAssessment | None = None
        self.guided_refit_last_message = "No Guided Refit has been run."
        self._plot_preview_tasks: dict[str, str] = {}
        self._boundary_task_id: str | None = None
        self._explorer_task_id: str | None = None
        self.boundary_plot_result: BoundaryPlotComputation | None = None
        self.explorer_plot_result: ExplorerPlotComputation | None = None
        self._closed = False
        from pyecr.gui.simulation import SimulationWorkspace
        from pyecr.services.simulation_service import SimulationService
        self.simulation = SimulationWorkspace()
        self.simulation_service = simulation_service or SimulationService()
        self._simulation_tasks = {}
        self._initialize_simulation_io()

    def copy_analysis_to_simulation(self):
        """Explicit one-way copy of applied values; no Fit or Analysis edit."""
        return self.simulation.copy_analysis(self.analysis_session)

    def run_simulation(self):
        run = self.simulation.prepare()
        token = CancellationToken()
        def execute(progress):
            token.raise_if_cancelled()
            progress(0., 'Running frozen given-parameter request; numerical progress fraction is unavailable')
            result = self.simulation_service.run(run.request, cancel_token=token, progress_callback=progress)
            token.raise_if_cancelled()
            return result
        try:
            task_id = self.runner.submit(GuiPage.SIMULATION, execute)
        except Exception as exc:
            self.simulation.fail(run.job_id, str(exc))
            raise
        self._simulation_tasks[task_id] = run.job_id
        self._cancellation_tokens[task_id] = token
        task = self.state.tasks[GuiPage.SIMULATION]
        task.status='running';task.progress=0.;task.task_id=task_id;task.cancellation_requested=False
        task.message='Queued Simulation '+run.job_id
        self._sync_simulation_task_view()
        self.state.current_errors.pop(GuiPage.SIMULATION,None)
        return run.job_id

    def cancel_simulation(self, job_id=None):
        from pyecr.gui.simulation import CANCEL_NOTICE
        key=job_id or self.simulation.selected_id
        if key not in self.simulation.active_ids:return False
        self.simulation.update(key,status='cancelling',message=CANCEL_NOTICE)
        task=self.state.tasks[GuiPage.SIMULATION]
        task_id=next((t for t,k in self._simulation_tasks.items() if k==key),None)
        if task.task_id==task_id:task.cancellation_requested=True;task.message=CANCEL_NOTICE
        token=self._cancellation_tokens.get(task_id)
        self._sync_simulation_task_view()
        return False if token is None else token.cancel()

    def _sync_simulation_task_view(self):
        """Page badge summarizes pending jobs; each record owns its progress."""
        if not self.simulation.history:return
        latest=self.simulation.history[-1];task=self.state.tasks[GuiPage.SIMULATION]
        pending=self.simulation.active_ids
        task.status='running' if pending else latest.status if latest.status in {'completed','warning','cancelled'} else 'error'
        task.progress=latest.progress if latest.job_id in pending or not pending else 0.
        task.cancellation_requested=latest.status=='cancelling'
        task.message=f'{len(pending)} pending; latest submitted {latest.job_id} · {latest.status}: '+latest.message

    def _apply_simulation_event(self,event):
        key=self._simulation_tasks[event.task_id];task=self.state.tasks[GuiPage.SIMULATION]
        run=next(r for r in self.simulation.history if r.job_id==key)
        if event.kind=='progress':
            if run.status!='cancelling':
                self.simulation.update(key,status='running',message=event.message,progress=event.progress or 0.)
                if task.task_id==event.task_id:task.message=event.message;task.progress=event.progress or 0.
            self._sync_simulation_task_view()
            return
        token=self._cancellation_tokens.get(event.task_id)
        if event.kind=='cancelled' or run.status=='cancelling' or (token is not None and token.is_cancelled):
            self.simulation.finish_cancel(key)
        elif event.kind=='completed':
            try:self.simulation.complete(key,event.result)
            except Exception as exc:self.simulation.fail(key,str(exc))
        else:self.simulation.fail(key,event.message,event.traceback_text or '')
        record=next(r for r in self.simulation.history if r.job_id==key)
        if task.task_id==event.task_id:
            task.status=record.status if record.status in {'completed','warning','cancelled'} else 'error'
            task.message=record.message;task.progress=1. if record.result is not None and record.result.state.ready else 0.
            if record.result is not None and record.result.state.ready:self.state.latest_results[GuiPage.SIMULATION]=record.result
        self._simulation_tasks.pop(event.task_id,None);self._cancellation_tokens.pop(event.task_id,None)
        self._sync_simulation_task_view()

    def new_project(self) -> None:
        """Clear the current project and start a fresh single-dataset workflow."""

        self._ensure_no_running_tasks()
        self.signed_dct_service.bind_inputs(None, force=True)
        self.session.import_preview = None
        self.session.import_result = None
        self.session.project_snapshot = None
        self.session.batch_inputs.clear()
        self.session.batch_result = None
        self.session.global_result = None
        self.session.dct_config = None
        self.session.dct_result = None
        self.session.report_snapshot = None
        self.session.report_export_result = None
        self.session.pair_design_state = PairDesignState()
        self._legacy_dct_sources.clear()
        self._legacy_dct_context_results.clear()
        self.dct_context_store.history.clear()
        self.dct_context_store.selected.clear()
        self.monitored_runs.clear()
        self.monitored_fit_records.clear()
        self._fit_control_records.clear()
        self._guided_refit_task_id = None
        self._robustness_task_id = None
        self._robustness_request = None
        self.robustness_assessment = None
        self.guided_refit_last_message = "No Guided Refit has been run."
        self.session.analysis_session = AnalysisSession()
        self.workflow = WorkflowController(
            session=self.session.analysis_session,
            analysis_service=self.analysis_service,
        )
        self.component_dct_service.bind_session(self.analysis_session)
        self.state = GuiState(active_page=GuiPage.PROJECT)
        task = self.state.tasks[GuiPage.PROJECT]
        task.status = "completed"
        task.progress = 1.0
        task.message = "New project ready"

    def save_project(
        self,
        project_path: str | Path | None = None,
        *,
        raw_data_mode: RawDataMode | str | None = None,
    ) -> ProjectSnapshot:
        """Atomically save the current reproducible analysis state."""

        self._ensure_no_running_tasks()
        session = self.analysis_session
        dataset = session.dataset
        pipeline = session.preprocessing
        if dataset is None or pipeline is None:
            raise RuntimeError("saving a project requires an imported dataset")
        target_value = project_path or self.state.project_path
        if target_value is None:
            raise RuntimeError("Save As is required before the first project save")
        previous = self.session.project_snapshot
        mode = RawDataMode(
            raw_data_mode
            if raw_data_mode is not None
            else (
                previous.raw_data_mode
                if previous is not None
                else RawDataMode.LINKED
            )
        )
        metadata_source = dataset.metadata.get("source_path")
        metadata_hash = dataset.metadata.get("source_sha256")
        raw_source = (
            previous.raw_data_path
            if (
                previous is not None
                and metadata_hash == previous.raw_data_sha256
            )
            else metadata_source
        )
        if not isinstance(raw_source, (str, Path)) or not str(raw_source).strip():
            raise RuntimeError(
                "the imported dataset has no source_path; it cannot be linked or copied"
            )
        fit_result = None if session.fit_stale else session.fit_result
        diagnostics = None if session.diagnostics_stale else session.diagnostics
        scientific=None
        if fit_result is not None:
            scientific=self.report_service.capture_snapshot(session,include_export_context=True,
                execution_evidence=self.monitored_fit_records.get(id(fit_result)),
                control_request=self._fit_control_records.get(id(fit_result))).scientific_evidence or None
        archive=self.capture_dct_archive()
        interpretations=self.capture_dct_interpretations(archive)
        snapshot = self.project_repository.save(
            target_value,
            dataset=dataset,
            source_dataset=pipeline.source,
            preprocess_config=pipeline.config,
            analysis_config=session.current_analysis_config,
            fit_result=fit_result,
            diagnostics_result=diagnostics,  # type: ignore[arg-type]
            batch_result=self.session.batch_result,
            global_result=self.session.global_result,
            dct_result=self.session.dct_result,
            dct_archive=archive,
            scientific_evidence=scientific,
            dct_interpretations=interpretations,
            dual_error_map_result=session.dual_error_map_result,
            dual_error_map_config=session.dual_error_map_config,
            dual_error_map_stale=(
                session.dual_error_map_result is not None
                and self.current_dual_error_map_result is None
            ),
            report_snapshot=self.session.report_snapshot,
            fit_history=session.fit_history,
            pair_design_state=self.session.pair_design_state,
            raw_data_mode=mode,
            raw_source_path=raw_source,
            seed=(
                previous.seed
                if (
                    previous is not None
                    and session.current_analysis_config == previous.analysis_config
                )
                else None
            ),
            project_metadata=self._frozen_result_metadata(),
        )
        self.session.project_snapshot = snapshot
        self.session.pair_design_state = snapshot.pair_design_state or PairDesignState()
        self.session.batch_inputs.clear()
        self.session.batch_result = snapshot.batch_result
        self.session.global_result = snapshot.global_result
        self.session.dct_result = snapshot.dct_result
        self.session.dct_config = (
            None if snapshot.dct_result is None else snapshot.dct_result.config
        )
        self.session.report_snapshot = snapshot.report_snapshot
        self.session.report_export_result = None
        self.state.project_path = str(snapshot.project_path)
        task = self.state.tasks[GuiPage.PROJECT]
        task.status = "completed"
        task.progress = 1.0
        task.message = f"Saved {snapshot.raw_data_mode.value} project"
        self.state.latest_results[GuiPage.PROJECT] = snapshot
        return snapshot

    def open_project(
        self, project_path: str | Path, *, confirm_unknown_units: bool = False
    ) -> ProjectSnapshot:
        """Open, verify, replay, and publish a directory project."""

        self._ensure_no_running_tasks()
        snapshot = self.project_repository.open(project_path)
        source = snapshot.source_dataset.metadata
        source_type = str(source.get("raw_signal_type", source.get("signal_type", "unknown")))
        source_unit = str(source.get("raw_signal_unit", source.get("signal_unit", "arbitrary")))
        unit_warning = None
        try:
            validate_pair(source_type, source_unit)
        except ValueError as exc:
            unit_warning = (
                f"Legacy project has unrecognized/incompatible signal type–unit pair "
                f"{source_type!r} / {source_unit!r}: {exc}. Confirm to open without unit conversion."
            )
        if unit_warning and not confirm_unknown_units:
            raise UnitConfirmationRequired(unit_warning)
        self._restore_project_snapshot(snapshot)
        if unit_warning:
            task = self.state.tasks[GuiPage.PROJECT]
            task.status = "warning"
            task.message = unit_warning
        return snapshot

    def make_project_portable(self) -> ProjectSnapshot:
        """Copy the verified linked raw source into the current project."""

        self._ensure_no_running_tasks()
        if self.state.project_path is None:
            raise RuntimeError("save or open a project before making it portable")
        snapshot = self.project_repository.make_portable(self.state.project_path)
        self._restore_project_snapshot(snapshot)
        self.state.tasks[GuiPage.PROJECT].message = "Project is now portable"
        return snapshot

    def _restore_project_snapshot(self, snapshot: ProjectSnapshot) -> None:
        self.signed_dct_service.bind_inputs(None, force=True)
        self.session.import_preview = None
        self.session.import_result = None
        self.session.project_snapshot = snapshot
        self.session.pair_design_state = snapshot.pair_design_state or PairDesignState()
        self.session.batch_inputs.clear()
        self.session.batch_result = snapshot.batch_result
        self.session.global_result = snapshot.global_result
        self.session.dct_result = snapshot.dct_result
        self.session.dct_config = (
            None if snapshot.dct_result is None else snapshot.dct_result.config
        )
        self.session.report_snapshot = snapshot.report_snapshot
        self.session.report_export_result = None
        self._guided_refit_task_id = None
        self._robustness_task_id = None
        self._robustness_request = None
        self.robustness_assessment = None
        self.guided_refit_last_message = "No Guided Refit has been run."
        self.session.analysis_session = AnalysisSession()
        self.workflow = WorkflowController(
            session=self.session.analysis_session,
            analysis_service=self.analysis_service,
        )
        restored = self.workflow.restore_dataset(
            snapshot.source_dataset,
            snapshot.preprocess_config,
        )
        self.analysis_session.fit_history = list(snapshot.fit_history)
        if snapshot.analysis_config is not None:
            self.workflow.configure(snapshot.analysis_config)
        self.analysis_session.project_metadata.update(snapshot.project_metadata)
        if snapshot.scientific_evidence is not None:
            # Restore the persisted identity before capturing any map context.
            # Binding an H/O map at generation 0 and changing it afterwards
            # would immediately make the restored map appear out of date.
            self.analysis_session.fit_generation = snapshot.scientific_evidence['fit_generation']

        active_page = self.state.active_page
        self.state = GuiState(active_page=active_page)
        self.state.project_path = str(snapshot.project_path)
        self.state.dataset_name = str(
            restored.metadata.get("dataset_name")
            or restored.metadata.get("sample_id")
            or "Restored ECR dataset"
        )
        self.state.time_unit = str(restored.metadata.get("time_unit", "s"))
        self.state.signal_unit = str(restored.metadata.get("signal_unit", "")) or None
        if snapshot.analysis_config is not None:
            self.state.model_name = snapshot.analysis_config.model_id.value
            self.state.latest_results[GuiPage.MODEL] = snapshot.analysis_config
            model_task = self.state.tasks[GuiPage.MODEL]
            model_task.status = "completed"
            model_task.progress = 1.0
            model_task.message = "Restored validated AnalysisConfig"

        self.state.latest_results[GuiPage.DATA] = restored
        data_task = self.state.tasks[GuiPage.DATA]
        data_task.status = "completed"
        data_task.progress = 1.0
        data_task.message = "Restored dataset and preprocessing history"

        restored_batch = snapshot.global_result or snapshot.batch_result
        if restored_batch is not None:
            self.state.latest_results[GuiPage.BATCH] = restored_batch
            batch_task = self.state.tasks[GuiPage.BATCH]
            if snapshot.global_result is not None:
                batch_reliable = snapshot.global_result.reliable
            else:
                assert snapshot.batch_result is not None
                batch_reliable = all(
                    record.fit_success and record.reliable
                    for record in snapshot.batch_result.records
                )
            batch_task.status = "completed" if batch_reliable else "warning"
            batch_task.progress = 1.0
            batch_task.message = (
                "Restored Global/g-ECR result"
                if snapshot.global_result is not None
                else "Restored Batch/Arrhenius result"
            )

        if snapshot.dct_result is not None:
            self.state.latest_results[GuiPage.DCT] = snapshot.dct_result
            dct_task = self.state.tasks[GuiPage.DCT]
            has_warnings = bool(
                snapshot.dct_result.result.resolution_warnings
                or snapshot.dct_result.suggestion.artifact_risk
            )
            dct_task.status = "warning" if has_warnings else "completed"
            dct_task.progress = 1.0
            dct_task.message = "Restored DCT diagnostic"

        if snapshot.fit_result is not None:
            if snapshot.analysis_config is None:
                raise RuntimeError("saved FitResult has no AnalysisConfig")
            from pyecr.report.scientific_evidence import restore_gate_presentation
            presentation = restore_gate_presentation(
                restored,
                snapshot.analysis_config,
                snapshot.fit_result,
                snapshot.diagnostics_result,
                None if snapshot.scientific_evidence is None else snapshot.scientific_evidence['gate_fit_presentation'],
            )
            revision = self.analysis_session.current_revision
            self.analysis_session.fit_result = snapshot.fit_result
            self.analysis_session.fit_presentation = presentation
            self.analysis_session.fit_revision = revision
            self.analysis_session.fit_stale = False
            self.state.fit_presentation = presentation
            self.state.latest_results[GuiPage.FIT] = snapshot.fit_result
            fit_task = self.state.tasks[GuiPage.FIT]
            fit_task.status = "completed" if presentation.reliable else "warning"
            fit_task.progress = 1.0
            fit_task.message = "Restored FitResult"

        captured_map = snapshot.report_snapshot
        if (captured_map is not None and captured_map.error_map_result is not None
                and self.workflow.can_error_map):
            from pyecr.report.snapshot import dataset_sha256, _sha256_json
            session = self.analysis_session
            if (dataset_sha256(session.dataset) == captured_map.dataset_sha256
                    and session.current_analysis_config.to_dict() == captured_map.analysis_config.to_dict()
                    and _sha256_json(session.fit_result.to_dict()) == _sha256_json(captured_map.fit_result.to_dict())):
                # Bind the captured cells to the restored revision; never refit
                # or generate a grid while opening a project.
                context = self.workflow.prepare_error_map()
                self.workflow.accept_error_map(ErrorMapComputation(
                    context, captured_map.error_map_config, captured_map.error_map_result, True))

        if snapshot.diagnostics_result is not None:
            if snapshot.fit_result is None:
                raise RuntimeError("saved DiagnosticsResult has no FitResult")
            self.workflow.record_diagnostics(snapshot.diagnostics_result)
            if isinstance(snapshot.diagnostics_result, DiagnosticsPresentation):
                self.state.diagnostics_presentation = snapshot.diagnostics_result
            self.state.latest_results[GuiPage.DIAGNOSTICS] = (
                snapshot.diagnostics_result
            )
            diagnostics_task = self.state.tasks[GuiPage.DIAGNOSTICS]
            reliable = bool(
                getattr(snapshot.diagnostics_result, "reliable", False)
            )
            diagnostics_task.status = "completed" if reliable else "warning"
            diagnostics_task.progress = 1.0
            diagnostics_task.message = "Restored DiagnosticsResult"

        if snapshot.dual_error_map_result is not None:
            session = self.analysis_session
            session.dual_error_map_result = snapshot.dual_error_map_result
            session.dual_error_map_config = snapshot.dual_error_map_config
            session.dual_error_map_stale = snapshot.dual_error_map_stale
            session.dual_error_map_last_status = (
                "out_of_date" if snapshot.dual_error_map_stale else "completed"
            )
            if not snapshot.dual_error_map_stale:
                # Restore the saved revision namespace before binding a new
                # in-memory context. The result's original revision and arrays
                # remain untouched; no objective is evaluated on project open.
                original = snapshot.dual_error_map_result.revision
                if session.preprocessing is None or original.preprocessing_revision != session.preprocessing.revision:
                    raise RuntimeError("saved H/O map preprocessing revision disagrees with replayed project")
                session.data_revision = original.dataset_revision
                session.preprocessing_revision = original.preprocessing_revision
                session.config_revision = original.model_revision
                session.physical_input_revision = original.fit_revision.physical_input_revision
                session.fit_revision = original.fit_revision
                if session.diagnostics is not None:
                    session.diagnostics_fit_revision = original.fit_revision
                session.dual_error_map_context = DualErrorMapContext.capture(session)
                if self.current_dual_error_map_result is not session.dual_error_map_result:
                    raise RuntimeError("restored H/O Error Map failed current-result validation")
            self.state.latest_results[GuiPage.DIAGNOSTICS] = (
                snapshot.diagnostics_result or snapshot.dual_error_map_result
            )

        project_task = self.state.tasks[GuiPage.PROJECT]
        from pyecr.report.scientific_evidence import HISTORY_KEY,validate_execution_history,plain
        self.monitored_runs.clear();self.monitored_fit_records.clear();self._fit_control_records.clear()
        if HISTORY_KEY in snapshot.project_metadata:
            history=validate_execution_history(snapshot.project_metadata[HISTORY_KEY])
            self.monitored_runs.update(plain(history['records']))
        self._legacy_dct_sources.clear();self._legacy_dct_context_results.clear()
        self.dct_context_store.history.clear();self.dct_context_store.selected.clear()
        if snapshot.scientific_evidence is not None:
            evidence=snapshot.scientific_evidence;s=self.analysis_session
            if s.diagnostics is not None:s.diagnostics_fit_generation=s.fit_generation
            if evidence['selected_output']['record'] is not None:
                from pyecr.services.local_intervals import LocalIntervalRecord
                s.selected_local_intervals=LocalIntervalRecord.from_dict(evidence['selected_output']['record'])
            from pyecr.contracts import NoiseComparison
            s.noise_reference=NoiseComparison.from_dict(evidence['noise']).reference
            s.execution_evidence=evidence['termination']['execution']
            if s.execution_evidence is not None:self.monitored_fit_records[id(s.fit_result)]=s.execution_evidence
            if evidence['termination']['control_request'] is not None:self._fit_control_records[id(s.fit_result)]=evidence['termination']['control_request']
            from pyecr.services.advanced_intervals import advanced_from_dict
            from pyecr.report.scientific_evidence import plain
            s.advanced_diagnostics={k:advanced_from_dict(plain(v)) for k,v in evidence['advanced'].items()}
            self.state.advanced_diagnostics=dict(s.advanced_diagnostics)
        if snapshot.dct_interpretations is not None:
            from pyecr.project.interpretation_archive import restore_interpretations,legacy_targets
            restore_interpretations(snapshot.dct_interpretations,self.dct_context_store)
            self._legacy_dct_context_results=legacy_targets(snapshot.dct_interpretations)
            if self.session.dct_result is not None:
                from pyecr.contracts import canonical_sha256
                entry=self._legacy_dct_context_results.get(canonical_sha256(self.session.dct_result.to_dict()))
                if entry is not None:self._legacy_dct_sources[id(self.session.dct_result)]=(self.session.dct_result,entry[1])
        from pyecr.project.dct_archive import DctArchive
        (snapshot.dct_archive or DctArchive()).restore_history(self.signed_dct_service, self.component_dct_service)
        project_task.status = "completed"
        project_task.progress = 1.0
        project_task.message = (
            f"Opened {snapshot.raw_data_mode.value} project; SHA256 verified"
        )
        if snapshot.fit_result is not None and snapshot.scientific_evidence is None and self.analysis_session.fit_generation < 1:
            project_task.status = 'warning'
            project_task.message += ('; historical result has no persisted Fit generation. Saved results remain readable; '
                'component capture requires an explicitly run new Fit. No identity was inferred.')
        if snapshot.scientific_evidence is not None:
            from pyecr.report.scientific_evidence import historical_output_notice
            notice=historical_output_notice(snapshot.scientific_evidence)
            if notice:
                project_task.status='warning'
                project_task.message+='; '+notice
        self.state.latest_results[GuiPage.PROJECT] = snapshot

    def _ensure_no_running_tasks(self) -> None:
        if self._simulation_export_task_id is not None:
            raise RuntimeError('finish or cancel Simulation export before changing Analysis/project')
        if self._simulation_tasks:
            raise RuntimeError('finish or cancel all Simulation jobs before changing projects')
        if self._signed_dct_tasks or self._component_dct_tasks or self._dct_export_task_id:
            raise RuntimeError('finish or cancel DCT computation/export before changing projects')
        running = [
            page.value
            for page, task in self.state.tasks.items()
            if task.status == "running"
        ]
        if running:
            raise RuntimeError(
                "finish or cancel running tasks before changing projects: "
                + ", ".join(running)
            )
        if self._plot_preview_tasks:
            raise RuntimeError(
                "finish or cancel running plot preview tasks before changing projects"
            )

    def add_current_dataset_to_batch(
        self,
        *,
        record_id: str,
        temperature: float,
        temperature_unit: str,
        noise_weight: float = 1.0,
        fit_baseline: bool = True,
        fit_amplitude: bool = True,
    ) -> BatchDatasetInput:
        """Capture the current immutable dataset/config as one Batch entry."""

        dataset = self.analysis_session.dataset
        if dataset is None:
            raise RuntimeError("Batch requires a current imported dataset")
        entry = BatchDatasetInput(
            record_id=record_id,
            dataset=dataset,
            temperature=temperature,
            temperature_unit=temperature_unit,
            analysis_config=self.analysis_session.current_analysis_config,
            source_project=self.state.project_path,
            noise_weight=noise_weight,
            fit_baseline=fit_baseline,
            fit_amplitude=fit_amplitude,
        )
        self._append_batch_input(entry)
        return entry

    def add_project_to_batch(
        self,
        project_path: str | Path,
        *,
        record_id: str | None,
        temperature: float,
        temperature_unit: str,
        noise_weight: float = 1.0,
        fit_baseline: bool = True,
        fit_amplitude: bool = True,
    ) -> BatchDatasetInput:
        """Verify a project and add its restored dataset without opening it."""

        snapshot = self.project_repository.open(project_path)
        name = record_id or str(
            snapshot.dataset.metadata.get("dataset_name")
            or snapshot.project_path.name
        )
        entry = BatchDatasetInput(
            record_id=name,
            dataset=snapshot.dataset,
            temperature=temperature,
            temperature_unit=temperature_unit,
            analysis_config=snapshot.analysis_config,
            source_project=str(snapshot.project_path),
            noise_weight=noise_weight,
            fit_baseline=fit_baseline,
            fit_amplitude=fit_amplitude,
        )
        self._append_batch_input(entry)
        return entry

    def _append_batch_input(self, entry: BatchDatasetInput) -> None:
        if self.state.tasks[GuiPage.BATCH].status=='running':
            raise RuntimeError('Wait for Batch or Cancel before adding inputs')
        if any(item.record_id == entry.record_id for item in self.session.batch_inputs):
            raise ValueError(f"duplicate Batch record ID: {entry.record_id}")
        had_result=self.session.batch_result is not None or self.session.global_result is not None
        self.session.batch_inputs.append(entry)
        self.session.batch_result = None
        self.session.global_result = None
        task = self.state.tasks[GuiPage.BATCH]
        self.state.latest_results.pop(GuiPage.BATCH,None)
        task.status = "warning" if had_result else "idle"
        task.progress = 0.0
        task.message = ('Inputs changed. Run to update results.' if had_result else 'Input added. Ready to choose a method.')
        if had_result:self.record_notice(GuiPage.BATCH, "out_of_date", task.message)

    def clear_batch_inputs(self) -> None:
        self._ensure_no_running_tasks()
        self.session.batch_inputs.clear()
        self.session.batch_result = None
        self.session.global_result = None
        self.state.latest_results.pop(GuiPage.BATCH, None)
        task = self.state.tasks[GuiPage.BATCH]
        task.status = "idle"
        task.progress = 0.0
        task.message = "Batch inputs cleared"

    def remove_batch_input(self, record_id: str) -> None:
        """Remove one captured row explicitly; never edit it through a draft."""
        self._ensure_no_running_tasks()
        entries=self.session.batch_inputs
        index=next((i for i,e in enumerate(entries) if e.record_id==record_id),None)
        if index is None:raise ValueError('Batch input no longer exists')
        had_result=self.session.batch_result is not None or self.session.global_result is not None
        entries.pop(index)
        self.session.batch_result=None;self.session.global_result=None
        self.state.latest_results.pop(GuiPage.BATCH,None)
        task=self.state.tasks[GuiPage.BATCH];task.status='warning' if had_result else 'idle';task.progress=0.
        task.message='Input removed. Run to update results.' if had_result else 'Input removed.'

    def batch_result_context(self, result):
        """Frozen observations for a published result, never current drafts.

        Historical projects did not persist Batch inputs; absence stays explicit.
        This does not change result or project serialization.
        """
        return next((v for v in self._batch_result_contexts.values() if v[0] is result),None)

    def run_batch_independent(self) -> str:
        """Run production independent fits and Arrhenius in the worker pool."""

        return self._submit_batch_analysis("independent")

    def run_global_analysis(self) -> str:
        """Run the separate shared-D/k Global/g-ECR entry point."""

        return self._submit_batch_analysis("global")

    def _submit_batch_analysis(self, mode: str) -> str:
        self.state.current_errors.pop(GuiPage.BATCH, None)
        task = self.state.tasks[GuiPage.BATCH]
        if task.status == "running":
            raise RuntimeError("Batch already has a running task")
        config = self.analysis_session.current_analysis_config
        if config is None:
            raise RuntimeError("Batch requires an applied AnalysisConfig template")
        entries = tuple(self.session.batch_inputs)
        if not entries:
            raise RuntimeError("Batch requires at least one dataset/project entry")
        if mode == "global":
            reason = self.batch_analysis_service.global_unavailable_reason(entries, config)
            if reason:
                raise ValueError(reason)
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            if mode == "independent":
                return self.batch_analysis_service.run_independent(
                    entries,
                    config,
                    cancel_token=token,
                    progress_callback=progress,
                )
            return self.batch_analysis_service.run_global(
                entries,
                config,
                cancel_token=token,
                progress_callback=progress,
            )

        task_id = self.runner.submit(GuiPage.BATCH, execute)
        self._cancellation_tokens[task_id] = token
        task.status = "running"
        task.progress = 0.0
        task.message = (
            "Queued independent Batch fits"
            if mode == "independent"
            else "Queued Global/g-ECR fit"
        )
        task.task_id = task_id
        task.cancellation_requested = False
        self._batch_requests[task_id]=(self.session,entries,config,mode)
        return task_id

    @property
    def analysis_session(self) -> AnalysisSession:
        return self.workflow.session

    @property
    def fit_context_summary(self) -> str:
        session = self.analysis_session
        config = session.current_analysis_config
        configured_model = None if config is None else getattr(config, "model_id", None)
        model_id = (
            "Not configured"
            if configured_model is None
            else str(getattr(configured_model, "value", configured_model))
        )
        return (
            f"Dataset revision {session.data_revision} "
            f"(preprocessing revision {session.preprocessing_revision}) | "
            f"Model: {model_id} | Workflow: {session.state.value}"
        )

    @property
    def model_descriptors(self) -> tuple[ModelDescriptor, ...]:
        """Return Model-page choices in registry order."""

        return self.model_builder.descriptors

    @property
    def current_analysis_config(self) -> AnalysisConfig | None:
        """Latest validated Model-page configuration, if one was applied."""

        return self.session.current_analysis_config

    @property
    def can_run_batch_independent(self) -> bool:
        config = self.current_analysis_config
        return bool(
            self.session.batch_inputs
            and config is not None
            and self.batch_analysis_service.supports_independent(config.model_id)
            and "fixed_d" not in config.model_specific
        )

    @property
    def can_run_global_analysis(self) -> bool:
        return not self.batch_analysis_service.global_unavailable_reason(
            self.session.batch_inputs, self.current_analysis_config)

    def model_descriptor(self, model_id: ModelID | str) -> ModelDescriptor:
        """Resolve display and control metadata from the sole model registry."""

        return self.model_builder.descriptor(model_id)

    def default_model_form(self, model_id: ModelID | str) -> ModelFormValues:
        """Create descriptor-derived editable defaults without scientific work."""

        form = self.model_builder.default_form(model_id)
        dataset = self.analysis_session.dataset
        if form.model_id is ModelID.M7_SINGLE_K and dataset is not None and dataset.metadata.get("signal_type") != "normalized":
            import numpy as np
            y = dataset.processed_signal
            scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), 1.0)
            controls = dict(form.parameters)
            controls["amplitude"] = ModelParameterValue(float(y[-1] - y[0]), -10 * scale, 10 * scale, True)
            controls["baseline"] = ModelParameterValue(float(y[0]), float(np.min(y) - 2 * scale), float(np.max(y) + 2 * scale), True)
            form = replace(form, parameters=controls)
        if form.model_id is ModelID.M8_DUAL_K and dataset is not None:
            import numpy as np
            y=dataset.processed_signal
            scale=max(float(np.ptp(y)),float(np.max(np.abs(y))),1.)
            controls=dict(form.parameters)
            for name in ("A_1","A_2"):
                controls[name]=ModelParameterValue(.1,0.,10*scale,True)
            controls["k_1_cm_s"]=replace(controls["k_1_cm_s"],value=1e-3)
            if dataset.metadata.get("signal_type")!="normalized":
                controls["y0"]=ModelParameterValue(float(y[0]),float(np.min(y)-2*scale),float(np.max(y)+2*scale),True)
            form=replace(form,parameters=controls)
        return form

    def apply_model_configuration(
        self,
        values: ModelFormValues,
    ) -> AnalysisConfig:
        """Build, validate, and retain one AnalysisConfig without running a fit."""

        config = self.model_builder.build(values,parent_fit=self.analysis_session.fit_result if not self.analysis_session.fit_stale else None)
        previous_config = self.analysis_session.current_analysis_config
        self.workflow.configure(config)
        self._invalidate_fit_view_if_stale("AnalysisConfig changed")
        if config != previous_config:
            self.invalidate_dct("AnalysisConfig geometry changed")
        if config != previous_config and (
            self.session.batch_result is not None
            or self.session.global_result is not None
        ):
            self.session.batch_result = None
            self.session.global_result = None
            self.state.latest_results.pop(GuiPage.BATCH, None)
            batch_task = self.state.tasks[GuiPage.BATCH]
            if batch_task.status != "running":
                batch_task.status = "warning"
                batch_task.progress = 0.0
                batch_task.message = "Out of date — Batch template changed"
                self.record_notice(GuiPage.BATCH, "out_of_date", batch_task.message)
        self.state.model_name = config.model_id.value
        self.state.latest_results[GuiPage.MODEL] = config
        task = self.state.tasks[GuiPage.MODEL]
        task.status = "completed"
        task.progress = 1.0
        from pyecr.model_display import display_label
        task.message = f"Applied validated {display_label(config.model_id)} configuration"
        self.state.current_errors.pop(GuiPage.MODEL, None)
        return config

    def preview_data_file(
        self,
        source_path: str | Path,
        *,
        sheet_name: str | int | None = None,
        max_rows: int = 20,
        header_mode: HeaderMode | str = HeaderMode.AUTO,
        header_row_index: int | None = None,
        text_parse_config: TextParseConfig | None = None,
    ) -> ImportPreview:
        """Load Data-page headers and sample rows without importing a dataset."""

        preview = self.import_service.preview(
            source_path,
            sheet_name=sheet_name,
            max_rows=max_rows,
            header_mode=header_mode,
            header_row_index=header_row_index,
            text_parse_config=text_parse_config,
        )
        self.session.import_preview = preview
        self.session.import_result = None
        self.state.latest_results[GuiPage.DATA] = preview
        task = self.state.tasks[GuiPage.DATA]
        task.status = "completed"
        task.progress = 1.0
        task.message = f"Preview ready: {preview.row_count} rows"
        return preview

    def import_previewed_data(self, mapping: ImportMapping) -> ImportResult:
        """Import the selected columns and retain the immutable EcrDataset."""

        preview = self.session.import_preview
        if preview is None:
            raise RuntimeError("select and preview a data file before import")
        result = self.import_service.import_dataset(
            preview.source.source_path,
            mapping,
            expected_preview=preview,
        )
        self.session.import_result = result
        self.workflow.import_dataset(result.dataset)
        self.guided_refit_last_message = "No Guided Refit has been run."
        self._invalidate_fit_view_if_stale("Imported dataset changed")
        self.invalidate_dct("Imported dataset changed")
        self.state.latest_results[GuiPage.DATA] = result
        self.state.dataset_name = str(
            result.dataset.metadata.get("dataset_name") or "Imported ECR dataset"
        )
        self.state.time_unit = str(result.dataset.metadata.get("time_unit", "s"))
        self.state.signal_unit = (
            str(result.dataset.metadata.get("signal_unit", "")) or None
        )
        task = self.state.tasks[GuiPage.DATA]
        task.status = "completed"
        task.progress = 1.0
        task.message = f"Imported {result.dataset.raw_time.size} raw observations"
        self.state.current_errors.pop(GuiPage.DATA, None)
        return result

    def apply_preprocessing(self, step: ProcessingStep) -> EcrDataset:
        """Apply one serialized Data-page step and expose its immutable result."""

        dataset = self.workflow.apply_preprocessing(step)
        self._publish_processed_dataset(dataset, f"Applied: {step.operation}")
        self._invalidate_fit_view_if_stale("Preprocessing revision changed")
        self.invalidate_dct("Preprocessing revision changed")
        return dataset

    def undo_preprocessing(self) -> EcrDataset:
        """Remove the latest step and replay all preceding steps from raw data."""

        dataset = self.workflow.undo_preprocessing()
        self._publish_processed_dataset(dataset, "Undid latest preprocessing step")
        self._invalidate_fit_view_if_stale("Preprocessing revision changed")
        self.invalidate_dct("Preprocessing revision changed")
        return dataset

    def reset_preprocessing(self) -> EcrDataset:
        """Discard the recipe and restore the imported immutable source view."""

        dataset = self.workflow.reset_preprocessing()
        self._publish_processed_dataset(dataset, "Reset preprocessing pipeline")
        self._invalidate_fit_view_if_stale("Preprocessing revision changed")
        self.invalidate_dct("Preprocessing revision changed")
        return dataset

    def _publish_processed_dataset(self, dataset: EcrDataset, message: str) -> None:
        self.invalidate_report("Preprocessing revision changed")
        self.session.dataset = dataset
        self.state.latest_results[GuiPage.DATA] = dataset
        self.state.time_unit = str(dataset.metadata.get("time_unit", "s"))
        self.state.signal_unit = str(dataset.metadata.get("signal_unit", "")) or None
        task = self.state.tasks[GuiPage.DATA]
        task.status = "completed"
        task.progress = 1.0
        task.message = message

    def _invalidate_fit_view_if_stale(self, reason: str) -> None:
        if not self.analysis_session.fit_stale:
            return
        self.invalidate_report(reason)
        had_diagnostics = bool(self.state.diagnostics_presentation is not None
                               or self.state.advanced_diagnostics or GuiPage.DIAGNOSTICS in self.state.latest_results)
        self.state.fit_presentation = None
        self.state.diagnostics_presentation = None
        self.state.advanced_diagnostics.clear()
        self.state.latest_results.pop(GuiPage.DIAGNOSTICS, None)
        if had_diagnostics:
            diagnostics_task = self.state.tasks[GuiPage.DIAGNOSTICS]
            if diagnostics_task.status != "running":
                diagnostics_task.status = "warning"
                diagnostics_task.progress = 0.0
                diagnostics_task.message = f"Out of date — Diagnostics: {reason}"
                self.record_notice(GuiPage.DIAGNOSTICS, "out_of_date", diagnostics_task.message)
        if self.analysis_session.error_map_stale:
            if self.state.latest_results.get(GuiPage.DIAGNOSTICS) is self.state.error_map_result:
                self.state.latest_results.pop(GuiPage.DIAGNOSTICS, None)
            self.state.error_map_result = None
        self.state.latest_results.pop(GuiPage.FIT, None)
        task = self.state.tasks[GuiPage.FIT]
        if task.status != "running":
            task.status = "warning"
            task.progress = 0.0
            task.message = f"Out of date — Fit: {reason}"
            self.record_notice(GuiPage.FIT, "out_of_date", task.message)
        self.cancel_boundary_plot()
        self.cancel_explorer_preview()
        self.boundary_plot_result = None
        self.explorer_plot_result = None

    def invalidate_dct(self, reason: str) -> None:
        """Discard a DCT result whose data or geometry context changed."""

        self._sync_signed_dct_inputs()
        if self.session.dct_result is None:
            return
        self.session.dct_result = None
        self.state.latest_results.pop(GuiPage.DCT, None)
        task = self.state.tasks[GuiPage.DCT]
        if task.status != "running":
            task.status = "warning"
            task.progress = 0.0
            task.message = f"Out of date — DCT: {reason}"
            self.record_notice(GuiPage.DCT, "out_of_date", task.message)

    def invalidate_report(self, reason: str) -> None:
        """Retain an explicitly captured snapshot as historical, clear current output."""
        had_current = GuiPage.REPORT in self.state.latest_results or self.session.report_export_result is not None
        self.session.report_export_result = None
        self.state.latest_results.pop(GuiPage.REPORT, None)
        task = self.state.tasks[GuiPage.REPORT]
        if had_current and task.status != "running":
            task.status, task.progress = "warning", 0.0
            task.message = f"Out of date — Report: {reason}; captured snapshot is historical"
            self.record_notice(GuiPage.REPORT, "out_of_date", task.message)

    def bind_fit_context(self, dataset: EcrDataset, config: object) -> None:
        """Deprecated test/client adapter; the desktop workflow does not use it.

        New clients must import data and apply a real ``AnalysisConfig``.  This
        adapter remains only for callers with injected test backends predating
        ``AnalysisSession`` and stores their inputs in the same revisioned state.
        """

        if not isinstance(dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        validator = getattr(config, "validate", None)
        if not callable(validator):
            raise TypeError("config must provide validate()")
        validator()
        self.workflow.import_dataset(dataset)
        self.analysis_session.current_analysis_config = config  # type: ignore[assignment]
        self.analysis_session.config_revision += 1
        self.state.dataset_name = str(
            dataset.metadata.get("dataset_name")
            or dataset.metadata.get("sample_id")
            or "Bound ECR dataset"
        )
        model_id = getattr(config, "model_id", None)
        if model_id is not None:
            self.state.model_name = str(getattr(model_id, "value", model_id))
        self.state.time_unit = str(dataset.metadata.get("time_unit", "s"))
        self.state.signal_unit = str(dataset.metadata.get("signal_unit", "")) or None

    def run_page(self, page: GuiPage, *, fit_control_binding=None) -> str:
        """Dispatch a page service without blocking the calling GUI thread."""

        normalized_page = GuiPage(page)
        self.state.current_errors.pop(normalized_page, None)
        task_state = self.state.tasks[normalized_page]
        if task_state.status == "running":
            raise RuntimeError(f"{normalized_page.value} already has a running task")
        if normalized_page is GuiPage.FIT:
            task_id = self._submit_fit(fit_control_binding=fit_control_binding)
        elif normalized_page is GuiPage.DIAGNOSTICS:
            task_id = self._submit_diagnostics("basic")
        elif normalized_page is GuiPage.BATCH:
            return self.run_batch_independent()
        elif normalized_page is GuiPage.DCT:
            return self.run_dct(self.session.dct_config or DctAnalysisConfig())
        elif normalized_page is GuiPage.SIMULATION:
            return self.run_simulation()
        elif normalized_page is GuiPage.REPORT:
            raise RuntimeError(
                "Report uses the explicit Capture Snapshot and Export actions"
            )
        else:
            context = self.state.snapshot()
            task_id = self.runner.submit(
                normalized_page,
                lambda progress: self.service.execute(
                    normalized_page,
                    context,
                    progress,
                ),
            )
        task_state.status = "running"
        task_state.progress = 0.0
        task_state.message = "Queued"
        task_state.task_id = task_id
        return task_id

    @property
    def can_capture_snapshot(self) -> bool:
        """Whether the session has a fit for the exact current revision."""

        session = self.analysis_session
        return bool(
            session.dataset is not None
            and session.current_analysis_config is not None
            and session.fit_result is not None
            and not session.fit_stale
            and session.fit_revision is not None
            and session.fit_revision == session.current_revision
        )

    @property
    def decomposition_view_options(self):
        from pyecr.plotting.decomposition import DecompositionViewOptions
        value = self.analysis_session.project_metadata.get("decomposition_view")
        return DecompositionViewOptions() if value is None else DecompositionViewOptions.from_dict(value)

    def set_decomposition_view_options(self, options):
        from pyecr.plotting.decomposition import DecompositionViewOptions
        if not isinstance(options, DecompositionViewOptions):
            raise TypeError("decomposition options required")
        self.analysis_session.project_metadata["decomposition_view"] = options.to_dict()

    def current_decomposition(self):
        from pyecr.plotting.decomposition import build_decomposition, DUAL_IDS
        session = self.analysis_session
        if not self.can_capture_snapshot or session.current_analysis_config.model_id.value not in DUAL_IDS:
            return None
        options = self.decomposition_view_options
        inputs = (session.dataset, session.current_analysis_config, session.fit_result)
        previous = getattr(self, "_decomposition_cache_inputs", ())
        if len(previous) != 3 or any(a is not b for a, b in zip(inputs, previous)) or getattr(self, "_decomposition_cache_options", None) != options:
            self._decomposition_cache = build_decomposition(session.dataset, session.current_analysis_config, session.fit_result, options)
            self._decomposition_cache_inputs = inputs
            self._decomposition_cache_options = options
        return self._decomposition_cache

    def capture_snapshot(self, *, include_export_context: bool = False) -> AnalysisSnapshot:
        """Capture the current valid analysis for reproducible reporting."""

        snapshot = self.report_service.capture_snapshot(
            self.analysis_session,
            execution_evidence=self.monitored_fit_records.get(id(self.analysis_session.fit_result)),
            control_request=self._fit_control_records.get(id(self.analysis_session.fit_result)),
            include_export_context=(include_export_context or bool(self.monitored_runs) or bool(self.analysis_session.project_metadata.get('plot_views')) or bool(
                self.analysis_session.project_metadata.get("error_map_view")
                and (self.current_error_map_result is not None or self.current_dual_error_map_result is not None))),
        )
        if snapshot.export_context:
            snapshot=replace(snapshot,export_context={**snapshot.export_context,'project_metadata':self._frozen_result_metadata()})
        from pyecr.plotting.decomposition import DUAL_IDS
        if snapshot.analysis_config.model_id.value in DUAL_IDS:
            snapshot = replace(snapshot, decomposition_view=self.decomposition_view_options.to_dict())
        if not self.session.pair_design_state.empty:
            snapshot = replace(snapshot, supplementary_results=self.session.pair_design_state.to_dict())
        dct=self.capture_dct_report_evidence(snapshot)
        if dct is not None:snapshot=replace(snapshot,dct_evidence=dct)
        self.session.report_snapshot = snapshot
        task = self.state.tasks[GuiPage.REPORT]
        task.status = "completed"
        task.progress = 1.0
        task.message = "Captured current AnalysisSnapshot"
        self.state.latest_results[GuiPage.REPORT] = snapshot
        return snapshot

    def _frozen_result_metadata(self):
        from pyecr.report.scientific_evidence import HISTORY_KEY,execution_history,plain
        metadata=dict(self.analysis_session.project_metadata)
        if self.monitored_runs:metadata[HISTORY_KEY]=plain(execution_history(self.monitored_runs))
        return metadata

    def map_view_options(self, family):
        from pyecr.plotting.map_display import MapViewOptions
        if family not in ("single", "dual"): raise ValueError("unknown map family")
        value = self.analysis_session.project_metadata.get("error_map_view", {}).get(family)
        return MapViewOptions() if value is None else MapViewOptions.from_dict(value)

    def set_map_view_options(self, family, options):
        from pyecr.plotting.map_display import MapViewOptions
        if family not in ("single", "dual") or not isinstance(options, MapViewOptions):
            raise TypeError("map family and MapViewOptions required")
        stored = dict(self.analysis_session.project_metadata.get("error_map_view", {}))
        stored[family] = options.to_dict()
        self.analysis_session.project_metadata["error_map_view"] = stored

    def start_export(self, destination, *, options, historical=False):
        """Queue an immutable export; cancellation and current publication are guarded."""
        if getattr(self,'_figure_export_task_id',None):raise RuntimeError('Wait for the active figure export')
        task = self.state.tasks[GuiPage.REPORT]
        if task.status == "running":
            raise RuntimeError("Export already running")
        snapshot = self.session.report_snapshot if historical else self.capture_snapshot(include_export_context=True)
        from pyecr.plotting.view_state import METADATA_KEY, freeze_views
        metadata = snapshot.export_context.get('project_metadata', {}) if historical and snapshot else self.analysis_session.project_metadata
        options = replace(options, plot_views=freeze_views(options.plot_views if not historical and options.plot_views is not None else metadata.get(METADATA_KEY)))
        if snapshot is None:
            raise RuntimeError("capture an AnalysisSnapshot before historical export")
        captured_session = self.analysis_session
        revision = captured_session.current_revision
        context = (snapshot, captured_session, revision, historical)
        batch = self.session.batch_result
        token = CancellationToken()
        def guard():
            if not self._report_context_is_current(context):
                raise RuntimeError("Out of date current analysis; export was not published")
        if not historical:
            guard()
        def execute(progress):
            return self.report_service.export_report_bundle(snapshot, destination, export_options=options,
                export_scope="historical" if historical else "current", batch_result=batch,
                cancel_check=token.raise_if_cancelled, before_publish=None if historical else guard, progress_callback=progress)
        task_id = self.runner.submit(GuiPage.REPORT, execute)
        self._report_contexts[task_id] = context
        self._cancellation_tokens[task_id] = token
        task.status, task.task_id, task.progress, task.message = "running", task_id, 0., "Export queued"
        self.session.report_snapshot = snapshot
        return task_id

    def _report_context_is_current(self, context):
        snapshot, session, revision, _ = context
        return bool(self.analysis_session is session and not session.fit_stale
                    and session.current_revision == revision and session.fit_revision == revision
                    and session.dataset is snapshot.dataset and session.current_analysis_config == snapshot.analysis_config
                    and session.fit_result is snapshot.fit_result)

    def export_report_bundle(
        self,
        destination: str | Path,
        *,
        snapshot: AnalysisSnapshot | None = None,
    ) -> ReportExportResult:
        """Export current analysis, or an explicitly supplied historical snapshot."""

        selected = self.capture_snapshot() if snapshot is None else snapshot
        experimental = self.analysis_session.advanced_diagnostics.get("experimental_m1")
        deep = None if experimental is None else experimental.experimental_result
        if not isinstance(deep, ExperimentalM1UncertaintyResult):
            deep = None
        if deep is not None and deep.source_fit_id != fit_result_id(selected.fit_result):
            deep = None
        result = self.report_service.export_report_bundle(
            selected, destination, experimental_uncertainty=deep,
        )
        self.session.report_snapshot = selected
        self.session.report_export_result = result
        task = self.state.tasks[GuiPage.REPORT]
        task.status = "completed"
        task.progress = 1.0
        task.message = (
            f"Exported and verified {len(result.artifacts)} report artifacts"
        )
        self.state.latest_results[GuiPage.REPORT] = result
        return result

    @property
    def can_capture_pair_member(self) -> bool:
        session = self.analysis_session
        return bool(self.can_capture_snapshot and session.diagnostics is not None
                    and not session.diagnostics_stale
                    and session.diagnostics_fit_revision == session.fit_revision)

    def _require_pair_idle(self) -> None:
        if self.state.tasks[GuiPage.PAIR].status == "running":
            raise RuntimeError("Finish or cancel the planning scan before changing Pair / Design artifacts")

    def set_pair_member(self, slot: str, member: PairMember) -> PairMember:
        """Capture a complete independent lane; never modify the active analysis."""
        self._require_pair_idle()
        if slot not in {"A", "B"} or not isinstance(member, PairMember):
            raise ValueError("Choose pair slot A or B and a PairMember")
        field = "member_a" if slot == "A" else "member_b"
        self.session.pair_design_state = replace(self.session.pair_design_state, **{field: member}, comparison=None)
        self.state.tasks[GuiPage.PAIR].message = f"Captured independent {slot}: {member.label}; comparison needs review"
        self.state.latest_results.pop(GuiPage.PAIR, None)
        return member

    def capture_pair_member(self, slot: str, *, label: str, direction: str,
                            context: Mapping[str, object] | None = None) -> PairMember:
        if not self.can_capture_pair_member:
            raise RuntimeError("Complete this curve's independent Import → Preprocess → Fit → Diagnostics first")
        snapshot = self.report_service.capture_snapshot(self.analysis_session)
        recorded = {key: value for key, value in self.analysis_session.project_metadata.items()
                    if key in {"temperature_k", "initial_condition", "final_condition", "step_direction"}}
        if any(key in recorded and recorded[key] != value for key, value in (context or {}).items()):
            raise ValueError("Pair context contradicts recorded project metadata")
        recorded.update(context or {})
        return self.set_pair_member(slot, PairMember(label, direction, snapshot, recorded))

    def load_pair_project(self, slot: str, project_path: str | Path, *, label: str, direction: str,
                          context: Mapping[str, object] | None = None) -> PairMember:
        self._require_pair_idle()
        project = self.project_repository.open(project_path)
        if project.analysis_config is None or project.fit_result is None or project.diagnostics_result is None:
            raise RuntimeError("Pair input project requires its own completed Fit and Diagnostics")
        snapshot = AnalysisSnapshot.capture(project.dataset, project.analysis_config, project.fit_result,
                                            project.diagnostics_result)
        recorded = {key: value for key, value in project.project_metadata.items()
                    if key in {"temperature_k", "initial_condition", "final_condition", "step_direction"}}
        if any(key in recorded and recorded[key] != value for key, value in (context or {}).items()):
            raise ValueError("Pair context contradicts recorded project metadata")
        recorded.update(context or {})
        return self.set_pair_member(slot, PairMember(label, direction, snapshot, recorded))

    def compare_pair(self, kind: str) -> PairedComparison:
        self._require_pair_idle()
        state = self.session.pair_design_state
        if state.member_a is None or state.member_b is None:
            raise RuntimeError("Capture or load both independently analyzed pair members")
        result = compare_paired_ecr(state.member_a, state.member_b, kind)
        self.session.pair_design_state = replace(state, comparison=result)
        self.state.latest_results[GuiPage.PAIR] = result
        task = self.state.tasks[GuiPage.PAIR]
        task.status = "completed" if result.ready else "warning"
        task.progress = 1.0
        task.message = "Pair comparison ready" if result.ready else "Pair comparison BLOCKED: " + "; ".join(result.compatibility.blocking_reasons)
        return result

    def run_experiment_design(self, request: ExperimentDesignRequest) -> str:
        self._require_pair_idle()
        request.validate()
        token = CancellationToken()
        task_id = self.runner.submit(GuiPage.PAIR, lambda progress: scan_experiment_design(
            request, cancel_token=token, progress_callback=progress))
        self._cancellation_tokens[task_id] = token
        task = self.state.tasks[GuiPage.PAIR]
        task.status, task.progress, task.task_id = "running", 0.0, task_id
        task.message = "Queued independent 03 M2_SINGLE_DK planning scan"
        self.state.current_errors.pop(GuiPage.PAIR, None)
        return task_id

    def save_pair_design_snapshot(self, destination: str | Path) -> Path:
        self._require_pair_idle()
        return self.session.pair_design_state.save(destination)

    def open_pair_design_snapshot(self, source: str | Path) -> PairDesignState:
        self._require_pair_idle()
        state = PairDesignState.open(source)
        self.session.pair_design_state = state
        if state.comparison is None:
            self.state.latest_results.pop(GuiPage.PAIR, None)
        else:
            self.state.latest_results[GuiPage.PAIR] = state.comparison
        self.state.tasks[GuiPage.PAIR].message = "Restored independent Pair / Design snapshot"
        return state

    def clear_pair_design(self) -> None:
        self._require_pair_idle()
        self.session.pair_design_state = PairDesignState()
        self.state.latest_results.pop(GuiPage.PAIR, None)
        self.state.tasks[GuiPage.PAIR].message = "Pair / Design artifacts cleared"

    def _sync_signed_dct_inputs(self) -> None:
        session = self.analysis_session
        recipe = session.preprocessing.config if session.preprocessing else None
        # Dataset/recipe objects are immutable. Hash complete arrays only on an
        # input/revision change, not on every native UI polling frame.
        binding = (id(session), session.data_revision, id(session.dataset), id(recipe))
        if binding == self._signed_dct_binding:
            return
        self.signed_dct_service.bind_inputs(session.dataset, recipe if recipe is not None else PreprocessConfig(),
            owner_token=(id(session), session.data_revision))
        self._signed_dct_binding = binding

    @property
    def dct_activity(self):
        """Actual active operations, independent of the selected mode/history/tab."""
        task=self.state.tasks[GuiPage.DCT]
        rows=[('monotonic',task.task_id,task.cancellation_requested)] if task.status=='running' and task.task_id in self._dct_revisions else []
        rows.extend(('signed',j.job_id,j.cancellation_requested) for j in self.signed_dct_service.history if j.status in {'queued','running'})
        rows.extend(('component',j.job_id,j.cancellation_requested) for j in self.component_dct_service.history if j.status in {'queued','running'})
        return tuple(rows)

    def _require_dct_idle(self):
        if self.dct_activity:raise RuntimeError('Wait for the active DCT task to exit, or Cancel it before another Run.')

    def cancel_active_dct(self):
        requested=False
        for kind,key,_ in self.dct_activity:
            if kind=='monotonic':requested=self.cancel_page(GuiPage.DCT) or requested
            elif kind=='signed':requested=self.cancel_signed_dct(key) or requested
            else:requested=self.cancel_component_dct(key) or requested
        return requested

    @property
    def current_signed_dct(self):
        """Current whole-source job; kept separate from legacy DctPresentation."""
        self._sync_signed_dct_inputs()
        return self.signed_dct_service.current

    def run_signed_dct(self, config: SignedDctConfig, *,
            normalization_spec: SignedNormalizationSpec | None = None,
            diagnostic_config=None,convex_options=None) -> str:
        """P06 service adapter only; no Apply, Fit, page switch or widget mutation."""
        self._require_dct_idle()
        self._sync_signed_dct_inputs()
        session = self.analysis_session
        if session.dataset is None:
            raise RuntimeError('Signed-DCT requires an imported experimental dataset')
        recipe = session.preprocessing.config if session.preprocessing else PreprocessConfig()
        key = self.signed_dct_service.create_job(session.dataset, recipe, config,
            normalization_spec=normalization_spec, diagnostic_config=diagnostic_config,
            convex_options=convex_options,
            physical_context=DctPhysicalContext.capture(session.current_analysis_config))
        try:
            task_id = self.runner.submit(GuiPage.DCT, lambda progress:
                self.signed_dct_service.run(key, progress_callback=progress))
        except Exception as exc:
            self.signed_dct_service.fail_queued(key, exc)
            raise
        self._signed_dct_tasks[task_id] = key
        return key

    def capture_dct_archive(self):
        """Freeze stored runs, including their own historical datasets; no draft execution."""
        from pyecr.project.dct_archive import DctArchive
        self._sync_signed_dct_inputs()
        self.component_dct_service.bind_session(self.analysis_session)
        whole, components = self.signed_dct_service.history, self.component_dct_service.history
        return DctArchive(whole, components) if whole or components else None

    def capture_dct_interpretations(self,archive,*,scoped=False):
        from pyecr.project.interpretation_archive import capture_interpretations,targets
        from pyecr.services.dct_context_service import DctContextStore
        from pyecr.contracts import canonical_sha256
        result=self.session.dct_result;entry=None if result is None else self._legacy_dct_sources.get(id(result))
        legacy=dict(self._legacy_dct_context_results)
        if entry is not None and entry[0] is result:legacy[canonical_sha256(result.to_dict())]=entry
        if scoped:legacy={}
        keys=set(targets(archive))|set(legacy)
        store=DctContextStore();store.history=[r for r in self.dct_context_store.history if r.target_sha256 in keys] if scoped else list(self.dct_context_store.history)
        ids={r.content_sha256 for r in store.history}
        store.selected={k:r for k,r in self.dct_context_store.selected.items() if r.content_sha256 in ids}
        if archive is None and not legacy and not store.history:return None
        return capture_interpretations(store,archive,legacy_targets=legacy)

    def capture_dct_report_evidence(self,snapshot):
        from pyecr.project.dct_archive import DctArchive
        from pyecr.project.interpretation_archive import dct_evidence,capture_interpretations,legacy_targets
        from pyecr.contracts import SourceIdentity,canonical_sha256
        from pyecr.services.dct_context_service import DctContextStore
        if not snapshot.scientific_evidence:return None
        source=SourceIdentity.from_dict(snapshot.scientific_evidence['source']);archive=self.capture_dct_archive()
        def same(s,component=False):
            return (s.dataset_sha256,s.recipe_sha256)==(source.dataset_sha256,source.recipe_sha256) and (not component or s==source)
        if archive is not None:
            archive=DctArchive(tuple(j for j in archive.whole_jobs if same(j.request.identity)),
                tuple(j for j in archive.component_jobs if same(j.request.capture.identity,True)))
            if not archive.whole_jobs and not archive.component_jobs:archive=None
        full=self.capture_dct_interpretations(self.capture_dct_archive())
        legacy={} if full is None else {k:v for k,v in legacy_targets(full).items() if same(v[1].identity)}
        from pyecr.project.interpretation_archive import targets
        keys=set(targets(archive))|set(legacy);store=DctContextStore()
        store.history=[r for r in self.dct_context_store.history if r.target_sha256 in keys]
        ids={r.content_sha256 for r in store.history};store.selected={k:r for k,r in self.dct_context_store.selected.items() if r.content_sha256 in ids}
        if not keys:return None
        return dct_evidence(archive,capture_interpretations(store,archive,legacy_targets=legacy))

    def start_dct_export(self, destination, *, job_id, component=False, include_xlsx=False, figure_formats=('png','svg','pdf'),
                         legend_policy='follow', plot_views=None):
        """Explicitly export one selected captured run, independent of Analysis/Fit report gates."""
        if getattr(self,'_figure_export_task_id',None):raise RuntimeError('Wait for the active figure export')
        from pyecr.project.dct_archive import DctArchive
        from pyecr.report.dct_bundle import export_dct_bundle
        if self._dct_export_task_id:
            raise RuntimeError('DCT export already running')
        self.capture_dct_archive()
        service = self.component_dct_service if component else self.signed_dct_service
        job = service.get_job(job_id)
        if job.status in {'queued','running'}:
            raise RuntimeError('select a finished or historical DCT run before export')
        archive = DctArchive(component_jobs=(job,)) if component else DctArchive(whole_jobs=(job,))
        interpretations=self.capture_dct_interpretations(archive,scoped=True)
        from pyecr.report.export_options import ExportOptions
        from pyecr.plotting.view_state import freeze_views
        options = ExportOptions(plot_views=freeze_views(plot_views if plot_views is not None else self.analysis_session.project_metadata.get('plot_views')),
                                legend_policy=legend_policy)
        token = CancellationToken()
        self.dct_export_result = None
        task_id = self.runner.submit(GuiPage.DCT, lambda progress: export_dct_bundle(archive, destination,
            include_xlsx=include_xlsx, figure_formats=figure_formats,
            options=options,
            interpretations=interpretations,
            cancel_check=token.raise_if_cancelled, progress_callback=progress))
        self._dct_export_task_id = task_id
        self._cancellation_tokens[task_id] = token
        self.dct_export_status = f'Queued captured DCT export: {job.job_id} ({job.status})'
        return task_id

    def cancel_dct_export(self):
        if self._dct_export_task_id:
            self._cancellation_tokens[self._dct_export_task_id].cancel()
            self.dct_export_status = 'Cancellation requested; destination commits only after validation'

    def cancel_signed_dct(self, job_id: str) -> bool:
        return self.signed_dct_service.cancel(job_id)

    @property
    def current_component_dct(self):
        self.component_dct_service.bind_session(self.analysis_session)
        return self.component_dct_service.current

    def run_component_dct(self, config, component_configs, *, normalization_spec=None,
                          diagnostic_config=None):
        self._require_dct_idle()
        key=self.component_dct_service.create_job(self.analysis_session,config,component_configs,
            normalization_spec=normalization_spec,diagnostic_config=diagnostic_config)
        try:
            task_id=self.runner.submit(GuiPage.DCT,lambda progress:self.component_dct_service.run(key,progress_callback=progress))
        except Exception as exc:
            self.component_dct_service.fail_queued(key,exc);raise
        self._component_dct_tasks[task_id]=key
        self.state.current_errors.pop(GuiPage.DCT,None)
        return key

    def cancel_component_dct(self,key):
        return self.component_dct_service.cancel(key)

    def dct_unavailable_reason(self, *, use_classical_bi: bool = False) -> str:
        """One service/GUI rule for the existing model-independent DCT input."""
        config, dataset = self.current_analysis_config, self.analysis_session.dataset
        if dataset is None:
            return "DCT requires an imported dataset"
        if dataset.metadata.get("normalization_mode") == "signed_terminal":
            return ("Signed-DCT normalization is a separate terminal-referenced profile. "
                    "Use the explicit whole signed-DCT service; the legacy nonnegative DCT cannot use this profile. "
                    "Undo/Reset to the physical/source data and explicitly choose an appropriate progress representation for legacy analysis.")
        if dataset.metadata.get("signal_type") != "normalized":
            return "DCT requires a normalized processed signal; preprocess explicitly before running this diagnostic"
        if dataset.metadata.get("time_unit", "s") != "s":
            return "DCT requires processed time in seconds"
        return ""

    def run_dct(self, config: DctAnalysisConfig) -> str:
        """Launch DCT against the current processed dataset and geometry."""

        self._require_dct_idle()
        self.state.current_errors.pop(GuiPage.DCT, None)
        task = self.state.tasks[GuiPage.DCT]
        if task.status == "running":
            raise RuntimeError("DCT already has a running task")
        dataset = self.analysis_session.dataset
        analysis_config = self.analysis_session.current_analysis_config
        if dataset is None:
            raise RuntimeError("DCT requires an imported dataset")
        config.validate()
        reason = self.dct_unavailable_reason(use_classical_bi=config.use_classical_bi)
        if reason:
            raise ValueError(reason)
        token = CancellationToken()
        from pyecr.contracts import DctSourceSnapshot
        from pyecr.preprocessing.signed_normalization import cumulative_t0_s
        frozen_source = DctSourceSnapshot.capture_experiment(dataset, self.analysis_session.preprocessing.config, t0_s=cumulative_t0_s(dataset))
        frozen_geometry = None if analysis_config is None else analysis_config.model.geometry

        def execute(progress):  # type: ignore[no-untyped-def]
            return self.dct_service.analyze(
                dataset,
                frozen_geometry,
                config,
                cancel_token=token,
                progress_callback=progress,
            )

        task_id = self.runner.submit(GuiPage.DCT, execute)
        self._cancellation_tokens[task_id] = token
        self._dct_revisions[task_id] = self.analysis_session.current_revision
        self._dct_bindings[task_id] = (self.analysis_session, dataset, analysis_config)
        self._dct_input_sources[task_id] = frozen_source
        self.session.dct_config = config
        task.status = "running"
        task.progress = 0.0
        task.cancellation_requested = False
        task.message = "Queued DCT diagnostic"
        task.task_id = task_id
        return task_id

    def run_advanced_diagnostics(
        self, method: str, *, seed: int | None = None,
        acknowledge_exchangeable_residuals: bool = False,
        confidence_level: float = 0.95, options=None,
    ) -> str:
        """Explicitly launch bootstrap or profile likelihood in a worker."""

        normalized = str(method).strip().lower()
        if normalized not in {"bootstrap", "profile", "experimental_m1"}:
            raise ValueError("unknown advanced diagnostic method")
        from pyecr.services.advanced_intervals import checked_C
        confidence_level=checked_C(confidence_level)
        if normalized == "experimental_m1":
            if seed is None or not acknowledge_exchangeable_residuals:
                raise ValueError("Deep uncertainty requires a seed and explicit residual-assumption acknowledgement")
        self.state.current_errors.pop(GuiPage.DIAGNOSTICS, None)
        task_state = self.state.tasks[GuiPage.DIAGNOSTICS]
        if task_state.status == "running":
            raise RuntimeError("Diagnostics already has a running task")
        task_id = self._submit_diagnostics(
            normalized, seed=seed,
            acknowledge_exchangeable_residuals=acknowledge_exchangeable_residuals,
            confidence_level=confidence_level,options=options,
        )
        task_state.status = "running"
        task_state.progress = 0.0
        task_state.message = f"Queued {normalized}; frozen output C={100*confidence_level:.12g}% / gate 95%"
        task_state.task_id = task_id
        return task_id

    def recompute_saved_advanced(self,method,confidence_level):
        """Explicit saved quantiles/scan-threshold action, never a runner job."""
        from pyecr.services.advanced_intervals import recompute_saved
        if self.state.tasks[GuiPage.DIAGNOSTICS].status=='running':
            raise RuntimeError('Finish or explicitly Cancel running Diagnostics before recomputing saved evidence')
        old=self.analysis_session.advanced_diagnostics.get(method)
        if old is None:raise ValueError('Saved method/samples unavailable; explicitly Run the requested analysis')
        updated=recompute_saved(old,self.analysis_session,confidence_level)
        self.analysis_session.advanced_diagnostics[method]=updated
        self.state.advanced_diagnostics[method]=updated
        if not hasattr(self,'advanced_output_history'):self.advanced_output_history=[]
        self.advanced_output_history.append(dict(action='explicit_saved_evidence_recompute',method=method,presentation=updated,current_at_publication=True))
        return updated

    def save_experimental_m1_uncertainty(self, path: str | Path) -> None:
        presentation = self.analysis_session.advanced_diagnostics.get("experimental_m1")
        record = None if presentation is None else presentation.experimental_result
        if not isinstance(record, ExperimentalM1UncertaintyResult):
            raise RuntimeError("no experimental M1 uncertainty result for this fit")
        record.save(path)

    def reopen_experimental_m1_uncertainty(self, path: str | Path) -> ExperimentalM1UncertaintyResult:
        session = self.analysis_session
        if session.dataset is None or session.current_analysis_config is None or session.fit_result is None or session.fit_stale:
            raise RuntimeError("a current fit is required to reopen experimental uncertainty")
        record = ExperimentalM1UncertaintyResult.reopen(
            path, session.dataset, session.current_analysis_config, session.fit_result,
        )
        from pyecr.services.diagnostics_service import AdvancedDiagnosticsPresentation
        from pyecr.uncertainty import UncertaintyMethod
        presentation = AdvancedDiagnosticsPresentation(
            method=UncertaintyMethod.RESIDUAL_BOOTSTRAP,
            available=record.status == "complete", experimental_result=record,
            reason=None if record.status == "complete" else record.status,
            warnings=record.warnings,
        )
        session.advanced_diagnostics["experimental_m1"] = presentation
        self.state.advanced_diagnostics["experimental_m1"] = presentation
        return record

    def run_error_map(self, config: ErrorMapConfig | None = None) -> str:
        """Queue a current M1 FIXED Error Map without blocking the GUI thread."""

        self.state.current_errors.pop(GuiPage.DIAGNOSTICS, None)
        settings = ErrorMapConfig() if config is None else config
        if not isinstance(settings, ErrorMapConfig):
            raise TypeError("config must be ErrorMapConfig")
        task = self.state.tasks[GuiPage.DIAGNOSTICS]
        if task.status == "running":
            raise RuntimeError("Diagnostics already has a running task")
        context = self.workflow.prepare_error_map()
        self.error_map_service.validate_request(context, settings)
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            return self.error_map_service.compute(
                context,
                settings,
                cancel_token=token,
                progress_callback=progress,
            )

        task_id = self.runner.submit(GuiPage.DIAGNOSTICS, execute)
        self._cancellation_tokens[task_id] = token
        self._error_map_task_id = task_id
        task.status = "running"
        task.progress = 0.0
        task.message = "Queued 03 M2_SINGLE_DK Error Map"
        task.task_id = task_id
        task.cancellation_requested = False
        return task_id

    @property
    def current_error_map_result(self) -> ErrorMapResult | None:
        """Expose only a map still tied to the current dataset/model/fit."""

        session = self.analysis_session
        context = session.error_map_context
        revision = session.current_revision
        if (
            not self.workflow.can_error_map
            or session.error_map_stale
            or context is None
            or context.dataset is not session.dataset
            or context.analysis_config is not session.current_analysis_config
            or context.fit_result is not session.fit_result
            or context.revision != (
                revision.data_revision,
                revision.config_revision,
                revision.physical_input_revision,
            )
        ):
            return None
        return session.error_map_result

    @property
    def is_error_map_updating(self) -> bool:
        task = self.state.tasks[GuiPage.DIAGNOSTICS]
        return self._error_map_task_id is not None and task.task_id == self._error_map_task_id and task.status == "running"

    def run_dual_error_map(self, config: DualErrorMapConfig | None = None) -> str:
        """Queue both current M3 maps in the established diagnostics worker."""

        settings = self.dual_error_map_service.default_config() if config is None else config
        if not isinstance(settings, DualErrorMapConfig):
            raise TypeError("config must be DualErrorMapConfig")
        task = self.state.tasks[GuiPage.DIAGNOSTICS]
        if task.status == "running":
            raise RuntimeError("Diagnostics already has a running task")
        context = self.workflow.prepare_dual_error_map()
        self.dual_error_map_service.validate_request(context, settings)
        self.state.current_errors.pop(GuiPage.DIAGNOSTICS, None)
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            return self.dual_error_map_service.compute(
                context, settings, cancel_token=token, progress_callback=progress,
                phase_callback=None,
            )

        task_id = self.runner.submit(GuiPage.DIAGNOSTICS, execute)
        self._cancellation_tokens[task_id] = token
        self._dual_error_map_task_id = task_id
        task.status = "running"
        task.progress = 0.0
        task.message = "Queued 06 M5_DUAL_H2O_DK H/O Error Maps"
        task.task_id = task_id
        task.cancellation_requested = False
        self.analysis_session.dual_error_map_running = True
        self.analysis_session.dual_error_map_last_status = "running"
        self.analysis_session.dual_error_map_progress = DualMapProgress(
            0.0, 0.0, 0.0, "queued", task.message,
        )
        return task_id

    @property
    def current_dual_error_map_result(self) -> DualErrorMapResult | None:
        return self.workflow.current_dual_error_map_result

    @property
    def dual_error_map_status(self) -> str:
        return self.dual_error_map_service.status(self.analysis_session)

    @property
    def dual_error_map_progress(self) -> DualMapProgress | None:
        return self.analysis_session.dual_error_map_progress

    def cancel_dual_error_map(self) -> bool:
        task = self.state.tasks[GuiPage.DIAGNOSTICS]
        if task.task_id != self._dual_error_map_task_id:
            return False
        return self.cancel_page(GuiPage.DIAGNOSTICS)

    def _bind_live_fit(self, task_id):
        s = self.analysis_session
        self._fit_live_binding = (task_id, s, s.dataset, s.current_analysis_config,
                                  s.preprocessing.config, s.current_revision)

    @property
    def live_fit_is_current(self):
        binding = self._fit_live_binding
        if binding is None:
            return False
        task_id, owner, dataset, config, recipe, revision = binding
        if task_id == self._fit_published_job:
            return False
        s = self.analysis_session
        return (task_id == self.state.tasks[GuiPage.FIT].task_id and owner is s
                and dataset is s.dataset and config is s.current_analysis_config
                and s.preprocessing is not None and recipe is s.preprocessing.config
                and revision == s.current_revision)

    @property
    def current_fit_candidate(self):
        """Display-only endpoint; never enables Diagnostics, DCT or export."""
        if not self.live_fit_is_current:
            return None
        t = self.state.tasks[GuiPage.FIT]
        p = t.fit_progress
        if (p is None or p.job_id != t.task_id
                or p.revision != FitProgressRevision.capture(self.analysis_session.current_revision)):
            return None
        return p.best_candidate

    def _submit_fit(self, *, fit_control_binding=None) -> str:
        if fit_control_binding is not None:
            from pyecr.contracts import FitControlBinding,SourceIdentity
            if not isinstance(fit_control_binding,FitControlBinding):raise TypeError('Frozen FitControlBinding required')
            s=self.analysis_session
            fit_control_binding.assert_execution_source(SourceIdentity.capture(s.dataset,s.preprocessing.config,config=s.current_analysis_config))
        request = self.workflow.prepare_fit()
        if fit_control_binding is not None:
            # prepare_fit resolves the actual seed before any backend work.
            # Keep execution controls bound to that immutable request, rather
            # than the pre-allocation draft that may omit its default seed.
            fit_control_binding = FitControlBinding.capture(
                SourceIdentity.capture(request.dataset, self.analysis_session.preprocessing.config,
                                       config=request.config),
                fit_control_binding.controls,
            )
        task = self.state.tasks[GuiPage.FIT]
        task.fit_progress = None
        task.last_sequence = 0
        task.cancellation_requested = False
        token = CancellationToken()
        observer=None
        if fit_control_binding is not None and fit_control_binding.controls.termination.mode=='monitored':
            from pyecr.fitting.monitored import MonitoredFitObserver
            observer=MonitoredFitObserver(fit_control_binding.controls.termination,
                recipe=self.analysis_session.preprocessing.config,cancel_check=token.raise_if_cancelled)

        def execute(progress):  # type: ignore[no-untyped-def]
            try:
                outcome=self.workflow.execute_fit(request,cancel_token=token,progress_callback=progress,
                    **({} if observer is None else {'iteration_observer':observer}))
                if observer is not None:self.monitored_fit_records[id(outcome.fit_result)]=observer.snapshot()
                if fit_control_binding is not None:self._fit_control_records[id(outcome.fit_result)]=fit_control_binding.to_dict()
                return outcome
            finally:
                if observer is not None and observer.binding is not None:
                    self.monitored_runs[observer.job_id]=observer.snapshot()

        task_id = self.runner.submit(GuiPage.FIT, execute, fit_revision=request.revision)
        self._bind_live_fit(task_id)
        if observer is not None:self.monitored_task_jobs[task_id]=observer.job_id
        self._cancellation_tokens[task_id] = token
        return task_id

    def run_guided_refit(
        self, *, fixed_parameter: str, value_source: str,
        fixed_value: float | None = None,
    ) -> str:
        """Run an explicitly requested M1 constrained refit in a worker."""

        task = self.state.tasks[GuiPage.FIT]
        if task.status == "running":
            raise RuntimeError("Fit already has a running task")
        if self.state.tasks[GuiPage.DIAGNOSTICS].status == "running":
            raise RuntimeError("wait for Diagnostics before Guided Refit")
        request = self.workflow.prepare_guided_refit(
            fixed_parameter=fixed_parameter,
            value_source=value_source,
            fixed_value=fixed_value,
        )
        self.state.current_errors.pop(GuiPage.FIT, None)
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            return self.workflow.execute_guided_refit(
                request, cancel_token=token, progress_callback=progress,
            )

        task.fit_progress = None
        task.last_sequence = 0
        task.cancellation_requested = False
        task_id = self.runner.submit(GuiPage.FIT, execute, fit_revision=request.parent_revision)
        self._bind_live_fit(task_id)
        self._cancellation_tokens[task_id] = token
        self._guided_refit_task_id = task_id
        self.guided_refit_last_message = (
            f"Guided Refit queued: fix {fixed_parameter} from {value_source}"
        )
        task.status = "running"
        task.progress = 0.0
        task.message = f"Guided Refit queued: fix {fixed_parameter} from {value_source}"
        task.task_id = task_id
        return task_id

    def rollback_guided_refit(self) -> AnalysisSnapshot | None:
        """Restore the parent Fit and preserve the child as history."""

        if any(task.status == "running" for task in self.state.tasks.values()):
            raise RuntimeError("finish or cancel running tasks before rollback")
        child_snapshot = self.report_service.capture_snapshot(self.analysis_session)
        parent_snapshot = self.workflow.rollback_guided_refit(
            current_snapshot=child_snapshot
        )
        self.session.report_snapshot = parent_snapshot
        self.session.report_export_result = None
        self.cancel_boundary_plot()
        self.cancel_explorer_preview()
        self.boundary_plot_result = None
        self.explorer_plot_result = None
        self.state.fit_presentation = self.analysis_session.fit_presentation
        self.state.diagnostics_presentation = None
        self.state.advanced_diagnostics.clear()
        self.state.error_map_result = None
        self.state.latest_results[GuiPage.FIT] = self.analysis_session.fit_result
        self.state.latest_results.pop(GuiPage.DIAGNOSTICS, None)
        self.state.model_name = self.current_analysis_config.model_id.value  # type: ignore[union-attr]
        self.state.latest_results[GuiPage.MODEL] = self.current_analysis_config
        task = self.state.tasks[GuiPage.FIT]
        task.status = "completed"
        task.progress = 1.0
        task.message = "Guided Refit rolled back; parent Fit restored"
        self.guided_refit_last_message = task.message
        return parent_snapshot

    @property
    def guided_refit_running(self) -> bool:
        task = self.state.tasks[GuiPage.FIT]
        return bool(
            self._guided_refit_task_id is not None
            and task.task_id == self._guided_refit_task_id
            and task.status == "running"
        )

    def _submit_diagnostics(
        self, kind: str, *, seed: int | None = None,
        acknowledge_exchangeable_residuals: bool = False,
        confidence_level: float = 0.95, options=None,
    ) -> str:
        request = self.workflow.prepare_diagnostics()
        session = self.analysis_session
        maps_match_request = (
            request.dataset is session.dataset
            and request.config is session.current_analysis_config
            and request.fit_result is session.fit_result
            and request.revision == session.current_revision
        )
        error_map_result = (
            self.current_error_map_result if maps_match_request else None
        )
        dual_error_map_result = (
            self.current_dual_error_map_result if maps_match_request else None
        )
        token = CancellationToken()
        binding=None;generation=session.fit_generation
        if kind!='basic':
            from pyecr.contracts import SourceIdentity
            from pyecr.uncertainty import AdvancedUncertaintyConfig
            from pyecr.services.advanced_intervals import make_binding
            from pyecr.fitting.reproducibility import integer_seed
            settings=options or AdvancedUncertaintyConfig(enabled=True)
            if not isinstance(settings,AdvancedUncertaintyConfig):raise TypeError('AdvancedUncertaintyConfig required')
            seed=integer_seed(20260915 if seed is None else seed)
            source=SourceIdentity.capture(request.dataset,session.preprocessing.config,config=request.config,fit=request.fit_result)
            spec_options=(dict(adapter='experimental_m1',seed=seed,n_resamples=200,max_wall_seconds=90.,minimum_successful=180,
                assumption_acknowledged=acknowledge_exchangeable_residuals) if kind=='experimental_m1' else
                dict(adapter=kind,seed=seed if kind=='bootstrap' else None,settings=asdict(settings)))
            binding=make_binding(source,request.config,confidence_level,'profile_likelihood' if kind=='profile' else 'residual_bootstrap',spec_options)
            if not hasattr(self,'advanced_output_history'):self.advanced_output_history=[]

        def execute(progress):  # type: ignore[no-untyped-def]
            if kind == "basic":
                token.raise_if_cancelled()
                progress(0.1, "Collecting fit diagnostics")
                presentation = self.diagnostics_service.describe(
                    request.dataset,
                    request.config,
                    request.fit_result,
                    error_map_result=error_map_result,
                    dual_error_map_result=dual_error_map_result,
                )
                token.raise_if_cancelled()
                progress(1.0, "Credibility Gate complete")
            elif kind == "bootstrap":
                presentation = self.diagnostics_service.run_bootstrap(
                    request.dataset,
                    request.config,
                    request.fit_result,
                    cancel_token=token,
                    progress_callback=progress,
                    options=settings,random_seed=seed,confidence_level=confidence_level,
                )
            elif kind == "experimental_m1":
                assert seed is not None
                presentation = self.diagnostics_service.run_experimental_m1_uncertainty(
                    request.dataset, request.config, request.fit_result,
                    seed=seed,
                    acknowledge_exchangeable_residuals=acknowledge_exchangeable_residuals,
                    cancel_token=token, progress_callback=progress,
                    confidence_level=confidence_level,
                )
                token.raise_if_cancelled()
            else:
                presentation = self.diagnostics_service.run_profile(
                    request.dataset,
                    request.config,
                    request.fit_result,
                    cancel_token=token,
                    progress_callback=progress,
                    options=settings,confidence_level=confidence_level,
                )
            if binding is not None:
                from pyecr.services.advanced_intervals import bind_presentation
                presentation=bind_presentation(presentation,binding,generation)
            return WorkflowDiagnosticsOutcome(request, kind, presentation)

        task_id = self.runner.submit(GuiPage.DIAGNOSTICS, execute)
        self._cancellation_tokens[task_id] = token
        if binding is not None:
            if not hasattr(self,'advanced_request_bindings'):self.advanced_request_bindings={}
            self.advanced_request_bindings[task_id]=binding
        return task_id

    def _plot_preview_context(self, plot: FitPlotData) -> PlotPreviewContext:
        session = self.analysis_session
        config = session.current_analysis_config
        fit = session.fit_result
        if (
            config is None or fit is None or session.fit_stale
            or session.fit_revision != session.current_revision
        ):
            raise RuntimeError("plot preview requires a current, non-out-of-date Fit result")
        return PlotPreviewContext(plot, fit, config, session.current_revision)

    def run_boundary_plot(self, plot: FitPlotData) -> str:
        """Generate measured-history boundary presentation outside the GUI thread."""

        context = self._plot_preview_context(plot)
        self.state.current_errors.pop(GuiPage.FIT, None)
        boundary = context.analysis_config.boundary
        if boundary is None or boundary.mode != "measured_history":
            raise RuntimeError("boundary worker requires measured_history configuration")
        self.cancel_boundary_plot()
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            return self.plot_preview_service.boundary(
                context, cancel_token=token, progress_callback=progress
            )

        task_id = self.runner.submit(GuiPage.FIT, execute)
        self._cancellation_tokens[task_id] = token
        self._plot_preview_tasks[task_id] = "boundary"
        self._boundary_task_id = task_id
        return task_id

    def cancel_boundary_plot(self) -> bool:
        return self._cancel_plot_preview(self._boundary_task_id)

    def run_explorer_preview(
        self, plot: FitPlotData, parameter: str, multiplier: float
    ) -> str:
        """Queue the latest production-forward explorer request and supersede prior work."""

        context = self._plot_preview_context(plot)
        self.state.current_errors.pop(GuiPage.FIT, None)
        self.cancel_explorer_preview()
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            return self.plot_preview_service.explorer(
                context,
                parameter,
                multiplier,
                cancel_token=token,
                progress_callback=progress,
            )

        task_id = self.runner.submit(GuiPage.FIT, execute)
        self._cancellation_tokens[task_id] = token
        self._plot_preview_tasks[task_id] = "explorer"
        self._explorer_task_id = task_id
        return task_id

    def cancel_explorer_preview(self) -> bool:
        return self._cancel_plot_preview(self._explorer_task_id)

    def run_preprocessing_robustness(self, variations: tuple[RobustnessVariation, ...]) -> str:
        """Explicit assessment worker; current Fit and recipe stay untouched."""
        if self.state.tasks[GuiPage.DIAGNOSTICS].status == "running":
            raise RuntimeError("Finish Diagnostics before starting robustness assessment")
        if self.state.tasks[GuiPage.FIT].status == "running":
            raise RuntimeError("Finish Fit before starting robustness assessment")
        request = prepare_robustness(self.analysis_session, variations)
        token = CancellationToken()

        def execute(progress):  # type: ignore[no-untyped-def]
            return run_robustness(request, analysis_service=self.analysis_service,
                                  diagnostics_service=self.diagnostics_service,
                                  cancel_token=token, progress_callback=progress)

        task_id = self.runner.submit(GuiPage.DIAGNOSTICS, execute)
        self._cancellation_tokens[task_id] = token
        self._robustness_task_id = task_id
        self._robustness_request = request
        task = self.state.tasks[GuiPage.DIAGNOSTICS]
        task.task_id = task_id
        task.status = "running"
        task.progress = 0.0
        task.message = "Preprocessing robustness assessment queued"
        task.cancellation_requested = False
        return task_id

    def cancel_preprocessing_robustness(self) -> bool:
        if self._robustness_task_id is None:
            return False
        return self.cancel_page(GuiPage.DIAGNOSTICS)

    @property
    def robustness_current(self) -> bool:
        result = self.robustness_assessment
        session = self.analysis_session
        return bool(result is not None and session.fit_result is not None and not session.fit_stale
                    and result.parent_revision == session.current_revision
                    and result.parent_fit_generation == session.fit_generation
                    and result.parent_fit_id == fit_result_id(session.fit_result))

    def save_preprocessing_robustness(self, path: str | Path) -> Path:
        if self.robustness_assessment is None:
            raise RuntimeError("No completed robustness assessment to save")
        return self.robustness_assessment.save_json(path)

    def open_preprocessing_robustness(self, path: str | Path) -> RobustnessAssessment:
        if self._robustness_task_id is not None:
            raise RuntimeError("Wait for robustness worker to finish")
        result = RobustnessAssessment.load_json(path)
        self.robustness_assessment = result  # Read-only artifact; no project state changes.
        return result

    def _cancel_plot_preview(self, task_id: str | None) -> bool:
        if task_id is None:
            return False
        token = self._cancellation_tokens.get(task_id)
        return False if token is None else token.cancel()

    @property
    def explorer_preview_running(self) -> bool:
        return self._explorer_task_id is not None

    @property
    def boundary_plot_running(self) -> bool:
        return self._boundary_task_id is not None

    def cancel_page(self, page: GuiPage) -> bool:
        """Request cooperative cancellation for the page's active task."""

        normalized_page = GuiPage(page)
        if normalized_page is GuiPage.SIMULATION:
            return self.cancel_simulation()
        task = self.state.tasks[normalized_page]
        if task.status != "running" or task.task_id is None:
            return False
        token = self._cancellation_tokens.get(task.task_id)
        if token is None:
            return False
        requested = token.cancel()
        task.cancellation_requested = normalized_page in (GuiPage.FIT,GuiPage.BATCH,GuiPage.DCT) or task.task_id == self._robustness_task_id
        task.message = "Cancelling | waiting for worker to exit"
        return requested

    def poll_events(self) -> tuple[TaskEvent, ...]:
        """Apply queued worker events to presentation state on the main thread."""

        if self._closed:
            return ()
        self._sync_signed_dct_inputs()
        events = self.runner.drain_events()
        self.component_dct_service.bind_session(self.analysis_session)
        for event in events:
            if event.task_id == getattr(self,'_figure_export_task_id',None):
                self._apply_figure_export_event(event)
                continue
            if event.task_id == self._simulation_export_task_id:
                self._apply_simulation_export_event(event)
                continue
            if event.task_id in self._simulation_tasks:
                self._apply_simulation_event(event)
                continue
            if event.page is GuiPage.SIMULATION:
                # Duplicate/unknown late events have no frozen request binding.
                # Never feed them into the legacy generic page dispatcher.
                continue
            if event.task_id == self._dct_export_task_id:
                self.dct_export_status = f'{event.kind}: {event.message}'
                if event.kind in {'completed', 'error', 'cancelled'}:
                    if event.kind == 'completed':
                        self.dct_export_result = event.result
                    self._cancellation_tokens.pop(event.task_id, None)
                    self._dct_export_task_id = None
                continue
            if event.task_id in self._component_dct_tasks:
                key=self._component_dct_tasks[event.task_id]
                if event.kind=='cancelled':self.component_dct_service.cancel(key)
                elif event.kind=='error' and self.component_dct_service.get_job(key).status=='queued':
                    self.component_dct_service.fail_queued(key,RuntimeError(event.message))
                if event.kind in {'completed','cancelled','error'}:self._component_dct_tasks.pop(event.task_id,None)
                continue
            if event.task_id in self._signed_dct_tasks:
                key = self._signed_dct_tasks[event.task_id]
                if event.kind == 'cancelled':
                    self.signed_dct_service.cancel(key)
                elif event.kind == 'error' and self.signed_dct_service.get_job(key).status == 'queued':
                    self.signed_dct_service.fail_queued(key, RuntimeError(event.message))
                if event.kind in {'completed', 'cancelled', 'error'}:
                    self._signed_dct_tasks.pop(event.task_id, None)
                continue
            if event.task_id in self._plot_preview_tasks:
                self._apply_plot_preview_event(event)
                continue
            task_state = self.state.tasks[event.page]
            if task_state.task_id != event.task_id:
                if event.kind in {"completed", "cancelled", "error"}:
                    self._cancellation_tokens.pop(event.task_id, None)
                    self._report_contexts.pop(event.task_id, None)
                    self._batch_requests.pop(event.task_id, None)
                    self._dct_revisions.pop(event.task_id,None)
                    self._dct_bindings.pop(event.task_id,None)
                    self._dct_input_sources.pop(event.task_id,None)
                continue
            if event.kind=='completed' and event.task_id in (self._error_map_task_id,self._dual_error_map_task_id):
                token=self._cancellation_tokens.get(event.task_id)
                if token is not None and token.is_cancelled:
                    event=replace(event,kind='cancelled',result=None,message='Map cancelled before publication; previous complete grid retained')
            if event.page is GuiPage.BATCH and event.kind=='completed':
                request=self._batch_requests.get(event.task_id)
                current=bool(request is not None and request[0] is self.session
                    and request[2]==self.current_analysis_config
                    and len(request[1])==len(self.session.batch_inputs)
                    and all(a is b for a,b in zip(request[1],self.session.batch_inputs)))
                if task_state.cancellation_requested or not current:
                    task_state.status='cancelled' if task_state.cancellation_requested else 'warning'
                    task_state.message=('Batch cancelled; previous complete result retained' if task_state.cancellation_requested else
                        'Out of date — Batch inputs or template changed; result was not published')
                    self._cancellation_tokens.pop(event.task_id,None);self._batch_requests.pop(event.task_id,None)
                    continue
                self._batch_result_contexts[request[3]]=(event.result,request[1],request[2],event.task_id)
                self._remember_figure_batch_source(event.result,request[1],request[2],request[3])
            if event.fit_progress is not None:
                snapshot = event.fit_progress
                if snapshot.job_id != event.task_id or snapshot.sequence <= task_state.last_sequence:
                    continue
                task_state.last_sequence = snapshot.sequence
                if snapshot.revision != FitProgressRevision.capture(self.analysis_session.current_revision):
                    task_state.fit_progress = None
                    if event.kind == "progress":
                        if not task_state.cancellation_requested:
                            task_state.message = "Out of date | worker progress ignored"
                        continue
                else:
                    task_state.fit_progress = snapshot
                    task_state.progress_received_at = monotonic()
                if event.kind == "completed" and task_state.cancellation_requested:
                    event = replace(event, kind="cancelled", result=None,
                                    message="Cancelled before main-thread publication")
                    task_state.fit_progress = replace(snapshot, phase="Cancelled",
                                                     message=event.message)
            if event.kind == "progress":
                task_state.status = "running"
                task_state.progress = event.progress if event.progress is not None else 0.0
                if not task_state.cancellation_requested:
                    task_state.message = event.message or "Running"
                if event.task_id == self._dual_error_map_task_id:
                    fraction = task_state.progress
                    phase = ("queued" if fraction == 0.0 else
                             "complete" if fraction == 1.0 else
                             "H" if fraction <= 0.5 else "O")
                    self.analysis_session.dual_error_map_progress = DualMapProgress(
                        min(1.0, 2.0 * fraction),
                        max(0.0, 2.0 * fraction - 1.0),
                        fraction, phase, task_state.message,
                    )
            elif event.kind == "completed":
                self.state.current_errors.pop(event.page, None)
                if isinstance(event.result, RobustnessAssessment):
                    request = self._robustness_request
                    session = self.analysis_session
                    token = self._cancellation_tokens.get(event.task_id)
                    still_current = bool(request is not None and request.parent_revision == session.current_revision
                        and request.parent_fit_generation == session.fit_generation
                        and request.parent_fit is session.fit_result and request.parent_config is session.current_analysis_config
                        and request.parent_dataset is session.dataset and not session.fit_stale)
                    if token is not None and token.is_cancelled:
                        task_state.status = "cancelled"
                        task_state.message = "Robustness cancelled before publication; prior result retained"
                    elif not still_current:
                        task_state.status = "warning"
                        task_state.message = "Out of date — robustness parent changed; result was not published"
                    elif not any(item.status == "completed" for item in event.result.scenarios):
                        task_state.status = "warning"
                        reasons = "; ".join(item.reason for item in event.result.scenarios)
                        task_state.message = "All robustness scenarios failed; prior result retained: " + reasons[:250]
                    else:
                        self.robustness_assessment = event.result
                        failed = sum(item.status != "completed" for item in event.result.scenarios)
                        task_state.status = "warning" if failed else "completed"
                        task_state.message = f"Robustness assessment complete: {failed} invalid scenario(s)"
                elif isinstance(event.result, PlanningSnapshot):
                    state = self.session.pair_design_state
                    self.session.pair_design_state = replace(state, plans=state.plans + (event.result,))
                    self.state.latest_results[GuiPage.PAIR] = event.result
                    task_state.status = "completed" if event.result.recommended_candidate_ids else "warning"
                    task_state.message = ("Planning scan saved; suggested region available" if event.result.recommended_candidate_ids
                                          else "Planning scan saved; no suggested region under these assumptions")
                elif isinstance(event.result, DctPresentation):
                    expected_revision = self._dct_revisions.get(event.task_id)
                    binding = self._dct_bindings.get(event.task_id)
                    if task_state.cancellation_requested:
                        task_state.status = "cancelled"
                        task_state.message = "DCT cancelled before publication; previous complete result retained"
                    elif (expected_revision != self.analysis_session.current_revision or binding is None
                          or binding[0] is not self.analysis_session or binding[1] is not self.analysis_session.dataset
                          or binding[2] is not self.analysis_session.current_analysis_config):
                        task_state.status = "warning"
                        task_state.message = (
                            "Out of date — obsolete DCT completed for changed data/geometry; "
                            "result was not applied"
                        )
                    else:
                        self.session.dct_result = event.result
                        source = self._dct_input_sources.get(event.task_id)
                        if source is not None:
                            self._legacy_dct_sources.clear()  # legacy UI has one stored result, not an unbounded hidden array cache
                            self._legacy_dct_sources[id(event.result)] = (event.result, source)
                        self.session.dct_config = event.result.config
                        self.state.latest_results[GuiPage.DCT] = event.result
                        has_warnings = bool(
                            event.result.result.resolution_warnings
                            or event.result.suggestion.artifact_risk
                        )
                        task_state.status = (
                            "warning" if has_warnings else "completed"
                        )
                        task_state.message = (
                            "DCT complete with resolution/artifact warnings"
                            if has_warnings
                            else "DCT diagnostic complete"
                        )
                elif isinstance(event.result, BatchAnalysisResult):
                    self.session.batch_result = event.result
                    self.session.global_result = None
                    self.state.latest_results[GuiPage.BATCH] = event.result
                    failed = sum(
                        not record.fit_success for record in event.result.records
                    )
                    unreliable = sum(
                        record.fit_success and not record.reliable
                        for record in event.result.records
                    )
                    task_state.status = (
                        "warning"
                        if failed or unreliable or event.result.warnings
                        else "completed"
                    )
                    task_state.message = (
                        f"Batch complete: {failed} failed, "
                        f"{unreliable} unreliable"
                    )
                elif isinstance(event.result, GlobalAnalysisResult):
                    self.session.global_result = event.result
                    self.state.latest_results[GuiPage.BATCH] = event.result
                    reliable = event.result.reliable
                    task_state.status = "completed" if reliable else "warning"
                    task_state.message = (
                        "Global/g-ECR complete"
                        if reliable
                        else "Global/g-ECR complete with reliability warnings"
                    )
                elif isinstance(event.result, GuidedRefitOutcome):
                    request = event.result.request
                    session = self.analysis_session
                    parent_current = (
                        request.parent_revision == session.current_revision
                        and request.parent_fit_generation == session.fit_generation
                        and request.dataset is session.dataset
                        and request.parent_config is session.current_analysis_config
                        and request.parent_fit is session.fit_result
                        and not session.fit_stale
                    )
                    parent_snapshot = (
                        self.report_service.capture_snapshot(session)
                        if parent_current else None
                    )
                    if self.workflow.accept_guided_refit(
                        event.result, parent_snapshot=parent_snapshot,
                    ):
                        self._fit_published_job = event.task_id
                        # The parent snapshot belongs to fit_history; the active
                        # report slot must not point at the new child fit.
                        self.session.report_snapshot = None
                        self.session.report_export_result = None
                        self.cancel_boundary_plot()
                        self.cancel_explorer_preview()
                        self.boundary_plot_result = None
                        self.explorer_plot_result = None
                        self.state.fit_presentation = event.result.presentation
                        self.state.diagnostics_presentation = None
                        self.state.advanced_diagnostics.clear()
                        self.state.error_map_result = None
                        self.state.latest_results[GuiPage.FIT] = event.result.fit_result
                        self.state.latest_results[GuiPage.MODEL] = request.config
                        self.state.latest_results.pop(GuiPage.DIAGNOSTICS, None)
                        self.state.model_name = request.config.model_id.value
                        task_state.status = (
                            "completed" if event.result.presentation.reliable else "warning"
                        )
                        task_state.message = (
                            f"Guided Refit complete: fixed {request.fixed_parameter} "
                            f"at {request.fixed_value:.6g} ({request.value_source})"
                        )
                    else:
                        task_state.status = "warning"
                        task_state.message = (
                            "Out of date — Guided Refit parent changed; result was not applied"
                        )
                elif isinstance(event.result, WorkflowFitOutcome):
                    if self.workflow.accept_fit(event.result):
                        self._fit_published_job = event.task_id
                        self.cancel_boundary_plot()
                        self.cancel_explorer_preview()
                        self.boundary_plot_result = None
                        self.explorer_plot_result = None
                        presentation = event.result.presentation
                        self.state.fit_presentation = presentation
                        self.state.diagnostics_presentation = None
                        self.state.advanced_diagnostics.clear()
                        self.state.error_map_result = None
                        self.state.latest_results[event.page] = (
                            event.result.fit_result
                        )
                        task_state.status = (
                            "completed" if presentation.reliable else "warning"
                        )
                        task_state.message = (
                            "Completed — reliability checks passed"
                            if presentation.reliable
                            else "Completed with reliability warnings"
                        )
                    else:
                        task_state.status = "warning"
                        task_state.message = (
                            "Out of date — obsolete Fit completed for a changed dataset/config revision; "
                            "result was not applied"
                        )
                elif isinstance(event.result, WorkflowDiagnosticsOutcome):
                    if event.result.kind!='basic':
                        if not hasattr(self,'advanced_output_history'):self.advanced_output_history=[]
                        self.advanced_output_history.append(dict(action='explicit_new_run',job_id=event.task_id,method=event.result.kind,
                            presentation=event.result.presentation,current_at_publication=bool(event.result.request.revision==self.analysis_session.current_revision
                                and event.result.request.fit_result is self.analysis_session.fit_result and not self.analysis_session.fit_stale)))
                    if self.workflow.accept_diagnostics(event.result):
                        presentation = event.result.presentation
                        if event.result.kind == "basic":
                            self.state.diagnostics_presentation = presentation  # type: ignore[assignment]
                            reliable = bool(getattr(presentation, "reliable", False))
                            evaluation = getattr(presentation, "evaluation", None)
                            if evaluation is not None:
                                reliable = evaluation.status == "supported"
                            task_state.status = "completed" if reliable else "warning"
                            task_state.message = (
                                "Diagnostics complete — reliability Gate passed"
                                if reliable
                                else "Diagnostics complete with reliability warnings"
                            )
                            if evaluation is not None:
                                task_state.message = evaluation.summary
                        else:
                            self.state.advanced_diagnostics[event.result.kind] = presentation  # type: ignore[assignment]
                            available = bool(getattr(presentation, "available", False))
                            task_state.status = "completed" if available else "warning"
                            task_state.message = (
                                f"{event.result.kind.title()} complete"
                                if available
                                else f"{event.result.kind.title()} is N/A; see reason"
                            )
                        self.state.latest_results[event.page] = presentation
                    else:
                        task_state.status = "warning"
                        task_state.message = (
                            "Out of date — obsolete Diagnostics completed for a changed fit revision; "
                            "result was not applied"
                        )
                elif isinstance(event.result, ErrorMapComputation):
                    if self.workflow.accept_error_map(event.result):
                        result = event.result.result
                        self.state.error_map_result = result
                        self.state.latest_results[event.page] = result
                        warning_count = len(result.diagnostics.warnings)
                        task_state.status = "warning" if warning_count else "completed"
                        task_state.message = (
                            f"Error Map complete with {warning_count} warning(s): "
                            + result.diagnostics.warnings[0]
                            if warning_count else
                            "Error Map complete; fit-point SSE parity verified"
                        )
                    else:
                        task_state.status = "warning"
                        task_state.message = (
                            "Out of date — obsolete Error Map completed for changed data/model/fit; "
                            "result was not applied"
                        )
                elif isinstance(event.result, DualErrorMapComputation):
                    if self.workflow.accept_dual_error_map(event.result):
                        result = event.result.result
                        warning_count = len(result.warnings)
                        task_state.status = "warning" if warning_count else "completed"
                        task_state.message = (
                            f"06 M5_DUAL_H2O_DK Error Maps complete with {warning_count} warning(s): "
                            + result.warnings[0]
                            if warning_count else
                            "06 M5_DUAL_H2O_DK Error Maps complete; fitted-point SSE parity verified"
                        )
                        self.analysis_session.dual_error_map_last_status = task_state.status
                    else:
                        task_state.status = "warning"
                        task_state.message = (
                            "Out of date — obsolete 06 M5_DUAL_H2O_DK Error Maps completed for "
                            "changed data/model/fit; result was not applied"
                        )
                        self.analysis_session.dual_error_map_last_status = "out_of_date"
                elif isinstance(event.result, ReportExportResult):
                    context = self._report_contexts.get(event.task_id)
                    token = self._cancellation_tokens.get(event.task_id)
                    if (context is None or (not context[3] and not self._report_context_is_current(context))
                            or (token is not None and token.is_cancelled)):
                        task_state.status = "warning"
                        task_state.message = "Out of date — obsolete report completed after its source changed; result was not published as current"
                    else:
                        self.session.report_export_result = event.result
                        self.state.latest_results[GuiPage.REPORT] = event.result
                        task_state.status = "completed"
                        task_state.message = "Historical export committed and verified" if context[3] else "Export committed and verified"
                else:
                    task_state.status = "completed"
                    task_state.message = event.message or "Completed"
                    self.state.latest_results[event.page] = event.result
                task_state.progress = 1.0
                if task_state.status == "warning":
                    self.record_notice(
                        event.page,
                        "out_of_date" if task_state.message.startswith("Out of date") else "warning",
                        task_state.message,
                    )
            elif event.kind == "cancelled":
                task_state.status = "cancelled"
                task_state.message = event.message or "Cancelled"
                self.record_notice(event.page, "cancelled", task_state.message)
                if event.task_id == self._dual_error_map_task_id:
                    self.analysis_session.dual_error_map_last_status = "cancelled"
            elif event.page is GuiPage.FIT and event.exception_type in {'MonitoredStop','MonitoredBudgetExceeded'}:
                task_state.status='warning'
                task_state.message=event.message
                self.state.current_errors.pop(event.page,None)
                self.record_notice(event.page,'warning',event.message)
            else:
                task_state.status = "error"
                is_dual_map = event.task_id == self._dual_error_map_task_id
                task_state.message = (
                    f"06 M5_DUAL_H2O_DK Error Maps failed: {event.message or 'see developer traceback'}"
                    if is_dual_map else event.message or "Task failed"
                )
                self._record_error(
                    GuiError(
                        page=event.page,
                        exception_type=event.exception_type or "Exception",
                        message=task_state.message if is_dual_map else event.message,
                        traceback_text=event.traceback_text or event.message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                )
                if is_dual_map:
                    self.analysis_session.dual_error_map_last_status = "error"
            if event.kind in {"completed", "cancelled", "error"}:
                if event.task_id == self._error_map_task_id:
                    self._error_map_task_id = None
                if event.task_id == self._guided_refit_task_id:
                    self.guided_refit_last_message = task_state.message
                    self._guided_refit_task_id = None
                if event.task_id == self._robustness_task_id:
                    if event.kind == "cancelled":
                        task_state.message = "Robustness cancelled; prior result retained"
                    elif event.kind == "error":
                        task_state.message = "Robustness failed; prior result retained: " + task_state.message
                    self._robustness_task_id = None
                    self._robustness_request = None
                if event.task_id == self._dual_error_map_task_id:
                    self._dual_error_map_task_id = None
                    self.analysis_session.dual_error_map_running = False
                self._cancellation_tokens.pop(event.task_id, None)
                self._dct_revisions.pop(event.task_id, None)
                self._dct_bindings.pop(event.task_id, None)
                self._dct_input_sources.pop(event.task_id, None)
                self._report_contexts.pop(event.task_id, None)
                self._batch_requests.pop(event.task_id, None)
        return events

    def _apply_plot_preview_event(self, event: TaskEvent) -> None:
        """Apply only the newest preview for the still-current fit revision."""

        kind = self._plot_preview_tasks[event.task_id]
        current_id = (
            self._boundary_task_id if kind == "boundary" else self._explorer_task_id
        )
        is_latest = event.task_id == current_id
        if event.kind == "completed" and is_latest:
            self.state.current_errors.pop(GuiPage.FIT, None)
            result = event.result
            context = getattr(result, "context", None)
            current = (
                context is not None
                and context.revision == self.analysis_session.current_revision
                and not self.analysis_session.fit_stale
                and context.fit_result is self.analysis_session.fit_result
                and context.analysis_config is self.current_analysis_config
            )
            if current and isinstance(result, BoundaryPlotComputation):
                self.boundary_plot_result = result
            elif current and isinstance(result, ExplorerPlotComputation):
                self.explorer_plot_result = result
            else:
                self.record_notice(
                    GuiPage.FIT,
                    "out_of_date",
                    f"Out of date — {kind} preview result was not applied.",
                )
        elif event.kind == "error" and is_latest:
            self._record_error(
                GuiError(
                    page=GuiPage.FIT,
                    exception_type=event.exception_type or "Exception",
                    message=event.message or f"{kind} preview failed",
                    traceback_text=event.traceback_text or event.message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        if event.kind in {"completed", "cancelled", "error"}:
            self._plot_preview_tasks.pop(event.task_id, None)
            self._cancellation_tokens.pop(event.task_id, None)
            if kind == "boundary" and is_latest:
                self._boundary_task_id = None
            if kind == "explorer" and is_latest:
                self._explorer_task_id = None

    def record_notice(
        self, page: GuiPage, kind: str, message: str
    ) -> None:
        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in {"warning", "out_of_date", "cancelled"}:
            raise ValueError(f"unsupported notice kind: {kind!r}")
        self.state.notices.append(
            GuiNotice(
                page=GuiPage(page),
                kind=cast(Literal["warning", "out_of_date", "cancelled"], normalized_kind),
                message=str(message),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    def record_callback_exception(
        self,
        page: GuiPage,
        exception: Exception,
        traceback_text: str,
    ) -> None:
        """Retain a synchronous widget-callback error for display and copying."""

        normalized_page = GuiPage(page)
        task_state = self.state.tasks[normalized_page]
        task_state.status = "error"
        task_state.message = str(exception)
        self._record_error(
            GuiError(
                page=normalized_page,
                exception_type=type(exception).__name__,
                message=str(exception),
                traceback_text=traceback_text,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    def _record_error(self, error: GuiError) -> None:
        """Retain one plain current error plus its full developer traceback."""

        self.state.errors.append(error)
        self.state.current_errors[error.page] = error

    def shutdown(self, *, wait: bool = True) -> None:
        """Release worker resources."""

        self._closed = True
        for job in self.component_dct_service.history:
            self.component_dct_service.cancel(job.job_id)
        for job in self.signed_dct_service.history:
            self.signed_dct_service.cancel(job.job_id)
        for token in tuple(self._cancellation_tokens.values()):
            token.cancel()
        self.runner.shutdown(wait=wait)
        if wait:
            self._dct_revisions.clear();self._dct_bindings.clear();self._dct_input_sources.clear()
            self._signed_dct_tasks.clear();self._component_dct_tasks.clear()
