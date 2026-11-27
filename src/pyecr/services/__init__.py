"""Application services mediating between clients and the scientific core."""

from pyecr.services.analysis import (
    AnalysisBackend,
    AnalysisBackendNotRegisteredError,
    AnalysisBackendRegistry,
    AnalysisService,
    CancellationToken,
    CoreFitOutput,
    DualHOAnalysisConfig,
    DualHOBackend,
    FitParameterPresentation,
    FitPresentation,
    ProgressReporter,
    SingleEcrBackend,
)
from pyecr.services.import_service import ImportService
from pyecr.services.simulation_service import SimulationService, simulation_service_support
from pyecr.services.batch_analysis import (
    ArrheniusPresentation,
    BatchAnalysisResult,
    BatchAnalysisService,
    BatchDatasetInput,
    BatchRecordPresentation,
    GlobalAnalysisResult,
    GlobalCurvePresentation,
)
from pyecr.services.diagnostics_service import (
    AdvancedDiagnosticsPresentation,
    DiagnosticsPresentation,
    DiagnosticsService,
)
from pyecr.services.error_map_service import (
    ErrorMapComputation,
    ErrorMapContext,
    ErrorMapService,
    ObjectiveParityError,
)
from pyecr.services.dual_error_map_service import (
    DualErrorMapComputation,
    DualErrorMapContext,
    DualErrorMapService,
    DualMapProgress,
)
from pyecr.services.dct_service import (
    DctAnalysisConfig,
    DctPresentation,
    DctService,
    DctSuggestionPresentation,
)
from pyecr.services.dct_sources import PreparedSignedDctInput, prepare_signed_experiment
from pyecr.services.signed_dct_service import (
    DctPhysicalContext, SignedDctJob, SignedDctRequest, WholeSignedDctService,
)
from pyecr.services.workflow import (
    AnalysisSession,
    DiagnosticsRequest,
    FitRequest,
    WorkflowController,
    WorkflowDiagnosticsOutcome,
    WorkflowFitOutcome,
    WorkflowRevision,
    WorkflowState,
)
from pyecr.services.plot_preview_service import (
    BoundaryPlotComputation,
    ExplorerPlotComputation,
    PlotPreviewContext,
    PlotPreviewService,
)
from pyecr.services.report_service import (
    ReportArtifactStatus,
    ReportExportResult,
    ReportService,
)
from pyecr.services.paired_ecr import (
    PairMember, PairCompatibility, PairedComparison, check_pair_compatibility, compare_paired_ecr,
)
from pyecr.services.experiment_design import ExperimentDesignRequest, PlanningSnapshot, scan_experiment_design
from pyecr.services.pair_design_state import PairDesignState

__all__ = [
    'SimulationService',
    'simulation_service_support',
    "PreparedSignedDctInput", "prepare_signed_experiment", "DctPhysicalContext",
    "SignedDctJob", "SignedDctRequest", "WholeSignedDctService",
    "PairMember", "PairCompatibility", "PairedComparison", "check_pair_compatibility", "compare_paired_ecr",
    "ExperimentDesignRequest", "PlanningSnapshot", "scan_experiment_design", "PairDesignState",
    "AnalysisBackend",
    "AnalysisBackendNotRegisteredError",
    "AnalysisBackendRegistry",
    "AnalysisService",
    "AnalysisSession",
    "ArrheniusPresentation",
    "BatchAnalysisResult",
    "BatchAnalysisService",
    "BatchDatasetInput",
    "BatchRecordPresentation",
    "CancellationToken",
    "CoreFitOutput",
    "DualHOAnalysisConfig",
    "DualHOBackend",
    "AdvancedDiagnosticsPresentation",
    "DiagnosticsPresentation",
    "DiagnosticsRequest",
    "DiagnosticsService",
    "DualErrorMapComputation",
    "DualErrorMapContext",
    "DualErrorMapService",
    "DualMapProgress",
    "ErrorMapComputation",
    "ErrorMapContext",
    "ErrorMapService",
    "DctAnalysisConfig",
    "DctPresentation",
    "DctService",
    "DctSuggestionPresentation",
    "FitParameterPresentation",
    "FitPresentation",
    "FitRequest",
    "ImportService",
    "GlobalAnalysisResult",
    "GlobalCurvePresentation",
    "ProgressReporter",
    "ObjectiveParityError",
    "BoundaryPlotComputation",
    "ExplorerPlotComputation",
    "PlotPreviewContext",
    "PlotPreviewService",
    "ReportArtifactStatus",
    "ReportExportResult",
    "ReportService",
    "SingleEcrBackend",
    "WorkflowController",
    "WorkflowDiagnosticsOutcome",
    "WorkflowFitOutcome",
    "WorkflowRevision",
    "WorkflowState",
]
