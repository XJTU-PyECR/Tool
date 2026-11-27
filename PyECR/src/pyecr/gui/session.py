"""Non-presentation objects owned by one desktop analysis session."""

from __future__ import annotations

from dataclasses import dataclass, field

from pyecr.config import AnalysisConfig
from pyecr.io.dataset import EcrDataset
from pyecr.io.importer import ImportPreview, ImportResult
from pyecr.preprocessing import DownstreamStaleEvent, PreprocessingPipeline
from pyecr.project import ProjectSnapshot
from pyecr.report import AnalysisSnapshot
from pyecr.services.batch_analysis import (
    BatchAnalysisResult,
    BatchDatasetInput,
    GlobalAnalysisResult,
)
from pyecr.services.dct_service import DctAnalysisConfig, DctPresentation
from pyecr.services.report_service import ReportExportResult
from pyecr.services.workflow import AnalysisSession
from pyecr.services.pair_design_state import PairDesignState

__all__ = ["GuiSession"]


@dataclass(slots=True)
class GuiSession:
    """Core objects kept separate from mutable widget presentation state."""

    import_preview: ImportPreview | None = None
    import_result: ImportResult | None = None
    project_snapshot: ProjectSnapshot | None = None
    batch_inputs: list[BatchDatasetInput] = field(default_factory=list)
    batch_result: BatchAnalysisResult | None = None
    global_result: GlobalAnalysisResult | None = None
    dct_config: DctAnalysisConfig | None = None
    dct_result: DctPresentation | None = None
    report_snapshot: AnalysisSnapshot | None = None
    report_export_result: ReportExportResult | None = None
    analysis_session: AnalysisSession = field(default_factory=AnalysisSession)
    pair_design_state: PairDesignState = field(default_factory=PairDesignState)

    @property
    def dataset(self) -> EcrDataset | None:
        return self.analysis_session.dataset

    @dataset.setter
    def dataset(self, value: EcrDataset | None) -> None:
        self.analysis_session.dataset = value

    @property
    def preprocessing(self) -> PreprocessingPipeline | None:
        return self.analysis_session.preprocessing

    @preprocessing.setter
    def preprocessing(self, value: PreprocessingPipeline | None) -> None:
        self.analysis_session.preprocessing = value

    @property
    def downstream_stale_events(self) -> list[DownstreamStaleEvent]:
        return self.analysis_session.downstream_stale_events

    @property
    def current_analysis_config(self) -> AnalysisConfig | None:
        return self.analysis_session.current_analysis_config

    @current_analysis_config.setter
    def current_analysis_config(self, value: AnalysisConfig | None) -> None:
        self.analysis_session.current_analysis_config = value
