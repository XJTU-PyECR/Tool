"""Presentation-only state for the PyECR desktop shell."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping
from pyecr.fitting.progress import FitProgressEvent

if TYPE_CHECKING:
    from pyecr.diagnostics.error_map import ErrorMapResult
    from pyecr.services.analysis import FitPresentation
    from pyecr.services.diagnostics_service import (
        AdvancedDiagnosticsPresentation,
        DiagnosticsPresentation,
    )

__all__ = [
    "GUI_PAGES",
    "GuiError",
    "GuiPage",
    "GuiSnapshot",
    "GuiState",
    "GuiNotice",
    "TaskViewState",
]


class GuiPage(str, Enum):
    """Stable identifiers for the first-generation desktop pages."""

    PROJECT = "Project"
    DATA = "Data"
    MODEL = "Model"
    FIT = "Fit"
    DIAGNOSTICS = "Diagnostics"
    BATCH = "Batch"
    PAIR = "Pair"
    DCT = "DCT"
    REPORT = "Report"
    SIMULATION = "Simulation"


GUI_PAGES: tuple[GuiPage, ...] = tuple(GuiPage)


@dataclass(frozen=True, slots=True)
class GuiSnapshot:
    """Immutable presentation context safe to pass to a worker thread."""

    project_path: str | None
    dataset_name: str | None
    model_name: str | None
    time_unit: str
    length_unit: str
    diffusion_unit: str
    surface_exchange_unit: str
    signal_unit: str | None


@dataclass(slots=True)
class TaskViewState:
    """Main-thread view state for one page's latest task."""

    status: Literal[
        "idle", "running", "completed", "warning", "cancelled", "error"
    ] = "idle"
    progress: float = 0.0
    message: str = "Idle"
    task_id: str | None = None
    fit_progress: FitProgressEvent | None = None
    last_sequence: int = 0
    cancellation_requested: bool = False
    progress_received_at: float = 0.0


@dataclass(frozen=True, slots=True)
class GuiError:
    """A full worker or callback failure retained for copying."""

    page: GuiPage
    exception_type: str
    message: str
    traceback_text: str
    timestamp: str

    def as_text(self) -> str:
        heading = (
            f"[{self.timestamp}] {self.page.value}: "
            f"{self.exception_type}: {self.message}"
        )
        return f"{heading}\n{self.traceback_text.rstrip()}"


@dataclass(frozen=True, slots=True)
class GuiNotice:
    """A non-crash warning or out-of-date event for user-facing history."""

    page: GuiPage
    kind: Literal["warning", "out_of_date", "cancelled"]
    message: str
    timestamp: str

    def as_text(self) -> str:
        return f"[{self.timestamp}] {self.page.value}: {self.kind}: {self.message}"


@dataclass(slots=True)
class GuiState:
    """Mutable UI state with no optimizer or scientific configuration objects."""

    project_path: str | None = None
    dataset_name: str | None = None
    model_name: str | None = None
    time_unit: str = "s"
    length_unit: str = "cm"
    diffusion_unit: str = "cm^2/s"
    surface_exchange_unit: str = "cm/s"
    signal_unit: str | None = None
    active_page: GuiPage = GuiPage.PROJECT
    tasks: dict[GuiPage, TaskViewState] = field(
        default_factory=lambda: {page: TaskViewState() for page in GUI_PAGES}
    )
    latest_results: dict[GuiPage, object] = field(default_factory=dict)
    errors: list[GuiError] = field(default_factory=list)
    current_errors: dict[GuiPage, GuiError] = field(default_factory=dict)
    notices: list[GuiNotice] = field(default_factory=list)
    fit_presentation: FitPresentation | None = None
    diagnostics_presentation: DiagnosticsPresentation | None = None
    advanced_diagnostics: dict[str, AdvancedDiagnosticsPresentation] = field(
        default_factory=dict
    )
    error_map_result: ErrorMapResult | None = None

    def snapshot(self) -> GuiSnapshot:
        """Freeze only values needed by a service request."""

        return GuiSnapshot(
            project_path=self.project_path,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            time_unit=self.time_unit,
            length_unit=self.length_unit,
            diffusion_unit=self.diffusion_unit,
            surface_exchange_unit=self.surface_exchange_unit,
            signal_unit=self.signal_unit,
        )

    def status_values(self, page: GuiPage) -> Mapping[str, str]:
        """Return the dataset/model/unit summary shown on every page."""

        normalized_page = GuiPage(page)
        values = {
            "page": normalized_page.value,
            "dataset": self.dataset_name or "Not selected",
            "model": self.model_name or "Not selected",
            "units": (
                f"time={self.time_unit}, length={self.length_unit}, "
                f"D={self.diffusion_unit}, k={self.surface_exchange_unit}"
            ),
            "signal_unit": self.signal_unit or "Not specified",
        }
        return MappingProxyType(values)

    def status_summary(self, page: GuiPage) -> str:
        """Build a compact, consistently formatted page summary."""

        values = self.status_values(page)
        return (
            f"Current dataset: {values['dataset']}  |  "
            f"Model: {values['model']}  |  Units: {values['units']}  |  "
            f"Signal: {values['signal_unit']}"
        )

    def copyable_error_text(self) -> str:
        """Return every retained error with its complete traceback."""

        if not self.errors:
            return "No errors."
        return "\n\n".join(error.as_text() for error in self.errors)

    def activity_history_text(self) -> str:
        """Human-facing history without developer tracebacks."""

        entries = [
            (notice.timestamp, notice.as_text()) for notice in self.notices
        ] + [
            (
                error.timestamp,
                f"[{error.timestamp}] {error.page.value}: error: "
                f"{error.exception_type}: {error.message}",
            )
            for error in self.errors
        ]
        if not entries:
            return "No warnings or errors."
        return "\n".join(text for _, text in sorted(entries, key=lambda item: item[0]))
