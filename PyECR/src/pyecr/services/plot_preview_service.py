"""Worker-safe plot-preview orchestration using production presentation functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.plotting.data import FitPlotData
from pyecr.plotting.forward_views import (
    BoundaryPlotData,
    boundary_plot_data,
    parameter_variant,
)
from pyecr.services.analysis import CancellationToken
from pyecr.services.workflow import WorkflowRevision


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class PlotPreviewContext:
    """Immutable fit/config/plot capture associated with one workflow revision."""

    plot: FitPlotData
    fit_result: FitResult
    analysis_config: AnalysisConfig
    revision: WorkflowRevision


@dataclass(frozen=True, slots=True)
class BoundaryPlotComputation:
    context: PlotPreviewContext
    result: BoundaryPlotData


@dataclass(frozen=True, slots=True)
class ExplorerPlotComputation:
    context: PlotPreviewContext
    parameter: str
    multiplier: float
    values: tuple[float, ...]


class PlotPreviewService:
    """Run existing production plot-data functions in a caller-owned worker."""

    def boundary(
        self,
        context: PlotPreviewContext,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> BoundaryPlotComputation:
        _check_cancel(cancel_token)
        if progress_callback is not None:
            progress_callback(0.05, "Preparing measured boundary history")
        result = boundary_plot_data(
            context.plot, context.fit_result, context.analysis_config
        )
        _check_cancel(cancel_token)
        if progress_callback is not None:
            progress_callback(1.0, "Measured boundary view ready")
        return BoundaryPlotComputation(context, result)

    def explorer(
        self,
        context: PlotPreviewContext,
        parameter: str,
        multiplier: float,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ExplorerPlotComputation:
        _check_cancel(cancel_token)
        if progress_callback is not None:
            progress_callback(0.1, "Evaluating production forward preview")
        values = parameter_variant(
            context.plot,
            context.fit_result,
            context.analysis_config,
            parameter,
            multiplier,
        )
        _check_cancel(cancel_token)
        if progress_callback is not None:
            progress_callback(1.0, "Parameter preview ready")
        return ExplorerPlotComputation(
            context=context,
            parameter=str(parameter),
            multiplier=float(multiplier),
            values=values,
        )


def _check_cancel(token: CancellationToken | None) -> None:
    if token is not None:
        token.raise_if_cancelled()
