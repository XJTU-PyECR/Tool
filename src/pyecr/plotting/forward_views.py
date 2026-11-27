"""Presentation arrays derived only from production forward models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10
from typing import Mapping

import numpy as np

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.factory import ModelFactory
from pyecr.models.single import SingleEcrContext, SingleEcrModel
from pyecr.models.surface_k import SurfaceKModel
from pyecr.boundary.composition import BoundaryComposedSingleEcrModel
from pyecr.boundary.measured import MeasuredBoundary
from pyecr.plotting.data import FitPlotData


@dataclass(frozen=True, slots=True)
class BoundaryPlotData:
    time_s: tuple[float, ...]
    ideal_response: tuple[float, ...]
    corrected_response: tuple[float, ...]
    input_time_s: tuple[float, ...]
    input_progress: tuple[float, ...]
    mode: str
    input_note: str


def _single_theta(params: Mapping[str, float], config: AnalysisConfig) -> dict[str, float]:
    theta = {"log10_D": log10(float(params["D_cm2_s"]))}
    if config.model_id.value == "M1_SINGLE_DK":
        theta["log10_k"] = log10(float(params["k_cm_s"]))
    for name in ("amplitude", "baseline"):
        if name in params:
            theta[name] = float(params[name])
    return theta


def _context(config: AnalysisConfig) -> SingleEcrContext:
    return SingleEcrContext(
        term_tol=config.solver.relative_tolerance,
        max_terms=config.solver.max_terms,
    )


def parameter_variant(
    plot: FitPlotData,
    fit: FitResult,
    config: AnalysisConfig,
    parameter: str,
    multiplier: float,
) -> tuple[float, ...]:
    """Evaluate a one-parameter variant; never refit or duplicate a kernel."""

    if not isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("multiplier must be positive and finite")
    if parameter not in fit.params or not parameter.startswith(("D_", "k_")):
        raise ValueError("choose a fitted D or k parameter")
    model = ModelFactory().create(config)
    times = np.asarray(plot.time_s, dtype=np.float64)
    params = {**config.model.fixed_parameters, **fit.params}
    params[parameter] = float(params[parameter]) * multiplier
    if isinstance(model, SurfaceKModel):
        # M7 uses physical k and its independent analytic surface kernel.
        # Observation parameters remain those of the captured fit/config.
        values = model.predict(times, params)
    elif isinstance(model, (SingleEcrModel, BoundaryComposedSingleEcrModel)):
        values = model.predict(times, _single_theta(params, config), _context(config))
    elif isinstance(model, DualHOModel):
        required = ("D_H_cm2_s", "D_O_cm2_s", "A_H", "A_O", "y0")
        if config.model_id.value == "M3_DUAL_HO_DK":
            required += ("k_H_cm_s", "k_O_cm_s")
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(f"production forward model requires: {', '.join(missing)}")
        values = model.evaluate(
            times,
            **{name: float(params[name]) for name in required},
            kernel_options={
                "term_tol": config.solver.relative_tolerance,
                "max_terms": config.solver.max_terms,
            },
        ).total
    else:
        raise ValueError(f"parameter explorer is unavailable for {config.model_id.value}")
    return tuple(float(value) for value in values)


def boundary_plot_data(
    plot: FitPlotData, fit: FitResult, config: AnalysisConfig
) -> BoundaryPlotData:
    """Compare core-returned ideal/corrected responses for non-ideal M0/M1."""

    model = ModelFactory().create(config)
    if not isinstance(model, BoundaryComposedSingleEcrModel):
        raise ValueError("non-ideal M0/M1 boundary required")
    prediction = model.evaluate(
        np.asarray(plot.time_s),
        _single_theta({**config.model.fixed_parameters, **fit.params}, config),
        _context(config),
    )
    boundary = config.boundary
    assert boundary is not None
    if boundary.mode == "measured_history":
        measured = MeasuredBoundary(
            time_s=boundary.measured_time_s,
            values=boundary.measured_values,
            signal_type=boundary.measured_signal_type,
            initial_value=boundary.measured_initial_value,
            final_value=boundary.measured_final_value,
            value_unit=boundary.measured_value_unit,
        )
        input_time = tuple(float(x) for x in measured.time_s)
        input_progress = tuple(float(x) for x in measured.normalized_progress)
        note = f"Measured {boundary.measured_signal_type} normalized input"
    else:
        input_time = ()
        input_progress = ()
        note = f"CISTR exponential input; flush time = {boundary.flush_time_s:.4e} s"
    return BoundaryPlotData(
        time_s=plot.time_s,
        ideal_response=tuple(float(x) for x in prediction.ideal_response),
        corrected_response=tuple(float(x) for x in prediction.normalized_response),
        input_time_s=input_time,
        input_progress=input_progress,
        mode=boundary.mode,
        input_note=note,
    )
