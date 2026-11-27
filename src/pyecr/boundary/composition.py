"""Composition of validated gas boundaries with existing M0/M1 models."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.boundary.flush import FlushDiagnostic, exponential_flush_correction
from pyecr.boundary.measured import MeasuredBoundary
from pyecr.config import BoundaryConfig
from pyecr.models.ids import ModelID
from pyecr.models.single import SingleEcrContext, SingleEcrModel, SingleEcrPrediction

__all__ = [
    "BoundaryComposedPrediction",
    "BoundaryComposedSingleEcrModel",
    "dominant_material_time_constant",
]

# Poisson/Jacobi early-time form of the same two-face M0 slab response.
# See docs/specs/t12_flush.md. At Fo<=1e-3 the omitted image correction
# is bounded by 8*sqrt(Fo/pi)*exp(-1/(4*Fo)) < 4e-110 per axis.
_M0_INITIAL_LAYER_VERSION = "m0-cistr-forcing-initial-layer/1"
_M0_INITIAL_LAYER_FO_MAX = 1.0e-3
_M0_INITIAL_LAYER_BOUND = 4.0e-110


@dataclass(frozen=True, slots=True)
class BoundaryComposedPrediction:
    """Boundary-driven signal plus material and boundary diagnostics."""

    y: NDArray[np.float64]
    normalized_response: NDArray[np.float64]
    ideal_response: NDArray[np.float64]
    diagnostics: Mapping[str, object]
    material_prediction: SingleEcrPrediction | None = None


@dataclass(frozen=True, slots=True)
class BoundaryComposedSingleEcrModel:
    """M0/M1 material model driven by a non-ideal gas boundary."""

    material_model: SingleEcrModel
    boundary: BoundaryConfig
    _measured_boundary: MeasuredBoundary | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.material_model, SingleEcrModel):
            raise TypeError("material_model must be a SingleEcrModel")
        if not isinstance(self.boundary, BoundaryConfig):
            raise TypeError("boundary must be a BoundaryConfig")
        self.boundary.validate()
        if self.boundary.mode == "ideal_step":
            raise ValueError("ideal_step does not require a boundary composition wrapper")
        measured: MeasuredBoundary | None = None
        if self.boundary.mode == "measured_history":
            measured = MeasuredBoundary(
                time_s=self.boundary.measured_time_s,
                values=self.boundary.measured_values,
                signal_type=self.boundary.measured_signal_type,  # type: ignore[arg-type]
                initial_value=self.boundary.measured_initial_value,  # type: ignore[arg-type]
                final_value=self.boundary.measured_final_value,  # type: ignore[arg-type]
                value_unit=self.boundary.measured_value_unit,
            )
        object.__setattr__(self, "_measured_boundary", measured)

    @property
    def model_id(self) -> ModelID:
        return self.material_model.model_id

    @property
    def kernel(self) -> object:
        return self.material_model.kernel

    @property
    def optimizer_parameter_names(self) -> tuple[str, ...]:
        return self.material_model.optimizer_parameter_names

    def validate_time_coverage(self, t_s: ArrayLike) -> None:
        """Fail before optimization if measured history would require extrapolation."""

        if self._measured_boundary is not None:
            self._measured_boundary.resample(t_s)

    def predict(
        self,
        t: ArrayLike,
        theta: Mapping[str, float],
        context: object,
    ) -> NDArray[np.float64]:
        return self.evaluate(t, theta, context).y

    def evaluate(
        self,
        t: ArrayLike,
        theta: Mapping[str, float],
        context: object,
    ) -> BoundaryComposedPrediction:
        if not isinstance(theta, Mapping):
            raise TypeError("theta must be a mapping of optimizer coordinates")
        if not isinstance(context, SingleEcrContext):
            raise TypeError("context must be a SingleEcrContext")
        amplitude = _signal_coordinate(theta, "amplitude", context.amplitude)
        baseline = _signal_coordinate(theta, "baseline", context.baseline)
        material_theta = {
            name: value
            for name, value in theta.items()
            if name not in {"amplitude", "baseline"}
        }
        normalized_context = SingleEcrContext(
            amplitude=1.0,
            baseline=0.0,
            term_tol=context.term_tol,
            max_terms=context.max_terms,
        )
        material_at_target = self.material_model.evaluate(
            t,
            material_theta,
            normalized_context,
        )

        # Bounded evidence only: do not retain the potentially large internal
        # grids, alter their tolerances, or substitute a new convolution.
        summary = dict(n_evaluations=0, n_points=0, min_time_s=None, max_time_s=None,
            all_axes_converged=True, unconverged_evaluations=0,
            asymptotic_roots_used=False, max_terms_used_by_axis=[0, 0, 0])
        initial_layer = dict(algorithm_version=_M0_INITIAL_LAYER_VERSION,
            evaluations_corrected=0, axis_points_corrected=0,
            max_abs_forcing_change=0.0, max_initial_layer_error_bound=0.0,
            max_roundoff_allowance=0.0, error_bound_basis="image_remainder; separate float64 allowance",
            fourier_number_max=_M0_INITIAL_LAYER_FO_MAX,
            modal_budget_increased=False, raw_kernel_evidence_preserved=True)

        def record_kernel(prediction: SingleEcrPrediction, times: ArrayLike) -> None:
            values = np.asarray(times)
            summary['n_evaluations'] += 1
            summary['n_points'] += int(values.size)
            low, high = float(values.min()), float(values.max())
            summary['min_time_s'] = low if summary['min_time_s'] is None else min(low, summary['min_time_s'])
            summary['max_time_s'] = high if summary['max_time_s'] is None else max(high, summary['max_time_s'])
            converged = True
            for i, axis in enumerate(prediction.kernel_result.axis_results):
                if axis is None:continue
                converged = converged and axis.diagnostics['converged']
                summary['max_terms_used_by_axis'][i] = max(summary['max_terms_used_by_axis'][i], axis.n_terms_used)
                summary['asymptotic_roots_used'] |= bool(axis.diagnostics.get('asymptotic_roots_used', False))
            summary['all_axes_converged'] &= converged
            summary['unconverged_evaluations'] += int(not converged)

        record_kernel(material_at_target, t)

        def ideal_response(lags: NDArray[np.float64]) -> NDArray[np.float64]:
            prediction = self.material_model.evaluate(
                lags,
                material_theta,
                normalized_context,
            )
            record_kernel(prediction, lags)
            if (self.boundary.mode == "exponential_flush"
                    and self.model_id is ModelID.M0_SINGLE_D):
                return _resolve_m0_initial_layer(prediction, lags, context.term_tol, initial_layer)
            return prediction.y

        if self.boundary.mode == "exponential_flush":
            tau1_s = dominant_material_time_constant(
                material_at_target.kernel_result.axis_results
            )
            result = exponential_flush_correction(
                t,
                ideal_response,
                self.boundary.flush_time_s,  # type: ignore[arg-type]
                tau1_s=tau1_s,
            )
            normalized = result.normalized_response
            ideal = result.ideal_response
            severity = {
                FlushDiagnostic.NO_CORRECTION_NEEDED: "no_correction_needed",
                FlushDiagnostic.CORRECTION_RECOMMENDED: "correction_recommended",
                FlushDiagnostic.SEVERE_LIMITATION: "severe_limitation",
            }[result.diagnostic]
            diagnostics: dict[str, object] = {
                "type": "exponential_flush",
                "severity": severity,
                "composition_applied": True,
                "ratio": result.ratio,
                "tau1_s": tau1_s,
                "flush_time_s": self.boundary.flush_time_s,
                "core": dict(result.diagnostics),
            }
            if self.model_id is ModelID.M0_SINGLE_D:
                diagnostics["core"]["forcing_initial_layer"] = initial_layer
        else:
            measured = self._measured_boundary
            if measured is None:  # pragma: no cover - protected by construction
                raise RuntimeError("measured boundary adapter is unavailable")
            result = measured.apply_to_step_response(
                t,
                ideal_response,
                resample_time_step_s=self.boundary.resample_time_step_s,
                max_internal_points=self.boundary.max_internal_points,
            )
            normalized = result.normalized_response
            ideal = result.ideal_response
            diagnostics = {
                "type": "measured_history",
                "severity": "not_classified",
                "severity_reason": (
                    "Measured histories have no tau_f/tau1 severity classification."
                ),
                "composition_applied": True,
                "signal_type": measured.signal_type.value,
                "linear_small_step_assumption": True,
                "extrapolation_used": False,
                "core": dict(result.diagnostics),
            }

        scaled = baseline + amplitude * np.asarray(normalized, dtype=np.float64)
        scaled = _readonly(scaled)
        diagnostics['material_kernel_evaluation_summary'] = MappingProxyType({
            **summary, 'max_terms_used_by_axis': tuple(summary['max_terms_used_by_axis']),
            'scope': 'requested_grid_and_all_production_boundary_callbacks',
        })
        return BoundaryComposedPrediction(
            y=scaled,
            normalized_response=_readonly(normalized),
            ideal_response=_readonly(ideal),
            diagnostics=MappingProxyType(diagnostics),
            material_prediction=material_at_target,
        )


def _resolve_m0_initial_layer(prediction, times, tolerance, evidence):
    """Bounded alternative only for unresolved M0 CISTR forcing samples.

    Raw material predictions/diagnostics stay untouched. No extra modes or
    observation rows are used. Other boundary types never enter this helper.
    """
    if (prediction.kernel_result.diagnostics["all_axes_converged"]
            or tolerance < _M0_INITIAL_LAYER_BOUND):
        return prediction.y
    times = np.asarray(times, dtype=np.float64)
    axes = prediction.kernel_result.axis_results
    fractions, replacements = [], np.zeros(times.shape, dtype=np.int64)
    for axis in axes:
        if axis is None:
            continue
        d = axis.diagnostics
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            fo = prediction.D_cm2_s * times / float(d["full_dim_cm"])**2
        use = ((times > 0) & (fo > 0) & (fo <= _M0_INITIAL_LAYER_FO_MAX)
               & (np.asarray(d["truncation_error_by_time"]) > tolerance))
        values = axis.F
        if np.any(use):
            values = values.copy()
            values[use] = 1.0 - 4.0*np.sqrt(fo[use]/np.pi)
            replacements += use
        fractions.append(values)
    if not np.any(replacements):
        return prediction.y
    corrected = 1.0 - np.prod(fractions, axis=0)
    evidence["evaluations_corrected"] += 1
    evidence["axis_points_corrected"] += int(replacements.sum())
    evidence["max_abs_forcing_change"] = max(evidence["max_abs_forcing_change"],
        float(np.max(np.abs(corrected-prediction.y))))
    evidence["max_initial_layer_error_bound"] = max(evidence["max_initial_layer_error_bound"],
        int(replacements.max())*_M0_INITIAL_LAYER_BOUND)
    evidence["max_roundoff_allowance"] = max(evidence["max_roundoff_allowance"],
        8.0*np.finfo(float).eps*len(fractions))
    return corrected


def dominant_material_time_constant(axis_results: tuple[object, ...]) -> float:
    """Return the whole-specimen slowest modal time from active-axis rates."""
    first_rates: list[float] = []
    for axis_result in axis_results:
        if axis_result is None:
            continue
        diagnostics = getattr(axis_result, "diagnostics", None)
        if not isinstance(diagnostics, Mapping):
            raise TypeError("kernel axis diagnostics are unavailable")
        rates = np.asarray(diagnostics.get("rates_s_inv"), dtype=np.float64).reshape(-1)
        if rates.size == 0 or not np.isfinite(rates[0]) or rates[0] <= 0.0:
            raise ValueError("kernel first modal rate must be positive and finite")
        first_rates.append(float(rates[0]))
    if not first_rates:
        raise RuntimeError("material kernel has no active modal rate")
    return 1.0 / sum(first_rates)


def _signal_coordinate(
    theta: Mapping[str, object],
    name: str,
    default: float,
) -> float:
    value = theta.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"theta[{name!r}] must be a real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"theta[{name!r}] must be finite")
    return numeric


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    result.flags.writeable = False
    return result
