"""Staged fitting workflow for independent H/O relaxation models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite, log
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyecr.fitting.solver_strategy import least_squares

from pyecr.fitting.base import FitResult, ParameterSpace
from pyecr.fitting.coarse_to_fine import current_screening, screen_and_refine
from pyecr.fitting.dual_objective import DualHOObjectiveEvaluator, _context_for_level
from pyecr.fitting.experimental_jacobian import (
    EXPERIMENTAL_MODE, HybridFallback, dual_hybrid_jacobian, validate_mode,
)
from pyecr.models.dual_ho import DualHOModel, DualHOResult
from pyecr.models.ids import ModelID

__all__ = [
    "DualHOFitConfig",
    "DualHOFitResult",
    "FitCandidateSummary",
    "LabelConstraint",
    "M3TriggerEvidence",
    "fit_dual_ho",
]


class LabelConstraint(str, Enum):
    """Optional physical ordering used to prevent H/O label switching."""

    D_H_GE_D_O = "D_H>=D_O"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class M3TriggerEvidence:
    """Independent evidence that can justify escalating from M2a to M3."""

    bi_values: tuple[float, ...] = ()
    surface_sensitivity_score: float | None = None

    def __post_init__(self) -> None:
        values = tuple(_positive_finite_scalar("bi_values item", x) for x in self.bi_values)
        sensitivity = self.surface_sensitivity_score
        if sensitivity is not None:
            sensitivity = _nonnegative_finite_scalar(
                "surface_sensitivity_score", sensitivity
            )
        object.__setattr__(self, "bi_values", values)
        object.__setattr__(self, "surface_sensitivity_score", sensitivity)


@dataclass(frozen=True, slots=True)
class DualHOFitConfig:
    """Numerical policy for staged M2a and optional M3 fitting."""

    fit_baseline: bool = False
    fixed_baseline: float | None = None
    log10_D_bounds: tuple[float, float] = (-12.0, -2.0)
    log10_k_bounds: tuple[float, float] = (-10.0, 0.0)
    log10_amplitude_bounds: tuple[float, float] = (-14.0, 1.0)
    baseline_bounds: tuple[float, float] | None = None
    label_constraint: LabelConstraint | str = LabelConstraint.D_H_GE_D_O
    m2_n_starts: int = 8
    m3_n_starts: int = 20
    n_candidates: int = 5
    random_seed: int = 20260915
    max_nfev: int = 5_000
    loss: str = "linear"
    residual_autocorrelation_threshold: float = 0.30
    residual_relative_rmse_threshold: float = 1.0e-5
    bi_trigger_max: float = 30.0
    surface_sensitivity_trigger: float = 0.50
    correlation_warning_threshold: float = 0.95
    condition_warning_threshold: float = 1.0e8
    shape_grid_points: int = 2_048
    kernel_options: Mapping[str, object] = field(default_factory=dict)
    jacobian_mode: str = "legacy"
    initial_values: Mapping[str, float] = field(default_factory=dict)
    start_policy: str = "legacy"
    linear_projection: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fit_baseline, bool):
            raise TypeError("fit_baseline must be boolean")
        fixed_baseline = self.fixed_baseline
        if fixed_baseline is not None:
            fixed_baseline = _finite_scalar("fixed_baseline", fixed_baseline)
        if self.fit_baseline and fixed_baseline is not None:
            raise ValueError(
                "fixed_baseline cannot be supplied when fit_baseline=True"
            )

        d_bounds = _bounds("log10_D_bounds", self.log10_D_bounds)
        k_bounds = _bounds("log10_k_bounds", self.log10_k_bounds)
        amplitude_bounds = _bounds(
            "log10_amplitude_bounds", self.log10_amplitude_bounds
        )
        baseline_bounds = (
            None
            if self.baseline_bounds is None
            else _bounds("baseline_bounds", self.baseline_bounds)
        )
        label_constraint = _coerce_label_constraint(self.label_constraint)
        m2_n_starts = _positive_integer("m2_n_starts", self.m2_n_starts)
        m3_n_starts = _positive_integer("m3_n_starts", self.m3_n_starts)
        if m3_n_starts < 2:
            raise ValueError("m3_n_starts must be at least 2 for MultiStart")
        n_candidates = _positive_integer("n_candidates", self.n_candidates)
        random_seed = _nonnegative_integer("random_seed", self.random_seed)
        max_nfev = _positive_integer("max_nfev", self.max_nfev)
        shape_grid_points = _positive_integer(
            "shape_grid_points", self.shape_grid_points
        )
        if shape_grid_points < 64:
            raise ValueError("shape_grid_points must be at least 64")
        if self.loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError("loss is not supported by scipy.optimize.least_squares")
        residual_autocorrelation_threshold = _closed_unit_interval(
            "residual_autocorrelation_threshold",
            self.residual_autocorrelation_threshold,
        )
        residual_relative_rmse_threshold = _nonnegative_finite_scalar(
            "residual_relative_rmse_threshold",
            self.residual_relative_rmse_threshold,
        )
        bi_trigger_max = _positive_finite_scalar(
            "bi_trigger_max", self.bi_trigger_max
        )
        surface_sensitivity_trigger = _nonnegative_finite_scalar(
            "surface_sensitivity_trigger", self.surface_sensitivity_trigger
        )
        correlation_warning_threshold = _closed_unit_interval(
            "correlation_warning_threshold", self.correlation_warning_threshold
        )
        condition_warning_threshold = _positive_finite_scalar(
            "condition_warning_threshold", self.condition_warning_threshold
        )
        if not isinstance(self.kernel_options, Mapping):
            raise TypeError("kernel_options must be a mapping")
        validate_mode(self.jacobian_mode)
        if self.start_policy not in {"legacy", "informed_v2"}:
            raise ValueError("start_policy must be legacy or informed_v2")
        if not isinstance(self.linear_projection, bool):
            raise TypeError("linear_projection must be boolean")
        if not isinstance(self.initial_values, Mapping):
            raise TypeError("initial_values must be a mapping")
        initial = {name: _finite_scalar(f"initial_values[{name!r}]", value)
                   for name, value in self.initial_values.items()}
        unknown = set(initial) - {"D_H_cm2_s", "D_O_cm2_s", "k_H_cm_s", "k_O_cm_s", "A_H", "A_O", "y0"}
        if unknown:
            raise ValueError("unknown dual initial values: " + ", ".join(sorted(unknown)))
        object.__setattr__(self, "initial_values", MappingProxyType(initial))

        object.__setattr__(self, "fixed_baseline", fixed_baseline)
        object.__setattr__(self, "log10_D_bounds", d_bounds)
        object.__setattr__(self, "log10_k_bounds", k_bounds)
        object.__setattr__(self, "log10_amplitude_bounds", amplitude_bounds)
        object.__setattr__(self, "baseline_bounds", baseline_bounds)
        object.__setattr__(self, "label_constraint", label_constraint)
        object.__setattr__(self, "m2_n_starts", m2_n_starts)
        object.__setattr__(self, "m3_n_starts", m3_n_starts)
        object.__setattr__(self, "n_candidates", n_candidates)
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "max_nfev", max_nfev)
        object.__setattr__(self, "shape_grid_points", shape_grid_points)
        object.__setattr__(
            self,
            "residual_autocorrelation_threshold",
            residual_autocorrelation_threshold,
        )
        object.__setattr__(
            self,
            "residual_relative_rmse_threshold",
            residual_relative_rmse_threshold,
        )
        object.__setattr__(self, "bi_trigger_max", bi_trigger_max)
        object.__setattr__(
            self, "surface_sensitivity_trigger", surface_sensitivity_trigger
        )
        object.__setattr__(
            self, "correlation_warning_threshold", correlation_warning_threshold
        )
        object.__setattr__(
            self, "condition_warning_threshold", condition_warning_threshold
        )
        object.__setattr__(
            self, "kernel_options", MappingProxyType(dict(self.kernel_options))
        )


@dataclass(frozen=True, slots=True)
class FitCandidateSummary:
    """Serializable-sized summary of one valid MultiStart candidate."""

    model_id: ModelID
    start_index: int
    params: Mapping[str, float]
    SSE: float
    exit_status: int
    nfev: int
    success: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class DualHOFitResult:
    """Staged fit result with model-selection, shape, and H/O diagnostics."""

    selected_model_id: ModelID
    selected_fit: FitResult
    m2a_fit: FitResult
    m3_fit: FitResult | None
    m2a_candidates: tuple[FitCandidateSummary, ...]
    m3_candidates: tuple[FitCandidateSummary, ...]
    selected_forward: DualHOResult
    DO_over_DH: float
    extremum_time: float | None
    is_monotonic: bool
    component_H: NDArray[np.float64]
    component_O: NDArray[np.float64]
    m2a_parameter_correlation: NDArray[np.float64]
    m3_parameter_correlation: NDArray[np.float64] | None
    diagnostics: Mapping[str, object]
    warnings: tuple[str, ...]


@dataclass(slots=True)
class _Candidate:
    start_index: int
    params: dict[str, float]
    prediction: NDArray[np.float64]
    SSE: float
    status: int
    nfev: int
    success: bool


@dataclass(slots=True)
class _LevelFit:
    fit: FitResult
    candidates: tuple[FitCandidateSummary, ...]
    forward: DualHOResult
    correlation: NDArray[np.float64]
    warnings: tuple[str, ...]
    polish: Callable | None = None


class _Parameterization:
    def __init__(
        self,
        config: DualHOFitConfig,
        signal: NDArray[np.float64],
        *,
        is_m3: bool,
    ) -> None:
        self.config = config
        self.is_m3 = is_m3
        self.fixed_baseline = (
            float(signal[0])
            if config.fixed_baseline is None
            else config.fixed_baseline
        )

        lower: list[float] = []
        upper: list[float] = []
        if config.label_constraint is LabelConstraint.D_H_GE_D_O:
            lower.extend([config.log10_D_bounds[0], 0.0])
            upper.extend([config.log10_D_bounds[1], 1.0])
        else:
            lower.extend([config.log10_D_bounds[0], config.log10_D_bounds[0]])
            upper.extend([config.log10_D_bounds[1], config.log10_D_bounds[1]])

        lower.extend(
            [config.log10_amplitude_bounds[0], config.log10_amplitude_bounds[0]]
        )
        upper.extend(
            [config.log10_amplitude_bounds[1], config.log10_amplitude_bounds[1]]
        )
        if is_m3:
            lower.extend([config.log10_k_bounds[0], config.log10_k_bounds[0]])
            upper.extend([config.log10_k_bounds[1], config.log10_k_bounds[1]])
        if config.fit_baseline:
            baseline_bounds = config.baseline_bounds or _automatic_baseline_bounds(signal)
            lower.append(baseline_bounds[0])
            upper.append(baseline_bounds[1])

        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    @property
    def model_id(self) -> ModelID:
        return ModelID.M3_DUAL_HO_DK if self.is_m3 else ModelID.M2_DUAL_HO_D

    @property
    def physical_bounds(self) -> Mapping[str, tuple[float, float]]:
        """Report the same bounds used by the existing optimizer layout."""

        config = self.config
        d_bounds = tuple(10.0**value for value in config.log10_D_bounds)
        amplitude_bounds = tuple(10.0**value for value in config.log10_amplitude_bounds)
        bounds: dict[str, tuple[float, float]] = {
            "D_H_cm2_s": d_bounds,
            "D_O_cm2_s": d_bounds,
            "A_H": amplitude_bounds,
            "A_O": amplitude_bounds,
        }
        if self.is_m3:
            k_bounds = tuple(10.0**value for value in config.log10_k_bounds)
            bounds["k_H_cm_s"] = k_bounds
            bounds["k_O_cm_s"] = k_bounds
        bounds["y0"] = (
            (float(self.lower[-1]), float(self.upper[-1]))
            if config.fit_baseline else (self.fixed_baseline, self.fixed_baseline)
        )
        return MappingProxyType(bounds)

    def random_start(self, generator: np.random.Generator) -> NDArray[np.float64]:
        return generator.uniform(self.lower, self.upper)

    def encode(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Encode physical inputs in the same bounded chart used by decode."""
        params = self.decode((self.lower + self.upper) / 2)
        params.update(values)
        if self.config.label_constraint is LabelConstraint.D_H_GE_D_O:
            if "D_H_cm2_s" in values and "D_O_cm2_s" not in values:
                params["D_O_cm2_s"] = min(params["D_O_cm2_s"], params["D_H_cm2_s"])
            if "D_O_cm2_s" in values and "D_H_cm2_s" not in values:
                params["D_H_cm2_s"] = max(params["D_H_cm2_s"], params["D_O_cm2_s"])
        for name, (lo, hi) in self.physical_bounds.items():
            if name == "y0" and not self.config.fit_baseline:
                continue
            value = params[name]
            lo_slack, hi_slack = 8*abs(np.spacing(lo)), 8*abs(np.spacing(hi))
            if not np.isfinite(value) or value < lo-lo_slack or value > hi+hi_slack:
                raise ValueError(f"initial_values[{name!r}] lies outside parameter_bounds")
            params[name] = float(np.clip(value, lo, hi))
        dh, do = np.log10(params["D_H_cm2_s"]), np.log10(params["D_O_cm2_s"])
        if self.config.label_constraint is LabelConstraint.D_H_GE_D_O:
            if dh < do-64*np.finfo(float).eps:
                raise ValueError("initial diffusion values must satisfy D_H>=D_O")
            span = self.config.log10_D_bounds[1]-do
            q = [do, 0. if span == 0 else (dh-do)/span]
        else:
            q = [dh, do]
        q += [np.log10(params["A_H"]), np.log10(params["A_O"])]
        if self.is_m3:
            q += [np.log10(params["k_H_cm_s"]), np.log10(params["k_O_cm_s"])]
        if self.config.fit_baseline:
            q.append(params["y0"])
        return np.clip(np.asarray(q, dtype=float), self.lower, self.upper)

    def decode(self, q: NDArray[np.float64]) -> dict[str, float]:
        index = 0
        if self.config.label_constraint is LabelConstraint.D_H_GE_D_O:
            log_d_o = float(q[index])
            order_fraction = float(q[index + 1])
            log_d_h = log_d_o + order_fraction * (
                self.config.log10_D_bounds[1] - log_d_o
            )
        else:
            log_d_h = float(q[index])
            log_d_o = float(q[index + 1])
        index += 2

        params: dict[str, float] = {
            "D_H_cm2_s": 10.0**log_d_h,
            "D_O_cm2_s": 10.0**log_d_o,
            "A_H": 10.0 ** float(q[index]),
            "A_O": 10.0 ** float(q[index + 1]),
        }
        index += 2
        if self.is_m3:
            params["k_H_cm_s"] = 10.0 ** float(q[index])
            params["k_O_cm_s"] = 10.0 ** float(q[index + 1])
            index += 2
        if self.config.fit_baseline:
            params["y0"] = float(q[index])
        else:
            params["y0"] = self.fixed_baseline
        return params

    def reported_params(self, params: Mapping[str, float]) -> dict[str, float]:
        result = {
            "D_H_cm2_s": params["D_H_cm2_s"],
            "D_O_cm2_s": params["D_O_cm2_s"],
            "A_H": params["A_H"],
            "A_O": params["A_O"],
        }
        if self.is_m3:
            result["k_H_cm_s"] = params["k_H_cm_s"]
            result["k_O_cm_s"] = params["k_O_cm_s"]
        if self.config.fit_baseline:
            result["y0"] = params["y0"]
        return result


def fit_dual_ho(
    t_s: ArrayLike,
    y: ArrayLike,
    *,
    m2_model: DualHOModel,
    m3_model: DualHOModel | None = None,
    fit_config: DualHOFitConfig | None = None,
    run_m3: bool | None = None,
    m3_evidence: M3TriggerEvidence | None = None,
    cancel_check: Callable[[], None] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> DualHOFitResult:
    """Fit M2a first, then conditionally evaluate the higher-risk M3 model.

    ``run_m3=True`` is an explicit user request, ``False`` forces M2a only,
    and ``None`` enables evidence-based escalation.  Baseline is fixed to the
    first observation by default, so M2a has four rather than five variables.
    """

    from pyecr.fitting.progress import fit_counters_available
    fit_counters_available()

    times, signal = _validated_data(t_s, y)
    if not isinstance(m2_model, DualHOModel):
        raise TypeError("m2_model must be a DualHOModel")
    if m3_model is not None and not isinstance(m3_model, DualHOModel):
        raise TypeError("m3_model must be a DualHOModel or None")
    if run_m3 is not None and not isinstance(run_m3, bool):
        raise TypeError("run_m3 must be True, False, or None")
    config = DualHOFitConfig() if fit_config is None else fit_config
    if not isinstance(config, DualHOFitConfig):
        raise TypeError("fit_config must be a DualHOFitConfig")
    evidence = M3TriggerEvidence() if m3_evidence is None else m3_evidence
    if not isinstance(evidence, M3TriggerEvidence):
        raise TypeError("m3_evidence must be M3TriggerEvidence or None")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable or None")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")

    _check_cancel(cancel_check)
    _report_progress(progress_callback, 0.02, "Preparing M2a MultiStart")
    generator = np.random.default_rng(config.random_seed)
    m2_level = _fit_level(
        m2_model,
        times,
        signal,
        config,
        is_m3=False,
        n_starts=config.m2_n_starts,
        generator=generator,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        progress_range=(0.05, 0.62),
    )

    lag1 = _lag1_autocorrelation(m2_level.fit.residuals)
    signal_scale = _signal_scale(signal)
    relative_rmse = m2_level.fit.RMSE / signal_scale
    structured_residuals = bool(
        relative_rmse >= config.residual_relative_rmse_threshold
        and isfinite(lag1)
        and abs(lag1) >= config.residual_autocorrelation_threshold
    )

    trigger_reasons: list[str] = []
    decision: str
    if run_m3 is False:
        decision = "user_forced_m2a_only"
    else:
        if run_m3 is True:
            trigger_reasons.append("explicit_user_request")
        else:
            if structured_residuals:
                trigger_reasons.append("structured_m2_residuals")
            if any(value <= config.bi_trigger_max for value in evidence.bi_values):
                trigger_reasons.append("bi_evidence")
            if (
                evidence.surface_sensitivity_score is not None
                and evidence.surface_sensitivity_score
                >= config.surface_sensitivity_trigger
            ):
                trigger_reasons.append("surface_sensitivity_evidence")
        decision = "run_m3" if trigger_reasons else "retain_m2a"

    workflow_warnings = list(m2_level.warnings)
    m3_level: _LevelFit | None = None
    if trigger_reasons:
        if m3_model is None:
            decision = "m3_triggered_but_model_unavailable"
            workflow_warnings.append(
                "M3_TRIGGERED_BUT_MODEL_UNAVAILABLE: provide m3_model to run "
                "finite-surface fitting."
            )
        else:
            m3_level = _fit_level(
                m3_model,
                times,
                signal,
                config,
                is_m3=True,
                n_starts=config.m3_n_starts,
                generator=generator,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
                progress_range=(0.64, 0.90),
                warm_parameters=m2_level.candidates[0].params,
            )
            workflow_warnings.extend(m3_level.warnings)

    _check_cancel(cancel_check)
    _report_progress(progress_callback, 0.92, "Selecting and diagnosing fitted model")
    from pyecr.fitting.progress import fit_stage
    fit_stage("Diagnostics", "Selecting and diagnosing fitted model")
    selected_level = m2_level
    selected_model = m2_model
    selected_is_m3 = False
    selection_reason = "M2a retained because M3 was not run"
    if m3_level is not None:
        if _information_criterion_prefers_m3(m2_level.fit, m3_level.fit):
            selected_level = m3_level
            selected_model = m3_model  # type: ignore[assignment]
            selected_is_m3 = True
            selection_reason = "M3 selected by AICc/BIC after staged evaluation"
        else:
            selection_reason = "M2a retained after M3 comparison"

    if selected_level.polish is not None:
        polished=selected_level.polish(selection_reason)
        if polished is not None:
            selected_level=polished
            if selected_is_m3:m3_level=polished
            else:m2_level=polished
            workflow_warnings.extend(polished.warnings)
    selected_params = dict(selected_level.fit.params)
    if "y0" not in selected_params:
        selected_params["y0"] = (
            float(signal[0])
            if config.fixed_baseline is None
            else config.fixed_baseline
        )
    selected_forward = _evaluate_forward(
        selected_model,
        times,
        selected_params,
        is_m3=selected_is_m3,
        kernel_options=config.kernel_options,
    )

    extremum_time, extremum_times, is_monotonic, shape_diagnostics = (
        _analyse_fitted_curve_shape(
            selected_model,
            times,
            selected_params,
            is_m3=selected_is_m3,
            kernel_options=config.kernel_options,
            grid_points=config.shape_grid_points,
        )
    )

    label_candidates: tuple[Mapping[str, object], ...] = ()
    if config.label_constraint is LabelConstraint.NONE:
        workflow_warnings.append(
            "LABEL_SWITCHING_RISK: D_H/D_O ordering is unconstrained; inspect "
            "the as-fitted and H/O-swapped candidates."
        )
        label_candidates = _label_switch_candidates(
            selected_model,
            times,
            signal,
            selected_params,
            is_m3=selected_is_m3,
            kernel_options=config.kernel_options,
        )

    do_over_dh = (
        selected_level.fit.params["D_O_cm2_s"]
        / selected_level.fit.params["D_H_cm2_s"]
    )
    warnings = tuple(dict.fromkeys(workflow_warnings))
    diagnostics = MappingProxyType(
        {
            "workflow": "M2a_then_conditional_M3",
            "m2a_free_parameter_count": len(m2_level.fit.params),
            "m3_free_parameter_count": (
                None if m3_level is None else len(m3_level.fit.params)
            ),
            "m3_ran": m3_level is not None,
            "m3_decision": decision,
            "m3_trigger_reasons": tuple(trigger_reasons),
            "m2_residual_lag1_autocorrelation": lag1,
            "m2_relative_rmse": relative_rmse,
            "bi_evidence": evidence.bi_values,
            "surface_sensitivity_score": evidence.surface_sensitivity_score,
            "selection_reason": selection_reason,
            "label_constraint": config.label_constraint.value,
            "label_switch_candidates": label_candidates,
            "DO_over_DH": do_over_dh,
            "extremum_time": extremum_time,
            "extremum_times": extremum_times,
            "is_monotonic": is_monotonic,
            "shape_method": "fitted_curve_derivative_zero",
            **shape_diagnostics,
        }
    )

    from pyecr.fitting.reproducibility import record_randomness
    all_starts=list(m2_level.fit.diagnostics["fitting_randomness"]["initial_points"])
    if m3_level is not None:all_starts+=list(m3_level.fit.diagnostics["fitting_randomness"]["initial_points"])
    algorithm = ("dual-HO-PCG64-shared-stream-v1" if config.start_policy == "legacy"
                 else "dual-HO-PCG64-informed-v2")
    selected_fit=record_randomness(selected_level.fit,seed=config.random_seed,stochastic=True,algorithm=algorithm,starts=all_starts,shared_M2_M3_stream=True)
    if config.start_policy != "legacy" or config.linear_projection:
        selected_fit = replace(selected_fit, diagnostics={**selected_fit.diagnostics,
            "dual_optimization_stages": {
                "M2": m2_level.fit.diagnostics.get("dual_optimization", {}),
                "M3": {} if m3_level is None else m3_level.fit.diagnostics.get("dual_optimization", {})}})
    if selected_is_m3:m3_level=replace(m3_level,fit=selected_fit)
    else:m2_level=replace(m2_level,fit=selected_fit)
    result = DualHOFitResult(
        selected_model_id=selected_level.candidates[0].model_id,
        selected_fit=selected_fit,
        m2a_fit=m2_level.fit,
        m3_fit=None if m3_level is None else m3_level.fit,
        m2a_candidates=m2_level.candidates,
        m3_candidates=() if m3_level is None else m3_level.candidates,
        selected_forward=selected_forward,
        DO_over_DH=do_over_dh,
        extremum_time=extremum_time,
        is_monotonic=is_monotonic,
        component_H=selected_forward.component_H,
        component_O=selected_forward.component_O,
        m2a_parameter_correlation=m2_level.correlation,
        m3_parameter_correlation=(
            None if m3_level is None else m3_level.correlation
        ),
        diagnostics=diagnostics,
        warnings=warnings,
    )
    _check_cancel(cancel_check)
    _report_progress(progress_callback, 1.0, "Dual H/O fit complete")
    return result


def _fit_level(
    model: DualHOModel,
    times: NDArray[np.float64],
    signal: NDArray[np.float64],
    config: DualHOFitConfig,
    *,
    is_m3: bool,
    n_starts: int,
    generator: np.random.Generator,
    cancel_check: Callable[[], None] | None,
    progress_callback: Callable[[float, str], None] | None,
    progress_range: tuple[float, float],
    warm_parameters: Mapping[str, float] | None = None,
) -> _LevelFit:
    from pyecr.fitting.progress import fit_start, fit_candidate, fit_start_finished, fit_stage
    parameterization = _Parameterization(config, signal, is_m3=is_m3)
    evaluator = DualHOObjectiveEvaluator(_context_for_level(
        model, times, signal, parameterization=parameterization,
        fit_config=config, is_m3=is_m3,
    ))
    candidates: list[_Candidate] = []
    start_audit = []
    failures = 0
    experimental_starts: list[dict[str, object]] = []
    hybrid_jac = (
        dual_hybrid_jacobian(evaluator, parameterization, cancel_check)
        if is_m3 and config.jacobian_mode == EXPERIMENTAL_MODE else None
    )

    def residual_function(q: NDArray[np.float64]) -> NDArray[np.float64]:
        _check_cancel(cancel_check)
        params = parameterization.decode(q)
        return evaluator.evaluate_residual(params)

    from pyecr.fitting.dual_initialization import start_schedule
    from pyecr.fitting.solver_strategy import current_solver_options
    from pyecr.fitting.iteration_observation import projection_execution_supported, monitored_execution_active
    schedule, start_sources = start_schedule(parameterization, generator, n_starts, model, warm_parameters)
    projection = None
    if (config.linear_projection and config.loss == "linear" and hybrid_jac is None
            and current_screening() is None and current_solver_options().strategy == "trf"
            and projection_execution_supported()):
        from pyecr.fitting.dual_projection import ProjectedDualObjective
        projection = ProjectedDualObjective(evaluator, parameterization, cancel_check)
    # Full-observation, short-budget starts locate basins before spending a
    # full solve on capped endpoints. Never compose this implicit screening
    # with a Monitored execution or its declared stop budgets. A read-only
    # observer retains the same search and native calls.
    projection_screen = bool(projection is not None and config.start_policy == "informed_v2"
                             and not monitored_execution_active())
    projected_capped = []
    projected_refined = []

    refined = None
    screening_stage = None
    experimental_schedule = schedule
    if current_screening() is not None:
        if config.loss != "linear" or config.jacobian_mode != "legacy":
            raise ValueError("P60 requires linear loss and legacy numeric Jacobian")
        def subset_factory(indices):
            screening_evaluator = DualHOObjectiveEvaluator(replace(
                evaluator.context, processed_time_s=times[indices],
                processed_signal=signal[indices]))
            def screening_residual(q):
                _check_cancel(cancel_check)
                return screening_evaluator.evaluate_residual(parameterization.decode(q))
            return screening_residual
        def physical_parameters(q):
            params = parameterization.reported_params(parameterization.decode(q))
            scale = max(float(np.ptp(signal)), np.finfo(float).eps)
            return params, [float(v) / scale if name == "y0" else float(np.log10(v))
                            for name, v in params.items()]
        def screening_progress(fraction, message):
            _report_progress(progress_callback, progress_range[0] + fraction *
                             (progress_range[1] - progress_range[0]), message)
        refined, screening_stage = screen_and_refine(
            times=times, starts=experimental_schedule,
            lower=parameterization.lower, upper=parameterization.upper,
            subset_factory=subset_factory, full_residual=residual_function,
            full_solver=least_squares, max_nfev=config.max_nfev,
            model_id=parameterization.model_id.value, seed=config.random_seed,
            physical_parameters=physical_parameters, cancel_check=cancel_check,
            progress=screening_progress,
        )

    for start_index in range(n_starts):
        if refined is not None and start_index not in refined:
            continue
        _check_cancel(cancel_check)
        fit_start("M3" if is_m3 else "M2", start_index + 1, n_starts, config.loss)
        fraction = start_index / n_starts
        if refined is None:
            _report_progress(
                progress_callback,
                progress_range[0]
                + fraction * (progress_range[1] - progress_range[0]),
                f"{parameterization.model_id.value} MultiStart "
                f"{start_index + 1}/{n_starts}",
            )
        start = (parameterization.random_start(generator) if experimental_schedule is None
                 else experimental_schedule[start_index])
        audit = {"stage":"M3" if is_m3 else "M2", "start_index":start_index,
                 "coordinates":start.tolist()}
        if config.start_policy != "legacy":
            audit["source"] = start_sources[start_index]
        if projection is not None:
            audit.update(requested_full_coordinates=start.tolist(),
                         coordinates=start[projection.indices].tolist(),
                         coordinate_version=projection.solver_problem.parameterization.version)
        start_audit.append(audit)
        try:
            fallback_reason = None
            if refined is not None:
                optimizer_result = refined[start_index]
            elif projection is not None:
                optimizer_result = least_squares(projection.residual, start[projection.indices],
                    bounds=(projection.lower, projection.upper),loss="linear",
                    max_nfev=min(config.max_nfev,60) if projection_screen else config.max_nfev,
                    x_scale=1.,jac="3-point",diff_step=1e-5)
            elif hybrid_jac is not None:
                try:
                    optimizer_result = least_squares(
                        residual_function, start,
                        bounds=(parameterization.lower, parameterization.upper),
                        loss=config.loss, max_nfev=config.max_nfev,
                        jac=hybrid_jac,
                    )
                except HybridFallback as exc:
                    fallback_reason = str(exc)
                    optimizer_result = least_squares(
                        residual_function, start,
                        bounds=(parameterization.lower, parameterization.upper),
                        loss=config.loss, max_nfev=config.max_nfev,
                    )
            else:
                optimizer_result = least_squares(
                    residual_function,
                    start,
                    bounds=(parameterization.lower, parameterization.upper),
                    loss=config.loss,
                    max_nfev=config.max_nfev,
                )
            params = (parameterization.decode(optimizer_result.x) if projection is None
                      else projection.decode(optimizer_result.x))
            evaluated = evaluator.evaluate(params)
            prediction = evaluated.prediction
            sse = evaluated.raw_sse
            if projection_screen:
                audit.update(screening_nfev=int(optimizer_result.nfev),
                             screening_status=int(optimizer_result.status),screening_SSE=float(sse))
                if not optimizer_result.success:
                    projected_capped.append((float(sse),start_index,optimizer_result.x.copy()))
                    fit_start_finished()
                    continue
            if hybrid_jac is not None:
                experimental_starts.append({
                    "start_index": start_index,
                    "method": ("legacy_fallback" if fallback_reason else EXPERIMENTAL_MODE),
                    "fallback_reason": fallback_reason,
                    "nfev": int(optimizer_result.nfev),
                    "njev": int(optimizer_result.njev),
                    "raw_sse": float(sse),
                })
            if not isfinite(sse) or not np.all(np.isfinite(prediction)):
                failures += 1
                fit_start_finished()
                continue
            candidates.append(
                _Candidate(
                    start_index=start_index,
                    params=params,
                    prediction=prediction,
                    SSE=sse,
                    status=int(optimizer_result.status),
                    nfev=int(optimizer_result.nfev),
                    success=bool(optimizer_result.success),
                )
            )
            fit_candidate(sse, optimizer_result.cost,
                          params=parameterization.reported_params(params),
                          model_id=parameterization.model_id,
                          observation_count=len(signal), native_status=int(optimizer_result.status))
            fit_start_finished()
        except (FloatingPointError, RuntimeError, TypeError, ValueError):
            failures += 1
            fit_start_finished()

    if projection_screen and projected_capped:
        best_completed = min((c.SSE for c in candidates),default=float("inf"))
        promising = sorted(projected_capped,key=lambda row:(row[0],row[1]))[:max(3,config.n_candidates)]
        promising = [row for row in promising if row[0] <= best_completed*1.25+1e-12]
        for index, (_, start_index, endpoint) in enumerate(promising):
            _check_cancel(cancel_check)
            remaining = config.max_nfev-start_audit[start_index]["screening_nfev"]
            if remaining <= 0:
                continue
            start_audit[start_index]["refinement_budget"] = remaining
            fit_start("M3 refinement" if is_m3 else "M2 refinement",index+1,len(promising),config.loss)
            _report_progress(progress_callback,progress_range[1],
                             f"Full-observation refinement {index+1}/{len(promising)}")
            native = least_squares(projection.residual,endpoint,
                bounds=(projection.lower,projection.upper),loss="linear",max_nfev=remaining,
                x_scale=1.,jac="3-point",diff_step=1e-5)
            projected_refined.append(start_index)
            if native.success:
                params = projection.decode(native.x)
                evaluated = evaluator.evaluate(params)
                candidates.append(_Candidate(start_index,params,evaluated.prediction,evaluated.raw_sse,
                                             int(native.status),int(native.nfev),True))
                fit_candidate(evaluated.raw_sse,native.cost,
                    params=parameterization.reported_params(params),model_id=parameterization.model_id,
                    observation_count=len(signal),native_status=int(native.status))
            else:
                failures += 1
            fit_start_finished()

    _check_cancel(cancel_check)
    _report_progress(
        progress_callback,
        progress_range[1],
        f"{parameterization.model_id.value} MultiStart complete",
    )

    if not candidates:
        raise RuntimeError(
            f"{parameterization.model_id.value} produced no valid MultiStart candidates"
        )
    candidates.sort(key=lambda candidate: (candidate.SSE, candidate.start_index))
    best = candidates[0]
    def build_level(best):
        fit_stage("Diagnostics", f"Diagnosing {'M3' if is_m3 else 'M2'} best candidate")
        reported_params = parameterization.reported_params(best.params)
        param_space: dict[str, ParameterSpace] = {
            name: ("linear" if name == "y0" else "log")
            for name in reported_params
        }
        forward = _evaluate_forward(
            model,
            times,
            best.params,
            is_m3=is_m3,
            kernel_options=config.kernel_options,
        )
        jacobian = _physical_parameter_jacobian(
            model,
            times,
            best.params,
            tuple(reported_params),
            is_m3=is_m3,
            kernel_options=config.kernel_options,
        )
        correlation, condition_number = _parameter_correlation(jacobian)

        warnings: list[str] = []
        if failures:
            warnings.append(
                f"MULTISTART_INVALID_CANDIDATES: {failures} of {n_starts} starts failed."
            )
        public_candidates = tuple(
            FitCandidateSummary(
                model_id=parameterization.model_id,
                start_index=candidate.start_index,
                params=parameterization.reported_params(candidate.params),
                SSE=candidate.SSE,
                exit_status=candidate.status,
                nfev=candidate.nfev,
                success=candidate.success,
            )
            for candidate in candidates[: config.n_candidates]
        )
        if len(public_candidates) < config.n_candidates:
            warnings.append(
                "MULTISTART_INSUFFICIENT_VALID_CANDIDATES: fewer candidates than requested."
            )
        warnings.extend(_multistart_identifiability_warnings(public_candidates, signal))
        warnings.extend(
            _correlation_warnings(
                correlation,
                tuple(reported_params),
                condition_number,
                config,
                model_id=parameterization.model_id,
            )
        )

        fit = _build_fit_result(
            signal,
            best.prediction,
            reported_params,
            param_space,
            exit_status=best.status,
            nfev=best.nfev,
            jacobian=jacobian,
            warnings=warnings,
        )
        if screening_stage is not None:
            fit = replace(fit, diagnostics={**fit.diagnostics,
                          "experimental_coarse_to_fine": current_screening().to_dict()})
        if hybrid_jac is not None:
            fit = replace(fit, diagnostics={
                **fit.diagnostics,
                "experimental_jacobian": {
                    "mode": EXPERIMENTAL_MODE, "log10_step": 1.0e-5,
                    "starts": experimental_starts,
                },
            })
        from pyecr.fitting.reproducibility import record_randomness
        algorithm = ("dual-HO-PCG64-shared-stream-v1" if config.start_policy == "legacy"
                     else "dual-HO-PCG64-informed-v2")
        fit=record_randomness(fit,seed=config.random_seed,stochastic=True,algorithm=algorithm,starts=start_audit)
        if config.start_policy != "legacy" or config.linear_projection:
            fit = replace(fit, diagnostics={**fit.diagnostics, "dual_optimization": dict(
                start_policy=config.start_policy,
                observation_solver="bounded_linear_projection" if projection is not None else "joint",
                objective="full_observation_unweighted_SSE" if config.loss == "linear" else config.loss,
                starts=n_starts, sources=start_sources,
                projection=None if projection is None else projection.diagnostics())})
            if projection_screen:
                fit = replace(fit, diagnostics={**fit.diagnostics, "dual_optimization": {
                    **fit.diagnostics["dual_optimization"],
                    "initial_screening": dict(max_nfev=min(config.max_nfev,60),full_observations=True,
                        competitive_SSE_factor=1.25,
                        capped_starts=len(projected_capped),refined_start_indices=projected_refined,
                        publication="Only successful full-observation endpoints; capped starts are provenance")}})
            if config.linear_projection and not projection_execution_supported():
                fit = replace(fit, diagnostics={**fit.diagnostics, "dual_optimization": {
                    **fit.diagnostics["dual_optimization"],
                    "projection_reason": "Joint coordinates retain the declared Monitored policy when native ftol/xtol are disabled"}})
        return _LevelFit(
            fit=fit,
            candidates=public_candidates,
            forward=forward,
            correlation=correlation,
            warnings=tuple(warnings),
            polish=do_polish,
        )

    def do_polish(selection_reason):
        from .production_stop import polish_ensemble
        output=polish_ensemble('M3' if is_m3 else 'M2',selection_reason+'; minimum actual F within chosen level')
        if output is None:return None
        native,index=output
        params = parameterization.decode(native.x) if projection is None else projection.decode(native.x)
        evaluated=evaluator.evaluate(params)
        polished=_Candidate(index,params,evaluated.prediction,evaluated.raw_sse,int(native.status),int(native.nfev),bool(native.success))
        return build_level(polished)
    return build_level(best)


def _evaluate_forward(
    model: DualHOModel,
    times: NDArray[np.float64],
    params: Mapping[str, float],
    *,
    is_m3: bool,
    kernel_options: Mapping[str, object],
) -> DualHOResult:
    from pyecr.fitting.progress import count_fit_call
    count_fit_call("forward")
    return model.evaluate(
        times,
        D_H_cm2_s=params["D_H_cm2_s"],
        D_O_cm2_s=params["D_O_cm2_s"],
        A_H=params["A_H"],
        A_O=params["A_O"],
        y0=params["y0"],
        k_H_cm_s=params.get("k_H_cm_s") if is_m3 else None,
        k_O_cm_s=params.get("k_O_cm_s") if is_m3 else None,
        kernel_options=kernel_options,
    )


def _check_cancel(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _report_progress(
    progress_callback: Callable[[float, str], None] | None,
    fraction: float,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(float(fraction), message)


def _physical_parameter_jacobian(
    model: DualHOModel,
    times: NDArray[np.float64],
    params: Mapping[str, float],
    parameter_names: tuple[str, ...],
    *,
    is_m3: bool,
    kernel_options: Mapping[str, object],
) -> NDArray[np.float64]:
    jacobian = np.empty((times.size, len(parameter_names)), dtype=np.float64)
    for column, name in enumerate(parameter_names):
        plus = dict(params)
        minus = dict(params)
        if name == "y0":
            step = max(1.0, abs(params[name])) * 1.0e-6
            plus[name] += step
            minus[name] -= step
            denominator = 2.0 * step
        else:
            log_step = 1.0e-5
            plus[name] *= 10.0**log_step
            minus[name] /= 10.0**log_step
            denominator = 2.0 * log_step
        plus_values = _evaluate_forward(
            model,
            times,
            plus,
            is_m3=is_m3,
            kernel_options=kernel_options,
        ).total
        minus_values = _evaluate_forward(
            model,
            times,
            minus,
            is_m3=is_m3,
            kernel_options=kernel_options,
        ).total
        jacobian[:, column] = (plus_values - minus_values) / denominator
    return jacobian


def _parameter_correlation(
    jacobian: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] == 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    covariance_shape = np.linalg.pinv(jacobian.T @ jacobian)
    scales = np.sqrt(np.maximum(np.diag(covariance_shape), 0.0))
    denominator = np.outer(scales, scales)
    correlation = np.divide(
        covariance_shape,
        denominator,
        out=np.full_like(covariance_shape, np.nan),
        where=denominator > 0.0,
    )
    correlation.flags.writeable = False
    return correlation, condition_number


def _correlation_warnings(
    correlation: NDArray[np.float64],
    parameter_names: tuple[str, ...],
    condition_number: float,
    config: DualHOFitConfig,
    *,
    model_id: ModelID,
) -> list[str]:
    warnings: list[str] = []
    if condition_number >= config.condition_warning_threshold:
        warnings.append(
            f"{model_id.value}_JACOBIAN_ILL_CONDITIONED: condition number "
            f"{condition_number:.3g}."
        )
    for row in range(len(parameter_names)):
        for column in range(row + 1, len(parameter_names)):
            value = correlation[row, column]
            if isfinite(value) and abs(value) >= config.correlation_warning_threshold:
                warnings.append(
                    f"{model_id.value}_PARAMETER_CORRELATION: "
                    f"{parameter_names[row]} vs {parameter_names[column]} = {value:.3f}."
                )
    return warnings


def _multistart_identifiability_warnings(
    candidates: tuple[FitCandidateSummary, ...],
    signal: NDArray[np.float64],
) -> list[str]:
    if len(candidates) < 2:
        return []
    best_sse = candidates[0].SSE
    machine_scale = np.finfo(np.float64).eps * signal.size * _signal_scale(signal) ** 2
    near = [
        candidate
        for candidate in candidates
        if (candidate.SSE - best_sse) / max(best_sse, machine_scale) < 0.01
    ]
    warnings: list[str] = []
    for name in candidates[0].params:
        if not (name.startswith("D_") or name.startswith("k_")):
            continue
        logs = [np.log10(candidate.params[name]) for candidate in near]
        if logs and max(logs) - min(logs) > 1.0:
            warnings.append(
                f"IDENTIFIABILITY_MULTISTART: {name} spans >1 decade while "
                "SSE differs by <1%."
            )
    return warnings


def _build_fit_result(
    observed: NDArray[np.float64],
    fitted: NDArray[np.float64],
    params: Mapping[str, float],
    param_space: Mapping[str, ParameterSpace],
    *,
    exit_status: int,
    nfev: int,
    jacobian: NDArray[np.float64],
    warnings: list[str],
) -> FitResult:
    residuals = observed - fitted
    sse = float(np.dot(residuals, residuals))
    n_observations = observed.size
    n_parameters = len(params)
    rmse = float(np.sqrt(sse / n_observations))
    centered = observed - np.mean(observed)
    sst = float(np.dot(centered, centered))
    if sst == 0.0:
        r2 = float("nan")
        warnings.append("R2_UNDEFINED: observed signal has zero variance.")
    else:
        r2 = 1.0 - sse / sst
    log_floor = np.finfo(np.float64).tiny * max(1.0, _signal_scale(observed) ** 2)
    mean_sse = max(sse / n_observations, log_floor)
    aic = n_observations * log(mean_sse) + 2.0 * n_parameters
    bic = n_observations * log(mean_sse) + n_parameters * log(n_observations)
    if n_observations <= n_parameters + 1:
        aicc = float("inf")
        warnings.append("AICC_UNDEFINED: n_observations <= n_parameters + 1.")
    else:
        aicc = aic + (
            2.0 * n_parameters * (n_parameters + 1)
            / (n_observations - n_parameters - 1)
        )
    return FitResult(
        params=params,
        param_space=param_space,
        y_fit=fitted,
        residuals=residuals,
        SSE=sse,
        RMSE=rmse,
        R2=r2,
        AIC=aic,
        AICc=aicc,
        BIC=bic,
        exit_status=exit_status,
        nfev=nfev,
        jacobian=jacobian,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _analyse_fitted_curve_shape(
    model: DualHOModel,
    times: NDArray[np.float64],
    params: Mapping[str, float],
    *,
    is_m3: bool,
    kernel_options: Mapping[str, object],
    grid_points: int,
) -> tuple[float | None, tuple[float, ...], bool, Mapping[str, object]]:
    dense_times = np.linspace(times[0], times[-1], grid_points, dtype=np.float64)
    dense_curve = _evaluate_forward(
        model,
        dense_times,
        params,
        is_m3=is_m3,
        kernel_options=kernel_options,
    ).total
    derivative = np.gradient(dense_curve, dense_times, edge_order=2)
    derivative_scale = float(np.max(np.abs(derivative)))
    time_step = float(dense_times[1] - dense_times[0])
    curve_scale = max(1.0, float(np.max(np.abs(dense_curve))))
    derivative_tolerance = max(
        derivative_scale * 1.0e-8,
        np.finfo(np.float64).eps * curve_scale / time_step * 100.0,
    )
    nonzero = np.flatnonzero(np.abs(derivative) > derivative_tolerance)
    roots: list[float] = []
    for left, right in zip(nonzero[:-1], nonzero[1:], strict=True):
        if np.signbit(derivative[left]) == np.signbit(derivative[right]):
            continue
        denominator = derivative[right] - derivative[left]
        root = dense_times[left] - derivative[left] * (
            dense_times[right] - dense_times[left]
        ) / denominator
        if not roots or abs(root - roots[-1]) > time_step:
            roots.append(float(root))

    root_tuple = tuple(roots)
    extremum_time = None if not root_tuple else root_tuple[0]
    diagnostics = MappingProxyType(
        {
            "shape_grid_points": grid_points,
            "shape_derivative_tolerance": derivative_tolerance,
            "shape_derivative_min": float(np.min(derivative)),
            "shape_derivative_max": float(np.max(derivative)),
            "extremum_count": len(root_tuple),
        }
    )
    return extremum_time, root_tuple, len(root_tuple) == 0, diagnostics


def _label_switch_candidates(
    model: DualHOModel,
    times: NDArray[np.float64],
    signal: NDArray[np.float64],
    params: Mapping[str, float],
    *,
    is_m3: bool,
    kernel_options: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    original = dict(params)
    swapped = dict(params)
    swapped["D_H_cm2_s"], swapped["D_O_cm2_s"] = (
        original["D_O_cm2_s"],
        original["D_H_cm2_s"],
    )
    swapped["A_H"], swapped["A_O"] = original["A_O"], original["A_H"]
    if is_m3:
        swapped["k_H_cm_s"], swapped["k_O_cm_s"] = (
            original["k_O_cm_s"],
            original["k_H_cm_s"],
        )

    original_fit = _evaluate_forward(
        model,
        times,
        original,
        is_m3=is_m3,
        kernel_options=kernel_options,
    ).total
    swapped_fit = _evaluate_forward(
        model,
        times,
        swapped,
        is_m3=is_m3,
        kernel_options=kernel_options,
    ).total
    return (
        MappingProxyType(
            {
                "labeling": "as_fitted",
                "params": MappingProxyType(original),
                "SSE": float(np.sum((signal - original_fit) ** 2)),
            }
        ),
        MappingProxyType(
            {
                "labeling": "H_O_swapped",
                "params": MappingProxyType(swapped),
                "SSE": float(np.sum((signal - swapped_fit) ** 2)),
            }
        ),
    )


def _information_criterion_prefers_m3(m2: FitResult, m3: FitResult) -> bool:
    if isfinite(m2.AICc) and isfinite(m3.AICc):
        return m3.AICc < m2.AICc
    return m3.BIC < m2.BIC


def _lag1_autocorrelation(residuals: NDArray[np.float64]) -> float:
    if residuals.size < 3:
        return float("nan")
    left = residuals[:-1] - np.mean(residuals[:-1])
    right = residuals[1:] - np.mean(residuals[1:])
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _validated_data(
    t_s: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    times = _vector("t_s", t_s)
    signal = _vector("y", y)
    if times.size != signal.size:
        raise ValueError("t_s and y must have the same length")
    if times.size < 3:
        raise ValueError("at least three observations are required")
    if np.any(times < 0.0):
        raise ValueError("t_s values must be nonnegative")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_s must be strictly increasing with no duplicates")
    if times[-1] == times[0]:
        raise ValueError("t_s must span a positive interval")
    return times, signal


def _vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    return result


def _automatic_baseline_bounds(
    signal: NDArray[np.float64],
) -> tuple[float, float]:
    span = float(np.ptp(signal))
    pad = max(10.0 * span, float(np.max(np.abs(signal))) + 1.0)
    return float(np.min(signal) - pad), float(np.max(signal) + pad)


def _signal_scale(signal: NDArray[np.float64]) -> float:
    dynamic_range = float(np.ptp(signal))
    absolute_scale = float(np.max(np.abs(signal)))
    numerical_floor = np.finfo(np.float64).eps * max(1.0, absolute_scale) * 100.0
    return max(dynamic_range, numerical_floor, 1.0e-15)


def _coerce_label_constraint(value: LabelConstraint | str) -> LabelConstraint:
    if isinstance(value, LabelConstraint):
        return value
    if not isinstance(value, str):
        raise TypeError("label_constraint must be D_H>=D_O or none")
    normalized = value.strip().replace(" ", "")
    aliases = {
        "D_H>=D_O": LabelConstraint.D_H_GE_D_O,
        "d_h>=d_o": LabelConstraint.D_H_GE_D_O,
        "ordered": LabelConstraint.D_H_GE_D_O,
        "none": LabelConstraint.NONE,
        "unconstrained": LabelConstraint.NONE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("label_constraint must be D_H>=D_O or none") from exc


def _bounds(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    lower = _finite_scalar(f"{name}[0]", values[0])
    upper = _finite_scalar(f"{name}[1]", values[1])
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    return lower, upper


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_finite_scalar(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _positive_finite_scalar(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _closed_unit_interval(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result
