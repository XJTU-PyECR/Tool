"""Presentation service for fit credibility and opt-in uncertainty diagnostics.

The service composes existing fitting, local-uncertainty, model-selection, and
advanced-uncertainty modules.  It contains no ECR forward equations and never
changes a completed fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite, log10
from typing import Callable, Mapping

import numpy as np

from pyecr.config import AnalysisConfig, ModelConfig
from pyecr.diagnostics.model_selection import (
    CandidateAssessment,
    ParameterBound,
    describe_bound_proximity,
    select_model,
)
from pyecr.diagnostics.boundary_contact import bound_contact as _bound_contact, contact_reason
from pyecr.diagnostics.error_map import ErrorMapResult
from pyecr.diagnostics.dual_error_map import DualErrorMapResult
from pyecr.diagnostics.sensitivity import SensitivityPresentation, compute_sensitivity
from pyecr.diagnostics.evaluation import (
    DiagnosticEvaluation, EvaluationPolicy, VERSION as EVALUATION_VERSION, LEGACY_VERSION, evaluate_fit,
)
from pyecr.fitting.base import FitCandidate, FitResult
from pyecr.io.dataset import EcrDataset
from pyecr.models.ids import ModelID
from pyecr.services.analysis import AnalysisService, CancellationToken
from pyecr.uncertainty import (
    AdvancedUncertaintyConfig,
    LocalUncertaintyResult,
    ProgressUpdate,
    UncertaintyMethod,
    UncertaintyResult,
    estimate_local_uncertainty,
    profile_likelihood_single,
    residual_bootstrap_dual_ho,
    residual_bootstrap_single,
)

__all__ = [
    "AdvancedDiagnosticsPresentation",
    "BiDimensionPresentation",
    "BiPresentation",
    "CorrelationPresentation",
    "DiagnosticsPresentation",
    "DiagnosticsService",
    "InformationCriteriaPresentation",
    "JacobianPresentation",
    "MultiStartPresentation",
    "MultiStartSpreadPresentation",
    "ParameterDiagnosticsPresentation",
    "ParameterPresentation",
    "ResidualSummaryPresentation",
    "ReliabilityAdvisorPresentation",
    "SensitivityPresentation",
]


ProgressReporter = Callable[[float, str], None]
_AXES = ("x", "y", "z")
_FINITE_MODELS = {ModelID.M1_SINGLE_DK, ModelID.M3_DUAL_HO_DK}
_DUAL_MODELS = {ModelID.M2_DUAL_HO_D, ModelID.M3_DUAL_HO_DK}


@dataclass(frozen=True, slots=True)
class ResidualSummaryPresentation:
    count: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    lag1_autocorrelation: float | None
    lag1_reason: str | None = None


@dataclass(frozen=True, slots=True)
class BiDimensionPresentation:
    process: str
    axis: str
    full_dimension_cm: float
    value: float
    regime: str


@dataclass(frozen=True, slots=True)
class BiPresentation:
    available: bool
    rows: tuple[BiDimensionPresentation, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ParameterPresentation:
    name: str
    estimate: float
    unit: str
    ci_lower: float | None
    ci_upper: float | None
    ci_reason: str | None
    bound_hit: bool | None
    bound_side: str | None
    bound_reason: str | None
    reliable: bool
    reliability_reason: str


@dataclass(frozen=True, slots=True)
class ParameterDiagnosticsPresentation:
    available: bool
    rows: tuple[ParameterPresentation, ...]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CorrelationPresentation:
    available: bool
    parameter_names: tuple[str, ...] = ()
    matrix: tuple[tuple[float | None, ...], ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class JacobianPresentation:
    available: bool
    rank: int | None = None
    parameter_count: int | None = None
    condition_number: float | None = None
    source: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MultiStartSpreadPresentation:
    parameter: str
    minimum: float
    maximum: float
    spread_decades: float | None
    candidate_count: int


@dataclass(frozen=True, slots=True)
class MultiStartPresentation:
    available: bool
    total_candidates: int = 0
    comparable_candidates: int = 0
    rows: tuple[MultiStartSpreadPresentation, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class InformationCriteriaPresentation:
    SSE: float
    RMSE: float
    R2: float
    AIC: float
    AICc: float
    BIC: float


@dataclass(frozen=True, slots=True)
class ReliabilityAdvisorPresentation:
    """Evidence-based guidance, separate from model fit adequacy.

    Statuses are explanatory heuristics and never alter a FitResult or the
    model-selection gate. Each evidence line names its source and availability.
    """

    identifiability_status: str = "unavailable"
    identifiability_explanation: str = "No current identifiability evidence."
    model_adequacy_status: str = "unavailable"
    model_adequacy_explanation: str = "No current fit-adequacy evidence."
    bi_guidance: str = "Bi is unavailable for this model."
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticsPresentation:
    model_id: str
    time_s: tuple[float, ...]
    residuals: tuple[float, ...]
    residual_summary: ResidualSummaryPresentation
    bi: BiPresentation
    parameters: tuple[ParameterPresentation, ...]
    parameter_uncertainty: ParameterDiagnosticsPresentation
    correlation: CorrelationPresentation
    jacobian: JacobianPresentation
    multistart: MultiStartPresentation
    information_criteria: InformationCriteriaPresentation
    optimizer_success: bool
    reliable: bool
    reliability_label: str
    best_statistical: str
    best_reliable: str | None
    gate_failures: tuple[str, ...]
    reasoning: tuple[str, ...]
    warnings: tuple[str, ...]
    sensitivity: SensitivityPresentation = field(
        default_factory=lambda: SensitivityPresentation(
            available=False, reason="Sensitivity was not recorded."
        )
    )
    reliability_advisor: ReliabilityAdvisorPresentation = field(
        default_factory=ReliabilityAdvisorPresentation
    )
    evaluation: DiagnosticEvaluation | None = None


@dataclass(frozen=True, slots=True)
class AdvancedDiagnosticsPresentation:
    method: UncertaintyMethod
    available: bool
    result: UncertaintyResult | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    experimental_result: object | None = None
    binding: object | None = None
    fit_generation: int | None = None


class DiagnosticsService:
    """Create conservative diagnostics from one current fit.

    Basic diagnostics are read-only and inexpensive.  Bootstrap and profile
    likelihood are separate methods so callers cannot run them accidentally.
    """

    def __init__(self, *, analysis_service: AnalysisService | None = None,
                 evaluation_policy: EvaluationPolicy | None = None) -> None:
        if analysis_service is not None and not isinstance(
            analysis_service, AnalysisService
        ):
            raise TypeError("analysis_service must be an AnalysisService")
        self._analysis_service = analysis_service or AnalysisService()
        if evaluation_policy is not None and not isinstance(evaluation_policy, EvaluationPolicy):
            raise TypeError("evaluation_policy must be EvaluationPolicy")
        self.evaluation_policy = evaluation_policy or EvaluationPolicy()

    def describe(self, dataset, config, fit_result, *, error_map_result=None,
                 dual_error_map_result=None, policy_version=EVALUATION_VERSION, include_sensitivity=True):
        """Explicit public Diagnostics action using the current versioned policy."""
        return self._describe_evidence(dataset, config, fit_result,
            error_map_result=error_map_result, dual_error_map_result=dual_error_map_result,
            policy_version=policy_version, include_sensitivity=include_sensitivity)

    def _describe_evidence(
        self,
        dataset: EcrDataset,
        config: AnalysisConfig,
        fit_result: FitResult,
        *,
        error_map_result: ErrorMapResult | None = None,
        dual_error_map_result: DualErrorMapResult | None = None,
        policy_version: str = EVALUATION_VERSION,
        include_sensitivity: bool = True,
    ) -> DiagnosticsPresentation:
        """Build all immediately available credibility diagnostics."""

        if policy_version not in {EVALUATION_VERSION, LEGACY_VERSION}:
            raise ValueError("Unsupported Diagnostics policy")
        _validate_inputs(dataset, config, fit_result)
        model_id = config.model_id.value
        uncertainty, uncertainty_reason = _local_uncertainty(fit_result)
        bi = _bi_presentation(config, fit_result)
        bounds = {
            name: ParameterBound(pair[0], pair[1], space="physical")
            for name, pair in config.parameter_bounds.items()
            if name in fit_result.params
        }
        if config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}:
            bounds = {name: ParameterBound(*pair, space="physical") for name, pair in
                      _surface_record(fit_result)["parameter_bounds"].items()}
        selection = select_model(
            {model_id: fit_result},
            uncertainties=(
                None if uncertainty is None else {model_id: uncertainty}
            ),
            parameter_bounds={model_id: bounds},
        )
        assessment = selection.assessments[model_id]
        gate_failures = list(assessment.gate_failures)
        if uncertainty is not None and uncertainty.rank < len(fit_result.params):
            gate_failures.append(
                "JACOBIAN_RANK_DEFICIENT: "
                f"rank={uncertainty.rank}, parameters={len(fit_result.params)}."
            )
        gate_failures = list(dict.fromkeys(gate_failures))
        parameters = _parameter_rows(
            config,
            fit_result,
            uncertainty,
            assessment,
            uncertainty_reason,
        )
        warnings = list(fit_result.warnings)
        if uncertainty is not None:
            warnings.extend(uncertainty.warnings)
        elif uncertainty_reason:
            warnings.append(f"LOCAL_UNCERTAINTY_NA: {uncertainty_reason}")
        warnings.extend(selection.warnings)
        warnings.extend(gate_failures)
        if bi.available and any(row.regime != "mixed_control" for row in bi.rows):
            warnings.append(
                "BI_EXTREME: at least one active dimension is outside the mixed-control region."
            )
        warnings = list(dict.fromkeys(warnings))

        residual_summary = _residual_summary(fit_result.residuals)
        correlation = _correlation_presentation(fit_result, uncertainty, uncertainty_reason)
        jacobian = _jacobian_presentation(fit_result, uncertainty, uncertainty_reason)
        multistart = _multistart_presentation(fit_result.candidates)
        sensitivity = (compute_sensitivity(dataset, config, fit_result) if include_sensitivity
                       else SensitivityPresentation(available=False, reason="Not requested for evidence verification."))
        advisor = _reliability_advisor(
            fit_result, bi, parameters, correlation, jacobian, multistart,
            residual_summary,
            error_map_result=error_map_result,
            dual_error_map_result=dual_error_map_result,
        )
        reliable = assessment.reliable
        presentation = DiagnosticsPresentation(
            model_id=model_id,
            time_s=tuple(float(value) for value in dataset.processed_time),
            residuals=tuple(float(value) for value in fit_result.residuals),
            residual_summary=residual_summary,
            bi=bi,
            parameters=parameters,
            parameter_uncertainty=ParameterDiagnosticsPresentation(
                available=(uncertainty is not None and (config.model_id not in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}
                    or bool(_surface_record(fit_result)["ci_available"]))),
                rows=parameters,
                reason=("M7 local intervals unavailable; inspect rank/conditioning/boundary" if config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}
                    and not _surface_record(fit_result)["ci_available"] else uncertainty_reason if uncertainty is None else None),
            ),
            correlation=correlation,
            jacobian=jacobian,
            multistart=multistart,
            information_criteria=InformationCriteriaPresentation(
                SSE=fit_result.SSE,
                RMSE=fit_result.RMSE,
                R2=fit_result.R2,
                AIC=fit_result.AIC,
                AICc=fit_result.AICc,
                BIC=fit_result.BIC,
            ),
            optimizer_success=fit_result.exit_status > 0,
            reliable=reliable,
            reliability_label=(
                "Credibility gates passed; physical interpretation remains conditional"
                if reliable
                else "Credibility gates warn; optimizer completion is a separate status"
            ),
            best_statistical=selection.best_statistical,
            best_reliable=selection.best_reliable,
            gate_failures=tuple(gate_failures),
            reasoning=assessment.reasoning,
            warnings=tuple(warnings),
            sensitivity=sensitivity,
            reliability_advisor=advisor,
        )
        if policy_version == LEGACY_VERSION:
            return presentation
        return _evaluated_presentation(presentation, dataset, config, fit_result, self.evaluation_policy)

    def run_experimental_m1_uncertainty(
        self,
        dataset: EcrDataset,
        config: AnalysisConfig,
        fit_result: FitResult,
        *,
        seed: int,
        acknowledge_exchangeable_residuals: bool,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
        confidence_level: float = 0.95,
    ) -> AdvancedDiagnosticsPresentation:
        """P66B explicit M1 experiment; never replaces the traditional local CI."""

        from pyecr.services.m1_deep_uncertainty import run_experimental_m1_bootstrap

        result = run_experimental_m1_bootstrap(
            dataset, config, fit_result, seed=seed,
            acknowledge_exchangeable_residuals=acknowledge_exchangeable_residuals,
            service=self, cancel_token=cancel_token,
            progress_callback=progress_callback,
            confidence_level=confidence_level,
        )
        return AdvancedDiagnosticsPresentation(
            method=UncertaintyMethod.RESIDUAL_BOOTSTRAP,
            available=result.status == "complete",
            reason=None if result.status == "complete" else result.status,
            warnings=result.warnings,
            experimental_result=result,
        )

    def run_bootstrap(
        self,
        dataset: EcrDataset,
        config: AnalysisConfig,
        fit_result: FitResult,
        *,
        options: AdvancedUncertaintyConfig | None = None,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
        random_seed: int | None = None,
        confidence_level: float = 0.95,
    ) -> AdvancedDiagnosticsPresentation:
        """Run residual bootstrap only after an explicit caller request."""

        _validate_inputs(dataset, config, fit_result)
        from pyecr.services.advanced_intervals import checked_C, relevel_result, observation_conditions
        confidence_level=checked_C(confidence_level)
        if config.fit.loss != 'linear':
            return AdvancedDiagnosticsPresentation(method=UncertaintyMethod.RESIDUAL_BOOTSTRAP,available=False,
                reason='Unsupported: residual-bootstrap output is unverified for robust loss; no alternate interval substituted.')
        settings = options or AdvancedUncertaintyConfig()
        token = cancel_token or CancellationToken()
        reporter = progress_callback or _ignore_progress
        token.raise_if_cancelled()

        def refit(signal: np.ndarray) -> FitResult:
            token.raise_if_cancelled()
            synthetic = _dataset_with_processed_signal(dataset, signal)
            return self._analysis_service.fit(
                synthetic,
                config,
                cancel_token=token,
            )

        def progress(update: ProgressUpdate) -> None:
            token.raise_if_cancelled()
            reporter(update.fraction, update.message or update.stage)

        engine = (
            residual_bootstrap_dual_ho
            if config.model_id in _DUAL_MODELS and 'fixed_d' not in fit_result.diagnostics
            else residual_bootstrap_single
        )
        result = engine(
            fit_result,
            refit_function=refit,
            n_resamples=settings.bootstrap_resamples,
            random_seed=random_seed,
            n_jobs=settings.bootstrap_n_jobs,
            progress_callback=progress,
            confidence_level=confidence_level,
            residual_convention=('fit_minus_observed' if config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}
                or (config.model_id is ModelID.M3_DUAL_HO_DK and 'fixed_d' in fit_result.diagnostics) else 'observed_minus_fit'),
        )
        record=fit_result.diagnostics.get('fixed_d',fit_result.diagnostics.get('surface_k',fit_result.diagnostics.get('dual_surface_k',{})))
        result=replace(result,diagnostics={**result.diagnostics,**observation_conditions(dataset,config,fit_result),'physical_model_id':config.model_id.value,
            'assumption':'Additive centered exchangeable residuals, conditional fixed inputs; no calibrated coverage or preserved time correlation',
            'ci_scope':record.get('ci_scope','conditional_on_fixed_inputs'),'ci_conditions':record.get('ci_conditions',{}),
            'fixed_parameters':dict(config.model.fixed_parameters),'geometry':config.model.geometry.to_dict(),
            'boundary':config.boundary.to_dict(),'fit_row_scope':'full_observations','N':len(fit_result.y_fit),
            'propagates_fixed_input_uncertainty':False})
        result=relevel_result(fit_result,result,confidence_level)
        token.raise_if_cancelled()
        return AdvancedDiagnosticsPresentation(
            method=UncertaintyMethod.RESIDUAL_BOOTSTRAP,
            available=bool(result.intervals),
            reason=result.diagnostics.get('output_reason') or None,
            result=result,
            warnings=result.warnings,
        )

    def run_profile(
        self,
        dataset: EcrDataset,
        config: AnalysisConfig,
        fit_result: FitResult,
        *,
        options: AdvancedUncertaintyConfig | None = None,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
        confidence_level: float = 0.95,
    ) -> AdvancedDiagnosticsPresentation:
        """Run profile likelihood for production single-process fit adapters.

        The existing staged dual fitter requires shared H/O bounds and cannot
        hold one kinetic parameter fixed without changing that fitter.  M2/M3
        therefore return an explicit N/A result rather than a misleading scan.
        """

        _validate_inputs(dataset, config, fit_result)
        from pyecr.services.advanced_intervals import checked_C, relevel_result, observation_conditions
        confidence_level=checked_C(confidence_level)
        if config.fit.loss != 'linear':
            return AdvancedDiagnosticsPresentation(method=UncertaintyMethod.PROFILE_LIKELIHOOD,available=False,
                reason='Unsupported: the existing Gaussian SSE profile is unverified for robust loss.')
        if config.model_id in _DUAL_MODELS:
            return AdvancedDiagnosticsPresentation(
                method=UncertaintyMethod.PROFILE_LIKELIHOOD,
                available=False,
                reason=(
                    "N/A: the current production M2/M3 staged fitter has no "
                    "fixed-parameter refit adapter; no unconstrained refit is "
                    "reported as profile likelihood."
                ),
            )
        if config.model_id is ModelID.M7_SINGLE_K:
            return AdvancedDiagnosticsPresentation(
                method=UncertaintyMethod.PROFILE_LIKELIHOOD,
                available=False,
                reason="N/A: analytic single k has no constrained nuisance-refit adapter. "
                       "Conditional local CI and residual bootstrap are available; "
                       "a narrow-bound approximation is not reported as profile likelihood.",
            )
        settings = options or AdvancedUncertaintyConfig()
        token = cancel_token or CancellationToken()
        reporter = progress_callback or _ignore_progress
        parameters = tuple(
            name for name in ("D_cm2_s", "k_cm_s") if name in fit_result.params
        )
        if not parameters:
            return AdvancedDiagnosticsPresentation(
                method=UncertaintyMethod.PROFILE_LIKELIHOOD,
                available=False,
                reason="N/A: this fit has no supported logD/logk profile parameters.",
            )

        def refit(
            fixed_parameter: str,
            fixed_value: float,
            start_params: Mapping[str, float],
        ) -> FitResult:
            del start_params
            token.raise_if_cancelled()
            profile_config = _narrow_parameter_bound(
                config,
                fixed_parameter,
                fixed_value,
            )
            return self._analysis_service.fit(
                dataset,
                profile_config,
                cancel_token=token,
            )

        def progress(update: ProgressUpdate) -> None:
            token.raise_if_cancelled()
            reporter(update.fraction, update.message or update.stage)

        result = profile_likelihood_single(
            fit_result,
            refit_function=refit,
            parameters=parameters,
            span_decades=settings.profile_span_decades,
            n_points=settings.profile_points,
            n_jobs=1,
            progress_callback=progress,
            confidence_level=confidence_level,
        )
        result=replace(result,diagnostics={**result.diagnostics,**observation_conditions(dataset,config,fit_result),'physical_model_id':config.model_id.value,
            'assumption':'Existing Gaussian SSE likelihood-ratio approximation, conditional fixed inputs; no calibrated coverage',
            'ci_scope':'conditional_on_fixed_inputs','fixed_parameters':dict(config.model.fixed_parameters),
            'ci_conditions':fit_result.diagnostics.get('fixed_d',{}).get('ci_conditions',{}),
            'geometry':config.model.geometry.to_dict(),'boundary':config.boundary.to_dict(),
            'constraint_adapter':'existing relative half-width 5e-11 bound adapter; no new exact constrained solver',
            'N':len(fit_result.y_fit),'propagates_fixed_input_uncertainty':False})
        result=relevel_result(fit_result,result,confidence_level)
        token.raise_if_cancelled()
        return AdvancedDiagnosticsPresentation(
            method=UncertaintyMethod.PROFILE_LIKELIHOOD,
            available=bool(result.intervals),
            reason=result.diagnostics.get('output_reason') or None,
            result=result,
            warnings=result.warnings,
        )


def _validate_inputs(
    dataset: EcrDataset,
    config: AnalysisConfig,
    fit_result: FitResult,
) -> None:
    if not isinstance(dataset, EcrDataset):
        raise TypeError("dataset must be an EcrDataset")
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")
    if not isinstance(fit_result, FitResult):
        raise TypeError("fit_result must be a FitResult")
    config.validate()
    if fit_result.y_fit.size != dataset.processed_signal.size:
        raise ValueError("fit result length does not match the processed dataset")


def _evaluated_presentation(presentation, dataset, config, fit, policy):
    """Publish v2 evidence; legacy policy remains explicit for stored consumers."""
    bounds = {n: pair for n, pair in config.parameter_bounds.items() if n in fit.params}
    surface = config.model_id in {ModelID.M7_SINGLE_K, ModelID.M8_DUAL_K}
    if surface:
        bounds = dict(_surface_record(fit)["parameter_bounds"])
    evidence = evaluate_fit(fit, times=dataset.processed_time, bounds=bounds, loss=config.fit.loss, policy=policy)
    # Retain model-specific nonregular-inference restrictions. Replacing a
    # generic condition-number rule cannot validate near-coincident processes.
    if surface:
        record = _surface_record(fit)
        unavailable = {p.name for p in evidence.parameters if not record.get("ci_availability", {}).get(
            p.name, record.get("ci_available", False))}
        if unavailable:
            evidence = replace(evidence, interval_status="unavailable", parameters=tuple(
                replace(p, ci_lower=None, ci_upper=None, precision="unavailable",
                        reasons=(*p.reasons, "model-specific nonregular local interval is unavailable"))
                if p.name in unavailable else p for p in evidence.parameters),
                blocking_reasons=(*evidence.blocking_reasons,
                    "MODEL_LOCAL_INTERVAL_UNAVAILABLE: " + ", ".join(sorted(unavailable))),
                identifiability_status="weak")
    old_rows = {row.name: row for row in presentation.parameters}
    rows = []
    for item in evidence.parameters:
        reason = "; ".join(item.reasons) or "local precision criterion supported; CI assumptions are reported separately"
        distance = (f"Boundary: {item.boundary}; lower/upper distance="
                    f"{item.lower_distance}/{item.upper_distance} ({item.distance_basis}); "
                    "each edge checked independently of the opposite bound")
        rows.append(replace(old_rows[item.name], ci_lower=item.ci_lower, ci_upper=item.ci_upper,
            ci_reason=None if item.ci_lower is not None else reason,
            bound_hit=item.boundary == "contact" if item.boundary != "unconfigured" else None,
            bound_side=item.contact_side, bound_reason=distance,
            reliable=item.precision == "supported" and item.boundary not in {"contact", "interval_crosses", "outside"},
            reliability_reason=reason))
    rows = tuple(rows)
    residual = replace(presentation.residual_summary, lag1_autocorrelation=evidence.residual.lag1,
        lag1_reason=None if evidence.residual.lag1 is not None else evidence.residual.explanation)
    advisor = replace(presentation.reliability_advisor,
        identifiability_status=evidence.identifiability_status,
        identifiability_explanation=("Scaled Jacobian and parameter-specific local precision; "
            "residual dependence does not propagate a failure into every parameter row."),
        model_adequacy_status=evidence.adequacy_status,
        model_adequacy_explanation=(f"RMSE / fitted response span={evidence.normalized_rmse}; "
            "recorded engineering thresholds, separate from error independence."),
        evidence=(evidence.summary, evidence.residual.explanation,
            f"Scaled information condition={evidence.scaled_condition}; raw={evidence.raw_condition}",
            *presentation.reliability_advisor.evidence[:1]))
    warnings = list(evidence.warnings)
    warnings.extend("Fit-stage note: " + w for w in fit.warnings)
    warnings.extend(w for w in presentation.warnings if w.startswith("BI_EXTREME"))
    return replace(presentation, evaluation=evidence, residual_summary=residual, parameters=rows,
        parameter_uncertainty=ParameterDiagnosticsPresentation(
            available=any(p.ci_lower is not None for p in evidence.parameters), rows=rows,
            reason=evidence.interval_status),
        correlation=CorrelationPresentation(available=bool(evidence.correlation),
            parameter_names=tuple(fit.params), matrix=evidence.correlation,
            reason=None if evidence.correlation else "Scaled local covariance unavailable."),
        jacobian=replace(presentation.jacobian, available=fit.jacobian is not None,
            rank=evidence.rank, condition_number=evidence.raw_condition),
        reliable=evidence.reliable, reliability_label=evidence.summary,
        best_reliable=presentation.model_id if evidence.reliable else None,
        gate_failures=evidence.blocking_reasons, reasoning=(evidence.policy_version, evidence.residual.explanation),
        warnings=tuple(dict.fromkeys(warnings)), reliability_advisor=advisor)


def canonical_diagnostics(dataset, config, fit, diagnostics):
    """Reconstruct cheap evidence using the recorded policy; never Fit/forward."""
    version = EVALUATION_VERSION if diagnostics.evaluation is not None else LEGACY_VERSION
    policy = diagnostics.evaluation.policy if diagnostics.evaluation is not None else None
    return DiagnosticsService(evaluation_policy=policy)._describe_evidence(
        dataset, config, fit, policy_version=version, include_sensitivity=False)


def _local_uncertainty(
    fit_result: FitResult,
) -> tuple[LocalUncertaintyResult | None, str | None]:
    if fit_result.jacobian is None:
        return None, "the completed fit does not contain a Jacobian"
    try:
        return estimate_local_uncertainty(fit_result), None
    except (ArithmeticError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return None, f"local uncertainty failed: {type(exc).__name__}: {exc}"


def _residual_summary(residuals: np.ndarray) -> ResidualSummaryPresentation:
    values = np.asarray(residuals, dtype=np.float64)
    lag1: float | None
    reason: str | None = None
    if values.size < 3:
        lag1 = None
        reason = "at least three residuals are required"
    else:
        left = values[:-1] - float(np.mean(values[:-1]))
        right = values[1:] - float(np.mean(values[1:]))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= np.finfo(np.float64).eps:
            lag1 = None
            reason = "residual variance is too small for autocorrelation"
        else:
            lag1 = float(np.dot(left, right) / denominator)
    return ResidualSummaryPresentation(
        count=values.size,
        mean=float(np.mean(values)),
        standard_deviation=(float(np.std(values, ddof=1)) if values.size > 1 else 0.0),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        lag1_autocorrelation=lag1,
        lag1_reason=reason,
    )


def _surface_record(fit):
    return fit.diagnostics.get("surface_k",fit.diagnostics.get("dual_surface_k",{}))

def _bi_presentation(config: AnalysisConfig, fit: FitResult) -> BiPresentation:
    if config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}:
        return BiPresentation(available=False, reason="N/A: D_not_estimated; surface control is an assumption, not a measured Bi.")
    if config.model_id not in _FINITE_MODELS:
        return BiPresentation(
            available=False,
            reason=(
                "N/A: the selected diffusion-controlled model assumes "
                "instantaneous surface equilibration and does not fit k."
            ),
        )
    pairs: tuple[tuple[str, str, str], ...]
    if config.model_id is ModelID.M1_SINGLE_DK:
        pairs = (("single", "D_cm2_s", "k_cm_s"),)
    else:
        pairs = (
            ("H", "D_H_cm2_s", "k_H_cm_s"),
            ("O", "D_O_cm2_s", "k_O_cm_s"),
        )
    physical_params = {**config.model.fixed_parameters, **fit.params}
    missing = sorted(
        {name for _, d_name, k_name in pairs for name in (d_name, k_name)}
        - set(physical_params)
    )
    if missing:
        return BiPresentation(
            available=False,
            reason=(
                "N/A: the selected fit did not report the finite-surface "
                f"parameter(s) {', '.join(missing)}."
            ),
        )
    surface_limit, diffusion_limit = _bi_limits(config)
    # R11 unique point calculation; retain the old Bi presentation and thresholds
    # consumed by existing credibility gates. No new gate is introduced.
    from pyecr.diagnostics.transport import transport_points
    _, points = transport_points(config.model_id, config.model.geometry, physical_params,
                                surface_limit=surface_limit, diffusion_limit=diffusion_limit)
    if any(row.value is None for row in points):
        return BiPresentation(available=False, reason="N/A: finite positive D/k and finite Bi required.")
    rows = [BiDimensionPresentation(row.process,row.axis,row.full_dimension_cm,row.value,row.regime) for row in points]
    return BiPresentation(available=True, rows=tuple(rows))


def _bi_limits(config: AnalysisConfig) -> tuple[float, float]:
    raw = config.model_specific.get("single_ecr", {})
    if not isinstance(raw, Mapping):
        raw = {}
    return (
        float(raw.get("bi_surface_limit", 0.03)),
        float(raw.get("bi_diffusion_limit", 30.0)),
    )


def _reliability_advisor(
    fit: FitResult,
    bi: BiPresentation,
    parameters: tuple[ParameterPresentation, ...],
    correlation: CorrelationPresentation,
    jacobian: JacobianPresentation,
    multistart: MultiStartPresentation,
    residual_summary: ResidualSummaryPresentation,
    *,
    error_map_result: ErrorMapResult | None,
    dual_error_map_result: DualErrorMapResult | None,
) -> ReliabilityAdvisorPresentation:
    """Explain parameter separation using existing results, with no refitting.

    Thresholds are deliberately heuristic. A small or large Bi value alone
    never enters the model-selection reliability gate.
    """

    evidence: list[str] = []
    concerns: list[str] = []
    strong_concerns: list[str] = []
    if bi.available:
        extremes = tuple(
            f"{row.process}/{row.axis} ({row.regime}, Bi={row.value:.3g})"
            for row in bi.rows if row.regime != "mixed_control"
        )
        evidence.append(
            "Bi: " + (
                "extreme heuristic regime at " + ", ".join(extremes)
                if extremes else "all active dimensions in the heuristic mixed-control range"
            )
        )
        if extremes:
            concerns.append("extreme Bi may limit separate D and k information")
        bi_guidance = (
            "For each active axis, Bi_i = k × (full length_i / 2) / D. "
            "Regime thresholds are heuristics, not validity or reliability cutoffs."
        )
    else:
        evidence.append(f"Bi: unavailable ({bi.reason or 'no finite-surface parameter'})")
        bi_guidance = bi.reason or "Bi is unavailable for this model."

    if jacobian.available:
        rank = jacobian.rank
        count = jacobian.parameter_count
        condition = jacobian.condition_number
        evidence.append(
            f"Jacobian: rank {rank}/{count}; "
            + (f"condition {condition:.3g}" if condition is not None else "condition unavailable")
        )
        if rank is not None and count is not None and rank < count:
            strong_concerns.append("Jacobian rank is deficient")
        if condition is None or condition > 1.0e10:
            strong_concerns.append("Jacobian is severely ill-conditioned")
        elif condition > 1.0e7:
            concerns.append("Jacobian is ill-conditioned")
    else:
        evidence.append(f"Jacobian: unavailable ({jacobian.reason or 'no Jacobian'})")
        concerns.append("local Jacobian evidence is unavailable")

    widths: list[tuple[str, float]] = []
    for row in parameters:
        if row.ci_lower is None or row.ci_upper is None:
            continue
        if row.estimate > 0 and row.ci_lower > 0 and row.ci_upper > 0:
            width = log10(row.ci_upper / row.ci_lower)
            widths.append((row.name, width))
    if widths:
        widest = max(widths, key=lambda item: item[1])
        evidence.append(f"CI width: widest 95% interval is {widest[0]} ({widest[1]:.3g} decades)")
        if widest[1] > 2.0:
            strong_concerns.append("a parameter interval spans more than two decades")
        elif widest[1] > 1.0:
            concerns.append("a parameter interval spans more than one decade")
    else:
        evidence.append("CI width: unavailable")

    if correlation.available:
        matrix = correlation.matrix
        pairs = [
            (abs(value), correlation.parameter_names[i], correlation.parameter_names[j])
            for i, row in enumerate(matrix)
            for j, value in enumerate(row)
            if j > i and value is not None
        ]
        if pairs:
            maximum, left, right = max(pairs)
            evidence.append(f"Correlation: max |r|={maximum:.3g} ({left}, {right})")
            if maximum >= 0.98:
                strong_concerns.append("parameters are highly correlated")
            elif maximum >= 0.9:
                concerns.append("parameters are correlated")
        else:
            evidence.append("Correlation: no parameter pair available")
    else:
        evidence.append(f"Correlation: unavailable ({correlation.reason or 'no covariance estimate'})")

    if multistart.available:
        spreads = [
            (row.spread_decades, row.parameter)
            for row in multistart.rows if row.spread_decades is not None
        ]
        if spreads:
            widest, name = max(spreads)
            evidence.append(
                f"MultiStart: {multistart.comparable_candidates} comparable candidates; "
                f"largest spread {name}={widest:.3g} decades"
            )
            if widest > 1.0:
                strong_concerns.append("comparable MultiStart fits have broad parameter spread")
            elif widest > 0.5:
                concerns.append("comparable MultiStart fits differ in parameters")
        else:
            evidence.append("MultiStart: no log-parameter spread available")
    else:
        evidence.append(f"MultiStart: unavailable ({multistart.reason or 'no candidates'})")

    valley = _error_map_valley(error_map_result, dual_error_map_result)
    evidence.append(valley[0])
    if valley[1]:
        concerns.append("Error Map shows an elongated or edge-limited objective valley")

    signal_span = float(np.ptp(fit.y_fit))
    noise = float(np.std(fit.residuals, ddof=1)) if fit.residuals.size > 1 else 0.0
    ratio = signal_span / noise if noise > 0 else (float("inf") if signal_span > 0 else 0.0)
    evidence.append(
        f"Amplitude: fitted span {signal_span:.3g}; residual SD {noise:.3g}; "
        + (f"span/noise {ratio:.3g}" if isfinite(ratio) else "span/noise unbounded")
    )
    if ratio < 3.0:
        strong_concerns.append("signal amplitude is weak relative to residual noise")
    elif ratio < 10.0:
        concerns.append("signal amplitude is modest relative to residual noise")
    if "A_H" in fit.params and "A_O" in fit.params:
        amplitudes = {name: abs(float(fit.params[name])) for name in ("A_H", "A_O")}
        total = sum(amplitudes.values())
        for name, amplitude in amplitudes.items():
            fraction = amplitude / total if total > 0 else 0.0
            evidence.append(f"Amplitude: {name} fraction {fraction:.3g}")
            if fraction < 0.05:
                strong_concerns.append(f"{name} contributes under 5% of process amplitude")

    if strong_concerns:
        identifiability_status = "weak"
        explanation = "; ".join(dict.fromkeys(strong_concerns + concerns))
    elif concerns:
        identifiability_status = "caution"
        explanation = "; ".join(dict.fromkeys(concerns))
    else:
        identifiability_status = "supported"
        explanation = "Available local and repeated-fit evidence supports parameter separation"
    explanation += ". This is an identifiability assessment, not a model-adequacy test."

    lag1 = residual_summary.lag1_autocorrelation
    if lag1 is None:
        evidence.append("Residual structure: lag-1 correlation unavailable")
    else:
        evidence.append(f"Residual structure: lag-1 correlation {lag1:.3g}")
    if fit.exit_status <= 0:
        adequacy_status = "poor"
        adequacy_explanation = "Optimizer did not report success."
    elif signal_span <= 0 or fit.RMSE > 0.2 * signal_span or fit.R2 < 0.8:
        adequacy_status = "poor"
        adequacy_explanation = "Residual error is large relative to the fitted response."
    elif fit.RMSE > 0.05 * signal_span or fit.R2 < 0.95 or (lag1 is not None and abs(lag1) > 0.5):
        adequacy_status = "caution"
        adequacy_explanation = "Residual error, R², or residual structure suggests imperfect fit to observations."
    else:
        adequacy_status = "supported"
        adequacy_explanation = "Optimizer and residual metrics support fit to the observed signal."
    adequacy_explanation += " Model adequacy is separate from parameter identifiability."
    return ReliabilityAdvisorPresentation(
        identifiability_status=identifiability_status,
        identifiability_explanation=explanation,
        model_adequacy_status=adequacy_status,
        model_adequacy_explanation=adequacy_explanation,
        bi_guidance=bi_guidance,
        evidence=tuple(evidence),
    )


def _error_map_valley(
    single: ErrorMapResult | None,
    dual: DualErrorMapResult | None,
) -> tuple[str, bool]:
    if dual is not None:
        elongated = tuple(
            row.process for row in (dual.h_map, dual.o_map)
            if row.quality.elongated_valley
        )
        edge = tuple(
            row.process for row in (dual.h_map, dual.o_map)
            if row.quality.grid_min_on_edge
        )
        parts = []
        if elongated:
            parts.append("elongated " + ", ".join(elongated))
        if edge:
            parts.append("grid minimum on edge " + ", ".join(edge))
        return "Error Map valley: " + ("; ".join(parts) if parts else "no elongated H/O slice reported"), bool(parts)
    if single is None:
        return "Error Map valley: unavailable (map not run)", False
    quality = single.quality
    if quality is not None and quality.grid_min_on_edge:
        return "Error Map valley: grid minimum on edge; shape is only heuristic", True
    valid = np.asarray(single.valid_mask, dtype=bool) & np.isfinite(single.objective_grid)
    if np.count_nonzero(valid) < 12:
        return "Error Map valley: insufficient valid grid cells", False
    grid = np.asarray(single.objective_grid, dtype=np.float64)
    cutoff = float(np.quantile(grid[valid], 0.05))
    region = valid & (grid <= cutoff)
    if np.count_nonzero(region) < 6:
        return "Error Map valley: insufficient low-error cells", False
    d_index, k_index = np.nonzero(region)
    coordinates = np.column_stack((single.log10_D[d_index], single.log10_k[k_index]))
    eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False))
    if eigenvalues[0] <= 0:
        return "Error Map valley: low-error region is degenerate (heuristic)", True
    ratio = float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
    return f"Error Map valley: low-error axis ratio {ratio:.3g} (grid heuristic)", ratio > 5.0


def _parameter_rows(
    config: AnalysisConfig,
    fit: FitResult,
    uncertainty: LocalUncertaintyResult | None,
    assessment: CandidateAssessment,
    uncertainty_reason: str | None,
) -> tuple[ParameterPresentation, ...]:
    units = config.parameter_units
    rows: list[ParameterPresentation] = []
    for name, estimate in fit.params.items():
        interval = (
            None if uncertainty is None else uncertainty.confidence_intervals.get(name)
        )
        if (config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K}
                and not _surface_record(fit).get("ci_availability",{}).get(name,_surface_record(fit).get("ci_available", False))):
            interval = None
            uncertainty_reason = "M7 local CI unavailable: rank/conditioning/boundary; conditional on fixed inputs"
        bound = (_surface_record(fit)["parameter_bounds"].get(name)
                 if config.model_id in {ModelID.M7_SINGLE_K,ModelID.M8_DUAL_K} else config.parameter_bounds.get(name))
        bound_hit, bound_side = _bound_contact(
            estimate,
            bound,
            is_log=fit.param_space[name] == "log",
        )
        reasons: list[str] = []
        if interval is None:
            reasons.append(uncertainty_reason or "95% CI unavailable")
        elif (
            fit.param_space[name] == "log"
            and interval.lower > 0.0
            and interval.upper > 0.0
            and log10(interval.upper / interval.lower) > 2.0
        ):
            reasons.append("95% CI spans more than 2 decades")
        if bound_hit:
            reasons.append(f"estimate touches {bound_side} bound")
        if name in assessment.parameters_at_bounds:
            reasons.append("credibility proximity warning (physical range 1%); numerical contact is a separate check")
        if uncertainty is not None and not uncertainty.reliable_separation:
            reasons.append("Jacobian does not reliably separate parameters")
        if not assessment.reliable:
            reasons.append("model credibility Gate failed")
        rows.append(
            ParameterPresentation(
                name=name,
                estimate=float(estimate),
                unit=str(units.get(name, "—")),
                ci_lower=None if interval is None else float(interval.lower),
                ci_upper=None if interval is None else float(interval.upper),
                ci_reason=(
                    uncertainty_reason if interval is None else None
                ),
                bound_hit=bound_hit,
                bound_side=bound_side,
                bound_reason=(
                    contact_reason(estimate, bound, is_log=fit.param_space[name] == "log")
                    + (" | " + describe_bound_proximity(fit, name, ParameterBound(*bound, space="physical"))
                       if bound is not None else "")
                ),
                reliable=not reasons,
                reliability_reason="reliable" if not reasons else "; ".join(reasons),
            )
        )
    return tuple(rows)


def _correlation_presentation(
    fit: FitResult,
    uncertainty: LocalUncertaintyResult | None,
    reason: str | None,
) -> CorrelationPresentation:
    if uncertainty is None:
        return CorrelationPresentation(available=False, reason=reason)
    return CorrelationPresentation(
        available=True,
        parameter_names=tuple(fit.params),
        matrix=tuple(
            tuple(float(value) if isfinite(float(value)) else None for value in row)
            for row in uncertainty.correlation_matrix
        ),
    )


def _jacobian_presentation(
    fit: FitResult,
    uncertainty: LocalUncertaintyResult | None,
    reason: str | None,
) -> JacobianPresentation:
    if uncertainty is None:
        return JacobianPresentation(available=False, reason=reason)
    condition = float(uncertainty.condition_number)
    return JacobianPresentation(
        available=True,
        rank=uncertainty.rank,
        parameter_count=len(fit.params),
        condition_number=condition if isfinite(condition) else None,
        source=uncertainty.jacobian_source,
        reason=("condition number is infinite" if not isfinite(condition) else None),
    )


def _multistart_presentation(
    candidates: tuple[FitCandidate, ...],
) -> MultiStartPresentation:
    if not candidates:
        return MultiStartPresentation(
            available=False,
            reason="N/A: retained MultiStart candidates are not available.",
        )
    best_sse = min(candidate.SSE for candidate in candidates)
    denominator = max(abs(best_sse), np.finfo(np.float64).tiny)
    comparable = tuple(
        candidate
        for candidate in candidates
        if (candidate.SSE - best_sse) / denominator < 0.01
    )
    names = tuple(candidates[0].params)
    rows: list[MultiStartSpreadPresentation] = []
    for name in names:
        values = np.asarray(
            [candidate.params[name] for candidate in comparable],
            dtype=np.float64,
        )
        spread = (
            float(np.ptp(np.log10(values)))
            if np.all(values > 0.0)
            else None
        )
        rows.append(
            MultiStartSpreadPresentation(
                parameter=name,
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
                spread_decades=spread,
                candidate_count=values.size,
            )
        )
    return MultiStartPresentation(
        available=True,
        total_candidates=len(candidates),
        comparable_candidates=len(comparable),
        rows=tuple(rows),
    )


def _dataset_with_processed_signal(
    dataset: EcrDataset,
    signal: np.ndarray,
) -> EcrDataset:
    return EcrDataset(
        raw_time=dataset.raw_time,
        raw_signal=dataset.raw_signal,
        processed_time=dataset.processed_time,
        processed_signal=signal,
        mask=dataset.mask,
        metadata=dataset.metadata,
        history=dataset.history,
    )


def _narrow_parameter_bound(
    config: AnalysisConfig,
    parameter: str,
    value: float,
) -> AnalysisConfig:
    relative_half_width = 5.0e-11
    lower = value * (1.0 - relative_half_width)
    upper = value * (1.0 + relative_half_width)
    if lower == upper:
        lower = float(np.nextafter(value, 0.0))
        upper = float(np.nextafter(value, np.inf))
    bounds = dict(config.model.parameter_bounds)
    bounds[parameter] = (lower, upper)
    initial = dict(config.model.initial_values)
    initial[parameter] = value
    model: ModelConfig = replace(
        config.model,
        parameter_bounds=bounds,
        initial_values=initial,
    )
    result = replace(config, model=model)
    result.validate()
    return result


def _ignore_progress(fraction: float, message: str) -> None:
    del fraction, message
