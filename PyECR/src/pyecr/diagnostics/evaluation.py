"""Versioned, read-only evaluation of fit, precision and error assumptions.

The reference portmanteau p-value is exploratory for fitted nonlinear residuals.
It is never a calibrated ECR hypothesis test or a replacement noise model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from math import isfinite, log10
from typing import Mapping

import numpy as np
from scipy.stats import chi2, norm, t as student_t

from pyecr.fitting.base import FitResult

VERSION = "diagnostics-evaluation/2"
LEGACY_VERSION = "diagnostics-gates/1"


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    confidence_level: float = .95
    residual_alpha: float = .01
    minimum_test_count: int = 20
    maximum_lag: int = 10
    spacing_relative_tolerance: float = .05
    log_ci_width_limit: float = 2.
    linear_ci_width_limit: float = 1.
    correlation_warning: float = .98
    scaled_condition_limit: float = 1e12
    log_contact_tolerance: float = 1e-5
    linear_contact_rtol: float = 1e-8
    adequacy_caution_nrmse: float = .05
    adequacy_poor_nrmse: float = .20

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"Invalid diagnostic policy {field.name}")
        if not (0 < self.confidence_level < 1 and 0 < self.residual_alpha < 1
                and self.correlation_warning < 1):
            raise ValueError("Diagnostic probabilities must lie between zero and one")
        if self.confidence_level != .95:
            raise ValueError("Diagnostics precision uses 95%; selected output confidence is a separate control")
        if (type(self.minimum_test_count) is not int or self.minimum_test_count < 20
                or type(self.maximum_lag) is not int
                or self.adequacy_caution_nrmse >= self.adequacy_poor_nrmse):
            raise ValueError("Invalid diagnostic count or adequacy thresholds")


@dataclass(frozen=True, slots=True)
class ResidualEvidence:
    count: int
    lag1: float | None
    acf: tuple[float, ...]
    spacing: str
    median_spacing_s: float | None
    reference_band: float | None
    simultaneous_band: float | None
    ljung_box_q: float | None
    reference_pvalue: float | None
    dependence: str
    explanation: str
    method: str = "centred-acf/common-energy; Ljung-Box reference df=lags"


@dataclass(frozen=True, slots=True)
class ParameterEvidence:
    name: str
    estimate: float
    ci_lower: float | None
    ci_upper: float | None
    coordinate_se: float | None
    width: float | None
    width_basis: str
    precision: str
    boundary: str
    contact_side: str | None
    lower_distance: float | None
    upper_distance: float | None
    distance_basis: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticEvaluation:
    policy: EvaluationPolicy
    optimizer_status: str
    point_estimates_usable: bool
    adequacy_status: str
    normalized_rmse: float | None
    identifiability_status: str
    interval_status: str
    residual: ResidualEvidence
    parameters: tuple[ParameterEvidence, ...]
    correlation: tuple[tuple[float | None, ...], ...]
    raw_condition: float | None
    scaled_condition: float | None
    rank: int | None
    parameter_count: int
    degrees_of_freedom: int
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    policy_version: str = VERSION

    def __post_init__(self):
        if self.policy_version != VERSION:
            raise ValueError("Unsupported Diagnostics evaluation policy")
        if (not isinstance(self.policy, EvaluationPolicy) or not isinstance(self.residual, ResidualEvidence)
                or type(self.point_estimates_usable) is not bool
                or self.optimizer_status not in {"completed", "incomplete"}):
            raise ValueError("Invalid diagnostic evidence types or optimizer status")
        if (type(self.parameter_count) is not int or self.parameter_count < 1
                or type(self.residual.count) is not int or self.residual.count < 0
                or self.degrees_of_freedom != self.residual.count - self.parameter_count):
            raise ValueError("Diagnostic sample/parameter/degrees-of-freedom conflict")
        if self.rank is not None and (type(self.rank) is not int or not 0 <= self.rank <= min(self.parameter_count, self.residual.count)):
            raise ValueError("Invalid diagnostic rank")
        if self.interval_status not in {"conditional_iid", "assumptions_unverified", "boundary_limited", "unavailable"}:
            raise ValueError("Unknown diagnostic interval status")
        if self.adequacy_status not in {"supported", "caution", "poor", "unavailable"}:
            raise ValueError("Unknown diagnostic adequacy status")
        if self.identifiability_status not in {"supported", "caution", "weak", "unavailable"}:
            raise ValueError("Unknown diagnostic identifiability status")
        if self.residual.dependence not in {"not_detected", "warning", "unassessed"}:
            raise ValueError("Unknown residual dependence status")
        if len(self.parameters) != self.parameter_count:
            raise ValueError("Diagnostic parameter count mismatch")
        if len({p.name for p in self.parameters}) != self.parameter_count:
            raise ValueError("Duplicate diagnostic parameter identity")
        if any(p.precision not in {"supported", "poor", "unavailable"} or p.boundary not in {
            "unconfigured", "outside", "contact", "interval_crosses", "clear"} for p in self.parameters):
            raise ValueError("Unknown parameter precision or boundary state")
        if self.correlation and (len(self.correlation) != self.parameter_count
                or any(len(row) != self.parameter_count for row in self.correlation)):
            raise ValueError("Diagnostic correlation matrix shape conflict")
        if self.residual.reference_pvalue is not None and not 0 <= self.residual.reference_pvalue <= 1:
            raise ValueError("Invalid residual reference p-value")
        json.dumps(asdict(self), allow_nan=False)

    @property
    def reliable(self):
        """Qualified conditional local inference, not a claim of physical truth."""
        return (self.point_estimates_usable and not self.blocking_reasons
                and self.interval_status == "conditional_iid")

    @property
    def status(self):
        if not self.point_estimates_usable:
            return "incomplete" if self.optimizer_status != "completed" else "poor"
        return "supported" if self.reliable and not self.warnings else "warning"

    @property
    def summary(self):
        usable = "usable" if self.point_estimates_usable else "not supported"
        return (f"Diagnostics v2: point estimates {usable}; adequacy {self.adequacy_status}; "
                f"parameter precision {self.identifiability_status}; local CI {self.interval_label}")

    @property
    def interval_label(self):
        return interval_label(self.interval_status)


def interval_label(status):
    return {"conditional_iid":"conditional on independent Gaussian errors",
            "assumptions_unverified":"error assumptions unverified",
            "boundary_limited":"limited by parameter bounds",
            "unavailable":"unavailable"}.get(status, str(status))


def evaluation_from_dict(value):
    """Read saved evidence exactly; opening a project never re-evaluates it."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Diagnostic evaluation must be an object")
    payload = dict(value)
    def sequence(value):
        if not isinstance(value, (tuple, list)):
            raise ValueError("Diagnostic evidence arrays must be arrays")
        return tuple(value)
    payload["policy"] = EvaluationPolicy(**dict(payload["policy"]))
    residual = dict(payload["residual"])
    residual["acf"] = sequence(residual["acf"])
    payload["residual"] = ResidualEvidence(**residual)
    payload["parameters"] = tuple(ParameterEvidence(**{**p, "reasons": sequence(p["reasons"])})
                                  for p in sequence(payload["parameters"]))
    payload["correlation"] = tuple(sequence(row) for row in sequence(payload["correlation"]))
    for key in ("blocking_reasons", "warnings"):
        payload[key] = sequence(payload[key])
    return DiagnosticEvaluation(**payload)


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def residual_evidence(residuals, times=None, policy=None):
    policy = policy or EvaluationPolicy()
    values = np.asarray(residuals, dtype=float).reshape(-1)
    n = values.size
    if not np.all(np.isfinite(values)):
        raise ValueError("Residuals must be finite")
    spacing, median = "unknown", None
    if times is not None:
        ts = np.asarray(times, dtype=float)
        if ts.shape != values.shape or not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
            raise ValueError("Diagnostic times must be finite, increasing and aligned")
        if n > 1:
            dt = np.diff(ts)
            median = float(np.median(dt))
            spacing = ("regular" if np.all(np.abs(dt / median - 1) <= policy.spacing_relative_tolerance)
                       else "irregular")
    # Scaling before centering prevents overflow/underflow and removes any
    # absolute machine-epsilon cutoff in signal units.
    scale = float(np.max(np.abs(values))) if n else 0.
    centred = values / scale if scale else values.copy()
    centred = centred - np.mean(centred) if n else centred
    energy = float(centred @ centred)
    h = min(policy.maximum_lag, n // 5, max(0, n - 1))
    count = max(1, h) if n >= 3 else 0
    acf = tuple(float(centred[:-k] @ centred[k:] / energy)
                for k in range(1, count + 1)) if energy > 0 else ()
    lag1 = acf[0] if acf else None
    band = float(norm.ppf(1 - policy.residual_alpha / 2) / np.sqrt(n)) if n else None
    simultaneous = float(norm.ppf(1 - policy.residual_alpha / (2 * h)) / np.sqrt(n)) if h else None
    q = p = None
    dependence = "unassessed"
    if not acf:
        explanation = "Constant or fewer than three residuals; serial dependence is unassessed."
    elif spacing != "regular":
        # Ordering still exposes strong consecutive-observation patterns. Use
        # the sample-size-aware simultaneous iid band as a descriptive screen,
        # without assigning a regular-time p-value or a correlation timescale.
        index_pattern = n >= policy.minimum_test_count and simultaneous is not None and max(map(abs, acf)) > simultaneous
        dependence = "warning" if index_pattern else "unassessed"
        explanation = (f"{spacing.capitalize()} time spacing: index ACF is descriptive only; "
                       + ("a consecutive-observation pattern exceeds the simultaneous reference band; " if index_pattern else "")
                       + "no regular-time reference p-value, physical correlation timescale or calibrated CI is asserted.")
    elif n < policy.minimum_test_count:
        explanation = f"n={n} < {policy.minimum_test_count}; insufficient data for the reference dependence screen."
    else:
        q = float(n * (n + 2) * sum(r * r / (n - k) for k, r in enumerate(acf, 1)))
        p = float(chi2.sf(q, h))
        dependence = "warning" if p < policy.residual_alpha else "not_detected"
        explanation = (f"Ljung-Box reference Q({h})={q:.4g}, p={p:.4g}, alpha={policy.residual_alpha:g}; "
                       "exploratory after nonlinear fitting (not ECR-calibrated). "
                       "No detected dependence does not establish independent Gaussian errors.")
    return ResidualEvidence(n, lag1, acf, spacing, median, band, simultaneous,
                            q, p, dependence, explanation)


def _local_information(fit, policy):
    """SVD in unit-column coordinates; transform covariance back exactly."""
    n, p = fit.residuals.size, len(fit.params)
    if fit.jacobian is None:
        return None, None, None, None, None
    j = np.asarray(fit.jacobian, dtype=float)
    norms = np.linalg.norm(j, axis=0)
    safe = np.where(norms > 0, norms, 1.)
    z = j / safe
    _, s, vh = np.linalg.svd(z, full_matrices=False)
    tolerance = np.finfo(float).eps * max(z.shape) * (s[0] if s.size else 0)
    rank = int(np.count_nonzero(s > tolerance))
    raw_s = np.linalg.svd(j, compute_uv=False)
    raw = _finite((raw_s[0] / raw_s[-1]) ** 2) if raw_s.size and raw_s[-1] > 0 else None
    scaled = _finite((s[0] / s[-1]) ** 2) if rank == p and s[-1] > 0 else None
    if rank < p or n <= p:
        return rank, raw, scaled, None, None
    variance = float(fit.residuals @ fit.residuals) / (n - p)
    # Avoid forming J.T@J: that squares its condition number and can truncate
    # useful directions in pinv solely because another column uses other units.
    inverse = (vh.T / s) / safe[:, None]
    covariance = variance * (inverse @ inverse.T)
    if not np.all(np.isfinite(covariance)):
        return rank, raw, scaled, None, None
    se = np.sqrt(np.maximum(0., np.diag(covariance)))
    denom = np.outer(se, se)
    correlation = np.divide(covariance, denom, out=np.full_like(covariance, np.nan), where=denom > 0)
    return rank, raw, scaled, se, correlation


def _boundary(estimate, name, space, raw_bound, lower_ci, upper_ci, policy):
    if raw_bound is None:
        return "unconfigured", None, None, None, "not configured"
    lower, upper = raw_bound
    if estimate < lower or estimate > upper:
        return "outside", None, None, None, "physical"
    if space == "log" and estimate > 0:
        q = log10(estimate)
        dl = q - log10(lower) if lower > 0 and isfinite(lower) else None
        du = log10(upper) - q if upper > 0 and isfinite(upper) else None
        contact_lower = dl is not None and dl <= policy.log_contact_tolerance
        contact_upper = du is not None and du <= policy.log_contact_tolerance
        basis = "log10 decades"
    else:
        dl, du = _finite(estimate - lower), _finite(upper - estimate)
        # Each edge uses its own magnitude; the opposite edge never contributes.
        contact_lower = dl is not None and dl <= policy.linear_contact_rtol * max(abs(estimate), abs(lower), np.finfo(float).tiny)
        contact_upper = du is not None and du <= policy.linear_contact_rtol * max(abs(estimate), abs(upper), np.finfo(float).tiny)
        basis = "physical units"
    side = "lower" if contact_lower else "upper" if contact_upper else None
    if side:
        return "contact", side, dl, du, basis
    if lower_ci is not None and upper_ci is not None and (lower_ci < lower or upper_ci > upper):
        return "interval_crosses", None, dl, du, basis
    return "clear", None, dl, du, basis


def evaluate_fit(fit: FitResult, *, times=None, bounds=None, loss="linear", policy=None):
    policy = policy or EvaluationPolicy()
    residual = residual_evidence(fit.residuals, times, policy)
    n, p = fit.residuals.size, len(fit.params)
    rank, raw, scaled, se, corr = _local_information(fit, policy)
    blockers, warnings = [], []
    complete = fit.exit_status > 0
    if not complete:
        blockers.append("OPTIMIZER_NOT_CONVERGED: point estimate has no successful native termination.")
    span = float(np.ptp(fit.y_fit))
    rmse = float(np.linalg.norm(fit.residuals) / np.sqrt(n))
    nrmse = rmse / span if span > 0 else None
    adequacy = ("poor" if not complete or nrmse is None or nrmse > policy.adequacy_poor_nrmse
                else "caution" if nrmse > policy.adequacy_caution_nrmse else "supported")
    if adequacy != "supported":
        warnings.append(f"FIT_ADEQUACY_{adequacy.upper()}: residual RMSE / fitted span={nrmse}; engineering precision criterion.")
    if residual.dependence == "warning":
        warnings.append("RESIDUAL_DEPENDENCE: " + residual.explanation)
    elif residual.dependence == "unassessed":
        warnings.append("RESIDUAL_DEPENDENCE_UNASSESSED: " + residual.explanation)
    if loss != "linear":
        warnings.append("ROBUST_CI_UNCALIBRATED: ordinary local Gaussian intervals do not calibrate robust inference.")
    if rank is None:
        blockers.append("MISSING_UNCERTAINTY: no stored Jacobian; local parameter precision is unavailable.")
    elif rank < p:
        blockers.append(f"JACOBIAN_RANK_DEFICIENT: scaled rank={rank}/{p}.")
    elif scaled is None or scaled > policy.scaled_condition_limit:
        blockers.append(f"SCALED_INFORMATION_ILL_CONDITIONED: cond(Z^T Z)={scaled}; Z has unit column norms.")
    if n <= p:
        blockers.append("NONPOSITIVE_DEGREES_OF_FREEDOM: local covariance is unavailable.")
    if raw is not None and raw > policy.scaled_condition_limit:
        warnings.append("RAW_INFORMATION_SCALE: large raw condition number; use scaled rank, sensitivity and CI jointly.")
    if corr is not None:
        high = [(a, b) for a in range(p) for b in range(a + 1, p)
                if np.isfinite(corr[a, b]) and abs(corr[a, b]) > policy.correlation_warning]
        if high:
            names = tuple(fit.params)
            warnings.append("PARAMETER_CORRELATION: " + "; ".join(
                f"{names[a]}/{names[b]} |r|={abs(corr[a,b]):.4g}" for a, b in high)
                + "; inspect joint precision; correlation alone does not reject the estimates.")
    critical = float(student_t.ppf((1 + policy.confidence_level) / 2, n - p)) if n > p else None
    best_sse = min([fit.SSE] + [c.SSE for c in fit.candidates if c.success and c.exit_status > 0])
    comparable = [c for c in fit.candidates if c.success and c.exit_status > 0
                  and c.SSE <= best_sse * 1.01 + np.finfo(float).tiny]
    if loss != "linear":
        comparable = []
        if fit.candidates:
            warnings.append("MULTISTART_OBJECTIVE_UNASSESSED: raw SSE does not identify comparable robust-loss minima.")
    parameters = []
    for i, (name, estimate) in enumerate(fit.params.items()):
        space = fit.param_space[name]
        reasons, lower_ci, upper_ci, width = [], None, None, None
        precision = "unavailable"
        width_basis = "log10 decades" if space == "log" else "relative to estimate / response amplitude"
        coordinate_se = _finite(se[i]) if se is not None else None
        if not isfinite(estimate) or (space == "log" and estimate <= 0):
            blockers.append(f"PHYSICAL_PARAMETER_INVALID: {name}.")
            reasons.append("invalid parameter value")
        elif coordinate_se is not None and critical is not None:
            half = critical * coordinate_se
            if space == "log":
                lo, hi = log10(estimate) - half, log10(estimate) + half
                width = _finite(2 * half)
                if -307 < lo < 308 and -307 < hi < 308:
                    lower_ci, upper_ci = 10. ** lo, 10. ** hi
                precision = "supported" if width is not None and width <= policy.log_ci_width_limit and lower_ci is not None else "poor"
            else:
                lower_ci, upper_ci = _finite(estimate - half), _finite(estimate + half)
                signal_parameter = name.startswith("A") or name in {"bias", "offset", "y0"}
                reference = max(abs(estimate), span if signal_parameter else 0., np.finfo(float).tiny)
                width = _finite(2 * half / reference)
                precision = "supported" if width is not None and width <= policy.linear_ci_width_limit else "poor"
                if name.startswith("A") and lower_ci is not None and lower_ci <= 0 <= upper_ci:
                    precision = "poor"
                    reasons.append("component amplitude is not separated from zero")
            if precision == "poor":
                reasons.append("local interval exceeds the recorded precision criterion")
                blockers.append(f"PARAMETER_PRECISION_INSUFFICIENT: {name}.")
        else:
            reasons.append("local interval unavailable; inspect Jacobian rank and degrees of freedom")
        bound = (bounds or {}).get(name)
        boundary, side, dl, du, basis = _boundary(estimate, name, space, bound, lower_ci, upper_ci, policy)
        if boundary == "outside":
            blockers.append(f"PARAMETER_OUTSIDE_BOUND: {name}.")
            reasons.append("estimate lies outside the configured physical bounds")
        elif boundary in {"contact", "interval_crosses"}:
            reasons.append("constraint active or local interval crosses a configured boundary")
            warnings.append(f"PARAMETER_BOUNDARY_LIMITED: {name}; {boundary}; ordinary local interval requires boundary-aware inference.")
        values = [c.params[name] for c in comparable if name in c.params]
        if len(values) > 1:
            spread = (log10(max(values) / min(values)) if space == "log" and min(values) > 0
                      else (max(values) - min(values)) / max(abs(estimate), span, np.finfo(float).tiny))
            if spread > .5:
                warnings.append(f"MULTISTART_PARAMETER_SPREAD: {name} spread={spread:.4g}; "
                                f"{'decades' if space == 'log' else 'response-normalized'} among comparable minima.")
            if spread > 1.:
                precision = "poor"
                reasons.append("comparable MultiStart minima disagree on this parameter")
                blockers.append(f"MULTISTART_PARAMETER_UNSTABLE: {name}.")
        parameters.append(ParameterEvidence(name, float(estimate), lower_ci, upper_ci,
            coordinate_se, width, width_basis, precision, boundary, side, dl, du, basis, tuple(reasons)))
    boundary_limited = any(p.boundary in {"contact", "interval_crosses", "outside"} for p in parameters)
    local_available = se is not None and all(p.ci_lower is not None and p.ci_upper is not None for p in parameters)
    interval_status = ("unavailable" if not local_available else "boundary_limited" if boundary_limited
                       else "assumptions_unverified" if loss != "linear" or residual.dependence != "not_detected"
                       else "conditional_iid")
    if interval_status != "conditional_iid":
        warnings.append(f"LOCAL_CI_{interval_status.upper()}: endpoints are diagnostic local approximations, not calibrated coverage under unsupported assumptions.")
    precision_ok = all(p.precision == "supported" for p in parameters)
    identifiability = ("unavailable" if rank is None else "weak" if not precision_ok or rank < p
                       or scaled is None or scaled > policy.scaled_condition_limit else "caution"
                       if any(w.startswith("PARAMETER_CORRELATION") for w in warnings) or boundary_limited else "supported")
    usable = complete and adequacy != "poor" and not any(
        b.startswith(("PHYSICAL_PARAMETER_INVALID", "PARAMETER_OUTSIDE_BOUND")) for b in blockers)
    return DiagnosticEvaluation(policy, "completed" if complete else "incomplete", usable,
        adequacy, _finite(nrmse), identifiability, interval_status, residual, tuple(parameters),
        tuple(tuple(_finite(v) for v in row) for row in corr) if corr is not None else (),
        raw, scaled, rank, p, n - p, tuple(dict.fromkeys(blockers)), tuple(dict.fromkeys(warnings)))
