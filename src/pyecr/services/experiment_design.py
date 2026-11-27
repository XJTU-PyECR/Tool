"""Deterministic M1 planning scans using production forward and P50 sensitivity.

The Fisher proxy assumes known, independent equal Gaussian noise and fixed
normalized amplitude/baseline. It is conditional local information, never an
observed CI, an adequacy assessment, or an absolute optimum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite, log10
from typing import Callable, Mapping

import numpy as np

from pyecr.config import AnalysisConfig, BoundaryConfig, FitConfig, ModelConfig, SampleGeometry, SolverConfig
from pyecr.diagnostics.sensitivity import SensitivityCurve, SensitivityPresentation, compute_sensitivity
from pyecr.fitting.base import FitResult
from pyecr.models.factory import ModelFactory
from pyecr.models.ids import ModelID
from pyecr.models.single import SingleEcrContext
from pyecr.services.analysis import CancellationToken
from pyecr.services.diagnostics_service import BiDimensionPresentation, _bi_presentation


@dataclass(frozen=True, slots=True)
class ExperimentDesignRequest:
    D_range_cm2_s: tuple[float, float]
    k_range_cm_s: tuple[float, float]
    candidate_geometries: tuple[SampleGeometry, ...]
    durations_s: tuple[float, ...]
    noise_std_normalized: float
    sampling_points: int = 80
    sampling_mode: str = "linear"
    range_samples: int = 3
    boundary: BoundaryConfig = BoundaryConfig()
    solver: SolverConfig = SolverConfig(surface_mode="finite", max_terms=80)

    def __post_init__(self) -> None:
        object.__setattr__(self, "D_range_cm2_s", tuple(self.D_range_cm2_s))
        object.__setattr__(self, "k_range_cm_s", tuple(self.k_range_cm_s))
        object.__setattr__(self, "candidate_geometries", tuple(self.candidate_geometries))
        object.__setattr__(self, "durations_s", tuple(self.durations_s))
        self.validate()

    def validate(self) -> None:
        for name, bounds in (("D", self.D_range_cm2_s), ("k", self.k_range_cm_s)):
            if len(bounds) != 2 or any(isinstance(v, bool) or not isfinite(float(v)) or v <= 0 for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"{name} plausible range must be ordered, positive and finite (equal endpoints allowed)")
            # Keep all finite-difference perturbations in representable physical space.
            if not all(1e-250 <= v <= 1e250 for v in bounds):
                raise ValueError(f"{name} range exceeds numerical representability limits")
        if not self.candidate_geometries or not self.durations_s:
            raise ValueError("Provide candidate full geometries and durations")
        for geometry in self.candidate_geometries:
            if not isinstance(geometry, SampleGeometry):
                raise TypeError("Candidates must be SampleGeometry values in cm")
            geometry.validate()
        if len(set(self.candidate_geometries)) != len(self.candidate_geometries):
            raise ValueError("Duplicate candidate geometries")
        if any(isinstance(t, bool) or not isfinite(float(t)) or t <= 0 for t in self.durations_s):
            raise ValueError("Durations must be positive finite seconds")
        if len(set(self.durations_s)) != len(self.durations_s):
            raise ValueError("Duplicate candidate durations")
        if isinstance(self.noise_std_normalized, bool) or not isfinite(self.noise_std_normalized) or self.noise_std_normalized <= 0:
            raise ValueError("Noise must be positive finite standard deviation in normalized signal units")
        if isinstance(self.sampling_points, bool) or not isinstance(self.sampling_points, int) or not 8 <= self.sampling_points <= 1000:
            raise ValueError("sampling_points must be an integer in [8, 1000]")
        if self.sampling_mode not in {"linear", "log"}:
            raise ValueError("sampling_mode must be linear or log")
        divisor = (self.sampling_points - 1)**(2 if self.sampling_mode == "log" else 1)
        if any(duration / divisor == 0 for duration in self.durations_s):
            raise ValueError("Duration and sampling cannot form a representable increasing time grid")
        for duration in self.durations_s:
            grid = _times(duration, self.sampling_points, self.sampling_mode)
            if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
                raise ValueError("Duration and sampling cannot form a representable increasing time grid")
        if isinstance(self.range_samples, bool) or not isinstance(self.range_samples, int) or not 2 <= self.range_samples <= 5:
            raise ValueError("range_samples must be an integer in [2, 5]")
        if len(self.candidate_geometries) * len(self.durations_s) * self.range_samples**2 > 512:
            raise ValueError("Planning scan exceeds 512 scenarios; reduce candidates or range_samples")
        self.boundary.validate()
        self.solver.validate()
        if self.solver.surface_mode != "finite":
            raise ValueError("D/k planning uses the production finite-surface M1 model")

    def to_dict(self) -> dict[str, object]:
        return {"D_range_cm2_s": list(self.D_range_cm2_s), "k_range_cm_s": list(self.k_range_cm_s),
                "candidate_geometries": [g.to_dict() for g in self.candidate_geometries],
                "durations_s": list(self.durations_s), "noise_std_normalized": self.noise_std_normalized,
                "sampling_points": self.sampling_points, "sampling_mode": self.sampling_mode,
                "range_samples": self.range_samples, "boundary": self.boundary.to_dict(),
                "solver": self.solver.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExperimentDesignRequest:
        return cls(tuple(value["D_range_cm2_s"]), tuple(value["k_range_cm_s"]),
                   tuple(SampleGeometry.from_dict(g) for g in value["candidate_geometries"]),
                   tuple(value["durations_s"]), float(value["noise_std_normalized"]),
                   value["sampling_points"], str(value["sampling_mode"]), value["range_samples"],
                   BoundaryConfig.from_dict(value["boundary"]), SolverConfig.from_dict(value["solver"]))


@dataclass(frozen=True, slots=True)
class DesignScenario:
    D_cm2_s: float
    k_cm_s: float
    available: bool
    bi: tuple[BiDimensionPresentation, ...] = ()
    sensitivity: SensitivityPresentation = SensitivityPresentation(False)
    response_fraction: float | None = None
    amplitude_to_noise: float | None = None
    jacobian_rank: int | None = None
    condition_number: float | None = None
    correlation: float | None = None
    nominal_logD_sd: float | None = None
    nominal_logk_sd: float | None = None
    minimum_information_eigenvalue: float | None = None
    supports_separation: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DesignScenario:
        payload = dict(value)
        payload["bi"] = tuple(BiDimensionPresentation(**row) for row in payload["bi"])
        s = dict(payload["sensitivity"])
        curves = []
        for row in s["curves"]:
            curve = dict(row)
            curve["values"] = tuple(curve["values"])
            curve["normalized_values"] = tuple(curve["normalized_values"])
            curves.append(SensitivityCurve(**curve))
        s["curves"] = tuple(curves)
        payload["sensitivity"] = SensitivityPresentation(**s)
        payload["reasons"] = tuple(payload["reasons"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DesignCandidate:
    candidate_id: str
    geometry: SampleGeometry
    duration_s: float
    time_s: tuple[float, ...]
    scenarios: tuple[DesignScenario, ...]
    bi_ranges: tuple[tuple[str, float, float], ...]
    D_peak_sensitivity_range: tuple[float, float] | None
    k_peak_sensitivity_range: tuple[float, float] | None
    supported_fraction: float
    recommended_region: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"candidate_id": self.candidate_id, "geometry": self.geometry.to_dict(),
                "duration_s": self.duration_s, "time_s": list(self.time_s),
                "scenarios": [s.to_dict() for s in self.scenarios], "bi_ranges": list(self.bi_ranges),
                "D_peak_sensitivity_range": self.D_peak_sensitivity_range,
                "k_peak_sensitivity_range": self.k_peak_sensitivity_range,
                "supported_fraction": self.supported_fraction, "recommended_region": self.recommended_region,
                "reasons": list(self.reasons)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DesignCandidate:
        return cls(str(value["candidate_id"]), SampleGeometry.from_dict(value["geometry"]),
                   float(value["duration_s"]), tuple(value["time_s"]),
                   tuple(DesignScenario.from_dict(s) for s in value["scenarios"]),
                   tuple(tuple(row) for row in value["bi_ranges"]),
                   None if value["D_peak_sensitivity_range"] is None else tuple(value["D_peak_sensitivity_range"]),
                   None if value["k_peak_sensitivity_range"] is None else tuple(value["k_peak_sensitivity_range"]),
                   float(value["supported_fraction"]), bool(value["recommended_region"]), tuple(value["reasons"]))


@dataclass(frozen=True, slots=True)
class RecommendedDesignRegion:
    active_dimensions: tuple[bool, bool, bool]
    full_dimension_ranges_cm: tuple[tuple[float, float], ...]
    duration_range_s: tuple[float, float]
    candidate_ids: tuple[str, ...]
    explanation: str = "Bounds enclose supported sampled candidates; intermediate geometry/duration combinations have not been validated."


@dataclass(frozen=True, slots=True)
class PlanningSnapshot:
    request: ExperimentDesignRequest
    candidates: tuple[DesignCandidate, ...]
    recommended_candidate_ids: tuple[str, ...]
    notes: tuple[str, ...]
    model_id: str = ModelID.M1_SINGLE_DK.value
    schema_version: int = 1
    recommended_regions: tuple[RecommendedDesignRegion, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "model_id": self.model_id, "request": self.request.to_dict(),
                "candidates": [c.to_dict() for c in self.candidates],
                "recommended_candidate_ids": list(self.recommended_candidate_ids), "notes": list(self.notes),
                "recommended_regions": [asdict(r) for r in self.recommended_regions]}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PlanningSnapshot:
        if value.get("schema_version") != 1 or value.get("model_id") != ModelID.M1_SINGLE_DK.value:
            raise ValueError("Unsupported planning snapshot schema or model")
        regions = tuple(RecommendedDesignRegion(tuple(r["active_dimensions"]),
            tuple(tuple(bounds) for bounds in r["full_dimension_ranges_cm"]), tuple(r["duration_range_s"]),
            tuple(r["candidate_ids"]), str(r["explanation"])) for r in value.get("recommended_regions", ()))
        return cls(ExperimentDesignRequest.from_dict(value["request"]),
                   tuple(DesignCandidate.from_dict(c) for c in value["candidates"]),
                   tuple(value["recommended_candidate_ids"]), tuple(value["notes"]), recommended_regions=regions)


def _times(duration: float, count: int, mode: str) -> np.ndarray:
    if mode == "linear":
        return np.linspace(0.0, duration, count)
    return np.concatenate(([0.0], np.geomspace(duration / (count - 1)**2, duration, count - 1)))


def _config(request: ExperimentDesignRequest, geometry: SampleGeometry) -> AnalysisConfig:
    return AnalysisConfig(
        ModelConfig(ModelID.M1_SINGLE_DK, geometry, surface_mode="finite", boundary_mode=request.boundary.mode,
                    free_parameters=("D_cm2_s", "k_cm_s"), fixed_parameters={"amplitude": 1.0, "baseline": 0.0},
                    parameter_bounds={"D_cm2_s": (request.D_range_cm2_s[0] / 10, request.D_range_cm2_s[1] * 10),
                                      "k_cm_s": (request.k_range_cm_s[0] / 10, request.k_range_cm_s[1] * 10)}),
        request.solver, FitConfig(model_id=ModelID.M1_SINGLE_DK), boundary=request.boundary,
        model_specific={"single_ecr": {"normalized_data": True}})


def _scenario(request: ExperimentDesignRequest, config: AnalysisConfig, time: np.ndarray, d: float, k: float) -> DesignScenario:
    from pyecr.io.dataset import EcrDataset

    params = {"D_cm2_s": d, "k_cm_s": k}
    try:
        model = ModelFactory().create(config)
        context = SingleEcrContext(term_tol=config.solver.relative_tolerance, max_terms=config.solver.max_terms)
        y = np.asarray(model.predict(time, {"log10_D": log10(d), "log10_k": log10(k)}, context))
        if y.shape != time.shape or not np.all(np.isfinite(y)):
            raise ValueError("Production forward returned nonfinite or misaligned signal")
        dataset = EcrDataset.from_raw(time, y, signal_type="normalized", signal_unit="1")
        # A forward evaluation carrier, not a fitted experiment or observed CI.
        forward_result = FitResult(params=params, param_space={p: "log" for p in params}, y_fit=y,
                                   residuals=np.zeros_like(y), SSE=0.0, RMSE=0.0, R2=1.0, AIC=0.0, AICc=0.0, BIC=0.0,
                                   exit_status=1, nfev=0, diagnostics={"planning_forward_only": True})
        sensitivity = compute_sensitivity(dataset, config, forward_result)
        bi = _bi_presentation(config, forward_result)
        if not sensitivity.available or not bi.available:
            raise ValueError(sensitivity.reason or bi.reason)
        if not all(isfinite(row.value) for row in bi.rows):
            raise ValueError("Bi calculation exceeds finite numerical range")
        matrix = np.column_stack([c.values for c in sensitivity.curves]) / request.noise_std_normalized
        singular = np.linalg.svd(matrix, compute_uv=False)
        if not np.all(np.isfinite(singular)):
            raise ValueError("Sensitivity information exceeds finite numerical range")
        rank = int(np.count_nonzero(singular > singular[0] * np.finfo(float).eps * max(matrix.shape)))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        if condition is not None and not isfinite(condition):
            condition = None
        sd_d = sd_k = correlation = None
        minimum_eigenvalue = float(singular[-1]**2)
        if not isfinite(minimum_eigenvalue):
            minimum_eigenvalue = None
        if rank == 2:
            covariance = np.linalg.inv(matrix.T @ matrix)
            sd_d, sd_k = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            correlation = float(covariance[0, 1] / (sd_d * sd_k)) if sd_d * sd_k > 0 else None
            sd_d, sd_k = float(sd_d), float(sd_k)
            if not all(isfinite(v) for v in (sd_d, sd_k)):
                sd_d = sd_k = None
        response = float(y[-1] - y[0])
        strength = float(np.ptp(y) / request.noise_std_normalized)
        reasons: list[str] = []
        if all(row.value < 0.03 for row in bi.rows):
            reasons.append("surface_limited: every active-axis Bi < 0.03; D may be weak")
        if all(row.value > 30.0 for row in bi.rows):
            reasons.append("diffusion_limited: every active-axis Bi > 30; k may be weak")
        if response < 0.632:
            reasons.append("too_slow: duration captures <63.2% of the normalized response")
        if y[1] - y[0] > 0.95:
            reasons.append("sampling_too_coarse: first post-zero observation is already >95% response")
        if strength < 5:
            reasons.append("weak_amplitude: predicted observed span / noise < 5")
        if rank < 2 or condition is None or condition > 1e3:
            reasons.append("weak_separation: sensitivity Jacobian rank/condition is unfavorable")
        if correlation is None or abs(correlation) > 0.98:
            reasons.append("high_correlation: conditional D/k information is strongly coupled")
        if sd_d is None or sd_k is None or max(sd_d, sd_k) * 3.92 > 2.0:
            reasons.append("wide_proxy: nominal 95% log-parameter width exceeds 2 decades")
        if any(c.convergence_relative_error > 0.01 for c in sensitivity.curves):
            reasons.append("sensitivity_step: finite-difference convergence needs review")
        return DesignScenario(d, k, True, bi.rows, sensitivity, response, strength, rank, condition, correlation,
                              sd_d, sd_k, minimum_eigenvalue, not reasons, tuple(reasons))
    except (ArithmeticError, FloatingPointError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return DesignScenario(d, k, False, reasons=(f"Production evaluation unavailable: {exc}",))


def scan_experiment_design(request: ExperimentDesignRequest, *, cancel_token: CancellationToken | None = None,
                           progress_callback: Callable[[float, str], None] | None = None) -> PlanningSnapshot:
    """Evaluate a finite plausible-range grid; recommend a region of candidates."""
    request.validate()
    d_values = np.unique(np.geomspace(*request.D_range_cm2_s, request.range_samples))
    k_values = np.unique(np.geomspace(*request.k_range_cm_s, request.range_samples))
    total = len(request.candidate_geometries) * len(request.durations_s) * len(d_values) * len(k_values)
    done = 0
    candidates = []
    for index, (geometry, duration) in enumerate(product(request.candidate_geometries, request.durations_s), 1):
        config = _config(request, geometry)
        time = _times(duration, request.sampling_points, request.sampling_mode)
        scenarios = []
        for d, k in product(d_values, k_values):
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            scenarios.append(_scenario(request, config, time, float(d), float(k)))
            done += 1
            if progress_callback is not None:
                progress_callback(done / total, f"Planning scenario {done}/{total}")
        available = [s for s in scenarios if s.available]
        bi_ranges = []
        for axis in "xyz":
            values = [row.value for s in available for row in s.bi if row.axis == axis]
            if values:
                bi_ranges.append((axis, min(values), max(values)))
        peaks = [[curve.peak_absolute for s in available for curve in s.sensitivity.curves if curve.parameter == name]
                 for name in ("D_cm2_s", "k_cm_s")]
        peak_ranges = [None if not values else (min(values), max(values)) for values in peaks]
        supported = sum(s.supports_separation for s in scenarios) / len(scenarios)
        recommended = supported >= 2 / 3 and len(available) == len(scenarios)
        reasons = tuple(dict.fromkeys(reason for s in scenarios for reason in s.reasons))
        if recommended:
            reasons = ("Recommended region: at least two-thirds of sampled plausible D/k combinations support conditional separation",) + reasons
        candidates.append(DesignCandidate(f"candidate-{index}", geometry, float(duration), tuple(time), tuple(scenarios),
                                          tuple(bi_ranges), peak_ranges[0], peak_ranges[1], supported, recommended, reasons))
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    regions = []
    for active in dict.fromkeys(c.geometry.active_dimensions for c in candidates if c.recommended_region):
        selected = [c for c in candidates if c.recommended_region and c.geometry.active_dimensions == active]
        bounds = tuple((min(c.geometry.full_dims_cm[i] for c in selected), max(c.geometry.full_dims_cm[i] for c in selected)) for i in range(3))
        duration_range = (min(c.duration_s for c in selected), max(c.duration_s for c in selected))
        regions.append(RecommendedDesignRegion(active, bounds, duration_range, tuple(c.candidate_id for c in selected)))
    return PlanningSnapshot(
        request, tuple(candidates), tuple(c.candidate_id for c in candidates if c.recommended_region),
        ("Production M1 forward plus P50 S_logD(t)=∂y/∂log10(D), S_logk(t)=∂y/∂log10(k); default central step 0.001 decades with half-step convergence.",
         "Bi_i=k*(full_dimension_i/2)/D for every active axis; 0.03 and 30 are heuristic regimes, not physical cutoffs.",
         "Information proxy J=S/noise; covariance=(JᵀJ)⁻¹ assumes known equal independent Gaussian noise and fixed amplitude/baseline. Nominal log SD is not an observed fit CI.",
         "Recommended candidates form a sampled region, not a unique absolute optimum. Unsampled plausible values may behave differently; validate with pilot data.",
         "63.2% duration coverage, span/noise≥5, rank=2, condition≤1000, |correlation|≤0.98 and nominal 95% width≤2 decades are heuristic screening rules.",
         "Planning does not assess model adequacy or change the current AnalysisConfig. M2/M3 boundary mathematics are not introduced."),
        recommended_regions=tuple(regions))
