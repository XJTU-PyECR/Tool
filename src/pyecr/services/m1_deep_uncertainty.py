"""Experimental, opt-in M1 residual-bootstrap diagnostics.

This module is deliberately outside FitResult and project serialization.  It
uses the production refit adapter through DiagnosticsService and never changes
the forward model, optimizer defaults, or local confidence intervals.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitResult
from pyecr.io.dataset import EcrDataset
from pyecr.models.ids import ModelID
from pyecr.report.snapshot import dataset_sha256
from pyecr.services.analysis import CancellationToken
from pyecr.services.diagnostics_service import DiagnosticsService
from pyecr.services.guided_refit import fit_result_id
from pyecr.uncertainty import AdvancedUncertaintyConfig

METHOD_VERSION = "p66b-m1-residual-bootstrap-v1"
REQUESTED = 200
MAX_WALL_SECONDS = 90.0
MIN_SUCCESSFUL = 180
ProgressReporter = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class ExperimentalM1UncertaintyResult:
    """Standalone diagnostics record; intervals exist only after full completion."""

    source_fit_id: str
    dataset_sha256: str
    config_sha256: str
    seed: int
    status: str
    n_requested: int
    n_completed: int
    n_successful: int
    n_failed: int
    max_wall_seconds: float
    elapsed_seconds: float
    intervals: Mapping[str, Mapping[str, float]]
    samples: tuple[tuple[float, ...], ...]
    parameter_names: tuple[str, ...]
    evidence: Mapping[str, object]
    warnings: tuple[str, ...]
    failure_messages: tuple[str, ...] = ()
    guided_refit_fixed: Mapping[str, object] | None = None
    method_version: str = METHOD_VERSION
    method: str = "centered_residual_bootstrap_percentile"
    assumption: str = "additive centered exchangeable residuals; time correlation not retained"
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        from pyecr.contracts import ConfidencePolicy
        ConfidencePolicy(confidence_level=self.confidence_level)
        if self.method_version != METHOD_VERSION or self.n_requested != REQUESTED:
            raise ValueError("unsupported experimental uncertainty record version/budget")
        if self.status not in {"complete", "incomplete", "cancelled", "timeout"}:
            raise ValueError("invalid experimental uncertainty status")
        if (self.seed < 0 or self.n_successful < 0 or self.n_failed < 0
                or self.n_completed < 0 or self.n_successful + self.n_failed > self.n_completed
                or self.n_completed > REQUESTED):
            raise ValueError("inconsistent bootstrap counts")
        if self.status != "complete" and self.intervals:
            raise ValueError("partial analysis cannot publish an interval")
        if self.status == "complete" and (self.n_completed != REQUESTED or self.n_successful < MIN_SUCCESSFUL):
            raise ValueError("complete analysis requires the frozen success budget")
        if self.status == "complete" and (
            set(self.intervals) != set(self.parameter_names)
            or len(self.samples) != self.n_successful
            or any(len(row) != len(self.parameter_names) for row in self.samples)
        ):
            raise ValueError("complete analysis requires intervals and successful samples for every free parameter")
        for interval in self.intervals.values():
            lower, upper = float(interval["lower"]), float(interval["upper"])
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
                raise ValueError("experimental interval endpoints must be positive, finite and ordered")

    def to_dict(self) -> dict[str, object]:
        from pyecr.report.snapshot import _json_safe
        return {
            "schema_version": 1, "method": self.method,
            "method_version": self.method_version, "source_fit_id": self.source_fit_id,
            "dataset_sha256": self.dataset_sha256, "config_sha256": self.config_sha256,
            "seed": self.seed, "status": self.status,
            "n_requested": self.n_requested, "n_completed": self.n_completed,
            "n_successful": self.n_successful, "n_failed": self.n_failed,
            "n_unclassified": self.n_completed - self.n_successful - self.n_failed,
            "max_wall_seconds": self.max_wall_seconds, "elapsed_seconds": self.elapsed_seconds,
            "confidence_level": self.confidence_level, "assumption": self.assumption,
            "parameter_names": list(self.parameter_names),
            "intervals": {k: dict(v) for k, v in self.intervals.items()},
            "samples": [list(row) for row in self.samples],
            "evidence": _json_safe(self.evidence),
            "warnings": list(self.warnings),
            "failure_messages": list(self.failure_messages),
            "guided_refit_fixed": None if self.guided_refit_fixed is None else dict(self.guided_refit_fixed),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentalM1UncertaintyResult":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported experimental uncertainty schema")
        if value.get("method") != "centered_residual_bootstrap_percentile":
            raise ValueError("unsupported experimental uncertainty method")
        intervals = value.get("intervals", {})
        evidence = value.get("evidence", {})
        if not isinstance(intervals, Mapping) or not isinstance(evidence, Mapping):
            raise TypeError("intervals and evidence must be mappings")
        fixed = value.get("guided_refit_fixed")
        if fixed is not None and not isinstance(fixed, Mapping):
            raise TypeError("guided_refit_fixed must be a mapping")
        return cls(
            source_fit_id=str(value["source_fit_id"]), dataset_sha256=str(value["dataset_sha256"]),
            config_sha256=str(value["config_sha256"]), seed=int(value["seed"]),
            status=str(value["status"]), n_requested=int(value["n_requested"]),
            n_completed=int(value["n_completed"]), n_successful=int(value["n_successful"]),
            n_failed=int(value["n_failed"]), max_wall_seconds=float(value["max_wall_seconds"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            intervals={str(k): {str(ik): float(iv) for ik, iv in dict(v).items()} for k, v in intervals.items()},
            samples=tuple(tuple(float(item) for item in row) for row in value.get("samples", [])),
            parameter_names=tuple(str(item) for item in value.get("parameter_names", [])),
            evidence=dict(evidence), warnings=tuple(str(item) for item in value.get("warnings", [])),
            failure_messages=tuple(str(item) for item in value.get("failure_messages", [])),
            guided_refit_fixed=None if fixed is None else dict(fixed),
            method_version=str(value["method_version"]), method=str(value["method"]),
            assumption=str(value["assumption"]), confidence_level=float(value["confidence_level"]),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def reopen(cls, path: str | Path, dataset: EcrDataset, config: AnalysisConfig, fit: FitResult) -> "ExperimentalM1UncertaintyResult":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls.from_dict(value)
        result.validate_provenance(dataset, config, fit)
        return result

    def validate_provenance(self, dataset: EcrDataset, config: AnalysisConfig, fit: FitResult) -> None:
        if (self.source_fit_id != fit_result_id(fit)
                or self.dataset_sha256 != dataset_sha256(dataset)
                or self.config_sha256 != config_sha256(config)):
            raise ValueError("experimental uncertainty belongs to another fit, dataset, or config")


def config_sha256(config: AnalysisConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_experimental_m1_bootstrap(
    dataset: EcrDataset, config: AnalysisConfig, fit: FitResult, *, seed: int,
    acknowledge_exchangeable_residuals: bool,
    iid_noise_known_by_design: bool = False,
    service: DiagnosticsService | None = None,
    cancel_token: CancellationToken | None = None,
    progress_callback: ProgressReporter | None = None,
    confidence_level: float = 0.95,
) -> ExperimentalM1UncertaintyResult:
    """Run the frozen 200-refit experiment only after an explicit assumption check."""
    from pyecr.services.advanced_intervals import checked_C
    confidence_level=checked_C(confidence_level)

    if not acknowledge_exchangeable_residuals:
        raise ValueError("explicit exchangeable-residual assumption acknowledgement is required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config.model_id is not ModelID.M1_SINGLE_DK:
        raise ValueError("P66B experimental uncertainty supports M1 only")
    if config.fit.loss != "linear":
        raise ValueError("P66B experimental interval has not been validated for robust loss")
    if config.boundary.mode != "ideal_step":
        raise ValueError("P66B measured/nonideal boundary uncertainty is unverified")
    if fit.exit_status <= 0:
        raise ValueError("a converged parent FitResult is required")
    if tuple(fit.params) != tuple(config.model.free_parameters):
        raise ValueError("FitResult free parameters do not match AnalysisConfig")
    if not set(fit.params)<={'D_cm2_s','k_cm_s'}:
        raise ValueError('Unsupported experimental M1 free observation parameters; use the general conditional bootstrap')
    for name, estimate in fit.params.items():
        bounds = config.parameter_bounds.get(name)
        if bounds and min(np.log10(estimate / bounds[0]), np.log10(bounds[1] / estimate)) <= 0.01:
            raise ValueError(f"{name} approaches a parameter bound; nonregular interval unvalidated")
    residual = np.asarray(fit.residuals, dtype=float)
    if residual.size != dataset.processed_signal.size or residual.size < 5:
        raise ValueError("fit residuals must align with at least five observations")
    if not np.all(np.isfinite(residual)) or float(np.std(residual)) == 0.0:
        raise ValueError("finite, nonconstant residuals are required for resampling")
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    if not np.isfinite(lag1) or (abs(lag1) > 0.3 and not iid_noise_known_by_design):
        raise ValueError("suspected time-correlated residuals; iid bootstrap unvalidated (lag1 screen > 0.3)")

    token = cancel_token or CancellationToken()
    token.raise_if_cancelled()
    started = time.monotonic()
    completed = 0
    fixed = config.model_specific.get("guided_refit")
    fixed_record = dict(fixed) if isinstance(fixed, Mapping) else None
    evidence = _evidence(dataset, config, fit, service or DiagnosticsService())
    evidence["iid_noise_known_by_design"] = iid_noise_known_by_design
    evidence["lag1_screen_flag"] = abs(lag1) > 0.3
    evidence.update(sample_coordinate_definition='FitResult optimizer coordinates; log means log10',
        parameter_spaces=dict(fit.param_space),gate_confidence_level=.95,gate_unchanged=True,
        fixed_parameters=dict(config.model.fixed_parameters),geometry=config.model.geometry.to_dict(),
        fixed_d_conditions=config.model_specific.get('fixed_d'),coverage_calibrated=False)

    def progress(fraction: float, message: str) -> None:
        nonlocal completed
        completed = min(REQUESTED, max(completed, round(fraction * REQUESTED)))
        token.raise_if_cancelled()
        if time.monotonic() - started > MAX_WALL_SECONDS:
            raise TimeoutError("P66B 90 s budget exceeded after the current refit")
        if progress_callback is not None:
            progress_callback(fraction, f"{completed}/{REQUESTED} refits; elapsed {time.monotonic() - started:.1f} s; {message}")

    def record(status: str, *, successful: int = 0, failed: int = 0,
               intervals: Mapping[str, Mapping[str, float]] | None = None,
               samples: tuple[tuple[float, ...], ...] = (), warnings: tuple[str, ...] = (),
               failure_messages: tuple[str, ...] = ()) -> ExperimentalM1UncertaintyResult:
        return ExperimentalM1UncertaintyResult(
            source_fit_id=fit_result_id(fit), dataset_sha256=dataset_sha256(dataset),
            config_sha256=config_sha256(config), seed=seed, status=status,
            n_requested=REQUESTED, n_completed=completed, n_successful=successful,
            n_failed=failed, max_wall_seconds=MAX_WALL_SECONDS,
            elapsed_seconds=time.monotonic() - started, intervals=intervals or {},
            samples=samples, parameter_names=tuple(fit.params), evidence=evidence,
            warnings=warnings, failure_messages=failure_messages,
            guided_refit_fixed=fixed_record,
            confidence_level=confidence_level,
        )

    try:
        presentation = (service or DiagnosticsService()).run_bootstrap(
            dataset, config, fit,
            options=AdvancedUncertaintyConfig(enabled=True, bootstrap_resamples=REQUESTED, bootstrap_n_jobs=1),
            cancel_token=token, progress_callback=progress, random_seed=seed,
            **({'confidence_level':confidence_level} if confidence_level!=.95 else {}),
        )
    except CancelledError:
        return record("cancelled", warnings=("Cancelled; no interval was published.",))
    except TimeoutError:
        return record("timeout", warnings=("Time budget exceeded; no interval was published.",))
    except RuntimeError as exc:
        return record("incomplete", warnings=(f"Bootstrap did not produce a complete result: {exc}",))

    result = presentation.result
    assert result is not None
    completed = result.n_requested
    evidence['bootstrap_provenance']=dict(result.diagnostics)
    evidence['quantile_method']='linear'
    if result.n_successful < MIN_SUCCESSFUL:
        return record("incomplete", successful=result.n_successful,
                      failed=result.n_requested - result.n_successful,
                      warnings=(*result.warnings, "Below 180/200 successful refits; no interval was published."),
                      samples=tuple(tuple(float(x) for x in row) for row in result.coordinate_samples),
                      failure_messages=tuple(result.diagnostics.get("failed_refit_messages", ())))
    intervals = {name: {"estimate": item.estimate, "lower": item.lower,
                        "upper": item.upper, "coordinate_lower": item.coordinate_lower,
                        "coordinate_upper": item.coordinate_upper}
                 for name, item in result.intervals.items()}
    samples = tuple(tuple(float(x) for x in row) for row in result.coordinate_samples)
    if not intervals:
        return record('incomplete',successful=result.n_successful,failed=result.n_requested-result.n_successful,
            samples=samples,warnings=(*result.warnings,str(result.diagnostics.get('output_reason','No supported interval at this C'))))
    notes = list(result.warnings)
    if fixed_record:
        notes.append("Guided Refit: interval is conditional on the fixed value; fixed-value uncertainty is excluded and the fixed parameter has no free CI.")
    return record("complete", successful=result.n_successful,
                  failed=result.n_requested - result.n_successful, intervals=intervals,
                  samples=samples, warnings=tuple(notes),
                  failure_messages=tuple(result.diagnostics.get("failed_refit_messages", ())))


def recompute_experimental_samples(record,fit,confidence_level):
    """Existing coordinate samples only; no residual draws or nuisance refits."""
    from pyecr.services.advanced_intervals import relevel_result
    from pyecr.uncertainty import UncertaintyMethod,UncertaintyResult
    if record.source_fit_id!=fit_result_id(fit) or record.parameter_names!=tuple(fit.params):
        raise ValueError('Experimental saved sample/parent identity conflict')
    if len(record.samples)!=record.n_successful or not record.samples:
        raise ValueError('No complete saved experimental samples; explicitly Run analysis')
    coordinates=np.asarray(record.samples,dtype=float);physical=coordinates.copy()
    for i,name in enumerate(fit.params):
        if fit.param_space[name]=='log':physical[:,i]=10.**physical[:,i]
    result=UncertaintyResult(method=UncertaintyMethod.RESIDUAL_BOOTSTRAP,parameter_names=tuple(fit.params),estimates=fit.params,
        param_space=fit.param_space,intervals={},confidence_level=record.confidence_level,n_requested=record.n_requested,n_successful=record.n_successful,
        physical_samples=physical,coordinate_samples=coordinates,diagnostics=record.evidence.get('bootstrap_provenance',{}))
    result=relevel_result(fit,result,confidence_level,minimum_success_fraction=MIN_SUCCESSFUL/REQUESTED)
    intervals={name:{key:getattr(ci,key) for key in ('estimate','lower','upper','coordinate_lower','coordinate_upper')} for name,ci in result.intervals.items()}
    return replace(record,status='complete' if intervals else 'incomplete',confidence_level=result.confidence_level,intervals=intervals,
        evidence={**record.evidence,'bootstrap_provenance':dict(result.diagnostics),'quantile_method':'linear'},warnings=result.warnings)


def _evidence(dataset: EcrDataset, config: AnalysisConfig, fit: FitResult,
              service: DiagnosticsService) -> dict[str, object]:
    """Keep the advisor's evidence sources separate; no new threshold is inferred."""
    from pyecr.services.advanced_intervals import observation_conditions
    summary = service.describe(dataset, config, fit)
    return {
        **observation_conditions(dataset,config,fit),
        "Bi_heuristic": [{"axis": row.axis, "value": row.value, "regime": row.regime} for row in summary.bi.rows],
        "Jacobian": {"rank": summary.jacobian.rank, "condition": summary.jacobian.condition_number,
                     "reason": summary.jacobian.reason},
        "correlation": {"matrix": summary.correlation.matrix, "reason": summary.correlation.reason},
        "MultiStart_spread": [{"parameter": row.parameter, "spread_decades": row.spread_decades}
                              for row in summary.multistart.rows],
        "Error_Map_valley": "N/A: no current map supplied to this standalone analysis",
        "amplitude": float(np.ptp(dataset.processed_signal)),
        "identifiability": summary.reliability_advisor.identifiability_explanation,
        "model_adequacy": summary.reliability_advisor.model_adequacy_explanation,
        "lag1": summary.residual_summary.lag1_autocorrelation,
    }
