"""Application service for independent temperature fits and Global/g-ECR."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log10
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from pyecr.batch.temperature_series import (
    ArrheniusResult,
    BatchFitOutcome,
    TemperatureDataset,
    TemperatureUnit,
    fit_temperature_series,
)
from pyecr.config import AnalysisConfig
from pyecr.diagnostics.model_selection import ParameterBound
from pyecr.fitting.base import FitResult
from pyecr.fitting.global_fit import (
    GlobalCurve,
    GlobalFitConfig,
    GlobalFitResult,
    fit_global_ecr,
)
from pyecr.io.dataset import EcrDataset
from pyecr.models.ids import ModelID
from pyecr.models.rectangular import RectangularKernel
from pyecr.services.analysis import AnalysisService, CancellationToken, ProgressReporter
from pyecr.uncertainty.local import LocalUncertaintyResult, estimate_local_uncertainty

__all__ = [
    "ArrheniusPresentation",
    "BatchAnalysisResult",
    "BatchAnalysisService",
    "BatchDatasetInput",
    "BatchRecordPresentation",
    "GlobalAnalysisResult",
    "GlobalCurvePresentation",
]


_BATCH_MODELS = frozenset(
    {
        ModelID.M0_SINGLE_D,
        ModelID.M1_SINGLE_DK,
        ModelID.M2_DUAL_HO_D,
        ModelID.M3_DUAL_HO_DK,
        ModelID.M7_SINGLE_K,
        ModelID.M8_DUAL_K,
    }
)


@dataclass(frozen=True, slots=True)
class BatchDatasetInput:
    """One immutable GUI/service input shared by Batch and Global workflows."""

    record_id: str
    dataset: EcrDataset
    temperature: float
    temperature_unit: TemperatureUnit | str
    analysis_config: AnalysisConfig | None = None
    source_project: str | None = None
    noise_weight: float = 1.0
    fit_baseline: bool = True
    fit_amplitude: bool = True
    baseline: float | None = None
    amplitude: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, EcrDataset):
            raise TypeError("dataset must be an EcrDataset")
        validated = TemperatureDataset(
            self.record_id,
            self.temperature,
            self.temperature_unit,
            self.dataset.processed_time,
            self.dataset.processed_signal,
        )
        if self.analysis_config is not None:
            if not isinstance(self.analysis_config, AnalysisConfig):
                raise TypeError("analysis_config must be AnalysisConfig or None")
            self.analysis_config.validate()
        weight = _positive("noise_weight", self.noise_weight)
        if not isinstance(self.fit_baseline, bool) or not isinstance(
            self.fit_amplitude, bool
        ):
            raise TypeError("fit_baseline and fit_amplitude must be boolean")
        baseline = _optional_finite("baseline", self.baseline)
        amplitude = _optional_finite("amplitude", self.amplitude)
        source = None if self.source_project is None else str(self.source_project)
        object.__setattr__(self, "record_id", validated.record_id)
        object.__setattr__(self, "temperature", validated.temperature)
        object.__setattr__(self, "temperature_unit", validated.temperature_unit)
        object.__setattr__(self, "noise_weight", weight)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "source_project", source)

    @property
    def temperature_K(self) -> float:
        if self.temperature_unit is TemperatureUnit.KELVIN:
            return self.temperature
        return self.temperature + 273.15


@dataclass(frozen=True, slots=True)
class BatchRecordPresentation:
    record_id: str
    temperature_K: float
    model_id: str
    fit_success: bool
    reliable: bool
    reliability_reasons: tuple[str, ...]
    error: str | None
    params: Mapping[str, float]
    parameter_ci: Mapping[str, tuple[float, float] | None]
    SSE: float | None
    RMSE: float | None
    R2: float | None
    fit_result: FitResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reliability_reasons", tuple(self.reliability_reasons))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(
            self,
            "parameter_ci",
            MappingProxyType(
                {
                    str(name): None if interval is None else tuple(interval)
                    for name, interval in self.parameter_ci.items()
                }
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "temperature_K": self.temperature_K,
            "model_id": self.model_id,
            "fit_success": self.fit_success,
            "reliable": self.reliable,
            "reliability_reasons": list(self.reliability_reasons),
            "error": self.error,
            "params": dict(self.params),
            "parameter_ci": {
                name: None if interval is None else list(interval)
                for name, interval in self.parameter_ci.items()
            },
            "SSE": self.SSE,
            "RMSE": self.RMSE,
            "R2": self.R2,
            "fit_result": None if self.fit_result is None else self.fit_result.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BatchRecordPresentation:
        ci = _mapping(value.get("parameter_ci", {}), "parameter_ci")
        fit = value.get("fit_result")
        return cls(
            record_id=str(value["record_id"]),
            temperature_K=float(value["temperature_K"]),
            model_id=str(value["model_id"]),
            fit_success=bool(value["fit_success"]),
            reliable=bool(value["reliable"]),
            reliability_reasons=tuple(str(item) for item in value.get("reliability_reasons", ())),  # type: ignore[arg-type]
            error=None if value.get("error") is None else str(value["error"]),
            params={str(name): float(item) for name, item in _mapping(value.get("params", {}), "params").items()},
            parameter_ci={
                str(name): (
                    None
                    if interval is None or any(v is None or not isfinite(float(v)) for v in interval)
                    else (float(interval[0]), float(interval[1]))  # type: ignore[index]
                )
                for name, interval in ci.items()
            },
            SSE=_optional_float(value.get("SSE")),
            RMSE=_optional_float(value.get("RMSE")),
            R2=_optional_float(value.get("R2")),
            fit_result=(
                None
                if fit is None
                else FitResult.from_dict(_mapping(fit, "fit_result"))
            ),
        )


@dataclass(frozen=True, slots=True)
class ArrheniusPresentation:
    key: str
    model_id: str
    parameter: str
    n_points: int
    slope: float
    intercept: float
    activation_energy_J_mol: float
    activation_energy_CI_J_mol: tuple[float, float]
    pre_exponential_factor: float
    R2: float
    confidence_level: float

    @classmethod
    def from_core(cls, key: str, value: ArrheniusResult) -> ArrheniusPresentation:
        return cls(
            key=key,
            model_id=value.model_id.value,
            parameter=value.parameter,
            n_points=value.n_points,
            slope=value.slope,
            intercept=value.intercept,
            activation_energy_J_mol=value.activation_energy_J_mol,
            activation_energy_CI_J_mol=value.activation_energy_CI_J_mol,
            pre_exponential_factor=value.pre_exponential_factor,
            R2=value.R2,
            confidence_level=value.confidence_level,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "model_id": self.model_id,
            "parameter": self.parameter,
            "n_points": self.n_points,
            "slope": self.slope,
            "intercept": self.intercept,
            "activation_energy_J_mol": self.activation_energy_J_mol,
            "activation_energy_CI_J_mol": list(self.activation_energy_CI_J_mol),
            "pre_exponential_factor": self.pre_exponential_factor,
            "R2": self.R2,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ArrheniusPresentation:
        interval = value["activation_energy_CI_J_mol"]
        return cls(
            key=str(value["key"]),
            model_id=str(value["model_id"]),
            parameter=str(value["parameter"]),
            n_points=int(value["n_points"]),
            slope=float(value["slope"]),
            intercept=float(value["intercept"]),
            activation_energy_J_mol=float(value["activation_energy_J_mol"]),
            activation_energy_CI_J_mol=(float(interval[0]), float(interval[1])),  # type: ignore[index]
            pre_exponential_factor=float(value["pre_exponential_factor"]),
            R2=float(value["R2"]),
            confidence_level=float(value["confidence_level"]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class BatchAnalysisResult:
    model_id: str
    records: tuple[BatchRecordPresentation, ...]
    arrhenius: Mapping[str, ArrheniusPresentation]
    warnings: tuple[str, ...]
    reproducible_csv: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "arrhenius", MappingProxyType(dict(self.arrhenius)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BatchAnalysisResult) and self.to_dict() == other.to_dict()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "records": [record.to_dict() for record in self.records],
            "arrhenius": {
                key: value.to_dict() for key, value in self.arrhenius.items()
            },
            "warnings": list(self.warnings),
            "reproducible_csv": self.reproducible_csv,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BatchAnalysisResult:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported BatchAnalysisResult schema_version")
        records = value.get("records", ())
        arrhenius = _mapping(value.get("arrhenius", {}), "arrhenius")
        return cls(
            model_id=str(value["model_id"]),
            records=tuple(
                BatchRecordPresentation.from_dict(_mapping(item, "batch record"))
                for item in records  # type: ignore[union-attr]
            ),
            arrhenius={
                str(key): ArrheniusPresentation.from_dict(
                    _mapping(item, "Arrhenius result")
                )
                for key, item in arrhenius.items()
            },
            warnings=tuple(str(item) for item in value.get("warnings", ())),  # type: ignore[arg-type]
            reproducible_csv=str(value.get("reproducible_csv", "")),
        )


@dataclass(frozen=True, slots=True)
class GlobalCurvePresentation:
    curve_id: str
    geometry: Mapping[str, object]
    noise_weight: float
    baseline: float | None
    amplitude: float | None
    y_fit: tuple[float, ...]
    residuals: tuple[float, ...]
    weighted_residuals: tuple[float, ...]
    component_amplitudes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", MappingProxyType(dict(self.geometry)))
        object.__setattr__(self, "y_fit", tuple(self.y_fit))
        object.__setattr__(self, "residuals", tuple(self.residuals))
        object.__setattr__(self, "weighted_residuals", tuple(self.weighted_residuals))
        object.__setattr__(self, "component_amplitudes", MappingProxyType(dict(self.component_amplitudes)))

    def to_dict(self) -> dict[str, object]:
        return {
            "curve_id": self.curve_id,
            "geometry": _thaw(self.geometry),
            "noise_weight": self.noise_weight,
            "baseline": self.baseline,
            "amplitude": self.amplitude,
            "y_fit": list(self.y_fit),
            "residuals": list(self.residuals),
            "weighted_residuals": list(self.weighted_residuals),
            **({"component_amplitudes": dict(self.component_amplitudes)} if self.component_amplitudes else {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GlobalCurvePresentation:
        return cls(
            curve_id=str(value["curve_id"]),
            geometry=_mapping(value["geometry"], "geometry"),
            noise_weight=float(value["noise_weight"]),
            baseline=_optional_float(value.get("baseline")),
            amplitude=_optional_float(value.get("amplitude")),
            y_fit=tuple(float(item) for item in value["y_fit"]),  # type: ignore[union-attr]
            residuals=tuple(float(item) for item in value["residuals"]),  # type: ignore[union-attr]
            weighted_residuals=tuple(float(item) for item in value["weighted_residuals"]),  # type: ignore[union-attr]
            component_amplitudes={str(k): float(v) for k, v in _mapping(value.get("component_amplitudes", {}), "component_amplitudes").items()},
        )


@dataclass(frozen=True, slots=True, eq=False)
class GlobalAnalysisResult:
    fit_result: FitResult
    reliable: bool
    condition_number: float
    rank: int
    shared_params: Mapping[str, float]
    shared_confidence_intervals: Mapping[str, tuple[float, float] | None]
    curves: tuple[GlobalCurvePresentation, ...]
    global_covariance: tuple[tuple[float, ...], ...]
    global_correlation: tuple[tuple[float, ...], ...]
    diagnostics: Mapping[str, object]
    warnings: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_params", MappingProxyType(dict(self.shared_params)))
        object.__setattr__(
            self,
            "shared_confidence_intervals",
            MappingProxyType(dict(self.shared_confidence_intervals)),
        )
        object.__setattr__(self, "curves", tuple(self.curves))
        object.__setattr__(self, "global_covariance", tuple(tuple(row) for row in self.global_covariance))
        object.__setattr__(self, "global_correlation", tuple(tuple(row) for row in self.global_correlation))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GlobalAnalysisResult) and self.to_dict() == other.to_dict()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fit_result": self.fit_result.to_dict(),
            "reliable": self.reliable,
            "condition_number": self.condition_number,
            "rank": self.rank,
            "shared_params": dict(self.shared_params),
            "shared_confidence_intervals": {
                name: None if interval is None else list(interval)
                for name, interval in self.shared_confidence_intervals.items()
            },
            "curves": [curve.to_dict() for curve in self.curves],
            "global_covariance": [list(row) for row in self.global_covariance],
            "global_correlation": [list(row) for row in self.global_correlation],
            "diagnostics": _thaw(self.diagnostics),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GlobalAnalysisResult:
        if value.get("schema_version") != 1:
            raise ValueError("unsupported GlobalAnalysisResult schema_version")
        intervals = _mapping(
            value.get("shared_confidence_intervals", {}),
            "shared_confidence_intervals",
        )
        return cls(
            fit_result=FitResult.from_dict(_mapping(value["fit_result"], "fit_result")),
            reliable=bool(value["reliable"]),
            condition_number=float("inf") if value["condition_number"] is None else float(value["condition_number"]),
            rank=int(value["rank"]),
            shared_params={
                str(name): float(item)
                for name, item in _mapping(value["shared_params"], "shared_params").items()
            },
            shared_confidence_intervals={
                str(name): (None if interval is None or any(v is None or not isfinite(float(v)) for v in interval)
                            else (float(interval[0]), float(interval[1])))  # type: ignore[index]
                for name, interval in intervals.items()
            },
            curves=tuple(
                GlobalCurvePresentation.from_dict(_mapping(item, "global curve"))
                for item in value.get("curves", ())  # type: ignore[union-attr]
            ),
            global_covariance=tuple(
                tuple(float("nan") if item is None else float(item) for item in row)
                for row in value["global_covariance"]  # type: ignore[union-attr]
            ),
            global_correlation=tuple(
                tuple(float("nan") if item is None else float(item) for item in row)
                for row in value["global_correlation"]  # type: ignore[union-attr]
            ),
            diagnostics=_mapping(value.get("diagnostics", {}), "diagnostics"),
            warnings=tuple(str(item) for item in value.get("warnings", ())),  # type: ignore[arg-type]
        )


class BatchAnalysisService:
    """Connect Batch and Global core APIs to the production analysis dispatcher."""

    def __init__(self, *, analysis_service: AnalysisService | None = None) -> None:
        self.analysis_service = analysis_service or AnalysisService()

    @staticmethod
    def supports_independent(model_id: ModelID) -> bool:
        return model_id in _BATCH_MODELS

    @staticmethod
    def supports_global(model_id: ModelID) -> bool:
        return model_id in {ModelID.M1_SINGLE_DK, ModelID.M3_DUAL_HO_DK}

    @staticmethod
    def global_unavailable_reason(entries: Sequence[BatchDatasetInput], config: AnalysisConfig | None) -> str:
        """Explain settings the shared ideal-step D+k solver cannot honor."""
        if config is None:
            return "Open Analysis Model and apply a template."
        if not BatchAnalysisService.supports_global(config.model_id):
            return "Global requires 03 M2_SINGLE_DK or 06 M5_DUAL_H2O_DK."
        kinetic = {"D_cm2_s", "k_cm_s", "D_H_cm2_s", "D_O_cm2_s", "k_H_cm_s", "k_O_cm_s"}
        if "fixed_d" in config.model_specific or set(config.model.fixed_parameters) & kinetic:
            return "Global fits both D and k; fixed kinetic parameters are unsupported."
        if config.boundary.mode != "ideal_step":
            return "Global supports Ideal step only; the applied template uses " + config.boundary.mode + "."
        for entry in entries:
            source = entry.analysis_config
            if source is not None:
                if source.model_id is not config.model_id:
                    return entry.record_id + ": captured model must match the applied Global template (03 or 06)."
                if source.boundary.mode != "ideal_step":
                    return entry.record_id + ": Global cannot apply the captured " + source.boundary.mode + " boundary."
                if "fixed_d" in source.model_specific or set(source.model.fixed_parameters) & kinetic:
                    return entry.record_id + ": conditional fixed-D Global is unsupported."
            if entry.dataset.metadata.get("time_unit", "s") != "s":
                return entry.record_id + ": convert time to seconds, then add the input again."
        if len(entries) < 2:
            return "Add at least two compatible inputs."
        # Numerical tolerance only covers Celsius/Kelvin roundoff. This solver
        # has no temperature dependence: one D and one k describe every curve.
        if any(not np.isclose(e.temperature_K,entries[0].temperature_K,rtol=0.,atol=1e-6) for e in entries[1:]):
            return "Global shares D and k at one temperature. Use Independent for a temperature series."
        return ""

    def run_independent(
        self,
        entries: Sequence[BatchDatasetInput],
        config_template: AnalysisConfig,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
        minimum_arrhenius_points: int = 3,
    ) -> BatchAnalysisResult:
        values = _entries(entries)
        config_template.validate()
        if "fixed_d" in config_template.model_specific:
            raise ValueError("Conditional fixed-D Batch Unsupported until source/temperature comparison is validated")
        if not self.supports_independent(config_template.model_id):
            raise ValueError("Independent supports models 01 through 06")
        token = cancel_token or CancellationToken()
        report = progress_callback or _ignore_progress
        token.raise_if_cancelled()
        ordered = sorted(values, key=lambda item: (item.temperature_K, item.record_id))
        index_by_id = {item.record_id: index for index, item in enumerate(ordered)}
        input_by_id = {item.record_id: item for item in values}
        temperature_datasets = tuple(
            TemperatureDataset(
                item.record_id,
                item.temperature,
                item.temperature_unit,
                item.dataset.processed_time,
                item.dataset.processed_signal,
            )
            for item in values
        )

        def fit_one(value: TemperatureDataset) -> BatchFitOutcome:
            token.raise_if_cancelled()
            item = input_by_id[value.record_id]
            index = index_by_id[value.record_id]
            count = len(values)

            def nested_progress(fraction: float, message: str) -> None:
                report(
                    0.03 + 0.82 * (index + fraction) / count,
                    f"{value.record_id}: {message}",
                )

            fit_result = self.analysis_service.fit(
                item.dataset,
                config_template,
                cancel_token=token,
                progress_callback=nested_progress,
            )
            uncertainty = _optional_uncertainty(fit_result)
            surface_key = {ModelID.M7_SINGLE_K: "surface_k", ModelID.M8_DUAL_K: "dual_surface_k"}.get(config_template.model_id)
            actual_bounds = (fit_result.diagnostics[surface_key]["parameter_bounds"]
                             if surface_key is not None
                             else config_template.parameter_bounds)
            bounds = {
                name: ParameterBound(
                    log10(pair[0]),
                    log10(pair[1]),
                    space="coordinate",
                )
                for name, pair in actual_bounds.items()
                if name in fit_result.params
                and fit_result.param_space[name] == "log"
                and pair[0] > 0.0
            }
            return BatchFitOutcome(
                fit_result=fit_result,
                uncertainty=uncertainty,
                parameter_bounds=bounds,
                physical_checks=({surface_key + "_conditional_ci_available": bool(
                    fit_result.diagnostics[surface_key]["ci_available"])}
                    if surface_key is not None else {}),
            )

        report(0.01, "Starting independent production fits")
        core = fit_temperature_series(
            temperature_datasets,
            fitters={config_template.model_id: fit_one},
            models=(config_template.model_id,),
            minimum_arrhenius_points=minimum_arrhenius_points,
        )
        token.raise_if_cancelled()
        report(0.92, "Preparing reliability-filtered Arrhenius results")
        result = _batch_presentation(core, config_template.model_id)
        report(1.0, "Batch and Arrhenius analysis complete")
        return result

    def run_global(
        self,
        entries: Sequence[BatchDatasetInput],
        config_template: AnalysisConfig,
        *,
        cancel_token: CancellationToken | None = None,
        progress_callback: ProgressReporter | None = None,
    ) -> GlobalAnalysisResult:
        values = _entries(entries, minimum=2)
        config_template.validate()
        token = cancel_token or CancellationToken()
        token.raise_if_cancelled()
        reason = self.global_unavailable_reason(values, config_template)
        if reason:
            raise ValueError(reason)
        report = progress_callback or _ignore_progress
        curves: list[GlobalCurve] = []
        geometries: dict[str, Mapping[str, object]] = {}
        for item in values:
            curve_config = item.analysis_config or config_template
            curve_config.model.geometry.validate()
            geometry = RectangularKernel(
                curve_config.model.geometry.full_dims_cm,
                curve_config.model.geometry.active_dimensions,
            )
            geometries[item.record_id] = curve_config.model.geometry.to_dict()
            curves.append(
                GlobalCurve(
                    curve_id=item.record_id,
                    time_s=item.dataset.processed_time,
                    signal=item.dataset.processed_signal,
                    geometry=geometry,
                    noise_weight=item.noise_weight,
                    fit_baseline=item.fit_baseline,
                    fit_amplitude=item.fit_amplitude,
                    baseline=item.baseline,
                    amplitude=item.amplitude,
                )
            )
        from pyecr.fitting.reproducibility import bind_config,random_job_scope
        config_template,job=bind_config(config_template,section="global_ecr")
        policy = _global_config(config_template)
        with random_job_scope(job):
            if config_template.model_id is ModelID.M3_DUAL_HO_DK:
                from pyecr.fitting.global_dual_ho import fit_global_dual_ho
                core = fit_global_dual_ho(
                    curves, template=config_template, policy=policy,
                    curve_configs={item.record_id: item.analysis_config or config_template for item in values},
                    cancel_check=token.raise_if_cancelled, progress_callback=report,
                )
            else:
                core = fit_global_ecr(
                    curves, config=policy, cancel_check=token.raise_if_cancelled, progress_callback=report,
                )
        token.raise_if_cancelled()
        return _global_presentation(core, curves, geometries)


def _batch_presentation(core, model_id: ModelID) -> BatchAnalysisResult:  # type: ignore[no-untyped-def]
    records: list[BatchRecordPresentation] = []
    for record in core.records:
        outcome = record.outcome
        fit = None if outcome is None else outcome.fit_result
        uncertainty = None if outcome is None else outcome.uncertainty
        ci = {
            name: (
                None
                if uncertainty is None or name not in uncertainty.confidence_intervals
                   or not all(isfinite(v) for v in (uncertainty.confidence_intervals[name].lower, uncertainty.confidence_intervals[name].upper))
                else (
                    uncertainty.confidence_intervals[name].lower,
                    uncertainty.confidence_intervals[name].upper,
                )
            )
            for name in (() if fit is None else fit.params)
        }
        records.append(
            BatchRecordPresentation(
                record_id=record.dataset.record_id,
                temperature_K=record.dataset.temperature_K,
                model_id=record.model_id.value,
                fit_success=fit is not None and fit.exit_status > 0,
                reliable=record.reliability.reliable,
                reliability_reasons=record.reliability.reasons,
                error=record.error,
                params={} if fit is None else fit.params,
                parameter_ci=ci,
                SSE=None if fit is None else fit.SSE,
                RMSE=None if fit is None else fit.RMSE,
                R2=None if fit is None else fit.R2,
                fit_result=fit,
            )
        )
    return BatchAnalysisResult(
        model_id=model_id.value,
        records=tuple(records),
        arrhenius={
            key: ArrheniusPresentation.from_core(key, value)
            for key, value in core.arrhenius.items()
        },
        warnings=core.warnings,
        reproducible_csv=core.reproducible_csv(),
    )


def _global_presentation(
    core: GlobalFitResult,
    curves: Sequence[GlobalCurve],
    geometries: Mapping[str, Mapping[str, object]],
) -> GlobalAnalysisResult:
    return GlobalAnalysisResult(
        fit_result=core.fit_result,
        reliable=(
            core.fit_result.exit_status > 0
            and core.uncertainty.reliable_separation
            and not any(w.startswith(("GLOBAL_BOUND_PROXIMITY", "GLOBAL_LABEL_BOUNDARY", "GLOBAL_WEAK_COMPONENT")) for w in core.warnings)
        ),
        condition_number=core.uncertainty.condition_number,
        rank=core.uncertainty.rank,
        shared_params=core.shared_params,
        shared_confidence_intervals={
            name: (interval.lower, interval.upper) if isfinite(interval.lower) and isfinite(interval.upper) else None
            for name, interval in core.shared_confidence_intervals.items()
        },
        curves=tuple(
            GlobalCurvePresentation(
                curve_id=curve.curve_id,
                geometry=geometries[curve.curve_id],
                noise_weight=curve.noise_weight,
                baseline=core.curve_params[curve.curve_id]["baseline"],
                amplitude=core.curve_params[curve.curve_id]["amplitude"],
                y_fit=tuple(float(value) for value in core.y_fit_by_curve[curve.curve_id]),
                residuals=tuple(float(value) for value in core.residuals_by_curve[curve.curve_id]),
                weighted_residuals=tuple(
                    float(value)
                    for value in core.weighted_residuals_by_curve[curve.curve_id]
                ),
                component_amplitudes={name: core.curve_params[curve.curve_id][name]
                                      for name in ("A_H", "A_O") if name in core.curve_params[curve.curve_id]},
            )
            for curve in curves
        ),
        global_covariance=tuple(tuple(float(value) for value in row) for row in core.global_covariance),
        global_correlation=tuple(tuple(float(value) for value in row) for row in core.global_correlation),
        diagnostics=_thaw(core.diagnostics),
        warnings=core.warnings,
    )


def _global_config(config: AnalysisConfig) -> GlobalFitConfig:
    dual = config.model_id is ModelID.M3_DUAL_HO_DK
    d_bounds = config.parameter_bounds["D_H_cm2_s" if dual else "D_cm2_s"]
    k_bounds = config.parameter_bounds["k_H_cm_s" if dual else "k_cm_s"]
    raw = config.model_specific.get("global_ecr", {})
    if not isinstance(raw, Mapping):
        raise TypeError("model_specific['global_ecr'] must be a mapping")
    allowed = {
        "n_starts",
        "random_seed",
        "derivative_step_log10",
        "confidence_level",
        "condition_number_limit",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("unknown global_ecr options: " + ", ".join(unknown))
    return GlobalFitConfig(
        log10_D_bounds=(log10(d_bounds[0]), log10(d_bounds[1])),
        log10_k_bounds=(log10(k_bounds[0]), log10(k_bounds[1])),
        initial_D_cm2_s=config.model.initial_values.get("D_cm2_s"),
        initial_k_cm_s=config.model.initial_values.get("k_cm_s"),
        n_starts=int(raw.get("n_starts", 8)),
        random_seed=int(raw.get("random_seed", 20260916)),
        max_nfev=config.fit.max_nfev,
        loss=config.fit.loss,
        term_tol=config.solver.relative_tolerance,
        max_terms=config.solver.max_terms,
        derivative_step_log10=float(raw.get("derivative_step_log10", 1e-5)),
        confidence_level=float(raw.get("confidence_level", 0.95)),
        condition_number_limit=float(raw.get("condition_number_limit", 1e12)),
    )


def _entries(
    values: Sequence[BatchDatasetInput],
    *,
    minimum: int = 1,
) -> tuple[BatchDatasetInput, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("entries must be a sequence of BatchDatasetInput")
    result = tuple(values)
    if len(result) < minimum:
        raise ValueError(f"at least {minimum} batch dataset(s) are required")
    if not all(isinstance(item, BatchDatasetInput) for item in result):
        raise TypeError("entries must contain only BatchDatasetInput")
    identifiers = tuple(item.record_id for item in result)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("batch record_id values must be unique")
    return result


def _optional_uncertainty(fit_result: FitResult) -> LocalUncertaintyResult | None:
    if ("surface_k" in fit_result.diagnostics
            and not fit_result.diagnostics["surface_k"].get("ci_available", False)):
        return None  # Do not publish intervals the analytic adapter withheld.
    if fit_result.jacobian is None:
        return None
    try:
        return estimate_local_uncertainty(fit_result)
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return None


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optional_finite(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ignore_progress(fraction: float, message: str) -> None:
    del fraction, message
