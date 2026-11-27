"""Independent temperature-series fits and reliability-filtered Arrhenius laws."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from pyecr.diagnostics.model_selection import ParameterBound, select_model
from pyecr.fitting.base import FitResult
from pyecr.models.ids import ModelID
from pyecr.uncertainty.local import LocalUncertaintyResult

__all__ = [
    "GAS_CONSTANT_J_MOL_K",
    "ArrheniusResult",
    "BatchFitOutcome",
    "BatchFitRecord",
    "ModelFitter",
    "ReliabilityDecision",
    "ReliabilityFilter",
    "TemperatureDataset",
    "TemperatureSeriesResult",
    "TemperatureUnit",
    "fit_temperature_series",
]


GAS_CONSTANT_J_MOL_K = 8.31446261815324
"""Molar gas constant used for activation energies, in J mol^-1 K^-1."""


class TemperatureUnit(str, Enum):
    """Accepted explicit temperature units."""

    KELVIN = "K"
    CELSIUS = "degC"


_DEFAULT_MODELS = (
    ModelID.M0_SINGLE_D,
    ModelID.M1_SINGLE_DK,
    ModelID.M2_DUAL_HO_D,
)
_SUPPORTED_MODELS = (
    *_DEFAULT_MODELS,
    ModelID.M3_DUAL_HO_DK,
    ModelID.M7_SINGLE_K,
    ModelID.M8_DUAL_K,
)
_MODEL_PARAMETERS: Mapping[ModelID, tuple[str, ...]] = MappingProxyType(
    {
        ModelID.M0_SINGLE_D: ("D_cm2_s",),
        ModelID.M1_SINGLE_DK: ("D_cm2_s", "k_cm_s"),
        ModelID.M2_DUAL_HO_D: ("D_H_cm2_s", "D_O_cm2_s"),
        ModelID.M3_DUAL_HO_DK: ("D_H_cm2_s", "D_O_cm2_s", "k_H_cm_s", "k_O_cm_s"),
        ModelID.M7_SINGLE_K: ("k_cm_s",),
        ModelID.M8_DUAL_K: ("k_1_cm_s", "k_2_cm_s"),
    }
)
_OUTPUT_PARAMETERS = (
    "D_cm2_s",
    "k_cm_s",
    "D_H_cm2_s",
    "D_O_cm2_s",
    "k_H_cm_s",
    "k_O_cm_s",
    "k_1_cm_s",
    "k_2_cm_s",
)


@dataclass(frozen=True, slots=True)
class TemperatureDataset:
    """One ECR record with an explicit experimental temperature unit."""

    record_id: str
    temperature: float
    temperature_unit: TemperatureUnit | str
    time_s: ArrayLike
    signal: ArrayLike
    context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be a nonempty string")
        temperature = _finite_scalar("temperature", self.temperature)
        unit = _coerce_temperature_unit(self.temperature_unit)
        temperature_k = temperature if unit is TemperatureUnit.KELVIN else temperature + 273.15
        if temperature_k <= 0.0:
            raise ValueError("temperature must be above absolute zero")
        times = _vector("time_s", self.time_s)
        signal = _vector("signal", self.signal)
        if times.size != signal.size:
            raise ValueError("time_s and signal must have the same length")
        if times.size < 2:
            raise ValueError("each temperature dataset needs at least two observations")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "temperature_unit", unit)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    @property
    def temperature_K(self) -> float:
        """Temperature converted to kelvin exactly once."""

        if self.temperature_unit is TemperatureUnit.KELVIN:
            return self.temperature
        return self.temperature + 273.15


@dataclass(frozen=True, slots=True)
class BatchFitOutcome:
    """One fitter output plus optional evidence for the reliability Gate."""

    fit_result: FitResult
    uncertainty: LocalUncertaintyResult | None = None
    parameter_bounds: Mapping[
        str,
        ParameterBound | tuple[float, float],
    ] = field(default_factory=dict)
    physical_checks: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TypeError("fit_result must be a FitResult")
        if self.uncertainty is not None and not isinstance(
            self.uncertainty, LocalUncertaintyResult
        ):
            raise TypeError("uncertainty must be LocalUncertaintyResult or None")
        if not isinstance(self.parameter_bounds, Mapping):
            raise TypeError("parameter_bounds must be a mapping")
        if not isinstance(self.physical_checks, Mapping):
            raise TypeError("physical_checks must be a mapping")
        object.__setattr__(
            self,
            "parameter_bounds",
            MappingProxyType(dict(self.parameter_bounds)),
        )
        object.__setattr__(
            self,
            "physical_checks",
            MappingProxyType(dict(self.physical_checks)),
        )


@dataclass(frozen=True, slots=True)
class ReliabilityDecision:
    """Fit-level and parameter-level permission for Arrhenius inclusion."""

    reliable: bool
    parameter_reliable: Mapping[str, bool]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reliable, bool):
            raise TypeError("reliable must be boolean")
        values = dict(self.parameter_reliable)
        if not all(isinstance(name, str) and name for name in values):
            raise ValueError("parameter_reliable keys must be nonempty strings")
        if not all(isinstance(value, bool) for value in values.values()):
            raise TypeError("parameter_reliable values must be boolean")
        object.__setattr__(self, "parameter_reliable", MappingProxyType(values))
        object.__setattr__(self, "reasons", tuple(self.reasons))


FitOutput = FitResult | BatchFitOutcome | object
ModelFitter = Callable[[TemperatureDataset], FitOutput]
ReliabilityFilter = Callable[
    [TemperatureDataset, ModelID, BatchFitOutcome],
    ReliabilityDecision,
]


@dataclass(frozen=True, slots=True)
class BatchFitRecord:
    """One independently fitted dataset/model pair."""

    series_index: int
    dataset: TemperatureDataset
    model_id: ModelID
    outcome: BatchFitOutcome | None
    reliability: ReliabilityDecision
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ArrheniusResult:
    """Ordinary least-squares Arrhenius regression for one model parameter."""

    model_id: ModelID
    parameter: str
    n_points: int
    temperatures_K: NDArray[np.float64]
    values: NDArray[np.float64]
    slope: float
    intercept: float
    slope_CI: tuple[float, float]
    intercept_CI: tuple[float, float]
    activation_energy_J_mol: float
    activation_energy_CI_J_mol: tuple[float, float]
    pre_exponential_factor: float
    R2: float
    confidence_level: float

    def __post_init__(self) -> None:
        temperatures = _vector("temperatures_K", self.temperatures_K)
        values = _vector("values", self.values)
        if temperatures.size != self.n_points or values.size != self.n_points:
            raise ValueError("Arrhenius arrays must contain n_points entries")
        object.__setattr__(self, "temperatures_K", temperatures)
        object.__setattr__(self, "values", values)

    @property
    def activation_energy_kJ_mol(self) -> float:
        return self.activation_energy_J_mol / 1000.0

    def to_dict(self) -> dict[str, object]:
        """Return a detached representation for reports and GUI state."""

        return {
            "model_id": self.model_id.value,
            "parameter": self.parameter,
            "n_points": self.n_points,
            "temperatures_K": self.temperatures_K.tolist(),
            "values": self.values.tolist(),
            "slope": self.slope,
            "intercept": self.intercept,
            "slope_CI": list(self.slope_CI),
            "intercept_CI": list(self.intercept_CI),
            "activation_energy_J_mol": self.activation_energy_J_mol,
            "activation_energy_CI_J_mol": list(
                self.activation_energy_CI_J_mol
            ),
            "pre_exponential_factor": self.pre_exponential_factor,
            "R2": self.R2,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True, slots=True)
class TemperatureSeriesResult:
    """Independent fits, deterministic result table, and Arrhenius regressions."""

    records: tuple[BatchFitRecord, ...]
    table: pd.DataFrame
    arrhenius: Mapping[str, ArrheniusResult]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "table", self.table.copy(deep=True))
        object.__setattr__(self, "arrhenius", MappingProxyType(dict(self.arrhenius)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def reproducible_csv(self) -> str:
        """Return a stable CSV representation without timestamps or row indices."""

        buffer = StringIO()
        self.table.to_csv(
            buffer,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
            na_rep="",
        )
        return buffer.getvalue()


def fit_temperature_series(
    datasets: Sequence[TemperatureDataset],
    *,
    fitters: Mapping[ModelID | str, ModelFitter],
    models: Sequence[ModelID | str] = _DEFAULT_MODELS,
    reliability_filter: ReliabilityFilter | None = None,
    confidence_level: float = 0.95,
    minimum_arrhenius_points: int = 3,
) -> TemperatureSeriesResult:
    """Fit each temperature independently, then fit trusted Arrhenius trends.

    Fit callbacks receive one immutable ``TemperatureDataset`` and may return a
    ``FitResult``, ``BatchFitOutcome``, or an object exposing ``selected_fit``
    as a ``FitResult`` (such as ``DualHOFitResult``).  No result enters an
    Arrhenius regression unless the parameter is explicitly accepted by the
    reliability filter.  The default filter applies PyECR's strict model Gate.
    """

    validated_datasets = _validate_datasets(datasets)
    selected_models = _coerce_models(models)
    fitter_map = _coerce_fitters(fitters, selected_models)
    if reliability_filter is not None and not callable(reliability_filter):
        raise TypeError("reliability_filter must be callable or None")
    confidence = _confidence_level(confidence_level)
    minimum_points = _minimum_points(minimum_arrhenius_points)

    ordered_datasets = sorted(
        validated_datasets,
        key=lambda item: (item.temperature_K, item.record_id),
    )
    records: list[BatchFitRecord] = []
    warnings: list[str] = []
    for series_index, dataset in enumerate(ordered_datasets):
        for model_id in selected_models:
            try:
                raw_output = fitter_map[model_id](dataset)
                outcome = _coerce_outcome(raw_output)
                decision = (
                    _default_reliability(dataset, model_id, outcome)
                    if reliability_filter is None
                    else reliability_filter(dataset, model_id, outcome)
                )
                decision = _complete_reliability_decision(
                    model_id,
                    outcome,
                    decision,
                )
                record = BatchFitRecord(
                    series_index=series_index,
                    dataset=dataset,
                    model_id=model_id,
                    outcome=outcome,
                    reliability=decision,
                )
            except CancelledError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                warnings.append(
                    f"FIT_FAILED[{dataset.record_id},{model_id.value}]: {message}"
                )
                record = BatchFitRecord(
                    series_index=series_index,
                    dataset=dataset,
                    model_id=model_id,
                    outcome=None,
                    reliability=ReliabilityDecision(
                        reliable=False,
                        parameter_reliable={},
                        reasons=(f"FIT_FAILED: {message}",),
                    ),
                    error=message,
                )
            records.append(record)

    table = _build_result_table(records, selected_models)
    arrhenius, arrhenius_warnings = _fit_all_arrhenius(
        table,
        selected_models,
        confidence_level=confidence,
        minimum_points=minimum_points,
    )
    warnings.extend(arrhenius_warnings)
    if any(record.model_id is ModelID.M8_DUAL_K and record.outcome is not None
           and record.outcome.fit_result.diagnostics.get("dual_surface_k", {}).get("label_semantics", "effective_modes") == "effective_modes"
           for record in records):
        warnings.append("EFFECTIVE_MODE_ARRHENIUS: k_1/k_2 track fast/slow effective modes; rate ordering can change with temperature. These are not automatically H/O activation energies.")
    return TemperatureSeriesResult(
        records=tuple(records),
        table=table,
        arrhenius=arrhenius,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _default_reliability(
    dataset: TemperatureDataset,
    model_id: ModelID,
    outcome: BatchFitOutcome,
) -> ReliabilityDecision:
    del dataset
    fit = outcome.fit_result
    uncertainties = (
        None
        if outcome.uncertainty is None
        else {model_id: outcome.uncertainty}
    )
    bounds = (
        None
        if not outcome.parameter_bounds
        else {model_id: outcome.parameter_bounds}
    )
    checks = (
        None
        if not outcome.physical_checks
        else {model_id: outcome.physical_checks}
    )
    available_keys = tuple(
        name for name in _MODEL_PARAMETERS[model_id] if name in fit.params
    )
    selection = select_model(
        {model_id: fit},
        uncertainties=uncertainties,
        parameter_bounds=bounds,
        physical_checks=checks,
        key_parameters={model_id: available_keys},
    )
    assessment = selection.assessments[model_id.value]
    return ReliabilityDecision(
        reliable=assessment.reliable,
        parameter_reliable={
            name: assessment.reliable for name in available_keys
        },
        reasons=assessment.gate_failures,
    )


def _complete_reliability_decision(
    model_id: ModelID,
    outcome: BatchFitOutcome,
    decision: ReliabilityDecision,
) -> ReliabilityDecision:
    if not isinstance(decision, ReliabilityDecision):
        raise TypeError("reliability_filter must return ReliabilityDecision")
    fit = outcome.fit_result
    values = dict(decision.parameter_reliable)
    reasons = list(decision.reasons)
    for name in _MODEL_PARAMETERS[model_id]:
        parameter = fit.params.get(name)
        intrinsically_valid = (
            parameter is not None
            and np.isfinite(parameter)
            and float(parameter) > 0.0
        )
        if name not in values:
            values[name] = False
            if parameter is not None:
                reasons.append(f"RELIABILITY_FILTER_MISSING_PARAMETER: {name}.")
        values[name] = bool(values[name] and decision.reliable and intrinsically_valid)
        if parameter is None:
            reasons.append(f"MODEL_PARAMETER_MISSING: {name}.")
        elif not intrinsically_valid:
            reasons.append(f"MODEL_PARAMETER_INVALID: {name}.")
    reliable = bool(
        decision.reliable
        and all(name in fit.params for name in _MODEL_PARAMETERS[model_id])
    )
    return ReliabilityDecision(
        reliable=reliable,
        parameter_reliable=values,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _build_result_table(
    records: Sequence[BatchFitRecord],
    models: Sequence[ModelID],
) -> pd.DataFrame:
    model_order = {model_id: index for index, model_id in enumerate(models)}
    rows: list[dict[str, object]] = []
    for record in records:
        fit = None if record.outcome is None else record.outcome.fit_result
        row: dict[str, object] = {
            "series_index": record.series_index,
            "record_id": record.dataset.record_id,
            "temperature_input": record.dataset.temperature,
            "temperature_unit": record.dataset.temperature_unit.value,
            "temperature_K": record.dataset.temperature_K,
            "inverse_temperature_K": 1.0 / record.dataset.temperature_K,
            "model_id": record.model_id.value,
            "fit_success": fit is not None,
            "reliable": record.reliability.reliable,
            "reliability_reasons": " | ".join(record.reliability.reasons),
            "error": record.error or "",
            "SSE": np.nan if fit is None else fit.SSE,
            "RMSE": np.nan if fit is None else fit.RMSE,
            "R2": np.nan if fit is None else fit.R2,
            "AICc": np.nan if fit is None else fit.AICc,
            "BIC": np.nan if fit is None else fit.BIC,
            "exit_status": pd.NA if fit is None else fit.exit_status,
            "nfev": pd.NA if fit is None else fit.nfev,
            "_model_order": model_order[record.model_id],
        }
        for parameter in _OUTPUT_PARAMETERS:
            row[parameter] = (
                np.nan
                if fit is None or parameter not in fit.params
                else fit.params[parameter]
            )
            row[f"{parameter}_reliable"] = bool(
                record.reliability.parameter_reliable.get(parameter, False)
            )
            interval = (
                None
                if record.outcome is None or record.outcome.uncertainty is None
                else record.outcome.uncertainty.confidence_intervals.get(parameter)
            )
            row[f"{parameter}_CI_lower"] = (
                np.nan if interval is None else interval.lower
            )
            row[f"{parameter}_CI_upper"] = (
                np.nan if interval is None else interval.upper
            )
        rows.append(row)

    columns = [
        "series_index",
        "record_id",
        "temperature_input",
        "temperature_unit",
        "temperature_K",
        "inverse_temperature_K",
        "model_id",
        "fit_success",
        "reliable",
        "reliability_reasons",
        "error",
        "SSE",
        "RMSE",
        "R2",
        "AICc",
        "BIC",
        "exit_status",
        "nfev",
    ]
    for parameter in _OUTPUT_PARAMETERS:
        columns.extend(
            (
                parameter,
                f"{parameter}_reliable",
                f"{parameter}_CI_lower",
                f"{parameter}_CI_upper",
            )
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["temperature_K", "record_id", "_model_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    return table.loc[:, columns]


def _fit_all_arrhenius(
    table: pd.DataFrame,
    models: Sequence[ModelID],
    *,
    confidence_level: float,
    minimum_points: int,
) -> tuple[dict[str, ArrheniusResult], list[str]]:
    results: dict[str, ArrheniusResult] = {}
    warnings: list[str] = []
    for model_id in models:
        model_rows = table.loc[table["model_id"] == model_id.value]
        for parameter in _MODEL_PARAMETERS[model_id]:
            values = pd.to_numeric(model_rows[parameter], errors="coerce")
            reliable = model_rows[f"{parameter}_reliable"].astype(bool)
            available = values.notna() & np.isfinite(values) & (values > 0.0)
            accepted = available & reliable & model_rows["reliable"].astype(bool)
            n_accepted = int(accepted.sum())
            key = f"{model_id.value}:{parameter}"
            if n_accepted < minimum_points:
                if int(available.sum()) > 0:
                    warnings.append(
                        f"ARRHENIUS_SKIPPED[{key}]: {n_accepted} reliable points; "
                        f"at least {minimum_points} required."
                    )
                continue
            temperatures = model_rows.loc[accepted, "temperature_K"].to_numpy(
                dtype=np.float64
            )
            accepted_values = values.loc[accepted].to_numpy(dtype=np.float64)
            try:
                result = _fit_arrhenius(
                    model_id,
                    parameter,
                    temperatures,
                    accepted_values,
                    confidence_level=confidence_level,
                )
            except ValueError as exc:
                warnings.append(f"ARRHENIUS_FAILED[{key}]: {exc}")
                continue
            results[key] = result
            if result.activation_energy_J_mol < 0.0:
                warnings.append(
                    f"NEGATIVE_ACTIVATION_ENERGY[{key}]: inspect model and data."
                )
    return results, warnings


def _fit_arrhenius(
    model_id: ModelID,
    parameter: str,
    temperatures_K: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    confidence_level: float,
) -> ArrheniusResult:
    n_points = temperatures_K.size
    if n_points < 3:
        raise ValueError("at least three reliable points are required for CI")
    inverse_temperature = 1.0 / temperatures_K
    log_values = np.log(values)
    x_mean = float(np.mean(inverse_temperature))
    y_mean = float(np.mean(log_values))
    centred_x = inverse_temperature - x_mean
    sxx = float(np.dot(centred_x, centred_x))
    if sxx <= np.finfo(np.float64).tiny:
        raise ValueError("at least two distinct temperatures are required")
    slope = float(np.dot(centred_x, log_values - y_mean) / sxx)
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * inverse_temperature
    residuals = log_values - fitted
    sse = float(np.dot(residuals, residuals))
    total = float(np.dot(log_values - y_mean, log_values - y_mean))
    r_squared = 1.0 if total <= np.finfo(np.float64).tiny else 1.0 - sse / total

    degrees_of_freedom = n_points - 2
    residual_variance = sse / degrees_of_freedom
    slope_se = float(np.sqrt(max(residual_variance / sxx, 0.0)))
    intercept_se = float(
        np.sqrt(
            max(
                residual_variance * (1.0 / n_points + x_mean**2 / sxx),
                0.0,
            )
        )
    )
    critical = float(
        student_t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom)
    )
    slope_ci = (slope - critical * slope_se, slope + critical * slope_se)
    intercept_ci = (
        intercept - critical * intercept_se,
        intercept + critical * intercept_se,
    )
    activation_energy = -slope * GAS_CONSTANT_J_MOL_K
    activation_ci = (
        -slope_ci[1] * GAS_CONSTANT_J_MOL_K,
        -slope_ci[0] * GAS_CONSTANT_J_MOL_K,
    )
    pre_exponential = float(np.exp(intercept))
    return ArrheniusResult(
        model_id=model_id,
        parameter=parameter,
        n_points=n_points,
        temperatures_K=temperatures_K,
        values=values,
        slope=slope,
        intercept=intercept,
        slope_CI=slope_ci,
        intercept_CI=intercept_ci,
        activation_energy_J_mol=activation_energy,
        activation_energy_CI_J_mol=activation_ci,
        pre_exponential_factor=pre_exponential,
        R2=float(r_squared),
        confidence_level=confidence_level,
    )


def _coerce_outcome(value: FitOutput) -> BatchFitOutcome:
    if isinstance(value, BatchFitOutcome):
        return value
    if isinstance(value, FitResult):
        return BatchFitOutcome(value)
    selected_fit = getattr(value, "selected_fit", None)
    if isinstance(selected_fit, FitResult):
        return BatchFitOutcome(selected_fit)
    raise TypeError(
        "fitter must return FitResult, BatchFitOutcome, or an object with selected_fit"
    )


def _validate_datasets(
    datasets: Sequence[TemperatureDataset],
) -> tuple[TemperatureDataset, ...]:
    if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
        raise TypeError("datasets must be a sequence of TemperatureDataset")
    result = tuple(datasets)
    if not result:
        raise ValueError("datasets must not be empty")
    if not all(isinstance(item, TemperatureDataset) for item in result):
        raise TypeError("datasets must contain only TemperatureDataset")
    identifiers = [item.record_id for item in result]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("record_id values must be unique within a temperature series")
    return result


def _coerce_models(models: Sequence[ModelID | str]) -> tuple[ModelID, ...]:
    if isinstance(models, str) or not isinstance(models, Sequence):
        raise TypeError("models must be a sequence")
    result = tuple(_coerce_model_id(value) for value in models)
    if not result:
        raise ValueError("at least one model must be selected")
    if len(set(result)) != len(result):
        raise ValueError("models must not contain duplicates")
    unsupported = tuple(model for model in result if model not in _SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(
            "temperature series supports M0-M3 and analytic single/dual k (M7/M8); received "
            + ", ".join(model.value for model in unsupported)
        )
    return result


def _coerce_fitters(
    fitters: Mapping[ModelID | str, ModelFitter],
    models: Sequence[ModelID],
) -> dict[ModelID, ModelFitter]:
    if not isinstance(fitters, Mapping):
        raise TypeError("fitters must be a mapping")
    result = {_coerce_model_id(key): value for key, value in fitters.items()}
    missing = set(models) - set(result)
    if missing:
        raise ValueError(
            "missing fitters for: " + ", ".join(sorted(model.value for model in missing))
        )
    if not all(callable(result[model]) for model in models):
        raise TypeError("all selected fitters must be callable")
    return result


def _coerce_model_id(value: ModelID | str) -> ModelID:
    if isinstance(value, ModelID):
        return value
    try:
        return ModelID(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid model ID: {value!r}") from exc


def _coerce_temperature_unit(value: TemperatureUnit | str) -> TemperatureUnit:
    if isinstance(value, TemperatureUnit):
        return value
    aliases = {
        "K": TemperatureUnit.KELVIN,
        "kelvin": TemperatureUnit.KELVIN,
        "C": TemperatureUnit.CELSIUS,
        "°C": TemperatureUnit.CELSIUS,
        "degC": TemperatureUnit.CELSIUS,
        "celsius": TemperatureUnit.CELSIUS,
    }
    if value not in aliases:
        raise ValueError("temperature_unit must explicitly be K or degC/°C")
    return aliases[value]


def _vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _confidence_level(value: float) -> float:
    result = _finite_scalar("confidence_level", value)
    if not 0.0 < result < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return result


def _minimum_points(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("minimum_arrhenius_points must be an integer")
    result = int(value)
    if result < 3:
        raise ValueError("minimum_arrhenius_points must be at least 3 for CI")
    return result
