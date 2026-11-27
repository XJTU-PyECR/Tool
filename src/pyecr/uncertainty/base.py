"""Shared result objects for opt-in, computationally expensive diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.fitting.base import ParameterSpace

__all__ = [
    "AdvancedUncertaintyConfig",
    "ProgressCallback",
    "ProgressUpdate",
    "ProfileTrace",
    "UncertaintyInterval",
    "UncertaintyMethod",
    "UncertaintyResult",
]


class UncertaintyMethod(str, Enum):
    """Slow uncertainty method used to create a result."""

    RESIDUAL_BOOTSTRAP = "residual_bootstrap"
    PROFILE_LIKELIHOOD = "profile_likelihood"


@dataclass(frozen=True, slots=True)
class AdvancedUncertaintyConfig:
    """GUI-facing defaults for diagnostics that must be explicitly enabled.

    ``enabled=False`` is intentional: fitting workflows must not run bootstrap
    or profile likelihood until the user requests advanced uncertainty.
    """

    enabled: bool = False
    bootstrap_resamples: int = 200
    bootstrap_n_jobs: int = 1
    profile_points: int = 41
    profile_span_decades: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        _integer_in_range("bootstrap_resamples", self.bootstrap_resamples, 200, 1000)
        _validate_n_jobs(self.bootstrap_n_jobs)
        if (
            isinstance(self.profile_points, bool)
            or not isinstance(self.profile_points, Integral)
            or int(self.profile_points) < 3
            or int(self.profile_points) % 2 == 0
        ):
            raise ValueError("profile_points must be an odd integer of at least 3")
        _positive_finite("profile_span_decades", self.profile_span_decades)


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """Progress event emitted from the caller's thread."""

    stage: str
    completed: int
    total: int
    message: str = ""

    @property
    def fraction(self) -> float:
        return self.completed / self.total if self.total else 1.0


ProgressCallback = Callable[[ProgressUpdate], None]


@dataclass(frozen=True, slots=True)
class UncertaintyInterval:
    """Confidence interval in optimizer and physical coordinates."""

    parameter: str
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    space: ParameterSpace
    coordinate_estimate: float
    coordinate_lower: float
    coordinate_upper: float


@dataclass(frozen=True, slots=True)
class ProfileTrace:
    """One fixed-parameter profile with all nuisance-parameter optima."""

    parameter: str
    space: ParameterSpace
    parameter_names: tuple[str, ...]
    coordinates: NDArray[np.float64]
    values: NDArray[np.float64]
    SSE: NDArray[np.float64]
    likelihood_ratio: NDArray[np.float64]
    successful: NDArray[np.bool_]
    optimized_parameters: NDArray[np.float64]
    cutoff: float
    reference_SSE: float

    def __post_init__(self) -> None:
        coordinates = _read_only_array(self.coordinates, ndim=1)
        n_points = coordinates.size
        values = _read_only_array(self.values, ndim=1)
        sse = _read_only_array(self.SSE, ndim=1, allow_nan=True)
        likelihood_ratio = _read_only_array(
            self.likelihood_ratio,
            ndim=1,
            allow_nan=True,
        )
        successful = np.asarray(self.successful, dtype=np.bool_).reshape(-1).copy()
        successful.flags.writeable = False
        optimized = _read_only_array(
            self.optimized_parameters,
            ndim=2,
            allow_nan=True,
        )
        if not all(
            array.size == n_points
            for array in (values, sse, likelihood_ratio, successful)
        ):
            raise ValueError("profile arrays must contain the same number of points")
        if optimized.shape != (n_points, len(self.parameter_names)):
            raise ValueError(
                "optimized_parameters shape must be (profile points, parameters)"
            )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "SSE", sse)
        object.__setattr__(self, "likelihood_ratio", likelihood_ratio)
        object.__setattr__(self, "successful", successful)
        object.__setattr__(self, "optimized_parameters", optimized)


@dataclass(frozen=True, slots=True)
class UncertaintyResult:
    """Standalone output from bootstrap or profile likelihood.

    This intentionally does not modify or replace ``FitResult``.  Bootstrap
    stores sample matrices, whereas profile likelihood stores named traces.
    Unused payloads remain ``None`` or empty.
    """

    method: UncertaintyMethod
    parameter_names: tuple[str, ...]
    estimates: Mapping[str, float]
    param_space: Mapping[str, ParameterSpace]
    intervals: Mapping[str, UncertaintyInterval]
    confidence_level: float
    n_requested: int
    n_successful: int
    physical_samples: NDArray[np.float64] | None = None
    coordinate_samples: NDArray[np.float64] | None = None
    profiles: Mapping[str, ProfileTrace] = MappingProxyType({})
    diagnostics: Mapping[str, object] = MappingProxyType({})
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        estimates = dict(self.estimates)
        spaces = dict(self.param_space)
        if tuple(estimates) != names or tuple(spaces) != names:
            raise ValueError(
                "parameter_names, estimates, and param_space must use the same order"
            )
        physical = _optional_sample_matrix(
            self.physical_samples,
            n_parameters=len(names),
        )
        coordinates = _optional_sample_matrix(
            self.coordinate_samples,
            n_parameters=len(names),
        )
        if physical is not None and physical.shape[0] != self.n_successful:
            raise ValueError("physical sample count must equal n_successful")
        if coordinates is not None and coordinates.shape[0] != self.n_successful:
            raise ValueError("coordinate sample count must equal n_successful")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "estimates", MappingProxyType(estimates))
        object.__setattr__(self, "param_space", MappingProxyType(spaces))
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "physical_samples", physical)
        object.__setattr__(self, "coordinate_samples", coordinates)
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        """Return a detached representation suitable for GUI/report export."""

        return {
            "schema_version": 1,
            "method": self.method.value,
            "parameter_names": list(self.parameter_names),
            "estimates": dict(self.estimates),
            "param_space": dict(self.param_space),
            "intervals": {
                name: {
                    "parameter": interval.parameter,
                    "estimate": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "confidence_level": interval.confidence_level,
                    "space": interval.space,
                    "coordinate_estimate": interval.coordinate_estimate,
                    "coordinate_lower": interval.coordinate_lower,
                    "coordinate_upper": interval.coordinate_upper,
                }
                for name, interval in self.intervals.items()
            },
            "confidence_level": self.confidence_level,
            "n_requested": self.n_requested,
            "n_successful": self.n_successful,
            "physical_samples": (
                None if self.physical_samples is None else self.physical_samples.tolist()
            ),
            "coordinate_samples": (
                None
                if self.coordinate_samples is None
                else self.coordinate_samples.tolist()
            ),
            "profiles": {
                name: {
                    "parameter": trace.parameter,
                    "space": trace.space,
                    "parameter_names": list(trace.parameter_names),
                    "coordinates": trace.coordinates.tolist(),
                    "values": trace.values.tolist(),
                    "SSE": trace.SSE.tolist(),
                    "likelihood_ratio": trace.likelihood_ratio.tolist(),
                    "successful": trace.successful.tolist(),
                    "optimized_parameters": trace.optimized_parameters.tolist(),
                    "cutoff": trace.cutoff,
                    "reference_SSE": trace.reference_SSE,
                }
                for name, trace in self.profiles.items()
            },
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value):
        """Strict R07 read of existing uncertainty types, including null scan gaps.

        Only a completely absent schema_version is the known old representation.
        Unknown versions/fields are never interpreted as a different method.
        """
        from pyecr.contracts.common import exact_fields, require, integer, number
        names=('method','parameter_names','estimates','param_space','intervals','confidence_level','n_requested',
               'n_successful','physical_samples','coordinate_samples','profiles','diagnostics','warnings')
        exact_fields(value,(*names,'schema_version') if 'schema_version' in value else names,'UncertaintyResult')
        require(integer(value.get('schema_version',1),'schema_version',minimum=1)==1,'unknown uncertainty result version')
        confidence=number(value['confidence_level'],'confidence_level');require(0<confidence<1,'invalid confidence level')
        count=integer(value['n_successful'],'n_successful');requested=integer(value['n_requested'],'n_requested')
        require(count<=requested,'successful count exceeds budget')
        intervals={}
        for name,row in value['intervals'].items():
            exact_fields(row,UncertaintyInterval.__dataclass_fields__,'UncertaintyInterval')
            interval=UncertaintyInterval(**row)
            require(interval.parameter==name and interval.confidence_level==confidence,'CI label/level conflict')
            for key,item in row.items():
                if key not in {'parameter','space'}:number(item,'interval '+key)
            require(interval.space in {'log','linear'} and interval.lower<=interval.upper,'invalid interval coordinate/order')
            intervals[name]=interval
        profiles={}
        for name,row in value['profiles'].items():
            exact_fields(row,ProfileTrace.__dataclass_fields__,'ProfileTrace')
            require(row['parameter']==name,'profile parameter mismatch')
            require(all(isinstance(flag,bool) for flag in row['successful']),'profile success mask must be boolean')
            require(row['space']=='log' and number(row['cutoff'],'profile cutoff')>0
                    and number(row['reference_SSE'],'profile reference SSE')>0,'invalid profile coordinate/threshold')
            profiles[name]=ProfileTrace(**row)
        samples={}
        for field in ('physical_samples','coordinate_samples'):
            sample=value[field]
            if sample==[]:sample=np.empty((0,len(value['parameter_names'])))
            samples[field]=sample
        names=tuple(value['parameter_names'])
        require(set(value['estimates'])==set(names) and set(value['param_space'])==set(names),'Missing or unknown uncertainty parameter identity')
        # JSON object order is not a coordinate contract. Sample columns use
        # the explicitly stored parameter_names, including after sort_keys.
        result=cls(method=UncertaintyMethod(value['method']),parameter_names=names,
            estimates={n:value['estimates'][n] for n in names},param_space={n:value['param_space'][n] for n in names},intervals=intervals,confidence_level=confidence,
            n_requested=requested,n_successful=count,**samples,
            profiles=profiles,diagnostics=value['diagnostics'],warnings=tuple(value['warnings']))
        require(set(intervals)<=set(result.parameter_names) and all(s in {'log','linear'} for s in result.param_space.values()),'unknown result parameters/coordinates')
        for name,estimate in result.estimates.items():
            number(estimate,'estimate');require(result.param_space[name]!='log' or estimate>0,'nonpositive log estimate')
        for name,interval in result.intervals.items():
            require(interval.estimate==result.estimates[name] and interval.space==result.param_space[name],'interval parent conflict')
            expected=(10.**np.array([interval.coordinate_estimate,interval.coordinate_lower,interval.coordinate_upper])
                if interval.space=='log' else np.array([interval.coordinate_estimate,interval.coordinate_lower,interval.coordinate_upper]))
            require(np.allclose(expected,[interval.estimate,interval.lower,interval.upper],rtol=1e-12,atol=0),'interval coordinate/physical conflict')
        if result.method is UncertaintyMethod.RESIDUAL_BOOTSTRAP:
            require(not result.profiles,'bootstrap cannot contain profile scans')
            for sample in (result.physical_samples,result.coordinate_samples):
                require(sample is None or len(sample)==count,'sample count conflict')
            if result.coordinate_samples is not None and count>=2:
                for name,ci in result.intervals.items():
                    index=result.parameter_names.index(name)
                    expected=np.quantile(result.coordinate_samples[:,index],[(1-confidence)/2,(1+confidence)/2],method='linear')
                    require(np.allclose(expected,[ci.coordinate_lower,ci.coordinate_upper],rtol=1e-12,atol=1e-14),'saved quantile/C conflict; explicitly recompute')
        else:
            require(result.physical_samples is None and result.coordinate_samples is None,'profile cannot contain bootstrap samples')
            for name,trace in result.profiles.items():
                require(name in result.parameter_names and tuple(trace.parameter_names)==result.parameter_names,'profile parent conflict')
                from scipy.stats import chi2
                require(np.isclose(trace.cutoff,chi2.ppf(confidence,1),rtol=1e-12,atol=1e-14),'profile threshold/C conflict; explicitly recompute')
        return result


def _optional_sample_matrix(
    values: ArrayLike | None,
    *,
    n_parameters: int,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    result = _read_only_array(values, ndim=2)
    if result.shape[1] != n_parameters:
        raise ValueError("sample matrix column count must match parameter_names")
    return result


def _read_only_array(
    values: ArrayLike,
    *,
    ndim: int,
    allow_nan: bool = False,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != ndim:
        raise ValueError(f"array must have {ndim} dimensions")
    if allow_nan:
        if np.any(np.isinf(result)):
            raise ValueError("array cannot contain infinity")
    elif not np.all(np.isfinite(result)):
        raise ValueError("array values must be finite")
    result.flags.writeable = False
    return result


def _integer_in_range(name: str, value: int, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _validate_n_jobs(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("n_jobs must be an integer")
    result = int(value)
    if result == 0 or result < -1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    return result


def _positive_finite(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)
