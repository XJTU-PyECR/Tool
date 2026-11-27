"""Model-independent fitting contracts and serializable result objects.

This module defines shared interfaces only.  It intentionally contains no ECR
forward-model equations and no optimizer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FitCandidate",
    "FitResult",
    "ModelProtocol",
    "ParameterSpace",
    "Theta",
]


ParameterSpace = Literal["log", "linear"]
"""Optimizer coordinate used for an individual fitted parameter."""

Theta = Mapping[str, float] | Sequence[float] | NDArray[np.float64]
"""Named or ordered parameter coordinates accepted by a forward model."""

_VALID_PARAMETER_SPACES = frozenset({"log", "linear"})


@runtime_checkable
class ModelProtocol(Protocol):
    """Minimal prediction interface required by model-independent fitters."""

    def predict(
        self,
        t: ArrayLike,
        theta: Theta,
        context: object,
    ) -> NDArray[np.float64]:
        """Return model predictions corresponding to the supplied times."""
        ...


@dataclass(frozen=True, slots=True)
class FitCandidate:
    """Serializable summary of one valid MultiStart optimization result."""

    start_index: int
    params: Mapping[str, float]
    SSE: float
    exit_status: int
    nfev: int
    success: bool
    initial_coordinates: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        params = _normalise_params(self.params)
        coordinates = tuple(float(value) for value in self.initial_coordinates)
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("initial_coordinates must be finite")
        if not isinstance(self.success, bool):
            raise TypeError("success must be boolean")
        object.__setattr__(self, "params", MappingProxyType(params))
        object.__setattr__(
            self,
            "start_index",
            _integer("start_index", self.start_index, nonnegative=True),
        )
        object.__setattr__(self, "SSE", _metric("SSE", self.SSE, nonnegative=True))
        object.__setattr__(self, "exit_status", _integer("exit_status", self.exit_status))
        object.__setattr__(self, "nfev", _integer("nfev", self.nfev, nonnegative=True))
        object.__setattr__(self, "initial_coordinates", coordinates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "params": dict(self.params),
            "SSE": self.SSE,
            "exit_status": self.exit_status,
            "nfev": self.nfev,
            "success": self.success,
            "initial_coordinates": list(self.initial_coordinates),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitCandidate:
        if not isinstance(data, Mapping):
            raise TypeError("FitCandidate data must be a mapping")
        return cls(
            start_index=data["start_index"],
            params=data["params"],
            SSE=data["SSE"],
            exit_status=data["exit_status"],
            nfev=data["nfev"],
            success=data["success"],
            initial_coordinates=tuple(data.get("initial_coordinates", ())),
        )


@dataclass(frozen=True, slots=True)
class FitResult:
    """Portable result of a completed fit.

    ``params`` contains reported parameter values.  ``param_space`` uses the
    same keys and records whether each parameter was optimized in logarithmic
    or linear coordinates.  Their insertion order defines Jacobian column
    order.  Fixed experimental context does not belong in ``params``.

    Residuals use the convention ``observed - fitted``.  Fitters are
    responsible for calculating the supplied statistics consistently and for
    documenting any weighting or effective sample-size convention.
    """

    params: Mapping[str, float]
    param_space: Mapping[str, ParameterSpace]
    y_fit: NDArray[np.float64]
    residuals: NDArray[np.float64]
    SSE: float
    RMSE: float
    R2: float
    AIC: float
    AICc: float
    BIC: float
    exit_status: int
    nfev: int
    jacobian: NDArray[np.float64] | None = None
    warnings: tuple[str, ...] = ()
    candidates: tuple[FitCandidate, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def random_seed(self) -> int | None:
        from pyecr.fitting.reproducibility import fit_randomness
        record=fit_randomness(self)
        return None if record is None else record['random_seed']

    @property
    def stochastic(self) -> bool | None:
        from pyecr.fitting.reproducibility import fit_randomness
        record=fit_randomness(self)
        return None if record is None else record['stochastic']

    def __post_init__(self) -> None:
        params = _normalise_params(self.params)
        param_space = _normalise_param_space(self.param_space)
        if tuple(params) != tuple(param_space):
            raise ValueError(
                "params and param_space must contain the same keys in the same order"
            )

        y_fit = _read_only_vector("y_fit", self.y_fit)
        residuals = _read_only_vector("residuals", self.residuals)
        if y_fit.shape != residuals.shape:
            raise ValueError("y_fit and residuals must have the same shape")

        jacobian = _normalise_jacobian(
            self.jacobian,
            n_observations=y_fit.size,
            n_parameters=len(params),
        )
        warnings = _normalise_warnings(self.warnings)
        candidates = tuple(
            item if isinstance(item, FitCandidate) else FitCandidate.from_dict(item)
            for item in self.candidates
        )
        diagnostics = _freeze_json_mapping(self.diagnostics, "diagnostics")

        object.__setattr__(self, "params", MappingProxyType(params))
        object.__setattr__(self, "param_space", MappingProxyType(param_space))
        object.__setattr__(self, "y_fit", y_fit)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "SSE", _metric("SSE", self.SSE, nonnegative=True))
        object.__setattr__(self, "RMSE", _metric("RMSE", self.RMSE, nonnegative=True))
        object.__setattr__(self, "R2", _metric("R2", self.R2))
        object.__setattr__(self, "AIC", _metric("AIC", self.AIC))
        object.__setattr__(self, "AICc", _metric("AICc", self.AICc))
        object.__setattr__(self, "BIC", _metric("BIC", self.BIC))
        object.__setattr__(self, "exit_status", _integer("exit_status", self.exit_status))
        object.__setattr__(self, "nfev", _integer("nfev", self.nfev, nonnegative=True))
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible representation."""

        return {
            "params": dict(self.params),
            **({'random_seed':self.random_seed,'stochastic':self.stochastic} if 'fitting_randomness' in self.diagnostics else {}),
            "param_space": dict(self.param_space),
            "y_fit": self.y_fit.tolist(),
            "residuals": self.residuals.tolist(),
            "SSE": self.SSE,
            "RMSE": self.RMSE,
            "R2": self.R2,
            "AIC": self.AIC,
            "AICc": self.AICc,
            "BIC": self.BIC,
            "exit_status": self.exit_status,
            "nfev": self.nfev,
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "warnings": list(self.warnings),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "diagnostics": _thaw_json(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitResult:
        """Reconstruct a result from :meth:`to_dict` output."""

        if not isinstance(data, Mapping):
            raise TypeError("FitResult data must be a mapping")

        required_fields = (
            "params",
            "param_space",
            "y_fit",
            "residuals",
            "SSE",
            "RMSE",
            "R2",
            "AIC",
            "AICc",
            "BIC",
            "exit_status",
            "nfev",
        )
        missing = [name for name in required_fields if name not in data]
        if missing:
            raise ValueError(f"FitResult data is missing fields: {', '.join(missing)}")

        result = cls(
            params=data["params"],
            param_space=data["param_space"],
            y_fit=np.asarray(data["y_fit"], dtype=np.float64),
            residuals=np.asarray(data["residuals"], dtype=np.float64),
            SSE=data["SSE"],
            RMSE=data["RMSE"],
            R2=data["R2"],
            AIC=data["AIC"],
            AICc=data["AICc"],
            BIC=data["BIC"],
            exit_status=data["exit_status"],
            nfev=data["nfev"],
            jacobian=(
                None
                if data.get("jacobian") is None
                else np.asarray(data["jacobian"], dtype=np.float64)
            ),
            warnings=tuple(data.get("warnings", ())),
            candidates=tuple(
                FitCandidate.from_dict(item) for item in data.get("candidates", ())
            ),
            diagnostics=data.get("diagnostics", {}),
        )
        if 'stochastic' in data and data['stochastic']!=result.stochastic:
            raise ValueError('Serialized stochastic flag disagrees with actual fit provenance')
        if 'random_seed' in data and data['random_seed']!=result.random_seed:
            raise ValueError('Serialized random_seed disagrees with actual fit provenance')
        return result


def _normalise_params(params: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    result: dict[str, float] = {}
    for name, value in params.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"parameter {name!r} must be a real number")
        result[name] = float(value)
    return result


def _normalise_param_space(
    param_space: Mapping[str, ParameterSpace],
) -> dict[str, ParameterSpace]:
    if not isinstance(param_space, Mapping):
        raise TypeError("param_space must be a mapping")

    result: dict[str, ParameterSpace] = {}
    for name, space in param_space.items():
        if not isinstance(name, str) or not name:
            raise ValueError("param_space names must be nonempty strings")
        if space not in _VALID_PARAMETER_SPACES:
            raise ValueError(
                f"param_space for {name!r} must be 'log' or 'linear'; received {space!r}"
            )
        result[name] = space
    return result


def _read_only_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} values must be finite")
    result.flags.writeable = False
    return result


def _normalise_jacobian(
    jacobian: ArrayLike | None,
    *,
    n_observations: int,
    n_parameters: int,
) -> NDArray[np.float64] | None:
    if jacobian is None:
        return None
    try:
        result = np.asarray(jacobian, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError("jacobian must contain real numeric values") from exc
    if result.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    if result.shape != (n_observations, n_parameters):
        raise ValueError(
            "jacobian shape must be (number of observations, number of parameters)"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("jacobian values must be finite")
    result.flags.writeable = False
    return result


def _normalise_warnings(warnings: Sequence[str]) -> tuple[str, ...]:
    if isinstance(warnings, str):
        raise TypeError("warnings must be a sequence of strings, not one string")
    result = tuple(warnings)
    if not all(isinstance(message, str) for message in result):
        raise TypeError("warnings must contain only strings")
    return result


def _metric(name: str, value: float, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(name: str, value: int, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _freeze_json_mapping(
    value: Mapping[str, object],
    path: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"{path} keys must be nonempty strings")
    return MappingProxyType(
        {key: _freeze_json(item, f"{path}.{key}") for key, item in value.items()}
    )


def _freeze_json(value: object, path: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Real):
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return int(value) if isinstance(value, Integral) else result
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, path)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains a non-JSON-compatible value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
