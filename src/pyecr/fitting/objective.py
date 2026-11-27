"""Public, reusable M0/M1 residual and reported-SSE evaluation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.boundary.composition import BoundaryComposedSingleEcrModel
from pyecr.fitting.base import ParameterSpace
from pyecr.models.single import SingleEcrContext, SingleEcrModel
from pyecr.fitting.progress import count_fit_call

__all__ = ["ObjectiveContext", "ObjectiveEvaluation", "SingleObjectiveEvaluator"]


@dataclass(frozen=True, slots=True)
class ObjectiveContext:
    """Validated arrays, optimizer layout, and production forward-model context."""

    model: SingleEcrModel | BoundaryComposedSingleEcrModel
    time_s: NDArray[np.float64]
    observed: NDArray[np.float64]
    forward_context: SingleEcrContext
    parameter_names: tuple[str, ...]
    parameter_spaces: tuple[ParameterSpace, ...]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    fixed_coordinates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model, (SingleEcrModel, BoundaryComposedSingleEcrModel)):
            raise TypeError("model must be a production single-process ECR model")
        if not isinstance(self.forward_context, SingleEcrContext):
            raise TypeError("forward_context must be SingleEcrContext")
        time_s = _readonly_vector(self.time_s, "time_s")
        observed = _readonly_vector(self.observed, "observed")
        lower = _readonly_vector(self.lower_bounds, "lower_bounds")
        upper = _readonly_vector(self.upper_bounds, "upper_bounds")
        if time_s.size != observed.size or observed.size < 3:
            raise ValueError("objective needs at least three aligned observations")
        if len(self.parameter_names) != len(self.parameter_spaces) or len(self.parameter_names) != lower.size or lower.size != upper.size:
            raise ValueError("optimizer names, spaces, and bounds must align")
        if not np.all(lower < upper):
            raise ValueError("optimizer lower bounds must be below upper bounds")
        fixed = dict(self.fixed_coordinates)
        if set(fixed) & set(self.parameter_names):
            raise ValueError("fixed and optimized coordinates must not overlap")
        if any(
            name not in {"log10_D", "log10_k"}
            or not isfinite(float(value))
            for name, value in fixed.items()
        ):
            raise ValueError("fixed coordinates must be finite log10_D/log10_k values")
        object.__setattr__(self, "fixed_coordinates", MappingProxyType(fixed))
        for name, values in (("time_s", time_s), ("observed", observed), ("lower_bounds", lower), ("upper_bounds", upper)):
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    """One forward call with FitResult-sign residuals and raw SSE."""

    prediction: NDArray[np.float64]
    residuals: NDArray[np.float64]
    sse: float


@dataclass(frozen=True, slots=True)
class SingleObjectiveEvaluator:
    """The single shared objective path for least_squares and fixed D-k maps.

    Optimizer residuals are prediction-observed. Reported residuals have the
    opposite sign. Their raw unweighted squared norm is FitResult.SSE even
    when scipy's optimizer uses a robust loss; no point weights currently exist.
    """

    context: ObjectiveContext
    cancel_check: Callable[[], None] | None = None

    def predict(self, coordinates: ArrayLike) -> NDArray[np.float64]:
        count_fit_call("forward")
        values = self._coordinates(coordinates)
        theta = dict(self.context.fixed_coordinates)
        theta.update(zip(self.context.parameter_names, map(float, values), strict=True))
        prediction = np.asarray(
            self.context.model.predict(
                self.context.time_s, theta, self.context.forward_context
            ),
            dtype=np.float64,
        )
        if prediction.shape != self.context.observed.shape:
            raise ValueError("forward prediction shape differs from observations")
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("forward prediction contains non-finite values")
        return prediction

    def optimizer_residual(self, coordinates: ArrayLike) -> NDArray[np.float64]:
        if self.cancel_check is not None:
            self.cancel_check()
        count_fit_call("residual")
        return self.predict(coordinates) - self.context.observed

    def evaluate(self, coordinates: ArrayLike) -> ObjectiveEvaluation:
        prediction = self.predict(coordinates)
        residuals = self.context.observed - prediction
        return ObjectiveEvaluation(
            prediction=prediction,
            residuals=residuals,
            sse=float(residuals @ residuals),
        )

    def _coordinates(self, coordinates: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (len(self.context.parameter_names),):
            raise ValueError("optimizer coordinate shape does not match parameter layout")
        if not np.all(np.isfinite(values)):
            raise ValueError("optimizer coordinates must be finite")
        return values


def _readonly_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.flags.writeable = False
    return result
