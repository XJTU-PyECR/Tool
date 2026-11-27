"""Production forward model for traditional single-process ECR."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.models.ids import ModelID
from pyecr.models.rectangular import (
    DiffusionControlledRectangularKernel,
    RectangularKernel,
    RectangularKernelResult,
)

__all__ = ["SingleEcrContext", "SingleEcrModel", "SingleEcrPrediction"]


Theta = Mapping[str, float] | Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SingleEcrContext:
    """Fixed signal and numerical values supplied to ``SingleEcrModel``."""

    amplitude: float = 1.0
    baseline: float = 0.0
    term_tol: float = 1.0e-10
    max_terms: int = 512

    def __post_init__(self) -> None:
        for name, value in (("amplitude", self.amplitude), ("baseline", self.baseline)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.term_tol, bool) or not isinstance(self.term_tol, Real):
            raise TypeError("term_tol must be a real number")
        if not isfinite(float(self.term_tol)) or self.term_tol <= 0.0:
            raise ValueError("term_tol must be positive and finite")
        if isinstance(self.max_terms, bool) or not isinstance(self.max_terms, Integral):
            raise TypeError("max_terms must be an integer")
        if self.max_terms <= 0:
            raise ValueError("max_terms must be positive")


@dataclass(frozen=True, slots=True)
class SingleEcrPrediction:
    """Completion signal and underlying remaining-fraction kernel result."""

    y: NDArray[np.float64]
    F: NDArray[np.float64]
    kernel_result: RectangularKernelResult
    D_cm2_s: float
    k_cm_s: float | None
    amplitude: float
    baseline: float


@dataclass(frozen=True, slots=True)
class SingleEcrModel:
    """Single-process M0/M1 signal model using an injected production kernel."""

    model_id: ModelID
    kernel: DiffusionControlledRectangularKernel | RectangularKernel

    def __post_init__(self) -> None:
        if self.model_id is ModelID.M0_SINGLE_D:
            if not isinstance(self.kernel, DiffusionControlledRectangularKernel):
                raise TypeError("M0 requires DiffusionControlledRectangularKernel")
        elif self.model_id is ModelID.M1_SINGLE_DK:
            if not isinstance(self.kernel, RectangularKernel):
                raise TypeError("M1 requires RectangularKernel")
        else:
            raise ValueError("SingleEcrModel supports only M0_SINGLE_D or M1_SINGLE_DK")

    @property
    def optimizer_parameter_names(self) -> tuple[str, ...]:
        if self.model_id is ModelID.M0_SINGLE_D:
            return ("log10_D",)
        return ("log10_D", "log10_k")

    def predict(
        self,
        t: ArrayLike,
        theta: Theta,
        context: object,
    ) -> NDArray[np.float64]:
        """Return baseline plus amplitude times the completed fraction."""

        return self.evaluate(t, theta, context).y

    def evaluate(
        self,
        t: ArrayLike,
        theta: Theta,
        context: object,
    ) -> SingleEcrPrediction:
        if not isinstance(theta, Mapping):
            raise TypeError("theta must be a mapping of optimizer coordinates")
        if not isinstance(context, SingleEcrContext):
            raise TypeError("context must be a SingleEcrContext")

        log10_d = _finite_coordinate(theta, "log10_D")
        diffusion = float(10.0**log10_d)
        amplitude = _optional_coordinate(theta, "amplitude", context.amplitude)
        baseline = _optional_coordinate(theta, "baseline", context.baseline)

        if self.model_id is ModelID.M0_SINGLE_D:
            if "log10_k" in theta:
                raise ValueError("M0 theta must not contain log10_k")
            kernel_result = self.kernel.evaluate(  # type: ignore[union-attr]
                t,
                diffusion,
                term_tol=context.term_tol,
                max_terms=context.max_terms,
            )
            surface_exchange = None
        else:
            log10_k = _finite_coordinate(theta, "log10_k")
            surface_exchange = float(10.0**log10_k)
            kernel_result = self.kernel.evaluate(  # type: ignore[union-attr]
                t,
                diffusion,
                surface_exchange,
                term_tol=context.term_tol,
                max_terms=context.max_terms,
            )

        fraction = np.asarray(kernel_result.F, dtype=np.float64)
        prediction = baseline + amplitude * (1.0 - fraction)
        prediction = prediction.copy()
        fraction = fraction.copy()
        prediction.flags.writeable = False
        fraction.flags.writeable = False
        return SingleEcrPrediction(
            y=prediction,
            F=fraction,
            kernel_result=kernel_result,
            D_cm2_s=diffusion,
            k_cm_s=surface_exchange,
            amplitude=amplitude,
            baseline=baseline,
        )


def _finite_coordinate(theta: Mapping[str, object], name: str) -> float:
    if name not in theta:
        raise ValueError(f"theta is missing {name}")
    return _optional_coordinate(theta, name, 0.0)


def _optional_coordinate(
    theta: Mapping[str, object],
    name: str,
    default: float,
) -> float:
    value = theta.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"theta[{name!r}] must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"theta[{name!r}] must be finite")
    return result
