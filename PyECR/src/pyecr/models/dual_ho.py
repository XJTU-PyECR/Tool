"""Independent H/O process composition for dual-relaxation ECR models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.models.component_cache import evaluate_unit_kernel

__all__ = [
    "DualHOModel",
    "DualHOResult",
    "FractionKernelProtocol",
    "StepType",
]


class StepType(str, Enum):
    """Direction of the gas-composition step used for H/O signs."""

    HYDRATION = "hydration"
    DEHYDRATION = "dehydration"
    CUSTOM = "custom"


class FractionKernelResultProtocol(Protocol):
    """Minimum result interface required from a shared transport kernel."""

    F: ArrayLike


class FractionKernelProtocol(Protocol):
    """Common H/O kernel interface; the model calls it once per process."""

    def evaluate(
        self,
        t_s: ArrayLike,
        diffusion_coefficient_cm2_s: float,
        surface_exchange_coefficient_cm_s: float | None,
        **options: object,
    ) -> FractionKernelResultProtocol: ...


@dataclass(frozen=True, slots=True)
class DualHOResult:
    """Two process fractions, signed components, and their total signal."""

    component_H: NDArray[np.float64]
    component_O: NDArray[np.float64]
    F_H: NDArray[np.float64]
    F_O: NDArray[np.float64]
    total: NDArray[np.float64]
    step_type: StepType
    sign_H: float
    sign_O: float
    kernel_H_result: object
    kernel_O_result: object
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DualHOModel:
    """Compose two independent transport kernels into one observed signal.

    This class is a forward model only.  It does not decide which of D, k,
    amplitude, or baseline are free fitting parameters.
    """

    kernel: FractionKernelProtocol
    step_type: StepType | str = StepType.HYDRATION
    sign_H: float | None = None
    sign_O: float | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.kernel, "evaluate", None)):
            raise TypeError("kernel must provide an evaluate(...) method")

        step_type = _coerce_step_type(self.step_type)
        if step_type is StepType.HYDRATION:
            _reject_unused_custom_signs(self.sign_H, self.sign_O, step_type)
            sign_h, sign_o = -1.0, 1.0
        elif step_type is StepType.DEHYDRATION:
            _reject_unused_custom_signs(self.sign_H, self.sign_O, step_type)
            sign_h, sign_o = 1.0, -1.0
        else:
            if self.sign_H is None or self.sign_O is None:
                raise ValueError("custom step_type requires both sign_H and sign_O")
            sign_h = _finite_scalar("sign_H", self.sign_H)
            sign_o = _finite_scalar("sign_O", self.sign_O)

        object.__setattr__(self, "step_type", step_type)
        object.__setattr__(self, "sign_H", sign_h)
        object.__setattr__(self, "sign_O", sign_o)

    def evaluate(
        self,
        t_s: ArrayLike,
        *,
        D_H_cm2_s: float,
        D_O_cm2_s: float,
        A_H: float,
        A_O: float,
        y0: float,
        k_H_cm_s: float | None = None,
        k_O_cm_s: float | None = None,
        kernel_options: Mapping[str, object] | None = None,
    ) -> DualHOResult:
        """Evaluate H and O independently and combine their signed changes."""

        times = _validated_times(t_s)
        diffusion_h = _positive_finite_scalar("D_H_cm2_s", D_H_cm2_s)
        diffusion_o = _positive_finite_scalar("D_O_cm2_s", D_O_cm2_s)
        amplitude_h = _nonnegative_finite_scalar("A_H", A_H)
        amplitude_o = _nonnegative_finite_scalar("A_O", A_O)
        baseline = _finite_scalar("y0", y0)
        surface_exchange_h = _optional_positive_finite_scalar(
            "k_H_cm_s", k_H_cm_s
        )
        surface_exchange_o = _optional_positive_finite_scalar(
            "k_O_cm_s", k_O_cm_s
        )
        options = _validated_kernel_options(kernel_options)

        kernel_h_result = evaluate_unit_kernel(
            kernel=self.kernel, original_times=t_s, times=times,
            diffusion=diffusion_h, surface=surface_exchange_h,
            options=options, step_type=self.step_type.value,
            sign_h=self.sign_H, sign_o=self.sign_O,
        )
        kernel_o_result = evaluate_unit_kernel(
            kernel=self.kernel, original_times=t_s, times=times,
            diffusion=diffusion_o, surface=surface_exchange_o,
            options=options, step_type=self.step_type.value,
            sign_h=self.sign_H, sign_o=self.sign_O,
        )
        fraction_h = _fraction_from_result("H", kernel_h_result, times.size)
        fraction_o = _fraction_from_result("O", kernel_o_result, times.size)

        component_h = self.sign_H * amplitude_h * (1.0 - fraction_h)
        component_o = self.sign_O * amplitude_o * (1.0 - fraction_o)
        total = baseline + component_h + component_o

        fraction_h = _readonly_copy(fraction_h)
        fraction_o = _readonly_copy(fraction_o)
        component_h = _readonly_copy(component_h)
        component_o = _readonly_copy(component_o)
        total = _readonly_copy(total)

        final_change = self.sign_H * amplitude_h + self.sign_O * amplitude_o
        amplitude_sum = amplitude_h + amplitude_o
        cancellation_index = (
            None if amplitude_sum == 0.0 else abs(final_change) / amplitude_sum
        )
        diagnostics = MappingProxyType(
            {
                "step_type": self.step_type.value,
                "sign_H": self.sign_H,
                "sign_O": self.sign_O,
                "D_H_cm2_s": diffusion_h,
                "D_O_cm2_s": diffusion_o,
                "k_H_cm_s": surface_exchange_h,
                "k_O_cm_s": surface_exchange_o,
                "A_H": amplitude_h,
                "A_O": amplitude_o,
                "y0": baseline,
                "D_O_over_D_H": diffusion_o / diffusion_h,
                "k_O_over_k_H": (
                    None
                    if surface_exchange_h is None or surface_exchange_o is None
                    else surface_exchange_o / surface_exchange_h
                ),
                "delta_y_infinity": final_change,
                "amplitude_cancellation_index": cancellation_index,
                "kernel_options": MappingProxyType(options.copy()),
                "independent_kernel_calls": 2,
                "parameters_designated_free": (),
            }
        )

        return DualHOResult(
            component_H=component_h,
            component_O=component_o,
            F_H=fraction_h,
            F_O=fraction_o,
            total=total,
            step_type=self.step_type,
            sign_H=self.sign_H,
            sign_O=self.sign_O,
            kernel_H_result=kernel_h_result,
            kernel_O_result=kernel_o_result,
            diagnostics=diagnostics,
        )


def _coerce_step_type(value: StepType | str) -> StepType:
    if isinstance(value, StepType):
        return value
    if not isinstance(value, str):
        raise TypeError("step_type must be hydration, dehydration, or custom")
    try:
        return StepType(value.strip().lower())
    except ValueError as exc:
        raise ValueError(
            "step_type must be hydration, dehydration, or custom"
        ) from exc


def _reject_unused_custom_signs(
    sign_h: float | None,
    sign_o: float | None,
    step_type: StepType,
) -> None:
    if sign_h is not None or sign_o is not None:
        raise ValueError(
            f"sign_H/sign_O are only valid when step_type='custom', not "
            f"{step_type.value!r}"
        )


def _fraction_from_result(
    process: str,
    result: object,
    expected_size: int,
) -> NDArray[np.float64]:
    if not hasattr(result, "F"):
        raise TypeError(f"{process} kernel result must provide an F array")
    try:
        fraction = np.asarray(result.F, dtype=np.float64)  # type: ignore[attr-defined]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{process} kernel F must contain real values") from exc
    if fraction.ndim != 1 or fraction.size != expected_size:
        raise ValueError(
            f"{process} kernel F must return one value for every time"
        )
    if not np.all(np.isfinite(fraction)):
        raise ValueError(f"{process} kernel F must contain only finite values")
    return fraction


def _validated_kernel_options(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("kernel_options must be a mapping")
    options = dict(value)
    reserved = {
        "t_s",
        "diffusion_coefficient_cm2_s",
        "surface_exchange_coefficient_cm_s",
    }
    conflicts = reserved.intersection(options)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"kernel_options contains reserved arguments: {names}")
    return options


def _validated_times(t_s: ArrayLike) -> NDArray[np.float64]:
    try:
        times = np.asarray(t_s, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError("t_s must contain real numeric values") from exc
    if times.size == 0:
        raise ValueError("t_s must contain at least one time")
    if not np.all(np.isfinite(times)):
        raise ValueError("t_s values must be finite")
    if np.any(times < 0.0):
        raise ValueError("t_s values must be nonnegative")
    return times


def _readonly_copy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_finite_scalar(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _positive_finite_scalar(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _optional_positive_finite_scalar(
    name: str, value: float | None
) -> float | None:
    if value is None:
        return None
    return _positive_finite_scalar(name, value)
