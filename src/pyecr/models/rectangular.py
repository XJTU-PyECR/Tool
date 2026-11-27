"""Composable finite-surface kernels for rectangular specimens."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyecr.config import SampleGeometry
from pyecr.models.kernel import (
    KernelResult,
    diffusion_controlled_fraction_1d,
    finite_surface_fraction_1d,
)

__all__ = [
    "DiffusionControlledRectangularKernel",
    "RectangularKernel",
    "RectangularKernelResult",
]


AxisResults = tuple[
    KernelResult | None,
    KernelResult | None,
    KernelResult | None,
]


@dataclass(frozen=True, slots=True)
class RectangularKernelResult:
    """Combined rectangular result and the three ordered axis results."""

    F: NDArray[np.float64]
    axis_results: AxisResults
    dimension_count: int
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DiffusionControlledRectangularKernel:
    """Explicit diffusion-controlled kernel over selected rectangular axes."""

    full_dims_cm: tuple[float, float, float]
    active_dimensions: tuple[bool, bool, bool] = (True, False, False)

    def __post_init__(self) -> None:
        geometry = SampleGeometry(
            full_dims_cm=self.full_dims_cm,
            active_dimensions=self.active_dimensions,
        )
        geometry.validate()
        object.__setattr__(self, "full_dims_cm", geometry.full_dims_cm)
        object.__setattr__(self, "active_dimensions", geometry.active_dimensions)

    def evaluate(
        self,
        t_s: ArrayLike,
        diffusion_coefficient_cm2_s: float,
        surface_exchange_coefficient_cm_s: float | None = None,
        *,
        term_tol: float = 1.0e-10,
        max_terms: int = 512,
    ) -> RectangularKernelResult:
        """Multiply explicit odd-series kernels over active axes."""

        if surface_exchange_coefficient_cm_s is not None:
            raise ValueError(
                "diffusion-controlled kernel does not accept a surface-exchange "
                "coefficient"
            )
        axis_results_list: list[KernelResult | None] = []
        total_fraction: NDArray[np.float64] | None = None
        for full_dim_cm, is_active in zip(
            self.full_dims_cm, self.active_dimensions, strict=True
        ):
            if not is_active:
                axis_results_list.append(None)
                continue
            axis_result = diffusion_controlled_fraction_1d(
                t_s,
                diffusion_coefficient_cm2_s,
                full_dim_cm,
                term_tol=term_tol,
                max_terms=max_terms,
            )
            axis_results_list.append(axis_result)
            if total_fraction is None:
                total_fraction = axis_result.F.copy()
            else:
                total_fraction *= axis_result.F

        if total_fraction is None:
            raise RuntimeError("validated geometry unexpectedly has no active dimension")
        axis_results = tuple(axis_results_list)
        if len(axis_results) != 3:
            raise RuntimeError("validated geometry unexpectedly has an invalid axis count")
        typed_axis_results: AxisResults = (
            axis_results[0],
            axis_results[1],
            axis_results[2],
        )
        active_results = tuple(
            result for result in typed_axis_results if result is not None
        )
        axis_labels = ("x", "y", "z")
        diagnostics = MappingProxyType(
            {
                "surface_mode": "instant",
                "full_dims_cm": self.full_dims_cm,
                "active_dimensions": self.active_dimensions,
                "active_axes": tuple(
                    label
                    for label, active in zip(
                        axis_labels, self.active_dimensions, strict=True
                    )
                    if active
                ),
                "half_dims_cm": tuple(
                    None
                    if result is None
                    else result.diagnostics["half_dim_cm"]
                    for result in typed_axis_results
                ),
                "n_terms_used_by_axis": tuple(
                    None if result is None else result.n_terms_used
                    for result in typed_axis_results
                ),
                "coefficient_sum_by_axis": tuple(
                    None if result is None else result.coefficient_sum
                    for result in typed_axis_results
                ),
                "truncation_error_by_axis": tuple(
                    None
                    if result is None
                    else result.diagnostics["truncation_error"]
                    for result in typed_axis_results
                ),
                "all_axes_converged": all(
                    result.diagnostics["converged"] for result in active_results
                ),
                "normalization_applied": False,
            }
        )
        return RectangularKernelResult(
            F=total_fraction,
            axis_results=typed_axis_results,
            dimension_count=len(active_results),
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True)
class RectangularKernel:
    """Finite-surface kernel composed over explicitly active dimensions."""

    full_dims_cm: tuple[float, float, float]
    active_dimensions: tuple[bool, bool, bool] = (True, False, False)

    def __post_init__(self) -> None:
        geometry = SampleGeometry(
            full_dims_cm=self.full_dims_cm,
            active_dimensions=self.active_dimensions,
        )
        geometry.validate()
        object.__setattr__(self, "full_dims_cm", geometry.full_dims_cm)
        object.__setattr__(self, "active_dimensions", geometry.active_dimensions)

    def evaluate(
        self,
        t_s: ArrayLike,
        diffusion_coefficient_cm2_s: float,
        surface_exchange_coefficient_cm_s: float,
        *,
        term_tol: float = 1.0e-10,
        max_terms: int = 512,
    ) -> RectangularKernelResult:
        """Multiply one-dimensional kernels over active x, y, and z axes."""

        axis_results_list: list[KernelResult | None] = []
        total_fraction: NDArray[np.float64] | None = None

        for full_dim_cm, is_active in zip(
            self.full_dims_cm, self.active_dimensions, strict=True
        ):
            if not is_active:
                axis_results_list.append(None)
                continue

            axis_result = finite_surface_fraction_1d(
                t_s,
                diffusion_coefficient_cm2_s,
                surface_exchange_coefficient_cm_s,
                full_dim_cm,
                term_tol=term_tol,
                max_terms=max_terms,
            )
            axis_results_list.append(axis_result)

            if total_fraction is None:
                total_fraction = axis_result.F.copy()
            else:
                total_fraction *= axis_result.F

        if total_fraction is None:
            raise RuntimeError("validated geometry unexpectedly has no active dimension")

        axis_results = tuple(axis_results_list)
        if len(axis_results) != 3:
            raise RuntimeError("validated geometry unexpectedly has an invalid axis count")

        typed_axis_results: AxisResults = (
            axis_results[0],
            axis_results[1],
            axis_results[2],
        )
        active_results = tuple(
            result for result in typed_axis_results if result is not None
        )
        axis_labels = ("x", "y", "z")

        diagnostics = MappingProxyType(
            {
                "full_dims_cm": self.full_dims_cm,
                "active_dimensions": self.active_dimensions,
                "active_axes": tuple(
                    label
                    for label, active in zip(
                        axis_labels, self.active_dimensions, strict=True
                    )
                    if active
                ),
                "shared_surface_exchange_coefficient_cm_s": (
                    surface_exchange_coefficient_cm_s
                ),
                "half_dims_cm": tuple(
                    None
                    if result is None
                    else result.diagnostics["half_dim_cm"]
                    for result in typed_axis_results
                ),
                "bi_by_axis": tuple(
                    None if result is None else result.diagnostics["bi"]
                    for result in typed_axis_results
                ),
                "n_terms_used_by_axis": tuple(
                    None if result is None else result.n_terms_used
                    for result in typed_axis_results
                ),
                "coefficient_sum_by_axis": tuple(
                    None if result is None else result.coefficient_sum
                    for result in typed_axis_results
                ),
                "truncation_error_by_axis": tuple(
                    None
                    if result is None
                    else result.diagnostics["truncation_error"]
                    for result in typed_axis_results
                ),
                "all_axes_converged": all(
                    result.diagnostics["converged"] for result in active_results
                ),
                "normalization_applied": False,
            }
        )

        return RectangularKernelResult(
            F=total_fraction,
            axis_results=typed_axis_results,
            dimension_count=len(active_results),
            diagnostics=diagnostics,
        )
