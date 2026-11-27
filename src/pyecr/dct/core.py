"""Nonnegative Tikhonov inversion of characteristic-time distributions.

This module is diagnostic only.  Peaks are deliberately reported as candidate
time scales and are not assigned to any physical species or transport step.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks
from pyecr.dct.execution import isolated_numeric, check_cancel, numeric_phase

__all__ = ["CandidateTimescale", "DCTResult", "fit_dct"]


@dataclass(frozen=True, slots=True)
class CandidateTimescale:
    """One non-physically-labelled peak in the recovered distribution."""

    position_s: float
    area: float
    width_decades: float
    grid_index: int
    label: str = "candidate_timescale"

    def to_dict(self) -> dict[str, object]:
        """Return the candidate in JSON-compatible form."""

        return {
            "position_s": self.position_s,
            "area": self.area,
            "width_decades": self.width_decades,
            "grid_index": self.grid_index,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CandidateTimescale:
        """Restore a candidate without adding a physical interpretation."""

        return cls(
            position_s=float(value["position_s"]),
            area=float(value["area"]),
            width_decades=float(value["width_decades"]),
            grid_index=int(value["grid_index"]),
            label=str(value.get("label", "candidate_timescale")),
        )


@dataclass(frozen=True, slots=True)
class DCTResult:
    """Recovered DCT spectrum, reconstruction, peaks, and diagnostics."""

    tau_s: NDArray[np.float64]
    chi: NDArray[np.float64]
    mass_distribution: NDArray[np.float64]
    y_fit: NDArray[np.float64]
    residuals: NDArray[np.float64]
    regularization_lambda: float
    lambda_source: str
    gcv_lambdas: NDArray[np.float64] | None
    gcv_scores: NDArray[np.float64] | None
    candidate_timescales: tuple[CandidateTimescale, ...]
    peak_positions_s: tuple[float, ...]
    peak_areas: tuple[float, ...]
    normalization_integral: float
    SSE: float
    RMSE: float
    resolution_warnings: tuple[str, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        tau = _read_only_vector(self.tau_s)
        chi = _read_only_vector(self.chi)
        mass = _read_only_vector(self.mass_distribution)
        fitted = _read_only_vector(self.y_fit)
        residuals = _read_only_vector(self.residuals)
        if tau.size != chi.size or tau.size != mass.size:
            raise ValueError("tau_s, chi, and mass_distribution must align")
        if fitted.size != residuals.size:
            raise ValueError("y_fit and residuals must align")
        gcv_lambdas = _optional_read_only_vector(self.gcv_lambdas)
        gcv_scores = _optional_read_only_vector(self.gcv_scores)
        if (gcv_lambdas is None) != (gcv_scores is None):
            raise ValueError("gcv_lambdas and gcv_scores must both be present or absent")
        if gcv_lambdas is not None and gcv_lambdas.size != gcv_scores.size:
            raise ValueError("GCV lambda and score arrays must align")
        object.__setattr__(self, "tau_s", tau)
        object.__setattr__(self, "chi", chi)
        object.__setattr__(self, "mass_distribution", mass)
        object.__setattr__(self, "y_fit", fitted)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "gcv_lambdas", gcv_lambdas)
        object.__setattr__(self, "gcv_scores", gcv_scores)
        object.__setattr__(self, "candidate_timescales", tuple(self.candidate_timescales))
        object.__setattr__(self, "peak_positions_s", tuple(self.peak_positions_s))
        object.__setattr__(self, "peak_areas", tuple(self.peak_areas))
        object.__setattr__(self, "resolution_warnings", tuple(self.resolution_warnings))
        diagnostics = dict(self.diagnostics)
        if 'lambda_scan' in diagnostics:
            scan = diagnostics['lambda_scan']
            if not isinstance(scan, Mapping) or scan.get('version') != 'stored-lambda-spectra/1':
                raise ValueError('Unknown stored lambda scan')
            frozen = []
            for row in scan.get('candidates', ()):
                chi_row, mass_row = _read_only_vector(row['chi']), _read_only_vector(row['mass'])
                if len(chi_row) != len(tau) or len(mass_row) != len(tau):
                    raise ValueError('Stored lambda spectrum and tau grid must align')
                if not np.all(np.isfinite(chi_row)) or not np.all(np.isfinite(mass_row)):
                    raise ValueError('Stored lambda spectrum must be finite')
                frozen.append(MappingProxyType({**row, 'chi': tuple(map(float, chi_row)),
                                               'mass': tuple(map(float, mass_row))}))
            diagnostics['lambda_scan'] = MappingProxyType(dict(version=scan['version'],candidates=tuple(frozen)))
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    def to_dict(self) -> dict[str, object]:
        """Serialize the complete diagnostic result for project persistence."""

        return {
            "tau_s": self.tau_s.tolist(),
            "chi": self.chi.tolist(),
            "mass_distribution": self.mass_distribution.tolist(),
            "y_fit": self.y_fit.tolist(),
            "residuals": self.residuals.tolist(),
            "regularization_lambda": self.regularization_lambda,
            "lambda_source": self.lambda_source,
            "gcv_lambdas": (
                None if self.gcv_lambdas is None else self.gcv_lambdas.tolist()
            ),
            "gcv_scores": (
                None if self.gcv_scores is None else self.gcv_scores.tolist()
            ),
            "candidate_timescales": [
                item.to_dict() for item in self.candidate_timescales
            ],
            "peak_positions_s": list(self.peak_positions_s),
            "peak_areas": list(self.peak_areas),
            "normalization_integral": self.normalization_integral,
            "SSE": self.SSE,
            "RMSE": self.RMSE,
            "resolution_warnings": list(self.resolution_warnings),
            "diagnostics": _json_compatible(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DCTResult:
        """Restore a result saved by :meth:`to_dict`."""

        candidates = value.get("candidate_timescales", ())
        if not isinstance(candidates, (list, tuple)):
            raise TypeError("candidate_timescales must be an array")
        diagnostics = value.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be an object")
        return cls(
            tau_s=value["tau_s"],
            chi=value["chi"],
            mass_distribution=value["mass_distribution"],
            y_fit=value["y_fit"],
            residuals=value["residuals"],
            regularization_lambda=float(value["regularization_lambda"]),
            lambda_source=str(value["lambda_source"]),
            gcv_lambdas=value.get("gcv_lambdas"),
            gcv_scores=value.get("gcv_scores"),
            candidate_timescales=tuple(
                CandidateTimescale.from_dict(_as_mapping(item, "candidate"))
                for item in candidates
            ),
            peak_positions_s=tuple(float(item) for item in value["peak_positions_s"]),  # type: ignore[union-attr]
            peak_areas=tuple(float(item) for item in value["peak_areas"]),  # type: ignore[union-attr]
            normalization_integral=float(value["normalization_integral"]),
            SSE=float(value["SSE"]),
            RMSE=float(value["RMSE"]),
            resolution_warnings=tuple(
                str(item) for item in value.get("resolution_warnings", ())  # type: ignore[union-attr]
            ),
            diagnostics=dict(diagnostics),
        )


@dataclass(slots=True)
class _SpectrumSolution:
    mass: NDArray[np.float64]
    chi: NDArray[np.float64]
    y_fit: NDArray[np.float64]
    residuals: NDArray[np.float64]
    data_residual_norm: float
    regularization_norm: float
    success: bool


@isolated_numeric('monotonic')
def fit_dct(
    t_s: ArrayLike,
    y: ArrayLike,
    *,
    tau_min_s: float | None = None,
    tau_max_s: float | None = None,
    n_tau: int = 121,
    regularization_lambda: float | None = None,
    gcv_lambdas: ArrayLike | None = None,
    normalization_weight: float = 100.0,
    peak_prominence_fraction: float = 0.03,
    minimum_peak_area: float = 0.05,
    resolution_limit_decades: float = 0.50,
) -> DCTResult:
    """Recover a nonnegative distribution from ``y = 1 - K @ mass``.

    ``K[i, j] = exp(-t[i] / tau[j])`` and ``mass`` is the nonnegative
    quadrature mass at each logarithmic grid point.  ``chi`` is the associated
    density per log10(tau), so trapezoidal integration of ``chi`` is one for a
    normalized relaxation.  The regularized objective is
    ``||K mass - (1-y)||^2 + lambda ||L chi||^2``.

    If ``regularization_lambda`` is omitted, generalized cross-validation
    selects one value from ``gcv_lambdas``.  No peak receives a physical label.
    """

    check_cancel()
    times, signal = _validated_data(t_s, y)
    tau_min, tau_max = _tau_bounds(times, tau_min_s, tau_max_s)
    count = _positive_integer("n_tau", n_tau)
    if count < 10:
        raise ValueError("n_tau must be at least 10")
    norm_weight = _positive_finite("normalization_weight", normalization_weight)
    prominence = _unit_interval(
        "peak_prominence_fraction",
        peak_prominence_fraction,
        include_zero=False,
    )
    minimum_area = _unit_interval(
        "minimum_peak_area",
        minimum_peak_area,
        include_zero=True,
    )
    resolution_limit = _positive_finite(
        "resolution_limit_decades",
        resolution_limit_decades,
    )

    log_tau = np.linspace(np.log10(tau_min), np.log10(tau_max), count)
    tau = np.power(10.0, log_tau)
    log_step = float(log_tau[1] - log_tau[0])
    quadrature_weights = np.full(count, log_step)
    quadrature_weights[[0, -1]] *= 0.5
    kernel = np.exp(-times[:, None] / tau[None, :])
    target_remaining = 1.0 - signal
    second_difference = _second_difference_matrix(count)
    regularizer_mass = second_difference / quadrature_weights[None, :]
    check_cancel()

    gcv_grid: NDArray[np.float64] | None = None
    gcv_scores: NDArray[np.float64] | None = None
    scan_solutions: list[tuple[float, _SpectrumSolution]] = []
    sensitivity_solutions: list[tuple[float, _SpectrumSolution]] = []
    selection_warnings: list[str] = []
    if regularization_lambda is None:
        gcv_grid = _gcv_grid(gcv_lambdas)
        solutions: list[_SpectrumSolution] = []
        scores = np.empty(gcv_grid.size, dtype=np.float64)
        for index, candidate_lambda in enumerate(gcv_grid):
            check_cancel()
            solution = _solve_spectrum(
                kernel,
                target_remaining,
                signal,
                quadrature_weights,
                regularizer_mass,
                regularization_lambda=float(candidate_lambda),
                normalization_weight=norm_weight,
            )
            solutions.append(solution)
            scores[index] = _gcv_score(
                kernel,
                regularizer_mass,
                solution,
                regularization_lambda=float(candidate_lambda),
            )
        finite = np.flatnonzero(np.isfinite(scores))
        if finite.size == 0:
            raise RuntimeError("GCV produced no finite regularization score")
        selected_index = int(finite[np.argmin(scores[finite])])
        selected_lambda = float(gcv_grid[selected_index])
        solution = solutions[selected_index]
        sensitivity_indices = sorted(
            {
                max(0, selected_index - 1),
                selected_index,
                min(gcv_grid.size - 1, selected_index + 1),
            }
        )
        if len(sensitivity_indices) < 3 and gcv_grid.size >= 3:
            sensitivity_indices = (
                [0, 1, 2]
                if selected_index == 0
                else [gcv_grid.size - 3, gcv_grid.size - 2, gcv_grid.size - 1]
            )
        sensitivity_solutions = [
            (float(gcv_grid[index]), solutions[index])
            for index in sensitivity_indices
        ]
        gcv_scores = scores
        scan_solutions = list(zip(map(float, gcv_grid), solutions))
        lambda_source = "gcv"
        if selected_index in (0, gcv_grid.size - 1):
            selection_warnings.append(
                "DCT_GCV_BOUNDARY: selected lambda is at the search-grid boundary; "
                "expand the lambda range before interpreting fine structure."
            )
    else:
        selected_lambda = _nonnegative_finite(
            "regularization_lambda",
            regularization_lambda,
        )
        solution = _solve_spectrum(
            kernel,
            target_remaining,
            signal,
            quadrature_weights,
            regularizer_mass,
            regularization_lambda=selected_lambda,
            normalization_weight=norm_weight,
        )
        sensitivity_lambdas = (
            (0.0, 1.0e-8, 1.0e-6)
            if selected_lambda == 0.0
            else (
                selected_lambda / 10.0,
                selected_lambda,
                selected_lambda * 10.0,
            )
        )
        for candidate_lambda in sensitivity_lambdas:
            check_cancel()
            if candidate_lambda == selected_lambda:
                sensitivity_solutions.append((candidate_lambda, solution))
                continue
            try:
                candidate_solution = _solve_spectrum(
                    kernel,
                    target_remaining,
                    signal,
                    quadrature_weights,
                    regularizer_mass,
                    regularization_lambda=candidate_lambda,
                    normalization_weight=norm_weight,
                )
            except RuntimeError:
                continue
            sensitivity_solutions.append((candidate_lambda, candidate_solution))
        lambda_source = "manual"

    check_cancel()
    if not solution.success:
        selection_warnings.append(
            "DCT_SOLVER_STATUS: bounded least-squares stopped without declaring success."
        )
    peak_readout = {}
    candidates, peak_warnings = _candidate_timescales(
        tau,
        log_tau,
        solution.chi,
        solution.mass,
        prominence_fraction=prominence,
        minimum_area=minimum_area,
        resolution_limit_decades=resolution_limit,
        evidence=peak_readout,
    )
    normalization_integral = float(
        np.dot(quadrature_weights, solution.chi)
    )
    sse = float(np.dot(solution.residuals, solution.residuals))
    rmse = float(np.sqrt(sse / signal.size))
    warnings = tuple(dict.fromkeys((*selection_warnings, *peak_warnings)))
    peak_positions = tuple(candidate.position_s for candidate in candidates)
    peak_areas = tuple(candidate.area for candidate in candidates)
    lambda_peak_snapshots = _lambda_peak_snapshots(
        sensitivity_solutions,
        tau,
        log_tau,
        prominence_fraction=prominence,
        minimum_area=minimum_area,
        resolution_limit_decades=resolution_limit,
        selected_lambda=selected_lambda,
    )

    return DCTResult(
        tau_s=tau,
        chi=solution.chi,
        mass_distribution=solution.mass,
        y_fit=solution.y_fit,
        residuals=solution.residuals,
        regularization_lambda=selected_lambda,
        lambda_source=lambda_source,
        gcv_lambdas=gcv_grid,
        gcv_scores=gcv_scores,
        candidate_timescales=candidates,
        peak_positions_s=peak_positions,
        peak_areas=peak_areas,
        normalization_integral=normalization_integral,
        SSE=sse,
        RMSE=rmse,
        resolution_warnings=warnings,
        diagnostics={
            "model_id": "M6_DCT",
            "interpretation": "diagnostic_candidate_timescales_only",
            "kernel_definition": "exp(-t/tau)",
            "tau_grid": "uniform_log10",
            "log10_tau_step": log_step,
            "tau_min_s": tau_min,
            "tau_max_s": tau_max,
            "n_tau": count,
            "nonnegative_constraint": True,
            "normalization_constraint": "sum(mass)=1",
            "regularization": "second_difference_tikhonov",
            "gcv_score_semantics": "unconstrained_trace_heuristic_for_bounded_normalized_solution",
            "data_residual_norm": solution.data_residual_norm,
            "regularization_norm": solution.regularization_norm,
            "lambda_peak_snapshots": lambda_peak_snapshots,
            # Retain the already-computed spectra. Reconstruction for browsing
            # can be projected from these masses without another inverse solve.
            # O(n_lambda*n_tau) storage, independent of observation count.
            "lambda_scan": {
                "version": "stored-lambda-spectra/1",
                "candidates": [dict(regularization_lambda=lam,
                    chi=s.chi.tolist(), mass=s.mass.tolist(),
                    RMSE=float(np.sqrt(np.mean(s.residuals**2))),
                    solver_success=bool(s.success)) for lam, s in scan_solutions],
            },
            "peak_readout": peak_readout,
            "peak_labels_are_physical_assignments": False,
        },
    )


def _lambda_peak_snapshots(
    solutions: list[tuple[float, _SpectrumSolution]],
    tau: NDArray[np.float64],
    log_tau: NDArray[np.float64],
    *,
    prominence_fraction: float,
    minimum_area: float,
    resolution_limit_decades: float,
    selected_lambda: float,
) -> tuple[Mapping[str, object], ...]:
    snapshots: list[Mapping[str, object]] = []
    for candidate_lambda, solution in sorted(solutions, key=lambda item: item[0]):
        check_cancel()
        candidates, _ = _candidate_timescales(
            tau,
            log_tau,
            solution.chi,
            solution.mass,
            prominence_fraction=prominence_fraction,
            minimum_area=minimum_area,
            resolution_limit_decades=resolution_limit_decades,
        )
        snapshots.append(
            MappingProxyType(
                {
                    "regularization_lambda": float(candidate_lambda),
                    "is_selected": bool(
                        np.isclose(candidate_lambda, selected_lambda, rtol=1.0e-12)
                    ),
                    "peak_positions_s": tuple(
                        candidate.position_s for candidate in candidates
                    ),
                    "peak_areas": tuple(candidate.area for candidate in candidates),
                    "peak_widths_decades": tuple(
                        candidate.width_decades for candidate in candidates
                    ),
                }
            )
        )
    return tuple(snapshots)


def _solve_spectrum(
    kernel: NDArray[np.float64],
    target_remaining: NDArray[np.float64],
    signal: NDArray[np.float64],
    quadrature_weights: NDArray[np.float64],
    regularizer_mass: NDArray[np.float64],
    *,
    regularization_lambda: float,
    normalization_weight: float,
) -> _SpectrumSolution:
    blocks = [kernel]
    targets = [target_remaining]
    if regularization_lambda > 0.0:
        blocks.append(np.sqrt(regularization_lambda) * regularizer_mass)
        targets.append(np.zeros(regularizer_mass.shape[0]))
    blocks.append(normalization_weight * np.ones((1, kernel.shape[1])))
    targets.append(np.array([normalization_weight]))
    augmented = np.vstack(blocks)
    augmented_target = np.concatenate(targets)
    with numeric_phase('bounded_lsq'):
        optimizer = lsq_linear(
            augmented,
            augmented_target,
            bounds=(0.0, np.inf),
            method="trf",
            tol=1.0e-10,
            lsmr_tol="auto",
            max_iter=500,
        )
    mass = np.maximum(np.asarray(optimizer.x, dtype=np.float64), 0.0)
    total_mass = float(np.sum(mass))
    if not np.isfinite(total_mass) or total_mass <= np.finfo(np.float64).tiny:
        raise RuntimeError("nonnegative DCT inversion returned zero total mass")
    mass /= total_mass
    chi = mass / quadrature_weights
    remaining_fit = kernel @ mass
    y_fit = 1.0 - remaining_fit
    residuals = signal - y_fit
    return _SpectrumSolution(
        mass=mass,
        chi=chi,
        y_fit=y_fit,
        residuals=residuals,
        data_residual_norm=float(np.linalg.norm(remaining_fit - target_remaining)),
        regularization_norm=float(np.linalg.norm(regularizer_mass @ mass)),
        success=bool(optimizer.success),
    )


def _gcv_score(
    kernel: NDArray[np.float64],
    regularizer_mass: NDArray[np.float64],
    solution: _SpectrumSolution,
    *,
    regularization_lambda: float,
) -> float:
    with numeric_phase('gcv_trace'):
        gram_data = kernel.T @ kernel
        gram_regularizer = regularizer_mass.T @ regularizer_mass
        normal = gram_data + regularization_lambda * gram_regularizer
        scale = max(float(np.linalg.norm(normal, ord=2)), 1.0)
        normal = normal + np.eye(normal.shape[0]) * np.finfo(np.float64).eps * scale
        effective_df = float(np.trace(np.linalg.pinv(normal) @ gram_data))
    n_observations = kernel.shape[0]
    denominator = max(n_observations - effective_df, 1.0e-6)
    rss = solution.data_residual_norm**2
    return float(rss / denominator**2)


def _candidate_timescales(
    tau: NDArray[np.float64],
    log_tau: NDArray[np.float64],
    chi: NDArray[np.float64],
    mass: NDArray[np.float64],
    *,
    prominence_fraction: float,
    minimum_area: float,
    resolution_limit_decades: float,
    evidence: dict | None = None,
) -> tuple[tuple[CandidateTimescale, ...], tuple[str, ...]]:
    from pyecr.dct.peak_geometry import half_height_bounds

    # Integrate the displayed piecewise-linear density, splitting at valleys.
    # Do not assign mass by distance to maxima, or reallocate rejected basins.
    # `mass` remains in the private API for existing callers; it is the
    # quadrature representation, not a separate density to partition.
    maximum = float(np.max(chi))
    if maximum <= 0.0:
        return (), ("DCT_NO_CANDIDATE_TIMESCALE: recovered spectrum is empty.",)
    peaks, properties = find_peaks(
        np.r_[0., chi, 0.],
        prominence=prominence_fraction * maximum,
        plateau_size=(None, None),
    )
    peak_indices = [int(index)-1 for index in peaks]
    edges = [0]+[a+int(np.argmin(chi[a:b+1])) for a,b in zip(peak_indices,peak_indices[1:])]+[len(chi)-1]
    candidates, records = [], []
    warnings: list[str] = []
    for j,index in enumerate(peak_indices):
        left,right = edges[j:j+2]
        bx,by = log_tau[left:right+1],chi[left:right+1]
        area = float(np.trapezoid(by,bx))
        if area < minimum_area or area <= 0 or right <= left:
            continue
        hl,hr,clipped = half_height_bounds(bx,by,index-left)
        tail = ((left == 0 and by[0] >= prominence_fraction*chi[index]) or
                (right == len(chi)-1 and by[-1] >= prominence_fraction*chi[index]))
        flat = properties['plateau_sizes'][j] > 1
        limits = tuple(k for ok,k in ((clipped,'WIDTH_TRUNCATED'),(tail,'BASIN_TRUNCATED'),(flat,'FLAT_TOP_PEAK')) if ok)
        for flag in limits:
            message = {'WIDTH_TRUNCATED':'FWHM is clipped within its basin.',
                       'BASIN_TRUNCATED':'A material tail reaches the tau boundary; area may be incomplete.',
                       'FLAT_TOP_PEAK':'Tau is a representative of a flat maximum, not a unique peak position.'}[flag]
            warnings.append(f'DCT_{flag}: {message}')
        candidates.append(CandidateTimescale(position_s=float(tau[index]),area=area,width_decades=hr-hl,grid_index=index))
        records.append(dict(grid_index=index,basin_log10_tau=[float(bx[0]),float(bx[-1])],
                            half_height_log10_tau=[hl,hr],
                            warnings=limits+(('BOUNDARY_PEAK',) if index <= 1 or index >= len(tau)-2 else ())))
    candidates = tuple(candidates)
    if not candidates:
        warnings.append('DCT_NO_CANDIDATE_TIMESCALE: no basin meets the reporting thresholds; the spectrum remains available.')
    if evidence is not None:
        evidence.update(version='valley-basins-linear-fwhm/2',basins=records,
            reported_area=sum(p.area for p in candidates),unreported_area=max(0.,float(np.trapezoid(chi,log_tau))-sum(p.area for p in candidates)),
            minimum_area=minimum_area,prominence_fraction=prominence_fraction)
    if any(candidate.grid_index <= 1 or candidate.grid_index >= tau.size - 2 for candidate in candidates):
        warnings.append(
            "DCT_PEAK_NEAR_GRID_BOUNDARY: extend the tau range before interpreting "
            "the boundary candidate time scale."
        )
    for j,(left, right) in enumerate(zip(candidates[:-1], candidates[1:], strict=True)):
        separation = np.log10(right.position_s / left.position_s)
        overlap = records[j]['half_height_log10_tau'][1] >= records[j+1]['half_height_log10_tau'][0]
        if separation < resolution_limit_decades or overlap:
            warnings.append(
                "DCT_LIMITED_TIMESCALE_RESOLUTION: adjacent candidate time scales "
                f"have separation {separation:.3g} decades or overlapping/clipped half-height intervals."
            )
            break
    if len(candidates) == 1 and candidates[0].width_decades >= 0.25:
        warnings.append(
            "DCT_BROAD_CANDIDATE_TIMESCALE: one broad peak may contain unresolved "
            "nearby time scales."
        )
    return candidates, tuple(dict.fromkeys(warnings))


def _fwhm_decades(
    log_tau: NDArray[np.float64],
    chi: NDArray[np.float64],
    peak_index: int,
) -> float:
    from pyecr.dct.peak_geometry import half_height_bounds
    left,right,_ = half_height_bounds(log_tau,chi,peak_index)
    return right-left


def _second_difference_matrix(n_values: int) -> NDArray[np.float64]:
    matrix = np.zeros((n_values - 2, n_values), dtype=np.float64)
    rows = np.arange(n_values - 2)
    matrix[rows, rows] = 1.0
    matrix[rows, rows + 1] = -2.0
    matrix[rows, rows + 2] = 1.0
    return matrix


def _gcv_grid(values: ArrayLike | None) -> NDArray[np.float64]:
    if values is None:
        return np.logspace(-8.0, 2.0, 25)
    grid = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if grid.size < 3:
        raise ValueError("gcv_lambdas must contain at least three values")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("gcv_lambdas must be positive and finite")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("gcv_lambdas must be strictly increasing")
    return grid


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _json_compatible(value: object) -> object:
    """Thaw immutable diagnostic containers into ordinary JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tau_bounds(
    times: NDArray[np.float64],
    supplied_min: float | None,
    supplied_max: float | None,
) -> tuple[float, float]:
    positive_times = times[times > 0.0]
    if positive_times.size == 0:
        raise ValueError("time axis must contain at least one positive value")
    tau_min = (
        float(np.min(positive_times)) / 10.0
        if supplied_min is None
        else _positive_finite("tau_min_s", supplied_min)
    )
    tau_max = (
        float(np.max(times)) * 10.0
        if supplied_max is None
        else _positive_finite("tau_max_s", supplied_max)
    )
    if tau_min >= tau_max:
        raise ValueError("tau_min_s must be smaller than tau_max_s")
    return tau_min, tau_max


def _validated_data(
    t_s: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    times = _read_only_vector(t_s)
    signal = _read_only_vector(y)
    if times.size != signal.size:
        raise ValueError("t_s and y must have the same length")
    if times.size < 6:
        raise ValueError("DCT requires at least six observations")
    if np.any(times < 0.0):
        raise ValueError("t_s values must be nonnegative")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_s must be strictly increasing")
    return times, signal


def _optional_read_only_vector(
    values: ArrayLike | None,
) -> NDArray[np.float64] | None:
    return None if values is None else _read_only_vector(values)


def _read_only_vector(values: ArrayLike) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError("array must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("array values must be finite")
    result.flags.writeable = False
    return result


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_finite(name: str, value: float) -> float:
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit_interval(
    name: str,
    value: float,
    *,
    include_zero: bool,
) -> float:
    result = _finite_scalar(name, value)
    lower_valid = result >= 0.0 if include_zero else result > 0.0
    if not lower_valid or result > 1.0:
        bracket = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {bracket}")
    return result
