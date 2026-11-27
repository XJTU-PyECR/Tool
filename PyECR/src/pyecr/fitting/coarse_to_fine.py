"""Opt-in P60 screening. Only converged full-observation TRF runs are publishable.

This module never constructs a FitResult or a preprocessed dataset. Screening
scores and endpoints are candidate provenance, not scientific fit statistics.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from numbers import Integral
import time

import numpy as np
from scipy.optimize import least_squares as screening_trf

VERSION = "p60-coarse-to-fine-v1"
SCREENING_ALGORITHM = "early-log-linear-index-v1"
SCREENING_RULE = (
    "zero-based eligible positions; endpoints plus first min(16,budget//4) rows; "
    "round-robin nearest log1p elapsed-time, linear elapsed-time and uniform-index "
    "targets; earlier row wins ties; ascending-index fill; no RNG, no interpolation"
)


class NoCompletedRefinement(RuntimeError):
    """No candidate may replace the current fit; the exception carries its audit."""


@dataclass(frozen=True)
class CoarseToFineOptions:
    enabled: bool = False
    screening_points: int = 48
    coarse_max_nfev: int = 60
    top_k: int = 3
    basin_distance: float = 0.15

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean")
        for name in ("screening_points", "coarse_max_nfev", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.screening_points < 3:
            raise ValueError("screening_points must be at least three")
        if isinstance(self.basin_distance, bool) or not np.isfinite(self.basin_distance) or self.basin_distance <= 0:
            raise ValueError("basin_distance must be finite and positive")


def screening_indices(times, original_indices, budget):
    """Return deterministic internal positions and exact original-row provenance."""
    t = np.asarray(times, dtype=float)
    original = np.asarray(original_indices)
    if (t.ndim != 1 or t.size < 3 or not np.all(np.isfinite(t))
            or not np.all(np.diff(t) > 0)):
        raise ValueError("screening times must be finite and strictly increasing")
    if (original.shape != t.shape or original.dtype.kind not in "iu"
            or np.any(original < 0) or not np.all(np.diff(original) > 0)):
        raise ValueError("original indices must be aligned, increasing integer row IDs")
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 3:
        raise ValueError("screening budget must be an integer >= 3")
    requested_budget = int(budget)
    budget = min(requested_budget, t.size)
    selected = set(range(min(16, budget // 4))) | {0, t.size - 1}
    elapsed = t - t[0]
    positive = elapsed[elapsed > 0]
    log_elapsed = np.log1p(elapsed / positive[0])
    transforms = (log_elapsed, elapsed, np.arange(t.size, dtype=float))
    for targets in zip(*(np.linspace(x[0], x[-1], budget) for x in transforms)):
        for x, target in zip(transforms, targets):
            selected.add(int(np.argmin(np.abs(x - target))))
            if len(selected) == budget:
                break
        if len(selected) == budget:
            break
    for i in range(t.size):
        if len(selected) == budget:
            break
        selected.add(i)
    positions = np.array(sorted(selected), dtype=int)
    return positions, {
        "algorithm": SCREENING_ALGORITHM, "version": 1, "seed": None,
        "rule": SCREENING_RULE, "eligible_original_indices": original.tolist(),
        "screening_original_indices": original[positions].tolist(),
        "screening_positions": positions.tolist(), "full_points": int(t.size),
        "screening_points": int(positions.size), "requested_budget": requested_budget,
        "data_scope": "current full fit-eligible processed observations",
    }


@dataclass
class ScreeningLedger:
    options: CoarseToFineOptions
    original_indices: object = None
    stages: list = field(default_factory=list)
    wall_start: float = field(default_factory=time.perf_counter)

    def to_dict(self):
        return {
            "version": VERSION, "experimental": True, "default_enabled": False,
            "options": {name: getattr(self.options, name) for name in
                        ("screening_points", "coarse_max_nfev", "top_k", "basin_distance")},
            "stages": self.stages,
            "screening_wall_s": sum(s["screening_wall_s"] for s in self.stages),
            "refinement_wall_s": sum(s["refinement_wall_s"] for s in self.stages),
            "total_wall_s": time.perf_counter() - self.wall_start,
            "statistics_scope": "full fit-eligible data; screening scores are candidates only",
        }


_active = ContextVar("p60_screening", default=None)


def current_screening():
    return _active.get()


@contextmanager
def coarse_to_fine_scope(options=None, *, original_indices=None):
    """Explicit experimental entry point, also available to direct core callers."""
    options = options or CoarseToFineOptions()
    if not isinstance(options, CoarseToFineOptions):
        raise TypeError("options must be CoarseToFineOptions")
    ledger = ScreeningLedger(options, original_indices) if options.enabled else None
    token = _active.set(ledger)
    try:
        yield ledger
    finally:
        _active.reset(token)


def select_distinct_candidates(candidates, top_k, distance):
    """Stable SSE rank, then greedy L-infinity distance in physical log coordinates."""
    selected = []
    for row in sorted(candidates, key=lambda c: (c["screening_score"], c["start_index"])):
        v = np.asarray(row["basin_coordinates"])
        if all(np.max(np.abs(v - np.asarray(other["basin_coordinates"]))) >= distance
               for other in selected):
            selected.append(row)
            if len(selected) == top_k:
                break
    return selected


def screen_and_refine(*, times, starts, lower, upper, subset_factory,
                      full_residual, full_solver, max_nfev, model_id, seed,
                      physical_parameters, cancel_check=None, progress=None):
    """Execute the entire coarse ensemble, distinct top-K, then full-data TRF.

    Returns only successful finite full-data OptimizeResults, keyed by the
    original start index. Nonconvergence is audited and never published.
    """
    ledger = current_screening()
    if ledger is None:
        raise RuntimeError("screen_and_refine requires explicit experimental scope")
    # Core callers may create the two task scopes in either order. The local
    # portfolio adapter overrides SciPy's method keyword, so reject it here too.
    from pyecr.fitting.solver_strategy import _active as active_solver
    solver_ledger = active_solver.get()
    if solver_ledger is not None and (
            solver_ledger.options.strategy != "trf" or solver_ledger.full_provider is not None):
        raise ValueError("P60 requires full-data TRF; solver portfolio composition is unsupported")
    options = ledger.options
    from pyecr.fitting.production_stop import execution,candidate_eligible
    from pyecr.fitting.iteration_observation import observed_least_squares
    from pyecr.fitting.progress import fit_start,fit_start_finished
    from contextlib import nullcontext
    monitored=execution()
    observation_context=None
    if monitored is not None:
        if not monitored.screening_enabled:raise ValueError('P60 needs the explicit verified Monitored screening/refinement scope')
        from pyecr.fitting.solver_strategy import _production_problem
        problem=_production_problem(full_residual,lower,upper,solver_ledger)
        names=problem.parameterization.coordinate_names
        observation_context=dict(names=names,kinds=tuple('log10' if n.startswith('log10_') else 'linear' for n in names),stage_id=problem.model_id)
    original = (np.arange(len(times), dtype=int) if ledger.original_indices is None
                else ledger.original_indices)
    if options.screening_points <= len(lower):
        raise ValueError("screening_points must exceed the free parameter count")
    positions, sampling = screening_indices(times, original, options.screening_points)
    if len(positions) <= len(lower):
        raise ValueError("screening observations must exceed the free parameter count")
    stage = dict(model_id=model_id, sampling=sampling, start_seed=int(seed),
                 start_rule="unchanged production seeded MultiStart schedule",
                 basin_rule="greedy coarse-score order; L-infinity physical parameters in their log10 optimizer space; linear nuisance scaled by full signal span",
                 basin_distance=options.basin_distance, candidates=[], selected_start_indices=[],
                 refinements=[], screening_wall_s=0., refinement_wall_s=0., status="running")
    ledger.stages.append(stage)
    def check():
        if cancel_check is not None:
            cancel_check()
    began = time.perf_counter()
    phase = "screening"
    try:
        residual = subset_factory(positions)
        for i, start in enumerate(starts):
            check()
            if progress:
                progress(i / len(starts) * .5, f"P60 {model_id} screening {i+1}/{len(starts)}")
            row = dict(start_index=i, initial_coordinates=np.asarray(start).tolist(),
                       status="failed", error=None, nfev=None)
            stage["candidates"].append(row)
            try:
                if monitored is not None:fit_start(model_id+' screening',i+1,len(starts),'linear')
                with monitored.objective_scope('screening',positions) if monitored is not None else nullcontext():
                    opt = (observed_least_squares(residual,start,_observation_context=observation_context,
                        bounds=(lower,upper),method='trf',loss='linear',max_nfev=min(max_nfev,options.coarse_max_nfev)) if monitored is not None else
                        screening_trf(residual,start,bounds=(lower,upper),method='trf',loss='linear',max_nfev=min(max_nfev,options.coarse_max_nfev)))
                check()
                params, basin = physical_parameters(opt.x)
                score = float(opt.fun @ opt.fun)
                if not np.isfinite(score) or not np.all(np.isfinite(opt.x)) or not np.all(np.isfinite(basin)):
                    raise FloatingPointError("nonfinite screening endpoint")
                row.update(status="candidate", coordinates=opt.x.tolist(), params=params,
                           basin_coordinates=list(basin), screening_score=score,
                           coarse_exit_status=int(opt.status), nfev=int(opt.nfev))
            except (ArithmeticError, ValueError) as exc:
                check()
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if monitored is not None:fit_start_finished()
        stage["screening_wall_s"] = time.perf_counter() - began
        stage["coarse_total_nfev"] = sum(r["nfev"] or 0 for r in stage["candidates"])
        selected = select_distinct_candidates(
            [r for r in stage["candidates"] if r["status"] == "candidate"],
            options.top_k, options.basin_distance)
        stage["selected_start_indices"] = [r["start_index"] for r in selected]
        phase = "refinement"
        began = time.perf_counter()
        results = {}
        for rank, row in enumerate(selected):
            check()
            if progress:
                progress(.5 + .5 * rank / len(selected), f"P60 {model_id} full-data TRF {rank+1}/{len(selected)}")
            q = np.asarray(row["coordinates"])
            q = np.clip(q, np.nextafter(lower, upper), np.nextafter(upper, lower))
            record = dict(start_index=row["start_index"], status="failed", error=None,
                          initial_coordinates=q.tolist(), full_points=len(times))
            stage["refinements"].append(record)
            now = time.perf_counter()
            try:
                if monitored is not None:fit_start(model_id+' refinement',row['start_index']+1,len(starts),'linear')
                with monitored.objective_scope('refinement',range(len(times))) if monitored is not None else nullcontext():
                    opt = full_solver(full_residual, q, bounds=(lower, upper), method="trf",loss="linear", max_nfev=max_nfev)
                check()
                completed = (bool(opt.success) and int(opt.status) > 0
                             and np.all(np.isfinite(opt.x)) and np.all(np.isfinite(opt.fun))
                             and np.asarray(opt.fun).shape == (len(times),)
                             and np.asarray(opt.jac).shape == (len(times), len(lower))
                             and np.all(np.isfinite(opt.jac)))
                params, basin = physical_parameters(opt.x)
                record.update(status="completed" if completed else "nonconverged",
                              exit_status=int(opt.status), nfev=int(opt.nfev), params=params,
                              coordinates=opt.x.tolist(), basin_coordinates=list(basin),
                              full_data_sse=float(opt.fun @ opt.fun) if np.all(np.isfinite(opt.fun)) else None)
                pending=(monitored is not None and candidate_eligible(opt) and np.shape(opt.fun)==(len(times),)
                    and np.all(np.isfinite(opt.fun)) and np.all(np.isfinite(opt.jac)))
                if pending and not completed:record['status']='candidate_pending_global_polish'
                if completed or pending:
                    results[row["start_index"]] = opt
            except (ArithmeticError, ValueError) as exc:
                check()
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if monitored is not None:fit_start_finished()
                record["wall_s"] = time.perf_counter() - now
        stage["refinement_wall_s"] = time.perf_counter() - began
        stage["refinement_total_nfev"] = sum(r.get("nfev", 0) for r in stage["refinements"])
        if not results:
            raise NoCompletedRefinement("P60 has no completed full-data TRF refinement")
        stage["status"] = "completed" if monitored is None else "pending_global_polish"
        return results, stage
    except BaseException as exc:
        stage[phase + "_wall_s"] = time.perf_counter() - began
        stage["status"] = "cancelled" if isinstance(exc, CancelledError) else "failed"
        stage["error"] = f"{type(exc).__name__}: {exc}"
        exc.coarse_to_fine_audit = ledger.to_dict()
        raise
