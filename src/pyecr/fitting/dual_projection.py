"""Bounded variable projection over the production H/O unit fractions.

The objective remains the same full-observation unweighted linear-loss SSE.
Only kinetic coordinates go to TRF; final inference uses the full physical
Jacobian from dual_ho, including every fitted observation coefficient.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from pyecr.fitting.progress import count_fit_call

VERSION = "dual-bounded-linear-projection/1"


@dataclass(frozen=True)
class ProjectedPoint:
    coordinates: np.ndarray
    params: dict
    prediction: np.ndarray
    residuals: np.ndarray


class ProjectedDualObjective:
    def __init__(self, evaluator, parameterization, cancel_check=None):
        self.evaluator = evaluator
        self.parameterization = parameterization
        self.cancel_check = cancel_check or (lambda: None)
        self.indices = np.array([0, 1, 4, 5] if parameterization.is_m3 else [0, 1])
        self.lower = parameterization.lower[self.indices]
        self.upper = parameterization.upper[self.indices]
        self.template = (parameterization.lower+parameterization.upper)/2
        self.names = ("A_H", "A_O", "y0") if parameterization.config.fit_baseline else ("A_H", "A_O")
        bounds = parameterization.physical_bounds
        self.linear_lower = np.array([bounds[n][0] for n in self.names])
        self.linear_upper = np.array([bounds[n][1] for n in self.names])
        self.calls = self.inner_iterations = self.active_bound_calls = self.rank_deficient_calls = 0
        self._last_q = self._last_point = None

    def project(self, nonlinear):
        self.cancel_check()
        nonlinear = np.asarray(nonlinear, dtype=float)
        if nonlinear.shape != self.lower.shape or not np.all(np.isfinite(nonlinear)):
            raise ValueError("invalid projected kinetic coordinates")
        if self._last_q is not None and np.array_equal(nonlinear, self._last_q):
            return self._last_point
        q = self.template.copy()
        q[self.indices] = nonlinear
        p = self.parameterization.decode(q)
        ctx = self.evaluator.context
        count_fit_call("forward")
        result = ctx.model.evaluate(ctx.processed_time_s, **p, kernel_options=ctx.kernel_options)
        columns = {"A_H": ctx.sign_H*(1-result.F_H),
                   "A_O": ctx.sign_O*(1-result.F_O),
                   "y0": np.ones_like(ctx.processed_time_s)}
        design = np.column_stack([columns[n] for n in self.names])
        rhs = ctx.processed_signal if "y0" in self.names else ctx.processed_signal-p["y0"]
        self.cancel_check()
        inner = lsq_linear(design, rhs, bounds=(self.linear_lower, self.linear_upper),
                           method="bvls", lsq_solver="exact", tol=1e-12, max_iter=100)
        self.cancel_check()
        if not inner.success or not np.all(np.isfinite(inner.x)):
            raise RuntimeError("dual linear subproblem did not converge")
        feasible = np.clip(inner.x, self.linear_lower, self.linear_upper)
        tolerance = 64*np.finfo(float).eps*np.maximum(1., np.maximum(abs(self.linear_lower), abs(self.linear_upper)))
        if np.any(abs(feasible-inner.x) > tolerance):
            raise RuntimeError("dual linear subproblem returned infeasible coefficients")
        p.update(zip(self.names, map(float, feasible)))
        q[2:4] = np.log10([p["A_H"], p["A_O"]])
        if "y0" in self.names:
            q[-1] = p["y0"]
        # Keep the actual log/power round-trip used by the production chart.
        p = self.parameterization.decode(q)
        prediction = p["y0"] + p["A_H"]*columns["A_H"] + p["A_O"]*columns["A_O"]
        residual = prediction-ctx.processed_signal
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("nonfinite dual projected residual")
        self.calls += 1
        self.inner_iterations += int(inner.nit)
        self.active_bound_calls += int(np.any(inner.active_mask))
        self.rank_deficient_calls += int(np.linalg.matrix_rank(design) < design.shape[1])
        for array in (q, prediction, residual):
            array.setflags(write=False)
        point = ProjectedPoint(q, p, prediction, residual)
        self._last_q = nonlinear.copy()
        self._last_point = point
        return point

    def residual(self, q):
        count_fit_call("residual")
        return self.project(q).residuals.copy()

    def decode(self, q):
        return dict(self.project(q).params)

    @property
    def solver_problem(self):
        from pyecr.fitting.solver_strategy import SolverProblem, SolverParameterization
        constrained = self.parameterization.config.label_constraint.value == "D_H>=D_O"
        names = ("log10_D_O", "eta") if constrained else ("log10_D_H", "log10_D_O")
        if self.parameterization.is_m3:
            names += ("log10_k_H", "log10_k_O")
        codec = SolverParameterization(names, self.decode,
            {} if "y0" in self.names else {"y0": self.parameterization.fixed_baseline},
            VERSION, len(self.parameterization.lower))
        return SolverProblem(self.residual, self.residual, self.lower, self.upper, codec,
                             "M3" if self.parameterization.is_m3 else "M2",
                             self.parameterization.config.random_seed)

    def diagnostics(self):
        return dict(version=VERSION, nonlinear_coordinates=len(self.indices),
                    fitted_parameter_count=len(self.parameterization.lower),
                    inner_free_parameters=list(self.names), inner_calls=self.calls,
                    inner_iterations=self.inner_iterations, active_bound_calls=self.active_bound_calls,
                    rank_deficient_calls=self.rank_deficient_calls,
                    inference="full physical Jacobian; projected coefficients count as fitted parameters")
