"""Shared H/O D+k inversion with curve-specific signed observation terms.

Linear loss profiles the bounded amplitudes/baselines on every observation.
The final covariance uses the full joint Jacobian, including all fitted local
terms, in reported log10-kinetic / linear-observation coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from pyecr.config import AnalysisConfig
from pyecr.fitting.global_fit import (
    GlobalCurve, GlobalFitConfig, GlobalFitResult, _Candidate,
    _automatic_baseline_bounds, _build_fit_result, _validate_curves,
)
from pyecr.fitting.reproducibility import record_randomness
from pyecr.models.factory import ModelFactory
from pyecr.models.dual_ho import DualHOModel
from pyecr.models.ids import ModelID
from pyecr.models.root_cache import RootCacheScope
from pyecr.uncertainty.local import estimate_local_uncertainty

KINETICS = ("D_H_cm2_s", "D_O_cm2_s", "k_H_cm_s", "k_O_cm_s")
LOCALS = ("baseline", "A_H", "A_O")
__all__ = ["fit_global_dual_ho"]


@dataclass
class _Layout:
    curve: GlobalCurve
    model: DualHOModel
    rows: slice
    fixed: dict[str, float]
    columns: dict[str, int]


class _Problem:
    def __init__(self, curves, template, curve_configs, policy, check):
        self.policy, self.check = policy, check
        self.label = template.model_specific.get("dual_ho", {}).get("label_constraint", "D_H>=D_O")
        if self.label == "dh_ge_do":
            self.label = "D_H>=D_O"
        if self.label not in {"none", "D_H>=D_O"}:
            raise ValueError("Global dual label_constraint must be D_H>=D_O or none")
        bounds = [template.parameter_bounds[name] for name in KINETICS]
        if any(lo <= 0 or lo >= hi for lo, hi in bounds):
            raise ValueError("Global dual kinetic bounds must be positive and increasing")
        self.kinetic_lower = np.log10([b[0] for b in bounds])
        self.kinetic_upper = np.log10([b[1] for b in bounds])
        initial = np.array([np.log10(template.model.initial_values[name])
                            if name in template.model.initial_values else (lo+hi)/2
                            for name, lo, hi in zip(KINETICS, self.kinetic_lower, self.kinetic_upper)])
        if np.any(initial < self.kinetic_lower) or np.any(initial > self.kinetic_upper):
            raise ValueError("Global dual initial kinetics lie outside their bounds")
        if self.label != "none" and initial[0] < initial[1]:
            raise ValueError("Global dual initial D_H must be >= D_O (or disable the label constraint)")
        self.lower = list(self.kinetic_lower)
        self.upper = list(self.kinetic_upper)
        self.initial = list(initial)
        self.names = list(KINETICS)
        self.spaces = ["log"] * 4
        self.layouts = []
        row = 0
        for curve in curves:
            cfg = curve_configs[curve.curve_id]
            model = ModelFactory().create(cfg)
            fixed, columns = {}, {}
            span = max(float(np.ptp(curve.signal)), 1e-8)
            for name in LOCALS:
                key = "y0" if name == "baseline" else name
                free = curve.fit_baseline if name == "baseline" else curve.fit_amplitude
                # Global's captured observation switches own these degrees of
                # freedom, independently of the source single-curve fit flags.
                default = float(curve.signal[0]) if name == "baseline" else span
                value = cfg.model.fixed_parameters.get(key, cfg.model.initial_values.get(key, default))
                if name == "baseline" and curve.baseline is not None:
                    value = curve.baseline
                value = float(value)
                if not np.isfinite(value) or (name != "baseline" and value < 0):
                    raise ValueError("Global dual observation parameters must be finite; amplitudes nonnegative")
                limits = cfg.parameter_bounds.get(key, _automatic_baseline_bounds(curve.signal)
                                                 if name == "baseline" else (0., 10*max(span, 1.)))
                if name != "baseline" and limits[0] < 0:
                    raise ValueError("Global dual amplitude bounds must be nonnegative")
                if free:
                    if not limits[0] <= value <= limits[1]:
                        raise ValueError(f"{curve.curve_id}: initial {key} lies outside its bounds")
                    columns[name] = len(self.names)
                    self.names.append(curve.curve_id + ":" + name)
                    self.spaces.append("linear")
                    self.lower.append(limits[0]); self.upper.append(limits[1]); self.initial.append(value)
                else:
                    fixed[name] = value
            self.layouts.append(_Layout(curve, model, slice(row, row+curve.time_s.size), fixed, columns))
            row += curve.time_s.size
        self.n_observations = row
        self.lower, self.upper, self.initial = map(np.asarray, (self.lower, self.upper, self.initial))
        self.opt_lower, self.opt_upper = self.lower.copy(), self.upper.copy()
        if self.label != "none":
            # Solver coordinates are log10(D_O), a fraction of the permitted
            # D_H interval, log10(k_H), log10(k_O). Reported coordinates remain
            # log10(D_H), log10(D_O), log10(k_H), log10(k_O).
            self.opt_lower[0] = self.kinetic_lower[1]
            self.opt_upper[0] = min(self.kinetic_upper[1], self.kinetic_upper[0])
            self.opt_lower[1], self.opt_upper[1] = 0., 1.
            if self.opt_lower[0] >= self.opt_upper[0]:
                raise ValueError("Global dual D bounds have no interval satisfying D_H>=D_O")
        self._basis_key = None
        self._basis_values = None

    def to_solver(self, q):
        result = q.copy()
        if self.label != "none":
            low = max(self.kinetic_lower[0], q[1])
            width = self.kinetic_upper[0] - low
            result[0] = q[1]
            result[1] = (q[0]-low)/width if width > 0 else 0.
        return np.clip(result, self.opt_lower[:len(q)], self.opt_upper[:len(q)])

    def from_solver(self, u):
        result = u.copy()
        if self.label != "none":
            low = max(self.kinetic_lower[0], u[0])
            result[0] = low + u[1]*(self.kinetic_upper[0]-low)
            result[1] = u[0]
        return result

    def start(self, index, rng):
        q = self.initial.copy()
        if index:
            if index % 3 == 0:
                q[:4] = rng.uniform(self.kinetic_lower, self.kinetic_upper)
            else:
                q[:4] = np.clip(q[:4]+rng.uniform(-2., 2., 4), self.kinetic_lower, self.kinetic_upper)
            if self.label != "none" and q[0] < q[1]:
                q[1] = min(q[1], self.kinetic_upper[0])
                q[0] = rng.uniform(max(self.kinetic_lower[0], q[1]), self.kinetic_upper[0])
        return self.to_solver(q)

    def basis(self, kinetics):
        self.check()
        if self._basis_key is not None and np.array_equal(kinetics, self._basis_key):
            return self._basis_values
        dh, do, kh, ko = 10.**kinetics
        values = []
        for layout in self.layouts:
            self.check()
            c, m = layout.curve, layout.model
            h = 1.-m.kernel.evaluate(c.time_s, dh, kh, term_tol=self.policy.term_tol, max_terms=self.policy.max_terms).F
            o = 1.-m.kernel.evaluate(c.time_s, do, ko, term_tol=self.policy.term_tol, max_terms=self.policy.max_terms).F
            values.append(np.column_stack((np.ones(c.time_s.size), m.sign_H*h, m.sign_O*o)))
        self._basis_key, self._basis_values = kinetics.copy(), values
        return values

    def local(self, q, layout):
        return {name: float(q[layout.columns[name]]) if name in layout.columns else layout.fixed[name]
                for name in LOCALS}

    def evaluate(self, q):
        residuals, predictions, locals_ = np.empty(self.n_observations), {}, {}
        for layout, basis in zip(self.layouts, self.basis(q[:4])):
            local = self.local(q, layout)
            prediction = basis @ np.array([local[name] for name in LOCALS])
            c = layout.curve
            predictions[c.curve_id] = prediction
            residuals[layout.rows] = c.noise_weight * (prediction-c.signal)
            locals_[c.curve_id] = {**local, "amplitude": layout.model.sign_H*local["A_H"]+layout.model.sign_O*local["A_O"]}
        return residuals, predictions, locals_

    def profile(self, u):
        q = self.initial.copy()
        q[:4] = self.from_solver(u)[:4]
        for layout, basis in zip(self.layouts, self.basis(q[:4])):
            self.check()
            free = list(layout.columns)
            if not free:
                continue
            target = layout.curve.signal.copy()
            for name, value in layout.fixed.items():
                target -= basis[:, LOCALS.index(name)]*value
            matrix = basis[:, [LOCALS.index(name) for name in free]]
            indices = [layout.columns[name] for name in free]
            lo, hi = self.lower[indices], self.upper[indices]
            values = np.linalg.lstsq(matrix, target, rcond=None)[0]
            if np.any(values < lo) or np.any(values > hi):
                solution = lsq_linear(matrix, target, bounds=(lo, hi), tol=1e-11, max_iter=200)
                if not solution.success:
                    raise RuntimeError("Global dual bounded observation solve failed")
                values = solution.x
            q[indices] = values
        return q

    def jacobian(self, q):
        # Derivatives are taken in reported coordinates with local parameters
        # held fixed, even when optimization profiles them out.
        jac = np.zeros((self.n_observations, len(self.names)))
        step = self.policy.derivative_step_log10
        for col in range(4):
            self.check()
            upper, lower = q.copy(), q.copy()
            upper[col] += step; lower[col] -= step
            jac[:, col] = (self.evaluate(upper)[0]-self.evaluate(lower)[0])/(2*step)
        for layout, basis in zip(self.layouts, self.basis(q[:4])):
            for name, col in layout.columns.items():
                jac[layout.rows, col] = layout.curve.noise_weight*basis[:, LOCALS.index(name)]
        return jac


def fit_global_dual_ho(curves, *, template: AnalysisConfig,
                       curve_configs: Mapping[str, AnalysisConfig], policy: GlobalFitConfig,
                       cancel_check=None, progress_callback=None) -> GlobalFitResult:
    """Fit four shared kinetics; preserve each curve's signs and local terms."""
    curves = _validate_curves(curves)
    template.validate()
    if template.model_id is not ModelID.M3_DUAL_HO_DK or template.boundary.mode != "ideal_step":
        raise ValueError("Global dual requires an ideal-step M3_DUAL_HO_DK template")
    if set(template.model.fixed_parameters) & set(KINETICS):
        raise ValueError("Global dual requires all four shared kinetics to be free")
    if set(curve_configs) != {c.curve_id for c in curves}:
        raise ValueError("Global dual needs one captured configuration per curve")
    for cfg in curve_configs.values():
        cfg.validate()
        if cfg.model_id is not ModelID.M3_DUAL_HO_DK or cfg.boundary.mode != "ideal_step":
            raise ValueError("Global dual curve configurations must be ideal-step M3_DUAL_HO_DK")
        if set(cfg.model.fixed_parameters) & set(KINETICS):
            raise ValueError("Global dual does not support fixed source kinetics")
    check = cancel_check or (lambda: None)
    report = progress_callback or (lambda fraction, message: None)
    check()
    p = _Problem(curves, template, curve_configs, policy, check)
    if p.n_observations <= len(p.names):
        raise ValueError("Global dual needs more observations than free parameters")
    rng = np.random.default_rng(policy.random_seed)
    candidates, starts, failures = [], [], []
    projected = policy.loss == "linear"
    with RootCacheScope(root_mode="on_demand_roots", cancel_check=check) as roots:
        for index in range(policy.n_starts):
            check()
            report(.03+.78*index/policy.n_starts, f"Global dual multistart {index+1}/{policy.n_starts}")
            start = p.start(index, rng)
            starts.append(dict(start_index=index, coordinates=start.tolist()))
            def residual(u):
                check()
                q = p.profile(u) if projected else p.from_solver(u)
                return p.evaluate(q)[0]
            try:
                n = 4 if projected else len(p.names)
                opt = least_squares(residual, start[:n], bounds=(p.opt_lower[:n], p.opt_upper[:n]),
                                    loss=policy.loss, max_nfev=policy.max_nfev, x_scale="jac",
                                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
                q = p.profile(opt.x) if projected else p.from_solver(opt.x)
                r = p.evaluate(q)[0]
                jac = p.jacobian(q)
                sse = float(r @ r)
                if not np.isfinite(sse) or not np.all(np.isfinite(jac)):
                    raise FloatingPointError("Non-finite Global dual endpoint")
                candidates.append((float(opt.cost), _Candidate(q, r, jac, sse, int(opt.status), int(opt.nfev), bool(opt.success))))
                starts[-1].update(SSE=sse, objective_cost=float(opt.cost), exit_status=int(opt.status), nfev=int(opt.nfev))
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                failures.append(f"start {index}: {type(exc).__name__}: {exc}")
        if not candidates:
            raise RuntimeError("Global dual produced no valid candidate: "+"; ".join(failures))
        candidates.sort(key=lambda item: item[0])
        best = candidates[0][1]
        _, predictions, local = p.evaluate(best.q)
        cache = roots.diagnostics.to_dict()
    check()
    warnings = []
    if not best.success:
        warnings.append("GLOBAL_OPTIMIZER_NOT_CONVERGED: dual joint fit exhausted its evaluation budget.")
    if failures:
        warnings.append(f"GLOBAL_MULTISTART_FAILURES: {len(failures)} starts failed.")
    if policy.loss != "linear":
        warnings.append("GLOBAL_ROBUST_LOSS_COVARIANCE_APPROXIMATE: local ordinary least-squares covariance.")
    proximity = np.minimum(best.q[:4]-p.lower[:4], p.upper[:4]-best.q[:4])/(p.upper[:4]-p.lower[:4])
    local_distance = np.minimum(best.q[4:]-p.lower[4:], p.upper[4:]-best.q[4:])
    if np.any(proximity < 1e-5) or np.any(local_distance < 1e-8*np.maximum(1., np.abs(best.q[4:]))):
        warnings.append("GLOBAL_BOUND_PROXIMITY: local intervals may be invalid at an active parameter bound.")
    if p.label != "none" and best.q[0]-best.q[1] < 1e-5:
        warnings.append("GLOBAL_LABEL_BOUNDARY: D_H and D_O lie on the ordering boundary.")
    for component in ("A_H", "A_O"):
        if all(values[component] < .01*max(values["A_H"]+values["A_O"], 1e-30) for values in local.values()):
            warnings.append("GLOBAL_WEAK_COMPONENT: "+component+" is weak across every input curve.")
    shared = {name: float(10.**value) for name, value in zip(KINETICS, best.q[:4])}
    params = dict(shared)
    for layout in p.layouts:
        params.update({layout.curve.curve_id+":"+name: local[layout.curve.curve_id][name] for name in layout.columns})
    observed = np.concatenate([c.noise_weight*c.signal for c in curves])
    fitted = np.concatenate([c.noise_weight*predictions[c.curve_id] for c in curves])
    fit = _build_fit_result(observed, fitted, params=params, param_space=dict(zip(p.names, p.spaces)),
                            exit_status=best.status, nfev=best.nfev, jacobian=best.jacobian, warnings=warnings)
    uncertainty = estimate_local_uncertainty(fit, confidence_level=policy.confidence_level,
                                            condition_threshold=policy.condition_number_limit)
    fit = record_randomness(uncertainty.annotated_fit_result, seed=policy.random_seed,
                            stochastic=policy.n_starts > 1, algorithm="global-dual-profile-PCG64-v1", starts=starts)
    residuals = {c.curve_id: c.signal-predictions[c.curve_id] for c in curves}
    diagnostics = dict(model_id=ModelID.M3_DUAL_HO_DK.value, n_curves=len(curves),
        curve_ids=tuple(c.curve_id for c in curves), n_observations=p.n_observations,
        n_free_parameters=len(p.names), parameter_order=tuple(p.names), shared_parameters=KINETICS,
        noise_weights={c.curve_id: c.noise_weight for c in curves}, weighted_objective=True,
        observation_solver="bounded_linear_projection" if projected else "joint_robust",
        label_constraint=p.label, root_cache=cache, failed_start_messages=tuple(failures),
        candidate_SSE=tuple(c.SSE for _, c in candidates), candidate_cost=tuple(cost for cost, _ in candidates),
        selection_objective=policy.loss, parameter_bounds={name: (float(lo),float(hi))
            for name,lo,hi in zip(p.names,p.lower,p.upper)},
        bounds_space="log10 kinetics / linear observations",
        solver_coordinate_order=(("log10_D_O", "D_H_interval_fraction", "log10_k_H", "log10_k_O")
                                  if p.label!="none" else tuple("log10_"+name for name in KINETICS)),
        curve_params=local, signs={x.curve.curve_id: (x.model.sign_H, x.model.sign_O) for x in p.layouts},
        shared_scope="one temperature; four shared kinetics; independent local observation terms")
    report(1., "Global dual fit complete")
    check()
    return GlobalFitResult(ModelID.M5_GLOBAL, fit, uncertainty, shared, local, predictions, residuals,
        {c.curve_id: c.noise_weight*residuals[c.curve_id] for c in curves}, uncertainty.covariance,
        uncertainty.correlation_matrix, uncertainty.covariance[:4, :4],
        {name: uncertainty.confidence_intervals[name] for name in KINETICS}, diagnostics,
        tuple(dict.fromkeys((*fit.warnings, *warnings))))
