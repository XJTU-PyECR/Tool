"""Full-data M7 adapter. No screening, diffusion kernel or legacy fit changes."""
from pyecr.fitting.reproducibility import seeded_analysis_fit,record_randomness

from dataclasses import replace
from math import isfinite
from numbers import Integral
from typing import Mapping
import numpy as np
from pyecr.config import AnalysisConfig
from pyecr.fitting.base import FitCandidate, FitResult
from pyecr.fitting.single import _fit_statistics, _bound_warnings
from pyecr.fitting.iteration_observation import observed_least_squares as least_squares, observe_direct_forward
from pyecr.fitting.solver_strategy import SolverOptions
from pyecr.fitting.coarse_to_fine import current_screening
from pyecr.fitting.progress import (fit_counters_available, fit_start, fit_start_finished,
                                   fit_candidate, count_fit_call)
from pyecr.io.dataset import EcrDataset
from pyecr.models.ids import ModelID
from pyecr.models.factory import ModelFactory
from pyecr.models.surface_k import MODEL_VERSION, CONTRACT_VERSION, EXPOSURE_RULE
from pyecr.uncertainty.local import estimate_local_uncertainty

def _integer(options, name, default):
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if name == "random_seed" else 1):
        raise ValueError(f"surface_k.{name} must be a {'nonnegative' if name == 'random_seed' else 'positive'} integer")
    return int(value)

@seeded_analysis_fit
def fit_surface_k(dataset: EcrDataset, config: AnalysisConfig, *, cancel_check=None, progress_callback=None):
    config.validate()
    if config.model_id is not ModelID.M7_SINGLE_K:
        raise ValueError("M7 fitter requires M7_SINGLE_K")
    if config.fit.loss != "linear" or SolverOptions.from_config(config).strategy != "trf":
        raise ValueError("M7 supports ideal_step, linear loss and TRF only")
    if current_screening() is not None:
        raise ValueError("M7 does not support P60 screening")
    if set(config.model_specific) - {"surface_k", "local_solver"}:
        raise ValueError("Unsupported model_specific composition for M7")
    opts = config.model_specific.get("surface_k", {})
    if not isinstance(opts, Mapping) or set(opts) - {"n_starts", "n_candidates", "random_seed"}:
        raise ValueError("Unsupported surface_k options; analytic Jacobian is mandatory")
    n_starts = _integer(opts, "n_starts", 8)
    n_candidates = _integer(opts, "n_candidates", 5)
    seed = _integer(opts, "random_seed", 20260915)
    model = ModelFactory().create(config)
    times = dataset.processed_time
    observed = dataset.processed_signal
    if (times.size < 3 or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(observed))):
        raise ValueError("M7 needs aligned finite nonnegative increasing fit times")
    if dataset.metadata.get("time_unit", "s") != "s":
        raise ValueError("M7 requires fit time in seconds; use explicit preprocessing conversion")
    normalized = dataset.metadata.get("signal_type") == "normalized"
    fixed = dict(config.model.fixed_parameters)
    if "k_cm_s" in fixed:
        raise ValueError("M7 requires k_cm_s to be free")
    declared = config.model.free_parameters
    names = tuple(p for p in ("k_cm_s", "amplitude", "baseline") if p in (
        declared if declared else ({"k_cm_s"} if normalized else {"k_cm_s", "amplitude", "baseline"}) - set(fixed)))
    if "k_cm_s" not in names:
        raise ValueError("M7 free_parameters must include k_cm_s")
    if normalized and (set(names) != {"k_cm_s"} or fixed.get("amplitude", 1.0) != 1 or fixed.get("baseline", 0.0) != 0):
        raise ValueError("Normalized M7 fixes amplitude=1 and baseline=0; do not renormalize silently")
    fixed.update({p: fixed.get(p, default) for p, default in (("amplitude", 1.0), ("baseline", 0.0)) if p not in names})
    if times.size <= len(names):
        raise ValueError("M7 requires N > all free kinetic and observation parameters")
    scale = max(float(np.ptp(observed)), float(np.max(np.abs(observed))), 1.0)
    default_bounds = {"k_cm_s": (1e-10, 1.0), "amplitude": (-10 * scale, 10 * scale),
                      "baseline": (float(np.min(observed) - 2 * scale), float(np.max(observed) + 2 * scale))}
    bounds = {p: config.model.parameter_bounds.get(p, default_bounds[p]) for p in names}
    lower = np.array([np.log10(bounds[p][0]) if p == "k_cm_s" else bounds[p][0] for p in names])
    upper = np.array([np.log10(bounds[p][1]) if p == "k_cm_s" else bounds[p][1] for p in names])
    guessed = {"k_cm_s": 1e-4, "amplitude": float(observed[-1] - observed[0]), "baseline": float(observed[0])}
    initial = {p: config.model.initial_values.get(p, guessed[p]) for p in names}
    q0 = np.array([np.log10(initial[p]) if p == "k_cm_s" else initial[p] for p in names])
    if not np.all(np.isfinite(q0)) or np.any(q0 < lower) or np.any(q0 > upper):
        raise ValueError("M7 initial values must lie within physical parameter bounds")
    def check():
        if cancel_check is not None:
            cancel_check()
    def decode(q):
        return {**fixed, **{p: float(10.0 ** v) if p == "k_cm_s" else float(v) for p, v in zip(names, q, strict=True)}}
    def residual(q):
        check()
        count_fit_call("residual")
        count_fit_call("forward")
        return model.predict(times, decode(q)) - observed
    def jac(q):
        check()
        count_fit_call("jacobian")
        return model.jacobian(times, decode(q), names)
    check()
    fit_counters_available()
    rng = np.random.default_rng(seed)
    starts = np.vstack((q0, rng.uniform(np.nextafter(lower, upper), np.nextafter(upper, lower), size=(n_starts - 1, len(names)))))
    completed = []
    audit = []
    for index, start in enumerate(starts):
        check()
        fit_start("M7", index + 1, n_starts, "linear")
        if progress_callback:
            progress_callback(index / n_starts, f"M7 full-data TRF {index + 1}/{n_starts}")
        try:
            result = least_squares(residual, start, method="trf", jac=jac, bounds=(lower, upper),
                                   _observation_context={'names':names,'kinds':tuple('log10' if n=='k_cm_s' else 'linear' for n in names),'stage_id':'M7'},
                                   loss="linear", max_nfev=config.fit.max_nfev,
                                   ftol=config.solver.relative_tolerance,
                                   xtol=config.solver.relative_tolerance,
                                   gtol=config.solver.relative_tolerance, x_scale="jac")
            check()
            from .production_stop import candidate_eligible
            valid = bool(candidate_eligible(result)
                and np.shape(result.x) == (len(names),)
                and np.shape(result.fun) == observed.shape and np.shape(result.jac) == (times.size, len(names))
                and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.fun)) and np.all(np.isfinite(result.jac)))
            audit.append({"start_index": index, "initial_coordinates": start.tolist(),
                          "success": bool(result.success), "status": int(result.status), "nfev": int(result.nfev), "published_candidate": valid and result.status>0})
            if valid:
                theta = decode(result.x)
                observe_direct_forward()
                prediction = model.predict(times, theta)
                r = observed - prediction
                sse = float(r @ r)
                j = jac(result.x)
                if not isfinite(sse):
                    audit[-1]["published_candidate"] = False
                    continue
                audit[-1].update(params={p: theta[p] for p in names}, SSE=sse,
                                  jacobian_shape=list(j.shape), fit_row_scope="full_fit_eligible")
                completed.append((sse, index, result, prediction, r, j, theta))
                fit_candidate(sse, sse / 2, params={p: theta[p] for p in names},
                              model_id=config.model_id, observation_count=len(observed),
                              native_status=int(result.status))
        except (ArithmeticError, ValueError, RuntimeError) as exc:
            audit.append({"start_index": index, "initial_coordinates": start.tolist(), "success": False,
                          "published_candidate": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            fit_start_finished()
    check()
    if not completed:
        raise RuntimeError("All M7 full-data TRF starts failed or did not converge")
    completed.sort(key=lambda c: (c[0], c[1]))
    sse, index, result, prediction, r, j, theta = completed[0]
    from .production_stop import polish_ensemble
    polished=polish_ensemble('M7')
    if polished is not None:
        result,index=polished;theta=decode(result.x)
        observe_direct_forward();prediction=model.predict(times,theta);r=observed-prediction;sse=float(r@r);j=jac(result.x)
    statistics, warnings = _fit_statistics(observed, r, n_parameters=len(names))
    warnings = list(warnings) + list(_bound_warnings(names, result.x, lower, upper))
    if theta["amplitude"] == 0:
        warnings.append("IDENTIFIABILITY_ZERO_AMPLITUDE: k is not identifiable")
    rate, tau = model.kernel.rates(theta["k_cm_s"])
    if times[-1] < tau:
        warnings.append("SURFACE_K_SHORT_WINDOW: observed window does not cover one surface-exchange time constant")
    geometry = config.model.geometry
    from pyecr.report.snapshot import _sha256_json, dataset_sha256
    with np.errstate(over="ignore"):
        saturated = bool(np.any(~np.isfinite(rate * times)))
    metadata = {"model_version": MODEL_VERSION, "contract_version": CONTRACT_VERSION,
        "inference_mode": "analytic_surface_controlled", "full_dims_cm": list(geometry.full_dims_cm),
        "active_dimensions": list(geometry.active_dimensions), "exposure_rule": EXPOSURE_RULE,
        "g_cm_inv": model.kernel.g_cm_inv, "lambda_s_inv": rate, "tau_surface_exchange_s": tau,
        "tau_definition": "1/(k*S_exposed/V); not the P51 finite-D dominant material mode",
        "parameter_roles": {**{p: "free" for p in names}, **{p: "fixed" for p in fixed},
            "g_cm_inv": "derived", "lambda_s_inv": "derived", "tau_surface_exchange_s": "derived"},
        "fixed_parameters": fixed, "free_parameter_count": len(names), "kinetic_free_count": 1,
        "observation_free_count": len(names) - 1, "parameter_bounds": {p: list(bounds[p]) for p in names},
        "observation_mode": "normalized" if normalized else "raw", "initial_values": initial,
        "initial_value_sources": {p: "explicit_config" if p in config.model.initial_values else "full_data_endpoint_guess" if p != "k_cm_s" else "contract_default" for p in names},
        "original_raw_indices": np.flatnonzero(dataset.mask).tolist(), "processed_indices": list(range(times.size)),
        "fit_eligible_count": times.size, "eligible_rule": "all current processed rows; no additional screening/filtering",
        "dataset_sha256": dataset_sha256(dataset), "config_sha256": _sha256_json(config.to_dict()),
        "units": {"k_cm_s": "cm/s", "g_cm_inv": "cm^-1", "lambda_s_inv": "s^-1", "tau_surface_exchange_s": "s",
                  "amplitude": str(dataset.metadata.get("signal_unit", "signal")), "baseline": str(dataset.metadata.get("signal_unit", "signal"))},
        "ci_scope": "conditional_on_fixed_inputs", "ci_conditions": {"geometry": geometry.to_dict(), **fixed},
        "bi": {"available": False, "reason": "D_not_estimated"}, "jacobian": "analytic; d(prediction-observed)/d(log10 k, linear observations)",
        "random_seed": seed, "start_rule": "explicit-or-recorded initial followed by PCG64 uniform optimizer-coordinate draws",
        "max_terms": "not_used_by_analytic_kernel", "fit_row_scope": "full_fit_eligible", "starts": audit,
        "solver_method": "trf", "max_nfev": config.fit.max_nfev,
        "stopping_tolerance": config.solver.relative_tolerance,
        "numerical_saturation": saturated }
    candidates = tuple(FitCandidate(i, {p: th[p] for p in names}, ss, int(res.status), int(res.nfev), bool(res.success),
                                   tuple(float(v) for v in starts[i])) for ss, i, res, _pr, _r, _j, th in completed[:n_candidates])
    fit = FitResult(params={p: theta[p] for p in names}, param_space={p: "log" if p == "k_cm_s" else "linear" for p in names},
        y_fit=prediction, residuals=r, SSE=sse, exit_status=int(result.status), nfev=int(result.nfev),
        jacobian=j, warnings=tuple(warnings), candidates=candidates, diagnostics={"model_id": "M7_SINGLE_K", "surface_k": metadata}, **statistics)
    uncertainty = estimate_local_uncertainty(fit)
    ci_ok = uncertainty.reliable_separation and not any("PARAMETER_BOUND_HIT" in w for w in warnings)
    metadata["ci_available"] = ci_ok
    metadata["ci_unavailable_reason"] = None if ci_ok else "rank/conditioning/boundary prevents reliable local intervals"
    k_ci = uncertainty.confidence_intervals.get("k_cm_s")
    ci_ok = bool(ci_ok and k_ci is not None and isfinite(k_ci.lower) and isfinite(k_ci.upper) and k_ci.lower > 0)
    metadata["ci_available"] = ci_ok
    if not ci_ok:
        metadata["ci_unavailable_reason"] = "rank/conditioning/boundary/nonfinite interval prevents reliable local intervals"
    if ci_ok and k_ci is not None and isfinite(k_ci.lower) and isfinite(k_ci.upper) and k_ci.lower > 0:
        metadata["tau_surface_exchange_ci_s"] = [1 / (model.kernel.g_cm_inv * k_ci.upper), 1 / (model.kernel.g_cm_inv * k_ci.lower)]
        metadata["lambda_ci_s_inv"] = [model.kernel.g_cm_inv * k_ci.lower, model.kernel.g_cm_inv * k_ci.upper]
    check()
    if progress_callback:
        progress_callback(1.0, "M7 full-data fit complete")
    final=replace(uncertainty.annotated_fit_result, diagnostics={"model_id": "M7_SINGLE_K", "surface_k": metadata})
    return record_randomness(final,seed=seed,stochastic=n_starts>1,algorithm='surface-k-explicit-then-PCG64-v1',
                             starts=[{'start_index':i,'coordinates':q.tolist()} for i,q in enumerate(starts)])
