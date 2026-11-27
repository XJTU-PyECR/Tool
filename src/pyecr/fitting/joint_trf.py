"""Independent full-row joint TRF engine; no legacy staged fitting or screening."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from pyecr.fitting.iteration_observation import observed_least_squares as least_squares
from pyecr.fitting.progress import fit_start, fit_start_finished, fit_candidate, count_fit_call, fit_counters_available

@dataclass
class JointEndpoint:
    index: int
    params: dict
    coordinates: np.ndarray
    prediction: np.ndarray
    residuals: np.ndarray
    sse: float
    jacobian: np.ndarray
    status: int
    nfev: int

def solve_joint(*, times, observed, names, lower, upper, starts, decode, predict,
                jacobian, max_nfev, tolerance, phase, cancel_check=None, progress=None):
    times, observed = np.asarray(times), np.asarray(observed)
    if (times.ndim != 1 or times.shape != observed.shape or times.size <= len(names)
            or times.size < 3 or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(observed))):
        raise ValueError("Joint fit requires all aligned finite increasing eligible rows and N>p")
    def check():
        if cancel_check is not None: cancel_check()
    def residual(q):
        check()
        count_fit_call("residual")
        count_fit_call("forward")
        return predict(decode(q)) - observed
    def jac(q):
        check()
        count_fit_call("jacobian")
        return jacobian(decode(q))
    fit_counters_available()
    endpoints, audit = [], []
    for i, start in enumerate(starts):
        check()
        row = {"start_index": i, "initial_coordinates": list(map(float, start)),
               "initial_parameters": decode(start), "published_candidate": False}
        audit.append(row)
        fit_start(phase, i+1, len(starts), "linear")
        try:
            if progress: progress(i/len(starts), f"{phase} full-time joint TRF {i+1}/{len(starts)}")
            result = least_squares(residual, start, jac=jac if jacobian is not None else "3-point",
                _observation_context={'names':names,'kinds':tuple('log10' if n.startswith('k_') else 'linear' for n in names),'stage_id':phase},
                method="trf", loss="linear", bounds=(lower, upper), x_scale="jac",
                max_nfev=max_nfev, ftol=tolerance, xtol=tolerance, gtol=tolerance)
            check()
            from .production_stop import candidate_eligible
            valid = bool(candidate_eligible(result)
                and np.shape(result.x) == (len(names),) and np.shape(result.fun) == observed.shape
                and np.shape(result.jac) == (times.size,len(names))
                and np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.fun)) and np.all(np.isfinite(result.jac)))
            row.update(success=bool(result.success), status=int(result.status), nfev=int(result.nfev),
                       final_coordinates=list(map(float,result.x)) if np.all(np.isfinite(result.x)) else None)
            if not valid:
                row["reason"] = "nonconverged/nonfinite/wrong full-row shape"
                continue
            params = decode(result.x)
            prediction = np.asarray(predict(params))
            r = observed-prediction
            j = np.asarray(jacobian(params) if jacobian is not None else result.jac)
            sse = float(r@r)
            if prediction.shape != observed.shape or not np.all(np.isfinite(prediction)) or not np.isfinite(sse):
                row["reason"] = "invalid full-row scientific finalization"
                continue
            row.update(params=params, SSE=sse, jacobian_shape=list(j.shape), published_candidate=result.status>0,
                       fit_row_scope="full_fit_eligible")
            endpoints.append(JointEndpoint(i,params,np.array(result.x),prediction,r,sse,j,int(result.status),int(result.nfev)))
            fit_candidate(sse,sse/2, params={n: params[n] for n in names},
                          model_id='M8_DUAL_K' if phase == 'M8' else 'M3_DUAL_HO_DK',
                          observation_count=len(observed), native_status=int(result.status))
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            row.update(success=False, reason=f"{type(exc).__name__}: {exc}")
        finally:
            fit_start_finished()
    check()
    if not endpoints:
        error = RuntimeError(f"All {phase} full-data joint TRF starts failed or did not converge")
        error.joint_start_audit = audit
        raise error
    endpoints.sort(key=lambda e:(e.sse,e.index))
    return endpoints, audit
