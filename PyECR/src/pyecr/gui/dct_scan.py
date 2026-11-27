"""Read-only adapters for stored lambda candidates; never run an inverse solve."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScanCandidate:
    lam: float
    score: float | None
    tau: tuple = ()
    chi: tuple = ()
    mass: tuple = ()
    reconstruction: tuple = ()
    rmse: float | None = None
    note: str = ''


@dataclass(frozen=True)
class LambdaScan:
    selected: float | None = None
    candidates: tuple = ()
    time: tuple = ()
    observed: tuple = ()
    signed: bool = False
    scope: str = ''
    score_name: str = 'GCV'
    manual_draft_allowed: bool = True


def monotonic_scan(presentation):
    if presentation is None:
        return LambdaScan(scope='Run DCT to create a stored regularization scan.')
    r = presentation.result
    saved = {float(s['regularization_lambda']): s for s in r.diagnostics.get('lambda_scan', {}).get('candidates', ())}
    rows = []
    lambdas = () if r.gcv_lambdas is None else r.gcv_lambdas
    for i, lam in enumerate(lambdas):
        s = saved.get(float(lam))
        score = float(r.gcv_scores[i])
        rows.append(ScanCandidate(float(lam), score if np.isfinite(score) else None,
                    tuple(r.tau_s) if s else (), tuple(s['chi']) if s else (), tuple(s['mass']) if s else (),
                    rmse=None if s is None else s['RMSE'],
                    note=('Stored spectrum' if s.get('solver_success',True) else 'Stored candidate; solver did not converge') if s else 'Older result: rerun GCV to retain candidate spectra'))
    if not rows:
        rows.append(ScanCandidate(r.regularization_lambda, None, tuple(r.tau_s), tuple(r.chi),
                                 tuple(r.mass_distribution), tuple(r.y_fit), r.RMSE, 'Manual / selected result only'))
    return LambdaScan(r.regularization_lambda, tuple(rows), tuple(presentation.time_s), tuple(presentation.signal),
                      scope='Monotonic nonnegative GCV is a tuning heuristic; correlated residuals and constraints affect its interpretation.')


def signed_scan(analysis):
    if analysis is None or analysis.selected_result is None:
        return LambdaScan(scope='Select a saved Signed run with a usable result.')
    selected = analysis.selected_result
    rows = []
    for c in analysis.candidates:
        r = c.result
        ready = r.state.ready
        residual = np.asarray(r.residuals)
        rows.append(ScanCandidate(r.selected_lambda, c.gcv_score,
                    tuple(r.tau_s) if ready else (), tuple(r.chi) if ready else (), (),
                    tuple(r.reconstruction) if ready else (),
                    float(np.sqrt(np.mean(residual**2))) if ready and residual.size else None,
                    c.failure or 'Stored spectrum'))
    return LambdaScan(selected.selected_lambda, tuple(rows), tuple(selected.source.time_s),
                      tuple(selected.source.signal), True,
                      'Signed terminal-deviation spectra in normalized Y units. GCV is conditional on the recorded terminal, scale and weights.')


def projected_reconstruction(scan, candidate):
    if candidate.reconstruction:
        return candidate.reconstruction
    if not candidate.mass:
        return ()
    time, tau, mass = np.asarray(scan.time), np.asarray(candidate.tau), np.asarray(candidate.mass)
    # Bounded temporary allocation; keep every stored observation.
    values = np.empty(len(time))
    for start in range(0, len(time), 2048):
        chunk = slice(start, start+2048)
        values[chunk] = 1. - np.exp(-time[chunk, None]/tau) @ mass
    return tuple(values)


def displayed_candidates(scan, index, all_candidates=False):
    """Selected optimum, browsing choice and neighbors; preserve lambda order."""
    if not scan.candidates:
        return ()
    optimum = min(range(len(scan.candidates)), key=lambda i: abs(scan.candidates[i].lam-scan.selected)) if scan.selected is not None else 0
    indices = range(len(scan.candidates)) if all_candidates else sorted({0,len(scan.candidates)-1,optimum,index,max(0,index-1),min(len(scan.candidates)-1,index+1)})
    return tuple(scan.candidates[i] for i in indices if scan.candidates[i].tau)
