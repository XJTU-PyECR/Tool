"""Terminal-referenced, unrestricted signed Tikhonov inversion (Manual lambda).

This pure entry point consumes a frozen, already normalized source. It never
normalizes, fits a physical model, changes observations, or publishes GUI state.
The legacy nonnegative fit_dct and DCTResult remain separate.
"""
from __future__ import annotations

from uuid import uuid4

import numpy as np
from scipy import linalg
from pyecr.dct.execution import numeric_phase, check_cancel

from pyecr.contracts import (
    ContractError, DctSourceSnapshot, ResultState, SignedDctConfig,
    SignedDctResult, SignedDctSolveDiagnostics, SourceKind,
)
from pyecr.contracts.common import require

__all__ = ['fit_signed_dct']


def _operators(time_s, config):
    """signed-tikhonov/1: uniform log10 grid, trapezoid, interior curvature.

    L[k,k:k+3] = sqrt(h)/h**2 * [1,-2,1], k=0..M-3.
    There are NO extra endpoint rows/anchors or periodic wrapping. Thus linear
    and constant densities are in the penalty null space. The objective uses
    lambda*||L chi||**2, so the augmented rows use sqrt(lambda)*L.
    """
    check_cancel()
    g = config.grid
    x = np.linspace(np.log10(g.tau_min_s), np.log10(g.tau_max_s), g.n_tau)
    h = float((x[-1]-x[0])/(g.n_tau-1))
    tau = np.power(10., x)
    tau[0], tau[-1] = g.tau_min_s, g.tau_max_s
    require(h > 0 and np.all(np.diff(x) > 0) and np.all(np.diff(tau) > 0),
            'signed log10 grid collapses at floating-point precision')
    q = np.full(g.n_tau, h)
    q[[0, -1]] = h/2
    # t/tau may overflow to +inf for valid extreme times; exp(-inf)=0.
    with np.errstate(over='ignore', under='ignore'):
        a = np.exp(-np.asarray(time_s)[:, None]/tau[None, :])*q
    l = np.zeros((g.n_tau-2, g.n_tau))
    ix = np.arange(g.n_tau-2)
    scale = np.sqrt(h)/(h*h)
    l[ix, ix], l[ix, ix+1], l[ix, ix+2] = scale, -2*scale, scale
    check_cancel()
    return tau, x, q, a, l, h


def fit_signed_dct(source: DctSourceSnapshot, config: SignedDctConfig, *, linear_cache=None) -> SignedDctResult:
    """Solve unrestricted chi using augmented SVD or pivoted QR, b=0.

    Malformed types/weights raise; a valid but unavailable source, unsupported
    mode, or numeric failure returns an explicit result without fabricated data.
    Condition/rank describe the augmented numerical solve, NOT physical Fit or
    species reliability. No peak extraction, automatic lambda, or RNG is run.
    """
    if not isinstance(source, DctSourceSnapshot) or not isinstance(config, SignedDctConfig):
        raise TypeError('fit_signed_dct requires DctSourceSnapshot and SignedDctConfig')
    run_id = uuid4().hex

    def unavailable(status, reason):
        return SignedDctResult(schema_version=3 if config.schema_version==3 else 2, source=source, config=config,
                               state=ResultState(status=status, blocking_reasons=(reason,)), run_id=run_id)

    if config.grid.lambda_mode != 'manual':
        return unavailable('unsupported', 'P04 supports explicit Manual lambda only; GCV selection has not been implemented')
    try:
        source.require_usable()
    except ContractError as exc:
        return unavailable('unsupported', str(exc))
    if source.representation != 'signed_terminal' or source.source_kind is SourceKind.FIT_COMPONENT:
        return unavailable('unsupported', 'P04 requires recorded signed terminal normalization; component-parent/unit adapter is a later stage')
    if source.normalization.spec.reference_kind not in {'accepted_final_window', 'external_known_terminal'}:
        return unavailable('unsupported', 'Model-inferred terminal requires a real parent-evidence adapter; a ready flag or hashes alone are insufficient')
    if all(v == source.signal[0] for v in source.signal):
        return unavailable('unsupported', 'Constant normalized observations have no temporal information; review the source and terminal')
    require(not config.observation_weights or len(config.observation_weights) == len(source.time_s),
            'observation weights must cover every scientific observation')
    if config.schema_version==3:
        from pyecr.dct.signed_priors import fit_prior
        return fit_prior(source,config,run_id,linear_cache)
    if config.schema_version == 2:
        from pyecr.dct.signed_linear import fit_reused
        return fit_reused(source, config, run_id, linear_cache)
    weights = np.asarray(config.observation_weights or (1.,)*len(source.time_s))
    y = np.asarray(source.signal)
    lam = float(config.grid.regularization_lambda)
    driver = 'gelsd' if config.solver == 'augmented_svd' else 'gelsy'
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            tau, x, q, a, l, h, root_w, b, rhs, cutoff = _augmented_system(source, config)
            require(np.all(np.isfinite(b)) and np.all(np.isfinite(rhs)), 'nonfinite augmented system')
            with numeric_phase('signed_lstsq'):
                chi, _, rank, s = linalg.lstsq(b, rhs, cond=cutoff, lapack_driver=driver)
            if s is None:  # gelsy has no singular-value output; audit B, not B.T@B.
                s = linalg.svdvals(b)
            svd_rank = int(np.count_nonzero(s > s[0]*cutoff))
            require(rank > 0 and svd_rank > 0, 'numerically zero augmented rank')
            mass = chi*q
            reconstruction = a@chi
            residuals = y-reconstruction
            weighted_sse = float(np.dot(root_w*residuals, root_w*residuals))
            # Evaluate the exact augmented penalty, not a normal-equation product.
            penalty_vector = (np.sqrt(lam)*l)@chi
            penalty = float(np.dot(penalty_vector, penalty_vector))
            full_condition = None
            if s[-1] > 0:
                with np.errstate(over='ignore'):
                    candidate = float(s[0]/s[-1])
                if np.isfinite(candidate):
                    full_condition = candidate
            diagnostics = SignedDctSolveDiagnostics(
                solver_status='rank_deficient' if rank < len(tau) else 'converged',
                lapack_driver=driver, augmented_shape=b.shape, effective_rank=int(rank), svd_rank=svd_rank,
                relative_cutoff=float(cutoff), singular_values=tuple(s), condition_number=full_condition,
                effective_condition_number=float(s[0]/s[svd_rank-1]), grid_step_decades=h,
                regularizer_rows=len(l), weighted_sse=weighted_sse, regularization_penalty=penalty,
                objective=weighted_sse+penalty,
            )
            require(all(np.all(np.isfinite(v)) for v in (chi, mass, reconstruction, residuals)), 'nonfinite signed solution')
            norm = source.normalization
            # Refuse an apparently usable result whose physical restoration overflows.
            require(all(np.all(np.isfinite(v)) for v in (
                norm.reference_value+norm.scale_value*reconstruction, norm.scale_value*residuals,
                norm.scale_value*chi, norm.scale_value*mass)), 'nonfinite physical-unit restoration')
    except (ContractError, FloatingPointError, OverflowError, linalg.LinAlgError, ValueError) as exc:
        return unavailable('error', f'Signed augmented solve failed: {exc}')

    warnings = list(source.state.warnings)+list(source.normalization.state.warnings)
    if rank < len(tau) or svd_rank < len(tau):
        warnings.append('SIGNED_DCT_RANK_DEFICIENT: augmented system is numerically rank deficient; truncated numerical solution is not unique spectral evidence')
    if int(rank) != svd_rank:
        warnings.append('SIGNED_DCT_RANK_DISAGREEMENT: pivoted QR and SVD rank estimates differ at the recorded cutoff')
    if full_condition is None or full_condition > 1/np.sqrt(np.finfo(float).eps):
        warnings.append('SIGNED_DCT_ILL_CONDITIONED: augmented condition exceeds 1/sqrt(float64 eps); coefficients may be sensitive')
    warnings = tuple(dict.fromkeys(warnings))
    return SignedDctResult(schema_version=2, source=source, config=config,
        state=ResultState(status='warning' if warnings else 'completed', warnings=warnings), run_id=run_id,
        tau_s=tuple(tau), log10_tau=tuple(x), chi=tuple(chi), quadrature_weights=tuple(q),
        quadrature_mass=tuple(mass), reconstruction=tuple(reconstruction), residuals=tuple(residuals),
        selected_lambda=lam, solve_diagnostics=diagnostics)


def _augmented_system(source, config):
    """One construction shared by the numeric solve and actual influence audit."""
    tau, x, q, a, l, h = _operators(source.time_s, config)
    root_w = np.sqrt(np.asarray(config.observation_weights or (1.,)*len(source.signal)))
    b = np.vstack((root_w[:, None]*a, np.sqrt(config.grid.regularization_lambda)*l))
    rhs = np.concatenate((root_w*np.asarray(source.signal), np.zeros(len(l))))
    return tau, x, q, a, l, h, root_w, b, rhs, max(b.shape)*np.finfo(float).eps
