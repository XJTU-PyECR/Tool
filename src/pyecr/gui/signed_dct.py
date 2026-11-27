"""Read-only signed-DCT presentation; no normalization, inversion or model calls."""
from dataclasses import dataclass
import json
from math import log10, sqrt

from pyecr.services.signed_dct_service import SignedDctJob
from pyecr.preprocessing import inverse_signed_dct

SIGN_LEGEND = ('Terminal-deviation spectrum (individual exponential): chi > 0 decays down from above the terminal; '
               'chi < 0 recovers up from below it. Opposite contributions can cancel; no automatic H/O assignment.')
DISPLAY_NOTICE = ('All valid observations enter reconstruction/residual/RMSE. Lines connect stored samples only; '
                  'display grid / zoom never enters error calculation. τ nodes are the inversion grid, not extra observations.')


@dataclass(frozen=True, slots=True)
class SignedDctDisplay:
    x: tuple = ()
    chi: tuple = ()
    time: tuple = ()
    observed: tuple = ()
    reconstruction: tuple = ()
    residual: tuple = ()
    lambda_x: tuple = ()
    lambda_score: tuple = ()
    peak_x: tuple = ()
    peak_shift: tuple = ()
    unit: str = '1'
    summary: str = 'No stored signed-DCT run.'
    details: str = 'No candidate evidence.'

    # Shared UIR6 viewport fitting reads these aliases; no new sampling grid.
    log10_tau = property(lambda self: self.x)
    time_s = property(lambda self: self.time)
    signal = property(lambda self: self.observed)
    residuals = property(lambda self: self.residual)


def display_signed_job(job: SignedDctJob | None, *, physical=False) -> SignedDctDisplay:
    if job is None:
        return SignedDctDisplay()
    if job.request.convex_options is not None:
        from pyecr.gui.convex_dct import display_convex_job
        return display_convex_job(job,physical=physical)
    lines = [f'job_id={job.job_id} | status={job.status}',
        f'source={job.request.dataset.metadata.get("source_path", job.request.dataset.metadata.get("dataset_name", "in-memory"))}',
        f'identity={job.request.identity.to_dict()}',
        f'captured physical context (optional): {job.request.physical_context.to_dict()}',
        *job.stale_reasons, *(('Failure: '+job.failure,) if job.failure else ()),
        'Warnings / limitations:', *job.warnings]
    if job.prepared:
        n = job.prepared.normalization
        lines += [f'profile={n.spec.profile_id} | reference={n.spec.reference_kind}',
            f'sigma_infinity={n.reference_value!r} {n.spec.signal_unit} | S={n.scale_value!r} {n.spec.signal_unit}',
            f'max raw index (zero-based)={n.scale_raw_index}; terminal rows={n.spec.reference_rows}',
            f'raw_count={n.spec.raw_count}; valid rows={n.spec.raw_indices}; t0_s={n.spec.t0_s}',
            f'raw_sha256={job.prepared.raw_sha256}']
    a = job.analysis
    if a is None or a.selected_result is None:
        if a:
            lines += [*a.state.blocking_reasons, *a.interpretation]
        return SignedDctDisplay(summary=f'{job.status}: no usable stored signed spectrum', details='\n'.join(lines))
    r = a.selected_result
    unit = r.physical_unit if physical else '1'
    chi = r.physical_chi if physical else r.chi
    reconstruction = r.physical_reconstruction if physical else r.reconstruction
    residual = r.physical_residuals if physical else r.residuals
    observed = (job.prepared.processed.signal if physical else r.source.signal)
    # For an explicitly applied profile, physical means its recorded input unit,
    # not the current normalized dataset's signal. Use the saved P02 inverse data.
    if physical:
        n = r.source.normalization
        observed = inverse_signed_dct(n)
    rmse = sqrt(sum(v*v for v in r.residuals)/len(r.residuals))
    lines += [f'λ={r.selected_lambda:.17g}; mode={r.config.grid.lambda_mode}; grid={r.config.grid.to_dict()}',
        f'full observed RMSE (normalized)={rmse:.17g}; b=0; mass=chi*quadrature_weight',
        f'resolution={a.resolution_status}; flags={a.flags}', SIGN_LEGEND, DISPLAY_NOTICE,
        'Signed peak candidates: canonical signed/absolute areas below have normalized unit 1. '
        'Position stability is not area accuracy or a calibrated CI.']
    for i, p in enumerate(a.peak_diagnostics, 1):
        lines.append(f'{i}: τ={p.candidate.position_s:.17g}s | {p.direction} | signed_area={p.signed_area:.17g} | '
            f'absolute_area={p.absolute_area:.17g} | width={p.candidate.width_decades:.17g} decade | '
            f'{p.stability_status} {p.matched_comparisons}/{p.required_comparisons}, '
            f'max shift={p.max_position_shift_decades!r} decade | {p.warnings}')
        if physical:
            scale=r.source.normalization.scale_value
            lines.append(f'  restored signed_area={scale*p.signed_area:.17g}, absolute_area={scale*p.absolute_area:.17g} {unit}')
    if not a.peak_diagnostics:
        lines.append('No reliable / resolvable candidate peak; stored spectrum and residual remain visible. No old peaks substituted.')
    lines += ['Complete λ evidence:']
    for c in a.candidates:
        lines.append(f'λ={c.result.config.grid.regularization_lambda:.17g} | df={c.effective_df!r} | '
                     f'GCV={c.gcv_score!r} | {c.failure or c.result.state.status}')
    lines += [f'{c.label}: {c.failure or "computed"}; matches={c.matches}; shifts={c.position_shifts_decades}' for c in a.comparisons]
    lines += list(a.interpretation)
    # Failed scores are not plotted at zero; complete failures remain in details.
    scores = [(log10(c.result.config.grid.regularization_lambda), c.gcv_score) for c in a.candidates
              if c.gcv_score is not None and c.result.config.grid.regularization_lambda > 0]
    shifts = [(log10(p.candidate.position_s), p.max_position_shift_decades) for p in a.peak_diagnostics
              if p.max_position_shift_decades is not None]
    return SignedDctDisplay(x=r.log10_tau, chi=chi, time=r.source.time_s, observed=observed,
        reconstruction=reconstruction, residual=residual, lambda_x=tuple(v[0] for v in scores),
        lambda_score=tuple(v[1] for v in scores), peak_x=tuple(v[0] for v in shifts),
        peak_shift=tuple(v[1] for v in shifts), unit=unit,
        summary=f'{job.status} | {len(r.source.signal)} full observations | λ={r.selected_lambda:.6g} '
                f'({r.config.grid.lambda_mode}) | RMSE(Y)={rmse:.6g} | {len(a.peak_diagnostics)} candidate peaks\n'
                f'Stored signed-terminal profile: σ∞={r.source.normalization.reference_value:.9g}, '
                f'S={r.source.normalization.scale_value:.9g} {r.source.normalization.spec.signal_unit}',
        details='\n'.join(lines))


def complete_signed_evidence(job, draft):
    view = display_signed_job(job)
    return ('Signed DCT run record\n'+view.details+'\n\nCurrent draft (not applied):\n'+
        json.dumps(draft, ensure_ascii=False, indent=2, allow_nan=False)+'\n\nStored job JSON (all arrays):\n'+
        ('null' if job is None else json.dumps(job.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)))
