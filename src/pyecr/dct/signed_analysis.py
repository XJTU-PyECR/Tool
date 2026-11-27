"""Manual/GCV selection and deterministic sensitivity of signed spectra.

All numeric spectra use the P04 solver and its single augmented construction.
No RNG, bootstrap, CI, active-data changes, model suggestions or species labels.
"""
from dataclasses import replace
from math import isclose

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from pyecr.contracts import (ContractError, ResultState, SignedDctResult, SignedDctAnalysis,
    SignedDctDiagnosticConfig, SignedLambdaCandidate, SignedDctComparison, DctSourceSnapshot,
    SignedDctConfig, SignedNormalizationSpec, SourceIdentity)
from pyecr.contracts.common import require, number
from pyecr.contracts.signed_analysis import default_diagnostic_config
from pyecr.dct.signed import fit_signed_dct, _augmented_system
from pyecr.dct.signed_peaks import extract_signed_peaks
from pyecr.preprocessing.signed_normalization import _local_noise
from pyecr.dct.execution import isolated_numeric, numeric_phase

__all__ = ['analyze_signed_dct', 'prepare_terminal_sensitivity', 'effective_degrees_of_freedom']

INTERPRETATION = (
    'Candidate peaks describe time scales conditional on the exponential kernel, terminal value, grid and regularization. Signs do not identify H/O, and peak count does not identify the number of chemical processes.',
    'GCV predicts error conditional on fixed W, sigma_infinity and S. It is not a confidence interval and excludes uncertainty from lambda selection and normalization.',
    'Stability covers the listed deterministic checks only. Bootstrap and calibrated confidence intervals are unavailable. A good reconstruction does not establish a unique spectrum.',
)
FLAG_EXPLANATIONS = {
    'GCV_LAMBDA_AT_BOUNDARY': 'The selected GCV candidate is at the search boundary. Values beyond this range were not tested; review the range before interpreting the spectrum.',
    'OSCILLATORY_SPECTRUM': 'Alternating signs or oscillatory candidates may distort basin areas. Individual candidates cannot each be identified as physical processes.',
    'MANY_CANDIDATES': 'Many candidate maxima are reported. Their count alone does not establish oscillation or multiple chemical processes; review resolution and regularization sensitivity.',
    'RESIDUAL_STRUCTURE': 'DCT residuals show an observation-order pattern. Review model mismatch and noise assumptions before interpreting peaks or GCV selection; this is an exploratory screen, not calibrated confidence evidence.',
    'BOUNDARY_PEAK': 'A candidate is near the tau boundary. Its position and area may depend on the chosen range.',
    'WIDTH_TRUNCATED': 'Half-height crossings are incomplete within the basin; the reported width is clipped at the basin boundary.',
    'BASIN_TRUNCATED': 'A material tail reaches a tau boundary. The basin area is incomplete even if both half-height crossings are present; extend the range. The tail test is a relative-height heuristic.',
    'FLAT_TOP_PEAK': 'A flat maximum spans multiple grid nodes. The reported tau is a grid representative, not a uniquely located maximum.',
    'NO_RESOLVABLE_PEAK': 'No candidate meets the current reporting thresholds. This does not establish the absence of hidden processes.',
    'UNRESOLVED_NEIGHBOURING_CANDIDATES': 'Neighbouring candidates are unresolved by separation or overlapping half-height widths.',
    'UNSTABLE_CANDIDATE_PEAKS': 'At least one candidate lacks a same-sign position match in every listed check, or is affected by boundary or truncation limits.',
    'SENSITIVITY_INCOMPLETE': 'Some checks were not run or failed. Review their reasons before assessing stability.',
    'LOW_SNR_HEURISTIC': 'Observed contrast is at most five times the local noise estimate. This is a low-information heuristic, not calibrated noise or confidence evidence.',
    'GCV_CANDIDATE_FAILURES': 'Some lambda solves or effective-degrees-of-freedom scores failed. Their numerical results and reasons are retained.',
    'MANUAL_GCV_SCORE_UNAVAILABLE': 'The explicit manual lambda is retained. Its GCV score is unavailable.',
}


def effective_degrees_of_freedom(result):
    """tr(Hw), Hw=Aw * B^+ * [I;0], using the actual cutoff/driver.

    W, trapezoidal density measure, curvature scaling and both unpenalized
    linear-density modes enter B. There is no free residual intercept b.
    Solve identity right-hand sides, never a Gram inverse or a guessed rank df.
    """
    require(isinstance(result, SignedDctResult) and result.state.ready and result.solve_diagnostics is not None,
            'df requires an actual ready numeric result with solve diagnostics')
    if result.solve_diagnostics.schema_version == 3:
        require(result.solve_diagnostics.df_semantics=='fixed_linear','Frozen UP weights do not provide the full nonlinear df/GCV')
        return result.solve_diagnostics.effective_df
    if result.solve_diagnostics.schema_version == 2:
        return result.solve_diagnostics.effective_df
    actual_config = _manual_config(result.config, result.selected_lambda)
    _, _, _, a, _, _, root_w, b, _, cutoff = _augmented_system(result.source, actual_config)
    n = len(result.source.signal)
    rhs = np.zeros((len(b), n)); rhs[:n] = np.eye(n)
    driver = result.solve_diagnostics.lapack_driver
    with numeric_phase('signed_gcv_df'):
        c, _, rank, _ = linalg.lstsq(b, rhs, cond=cutoff, lapack_driver=driver)
    require(rank == result.solve_diagnostics.effective_rank, 'influence/numeric rank differs at recorded cutoff')
    raw = float(np.sum((root_w[:, None]*a)*c.T))
    require(np.isfinite(raw), 'nonfinite actual effective degrees of freedom')
    return raw


def _manual_config(config, lam, **grid):
    return replace(config, grid=replace(config.grid, lambda_mode='manual', regularization_lambda=float(lam), **grid), gcv_lambdas=())


def _candidate(source, config, policy, cancel_check=lambda: None, *, linear_cache=None):
    cancel_check()
    result = fit_signed_dct(source, config, linear_cache=linear_cache) if linear_cache is not None else fit_signed_dct(source, config)
    cancel_check()
    peaks = extract_signed_peaks(result.tau_s, result.chi, policy)[0] if result.state.ready else ()
    raw = None
    failure, df, score = None, None, None
    if not result.state.ready:
        failure = '; '.join(result.state.blocking_reasons)
    else:
        try:
            cancel_check()
            raw = effective_degrees_of_freedom(result)
            cancel_check()
            n = len(source.signal)
            require(0 <= raw <= n, 'actual df outside [0,N]')
            require(n-raw > 64*np.finfo(float).eps*n, 'GCV denominator saturated at actual df=N')
            df = raw
            score = float(n*result.solve_diagnostics.weighted_sse/(n-df)**2)
            require(np.isfinite(score), 'nonfinite GCV score')
        except (ContractError, linalg.LinAlgError, ValueError, FloatingPointError, OverflowError) as exc:
            failure, df, score = f'GCV unavailable: {exc}', None, None
    return SignedLambdaCandidate(result=result, effective_df=df, raw_effective_df=raw,
        gcv_score=score, failure=failure, peak_diagnostics=peaks)


def _match(base, variants, tolerance, *, legacy=False):
    matches, shifts = [None]*len(base), [None]*len(base)
    if base and variants:
        # Lexicographic objective: first maximize valid one-to-one matches,
        # then minimize their total log-position shift. One unmatched peak must
        # cost more than *all* valid shifts, or Hungarian assignment can discard
        # a valid pair merely to get a different exact-position match.
        unmatched = tolerance+1e-12 if legacy else (len(base)+1)*tolerance
        cost = np.full((len(base), len(variants)+len(base)), unmatched)
        for i, p in enumerate(base):
            for j, q in enumerate(variants):
                distance = abs(np.log10(p.candidate.position_s/q.candidate.position_s))
                cost[i, j] = distance if p.direction == q.direction and (legacy or distance <= tolerance) else (len(base)+1)*unmatched+1
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if j < len(variants) and cost[i, j] <= tolerance:
                matches[i], shifts[i] = int(j), float(cost[i, j])
    return tuple(matches), tuple(shifts)


def _check_terminal_source(base, variant, offset):
    require(base.normalization is not None and isinstance(variant, DctSourceSnapshot)
            and variant.normalization is not None, 'terminal comparison requires explicit normalized source copies')
    base.normalization.spec.identity.assert_matches(variant.normalization.spec.identity)
    require(base.raw_indices == variant.raw_indices and base.raw_count == variant.raw_count
            and base.time_s == variant.time_s and base.t0_s == variant.t0_s, 'terminal comparison raw/time mapping conflict')
    b, v = base.normalization, variant.normalization
    require(v.spec.signal_unit == b.spec.signal_unit and v.spec.signal_type == b.spec.signal_type, 'terminal comparison units conflict')
    require(isclose((v.reference_value-b.reference_value)/b.scale_value, offset, rel_tol=1e-10, abs_tol=1e-12), 'terminal offset differs from explicit request')
    original = b.reference_value+b.scale_value*np.asarray(base.signal)
    other = v.reference_value+v.scale_value*np.asarray(variant.signal)
    require(np.allclose(original, other, rtol=1e-12, atol=1e-12*max(1., b.scale_value)), 'terminal comparison changed physical observations')


@isolated_numeric('signed')
def analyze_signed_dct(source, config, diagnostic_config=None, *, terminal_sources=(),
                       cancel_check=None, progress_callback=None):
    """Run a finite, explicit deterministic analysis on copies; no active Apply.

    terminal_sources are explicit P02 preparations in the requested offset order.
    Missing preparations are recorded NOT_RUN and stability remains incomplete.
    Controls hold selected lambda fixed to isolate tau/grid/terminal effects.
    """
    if diagnostic_config is None and isinstance(config, SignedDctConfig):
        diagnostic_config = default_diagnostic_config(config.grid)
    require(isinstance(source, DctSourceSnapshot) and isinstance(config, SignedDctConfig)
            and isinstance(diagnostic_config, SignedDctDiagnosticConfig), 'analysis requires explicit contracts')
    check = cancel_check or (lambda: None)
    progress = progress_callback or (lambda fraction, message: None)
    check()
    p = diagnostic_config
    linear_cache = None
    if config.schema_version >= 2:
        from pyecr.dct.signed_linear import LinearCache
        linear_cache = LinearCache()
    lambdas = config.gcv_lambdas if config.grid.lambda_mode == 'gcv' else (config.grid.regularization_lambda,)
    requested = len(lambdas)+len(p.lambda_factors)+len(p.tau_ranges_s)+len(p.grid_sizes)+len(p.terminal_offsets_in_S)
    require(requested <= p.max_evaluations, 'requested candidate evaluations exceed explicit budget')
    require(len(terminal_sources) in {0, len(p.terminal_offsets_in_S)}, 'terminal source count must match requested offsets')
    for offset, variant in zip(p.terminal_offsets_in_S, terminal_sources):
        _check_terminal_source(source, variant, offset)
    progress(0., 'Validating frozen signed-DCT source')
    check()
    candidates = []
    completed = 0
    for lam in lambdas:
        candidates.append(_candidate(source, _manual_config(config, lam), p, check, linear_cache=linear_cache))
        completed += 1
        progress(completed/requested, f'Lambda candidate {completed}/{len(lambdas)}: {lam:g}')
        check()
    candidates = tuple(candidates)
    used = len(candidates)
    usable = [c for c in candidates if c.failure is None] if config.grid.lambda_mode == 'gcv' else [c for c in candidates if c.result.state.ready]
    if not usable:
        progress(1., 'No usable lambda candidate; failures retained')
        check()
        return SignedDctAnalysis(source=source, config=config, diagnostic_config=p, state=ResultState(status='error',
            blocking_reasons=('No usable lambda candidate; inspect every saved failure',)), candidates=candidates,
            evaluations_used=used, flags=('NO_RESOLVABLE_PEAK',), interpretation=INTERPRETATION)
    chosen = min(usable, key=lambda c: c.gcv_score) if config.grid.lambda_mode == 'gcv' else usable[0]
    base = chosen.result
    peaks, peak_flags, resolution = extract_signed_peaks(base.tau_s, base.chi, p)
    flags = set(peak_flags)
    if config.grid.lambda_mode == 'gcv':
        index = candidates.index(chosen)
        if index in {0, len(candidates)-1}:
            flags.add('GCV_LAMBDA_AT_BOUNDARY')
        if any(c.failure is not None for c in candidates):
            flags.add('GCV_CANDIDATE_FAILURES')
    elif chosen.failure is not None:
        flags.add('MANUAL_GCV_SCORE_UNAVAILABLE')
    comparisons = []

    def compare(label, kind, cfg, variant=source, failure=None):
        nonlocal used, completed
        check()
        result, cp = None, ()
        if failure is None:
            if variant == source and cfg == base.config:
                result = base
            else:
                result = (fit_signed_dct(variant, cfg, linear_cache=linear_cache) if linear_cache is not None
                          else fit_signed_dct(variant, cfg)); used += 1
                check()
            if result.state.ready:
                cp = extract_signed_peaks(result.tau_s, result.chi, p)[0]
            else:
                failure = '; '.join(result.state.blocking_reasons)
        matches, shifts = _match(peaks, cp, p.match_tolerance_decades,
                                legacy=p.algorithm_version == 'signed-diagnostics/1')
        comparisons.append(SignedDctComparison(label=label, kind=kind,
            normalization_policy='reestimate_S_at_declared_terminal' if kind == 'terminal' else 'fixed_sigma_infinity_and_S',
            result=result, peak_diagnostics=cp, matches=matches, position_shifts_decades=shifts, failure=failure))
        completed += 1
        progress(completed/requested, label + ('; ' + failure if failure else ''))
        check()

    for factor in p.lambda_factors:
        compare(f'lambda x{factor:g}', 'lambda', _manual_config(config, base.selected_lambda*factor),
            failure='NOT_RUN: multiplicative lambda perturbation is undefined at lambda=0' if base.selected_lambda == 0 else None)
    for lo, hi in p.tau_ranges_s:
        compare(f'tau [{lo:g}, {hi:g}] s', 'tau_range', _manual_config(config, base.selected_lambda, tau_min_s=lo, tau_max_s=hi))
    for n in p.grid_sizes:
        compare(f'grid {n}', 'grid', _manual_config(config, base.selected_lambda, n_tau=n))
    for i, offset in enumerate(p.terminal_offsets_in_S):
        compare(f'terminal {offset:+g} S; reestimate max S', 'terminal', _manual_config(config, base.selected_lambda),
            variant=terminal_sources[i] if terminal_sources else source,
            failure=None if terminal_sources else 'NOT_RUN: provide explicit P02 terminal-source preparations from the real input dataset/recipe')
    updated = []
    for i, peak in enumerate(peaks):
        matched = sum(c.matches[i] is not None for c in comparisons)
        shifts = [c.position_shifts_decades[i] for c in comparisons if c.matches[i] is not None]
        if not comparisons:
            status = 'not_checked'
        elif any(c.failure is not None for c in comparisons):
            status = 'incomplete'
        elif matched != len(comparisons) or peak.boundary or peak.width_truncated:
            status = 'unstable'
        else:
            status = 'stable_conditional'
        updated.append(replace(peak, stability_status=status, matched_comparisons=matched,
            required_comparisons=len(comparisons), max_position_shift_decades=max(shifts) if shifts else None))
    if any(p.stability_status == 'unstable' for p in updated):
        flags.add('UNSTABLE_CANDIDATE_PEAKS')
    if any(c.failure is not None for c in comparisons):
        flags.add('SENSITIVITY_INCOMPLETE')
    time = np.asarray(source.time_s); time = (time-time[0])/(time[-1]-time[0])
    noise = float(_local_noise(time, np.asarray(source.signal)))
    contrast = float(np.percentile(source.signal, 90)-np.percentile(source.signal, 10))
    ratio = contrast/noise if noise > 0 else None
    if ratio is not None and not np.isfinite(ratio):
        ratio = None
    if noise > 0 and contrast <= 5*noise:
        flags.add('LOW_SNR_HEURISTIC')
        resolution = 'low_information_no_resolution_claim'
    residual_note = ()
    if p.algorithm_version != 'signed-diagnostics/1':
        # Reuse the sample-size/spacing-aware Diagnostics screen. This tests
        # the actual weighted residual, without treating irregular observation
        # indices as equally spaced time or changing/whitening any input.
        from pyecr.diagnostics.evaluation import residual_evidence
        weighted = np.asarray(base.residuals)*np.sqrt(config.observation_weights or np.ones(len(base.residuals)))
        evidence = residual_evidence(weighted, base.source.time_s)
        if evidence.dependence == 'warning':
            flags.add('RESIDUAL_STRUCTURE')
            residual_note = (evidence.explanation,)
    warnings = tuple(dict.fromkeys((*base.state.warnings, *sorted(flags))))
    selected = replace(base, config=config, peaks=tuple(q.candidate for q in updated),
        gcv_scores=tuple(c.gcv_score for c in candidates) if config.grid.lambda_mode == 'gcv' else (),
        gcv_effective_df=tuple(c.effective_df for c in candidates) if config.grid.lambda_mode == 'gcv' else (),
        gcv_failures=tuple(c.failure for c in candidates) if config.grid.lambda_mode == 'gcv' else (),
        state=ResultState(status='warning' if warnings else 'completed', warnings=warnings))
    check()
    analysis = SignedDctAnalysis(source=source, config=config, diagnostic_config=p,
        state=selected.state, candidates=candidates, selected_result=selected, peak_diagnostics=tuple(updated),
        comparisons=tuple(comparisons), flags=tuple(sorted(flags)), noise_sd_estimate=noise,
        contrast_to_noise=ratio, resolution_status=resolution, evaluations_used=used,
        interpretation=INTERPRETATION+tuple(FLAG_EXPLANATIONS[f] for f in sorted(flags))+residual_note)
    progress(1., 'Signed-DCT diagnostics complete; publication still requires source check')
    check()
    return analysis


def prepare_terminal_sensitivity(dataset, recipe, normalization, *, offsets_in_S=(-.01, .01), config=None):
    """Explicit read-only hypotheses using the actual pre-normalization input.

    Each hypothesis declares a shifted external value and recomputes the true
    max on all valid rows through P02. These copies are NOT accepted terminals,
    raw edits, bootstrap replicates or a model-inferred reference.
    """
    from pyecr.preprocessing import normalize_signed_dct
    SourceIdentity.capture(dataset, recipe, config=config).assert_matches(normalization.spec.identity)
    require(normalization.state.ready, 'terminal sensitivity needs a usable base normalization')
    result = []
    for offset in offsets_in_S:
        offset = number(offset, 'terminal sensitivity offset')
        require(offset != 0, 'terminal sensitivity offset must be nonzero')
        spec = replace(normalization.spec, reference_kind='external_known_terminal', reference_rows=(),
            reference_value=normalization.reference_value+float(offset)*normalization.scale_value,
            reference_evidence=f'Explicit deterministic terminal hypothesis {offset:+g} base S; not an accepted observed terminal or CI')
        norm = normalize_signed_dct(dataset, spec, recipe=recipe)
        require(norm.state.ready, '; '.join(norm.state.blocking_reasons))
        result.append(DctSourceSnapshot.capture_experiment(dataset, recipe, config=config, normalization=norm, t0_s=spec.t0_s))
    return tuple(result)
