"""Signed selection/peak/sensitivity evidence, separate from the spectrum itself."""
from dataclasses import dataclass, replace
from math import isclose, log10
from typing import ClassVar

from pyecr.dct.core import CandidateTimescale
from .common import Contract, ResultState, number, integer, require, vector, strings, text
from .dct import DctSourceSnapshot, SignedDctConfig, SignedDctResult


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedDctDiagnosticConfig(Contract):
    KIND: ClassVar[str] = 'signed_dct_diagnostic_config'
    prominence_fraction: float = .03
    minimum_absolute_area: float = .05
    resolution_decades: float = .5
    match_tolerance_decades: float = .10
    lambda_factors: tuple[float, ...] = (.3, 3.)
    tau_ranges_s: tuple[tuple[float, float], ...] = ((.05, 4000.), (.1, 2000.))
    grid_sizes: tuple[int, ...] = (121, 241, 481)
    terminal_offsets_in_S: tuple[float, ...] = (-.01, .01)
    max_evaluations: int = 128
    algorithm_version: str = 'signed-diagnostics/2'
    width_definition: str = 'FWHM_zero_baseline_log10_tau_v1'
    area_definition: str = 'piecewise_linear_sign_basin_dlog10_tau_v1'
    df_definition: str = 'trace_actual_augmented_whitened_operator_v1'
    uncertainty_semantics: str = 'deterministic_sensitivity_not_CI'

    def __post_init__(self):
        Contract.__post_init__(self)
        for k in ('prominence_fraction', 'minimum_absolute_area', 'resolution_decades', 'match_tolerance_decades'):
            object.__setattr__(self, k, number(getattr(self, k), k, positive=True))
        require(self.prominence_fraction < 1, 'prominence fraction must be <1')
        for k in ('lambda_factors', 'terminal_offsets_in_S'):
            object.__setattr__(self, k, vector(getattr(self, k), k, positive=k == 'lambda_factors'))
            require(len(set(getattr(self, k))) == len(getattr(self, k)), f'duplicate {k}')
        require(all(v != 0 for v in self.terminal_offsets_in_S), 'terminal offsets must be nonzero')
        require(isinstance(self.tau_ranges_s, (tuple, list)), 'tau ranges must be an array')
        ranges = tuple(vector(v, 'tau range', positive=True) for v in self.tau_ranges_s)
        require(all(len(v) == 2 and v[0] < v[1] for v in ranges), 'invalid tau ranges')
        object.__setattr__(self, 'tau_ranges_s', ranges)
        require(isinstance(self.grid_sizes, (tuple, list)), 'grid sizes must be an array')
        object.__setattr__(self, 'grid_sizes', tuple(integer(v, 'grid size', minimum=10) for v in self.grid_sizes))
        object.__setattr__(self, 'max_evaluations', integer(self.max_evaluations, 'max_evaluations', minimum=1))
        require(self.algorithm_version in {'signed-diagnostics/1', 'signed-diagnostics/2'} and
                (self.width_definition, self.area_definition, self.df_definition, self.uncertainty_semantics) ==
                ('FWHM_zero_baseline_log10_tau_v1', 'piecewise_linear_sign_basin_dlog10_tau_v1',
                 'trace_actual_augmented_whitened_operator_v1', 'deterministic_sensitivity_not_CI'), 'unknown signed diagnostics profile')


def default_diagnostic_config(grid):
    """Freeze meaningful controls around the requested grid, rather than seconds
    or node counts unrelated to the experiment. Explicit policies are untouched.
    """
    grid.validate()
    lo, hi, n = grid.tau_min_s, grid.tau_max_s, grid.n_tau
    ranges = [(lo/2, hi*2)]
    if lo*2 < hi/2:
        ranges.append((lo*2, hi/2))
    sizes = tuple(sorted({max(10, round((n-1)*factor)+1) for factor in (.75, 1.5)}-{n}))
    return SignedDctDiagnosticConfig(tau_ranges_s=tuple(ranges), grid_sizes=sizes)


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedPeakDiagnostic(Contract):
    KIND: ClassVar[str] = 'signed_peak_diagnostic'
    candidate: CandidateTimescale
    direction: str
    absolute_area: float
    absolute_height: float
    basin_log10_tau: tuple[float, float]
    half_height_log10_tau: tuple[float, float]
    boundary: bool
    width_truncated: bool
    warnings: tuple[str, ...] = ()
    stability_status: str = 'not_checked'
    matched_comparisons: int = 0
    required_comparisons: int = 0
    max_position_shift_decades: float | None = None

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.candidate, CandidateTimescale), 'reuse CandidateTimescale')
        p = self.candidate
        number(p.position_s, 'peak position', positive=True); integer(p.grid_index, 'peak grid index')
        require(p.label == 'candidate_timescale', 'no species labels in signed peaks')
        number(p.area, 'signed_area'); require(p.area != 0, 'zero-area peak is unavailable')
        require(self.direction in {'positive', 'negative'} and (p.area > 0) == (self.direction == 'positive'), 'peak sign/direction conflict')
        for k in ('absolute_area', 'absolute_height'):
            object.__setattr__(self, k, number(getattr(self, k), k, positive=True))
        require(isclose(self.absolute_area, abs(p.area), rel_tol=1e-12, abs_tol=1e-14), 'signed/absolute area conflict')
        for k in ('basin_log10_tau', 'half_height_log10_tau'):
            v = vector(getattr(self, k), k)
            require(len(v) == 2 and v[0] < v[1], f'invalid {k}')
            object.__setattr__(self, k, v)
        require(self.basin_log10_tau[0] <= self.half_height_log10_tau[0]
                < self.half_height_log10_tau[1] <= self.basin_log10_tau[1], 'width/basin conflict')
        require(self.basin_log10_tau[0] <= log10(p.position_s) <= self.basin_log10_tau[1], 'peak position outside basin')
        require(isclose(p.width_decades, self.half_height_log10_tau[1]-self.half_height_log10_tau[0], abs_tol=1e-12), 'FWHM definition conflict')
        require(isinstance(self.boundary, bool) and isinstance(self.width_truncated, bool), 'peak flags must be bool')
        object.__setattr__(self, 'warnings', strings(self.warnings, 'peak warnings'))
        require(self.stability_status in {'not_checked', 'incomplete', 'unstable', 'stable_conditional'}, 'unknown peak stability')
        for k in ('matched_comparisons', 'required_comparisons'):
            object.__setattr__(self, k, integer(getattr(self, k), k))
        require(self.matched_comparisons <= self.required_comparisons, 'invalid stability match count')
        require((self.max_position_shift_decades is None) == (self.matched_comparisons == 0), 'missing/extra measured stability shift')
        if self.max_position_shift_decades is not None:
            object.__setattr__(self, 'max_position_shift_decades', number(self.max_position_shift_decades, 'peak shift', nonnegative=True))
        require(self.stability_status != 'stable_conditional' or
                (self.required_comparisons > 0 and self.matched_comparisons == self.required_comparisons
                 and not self.boundary and not self.width_truncated), 'incomplete/boundary stability cannot be called stable')

    @property
    def signed_area(self):
        return self.candidate.area


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedLambdaCandidate(Contract):
    KIND: ClassVar[str] = 'signed_lambda_candidate'
    result: SignedDctResult
    effective_df: float | None
    raw_effective_df: float | None
    gcv_score: float | None
    failure: str | None
    peak_diagnostics: tuple[SignedPeakDiagnostic, ...] = ()

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.result, SignedDctResult) and self.result.config.grid.lambda_mode == 'manual', 'candidate must retain actual manual numeric solve')
        require(isinstance(self.peak_diagnostics, (tuple, list)) and all(isinstance(p, SignedPeakDiagnostic) for p in self.peak_diagnostics), 'invalid candidate peaks')
        object.__setattr__(self, 'peak_diagnostics', tuple(self.peak_diagnostics))
        for p in self.peak_diagnostics:
            require(self.result.state.ready and p.candidate.grid_index < len(self.result.tau_s)
                    and p.candidate.position_s == self.result.tau_s[p.candidate.grid_index], 'candidate peak/grid conflict')
        n = len(self.result.source.signal)
        if self.raw_effective_df is not None:
            object.__setattr__(self, 'raw_effective_df', number(self.raw_effective_df, 'raw_effective_df'))
        if self.failure is None:
            require(self.result.state.ready, 'successful candidate requires numeric payload')
            df = number(self.effective_df, 'effective_df', nonnegative=True)
            require(df < n and self.raw_effective_df is not None, 'invalid/saturated effective_df')
            score = number(self.gcv_score, 'gcv_score', nonnegative=True)
            require(isclose(score, n*self.result.solve_diagnostics.weighted_sse/(n-df)**2, rel_tol=1e-10, abs_tol=1e-14), 'GCV score/actual residual/df conflict')
        else:
            text(self.failure, 'candidate failure')
            require(self.effective_df is None and self.gcv_score is None, 'failed candidate must not fabricate score/df')


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedDctComparison(Contract):
    KIND: ClassVar[str] = 'signed_dct_comparison'
    label: str
    kind: str
    normalization_policy: str
    result: SignedDctResult | None
    peak_diagnostics: tuple[SignedPeakDiagnostic, ...]
    matches: tuple[int | None, ...]
    position_shifts_decades: tuple[float | None, ...]
    failure: str | None = None

    def __post_init__(self):
        Contract.__post_init__(self)
        text(self.label, 'comparison label')
        require(self.kind in {'lambda', 'tau_range', 'grid', 'terminal'}, 'unknown comparison kind')
        require(self.normalization_policy in {'fixed_sigma_infinity_and_S', 'reestimate_S_at_declared_terminal'}, 'unknown normalization policy')
        require((self.kind == 'terminal') == (self.normalization_policy == 'reestimate_S_at_declared_terminal'), 'comparison normalization policy conflict')
        for k in ('peak_diagnostics', 'matches', 'position_shifts_decades'):
            require(isinstance(getattr(self, k), (tuple, list)), f'{k} must be an array')
            object.__setattr__(self, k, tuple(getattr(self, k)))
        require(all(isinstance(p, SignedPeakDiagnostic) for p in self.peak_diagnostics), 'invalid comparison peaks')
        require(len(self.matches) == len(self.position_shifts_decades), 'comparison match/shift mismatch')
        for match, shift in zip(self.matches, self.position_shifts_decades):
            if match is None:
                require(shift is None, 'unmatched peak has no position shift')
            else:
                require(integer(match, 'peak match') < len(self.peak_diagnostics), 'peak match out of range')
                number(shift, 'position shift', nonnegative=True)
        if self.failure is None:
            require(isinstance(self.result, SignedDctResult) and self.result.state.ready, 'comparison needs actual ready result')
        else:
            text(self.failure, 'comparison failure')


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedDctAnalysis(Contract):
    """Evidence bundle reusing SignedDctResult, not a second spectrum/Fit type."""
    KIND: ClassVar[str] = 'signed_dct_analysis'
    source: DctSourceSnapshot
    config: SignedDctConfig
    diagnostic_config: SignedDctDiagnosticConfig
    state: ResultState
    candidates: tuple[SignedLambdaCandidate, ...] = ()
    selected_result: SignedDctResult | None = None
    peak_diagnostics: tuple[SignedPeakDiagnostic, ...] = ()
    comparisons: tuple[SignedDctComparison, ...] = ()
    flags: tuple[str, ...] = ()
    noise_sd_estimate: float | None = None
    contrast_to_noise: float | None = None
    resolution_status: str = 'not_evaluated'
    evaluations_used: int = 0
    bootstrap_status: str = 'not_implemented'
    interpretation: tuple[str, ...] = ()

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.source, DctSourceSnapshot) and isinstance(self.config, SignedDctConfig)
                and isinstance(self.diagnostic_config, SignedDctDiagnosticConfig) and isinstance(self.state, ResultState), 'analysis contract types required')
        for k, cls in (('candidates', SignedLambdaCandidate), ('peak_diagnostics', SignedPeakDiagnostic), ('comparisons', SignedDctComparison)):
            require(isinstance(getattr(self, k), (tuple, list)) and all(isinstance(v, cls) for v in getattr(self, k)), f'invalid {k}')
            object.__setattr__(self, k, tuple(getattr(self, k)))
        for k in ('flags', 'interpretation'):
            object.__setattr__(self, k, strings(getattr(self, k), k))
        for k in ('noise_sd_estimate', 'contrast_to_noise'):
            if getattr(self, k) is not None:
                object.__setattr__(self, k, number(getattr(self, k), k, nonnegative=True))
        require(self.bootstrap_status == 'not_implemented', 'bootstrap/CI not implemented by this profile')
        require(self.resolution_status in {'not_evaluated', 'no_resolvable_peak', 'single_candidate_not_process_count',
                'unresolved_overlap', 'separated_candidates_conditional', 'oscillatory_candidates_not_resolved',
                'boundary_limited_candidates', 'low_information_no_resolution_claim'}, 'unknown resolution status')
        integer(self.evaluations_used, 'evaluations_used')
        require(self.evaluations_used <= self.diagnostic_config.max_evaluations, 'evaluation budget exceeded')
        expected = self.config.gcv_lambdas if self.config.grid.lambda_mode == 'gcv' else (self.config.grid.regularization_lambda,)
        if self.candidates:
            require(tuple(c.result.config.grid.regularization_lambda for c in self.candidates) == expected, 'candidate list incomplete/reordered')
        for c in self.candidates:
            require(c.result.source_sha256 == self.source.content_sha256, 'candidate source identity conflict')
            expected_cfg = replace(self.config, grid=replace(self.config.grid, lambda_mode='manual',
                regularization_lambda=c.result.config.grid.regularization_lambda), gcv_lambdas=())
            require(c.result.config == expected_cfg, 'candidate changed weights/grid/solver/profile')
        if self.selected_result is not None:
            r = self.selected_result
            p = self.diagnostic_config
            expected_kinds = ('lambda',)*len(p.lambda_factors)+('tau_range',)*len(p.tau_ranges_s)+('grid',)*len(p.grid_sizes)+('terminal',)*len(p.terminal_offsets_in_S)
            require(tuple(c.kind for c in self.comparisons) == expected_kinds, 'comparison list incomplete/reordered; retain NOT_RUN reasons')
            require(r.source_sha256 == self.source.content_sha256 and r.config == self.config, 'selected source/config conflict')
            require(r.peaks == tuple(p.candidate for p in self.peak_diagnostics), 'selected peak evidence conflict')
            if self.config.grid.lambda_mode == 'gcv':
                valid = [c for c in self.candidates if c.failure is None]
                require(valid and r.selected_lambda == min(valid, key=lambda c: c.gcv_score).result.selected_lambda, 'GCV choice conflicts with candidates')
                chosen = min(valid, key=lambda c: c.gcv_score)
                require(r.gcv_scores == tuple(c.gcv_score for c in self.candidates)
                        and r.gcv_effective_df == tuple(c.effective_df for c in self.candidates)
                        and r.gcv_failures == tuple(c.failure for c in self.candidates), 'selected GCV evidence/candidates conflict')
            else:
                require(len(self.candidates) == 1, 'manual analysis needs its actual candidate')
                chosen = self.candidates[0]
            for key in ('tau_s', 'log10_tau', 'chi', 'quadrature_weights', 'quadrature_mass',
                        'reconstruction', 'residuals', 'selected_lambda', 'solve_diagnostics'):
                require(getattr(r, key) == getattr(chosen.result, key), 'selected numeric result differs from saved candidate')
        require(not self.state.ready or self.selected_result is not None and self.selected_result.state.ready, 'ready analysis needs selected payload')
        require(not self.state.ready or not self.flags or self.state.status == 'warning', 'diagnostic flags must not be hidden as completed')
        for c in self.comparisons:
            require(len(c.matches) == len(self.peak_diagnostics), 'comparison must describe every base peak')
            assigned = [j for j in c.matches if j is not None]
            require(len(assigned) == len(set(assigned)), 'comparison peak assigned more than once')
            if c.result is not None:
                if c.kind != 'terminal':
                    require(c.result.source_sha256 == self.source.content_sha256, 'nonterminal comparison changed source')
                else:
                    require(c.result.source.normalization is not None, 'terminal comparison normalization required')
                    self.source.normalization.spec.identity.assert_matches(c.result.source.normalization.spec.identity)
                if c.kind != 'lambda' and c.result.state.ready:
                    require(c.result.selected_lambda == self.selected_result.selected_lambda, 'control comparison changed selected lambda')
            for p, j, shift in zip(self.peak_diagnostics, c.matches, c.position_shifts_decades):
                if j is not None:
                    q = c.peak_diagnostics[j]
                    require(p.direction == q.direction and isclose(shift, abs(log10(p.candidate.position_s/q.candidate.position_s)), abs_tol=1e-12)
                            and shift <= self.diagnostic_config.match_tolerance_decades, 'comparison sign/shift/match conflict')
        for i, p in enumerate(self.peak_diagnostics):
            require(p.required_comparisons == len(self.comparisons)
                    and p.matched_comparisons == sum(c.matches[i] is not None for c in self.comparisons), 'peak stability count conflicts with comparisons')
            shifts = [c.position_shifts_decades[i] for c in self.comparisons if c.matches[i] is not None]
            require(p.max_position_shift_decades == (max(shifts) if shifts else None), 'peak stability shift conflicts with comparisons')
            require(p.stability_status != 'stable_conditional' or not any(c.failure is not None for c in self.comparisons), 'missing comparison cannot certify stability')
