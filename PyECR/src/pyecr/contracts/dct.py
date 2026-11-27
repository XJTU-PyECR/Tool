"""Signed spectra and frozen source bindings; no inverse solver or decomposition."""
from dataclasses import dataclass, fields
from enum import Enum
from math import isclose, isfinite, log10
from typing import ClassVar, get_type_hints

from pyecr.dct.core import CandidateTimescale
from pyecr.dct.config import DctAnalysisConfig
from pyecr.units import validate_pair
from .common import (Contract, ResultState, SourceIdentity, digest, integer, number, rows,
                     text, time_vector, vector, close_vectors, require)
from .normalization import SignedNormalizationResult
from .signed_solve import SignedDctSolveDiagnostics
from .signed_priors import SignedDctPriorSpec, SignedDctPriorEvidence
from .eligibility import FeatureEligibility


class SourceKind(str, Enum):
    EXPERIMENT = 'experiment'
    FIT_COMPONENT = 'fit_component'
    SIMULATION_CLEAN = 'simulation_clean'
    SIMULATION_OBSERVED = 'simulation_observed'


@dataclass(frozen=True, slots=True, kw_only=True)
class DctSourceSnapshot(Contract):
    KIND: ClassVar[str] = 'dct_source_snapshot'
    source_kind: SourceKind
    identity: SourceIdentity
    time_s: tuple[float, ...]
    signal: tuple[float, ...]
    raw_indices: tuple[int, ...]
    raw_count: int
    signal_type: str
    signal_unit: str
    representation: str
    state: ResultState
    normalization: SignedNormalizationResult | None = None
    parent_source_sha256: str | None = None
    component_key: str | None = None
    component_initial_value: float | None = None
    component_terminal_value: float | None = None
    parent_reliability: str = 'not_applicable'
    simulation_request_sha256: str | None = None
    time_unit: str = 's'
    t0_s: float = 0.0
    eligibility: FeatureEligibility | None = None

    def __post_init__(self):
        require(integer(self.schema_version,'schema_version',minimum=1) in {1,2},'unknown DCT source version')
        require((self.schema_version==2)==(self.eligibility is not None),'DCT source eligibility/version conflict')
        object.__setattr__(self, 'source_kind', SourceKind(self.source_kind))
        require(isinstance(self.identity, SourceIdentity) and isinstance(self.state, ResultState), 'identity/state types required')
        object.__setattr__(self, 'time_s', time_vector(self.time_s))
        object.__setattr__(self, 'signal', vector(self.signal, 'signal'))
        object.__setattr__(self, 'raw_count', integer(self.raw_count, 'raw_count', minimum=2))
        object.__setattr__(self, 'raw_indices', rows(self.raw_indices, self.raw_count))
        require(len(self.time_s) == len(self.signal) == len(self.raw_indices), 'source array/raw row mapping mismatch')
        require(self.time_unit == 's', 'DCT scientific time unit must be s')
        object.__setattr__(self, 't0_s', number(self.t0_s, 't0_s'))
        validate_pair(self.signal_type, self.signal_unit)
        require(self.representation in {'physical', 'progress', 'remaining', 'signed_terminal',
                                        'component_remaining', 'unqualified_normalized'}, 'unknown signal representation')
        require((self.representation == 'physical') == (self.signal_type != 'normalized'), 'signal representation/type conflict')
        if self.normalization is not None:
            require(isinstance(self.normalization, SignedNormalizationResult), 'normalization type required')
            self.identity.assert_matches(self.normalization.output_identity or self.normalization.spec.identity)
            require(self.normalization.state.has_payload and self.representation == 'signed_terminal', 'signed normalization payload required')
            require(self.raw_indices == self.normalization.spec.raw_indices and self.raw_count == self.normalization.spec.raw_count,
                    'normalization original row mapping conflict')
            require(self.t0_s == self.normalization.spec.t0_s, 'normalization t0 conflict')
            require(self.signal == self.normalization.normalized_signal and self.signal_unit == '1', 'normalized signal/unit conflict')
        require(self.representation != 'signed_terminal' or self.normalization is not None, 'signed source needs terminal and scale provenance')
        if self.source_kind is SourceKind.FIT_COMPONENT:
            self.identity.require_parent()
            expected_keys = {'M2_DUAL_HO_D': {'component_H', 'component_O'},
                             'M3_DUAL_HO_DK': {'component_H', 'component_O'},
                             'M8_DUAL_K': {'component_1', 'component_2'}}
            require(self.component_key in expected_keys.get(self.identity.model_id, set()), 'component key/parent model identity conflict')
            digest(self.parent_source_sha256, 'parent_source_sha256')
            text(self.component_key, 'component_key')
            require(self.parent_reliability in {'reliable', 'unreliable', 'unverified', 'unsupported'}, 'unknown parent reliability')
            for key in ('component_initial_value', 'component_terminal_value'):
                object.__setattr__(self, key, number(getattr(self, key), key))
            require(self.simulation_request_sha256 is None, 'component source cannot masquerade as simulated observations')
            if self.eligibility is not None:
                require(isinstance(self.eligibility,FeatureEligibility),'typed eligibility required')
                require(isinstance(self.eligibility.source,SourceIdentity),'component eligibility requires source identity')
                self.eligibility.source.assert_matches(self.identity)
                require(self.parent_reliability==self.eligibility.parent_reliability,'original parent reliability conflict')
        else:
            require(self.eligibility is None,'whole source cannot claim component eligibility')
            require(self.parent_source_sha256 is None and self.component_key is None
                    and self.component_initial_value is None and self.component_terminal_value is None
                    and self.parent_reliability == 'not_applicable', 'noncomponent source has component-only fields')
            require(self.representation != 'component_remaining', 'component_remaining requires a Fit component')
        if self.source_kind in {SourceKind.SIMULATION_CLEAN, SourceKind.SIMULATION_OBSERVED}:
            digest(self.simulation_request_sha256, 'simulation_request_sha256')
        else:
            require(self.simulation_request_sha256 is None, 'non-simulation source has simulation request identity')

    def require_usable(self):
        require(self.state.ready, 'source is unavailable or stale')
        if self.normalization is not None:
            require(self.normalization.state.ready, 'normalization is unavailable or stale')
        if self.source_kind is SourceKind.FIT_COMPONENT:
            if self.eligibility is not None:self.eligibility.require_component(self.identity,self.parent_reliability)
            else:require(self.parent_reliability == 'reliable', 'parent Fit/Diagnostics reliability not supported')
            require(self.component_initial_value != self.component_terminal_value, 'zero-amplitude component cannot be normalized')

    def to_dict(self):
        value=Contract.to_dict(self)
        if self.schema_version==1:value.pop('eligibility')
        return value

    @classmethod
    def from_dict(cls,value):
        from .common import exact_fields,decode
        require(hasattr(value,'keys') and 'schema_version' in value,"DctSourceSnapshot missing fields: ['schema_version']")
        version=integer(value['schema_version'],'schema_version',minimum=1)
        require(version in {1,2},'unknown DCT source version')
        names=[f.name for f in fields(cls) if version==2 or f.name!='eligibility']
        exact_fields(value,['contract_kind',*names],cls.__name__)
        require(value['contract_kind']==cls.KIND,'DCT source kind conflict')
        hints=get_type_hints(cls)
        return cls(**{n:decode(hints[n],value[n]) for n in names})

    @classmethod
    def capture_experiment(cls, dataset, recipe, *, config=None, normalization=None, t0_s=None):
        """Freeze existing observations; no preprocessing, fitting or resampling."""
        import numpy as np
        if t0_s is None:
            require(not any(r.operation == 'set_manual_t0' for r in dataset.history),
                    'provide cumulative t0_s explicitly after a time-origin operation; no unit/origin guessing')
            t0_s = 0.0
        identity = SourceIdentity.capture(dataset, recipe, config=config)
        normalized = normalization is not None
        source_kind = SourceKind.EXPERIMENT
        simulation_identity = None
        if dataset.metadata.get('simulation_origin') is not None or str(dataset.metadata.get('source_kind','')).startswith('simulation_'):
            from pyecr.services.simulation_transfer import validate_simulation_origin
            simulation = validate_simulation_origin(dataset)
            source_kind = SourceKind(dataset.metadata['source_kind'])
            simulation_identity = simulation.request_sha256
        return cls(source_kind=source_kind, identity=identity, simulation_request_sha256=simulation_identity,
                   time_s=tuple(dataset.processed_time),
                   signal=normalization.normalized_signal if normalized else tuple(dataset.processed_signal),
                   raw_indices=tuple(int(i) for i in np.flatnonzero(dataset.mask)), raw_count=len(dataset.raw_time),
                   signal_type='normalized' if normalized else str(dataset.metadata['signal_type']),
                   signal_unit='1' if normalized else str(dataset.metadata['signal_unit']),
                   representation='signed_terminal' if normalized else str(dataset.metadata.get('normalization_mode',
                       'unqualified_normalized' if dataset.metadata['signal_type'] == 'normalized' else 'physical')),
                   normalization=normalization, time_unit=str(dataset.metadata['time_unit']), t0_s=t0_s, state=ResultState())


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedDctConfig(Contract):
    KIND: ClassVar[str] = 'signed_dct_config'
    grid: DctAnalysisConfig
    gcv_lambdas: tuple[float, ...] = ()
    observation_weights: tuple[float, ...] = ()
    kernel: str = 'exp_decay'
    density_measure: str = 'dlog10_tau'
    regularizer: str = 'sqrt_dlog10_tau_second_derivative_v1'
    baseline_mode: str = 'fixed_zero'
    solver: str = 'augmented_svd'
    algorithm_version: str = 'signed-tikhonov/1'
    seed: int | None = None
    rng_algorithm: str | None = None
    prior_spec: SignedDctPriorSpec | None = None

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version', minimum=1) in {1, 2, 3},
                'unsupported signed_dct_config schema_version')
        object.__setattr__(self, 'schema_version', int(self.schema_version))
        require(isinstance(self.grid, DctAnalysisConfig), 'reuse DctAnalysisConfig for grid/lambda policy')
        self.grid.validate()
        require(not self.grid.use_classical_bi, 'signed inversion does not require geometry or physical-model suggestions')
        for key in ('gcv_lambdas', 'observation_weights'):
            object.__setattr__(self, key, vector(getattr(self, key), key, positive=True))
        if self.grid.lambda_mode == 'gcv':
            require(len(self.gcv_lambdas) >= 3 and all(a < b for a, b in zip(self.gcv_lambdas, self.gcv_lambdas[1:])),
                    'GCV requires an explicit increasing positive lambda grid')
        else:
            require(not self.gcv_lambdas, 'manual lambda must not carry a GCV grid')
        require((self.kernel, self.density_measure, self.baseline_mode) ==
                ('exp_decay', 'dlog10_tau', 'fixed_zero'),
                'unknown signed kernel/measure/regularizer/baseline/algorithm')
        require((self.schema_version == 1 and self.algorithm_version == 'signed-tikhonov/1'
                 and self.solver in {'augmented_svd', 'augmented_qr'}) or
                (self.schema_version == 2 and self.algorithm_version == 'signed-tikhonov/2'
                 and self.solver in {'reused_svd', 'projected_svd'}) or
                (self.schema_version == 3 and self.algorithm_version == 'signed-priors/1'
                 and self.solver == 'prior_svd'), 'unknown signed solver/version pair')
        if self.schema_version < 3:
            require(self.prior_spec is None and self.regularizer=='sqrt_dlog10_tau_second_derivative_v1',
                    'historical config cannot acquire a new prior')
        else:
            require(isinstance(self.prior_spec,SignedDctPriorSpec) and self.regularizer=='configured_signed_prior/1','explicit typed prior required')
            require(not self.prior_spec.adaptive or self.grid.lambda_mode=='manual','adaptive priors cannot use frozen-weight GCV as nonlinear selection')
        if self.seed is None:
            require(self.rng_algorithm is None, 'RNG requires an explicit actual seed')
        else:
            object.__setattr__(self, 'seed', integer(self.seed, 'seed'))
            require(self.rng_algorithm == 'PCG64', 'unknown sensitivity RNG algorithm')

    def to_dict(self):
        value=Contract.to_dict(self)
        if self.schema_version<3:value.pop('prior_spec')
        return value

    @classmethod
    def from_dict(cls, value):
        from .common import exact_fields, decode
        require('schema_version' in value,'missing signed config schema_version')
        names = [f.name for f in fields(cls) if value['schema_version']==3 or f.name!='prior_spec']
        exact_fields(value, ['contract_kind', *names], cls.__name__)
        require(value['contract_kind'] == cls.KIND, 'expected signed config kind')
        hints = get_type_hints(cls)
        return cls(**{name: decode(hints[name], value[name]) for name in names})


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedDctResult(Contract):
    KIND: ClassVar[str] = 'signed_dct_result'
    source: DctSourceSnapshot
    config: SignedDctConfig
    state: ResultState
    run_id: str
    tau_s: tuple[float, ...] = ()
    log10_tau: tuple[float, ...] = ()
    chi: tuple[float, ...] = ()
    quadrature_weights: tuple[float, ...] = ()
    quadrature_mass: tuple[float, ...] = ()
    reconstruction: tuple[float, ...] = ()
    residuals: tuple[float, ...] = ()
    selected_lambda: float | None = None
    gcv_scores: tuple[float | None, ...] = ()
    gcv_effective_df: tuple[float | None, ...] = ()
    gcv_failures: tuple[str | None, ...] = ()
    peaks: tuple[CandidateTimescale, ...] = ()
    chi_unit: str = '1/decade'
    mass_unit: str = '1'
    solve_diagnostics: SignedDctSolveDiagnostics | None = None
    prior_evidence: SignedDctPriorEvidence | None = None

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version', minimum=1) in {1, 2, 3},
                'unsupported signed_dct_result schema_version')
        require(self.schema_version != 1 or self.solve_diagnostics is None,
                'v1 signed result cannot claim newly recorded solve diagnostics')
        require(isinstance(self.source, DctSourceSnapshot) and isinstance(self.config, SignedDctConfig)
                and isinstance(self.state, ResultState), 'source/config/state types required')
        require(self.schema_version != 1 or self.config.schema_version == 1,
                'historical v1 spectrum cannot claim a v2 solve without new recorded evidence')
        require((self.schema_version==3)==(self.config.schema_version==3),'prior spectrum/config version conflict')
        require(self.schema_version==3 or self.prior_evidence is None,'historical spectrum cannot acquire prior evidence')
        if self.schema_version >= 2:
            require((isinstance(self.solve_diagnostics, SignedDctSolveDiagnostics) if self.state.has_payload
                     else self.solve_diagnostics is None), 'v2 payload requires actual solve diagnostics; unavailable result has none')
        text(self.run_id, 'run_id')
        for key in ('tau_s', 'log10_tau', 'chi', 'quadrature_weights', 'quadrature_mass', 'reconstruction', 'residuals'):
            object.__setattr__(self, key, vector(getattr(self, key), key))
        for key in ('gcv_scores', 'gcv_effective_df', 'gcv_failures', 'peaks'):
            require(isinstance(getattr(self, key), (list, tuple)), f'{key} must be an array')
            object.__setattr__(self, key, tuple(getattr(self, key)))
        require(self.chi_unit == '1/decade' and self.mass_unit == '1', 'signed spectrum units must retain density/mass distinction')
        if not self.state.has_payload:
            require(all(not getattr(self, k) for k in ('tau_s', 'log10_tau', 'chi', 'quadrature_weights',
                    'quadrature_mass', 'reconstruction', 'residuals', 'gcv_scores', 'gcv_effective_df', 'gcv_failures', 'peaks'))
                    and self.selected_lambda is None, 'unavailable signed result must not fabricate arrays')
            if self.prior_evidence is not None:
                require(self.prior_evidence.status.startswith('FAIL'),'unavailable result may retain failed attempt evidence only')
                require(not self.prior_evidence.attempted_chi or len(self.prior_evidence.attempted_chi)==self.config.grid.n_tau,'failed attempt/grid conflict')
                require(not self.prior_evidence.attempted_reconstruction or len(self.prior_evidence.attempted_reconstruction)==len(self.source.signal),'failed reconstruction/source conflict')
            return
        if self.state.ready:
            self.source.require_usable()
        require(self.source.representation in {'signed_terminal', 'component_remaining'}, 'signed solver requires explicit terminal/remaining representation')
        n = len(self.tau_s)
        require(n == self.config.grid.n_tau and all(len(getattr(self, k)) == n for k in
                ('log10_tau', 'chi', 'quadrature_weights', 'quadrature_mass')), 'spectrum arrays/grid length mismatch')
        require(all(v > 0 for v in self.tau_s) and all(a < b for a, b in zip(self.tau_s, self.tau_s[1:])), 'tau grid must be positive/increasing')
        close_vectors(self.log10_tau, tuple(log10(v) for v in self.tau_s), 'tau/log10_tau', atol=1e-12, rtol=1e-12)
        g = self.config.grid
        require(isclose(self.tau_s[0], g.tau_min_s, rel_tol=1e-12) and isclose(self.tau_s[-1], g.tau_max_s, rel_tol=1e-12), 'tau bounds/config conflict')
        h = (self.log10_tau[-1] - self.log10_tau[0]) / (n - 1)
        close_vectors(self.log10_tau, tuple(self.log10_tau[0] + i*h for i in range(n)), 'uniform log10 grid', atol=1e-12, rtol=1e-12)
        close_vectors(self.quadrature_weights, (h/2, *([h]*(n-2)), h/2), 'trapezoidal quadrature weights', atol=1e-12, rtol=1e-12)
        close_vectors(self.quadrature_mass, tuple(c*w for c, w in zip(self.chi, self.quadrature_weights)), 'chi/quadrature mass')
        close_vectors(self.residuals, tuple(y-f for y, f in zip(self.source.signal, self.reconstruction)), 'source/reconstruction residual')
        require(len(self.reconstruction) == len(self.source.signal), 'reconstruction must cover every scientific observation')
        if self.config.observation_weights:
            require(len(self.config.observation_weights) == len(self.source.signal), 'observation weights length mismatch')
        object.__setattr__(self, 'selected_lambda', number(self.selected_lambda, 'selected_lambda', nonnegative=True))
        if g.lambda_mode == 'manual':
            require(self.selected_lambda == g.regularization_lambda and not (self.gcv_scores or self.gcv_effective_df or self.gcv_failures), 'manual lambda/result conflict')
        else:
            require(self.selected_lambda in self.config.gcv_lambdas, 'selected lambda not in configured grid')
            require(len(self.gcv_scores) == len(self.gcv_effective_df) == len(self.gcv_failures) == len(self.config.gcv_lambdas), 'GCV evidence must cover all candidates')
            for score, df, failure in zip(self.gcv_scores, self.gcv_effective_df, self.gcv_failures):
                if failure is not None:
                    text(failure, 'GCV failure'); require(score is None and df is None, 'failed GCV candidate must not have a fake score/df')
                else:
                    number(score, 'GCV score', nonnegative=True)
                    require(0 <= number(df, 'GCV effective_df') <= len(self.source.signal), 'invalid effective degrees of freedom')
            require(self.gcv_failures[self.config.gcv_lambdas.index(self.selected_lambda)] is None, 'selected GCV candidate failed')
            valid = [i for i, failure in enumerate(self.gcv_failures) if failure is None]
            best = min(valid, key=lambda i: self.gcv_scores[i])
            require(self.selected_lambda == self.config.gcv_lambdas[best], 'GCV selected lambda conflicts with minimum finite score')
        for p in self.peaks:
            require(isinstance(p, CandidateTimescale), 'reuse CandidateTimescale for signed peaks')
            ix = integer(p.grid_index, 'peak grid_index')
            require(ix < n and p.position_s == self.tau_s[ix] and p.label == 'candidate_timescale', 'peak grid/interpretation conflict')
            require(number(p.area, 'signed peak area') != 0, 'zero-area peak has no signed contribution')
            number(p.width_decades, 'peak width', nonnegative=True)
        if self.schema_version >= 2:
            d = self.solve_diagnostics
            if self.schema_version==3:
                e=self.prior_evidence
                require(isinstance(e,SignedDctPriorEvidence) and e.family==self.config.prior_spec.family,'prior evidence/config conflict')
                require(e.status in {'CONVERGED','LINEAR_CONVERGED'},'failed adaptive attempt cannot publish a successful spectrum')
                expected_free=tuple(range(1,n-1)) if self.config.prior_spec.hard_zero_control else tuple(range(n))
                require(e.free_indices==expected_free,'prior constrained subspace conflict')
                require(d.augmented_shape==(len(self.source.signal)+d.regularizer_rows,len(expected_free)),'prior augmented shape conflict')
                require(d.df_semantics==e.df_semantics,'prior conditional trace semantics conflict')
                require(isclose(sum(v for _,v in e.penalties),d.regularization_penalty,rel_tol=1e-10,abs_tol=1e-13),'prior penalty breakdown conflict')
                if self.config.prior_spec.hard_zero_control:
                    require(self.chi[0]==0 and self.chi[-1]==0 and self.state.status=='warning','hard zero is a declared sensitivity control')
            else:require(d.augmented_shape == (len(self.source.signal)+n-2, n), 'augmented shape/source/grid conflict')
            require(isclose(d.grid_step_decades, h, rel_tol=1e-12), 'regularizer grid step conflict')
            expected_driver = {'augmented_svd': 'gelsd', 'augmented_qr': 'gelsy',
                               'reused_svd': 'gesdd', 'projected_svd': 'gesdd','prior_svd':'gesdd'}[self.config.solver]
            require(d.lapack_driver == expected_driver, 'solver/driver conflict')
            require(d.schema_version == self.config.schema_version, 'config/solve evidence version conflict')
            require(self.selected_lambda != 0 or d.regularization_penalty == 0, 'zero lambda must have zero penalty')
            weights = self.config.observation_weights or (1.,)*len(self.residuals)
            require(isclose(d.weighted_sse, sum(w*r*r for w, r in zip(weights, self.residuals)), rel_tol=1e-10, abs_tol=1e-14),
                    'weighted SSE/residual conflict')
            if self.state.ready and (d.effective_rank < n or d.svd_rank < n
                    or d.condition_number is None or d.condition_number > 1/(2.220446049250313e-16**.5)):
                require(self.state.status == 'warning', 'rank/condition warning must not be hidden as completed')
            if self.source.normalization is not None:
                require(all(isfinite(v) for values in (self.physical_reconstruction, self.physical_residuals,
                        self.physical_chi, self.physical_mass) for v in values), 'nonfinite physical-unit restoration')

    def to_dict(self):
        # Preserve the exact v1 wire shape and content hash. No invented diagnostics.
        result = Contract.to_dict(self)
        if self.schema_version == 1:
            result.pop('solve_diagnostics')
        if self.schema_version<3:result.pop('prior_evidence')
        return result

    @classmethod
    def from_dict(cls, value):
        from collections.abc import Mapping
        from .common import exact_fields, decode, ContractError
        require(isinstance(value, Mapping), 'SignedDctResult must be an object')
        require('schema_version' in value, 'SignedDctResult missing schema_version')
        version = integer(value['schema_version'], 'schema_version', minimum=1)
        require(version in {1, 2, 3}, 'unsupported signed_dct_result schema_version')
        names = [f.name for f in fields(cls) if (version>=2 or f.name!='solve_diagnostics') and (version==3 or f.name!='prior_evidence')]
        exact_fields(value, ['contract_kind', *names], cls.__name__)
        require(value['contract_kind'] == cls.KIND, 'expected contract_kind=signed_dct_result')
        hints = get_type_hints(cls)
        try:
            return cls(**{name: decode(hints[name], value[name]) for name in names})
        except (ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f'invalid {cls.KIND}: {exc}') from exc

    @property
    def physical_unit(self):
        """Unit of the source before signed normalization; no component-unit guessing."""
        return None if self.source.normalization is None else self.source.normalization.spec.signal_unit

    @property
    def physical_chi_unit(self):
        return None if self.physical_unit is None else self.physical_unit+'/decade'

    @property
    def physical_mass_unit(self):
        return self.physical_unit

    def _restore(self, values, *, add_terminal=False):
        norm = self.source.normalization
        if norm is None or not self.state.has_payload:
            return None
        offset = norm.reference_value if add_terminal else 0.
        return tuple(offset+norm.scale_value*v for v in values)

    @property
    def physical_reconstruction(self):
        return self._restore(self.reconstruction, add_terminal=True)

    @property
    def physical_residuals(self):
        return self._restore(self.residuals)

    @property
    def physical_chi(self):
        """Density in physical_unit/decade, relative to the recorded terminal."""
        return self._restore(self.chi)

    @property
    def physical_mass(self):
        return self._restore(self.quadrature_mass)

    @property
    def source_sha256(self):
        return self.source.content_sha256

    @property
    def seed(self):
        return self.config.seed


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentDctBundle(Contract):
    KIND: ClassVar[str] = 'component_dct_bundle'
    parent_source: DctSourceSnapshot
    parent_model_id: str
    parent_reliability: str
    component_keys: tuple[str, str]
    labels: tuple[str, str]
    label_semantics: str
    component_results: tuple[SignedDctResult | None, SignedDctResult | None]
    unavailable_reasons: tuple[str | None, str | None]
    B_j: tuple[float, float]
    S_total: float
    delta: float
    baseline: float
    state: ResultState
    overall_result: SignedDctResult | None = None
    component_capture: 'ComponentSourceCapture | None' = None
    independent_results: 'tuple[ComponentSpectrumResult | None, ComponentSpectrumResult | None]' = (None, None)
    overall_analysis: 'SignedDctAnalysis | None' = None

    def __post_init__(self):
        if self.schema_version == 2 and not isinstance(self.schema_version, bool):
            self._validate_v2()
            return
        Contract.__post_init__(self)
        require(self.component_capture is None and self.independent_results == (None,None)
                and self.overall_analysis is None, 'v1 bundle cannot carry P09 payload')
        require(isinstance(self.parent_source, DctSourceSnapshot) and isinstance(self.state, ResultState), 'parent source/state required')
        self.parent_source.identity.require_parent()
        require(self.parent_source.source_kind is not SourceKind.FIT_COMPONENT, 'bundle parent must be whole observations, not another component')
        require(self.parent_source.normalization is not None, 'common scale requires recorded whole-source terminal normalization')
        require(self.parent_model_id in {'M2_DUAL_HO_D', 'M3_DUAL_HO_DK', 'M8_DUAL_K'}, 'unsupported component parent model')
        require(self.parent_model_id == self.parent_source.identity.model_id, 'bundle model/config identity conflict')
        require(self.parent_reliability in {'reliable', 'unreliable', 'unverified', 'unsupported'}, 'unknown parent reliability')
        for key in ('component_keys', 'labels', 'component_results', 'unavailable_reasons', 'B_j'):
            require(isinstance(getattr(self, key), (tuple, list)) and len(getattr(self, key)) == 2, f'{key} needs exactly two slots')
            object.__setattr__(self, key, tuple(getattr(self, key)))
        expected = ('component_1', 'component_2') if self.parent_model_id == 'M8_DUAL_K' else ('component_H', 'component_O')
        require(self.component_keys == expected, 'component slots must match actual parent model')
        require(self.label_semantics == ('effective_modes_not_species' if self.parent_model_id == 'M8_DUAL_K' else 'declared_H_O_slots_not_species_proof'), 'component label semantics conflict')
        for label in self.labels:
            text(label, 'component label')
        object.__setattr__(self, 'B_j', vector(self.B_j, 'B_j'))
        for key in ('S_total', 'delta', 'baseline'):
            object.__setattr__(self, key, number(getattr(self, key), key, positive=key == 'S_total'))
        require(self.S_total == self.parent_source.normalization.scale_value, 'S_total conflicts with whole-source scale provenance')
        if self.state.ready:
            self.parent_source.require_usable()
            require(self.parent_reliability == 'reliable' and any(r is not None for r in self.component_results), 'bundle requires reliable parent and available component evidence')
        for key, result, reason, b in zip(self.component_keys, self.component_results, self.unavailable_reasons, self.B_j):
            if result is None:
                text(reason, 'component unavailable reason')
                continue
            require(isinstance(result, SignedDctResult) and reason is None and b != 0, 'available component conflicts with reason/zero amplitude')
            if self.state.ready:
                require(result.state.ready, 'ready bundle cannot contain unavailable/stale component results')
            source = result.source
            self.parent_source.identity.assert_matches(source.identity)
            require(source.source_kind is SourceKind.FIT_COMPONENT and source.component_key == key
                    and source.parent_source_sha256 == self.parent_source.content_sha256, 'component parent source identity conflict')
            require(source.time_s == self.parent_source.time_s and source.raw_indices == self.parent_source.raw_indices
                    and source.raw_count == self.parent_source.raw_count, 'component observation/raw mapping conflict')
            require(source.parent_reliability == self.parent_reliability
                    and isclose(source.component_initial_value-source.component_terminal_value, b, rel_tol=1e-10, abs_tol=1e-12), 'component B_j/reliability conflict')
        if self.overall_result is not None:
            require(isinstance(self.overall_result, SignedDctResult)
                    and self.overall_result.source_sha256 == self.parent_source.content_sha256, 'overall result parent conflict')
        if all(result is not None for result in self.component_results):
            fit_terminal = self.baseline + sum(result.source.component_terminal_value for result in self.component_results)
            expected_delta = (fit_terminal-self.parent_source.normalization.reference_value)/self.S_total
            require(isclose(self.delta, expected_delta, rel_tol=1e-10, abs_tol=1e-12), 'delta conflicts with common baseline/terminal reference')

    @property
    def common_scale_factors(self):
        return tuple(b/self.S_total for b in self.B_j)

    def _validate_v2(self):
        from .component_source import ComponentSourceCapture
        from .component_spectrum import ComponentSpectrumResult
        from .signed_analysis import SignedDctAnalysis
        require(integer(self.schema_version,'schema_version',minimum=1)==2,'unknown component bundle version')
        for name in ('S_total','delta','baseline'):number(getattr(self,name),name,positive=name=='S_total')
        require(isinstance(self.component_capture,ComponentSourceCapture), 'v2 requires actual P08 capture')
        c=self.component_capture;c.require_usable()
        require(isinstance(self.state,ResultState) and self.state.has_payload, 'v2 bundle requires numerical payload state')
        p=self.parent_source
        require(isinstance(p,DctSourceSnapshot) and p.normalization is not None,'v2 needs experimental P02 terminal/scale')
        p.require_usable();c.identity.assert_matches(p.identity)
        require(p.source_kind is c.parent_source.source_kind
                and p.simulation_request_sha256==c.parent_source.simulation_request_sha256,
                'whole/component parent source kind conflict')
        require(p.time_s == c.parent_source.time_s and p.raw_indices == c.parent_source.raw_indices
                and p.raw_count == c.parent_source.raw_count and p.t0_s == c.parent_source.t0_s,'whole/component observation binding conflict')
        norm=p.normalization
        close_vectors(tuple(norm.reference_value+norm.scale_value*y for y in p.signal),c.parent_source.signal,'whole normalization restoration')
        require(self.parent_model_id == c.identity.model_id and self.parent_reliability == c.components[0].parent_reliability
                and (self.parent_reliability=='reliable' or c.eligibility is not None and c.eligibility.can_run)
                and tuple(self.component_keys) == tuple(s.component_key for s in c.components)
                and tuple(self.labels) == c.labels and self.label_semantics == c.label_semantics,'parent/component roles conflict')
        require(tuple(self.component_results) == (None,None),'nonnegative component spectra must not masquerade as signed solver results')
        require(tuple(self.B_j) == c.B_j and self.baseline == c.baseline and self.S_total == norm.scale_value,'common scale/B/baseline conflict')
        expected=(c.asymptotic_total-norm.reference_value)/self.S_total
        require(isclose(self.delta,expected,rel_tol=1e-10,abs_tol=1e-12),'explicit delta conflict')
        for name in ('component_keys','labels','component_results','unavailable_reasons','B_j','independent_results'):
            require(isinstance(getattr(self,name),(tuple,list)) and len(getattr(self,name))==2,f'{name} requires two slots')
            object.__setattr__(self,name,tuple(getattr(self,name)))
        for physical,result,reason,b in zip(c.components,self.independent_results,self.unavailable_reasons,self.B_j):
            if result is None:
                text(reason,'component unavailable reason');continue
            require(isinstance(result,ComponentSpectrumResult) and result.state.ready and reason is None,'actual usable component spectrum required')
            require(result.physical_source == physical and result.source.parent_source_sha256 == p.content_sha256
                    and result.B == b and result.S_total == self.S_total,'independent/common spectrum source conflict')
        require(isinstance(self.overall_analysis,SignedDctAnalysis) and self.overall_result == self.overall_analysis.selected_result
                and self.overall_result is not None and self.overall_result.state.ready
                and self.overall_result.source == p,'actual whole signed analysis required')
        exact=tuple(self.delta+sum(b/self.S_total*(v-s.component_terminal_value)/b for b,v,s in
            zip(self.B_j,values,c.components)) for values in zip(*(s.signal for s in c.components)))
        close_vectors(self.normalized_parent_fit,exact,'explicit delta plus exact component reconstruction',atol=3e-12,rtol=3e-12)

    @property
    def normalized_parent_fit(self):
        if self.component_capture is None:return ()
        n=self.parent_source.normalization
        return tuple((v-n.reference_value)/self.S_total for v in self.component_capture.fit_total)

    @property
    def component_time_reconstruction(self):
        if not all(self.independent_results):return ()
        return tuple(self.delta+sum(b/self.S_total*r for b,r in zip(self.B_j,values))
                     for values in zip(*(s.reconstruction for s in self.independent_results)))

    @property
    def component_time_residuals(self):
        return tuple(a-b for a,b in zip(self.normalized_parent_fit,self.component_time_reconstruction))

    def to_dict(self):
        result=Contract.to_dict(self)
        if self.schema_version == 1:
            for name in ('component_capture','independent_results','overall_analysis'):result.pop(name)
        return result

    @classmethod
    def from_dict(cls,value):
        from .common import exact_fields,decode
        from .component_source import ComponentSourceCapture
        from .component_spectrum import ComponentSpectrumResult
        from .signed_analysis import SignedDctAnalysis
        require(isinstance(value,dict) or hasattr(value,'keys'),'bundle must be object')
        require('schema_version' in value,'bundle version missing')
        version=integer(value['schema_version'],'schema_version',minimum=1)
        require(version in {1,2},'unsupported component bundle version')
        added={'component_capture','independent_results','overall_analysis'}
        names=[f.name for f in fields(cls) if version==2 or f.name not in added]
        exact_fields(value,['contract_kind',*names],cls.__name__)
        require(value['contract_kind']==cls.KIND,'component bundle kind conflict')
        hints=get_type_hints(cls,localns=dict(ComponentSourceCapture=ComponentSourceCapture,
            ComponentSpectrumResult=ComponentSpectrumResult,SignedDctAnalysis=SignedDctAnalysis))
        return cls(**{name:decode(hints[name],value[name]) for name in names})


# Resolve versioned bundle annotations only after the shared source types exist.
from .component_source import ComponentSourceCapture
from .component_spectrum import ComponentSpectrumResult
from .signed_analysis import SignedDctAnalysis
