"""Given-parameter forward requests/results; never a FitResult or optimizer call."""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from math import log10, prod
from typing import ClassVar, get_type_hints
from pyecr.config import BoundaryConfig, ModelConfig, SolverConfig, _freeze_json
from pyecr.model_display import MODEL_DISPLAY, VERSION as DISPLAY_VERSION
from pyecr.models.registry import DEFAULT_MODEL_REGISTRY
from pyecr.models.ids import ModelID
from pyecr.units import validate_pair
from .common import (Contract, ResultState, SourceIdentity, require, text, integer,
                     number, time_vector, vector, close_vectors, exact_fields, decode, ContractError)
from .simulation_units import SimulationUnitInputs

SUPPORT_MATRIX_VERSION = 'simulation-support/1'
SIMULATION_MODEL_IDS = frozenset(item.model_id for item in MODEL_DISPLAY)
SIMULATION_POINT_BUDGET = 50000
SIMULATION_OBSERVATION_VERSION = 'simulation-observation/1'
SINGLE_SIMULATION_ADAPTER_VERSION = 'single-ideal-forward/1'
SINGLE_SIMULATION_INTERPRETATION = ('Forward calculation only; no Fit, statistical reliability, '
    'confidence interval or R2 is evaluated. Given parameters remain supplied values, not estimates. '
    'The signal uses the model linear small-step assumption; no nonlinear electrical conversion is performed.')
SINGLE_SIMULATION_EVIDENCE_FIELDS = ('adapter_version', 'request_sha256', 'source_kind', 'time_unit', 'signal_unit', 'length_unit',
    'parameter_units', 'geometry_convention', 'active_face_convention', 'signal_convention',
    'term_tol', 'absolute_tolerance_usage', 'root_mode', 'root_solver_policy', 'software_versions',
    'remaining_fraction', 'completion_fraction', 'axis_results', 'kernel_diagnostics',
    'nonfinite_diagnostic_paths', 't_zero_rule', 'evaluated_parameters', 'numerical_warning_codes', 'interpretation')
SIMULATION_ADAPTER_VERSION = 'six-model-production-forward/1'
SIMULATION_EVIDENCE_FIELDS = (*SINGLE_SIMULATION_EVIDENCE_FIELDS, 'model_display',
    'boundary_diagnostics', 'component_evidence', 'given_diffusion_background', 'limit_signals')
SIMULATION_COMPONENT_FIELDS = ('role', 'sign', 'amplitude', 'kinetics', 'remaining_fraction',
    'completion_fraction', 'ideal_completion_fraction', 'axis_results', 'kernel_diagnostics',
    'surface_rates', 'speed_label')
SIMULATION_INTERPRETATION = (SINGLE_SIMULATION_INTERPRETATION +
    ' H/O slots record parent-model roles only, not chemical identification. '
    'K-only modes have no D or Bi. All D values in a finite-D simulation are given '
    'background parameters; no conditional fit or uncertainty propagation is performed. '
    'A measured-history asymptote is conditional on eventual continuation at the declared final gas value.')


def model_support_matrix():
    """Reuse registry/display metadata; an immutable declared matrix, not a probe."""
    return _freeze_json({'version': SUPPORT_MATRIX_VERSION, 'display_version': DISPLAY_VERSION,
        'models': [{**DEFAULT_MODEL_REGISTRY.get(item.model_id).to_dict(),
                    'display_label': item.label} for item in MODEL_DISPLAY]}, path='support_matrix')


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationRequest(Contract):
    KIND: ClassVar[str] = 'simulation_request'
    model: ModelConfig
    boundary: BoundaryConfig
    solver_config: SolverConfig
    given_parameters: Mapping[str, float]
    time_s: tuple[float, ...]
    signal_type: str
    signal_unit: str
    job_id: str
    noise_sd: float = 0.0
    seed: int | None = None
    rng_algorithm: str | None = None
    sampling_kind: str = 'explicit'
    parameter_origin: str = 'manual'
    copied_from: SourceIdentity | None = None
    support_matrix_version: str = SUPPORT_MATRIX_VERSION
    algorithm_version: str = 'production-forward/1'
    name: str = 'Simulation'
    sequence_number: int = 0
    input_units: SimulationUnitInputs | None = None

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version', minimum=1) in {1, 2, 3},
                'unsupported simulation_request schema_version')
        require((self.schema_version==3)==(self.input_units is not None),'Simulation input units/version conflict')
        text(self.name, 'simulation name')
        require(len(self.name) <= 256, 'simulation name exceeds 256 characters')
        object.__setattr__(self, 'sequence_number', integer(self.sequence_number, 'sequence_number'))
        require(self.sequence_number < 2**32, 'sequence_number exceeds uint32 budget')
        require(self.schema_version >= 2 or (self.name == 'Simulation' and self.sequence_number == 0),
                'simulation request v1 cannot claim a new name/sequence')
        require(isinstance(self.model, ModelConfig) and isinstance(self.boundary, BoundaryConfig)
                and isinstance(self.solver_config, SolverConfig), 'reuse ModelConfig/BoundaryConfig/SolverConfig')
        require(isinstance(self.model.model_id, ModelID), 'simulation model_id must be a canonical ModelID')
        require(self.model.model_id.value in SIMULATION_MODEL_IDS, 'only six displayed production models are supported')
        require(self.support_matrix_version == SUPPORT_MATRIX_VERSION and self.algorithm_version == 'production-forward/1',
                'unknown simulation support/algorithm version')
        desc = DEFAULT_MODEL_REGISTRY.validate_config(self.model)
        self.boundary.validate(); self.solver_config.validate()
        require(self.model.boundary_mode == self.boundary.mode, 'model/boundary mode conflict')
        require(self.model.surface_mode == self.solver_config.surface_mode, 'model/solver surface mode conflict')
        require(not (self.model.initial_values or self.model.parameter_bounds or self.model.free_parameters or self.model.fixed_parameters),
                'simulation uses given_parameters; Fit policies/initial estimates must be empty')
        require(isinstance(self.given_parameters, Mapping) and set(self.given_parameters) == set(desc.parameter_names),
                'given_parameters must contain exactly every declared parameter, including amplitude/baseline')
        params = {}
        for p in desc.parameters:
            value = number(self.given_parameters[p.name], p.name)
            p._validate_value(value, p.name)
            params[p.name] = value
        object.__setattr__(self, 'given_parameters', MappingProxyType(params))
        object.__setattr__(self, 'time_s', time_vector(self.time_s))
        require(self.schema_version == 1 or len(self.time_s) <= SIMULATION_POINT_BUDGET,
                f'simulation scientific point budget is {SIMULATION_POINT_BUDGET}; no truncation')
        validate_pair(self.signal_type, self.signal_unit)
        text(self.job_id, 'job_id')
        require(self.sampling_kind in {'explicit', 'linear', 'log_with_zero'}, 'unknown sampling kind')
        if self.sampling_kind == 'linear':
            h = (self.time_s[-1]-self.time_s[0])/(len(self.time_s)-1)
            close_vectors(self.time_s, tuple(self.time_s[0]+i*h for i in range(len(self.time_s))), 'declared linear sampling')
        elif self.sampling_kind == 'log_with_zero':
            require(len(self.time_s) >= 3 and self.time_s[0] == 0, 'log_with_zero requires zero plus >=2 positive times')
            logs = tuple(log10(t) for t in self.time_s[1:])
            h = (logs[-1]-logs[0])/(len(logs)-1)
            close_vectors(logs, tuple(logs[0]+i*h for i in range(len(logs))), 'declared positive log sampling')
        require(self.parameter_origin in {'manual', 'analysis_config', 'fit_estimate'}, 'unknown given-parameter origin')
        require(self.copied_from is None or isinstance(self.copied_from, SourceIdentity), 'copied_from must be SourceIdentity')
        require(self.parameter_origin == 'manual' or self.copied_from is not None, 'copied parameters need parent identity')
        if self.parameter_origin == 'fit_estimate':
            require(self.copied_from.fit_id is not None, 'copied Fit estimates need an actual parent Fit identity')
        object.__setattr__(self, 'noise_sd', number(self.noise_sd, 'noise_sd', nonnegative=True))
        if self.noise_sd == 0:
            require(self.seed is None and self.rng_algorithm is None, 'noise-free simulation must not claim an RNG run')
        else:
            object.__setattr__(self, 'seed', integer(self.seed, 'seed'))
            require(self.schema_version == 1 or self.seed < 2**128, 'seed exceeds explicit 128-bit entropy budget')
            require(self.rng_algorithm == 'PCG64', 'noise simulation requires explicit PCG64 and actual seed')
        if self.boundary.mode == 'measured_history':
            require(self.boundary.measured_time_s[0] <= self.time_s[0]
                    and self.boundary.measured_time_s[-1] >= self.time_s[-1], 'measured boundary does not cover scientific time grid; no extrapolation')
        if self.input_units is not None:
            require(isinstance(self.input_units,SimulationUnitInputs),'Typed Simulation input unit evidence required')
            self.input_units.validate_request(self)

    def to_dict(self):
        result = Contract.to_dict(self)
        if self.schema_version == 1:
            result.pop('name'); result.pop('sequence_number')
        if self.schema_version<3:result.pop('input_units')
        return result

    @classmethod
    def from_dict(cls, value):
        return _read_versioned(cls, value, {1, 2, 3}, lambda f, v:
            (v>=2 or f not in {'name','sequence_number'}) and (v>=3 or f!='input_units'))

    def as_canonical(self):
        """Original v2 request identity, excluding the distinct input declaration."""
        if self.input_units is None:return self
        from dataclasses import replace
        return replace(self,schema_version=2,input_units=None)


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationResult(Contract):
    KIND: ClassVar[str] = 'simulation_result'
    request: SimulationRequest
    state: ResultState
    signal_clean: tuple[float, ...] = ()
    signal_observed: tuple[float, ...] = ()
    noise: tuple[float, ...] = ()
    contributions: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    numerical_evidence: Mapping[str, object] | None = None
    observation_evidence: Mapping[str, object] | None = None

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version', minimum=1) in {1, 2, 3, 4},
                'unsupported simulation_result schema_version')
        require(self.schema_version == 4 or self.observation_evidence is None,
                'legacy simulation cannot claim observation evidence')
        require(self.schema_version != 1 or self.numerical_evidence is None,
                'legacy simulation v1 cannot claim newly captured numerical evidence')
        require(isinstance(self.request, SimulationRequest) and isinstance(self.state, ResultState), 'request/state types required')
        for key in ('signal_clean', 'signal_observed', 'noise'):
            object.__setattr__(self, key, vector(getattr(self, key), key))
        supplied = self.contributions
        require(isinstance(supplied, Mapping), 'contributions must be a mapping')
        parts = {text(k, 'component key'): vector(v, 'contribution') for k, v in supplied.items()}
        object.__setattr__(self, 'contributions', MappingProxyType(parts))
        if self.numerical_evidence is not None:
            require(isinstance(self.numerical_evidence, Mapping), 'numerical evidence must be an object')
            object.__setattr__(self, 'numerical_evidence', _freeze_json(self.numerical_evidence, path='numerical_evidence'))
        if self.observation_evidence is not None:
            require(isinstance(self.observation_evidence, Mapping), 'observation evidence must be an object')
            object.__setattr__(self, 'observation_evidence', _freeze_json(self.observation_evidence, path='observation_evidence'))
        if not self.state.has_payload:
            require(not (self.signal_clean or self.signal_observed or self.noise or self.contributions), 'unavailable simulation must not fabricate arrays')
            require(self.numerical_evidence is None, 'unavailable simulation cannot claim numerical execution evidence')
            require(self.observation_evidence is None, 'unavailable simulation cannot claim an RNG execution')
            return
        n = len(self.request.time_s)
        require(all(len(v) == n for v in (self.signal_clean, self.signal_observed, self.noise)), 'simulation arrays must cover every requested time')
        mid = self.request.model.model_id.value
        expected = {'component_H', 'component_O'} if mid in {'M2_DUAL_HO_D', 'M3_DUAL_HO_DK'} else (
            {'component_1', 'component_2'} if mid == 'M8_DUAL_K' else {'component_1'})
        require(set(parts) == expected and all(len(v) == n for v in parts.values()), 'simulation contribution slots/lengths conflict with model')
        close_vectors(self.signal_observed, tuple(c+e for c, e in zip(self.signal_clean, self.noise)), 'clean/noise/observed')
        if self.request.noise_sd == 0:
            require(all(v == 0 for v in self.noise), 'noise-free request has nonzero noise')
        baseline = self.request.given_parameters['y0' if mid in {'M2_DUAL_HO_D', 'M3_DUAL_HO_DK', 'M8_DUAL_K'} else 'baseline']
        scale = max(1., abs(baseline), *(abs(v) for part in parts.values() for v in part))
        close_vectors(self.signal_clean, tuple(baseline+sum(part[i] for part in parts.values()) for i in range(n)),
                      'single common baseline + signed contributions', atol=1e-10*scale, rtol=0)
        if self.schema_version == 2:
            self._validate_single_evidence()
        elif self.schema_version in {3, 4}:
            # The reader verifies identities and algebra, never reruns a kernel.
            from .simulation_validation import validate_simulation_evidence
            validate_simulation_evidence(self)
            if self.schema_version == 4:
                from .simulation_observation import validate_observation_evidence
                validate_observation_evidence(self)

    def _validate_single_evidence(self):
        r, e = self.request, self.numerical_evidence
        require(r.model.model_id in {ModelID.M0_SINGLE_D, ModelID.M1_SINGLE_DK}
                and r.boundary.mode == 'ideal_step' and r.noise_sd == 0,
                'simulation result v2 covers only the validated P11 noiseless single ideal-step adapter')
        exact_fields(e, SINGLE_SIMULATION_EVIDENCE_FIELDS, 'simulation numerical evidence')
        require(e['adapter_version'] == SINGLE_SIMULATION_ADAPTER_VERSION and e['request_sha256'] == r.content_sha256,
                'simulation adapter/request identity conflict')
        require(e['source_kind'] == 'simulation_clean' and e['time_unit'] == 's'
                and e['signal_unit'] == r.signal_unit and e['length_unit'] == 'cm', 'simulation source/units conflict')
        desc = DEFAULT_MODEL_REGISTRY.get(r.model.model_id)
        units = {p.name: r.signal_unit if p.unit == 'signal' else p.unit for p in desc.parameters}
        require(e['parameter_units'] == units, 'given-parameter units conflict')
        require(e['geometry_convention'] == 'complete_dimensions_cm'
                and e['active_face_convention'] == 'both_opposite_faces_per_active_axis'
                and e['signal_convention'] == 'baseline_plus_signed_completion', 'simulation geometry/signal convention conflict')
        require(e['term_tol'] == r.solver_config.relative_tolerance
                and e['absolute_tolerance_usage'] == 'recorded_not_used_by_single_series_kernel', 'solver-control mapping conflict')
        from pyecr.models.root_cache import ROOT_SOLVER_POLICY
        require(e['root_mode'] == 'legacy_full_roots' and e['root_solver_policy'] == ROOT_SOLVER_POLICY,
                'unknown simulation root policy')
        require(e['interpretation'] == SINGLE_SIMULATION_INTERPRETATION, 'missing forward-only interpretation')
        require(e['t_zero_rule'] == 'production_analytic_initial_condition; series_bound_not_applicable_at_zero', 'unknown t=0 convention')
        require(isinstance(e['nonfinite_diagnostic_paths'], tuple)
                and all(isinstance(p, str) and p for p in e['nonfinite_diagnostic_paths']), 'nonfinite diagnostic paths required')
        exact_fields(e['evaluated_parameters'], r.given_parameters, 'evaluated physical parameters')
        for name, v in e['evaluated_parameters'].items():
            close_vectors((number(v, name),), (r.given_parameters[name],), 'physical/log10 adapter '+name, atol=0, rtol=2e-12)
        exact_fields(e['software_versions'], ('python', 'numpy', 'scipy', 'pyecr'), 'simulation software versions')
        for k, v in e['software_versions'].items():text(v, k)
        n = len(r.time_s)
        F = vector(e['remaining_fraction'], 'remaining fraction')
        p = vector(e['completion_fraction'], 'completion fraction')
        require(len(F) == len(p) == n, 'simulation evidence must cover every requested time')
        close_vectors(p, tuple(1.-f for f in F), 'remaining/completion identity', atol=0, rtol=0)
        a = r.given_parameters['amplitude']; b = r.given_parameters['baseline']
        close_vectors(self.contributions['component_1'], tuple(a*v for v in p), 'signed component identity', atol=0, rtol=0)
        close_vectors(self.signal_clean, tuple(b+c for c in self.contributions['component_1']), 'single baseline identity', atol=0, rtol=0)
        axes = e['axis_results']
        require(isinstance(axes, tuple) and len(axes) == 3, 'three ordered axis results required')
        active = []
        for dim, flag, axis in zip(r.model.geometry.full_dims_cm, r.model.geometry.active_dimensions, axes):
            require((axis is not None) == flag, 'inactive/active axis evidence conflict')
            if axis is None:continue
            exact_fields(axis, ('remaining_fraction', 'n_terms_used', 'coefficient_sum', 'diagnostics'), 'axis evidence')
            values = vector(axis['remaining_fraction'], 'axis remaining fraction')
            require(len(values) == n, 'axis result omits requested times')
            terms = integer(axis['n_terms_used'], 'n_terms_used', minimum=1)
            require(terms <= r.solver_config.max_terms, 'axis exceeded numerical term budget')
            number(axis['coefficient_sum'], 'coefficient_sum')
            d = axis['diagnostics']
            require(d['full_dim_cm'] == dim and d['half_dim_cm'] == dim/2
                    and d['max_terms'] == r.solver_config.max_terms and d['term_tol'] == e['term_tol']
                    and d['normalization_applied'] is False, 'axis dimensions/numerics conflict')
            if r.model.model_id is ModelID.M1_SINGLE_DK:
                expected = e['evaluated_parameters']['k_cm_s']*(dim/2)/e['evaluated_parameters']['D_cm2_s']
                close_vectors((d['bi'],), (expected,), 'two-face half-dimension Bi', atol=0, rtol=1e-14)
            require(isinstance(d['converged'], bool), 'axis convergence must be captured boolean')
            active.append(values)
        close_vectors(F, tuple(prod(axis[i] for axis in active) for i in range(n)), 'rectangular axis product', atol=2e-15, rtol=2e-15)
        k = e['kernel_diagnostics']
        require(tuple(k['full_dims_cm']) == r.model.geometry.full_dims_cm
                and tuple(k['active_dimensions']) == r.model.geometry.active_dimensions
                and k['all_axes_converged'] == all(a['diagnostics']['converged'] for a in axes if a is not None)
                and k['normalization_applied'] is False, 'rectangular numerical evidence conflict')
        codes = e['numerical_warning_codes']
        require(isinstance(codes, tuple) and all(isinstance(c,str) and c for c in codes)
                and len(set(codes)) == len(codes), 'invalid numerical warning codes')
        for code in codes:
            require(any(w.startswith(code+':') for w in self.state.warnings), 'numerical warning was hidden')
        if not k['all_axes_converged']:
            require('SERIES_TRUNCATION_UNRESOLVED' in codes, 'unconverged series must remain a warning')

    def to_dict(self):
        result = Contract.to_dict(self)
        if self.schema_version == 1:result.pop('numerical_evidence')
        if self.schema_version < 4:result.pop('observation_evidence')
        return result

    @classmethod
    def from_dict(cls, value):
        return _read_versioned(cls, value, {1, 2, 3, 4}, lambda f, v:
            (v >= 2 or f != 'numerical_evidence') and (v >= 4 or f != 'observation_evidence'))

    @property
    def time_s(self):
        return self.request.time_s

    @property
    def request_sha256(self):
        return self.request.content_sha256


def _read_versioned(cls, value, versions, include):
    require(isinstance(value, Mapping) and 'schema_version' in value, f'{cls.KIND} missing schema_version')
    version = integer(value['schema_version'], 'schema_version', minimum=1)
    require(version in versions, f'unsupported {cls.KIND} schema_version')
    names = [f.name for f in fields(cls) if include(f.name, version)]
    exact_fields(value, ['contract_kind', *names], cls.__name__)
    require(value['contract_kind'] == cls.KIND, f'expected contract_kind={cls.KIND}')
    hints = get_type_hints(cls)
    try:
        return cls(**{name: decode(hints[name], value[name]) for name in names})
    except (ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, ContractError):raise
        raise ContractError(f'invalid {cls.KIND}: {exc}') from exc
