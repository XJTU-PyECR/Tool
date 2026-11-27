"""P08 observation sources only. No component normalization or DCT result."""
from dataclasses import dataclass,fields
from typing import ClassVar,get_type_hints

from .common import (Contract, ResultState, SourceIdentity, close_vectors, digest,
                     integer, number, require, strings, vector)
from .dct import DctSourceSnapshot, SourceKind
from .simulation_parent import SimulationParentEvidence,CHANNEL_KINDS
from .eligibility import FeatureEligibility


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentSourceCapture(Contract):
    KIND: ClassVar[str] = 'component_source_capture'
    state: ResultState
    identity: SourceIdentity | None = None
    parent_source: DctSourceSnapshot | None = None
    components: tuple[DctSourceSnapshot, ...] = ()
    labels: tuple[str, ...] = ()
    label_semantics: str | None = None
    baseline: float | None = None
    signed_amplitudes: tuple[float, ...] = ()
    directions: tuple[str, ...] = ()
    parent_step_type: str | None = None
    fit_total: tuple[float, ...] = ()
    asymptotic_total: float | None = None
    mask: tuple[bool, ...] = ()
    raw_sha256: str | None = None
    reported_to_input: tuple[int, ...] = ()
    input_view_mode: str | None = None
    observation_parity_max_abs: float | None = None
    revision: tuple[int, ...] = ()
    fit_generation: int | None = None
    decomposition_version: str = 'continuous-decomposition/1.0.0'
    extraction_version: str = 'component-observation-source/1'
    simulation_parent: SimulationParentEvidence | None = None
    eligibility: FeatureEligibility | None = None

    def __post_init__(self):
        version=integer(self.schema_version,'schema_version',minimum=1)
        require(version in {1,2,3},'unsupported component_source_capture schema_version')
        require((version==3)==(self.eligibility is not None),'component capture eligibility/version conflict')
        require(isinstance(self.state, ResultState), 'component capture state required')
        require(self.decomposition_version == 'continuous-decomposition/1.0.0'
                and self.extraction_version == ('component-observation-source/1' if version==1 else
                    'component-observation-source/2-simulation' if version==2 else 'component-observation-source/3-capability'), 'unknown extraction version')
        require(version!=1 or self.simulation_parent is None,'v1 cannot claim simulated parent evidence')
        require(self.identity is None or isinstance(self.identity, SourceIdentity), 'source identity required')
        for name in ('components', 'mask', 'reported_to_input', 'revision'):
            require(isinstance(getattr(self, name), (tuple, list)), f'{name} requires an array')
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('labels', 'directions'):
            object.__setattr__(self, name, strings(getattr(self, name), name))
        for name in ('signed_amplitudes', 'fit_total'):
            object.__setattr__(self, name, vector(getattr(self, name), name))
        if self.parent_source is None:
            require(version==1 and self.simulation_parent is None,'Unavailable capture cannot claim simulation evidence')
            require(not self.state.ready and not any((self.components, self.labels, self.signed_amplitudes,
                    self.directions, self.fit_total, self.mask, self.reported_to_input, self.revision))
                    and all(getattr(self, n) is None for n in ('label_semantics', 'baseline', 'parent_step_type',
                        'asymptotic_total', 'raw_sha256', 'input_view_mode', 'observation_parity_max_abs', 'fit_generation')),
                    'unavailable capture must not fabricate component payload')
            return
        require(self.state.has_payload, 'unavailable status cannot carry components')
        p = self.parent_source
        require(isinstance(p,DctSourceSnapshot) and p.normalization is None,'Parent must be unnormalized processed observations')
        if self.simulation_parent is None:
            require(version!=2,'v2 requires simulation evidence')
            require(p.source_kind is SourceKind.EXPERIMENT,
                    'parent must be original processed experiment, without DCT normalization')
        else:
            e=self.simulation_parent
            require(isinstance(e,SimulationParentEvidence),'Simulation parent requires full Generate/Send evidence')
            require(p.source_kind.value==CHANNEL_KINDS[e.send.channel]
                and p.simulation_request_sha256==e.result.request_sha256,'Simulation capture source/channel conflict')
            require(p.raw_count==len(e.result.time_s) and self.raw_sha256==e.send.raw_sha256,
                    'Simulation capture must bind all original generated rows')
            close_vectors(p.time_s,tuple(e.result.time_s[i]-p.t0_s for i in p.raw_indices),
                          'Simulation capture observation time/origin',atol=1e-12,rtol=1e-12)
        require(isinstance(self.identity, SourceIdentity), 'parent identity required')
        self.identity.require_parent(); self.identity.assert_matches(p.identity)
        if self.eligibility is not None:
            require(isinstance(self.eligibility,FeatureEligibility),'typed component eligibility required')
            self.eligibility.require_component(self.identity,self.eligibility.parent_reliability)
        require(len(self.components) == len(self.labels) == len(self.signed_amplitudes) == len(self.directions) == 2,
                'two complete component slots required')
        m8 = self.identity.model_id == 'M8_DUAL_K'
        keys = ('component_1', 'component_2') if m8 else ('component_H', 'component_O')
        require(self.identity.model_id in {'M2_DUAL_HO_D', 'M3_DUAL_HO_DK', 'M8_DUAL_K'}, 'unsupported parent model')
        require(self.label_semantics == ('effective_modes_not_species' if m8 else 'declared_H_O_slots_not_species_proof'),
                'component roles cannot assert chemical attribution')
        require(self.parent_step_type in {'hydration', 'dehydration', 'custom'}, 'parent direction declaration required')
        require(self.input_view_mode in {'zero', 'shifted'}, 'known decomposition view required')
        require(len(self.revision) == 3, 'full workflow revision required')
        for r in self.revision: integer(r, 'revision')
        integer(self.fit_generation, 'fit_generation', minimum=1)
        for i in self.reported_to_input: integer(i, 'reported_to_input')
        require(sorted(self.reported_to_input) == [0, 1], 'reported/input component permutation required')
        digest(self.raw_sha256, 'raw_sha256')
        number(self.observation_parity_max_abs, 'observation_parity_max_abs', nonnegative=True)
        object.__setattr__(self, 'baseline', number(self.baseline, 'baseline'))
        object.__setattr__(self, 'asymptotic_total', number(self.asymptotic_total, 'asymptotic_total'))
        require(len(self.mask) == p.raw_count and all(isinstance(v, bool) for v in self.mask)
                and tuple(i for i, v in enumerate(self.mask) if v) == p.raw_indices, 'complete raw mask/mapping conflict')
        for c, key, a, direction in zip(self.components, keys, self.signed_amplitudes, self.directions):
            require(isinstance(c, DctSourceSnapshot), 'component snapshot required')
            require(c.eligibility==self.eligibility,'component/capture eligibility conflict')
            self.identity.assert_matches(c.identity)
            require(c.source_kind is SourceKind.FIT_COMPONENT and c.component_key == key
                    and c.parent_source_sha256 == p.content_sha256, 'component/parent binding conflict')
            require(c.time_s == p.time_s and c.t0_s == p.t0_s and c.raw_indices == p.raw_indices
                    and c.raw_count == p.raw_count and c.signal_unit == p.signal_unit and c.signal_type == p.signal_type,
                    'component must retain every parent observation, origin, row and unit')
            require(c.normalization is None and c.representation == ('physical' if p.signal_type != 'normalized' else 'unqualified_normalized')
                    and c.component_initial_value == 0 and c.component_terminal_value == a,
                    'component must be unnormalized zero-baseline signed contribution')
            require(a != 0 and direction == ('increase' if a > 0 else 'decrease'), 'zero/direction/amplitude conflict')
            if self.state.ready: c.require_usable()
        close_vectors(self.fit_total, tuple(self.baseline+a+b for a,b in zip(
            self.components[0].signal, self.components[1].signal)), 'all-observation reconstruction', atol=2e-12, rtol=2e-12)
        close_vectors((self.asymptotic_total,), (self.baseline+sum(self.signed_amplitudes),), 'analytic terminal')

    @property
    def B_j(self):
        """Terminal-deviation coefficients, opposite to progress amplitudes."""
        return tuple(-a for a in self.signed_amplitudes)

    def require_usable(self):
        require(self.state.ready and self.parent_source is not None, '; '.join(self.state.blocking_reasons) or 'source unavailable')
        for c in self.components: c.require_usable()

    def to_dict(self):
        value=Contract.to_dict(self)
        if self.schema_version==1:value.pop('simulation_parent')
        if self.schema_version<3:value.pop('eligibility')
        return value

    @classmethod
    def from_dict(cls,value):
        from .common import exact_fields,decode
        from collections.abc import Mapping
        require(isinstance(value,Mapping) and 'schema_version' in value,'Component capture schema_version required')
        version=integer(value['schema_version'],'schema_version',minimum=1)
        require(version in {1,2,3},'unsupported component_source_capture schema_version')
        names=[f.name for f in fields(cls) if (version>1 or f.name!='simulation_parent') and (version==3 or f.name!='eligibility')]
        exact_fields(value,['contract_kind',*names],cls.__name__)
        require(value['contract_kind']==cls.KIND,'Component capture kind conflict')
        hints=get_type_hints(cls)
        return cls(**{n:decode(hints[n],value[n]) for n in names})
