"""Terminal-reference normalization records, not a normalization implementation."""
from dataclasses import dataclass
from math import isclose
from typing import ClassVar
from pyecr.units import validate_pair
from .common import Contract, ResultState, SourceIdentity, number, integer, rows, vector, text, require


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedNormalizationSpec(Contract):
    KIND: ClassVar[str] = 'signed_normalization_spec'
    identity: SourceIdentity
    raw_count: int
    raw_indices: tuple[int, ...]
    signal_type: str
    signal_unit: str
    reference_kind: str
    reference_rows: tuple[int, ...] = ()
    reference_value: float | None = None
    reference_evidence: str = ''
    t0_s: float = 0.0
    profile_id: str = 'signed_dct_final_maxabs_v1'
    scale_kind: str = 'maxabs_final_residual'
    algorithm_version: str = 'signed-final-maxabs/1'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.identity, SourceIdentity), 'identity must be SourceIdentity')
        count = integer(self.raw_count, 'raw_count', minimum=2)
        object.__setattr__(self, 'raw_count', count)
        object.__setattr__(self, 'raw_indices', rows(self.raw_indices, count))
        object.__setattr__(self, 'reference_rows', rows(self.reference_rows, count, 'reference_rows'))
        require(len(self.raw_indices) >= 2, 'normalization requires complete eligible rows')
        require(set(self.reference_rows) <= set(self.raw_indices), 'reference rows must belong to eligible raw rows')
        validate_pair(self.signal_type, self.signal_unit)
        require(self.profile_id == 'signed_dct_final_maxabs_v1' and self.scale_kind == 'maxabs_final_residual',
                'unsupported normalization mode/scale; no initial-reference or quantile fallback')
        require(self.algorithm_version == 'signed-final-maxabs/1', 'unknown normalization algorithm_version')
        require(self.reference_kind in {'accepted_final_window', 'external_known_terminal', 'parent_fit_asymptote'},
                'unknown terminal reference kind')
        text(self.reference_evidence, 'reference_evidence')
        if self.reference_kind == 'accepted_final_window':
            require(bool(self.reference_rows) and self.reference_value is None,
                    'window spec needs raw rows and defers its mean to P02')
        else:
            require(not self.reference_rows and self.reference_value is not None,
                    'declared reference needs an explicit value and no implicit window')
        if self.reference_kind == 'parent_fit_asymptote':
            self.identity.require_parent()
        if self.reference_value is not None:
            object.__setattr__(self, 'reference_value', number(self.reference_value, 'reference_value'))
        object.__setattr__(self, 't0_s', number(self.t0_s, 't0_s'))


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedNormalizationResult(Contract):
    KIND: ClassVar[str] = 'signed_normalization_result'
    spec: SignedNormalizationSpec
    state: ResultState
    output_identity: SourceIdentity | None = None
    normalized_signal: tuple[float, ...] = ()
    reference_value: float | None = None
    scale_value: float | None = None
    scale_raw_index: int | None = None
    output_unit: str = '1'

    def __post_init__(self):
        Contract.__post_init__(self)
        require(isinstance(self.spec, SignedNormalizationSpec) and isinstance(self.state, ResultState), 'spec/state types required')
        object.__setattr__(self, 'normalized_signal', vector(self.normalized_signal, 'normalized_signal'))
        require(self.output_unit == '1', 'normalization output_unit must be 1')
        if not self.state.has_payload:
            require(not self.normalized_signal and all(v is None for v in
                    (self.reference_value, self.scale_value, self.scale_raw_index, self.output_identity)), 'unavailable normalization must not fabricate payload')
            return
        if self.output_identity is not None:
            require(isinstance(self.output_identity, SourceIdentity), 'output_identity must be SourceIdentity')
            require(self.output_identity.fit_id is None, 'applied normalization invalidates current Fit/Diagnostics identity')
            require(self.output_identity.dataset_sha256 != self.spec.identity.dataset_sha256
                    and self.output_identity.recipe_sha256 != self.spec.identity.recipe_sha256,
                    'applied normalization must record new dataset and recipe identities')
        require(len(self.normalized_signal) == len(self.spec.raw_indices), 'normalization raw row mapping mismatch')
        object.__setattr__(self, 'reference_value', number(self.reference_value, 'reference_value'))
        object.__setattr__(self, 'scale_value', number(self.scale_value, 'scale_value', positive=True))
        require(integer(self.scale_raw_index, 'scale_raw_index') in self.spec.raw_indices, 'maxabs index not in eligible raw rows')
        require(self.spec.reference_value is None or self.reference_value == self.spec.reference_value, 'declared terminal reference conflict')
        require(isclose(max(abs(v) for v in self.normalized_signal), 1.0, rel_tol=0, abs_tol=1e-14), 'normalized maxabs must be 1')
        require(isclose(abs(self.normalized_signal[self.spec.raw_indices.index(self.scale_raw_index)]), 1.0, rel_tol=0, abs_tol=1e-14),
                'scale_raw_index must identify an actual maximum residual')
