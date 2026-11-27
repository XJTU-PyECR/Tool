"""Numerical evidence for signed inversion; never a physical Fit reliability gate."""
from dataclasses import dataclass, fields
from math import isclose
from typing import ClassVar, get_type_hints

from .common import Contract, integer, number, require, vector


@dataclass(frozen=True, slots=True, kw_only=True)
class SignedDctSolveDiagnostics(Contract):
    KIND: ClassVar[str] = 'signed_dct_solve_diagnostics'
    solver_status: str
    lapack_driver: str
    augmented_shape: tuple[int, int]
    effective_rank: int
    svd_rank: int
    relative_cutoff: float
    singular_values: tuple[float, ...]
    condition_number: float | None
    effective_condition_number: float
    grid_step_decades: float
    regularizer_rows: int
    weighted_sse: float
    regularization_penalty: float
    objective: float
    residual_constant: float = 0.0
    effective_df: float | None = None
    factorization_shape: tuple[int, int] | None = None
    orthogonal_weighted_sse: float | None = None
    df_semantics: str | None = None

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version', minimum=1) in {1, 2, 3}, 'unknown signed solve evidence version')
        object.__setattr__(self, 'schema_version', int(self.schema_version))
        require(self.solver_status in {'converged', 'rank_deficient'}, 'unknown signed solve status')
        require((self.schema_version == 1 and self.lapack_driver in {'gelsd', 'gelsy'}) or
                (self.schema_version >= 2 and self.lapack_driver == 'gesdd'), 'unknown signed LAPACK driver/version')
        require(isinstance(self.augmented_shape, (tuple, list)) and len(self.augmented_shape) == 2,
                'augmented_shape must have two dimensions')
        object.__setattr__(self, 'augmented_shape', tuple(integer(v, 'augmented_shape', minimum=1) for v in self.augmented_shape))
        for key in ('effective_rank', 'svd_rank', 'regularizer_rows'):
            object.__setattr__(self, key, integer(getattr(self, key), key))
        m, n = self.augmented_shape
        require(1 <= self.effective_rank <= min(m, n) and 1 <= self.svd_rank <= min(m, n), 'invalid solve rank')
        require(self.regularizer_rows == n-2 if self.schema_version<3 else self.regularizer_rows>=1,
                'regularizer row count conflicts with numerical version')
        require((self.solver_status == 'rank_deficient') == (self.effective_rank < n), 'solve status/rank conflict')
        for key in ('relative_cutoff', 'effective_condition_number', 'grid_step_decades'):
            object.__setattr__(self, key, number(getattr(self, key), key, positive=True))
        require(self.relative_cutoff < 1, 'relative cutoff must be smaller than 1')
        object.__setattr__(self, 'singular_values', vector(self.singular_values, 'singular_values'))
        s = self.singular_values
        require(len(s) == min(m, n) and s[0] > 0 and s[-1] >= 0
                and all(a >= b for a, b in zip(s, s[1:])), 'invalid augmented singular values')
        require(self.svd_rank == sum(v > s[0]*self.relative_cutoff for v in s), 'SVD rank/cutoff conflict')
        require(isclose(self.effective_condition_number, s[0]/s[self.svd_rank-1], rel_tol=1e-12),
                'effective condition must use retained SVD singular values')
        if self.condition_number is not None:
            object.__setattr__(self, 'condition_number', number(self.condition_number, 'condition_number', positive=True))
            require(s[-1] > 0 and isclose(self.condition_number, s[0]/s[-1], rel_tol=1e-12), 'condition/singular values conflict')
        else:
            require(s[-1] == 0 or s[0]/s[-1] == float('inf'), 'finite full condition must be recorded')
        for key in ('weighted_sse', 'regularization_penalty', 'objective'):
            object.__setattr__(self, key, number(getattr(self, key), key, nonnegative=True))
        require(isclose(self.objective, self.weighted_sse+self.regularization_penalty, rel_tol=1e-12, abs_tol=1e-14),
                'objective/SSE/penalty conflict')
        require(number(self.residual_constant, 'residual_constant') == 0, 'signed v1 residual constant must be fixed zero')
        if self.schema_version == 1:
            require(self.effective_df is None and self.factorization_shape is None and self.orthogonal_weighted_sse is None,
                    'v1 evidence cannot claim v2 measurements')
        else:
            object.__setattr__(self, 'effective_df', number(self.effective_df, 'effective_df', nonnegative=True))
            require(self.effective_df <= m-self.regularizer_rows, 'df exceeds original observation count')
            require(isinstance(self.factorization_shape, (tuple, list)) and len(self.factorization_shape) == 2,
                    'factorization shape required')
            shape = tuple(integer(v, 'factorization shape', minimum=1) for v in self.factorization_shape)
            require(shape[1] == n and n <= shape[0] <= m, 'factorization shape conflict')
            object.__setattr__(self, 'factorization_shape', shape)
            object.__setattr__(self, 'orthogonal_weighted_sse', number(self.orthogonal_weighted_sse,
                'orthogonal_weighted_sse', nonnegative=True))
        require(self.df_semantics is None if self.schema_version<3 else self.df_semantics in {'fixed_linear','conditional_frozen_weights'},'unknown/misversioned df semantics')

    def to_dict(self):
        value = Contract.to_dict(self)
        if self.schema_version == 1:
            for key in ('effective_df', 'factorization_shape', 'orthogonal_weighted_sse'):
                value.pop(key)
        if self.schema_version<3:value.pop('df_semantics')
        return value

    @classmethod
    def from_dict(cls, value):
        from .common import exact_fields, decode
        require('schema_version' in value, 'signed solve evidence version required')
        version = integer(value['schema_version'], 'schema_version', minimum=1)
        require(version in {1, 2, 3}, 'unknown signed solve evidence version')
        new = {'effective_df', 'factorization_shape', 'orthogonal_weighted_sse'}
        names = [f.name for f in fields(cls) if (version>=2 or f.name not in new) and (version==3 or f.name!='df_semantics')]
        exact_fields(value, ['contract_kind', *names], cls.__name__)
        require(value['contract_kind'] == cls.KIND, 'signed solve evidence kind conflict')
        hints = get_type_hints(cls)
        return cls(**{name: decode(hints[name], value[name]) for name in names})
