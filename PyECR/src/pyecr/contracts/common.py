"""Versioned immutable payloads. No fitting, normalization or forward evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from hashlib import sha256
import json
from math import isclose, isfinite
from numbers import Integral, Real
from types import UnionType
from typing import ClassVar, Union, get_args, get_origin, get_type_hints

from pyecr.config import BoundaryConfig, ModelConfig, SampleGeometry, SolverConfig
from pyecr.dct.core import CandidateTimescale
from pyecr.models.ids import ModelID
from pyecr.dct.config import DctAnalysisConfig


class ContractError(ValueError):
    """Malformed, incomplete, unknown-version or unsupported contract."""


class IdentityConflictError(ContractError):
    """Two payloads claim incompatible dataset/config/recipe/Fit identities."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def number(value, name, *, positive=False, nonnegative=False):
    require(not isinstance(value, bool) and isinstance(value, Real), f"{name} must be a real number")
    value = float(value)
    require(isfinite(value), f"{name} must be finite")
    require(not positive or value > 0, f"{name} must be positive")
    require(not nonnegative or value >= 0, f"{name} must be nonnegative")
    return value


def integer(value, name, *, minimum=0):
    require(not isinstance(value, bool) and isinstance(value, Integral) and value >= minimum,
            f"{name} must be an integer >= {minimum}")
    return int(value)


def vector(values, name, *, positive=False):
    require(not isinstance(values, (str, bytes, Mapping)), f"{name} must be a vector")
    try:
        return tuple(number(v, name, positive=positive) for v in values)
    except TypeError as exc:
        raise ContractError(f"{name} must be a vector") from exc


def strings(values, name):
    require(isinstance(values, (tuple, list)), f"{name} must be an array")
    return tuple(text(v, name) for v in values)


def rows(values, raw_count, name="raw_indices"):
    require(isinstance(values, (tuple, list)), f"{name} must be an array")
    result = tuple(integer(v, name) for v in values)
    require(all(a < b for a, b in zip(result, result[1:])), f"{name} must be unique and increasing")
    require(all(v < raw_count for v in result), f"{name} exceeds original raw row count")
    return result


def time_vector(values):
    result = vector(values, "time_s")
    require(len(result) >= 2 and result[0] >= 0 and all(a < b for a, b in zip(result, result[1:])),
            "time_s requires >=2 strictly increasing nonnegative observations")
    return result


def digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            f"{name} must be a lowercase SHA256 digest")
    return value


def close_vectors(a, b, name, *, atol=1e-12, rtol=1e-10):
    require(len(a) == len(b) and all(isclose(x, y, abs_tol=atol, rel_tol=rtol) for x, y in zip(a, b)),
            f"{name} mismatch")


def encode(value):
    if isinstance(value, Contract):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, 'to_dict'):
        return encode(value.to_dict())
    if isinstance(value, Mapping):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return number(value, 'serialized number')
    return value


def canonical_sha256(value):
    """Strict finite JSON content identity, matching snapshot/config conventions."""
    return sha256(json.dumps(encode(value), sort_keys=True, ensure_ascii=False,
                            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def exact_fields(payload, names, name):
    require(isinstance(payload, Mapping), f"{name} must be an object")
    missing = set(names) - set(payload)
    extra = set(payload) - set(names)
    require(not missing, f"{name} missing fields: {sorted(missing)}")
    require(not extra, f"{name} unknown fields: {sorted(extra)}")


def strict_json_loads(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON field: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ContractError(f'nonfinite JSON constant: {value}')

    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f'invalid JSON: {exc}') from exc


FOREIGN_TYPES = (SampleGeometry, BoundaryConfig, ModelConfig, SolverConfig, DctAnalysisConfig, CandidateTimescale)


def read_foreign(cls, value):
    """Reuse legacy value types, without their permissive missing-field defaults."""
    names = [f.name for f in fields(cls)]
    # The optional search-grid extension is omitted by historical/default DCT
    # payloads. All original required fields and unknown-field checks stay strict.
    if cls is DctAnalysisConfig and isinstance(value, Mapping) and 'gcv_lambdas' not in value:
        names.remove('gcv_lambdas')
    exact_fields(value, names, cls.__name__)
    args = dict(value)
    if cls is DctAnalysisConfig and 'gcv_lambdas' in args:
        args['gcv_lambdas'] = tuple(args['gcv_lambdas'])
    if cls is ModelConfig:
        args['geometry'] = read_foreign(SampleGeometry, args['geometry'])
        args['model_id'] = ModelID(args['model_id'])
    result = cls(**args)
    if hasattr(result, 'validate'):
        result.validate()
    return result


def decode(annotation, value):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        alternatives = [t for t in args if t is not type(None)]
        require(len(alternatives) == 1, "ambiguous union in contract")
        return decode(alternatives[0], value)
    if origin is tuple:
        require(isinstance(value, (list, tuple)), "tuple field must be an array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], v) for v in value)
        require(len(args) == len(value), "fixed tuple length mismatch")
        return tuple(decode(t, v) for t, v in zip(args, value))
    if origin is Mapping:
        require(isinstance(value, Mapping), "mapping field must be an object")
        return {k: decode(args[1], v) for k, v in value.items()}
    if isinstance(annotation, type) and issubclass(annotation, Contract):
        return annotation.from_dict(value)
    if annotation in FOREIGN_TYPES:
        return read_foreign(annotation, value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    return value  # Scalar constraints are enforced in __post_init__, without coercion.


@dataclass(frozen=True, slots=True, kw_only=True)
class Contract:
    schema_version: int = 1
    KIND: ClassVar[str]

    def __post_init__(self):
        require(integer(self.schema_version, 'schema_version', minimum=1) == 1,
                f"unsupported {self.KIND} schema_version")
        object.__setattr__(self, 'schema_version', int(self.schema_version))

    def to_dict(self):
        return {'contract_kind': self.KIND, **{f.name: encode(getattr(self, f.name)) for f in fields(self)}}

    def to_json(self, *, indent=None):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False, indent=indent)

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(strict_json_loads(payload))

    @classmethod
    def from_dict(cls, value):
        exact_fields(value, ['contract_kind', *[f.name for f in fields(cls)]], cls.__name__)
        require(value['contract_kind'] == cls.KIND, f"expected contract_kind={cls.KIND}")
        require(integer(value['schema_version'], 'schema_version', minimum=1) == 1,
                f"unsupported {cls.KIND} schema_version")
        hints = get_type_hints(cls)
        try:
            return cls(**{f.name: decode(hints[f.name], value[f.name]) for f in fields(cls)})
        except (ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid {cls.KIND}: {exc}") from exc

    @property
    def content_sha256(self):
        return canonical_sha256(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultState(Contract):
    KIND: ClassVar[str] = 'result_state'
    status: str = 'completed'
    warnings: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        Contract.__post_init__(self)
        require(self.status in {'completed', 'warning', 'stale', 'cancelled', 'error', 'unsupported'}, 'unknown result status')
        object.__setattr__(self, 'warnings', strings(self.warnings, 'warnings'))
        object.__setattr__(self, 'blocking_reasons', strings(self.blocking_reasons, 'blocking_reasons'))
        require(self.status != 'warning' or bool(self.warnings), 'warning status needs warnings')
        require(self.status != 'completed' or not self.warnings, 'completed status must not hide warnings')
        require(not self.ready or not self.blocking_reasons, 'ready status conflicts with blocking reasons')
        require(self.ready or bool(self.blocking_reasons), 'unavailable/stale status needs explicit reasons')

    @property
    def ready(self):
        return self.status in {'completed', 'warning'}

    @property
    def has_payload(self):
        return self.status in {'completed', 'warning', 'stale'}


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceIdentity(Contract):
    KIND: ClassVar[str] = 'source_identity'
    dataset_sha256: str
    recipe_sha256: str
    config_sha256: str | None = None
    model_id: str | None = None
    fit_id: str | None = None
    fit_sha256: str | None = None
    diagnostics_sha256: str | None = None

    def __post_init__(self):
        Contract.__post_init__(self)
        for f in fields(self):
            if f.name not in {'schema_version', 'model_id'} and getattr(self, f.name) is not None:
                digest(getattr(self, f.name), f.name)
        require((self.config_sha256 is None) == (self.model_id is None), 'config identity and canonical model_id must be provided together')
        if self.model_id is not None:
            require(isinstance(self.model_id, str) and self.model_id in {m.value for m in ModelID}, 'unknown canonical model_id')
        require((self.fit_id is None) == (self.fit_sha256 is None), 'fit_id and fit_sha256 must be provided together')
        require(self.fit_id is None or self.config_sha256 is not None, 'Fit identity requires config identity')
        require(self.diagnostics_sha256 is None or self.fit_id is not None, 'Diagnostics identity requires Fit identity')

    def require_parent(self):
        require(all(getattr(self, k) is not None for k in ('config_sha256', 'fit_id', 'fit_sha256', 'diagnostics_sha256')),
                'component source requires config, Fit and Diagnostics identities')

    def assert_matches(self, other):
        if not isinstance(other, SourceIdentity) or self != other:
            raise IdentityConflictError('dataset/config/recipe/Fit/Diagnostics identity conflict')

    @classmethod
    def capture(cls, dataset, recipe, *, config=None, fit=None, diagnostics=None):
        from pyecr.report.snapshot import dataset_sha256, _sha256_json
        from pyecr.fitting.reproducibility import config_hash
        from pyecr.services.guided_refit import fit_result_id
        from pyecr.preprocessing import PreprocessConfig
        from pyecr.config import AnalysisConfig
        from pyecr.fitting.base import FitResult
        require(isinstance(recipe, PreprocessConfig), 'recipe must be the actual PreprocessConfig')
        require(config is None or isinstance(config, AnalysisConfig), 'config must be the actual AnalysisConfig')
        require(fit is None or isinstance(fit, FitResult), 'fit must be an actual FitResult')
        if diagnostics is not None and hasattr(diagnostics, 'to_dict'):
            diagnostics = diagnostics.to_dict()
        require(diagnostics is None or isinstance(diagnostics, Mapping), 'diagnostics must be captured structured evidence')
        return cls(dataset_sha256=dataset_sha256(dataset), recipe_sha256=canonical_sha256(recipe.to_dict()),
                   config_sha256=None if config is None else config_hash(config),
                   model_id=None if config is None else config.model_id.value,
                   fit_id=None if fit is None else fit_result_id(fit),
                   fit_sha256=None if fit is None else _sha256_json(fit.to_dict()),
                   diagnostics_sha256=None if diagnostics is None else _sha256_json(diagnostics))
