"""Exact, task-local reuse of amplitude-free dual H/O kernel results.

The cache stores immutable bytes and reconstructs a fresh result on every hit.
It never stores a signed/amplitude-scaled component or a final prediction.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields, is_dataclass
from math import isfinite
from numbers import Integral, Real
from struct import pack
from sys import getsizeof
from threading import local
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import scipy

from pyecr.models import kernel as kernel_module
from pyecr.models import roots
from pyecr.models.kernel import KernelResult
from pyecr.models.root_cache import ROOT_SOLVER_POLICY, ROOT_PREFIX_POLICY, current_root_mode


COMPONENT_CACHE_POLICY = "dual-unit-rectangular-exact-v1"
DEFAULT_MAX_ENTRIES = 1024
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
ROOT_PLUS_COMPONENT_MAX_BYTES = 32 * 1024 * 1024
_worker = local()


@dataclass(frozen=True, slots=True)
class ComponentCacheDiagnostics:
    enabled: bool
    hits: int
    misses: int
    evictions: int
    entries: int
    estimated_bytes: int
    peak_entries: int
    peak_estimated_bytes: int
    max_entries: int
    max_bytes: int
    bypasses: int
    original_kernel_calls: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _f64(value: Real) -> bytes:
    return pack(">d", float(value))


def _array_key(original: object, validated: np.ndarray) -> tuple | None:
    """Retain both caller dtype/shape/content and the evaluated f64 vector."""
    try:
        raw = np.asarray(original)
    except (TypeError, ValueError):
        return None
    if raw.dtype.kind not in "biuf" or raw.ndim == 0:
        return None
    return (raw.dtype.str, tuple(raw.shape), raw.tobytes(order="C"),
            validated.dtype.str, tuple(validated.shape), validated.tobytes(order="C"))


def _options_key(options: Mapping[str, object]) -> tuple[bytes, int] | None:
    if set(options) - {"term_tol", "max_terms"}:
        return None
    tolerance = options.get("term_tol", 1.0e-10)
    count = options.get("max_terms", 512)
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not isfinite(float(tolerance)) or tolerance <= 0):
        return None
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        return None
    return _f64(tolerance), int(count)


def _freeze(value: object) -> tuple:
    """Detach all mutable arrays/mappings into an immutable payload tree."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return ("array", array.dtype.str, tuple(array.shape),
                array.tobytes(order="C"), bool(value.flags.writeable))
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(item) for item in value))
    if is_dataclass(value) and (
        type(value) is KernelResult or type(value).__module__ == "pyecr.models.rectangular"
        and type(value).__name__ == "RectangularKernelResult"
    ):
        return ("result", type(value),
                tuple((item.name, _freeze(getattr(value, item.name))) for item in fields(value)))
    if isinstance(value, (str, bytes, int, float, bool, np.generic)) or value is None:
        return ("scalar", value)
    raise TypeError(f"unsupported component result value: {type(value).__name__}")


def _thaw(payload: tuple) -> object:
    tag = payload[0]
    if tag == "array":
        _tag, dtype, shape, data, writable = payload
        result = np.frombuffer(data, dtype=np.dtype(dtype)).copy().reshape(shape)
        if not writable:
            result.setflags(write=False)
        return result
    if tag == "mapping":
        return MappingProxyType({key: _thaw(item) for key, item in payload[1]})
    if tag == "tuple":
        return tuple(_thaw(item) for item in payload[1])
    if tag == "result":
        return payload[1](**{name: _thaw(item) for name, item in payload[2]})
    if tag == "scalar":
        return payload[1]
    raise RuntimeError("invalid component cache payload")


def _estimated_size(value: object) -> int:
    if isinstance(value, tuple):
        return getsizeof(value) + sum(_estimated_size(item) for item in value)
    if isinstance(value, (bytes, str)):
        return getsizeof(value)
    return getsizeof(value)


class ComponentCacheScope:
    """A fresh synchronous-worker LRU, independent of scientific artifacts."""

    def __init__(self, *, enabled: bool = False,
                 max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 cancel_check: Callable[[], None] | None = None,
                 model_id: str = "unknown", boundary_mode: str = "unknown",
                 signal_type: str = "unknown", signal_unit: str = "unknown") -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if max_bytes > DEFAULT_MAX_BYTES:
            raise ValueError("component max_bytes exceeds the P58A 16 MiB budget")
        self.enabled = enabled
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.cancel_check = cancel_check or (lambda: None)
        self.context = (model_id, boundary_mode, signal_type, signal_unit)
        self._values: OrderedDict[tuple, tuple[tuple, int]] = OrderedDict()
        self._hits = self._misses = self._evictions = self._bytes = 0
        self._bypasses = self._kernel_calls = 0
        self._peak_entries = self._peak_bytes = 0
        self._entered = self._closed = False
        self._final: ComponentCacheDiagnostics | None = None

    @property
    def diagnostics(self) -> ComponentCacheDiagnostics:
        return self._final or ComponentCacheDiagnostics(
            self.enabled, self._hits, self._misses, self._evictions,
            len(self._values), self._bytes, self._peak_entries, self._peak_bytes,
            self.max_entries, self.max_bytes, self._bypasses, self._kernel_calls)

    def __enter__(self) -> ComponentCacheScope:
        if self._entered or self._closed:
            raise RuntimeError("component cache scopes are single-use")
        self._entered = True
        self._previous = getattr(_worker, "scope", None)
        _worker.scope = self
        return self

    def __exit__(self, *exc: object) -> None:
        _worker.scope = self._previous
        self._previous = None
        self._final = self.diagnostics
        self._values.clear()
        self._bytes = 0
        self._closed = True

    def evaluate(self, *, kernel: object, original_times: object,
                 times: np.ndarray, diffusion: float, surface: float | None,
                 options: Mapping[str, object], step_type: str,
                 sign_h: float, sign_o: float) -> object:
        if not self._entered or self._closed or getattr(_worker, "scope", None) is not self:
            raise RuntimeError("component cache used outside its owning worker scope")
        self.cancel_check()
        if not self.enabled:
            self._bypasses += 1
            self._kernel_calls += 1
            return kernel.evaluate(times, diffusion, surface, **options)
        # Keep this import lazy: config imports models.dual_ho while defining
        # SampleGeometry, and rectangular imports SampleGeometry from config.
        from pyecr.models import rectangular as rectangular_module
        from pyecr.models.rectangular import (
            DiffusionControlledRectangularKernel, RectangularKernel,
        )
        if type(kernel) not in (RectangularKernel, DiffusionControlledRectangularKernel):
            self._bypasses += 1
            self._kernel_calls += 1
            return kernel.evaluate(times, diffusion, surface, **options)
        time_key = _array_key(original_times, times)
        option_key = _options_key(options)
        if time_key is None or option_key is None:
            self._bypasses += 1
            self._kernel_calls += 1
            return kernel.evaluate(times, diffusion, surface, **options)

        key = (COMPONENT_CACHE_POLICY, ROOT_SOLVER_POLICY, self.context,
               type(kernel), type(kernel).evaluate,
               rectangular_module.finite_surface_fraction_1d,
               rectangular_module.diffusion_controlled_fraction_1d,
               kernel_module.finite_surface_fraction_1d,
               kernel_module.diffusion_controlled_fraction_1d,
               kernel_module._select_term_count,
               kernel_module._truncation_error_by_time,
               kernel_module._diffusion_truncation_error_by_time,
               kernel_module.resolve_alpha_roots,
               roots.solve_alpha_roots, roots._brentq_roots,
               roots._large_bi_roots, roots.brentq,
               _f64(roots.ASYMPTOTIC_BI_THRESHOLD),
               np.__version__, scipy.__version__, np.dtype("float64").str,
               time_key, tuple(_f64(value) for value in kernel.full_dims_cm),
               tuple(kernel.active_dimensions), option_key,
               _f64(diffusion), None if surface is None else _f64(surface),
               step_type, _f64(sign_h), _f64(sign_o))
        if current_root_mode() != "legacy_full_roots":
            key = (ROOT_PREFIX_POLICY, current_root_mode()) + key
        if key in self._values:
            self._hits += 1
            payload, size = self._values.pop(key)
            self._values[key] = (payload, size)
            self.cancel_check()
            return _thaw(payload)

        self._misses += 1
        self._kernel_calls += 1
        result = kernel.evaluate(times, diffusion, surface, **options)
        self.cancel_check()
        try:
            payload = _freeze(result)
        except TypeError:
            self._bypasses += 1
            return result
        size = _estimated_size(key) + _estimated_size(payload) + 128
        if size <= self.max_bytes:
            self.cancel_check()
            while self._values and (len(self._values) >= self.max_entries
                                    or self._bytes + size > self.max_bytes):
                _, (_, old_size) = self._values.popitem(last=False)
                self._bytes -= old_size
                self._evictions += 1
            self._values[key] = (payload, size)
            self._bytes += size
            self._peak_entries = max(self._peak_entries, len(self._values))
            self._peak_bytes = max(self._peak_bytes, self._bytes)
        return result


def evaluate_unit_kernel(*, kernel: object, original_times: object,
                         times: np.ndarray, diffusion: float,
                         surface: float | None, options: Mapping[str, object],
                         step_type: str, sign_h: float, sign_o: float) -> object:
    """Evaluate the original kernel unless a task-owned scope is active."""
    scope = getattr(_worker, "scope", None)
    if scope is None:
        return kernel.evaluate(times, diffusion, surface, **options)
    return scope.evaluate(kernel=kernel, original_times=original_times,
                          times=times, diffusion=diffusion, surface=surface,
                          options=options, step_type=step_type,
                          sign_h=sign_h, sign_o=sign_o)


def current_component_cache_diagnostics() -> ComponentCacheDiagnostics | None:
    scope = getattr(_worker, "scope", None)
    return None if scope is None else scope.diagnostics
