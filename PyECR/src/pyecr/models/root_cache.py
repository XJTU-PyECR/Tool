"""Exact, bounded root memoization for one synchronous analysis worker.

No model state or persistent scientific result contains this cache. Standalone
model calls retain the original solver unless a scope is explicitly entered.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from struct import pack
from sys import getsizeof
from threading import local
from typing import Callable

import numpy as np
import scipy

from pyecr.models import roots

# Bump this policy version whenever solver arithmetic/configuration changes.
ROOT_SOLVER_POLICY = "alpha-tan-brentq-nextafter-f64-v1"
ROOT_PREFIX_POLICY = "complete-global-prefix-range-v1-initial1-block8"
ROOT_MODES = ("legacy_full_roots", "on_demand_roots")
PREFIX_BLOCK_SIZE = 8
DEFAULT_MAX_ENTRIES = 4096
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
_worker = local()


@dataclass(frozen=True, slots=True)
class RootCacheDiagnostics:
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
    root_mode: str = "legacy_full_roots"
    roots_requested: int = 0
    solver_batches: int = 0
    solver_roots_requested: int = 0
    solver_roots_returned: int = 0
    prefix_extensions: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RootCacheScope:
    """Always create a fresh thread-local LRU; release all values on exit.

    AnalysisService is synchronous. Each fit establishes a new scope, including
    nested fits and repeated fits on the same thread. Threads never inherit it.
    Diagnostics on exit describe occupancy immediately before disposal.
    Estimated bytes include payloads, array/key headers and a conservative
    per-entry mapping allowance; they are not process RSS measurements.
    """

    def __init__(self, *, enabled: bool = True,
                 max_entries: int = DEFAULT_MAX_ENTRIES,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 cancel_check: Callable[[], None] | None = None,
                 root_mode: str = "legacy_full_roots"):
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        if root_mode not in ROOT_MODES:
            raise ValueError(f"root_mode must be one of {ROOT_MODES}")
        self.root_mode = root_mode
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.enabled = enabled
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.cancel_check = cancel_check or (lambda: None)
        self._values: OrderedDict[tuple, tuple] = OrderedDict()
        self._hits = self._misses = self._evictions = self._bytes = 0
        self._peak_entries = self._peak_bytes = 0
        self._requested = self._batches = self._solver_requested = self._returned = 0
        self._extensions = 0
        self._entered = self._closed = False
        self._final: RootCacheDiagnostics | None = None

    @property
    def diagnostics(self) -> RootCacheDiagnostics:
        return self._final or RootCacheDiagnostics(
            self.enabled, self._hits, self._misses, self._evictions,
            len(self._values), self._bytes, self._peak_entries, self._peak_bytes,
            self.max_entries, self.max_bytes, self.root_mode, self._requested,
            self._batches, self._solver_requested, self._returned, self._extensions)

    def __enter__(self) -> RootCacheScope:
        if self._entered or self._closed:
            raise RuntimeError("root cache scopes are single-use")
        self._entered = True
        self._previous = getattr(_worker, "scope", None)
        _worker.scope = self
        return self

    def __exit__(self, *exc) -> None:
        _worker.scope = self._previous
        self._previous = None
        self._final = self.diagnostics
        self._values.clear()
        self._bytes = 0
        self._closed = True

    def solve(self, bi: float, n_terms: int, solver: Callable):
        if not self._entered or self._closed or getattr(_worker, "scope", None) is not self:
            raise RuntimeError("root cache used outside its owning worker scope")
        self.cancel_check()
        roots._validate_inputs(bi, n_terms)
        self._requested += int(n_terms)
        # Canonical big-endian IEEE-754 bits: no rounding, tolerance, or bins.
        # Solver identities protect instrumentation/hot replacement within a task.
        key = (pack(">d", float(bi)), int(n_terms), ROOT_SOLVER_POLICY,
               pack(">d", roots.ASYMPTOTIC_BI_THRESHOLD), np.dtype("float64").str,
               np.__version__, scipy.__version__,
               pack(">d", np.finfo(np.float64).tiny),
               pack(">d", 4.0 * np.finfo(np.float64).eps), 100,
               solver, roots.solve_alpha_roots, roots._brentq_roots,
               roots._large_bi_roots, roots.brentq)
        if self.enabled and key in self._values:
            self._hits += 1
            value = self._values.pop(key)
            self._values[key] = value
            self.cancel_check()
            return tuple(np.frombuffer(data, dtype=np.float64) for data in value[0])
        self._misses += 1
        self._batches += 1
        self._solver_requested += int(n_terms)
        solved = solver(bi, n_terms)  # The sole miss path: unchanged production solver.
        self._returned += int(n_terms)
        self.cancel_check()  # Cancellation during a solve must never insert it.
        if not self.enabled:
            return solved
        # Immutable bytes backing prevents callers from re-enabling write access.
        payload = tuple(array.tobytes() for array in solved)
        frozen = tuple(np.frombuffer(data, dtype=np.float64) for data in payload)
        size = (getsizeof(key) + sum(getsizeof(part) for part in key)
                + getsizeof(frozen) + sum(getsizeof(a) + getsizeof(a.base)
                                           for a in frozen) + 128)
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
        return frozen

    def solve_prefix(self, bi: float, max_terms: int, selector: Callable, solver: Callable):
        """Privately extend a complete global prefix; publish only at success.

        The max_terms key is a budget, not a completeness claim. Payload byte
        length declares the complete [0,count) prefix. Selection is recomputed
        from its first coefficient on every extension, with no block reductions.
        """
        if not self._entered or self._closed or getattr(_worker, "scope", None) is not self:
            raise RuntimeError("root cache used outside its owning worker scope")
        self.cancel_check()
        roots._validate_inputs(bi, max_terms)
        self._requested += int(max_terms)
        key = (ROOT_PREFIX_POLICY, pack(">d", float(bi)), int(max_terms),
               PREFIX_BLOCK_SIZE, ROOT_SOLVER_POLICY,
               pack(">d", roots.ASYMPTOTIC_BI_THRESHOLD), np.dtype("float64").str,
               np.__version__, scipy.__version__,
               pack(">d", np.finfo(np.float64).tiny),
               pack(">d", 4.0 * np.finfo(np.float64).eps), 100,
               solver, roots.solve_alpha_roots, roots._brentq_roots,
               roots._large_bi_roots, roots.brentq)
        cached = self._values.get(key) if self.enabled else None
        if cached is not None:
            arrays = tuple(np.frombuffer(data, dtype=np.float64) for data in cached[0])
            selected = selector(arrays[0])
            self.cancel_check()
            if selected[2] or arrays[0].size == max_terms:
                self._hits += 1
                self._values.move_to_end(key)
                return arrays, selected
            self._extensions += 1
        else:
            arrays = (np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64))
        self._misses += 1
        while True:
            count = arrays[0].size
            target = min(max_terms, 1 if count == 0 else
                         ((count // PREFIX_BLOCK_SIZE) + 1) * PREFIX_BLOCK_SIZE)
            self.cancel_check()
            self._batches += 1
            self._solver_requested += target - count
            # The original production solver is the only actual root path.
            block = solver(bi, target - count, start_index=count)
            self._returned += target - count
            self.cancel_check()
            arrays = tuple(np.concatenate((old, new)) for old, new in zip(arrays, block))
            selected = selector(arrays[0])
            self.cancel_check()
            if selected[2] or target == max_terms:
                break
        payload = tuple(array.tobytes() for array in arrays)
        frozen = tuple(np.frombuffer(data, dtype=np.float64) for data in payload)
        size = (getsizeof(key) + sum(getsizeof(part) for part in key)
                + getsizeof(frozen) + sum(getsizeof(a) + getsizeof(a.base) for a in frozen) + 128)
        if self.enabled and size <= self.max_bytes:
            self.cancel_check()
            if key in self._values:
                _, old_size = self._values.pop(key)
                self._bytes -= old_size
            while self._values and (len(self._values) >= self.max_entries
                                    or self._bytes + size > self.max_bytes):
                _, (_, old_size) = self._values.popitem(last=False)
                self._bytes -= old_size
                self._evictions += 1
            self._values[key] = (payload, size)
            self._bytes += size
            self._peak_entries = max(self._peak_entries, len(self._values))
            self._peak_bytes = max(self._peak_bytes, self._bytes)
        return frozen, selected


def resolve_alpha_roots(bi: float, n_terms: int, *, solver: Callable):
    """Delegate a finite-axis request to its worker, or directly to the solver."""
    scope = getattr(_worker, "scope", None)
    return solver(bi, n_terms) if scope is None else scope.solve(bi, n_terms, solver)


def current_root_cache_diagnostics() -> RootCacheDiagnostics | None:
    """Read a detached counter snapshot; never expose cache values."""
    scope = getattr(_worker, "scope", None)
    return None if scope is None else scope.diagnostics


def current_root_mode() -> str:
    scope = getattr(_worker, "scope", None)
    return "legacy_full_roots" if scope is None else scope.root_mode


def resolve_alpha_root_prefix(bi: float, max_terms: int, *, selector: Callable, solver: Callable):
    scope = getattr(_worker, "scope", None)
    if scope is None:
        raise RuntimeError("on-demand roots require an explicit task scope")
    return scope.solve_prefix(bi, max_terms, selector, solver)
