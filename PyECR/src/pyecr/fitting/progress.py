"""Transient fit telemetry. No result/config serialization or model arithmetic.

Counters observe existing production calls. SSE/cost are read from completed
finite candidates, never obtained with an additional forward evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite, sqrt
from threading import local
from time import monotonic
from typing import Callable, Iterator
from uuid import uuid4

from pyecr.models.root_cache import RootCacheDiagnostics, current_root_cache_diagnostics

_worker = local()
TERMINAL_PHASES = frozenset({"Completed", "Cancelled", "Failed"})


@dataclass(frozen=True, slots=True)
class FitProgressRevision:
    dataset_revision: int | None = None
    config_revision: int | None = None
    physical_input_revision: int | None = None

    @classmethod
    def capture(cls, revision: object | None) -> FitProgressRevision:
        if revision is None:
            return cls()
        return cls(revision.data_revision, revision.config_revision,
                   revision.physical_input_revision)


@dataclass(frozen=True, slots=True)
class FitProgressCandidate:
    """A finite completed full-data endpoint, never a published FitResult.

    Only small immutable parameter/metric values cross the worker boundary.
    Model identity belongs to the endpoint's actual stage, including a D-only
    preparation stage of a dual D+k request.
    """
    model_id: str
    params: tuple[tuple[str, float], ...]
    raw_sse: float
    observation_count: int
    start_index: int
    native_status: int | None = None

    @property
    def rmse(self) -> float:
        return sqrt(self.raw_sse / self.observation_count)


@dataclass(frozen=True, slots=True)
class FitProgressEvent:
    job_id: str
    revision: FitProgressRevision
    sequence: int
    phase: str
    elapsed_s: float
    message: str
    start_index: int | None = None  # 1-based currently executing start
    total_starts: int | None = None
    completed_starts: int | None = None
    best_raw_sse: float | None = None  # minimum completed finite candidate in stage
    optimizer_cost: float | None = None  # SciPy cost of that SAME candidate
    loss: str | None = None
    forward_calls: int | None = None
    residual_calls: int | None = None
    root_cache: RootCacheDiagnostics | None = None
    emitted_monotonic_s: float | None = None  # transport latency, transient only
    best_candidate: FitProgressCandidate | None = None

    @property
    def fraction(self) -> float | None:
        if self.phase not in {"M0", "M1", "M2", "M3", "M7", "M8", "M3 fixed-D conditional"} or not self.total_starts:
            return None
        return (self.completed_starts or 0) / self.total_starts


class FitTelemetry:
    """One worker scope; emit same-stage updates at most 10 Hz.

    Phase/start boundaries and terminal states bypass the throttle. Consumers
    must be nonblocking (the GUI runner coalesces snapshots under a short lock).
    A failed observer is detached so telemetry cannot change fit acceptance.
    """

    def __init__(self, callback: Callable[[FitProgressEvent], None], *,
                 job_id: str | None = None, revision: object | None = None,
                 clock: Callable[[], float] = monotonic):
        self.callback = callback
        self.job_id = job_id or uuid4().hex
        self.revision = FitProgressRevision.capture(revision)
        self.clock = clock
        self.started = clock()
        self.last_emit = float("-inf")
        self.sequence = 0
        self.phase = "Preparing"
        self.message = "Preparing fit"
        self.index = self.total = self.completed = None
        self.best = self.cost = self.loss = None
        self.best_candidate: FitProgressCandidate | None = None
        self.forward = self.residual = None
        self.cache = None
        self.last: FitProgressEvent | None = None
        self.observer_error: str | None = None

    def emit(self, *, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self.last_emit < 0.1:
            return
        self.cache = current_root_cache_diagnostics() or self.cache
        self.sequence += 1
        self.last = FitProgressEvent(
            self.job_id, self.revision, self.sequence, self.phase,
            max(0., now - self.started), self.message,
            self.index, self.total, self.completed, self.best, self.cost, self.loss,
            self.forward, self.residual, self.cache, now, self.best_candidate)
        self.last_emit = now
        if self.observer_error is None:
            try:
                self.callback(self.last)
            except Exception as exc:
                self.observer_error = f"{type(exc).__name__}: {exc}"

    def stage(self, phase: str, message: str = "") -> None:
        self.phase = phase
        self.message = message or phase
        self.emit(force=True)

    def start(self, phase: str, index: int, total: int, loss: str) -> None:
        if phase != self.phase:
            self.best = self.cost = None
        self.phase, self.index, self.total, self.completed = phase, index, total, index - 1
        self.loss = loss
        self.message = f"{phase} start {index}/{total}"
        self.emit(force=True)

    def candidate(self, sse: float, cost: float, *, params=None, model_id=None,
                  observation_count=None, native_status=None) -> None:
        if isfinite(sse) and (self.best is None or sse < self.best):
            self.best, self.cost = float(sse), float(cost)
            if params is not None and model_id is not None and observation_count:
                values = tuple((str(k), float(v)) for k, v in params.items())
                if sse >= 0 and values and all(isfinite(v) for _, v in values):
                    self.best_candidate = FitProgressCandidate(
                        str(getattr(model_id, 'value', model_id)), values,
                        float(sse), int(observation_count), self.index or 1,
                        native_status)
        self.emit()

    def finish_start(self) -> None:
        self.completed = self.index
        self.emit(force=True)

    def count(self, kind: str) -> None:
        if kind == "forward":
            self.forward = (self.forward or 0) + 1
        else:
            self.residual = (self.residual or 0) + 1
        self.emit()


@contextmanager
def fit_telemetry_scope(callback=None, *, job_id=None, revision=None) -> Iterator[FitTelemetry | None]:
    """Reuse a worker's scope; a direct service caller can opt into its own.

    Only the owning outer scope emits terminal events, after the whole worker
    (including presentation) exits. The thread-local previous scope is restored.
    """
    existing = current_telemetry()
    if callback is None or existing is not None:
        yield existing
        return
    telemetry = FitTelemetry(callback, job_id=job_id, revision=revision)
    _worker.telemetry = telemetry
    try:
        telemetry.stage("Preparing", "Preparing fit")
        yield telemetry
    except BaseException as exc:
        from concurrent.futures import CancelledError
        telemetry.stage("Cancelled" if isinstance(exc, CancelledError) else "Failed", str(exc))
        raise
    else:
        telemetry.stage("Completed", "Worker completed; awaiting main-thread acceptance")
    finally:
        _worker.telemetry = existing


def current_telemetry() -> FitTelemetry | None:
    return getattr(_worker, "telemetry", None)


def fit_stage(phase: str, message: str = "") -> None:
    if (telemetry := current_telemetry()) is not None:
        telemetry.stage(phase, message)


def fit_start(phase: str, index: int, total: int, loss: str) -> None:
    from pyecr.fitting.iteration_observation import observe_start
    observe_start(phase,index,total)
    if (telemetry := current_telemetry()) is not None:
        telemetry.start(phase, index, total, loss)


def fit_candidate(sse: float, cost: float, *, params=None, model_id=None,
                  observation_count=None, native_status=None) -> None:
    if (telemetry := current_telemetry()) is not None:
        telemetry.candidate(sse, cost, params=params, model_id=model_id,
                            observation_count=observation_count, native_status=native_status)


def fit_start_finished() -> None:
    if (telemetry := current_telemetry()) is not None:
        telemetry.finish_start()


def count_fit_call(kind: str) -> None:
    from pyecr.fitting.iteration_observation import count_observed_call
    count_observed_call(kind)
    if (telemetry := current_telemetry()) is not None:
        telemetry.count(kind)


def fit_counters_available() -> None:
    if (telemetry := current_telemetry()) is not None:
        telemetry.forward = telemetry.residual = 0


def capture_fit_cache() -> None:
    if (telemetry := current_telemetry()) is not None:
        telemetry.cache = current_root_cache_diagnostics()
        telemetry.emit(force=True)
