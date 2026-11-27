"""Threaded task execution with main-thread-consumable events."""

from __future__ import annotations

import traceback
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Lock
from typing import Callable, Literal
from uuid import uuid4

from pyecr.gui.state import GuiPage
from pyecr.fitting.progress import FitProgressEvent, FitProgressRevision, TERMINAL_PHASES, fit_telemetry_scope

__all__ = ["BackgroundTaskRunner", "ProgressCallback", "TaskEvent"]


ProgressCallback = Callable[[float, str], None]
WorkerCallable = Callable[[ProgressCallback], object]


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """One worker-to-main-thread event."""

    task_id: str
    page: GuiPage
    kind: Literal["progress", "completed", "cancelled", "error"]
    progress: float | None = None
    message: str = ""
    result: object | None = None
    exception_type: str | None = None
    traceback_text: str | None = None
    fit_progress: FitProgressEvent | None = None


class BackgroundTaskRunner:
    """Run services in workers and expose results through a thread-safe queue."""

    def __init__(self, *, max_workers: int = 2) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="pyecr-gui",
        )
        self._events: SimpleQueue[TaskEvent] = SimpleQueue()
        self._closed = False
        self._lock = Lock()
        self._active_task_ids: set[str] = set()
        self._submitted_count = 0
        self._terminal_count = 0
        self._fit_events: OrderedDict[str, TaskEvent] = OrderedDict()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active_task_ids)

    @property
    def submitted_count(self) -> int:
        with self._lock:
            return self._submitted_count

    @property
    def terminal_count(self) -> int:
        with self._lock:
            return self._terminal_count

    def submit(self, page: GuiPage, work: WorkerCallable, *,
               fit_revision: object | None = None) -> str:
        """Submit work immediately and return its unique task identifier."""

        if self._closed:
            raise RuntimeError("background task runner is closed")
        normalized_page = GuiPage(page)
        task_id = uuid4().hex
        latest_fit_progress: FitProgressEvent | None = None
        with self._lock:
            self._active_task_ids.add(task_id)
            self._submitted_count += 1

        def progress_callback(fraction: float, message: str = "") -> None:
            if fit_revision is not None:
                # Structured fit telemetry supplies real phase-local progress.
                return
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise TypeError("progress fraction must be a real number")
            normalized_fraction = float(fraction)
            if not 0.0 <= normalized_fraction <= 1.0:
                raise ValueError("progress fraction must be between 0 and 1")
            self._events.put(
                TaskEvent(
                    task_id=task_id,
                    page=normalized_page,
                    kind="progress",
                    progress=normalized_fraction,
                    message=str(message),
                )
            )

        def fit_callback(snapshot: FitProgressEvent) -> None:
            nonlocal latest_fit_progress
            latest_fit_progress = snapshot
            if snapshot.phase in TERMINAL_PHASES:
                return  # Attached to the lossless terminal event below.
            with self._lock:
                self._fit_events[task_id] = TaskEvent(
                    task_id, normalized_page, "progress", snapshot.fraction,
                    snapshot.message, fit_progress=snapshot)

        def run() -> None:
            try:
                with fit_telemetry_scope(
                    fit_callback if fit_revision is not None else None,
                    job_id=task_id, revision=fit_revision,
                ):
                    result = work(progress_callback)
            except CancelledError as exc:
                self._events.put(
                    TaskEvent(
                        task_id=task_id,
                        page=normalized_page,
                        kind="cancelled",
                        message=str(exc) or "Cancelled",
                        fit_progress=latest_fit_progress,
                    )
                )
                return
            except Exception as exc:  # displayed verbatim in the GUI error console
                self._events.put(
                    TaskEvent(
                        task_id=task_id,
                        page=normalized_page,
                        kind="error",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                        traceback_text=traceback.format_exc(),
                        fit_progress=latest_fit_progress,
                    )
                )
                return
            self._events.put(
                TaskEvent(
                    task_id=task_id,
                    page=normalized_page,
                    kind="completed",
                    progress=1.0,
                    message="Completed",
                    result=result,
                    fit_progress=latest_fit_progress,
                )
            )
            # Terminal accounting is completed in the finally clause below.

        def tracked_run() -> None:
            try:
                run()
            finally:
                with self._lock:
                    self._active_task_ids.discard(task_id)
                    self._terminal_count += 1
                    self._fit_events.pop(task_id, None)

        future = self._executor.submit(tracked_run)

        def cancelled_before_start(future) -> None:
            if future.cancelled():
                with self._lock:
                    self._active_task_ids.discard(task_id)
                    self._terminal_count += 1
                    self._fit_events.pop(task_id, None)
                self._events.put(TaskEvent(task_id, normalized_page, "cancelled",
                    message="Cancelled before worker started",
                    fit_progress=(FitProgressEvent(task_id, FitProgressRevision.capture(fit_revision),
                        1, "Cancelled", 0., "Cancelled before worker started")
                        if fit_revision is not None else None)))
        future.add_done_callback(cancelled_before_start)
        return task_id

    def drain_events(self, *, limit: int = 200) -> tuple[TaskEvent, ...]:
        """Remove currently queued events without blocking the GUI thread."""

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
        events: list[TaskEvent] = []
        with self._lock:
            while self._fit_events and len(events) < limit:
                _, event = self._fit_events.popitem(last=False)
                events.append(event)
        for _ in range(limit - len(events)):
            try:
                events.append(self._events.get_nowait())
            except Empty:
                break
        return tuple(events)

    def shutdown(self, *, wait: bool = True) -> None:
        """Stop accepting work and release executor resources."""

        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)
