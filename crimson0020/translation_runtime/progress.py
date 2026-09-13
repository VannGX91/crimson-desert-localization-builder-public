"""Immutable progress reporting and cooperative translation pause control."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable


class ProgressState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    COMPLETED_WITH_PENDING = "completed_with_pending"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    job_id: str
    state: ProgressState
    total_entries: int
    saved_entries: int
    translated_entries: int
    resumed_entries: int
    pending_entries: int
    failure_count: int
    request_count: int
    current_route: str | None
    current_model: str | None
    fallback_count: int
    cooldown_seconds: float
    failure_summary: str | None
    updated_at: float


class PauseRequested(Exception):
    """Internal signal used to stop scheduling after the current request."""


class TranslationControl:
    """Thread-safe cooperative pause/cancel control for one translation job."""

    def __init__(self) -> None:
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""

    def request_pause(self, reason: str = "Paused by user") -> None:
        with self._lock:
            self._reason = reason
        self._pause.set()

    def clear_pause(self) -> None:
        self._pause.clear()

    def cancel(self, reason: str = "Cancelled by user") -> None:
        with self._lock:
            self._reason = reason
        self._cancel.set()

    @property
    def pause_requested(self) -> bool:
        return self._pause.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def raise_if_stopped(self) -> None:
        if self.cancelled:
            raise PauseRequested(self.reason or "Cancelled by user")

    def should_pause_after_request(self) -> bool:
        return self.pause_requested or self.cancelled


class ProgressReporter:
    """Publishes monotonic immutable snapshots at most four times per second."""

    def __init__(
        self,
        job_id: str,
        total_entries: int,
        callback: Callable[[ProgressSnapshot], None] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        min_interval: float = 0.25,
    ) -> None:
        if not job_id or total_entries < 0 or min_interval < 0:
            raise ValueError("Invalid progress reporter configuration.")
        self._callback = callback
        self._clock = clock
        self._min_interval = min_interval
        self._last_emit = float("-inf")
        self._last_snapshot = ProgressSnapshot(
            job_id=job_id,
            state=ProgressState.IDLE,
            total_entries=total_entries,
            saved_entries=0,
            translated_entries=0,
            resumed_entries=0,
            pending_entries=total_entries,
            failure_count=0,
            request_count=0,
            current_route=None,
            current_model=None,
            fallback_count=0,
            cooldown_seconds=0.0,
            failure_summary=None,
            updated_at=self._clock(),
        )
        self.callback_errors: tuple[str, ...] = ()

    @property
    def snapshot(self) -> ProgressSnapshot:
        return self._last_snapshot

    def publish(self, **changes: object) -> ProgressSnapshot | None:
        previous = self._last_snapshot
        candidate = replace(previous, **changes, updated_at=self._clock())
        self._validate(candidate, previous)
        now = candidate.updated_at
        if (
            candidate.state not in {
                ProgressState.PAUSED,
                ProgressState.COMPLETED,
                ProgressState.COMPLETED_WITH_PENDING,
                ProgressState.FAILED,
            }
            and now - self._last_emit < self._min_interval
        ):
            self._last_snapshot = candidate
            return None
        self._last_snapshot = candidate
        self._last_emit = now
        if self._callback is not None:
            try:
                self._callback(candidate)
            except Exception as error:
                self.callback_errors = (*self.callback_errors, str(error))
        return candidate

    def flush(self) -> ProgressSnapshot:
        self._last_emit = float("-inf")
        return self.publish()

    @staticmethod
    def _validate(
        snapshot: ProgressSnapshot, previous: ProgressSnapshot | None = None
    ) -> None:
        if previous is not None:
            for field in (
                "saved_entries",
                "translated_entries",
                "resumed_entries",
                "request_count",
                "fallback_count",
            ):
                if getattr(snapshot, field) < getattr(previous, field):
                    raise ValueError(f"{field} cannot decrease.")
            if snapshot.failure_count < previous.failure_count:
                raise ValueError("failure_count cannot decrease.")
        if snapshot.saved_entries < max(
            snapshot.translated_entries, snapshot.resumed_entries
        ):
            raise ValueError("Saved progress cannot be below completed progress.")
        if snapshot.pending_entries < 0 or snapshot.failure_count < 0:
            raise ValueError("Progress counts cannot be negative.")
        if snapshot.saved_entries + snapshot.pending_entries > snapshot.total_entries:
            raise ValueError("Progress counts exceed total entries.")
