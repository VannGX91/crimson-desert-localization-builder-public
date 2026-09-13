"""Job control, state transitions, locking, and cancellation."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from crimson0020.translation_runtime.errors import TranslationRuntimeError

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class JobState(str, Enum):
    """Seven lifecycle states for translation jobs according to PLAN.md."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    COMPLETED_WITH_PENDING = "completed_with_pending"
    FAILED = "failed"


_VALID_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.IDLE: {JobState.RUNNING},
    JobState.RUNNING: {
        JobState.PAUSING,
        JobState.COMPLETED,
        JobState.COMPLETED_WITH_PENDING,
        JobState.FAILED,
    },
    JobState.PAUSING: {JobState.PAUSED, JobState.FAILED},
    JobState.PAUSED: {JobState.RUNNING, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.COMPLETED_WITH_PENDING: set(),
    JobState.FAILED: set(),
}


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Immutable snapshot of current job progress and status."""

    job_id: str
    state: JobState
    completed_count: int
    total_count: int
    pending_count: int
    failure_count: int
    updated_at: float


class CancellationToken:
    """Token allowing safe cancellation or pause of translation operations."""

    def __init__(self) -> None:
        self._cancelled = False
        self._cancel_reason = ""

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def cancel_reason(self) -> str:
        return self._cancel_reason

    def cancel(self, reason: str = "Operation cancelled") -> None:
        self._cancelled = True
        self._cancel_reason = reason


class FileLock:
    """Process-level file lock to prevent concurrent writers."""

    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = Path(lock_path)
        self._fd: int | None = None
        self._acquired = False

    def acquire(self) -> bool:
        """Attempts non-blocking acquisition of the file lock."""
        if self._acquired:
            return True

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(
                str(self.lock_path),
                os.O_RDWR | os.O_CREAT,
                0o666,
            )
            if sys.platform == "win32":
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._acquired = True
            return True
        except (OSError, IOError, BlockingIOError):
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            return False

    def release(self) -> None:
        """Releases the lock and closes the file descriptor."""
        if not self._acquired or self._fd is None:
            return

        try:
            if sys.platform == "win32":
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
        except OSError:
            pass
        finally:
            self._fd = None
            self._acquired = False
            if self.lock_path.exists():
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise TranslationRuntimeError(
                f"Could not acquire job lock: {self.lock_path}"
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


class JobController:
    """Controls lifecycle state transitions and tracking for a translation job."""

    def __init__(self, job_id: str, total_count: int) -> None:
        self.job_id = job_id
        self.total_count = total_count
        self._state = JobState.IDLE
        self._completed_count = 0
        self._pending_count = total_count
        self._failure_count = 0
        self._updated_at = time.time()

    @property
    def state(self) -> JobState:
        return self._state

    def _transition(self, target: JobState) -> None:
        valid_targets = _VALID_TRANSITIONS.get(self._state, set())
        if target not in valid_targets:
            raise TranslationRuntimeError(
                f"Invalid job state transition from {self._state.value} to {target.value}"
            )
        self._state = target
        self._updated_at = time.time()

    def start(self) -> None:
        self._transition(JobState.RUNNING)

    def pause_requested(self) -> None:
        self._transition(JobState.PAUSING)

    def paused(self) -> None:
        self._transition(JobState.PAUSED)

    def resume(self) -> None:
        self._transition(JobState.RUNNING)

    def complete(self, pending_count: int = 0) -> None:
        if pending_count > 0:
            self._pending_count = pending_count
            self._transition(JobState.COMPLETED_WITH_PENDING)
        else:
            self._pending_count = 0
            self._transition(JobState.COMPLETED)

    def fail(self) -> None:
        self._transition(JobState.FAILED)

    def record_progress(
        self, completed: int, pending: int, failures: int = 0
    ) -> None:
        self._completed_count = completed
        self._pending_count = pending
        self._failure_count = failures
        self._updated_at = time.time()

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            job_id=self.job_id,
            state=self._state,
            completed_count=self._completed_count,
            total_count=self.total_count,
            pending_count=self._pending_count,
            failure_count=self._failure_count,
            updated_at=self._updated_at,
        )
