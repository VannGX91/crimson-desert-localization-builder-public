"""Persistent route health tracking and circuit breaker implementation (RTR-10)."""

from __future__ import annotations

import datetime
import email.utils
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from crimson0020.translation_runtime.contracts import FailureKind
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase

FALLBACK_DELAYS: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0, 600.0)
MIN_RETRY_AFTER_SECONDS: float = 30.0
MAX_RETRY_AFTER_SECONDS: float = 1800.0  # 30 minutes
ANONYMOUS_AUTH_DISABLE_SECONDS: float = 6.0 * 3600.0  # 6 hours
MISSING_MODEL_DISABLE_SECONDS: float = 24.0 * 3600.0  # 24 hours
COMPACTION_RETENTION_SECONDS: float = 30.0 * 86400.0  # 30 days


class CircuitState(str, Enum):
    """Circuit breaker operational states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class RouteHealthRecord:
    """Snapshot of health metrics and circuit status for a route and model."""

    route_id: str
    model_id: str
    state: CircuitState
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_failure_kind: str | None = None
    last_status_code: int | None = None
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "model_id": self.model_id,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_until": self.cooldown_until,
            "last_failure_kind": self.last_failure_kind,
            "last_status_code": self.last_status_code,
            "updated_at": self.updated_at,
        }


def parse_retry_after(retry_after: str | int | float | None, now: float | None = None) -> float | None:
    """Parses Retry-After header as numeric seconds or HTTP-date, clamped to [30s, 1800s]."""
    if retry_after is None:
        return None

    ref_now = now if now is not None else time.time()

    if isinstance(retry_after, (int, float)):
        seconds = float(retry_after)
    else:
        text = str(retry_after).strip()
        if not text:
            return None
        # Try parsing numeric seconds first
        try:
            seconds = float(text)
        except ValueError:
            # Try parsing HTTP-date format (RFC 2822 / RFC 7231 / RFC 1123)
            try:
                parsed_dt = email.utils.parsedate_to_datetime(text)
                if parsed_dt.tzinfo is None:
                    parsed_dt = parsed_dt.replace(tzinfo=datetime.timezone.utc)
                target_ts = parsed_dt.timestamp()
                seconds = target_ts - ref_now
            except Exception:
                return None

    # Clamp to [30s, 1800s]
    if seconds < MIN_RETRY_AFTER_SECONDS:
        return MIN_RETRY_AFTER_SECONDS
    if seconds > MAX_RETRY_AFTER_SECONDS:
        return MAX_RETRY_AFTER_SECONDS
    return seconds


def compute_cooldown(
    failure_kind: FailureKind | str | None,
    status_code: int | None = None,
    retry_after: str | int | float | None = None,
    consecutive_failures: int = 1,
    now: float | None = None,
) -> float:
    """Classifies cooldown duration by failure kind, status code, and Retry-After header."""
    kind_str = failure_kind.value if isinstance(failure_kind, FailureKind) else str(failure_kind or "")

    # 401/403 or UNAUTHORIZED disables route for 6 hours
    if status_code in (401, 403) or kind_str == FailureKind.UNAUTHORIZED.value:
        return ANONYMOUS_AUTH_DISABLE_SECONDS

    # 404 or UNSUPPORTED_MODEL disables model for 24 hours
    if status_code == 404 or kind_str == FailureKind.UNSUPPORTED_MODEL.value:
        return MISSING_MODEL_DISABLE_SECONDS

    # Honor Retry-After if present
    parsed_ra = parse_retry_after(retry_after, now=now)
    if parsed_ra is not None:
        return parsed_ra

    # Fallback progressive backoff delays: 30, 60, 120, 300, 600
    idx = max(0, min(consecutive_failures - 1, len(FALLBACK_DELAYS) - 1))
    return FALLBACK_DELAYS[idx]


class RouteHealthTracker:
    """Tracks persistent circuit breaker states per route/model with single half-open probe."""

    def __init__(self, database: KnowledgeDatabase) -> None:
        self.database = database
        self._lock = threading.Lock()
        self._in_flight_probes: set[tuple[str, str]] = set()
        self._cache: dict[tuple[str, str], RouteHealthRecord] = {}

    def get_health(self, route_id: str, model_id: str, now: float | None = None) -> RouteHealthRecord:
        """Retrieves current health status for (route_id, model_id)."""
        ref_now = now if now is not None else time.time()
        with self._lock:
            return self._get_health_locked(route_id, model_id, ref_now)

    def _get_health_locked(self, route_id: str, model_id: str, now: float) -> RouteHealthRecord:
        key = (route_id, model_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with self.database.connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT state, consecutive_failures, cooldown_until, last_failure_kind, last_status_code, updated_at
                FROM route_health
                WHERE route_id = ? AND model_id = ?
                """,
                (route_id, model_id),
            )
            row = cur.fetchone()

        if not row:
            record = RouteHealthRecord(
                route_id=route_id,
                model_id=model_id,
                state=CircuitState.CLOSED,
                consecutive_failures=0,
                cooldown_until=0.0,
                last_failure_kind=None,
                last_status_code=None,
                updated_at=now,
            )
            self._cache[key] = record
            return record

        state_str, consecutive_failures, cooldown_until, last_failure_kind, last_status_code, updated_at = row
        state = CircuitState(state_str)

        record = RouteHealthRecord(
            route_id=route_id,
            model_id=model_id,
            state=state,
            consecutive_failures=consecutive_failures,
            cooldown_until=cooldown_until,
            last_failure_kind=last_failure_kind,
            last_status_code=last_status_code,
            updated_at=updated_at,
        )
        self._cache[key] = record
        return record

    def is_available(self, route_id: str, model_id: str, now: float | None = None) -> bool:
        """Checks if a route/model is open for general execution (CLOSED state)."""
        ref_now = now if now is not None else time.time()
        with self._lock:
            rec = self._get_health_locked(route_id, model_id, ref_now)
            return rec.state == CircuitState.CLOSED

    def retry_after_seconds(
        self, route_id: str, model_id: str, now: float | None = None
    ) -> float | None:
        """Returns the remaining cooldown for an unavailable route/model."""
        ref_now = now if now is not None else time.time()
        key = (route_id, model_id)
        with self._lock:
            rec = self._get_health_locked(route_id, model_id, ref_now)
            if rec.state == CircuitState.OPEN:
                return max(0.0, rec.cooldown_until - ref_now)
            if rec.state == CircuitState.HALF_OPEN and key in self._in_flight_probes:
                return 0.1
            return None

    def acquire_probe(self, route_id: str, model_id: str, now: float | None = None) -> bool:
        """Acquires execution permission.

        - If CLOSED: succeeds.
        - If OPEN and cooldown not expired: fails.
        - If OPEN and cooldown expired: transitions to HALF_OPEN and grants a single probe.
        - If HALF_OPEN: grants probe only if no probe is currently in flight.
        """
        ref_now = now if now is not None else time.time()
        key = (route_id, model_id)

        with self._lock:
            rec = self._get_health_locked(route_id, model_id, ref_now)

            if rec.state == CircuitState.CLOSED:
                return True

            if rec.state == CircuitState.OPEN:
                if ref_now < rec.cooldown_until:
                    return False
                # Cooldown expired: transition to HALF_OPEN
                if key in self._in_flight_probes:
                    return False
                self._set_state_locked(
                    route_id=route_id,
                    model_id=model_id,
                    state=CircuitState.HALF_OPEN,
                    consecutive_failures=rec.consecutive_failures,
                    cooldown_until=rec.cooldown_until,
                    last_failure_kind=rec.last_failure_kind,
                    last_status_code=rec.last_status_code,
                    now=ref_now,
                )
                self._in_flight_probes.add(key)
                return True

            if rec.state == CircuitState.HALF_OPEN:
                if key in self._in_flight_probes:
                    return False
                self._in_flight_probes.add(key)
                return True

            return False

    def release_probe(self, route_id: str, model_id: str) -> None:
        """Releases an in-flight probe without changing state."""
        with self._lock:
            self._in_flight_probes.discard((route_id, model_id))

    def record_success(self, route_id: str, model_id: str, now: float | None = None) -> RouteHealthRecord:
        """Records successful execution, resetting circuit state to CLOSED."""
        ref_now = now if now is not None else time.time()
        key = (route_id, model_id)

        with self._lock:
            self._in_flight_probes.discard(key)
            current = self._get_health_locked(route_id, model_id, ref_now)
            if current.state == CircuitState.CLOSED and current.consecutive_failures == 0:
                return current
            self._set_state_locked(
                route_id=route_id,
                model_id=model_id,
                state=CircuitState.CLOSED,
                consecutive_failures=0,
                cooldown_until=0.0,
                last_failure_kind=None,
                last_status_code=None,
                now=ref_now,
            )
            return self._get_health_locked(route_id, model_id, ref_now)

    def record_failure(
        self,
        route_id: str,
        model_id: str,
        failure_kind: FailureKind | str,
        status_code: int | None = None,
        retry_after: str | int | float | None = None,
        now: float | None = None,
    ) -> RouteHealthRecord:
        """Records a failure, transitions circuit to OPEN, and computes cooldown."""
        ref_now = now if now is not None else time.time()
        key = (route_id, model_id)

        with self._lock:
            self._in_flight_probes.discard(key)
            rec = self._get_health_locked(route_id, model_id, ref_now)

            new_failures = rec.consecutive_failures + 1
            cooldown_seconds = compute_cooldown(
                failure_kind=failure_kind,
                status_code=status_code,
                retry_after=retry_after,
                consecutive_failures=new_failures,
                now=ref_now,
            )
            cooldown_until = ref_now + cooldown_seconds

            kind_str = failure_kind.value if isinstance(failure_kind, FailureKind) else str(failure_kind)

            self._set_state_locked(
                route_id=route_id,
                model_id=model_id,
                state=CircuitState.OPEN,
                consecutive_failures=new_failures,
                cooldown_until=cooldown_until,
                last_failure_kind=kind_str,
                last_status_code=status_code,
                now=ref_now,
            )
            return self._get_health_locked(route_id, model_id, ref_now)

    def _set_state_locked(
        self,
        route_id: str,
        model_id: str,
        state: CircuitState,
        consecutive_failures: int,
        cooldown_until: float,
        last_failure_kind: str | None,
        last_status_code: int | None,
        now: float,
    ) -> None:
        record = RouteHealthRecord(
            route_id=route_id,
            model_id=model_id,
            state=state,
            consecutive_failures=consecutive_failures,
            cooldown_until=cooldown_until,
            last_failure_kind=last_failure_kind,
            last_status_code=last_status_code,
            updated_at=now,
        )
        with self.database.transaction() as con:
            con.execute(
                """
                INSERT INTO route_health (
                    route_id, model_id, state, consecutive_failures,
                    cooldown_until, last_failure_kind, last_status_code, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(route_id, model_id) DO UPDATE SET
                    state = excluded.state,
                    consecutive_failures = excluded.consecutive_failures,
                    cooldown_until = excluded.cooldown_until,
                    last_failure_kind = excluded.last_failure_kind,
                    last_status_code = excluded.last_status_code,
                    updated_at = excluded.updated_at
                """,
                (
                    route_id,
                    model_id,
                    state.value,
                    consecutive_failures,
                    cooldown_until,
                    last_failure_kind,
                    last_status_code,
                    now,
                ),
            )
        self._cache[(route_id, model_id)] = record

    def compact_stale_health(
        self,
        older_than_seconds: float = COMPACTION_RETENTION_SECONDS,
        now: float | None = None,
    ) -> int:
        """Deletes health records older than older_than_seconds without touching Translation Memory."""
        ref_now = now if now is not None else time.time()
        threshold = ref_now - older_than_seconds

        with self._lock:
            with self.database.transaction() as con:
                cur = con.cursor()
                cur.execute(
                    "DELETE FROM route_health WHERE updated_at < ?",
                    (threshold,),
                )
                deleted = cur.rowcount
            if deleted:
                self._cache.clear()
            return deleted
