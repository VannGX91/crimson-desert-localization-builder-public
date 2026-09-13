"""Tests for persistent route health tracking and circuit breaker (RTR-10)."""

from __future__ import annotations

import concurrent.futures
import datetime
from pathlib import Path

import pytest

from crimson0020.translation_runtime.contracts import FailureKind
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.router.health import (
    ANONYMOUS_AUTH_DISABLE_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    MIN_RETRY_AFTER_SECONDS,
    MISSING_MODEL_DISABLE_SECONDS,
    CircuitState,
    RouteHealthTracker,
    compute_cooldown,
    parse_retry_after,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> KnowledgeDatabase:
    db_file = tmp_path / "test_health.sqlite3"
    return KnowledgeDatabase(database_path=db_file)


def test_retry_after_parsing_and_clamping() -> None:
    now = 1700000000.0
    # None or empty
    assert parse_retry_after(None, now=now) is None
    assert parse_retry_after("", now=now) is None

    # Invalid text
    assert parse_retry_after("not-a-number-or-date", now=now) is None

    # Numeric seconds below min (5s -> 30s min)
    assert parse_retry_after(5, now=now) == MIN_RETRY_AFTER_SECONDS
    assert parse_retry_after("5", now=now) == MIN_RETRY_AFTER_SECONDS

    # Valid numeric seconds in range
    assert parse_retry_after(120, now=now) == 120.0
    assert parse_retry_after("120", now=now) == 120.0

    # Numeric seconds above max (3600s -> 1800s max)
    assert parse_retry_after(3600, now=now) == MAX_RETRY_AFTER_SECONDS
    assert parse_retry_after("3600", now=now) == MAX_RETRY_AFTER_SECONDS

    # HTTP-date string (RFC 2822 / RFC 7231)
    target_dt = datetime.datetime.fromtimestamp(now + 90.0, tz=datetime.timezone.utc)
    http_date = target_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    parsed = parse_retry_after(http_date, now=now)
    assert parsed is not None
    assert abs(parsed - 90.0) < 1.5

    # Past HTTP-date clamps to min
    past_dt = datetime.datetime.fromtimestamp(now - 100.0, tz=datetime.timezone.utc)
    past_http_date = past_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(past_http_date, now=now) == MIN_RETRY_AFTER_SECONDS


def test_cooldown_classification() -> None:
    now = 1000.0

    # 401/403 or UNAUTHORIZED -> 6 hours
    cd_401 = compute_cooldown(FailureKind.UNAUTHORIZED, status_code=401, now=now)
    assert cd_401 == ANONYMOUS_AUTH_DISABLE_SECONDS
    cd_403 = compute_cooldown(FailureKind.UNAUTHORIZED, status_code=403, now=now)
    assert cd_403 == ANONYMOUS_AUTH_DISABLE_SECONDS

    # 404 or UNSUPPORTED_MODEL -> 24 hours
    cd_404 = compute_cooldown(FailureKind.UNSUPPORTED_MODEL, status_code=404, now=now)
    assert cd_404 == MISSING_MODEL_DISABLE_SECONDS

    # Retry-After header present
    cd_ra = compute_cooldown(FailureKind.RATE_LIMIT, status_code=429, retry_after="60", now=now)
    assert cd_ra == 60.0

    # Fallback progressive delays: 30, 60, 120, 300, 600
    assert compute_cooldown(FailureKind.NETWORK, consecutive_failures=1, now=now) == 30.0
    assert compute_cooldown(FailureKind.TIMEOUT, consecutive_failures=2, now=now) == 60.0
    assert compute_cooldown(FailureKind.PROVIDER_INTERNAL, consecutive_failures=3, now=now) == 120.0
    assert compute_cooldown(FailureKind.RATE_LIMIT, consecutive_failures=4, now=now) == 300.0
    assert compute_cooldown(FailureKind.NETWORK, consecutive_failures=5, now=now) == 600.0
    assert compute_cooldown(FailureKind.NETWORK, consecutive_failures=10, now=now) == 600.0


def test_circuit_state_transitions(temp_db: KnowledgeDatabase) -> None:
    tracker = RouteHealthTracker(temp_db)
    route_id = "opencode"
    model_id = "free-model-1"
    t0 = 10000.0

    # 1. Initially CLOSED
    rec = tracker.get_health(route_id, model_id, now=t0)
    assert rec.state == CircuitState.CLOSED
    assert rec.consecutive_failures == 0
    assert rec.to_dict()["state"] == "closed"
    assert tracker.is_available(route_id, model_id, now=t0)
    assert tracker.acquire_probe(route_id, model_id, now=t0)

    # 2. First failure -> transitions to OPEN
    rec1 = tracker.record_failure(
        route_id,
        model_id,
        failure_kind=FailureKind.NETWORK,
        status_code=500,
        now=t0,
    )
    assert rec1.state == CircuitState.OPEN
    assert rec1.consecutive_failures == 1
    assert rec1.cooldown_until == t0 + 30.0
    assert not tracker.is_available(route_id, model_id, now=t0 + 10.0)
    assert not tracker.acquire_probe(route_id, model_id, now=t0 + 10.0)

    # 3. Before cooldown expires -> remains OPEN
    assert not tracker.is_available(route_id, model_id, now=t0 + 29.99)
    assert not tracker.acquire_probe(route_id, model_id, now=t0 + 29.99)

    # 4. At cooldown expiration -> transitions to HALF_OPEN when probe acquired
    t_probe = t0 + 30.0
    assert tracker.acquire_probe(route_id, model_id, now=t_probe)
    rec_half = tracker.get_health(route_id, model_id, now=t_probe)
    assert rec_half.state == CircuitState.HALF_OPEN

    # 5. Half-open probe succeeds -> transitions back to CLOSED
    rec_succ = tracker.record_success(route_id, model_id, now=t_probe + 1.0)
    assert rec_succ.state == CircuitState.CLOSED
    assert rec_succ.consecutive_failures == 0
    assert rec_succ.cooldown_until == 0.0
    assert tracker.is_available(route_id, model_id, now=t_probe + 1.0)


def test_healthy_fast_path_does_not_persist_redundant_success_rows(
    temp_db: KnowledgeDatabase,
) -> None:
    tracker = RouteHealthTracker(temp_db)

    assert tracker.acquire_probe("opencode", "healthy", now=1000.0)
    result = tracker.record_success("opencode", "healthy", now=1001.0)
    assert result.state == CircuitState.CLOSED

    with temp_db.connect() as con:
        count = con.execute(
            "SELECT COUNT(*) FROM route_health WHERE route_id = ? AND model_id = ?",
            ("opencode", "healthy"),
        ).fetchone()[0]
    assert count == 0


def test_half_open_failure_increases_cooldown(temp_db: KnowledgeDatabase) -> None:
    tracker = RouteHealthTracker(temp_db)
    route_id = "kilo"
    model_id = "qwen-32b"
    t0 = 20000.0

    # Initial failure
    tracker.record_failure(route_id, model_id, failure_kind=FailureKind.TIMEOUT, now=t0)
    assert tracker.get_health(route_id, model_id).cooldown_until == t0 + 30.0

    # Probe at t0 + 30.0
    assert tracker.acquire_probe(route_id, model_id, now=t0 + 30.0)
    # Probe fails! Consecutive failures becomes 2, next backoff is 60s
    rec_fail2 = tracker.record_failure(
        route_id,
        model_id,
        failure_kind=FailureKind.TIMEOUT,
        now=t0 + 31.0,
    )
    assert rec_fail2.state == CircuitState.OPEN
    assert rec_fail2.consecutive_failures == 2
    assert rec_fail2.cooldown_until == t0 + 31.0 + 60.0


def test_concurrent_half_open_single_probe(temp_db: KnowledgeDatabase) -> None:
    tracker = RouteHealthTracker(temp_db)
    route_id = "ddg"
    model_id = "gpt-4o-mini"
    t0 = 30000.0

    # Open circuit
    tracker.record_failure(route_id, model_id, failure_kind=FailureKind.RATE_LIMIT, now=t0)
    t_probe = t0 + 35.0

    def try_acquire() -> bool:
        return tracker.acquire_probe(route_id, model_id, now=t_probe)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(try_acquire) for _ in range(8)]
        results = [f.result() for f in futures]

    assert sum(results) == 1, "Exactly one caller must acquire the half-open probe"
    assert results.count(False) == 7

    assert tracker.get_health(route_id, model_id, now=t_probe).state == CircuitState.HALF_OPEN

    # Test release_probe without recording success/failure
    tracker.release_probe(route_id, model_id)
    # Now acquire_probe in HALF_OPEN should succeed once again
    assert tracker.acquire_probe(route_id, model_id, now=t_probe)
    # And second concurrent one fails
    assert not tracker.acquire_probe(route_id, model_id, now=t_probe)

    tracker.record_success(route_id, model_id, now=t_probe + 1.0)
    assert tracker.get_health(route_id, model_id).state == CircuitState.CLOSED


def test_route_model_isolation(temp_db: KnowledgeDatabase) -> None:
    tracker = RouteHealthTracker(temp_db)
    route1 = "opencode"
    route2 = "kilo"
    modelA = "model-a"
    modelB = "model-b"
    t0 = 40000.0

    tracker.record_failure(route1, modelA, failure_kind=FailureKind.RATE_LIMIT, now=t0)

    assert tracker.get_health(route1, modelA, now=t0).state == CircuitState.OPEN
    assert not tracker.is_available(route1, modelA, now=t0)

    assert tracker.get_health(route1, modelB, now=t0).state == CircuitState.CLOSED
    assert tracker.is_available(route1, modelB, now=t0)

    assert tracker.get_health(route2, modelA, now=t0).state == CircuitState.CLOSED
    assert tracker.is_available(route2, modelA, now=t0)


def test_persistence_across_restart(temp_db: KnowledgeDatabase) -> None:
    t0 = 50000.0
    route_id = "unclose"
    model_id = "gpt-3.5"

    tracker1 = RouteHealthTracker(temp_db)
    tracker1.record_failure(
        route_id,
        model_id,
        failure_kind=FailureKind.PROVIDER_INTERNAL,
        status_code=502,
        now=t0,
    )

    tracker2 = RouteHealthTracker(temp_db)
    rec = tracker2.get_health(route_id, model_id, now=t0 + 5.0)
    assert rec.state == CircuitState.OPEN
    assert rec.consecutive_failures == 1
    assert rec.cooldown_until == t0 + 30.0
    assert rec.last_status_code == 502
    assert rec.last_failure_kind == FailureKind.PROVIDER_INTERNAL.value
    assert not tracker2.is_available(route_id, model_id, now=t0 + 5.0)


def test_stale_compaction_does_not_touch_translation_memory(temp_db: KnowledgeDatabase) -> None:
    tracker = RouteHealthTracker(temp_db)
    t_now = 100000000.0

    with temp_db.connect() as con:
        with con:
            con.execute(
                """
                INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at)
                VALUES ('sc1', 'app1', 'pkg1', 'ja', 'vi', ?)
                """,
                (t_now,),
            )
            con.execute(
                """
                INSERT INTO lineages (lineage_id, scope_id, parent_lineage_id, created_at)
                VALUES ('lin1', 'sc1', NULL, ?)
                """,
                (t_now,),
            )
            con.execute(
                """
                INSERT INTO translation_segments (
                    segment_id, scope_id, lineage_id, marker, key, source_hash,
                    source_text, target_text, context_signature, hard_terms_hash, created_at, updated_at
                ) VALUES (
                    'seg1', 'sc1', 'lin1', 'm1', 'k1', 'h1', 'Hello', 'Xin chao', 'ctx1', 'hth1', ?, ?
                )
                """,
                (t_now, t_now),
            )

    seconds_per_day = 86400.0
    t_35_days_ago = t_now - (35 * seconds_per_day)
    t_10_days_ago = t_now - (10 * seconds_per_day)

    tracker.record_failure("route_old", "model_old", failure_kind=FailureKind.NETWORK, now=t_35_days_ago)
    tracker.record_failure("route_10d", "model_10d", failure_kind=FailureKind.NETWORK, now=t_10_days_ago)
    tracker.record_failure("route_fresh", "model_fresh", failure_kind=FailureKind.NETWORK, now=t_now)

    deleted = tracker.compact_stale_health(older_than_seconds=30 * seconds_per_day, now=t_now)
    assert deleted == 1

    rec_old = tracker.get_health("route_old", "model_old", now=t_now)
    assert rec_old.state == CircuitState.CLOSED
    assert rec_old.consecutive_failures == 0

    rec_10d = tracker.get_health("route_10d", "model_10d", now=t_now)
    assert rec_10d.consecutive_failures == 1
    rec_fresh = tracker.get_health("route_fresh", "model_fresh", now=t_now)
    assert rec_fresh.consecutive_failures == 1

    with temp_db.connect() as con:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM translation_segments WHERE segment_id = 'seg1'")
        count = cur.fetchone()[0]
        assert count == 1
