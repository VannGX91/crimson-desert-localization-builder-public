"""Specification tests for Router Scheduler, ranking, adaptive concurrency, bisection, and fallback (RTR-11)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TranslationBatchOutcome,
    TranslationBatchRequest,
    TranslationFailure,
    TranslationItemResult,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.job_control import CancellationToken
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.router.catalog import DiscoveredModel
from crimson0020.translation_runtime.router.health import CircuitState, RouteHealthTracker
from crimson0020.translation_runtime.router.ranking import (
    rank_candidates,
)
from crimson0020.translation_runtime.router.scheduler import (
    ConcurrencyManager,
    RoutePerformanceTracker,
    RouterScheduler,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> KnowledgeDatabase:
    db_file = tmp_path / "test_scheduler.sqlite3"
    return KnowledgeDatabase(database_path=db_file)


def make_item(work_id: str, text: str = "Test") -> TranslationRequestItem:
    return TranslationRequestItem(
        work_id=work_id,
        masked_text=text,
        context=(),
        context_signature="sig_" + work_id,
        hard_terms=(),
        soft_terms=(),
    )


class MockAdapter:
    """Configurable mock adapter for scheduler testing."""

    def __init__(
        self,
        route_id: str,
        model_id: str,
        outcomes: list[TranslationBatchOutcome] | None = None,
        side_effects: list[Any] | None = None,
    ) -> None:
        self.route_id = route_id
        self.model_id = model_id
        self.outcomes = list(outcomes or [])
        self.side_effects = list(side_effects or [])
        self.calls: list[TranslationBatchRequest] = []

    def translate_batch(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
        self.calls.append(batch)
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(batch)

        if self.outcomes:
            return self.outcomes.pop(0)

        # Default success
        successes = tuple(
            TranslationItemResult(
                work_id=item.work_id,
                translated_text=f"Translated: {item.masked_text}",
                route_id=self.route_id,
                model_id=self.model_id,
            )
            for item in batch.items
        )
        attempt = RouteAttempt(
            route_id=self.route_id,
            model_id=self.model_id,
            duration_ms=10.0,
        )
        return TranslationBatchOutcome(successes=successes, failures=(), attempts=(attempt,))


def test_priority_ranking_and_prohibited_routes() -> None:
    models = [
        DiscoveredModel(model_id="google/model-1", display_name="G1", route_id="google"),
        DiscoveredModel(model_id="mymemory/model-1", display_name="MM1", route_id="mymemory"),
        DiscoveredModel(model_id="aihorde/worker-1", display_name="Horde", route_id="aihorde", context_limit=4096),
        DiscoveredModel(model_id="kilo/auto-router", display_name="Kilo Auto", route_id="kilo_gateway", context_limit=8192),
        DiscoveredModel(model_id="kilo/qwen-32b", display_name="Kilo Qwen", route_id="kilo_gateway", context_limit=8192),
        DiscoveredModel(model_id="opencode/free-1", display_name="OpenCode", route_id="opencode", context_limit=8192),
        DiscoveredModel(model_id="duckduckgo/gpt-4o", display_name="DDG", route_id="duckduckgo", context_limit=8192),
    ]

    ranked = rank_candidates(models)
    route_ids = [m.route_id for m in ranked]

    # Prohibited routes are stripped
    assert "google" not in route_ids
    assert "mymemory" not in route_ids

    # Live-verified priority keeps OpenCode/Kilo first and moves working Horde
    # ahead of currently unavailable DuckDuckGo.
    assert ranked[0].route_id == "opencode"
    assert ranked[1].route_id == "kilo_gateway"
    assert ranked[2].route_id == "kilo_gateway"
    assert ranked[3].route_id == "aihorde"
    assert ranked[4].route_id == "duckduckgo"

    # Explicit Kilo models rank before Kilo meta routes
    assert ranked[1].model_id == "kilo/qwen-32b"
    assert ranked[2].model_id == "kilo/auto-router"


def test_later_route_succeeds_when_first_fails(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="opencode/m1", display_name="OpenCode", route_id="opencode")
    m2 = DiscoveredModel(model_id="kilo/m2", display_name="Kilo", route_id="kilo_gateway")

    # Adapter 1 fails with 500
    adapter1 = MockAdapter(
        "opencode",
        "opencode/m1",
        outcomes=[
            TranslationBatchOutcome(
                successes=(),
                failures=(
                    TranslationFailure(
                        work_id="w1",
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=True,
                        message="Server error 500",
                    ),
                ),
                attempts=(
                    RouteAttempt(
                        route_id="opencode",
                        model_id="opencode/m1",
                        duration_ms=20.0,
                        error_kind=FailureKind.PROVIDER_INTERNAL,
                        status_code=500,
                    ),
                ),
            )
        ],
    )
    # Adapter 2 succeeds
    adapter2 = MockAdapter("kilo_gateway", "kilo/m2")

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:opencode/m1": adapter1, "kilo_gateway:kilo/m2": adapter2},
    )

    batch = TranslationBatchRequest(
        job_id="b1",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[m1, m2])
    assert len(outcome.successes) == 1
    assert outcome.successes[0].work_id == "w1"
    assert outcome.successes[0].route_id == "kilo_gateway"
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].route_id == "opencode"
    assert outcome.attempts[1].route_id == "kilo_gateway"

    # Check health tracker updated: opencode is in cooldown, kilo is closed
    rec1 = health_tracker.get_health("opencode", "opencode/m1")
    assert rec1.state == CircuitState.OPEN
    rec2 = health_tracker.get_health("kilo_gateway", "kilo/m2")
    assert rec2.state == CircuitState.CLOSED


def test_partial_response_preservation(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")
    m2 = DiscoveredModel(model_id="m2", display_name="M2", route_id="kilo_gateway")

    # Route 1 succeeds for w1 but omits/fails w2
    adapter1 = MockAdapter(
        "opencode",
        "m1",
        outcomes=[
            TranslationBatchOutcome(
                successes=(
                    TranslationItemResult(
                        work_id="w1",
                        translated_text="Xin chao",
                        route_id="opencode",
                        model_id="m1",
                    ),
                ),
                failures=(
                    TranslationFailure(
                        work_id="w2",
                        kind=FailureKind.TIMEOUT,
                        retryable=True,
                        message="Item timed out",
                    ),
                ),
                attempts=(
                    RouteAttempt(
                        route_id="opencode",
                        model_id="m1",
                        duration_ms=50.0,
                    ),
                ),
            )
        ],
    )
    # Route 2 only gets w2
    adapter2 = MockAdapter("kilo_gateway", "m2")

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:m1": adapter1, "kilo_gateway:m2": adapter2},
    )

    batch = TranslationBatchRequest(
        job_id="b2",
        items=(make_item("w1", "Hello"), make_item("w2", "World")),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[m1, m2])
    # Both items succeed in the end
    assert len(outcome.successes) == 2
    res_map = {s.work_id: s for s in outcome.successes}
    assert res_map["w1"].route_id == "opencode"
    assert res_map["w2"].route_id == "kilo_gateway"

    # Adapter 2 was called with only w2 (w1 was not resent!)
    assert len(adapter2.calls) == 1
    assert [i.work_id for i in adapter2.calls[0].items] == ["w2"]


def test_bisect_malformed_batch(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")

    def custom_handler(req: TranslationBatchRequest) -> TranslationBatchOutcome:
        ids = [i.work_id for i in req.items]
        if "w3" in ids and len(ids) > 1:
            return TranslationBatchOutcome(
                successes=(),
                failures=tuple(
                    TranslationFailure(
                        work_id=wid,
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=True,
                        message="Malformed batch JSON",
                    )
                    for wid in ids
                ),
                attempts=(
                    RouteAttempt(
                        route_id="opencode",
                        model_id="m1",
                        duration_ms=10.0,
                        error_kind=FailureKind.INVALID_STRUCTURE,
                    ),
                ),
            )
        if ids == ["w3"]:
            return TranslationBatchOutcome(
                successes=(),
                failures=(
                    TranslationFailure(
                        work_id="w3",
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=False,
                        message="Malformed single item",
                    ),
                ),
                attempts=(
                    RouteAttempt(
                        route_id="opencode",
                        model_id="m1",
                        duration_ms=10.0,
                        error_kind=FailureKind.INVALID_STRUCTURE,
                    ),
                ),
            )
        # Sibling rows succeed
        return TranslationBatchOutcome(
            successes=tuple(
                TranslationItemResult(
                    work_id=item.work_id,
                    translated_text=f"Tr: {item.masked_text}",
                    route_id="opencode",
                    model_id="m1",
                )
                for item in req.items
            ),
            failures=(),
            attempts=(
                RouteAttempt(
                    route_id="opencode",
                    model_id="m1",
                    duration_ms=10.0,
                ),
            ),
        )

    adapter1 = MockAdapter("opencode", "m1", side_effects=[custom_handler] * 10)
    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:m1": adapter1},
    )

    batch = TranslationBatchRequest(
        job_id="b_bisect",
        items=(
            make_item("w1", "One"),
            make_item("w2", "Two"),
            make_item("w3", "Bad"),
            make_item("w4", "Four"),
        ),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[m1])
    success_ids = {s.work_id for s in outcome.successes}
    failure_ids = {f.work_id for f in outcome.failures}

    # w1, w2, and w4 were saved by bisection! Only w3 failed.
    assert success_ids == {"w1", "w2", "w4"}
    assert failure_ids == {"w3"}


def test_attempt_and_time_budget_exhaustion(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    # Create 8 candidate models
    candidates = [
        DiscoveredModel(model_id=f"m{i}", display_name=f"M{i}", route_id="opencode")
        for i in range(8)
    ]

    adapters = {}
    for c in candidates:
        adapters[f"{c.route_id}:{c.model_id}"] = MockAdapter(
            c.route_id,
            c.model_id,
            outcomes=[
                TranslationBatchOutcome(
                    successes=(),
                    failures=(
                        TranslationFailure(
                            work_id="w1",
                            kind=FailureKind.PROVIDER_INTERNAL,
                            retryable=True,
                            message="Temporary failure",
                        ),
                    ),
                    attempts=(
                        RouteAttempt(
                            route_id=c.route_id,
                            model_id=c.model_id,
                            duration_ms=5.0,
                            error_kind=FailureKind.PROVIDER_INTERNAL,
                        ),
                    ),
                )
            ],
        )

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters=adapters,
        max_attempts_per_batch=6,
        max_attempts_per_route=6,
    )

    batch = TranslationBatchRequest(
        job_id="b_budget",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=candidates)
    assert len(outcome.successes) == 0
    assert len(outcome.failures) == 1
    assert outcome.failures[0].work_id == "w1"
    # Max attempts capped at 6 even though 8 candidates existed
    assert len(outcome.attempts) == 6


def test_pause_and_cancellation(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")
    m2 = DiscoveredModel(model_id="m2", display_name="M2", route_id="kilo_gateway")

    cancel_token = CancellationToken()

    # When first adapter runs, it requests pause/cancellation
    def pause_side_effect(req: TranslationBatchRequest) -> TranslationBatchOutcome:
        cancel_token.cancel("User paused job")
        return TranslationBatchOutcome(
            successes=(),
            failures=(
                TranslationFailure(
                    work_id="w1",
                    kind=FailureKind.NETWORK,
                    retryable=True,
                    message="Network dropped",
                ),
            ),
            attempts=(
                RouteAttempt(
                    route_id="opencode",
                    model_id="m1",
                    duration_ms=10.0,
                    error_kind=FailureKind.NETWORK,
                ),
            ),
        )

    adapter1 = MockAdapter("opencode", "m1", side_effects=[pause_side_effect])
    adapter2 = MockAdapter("kilo_gateway", "m2")

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:m1": adapter1, "kilo_gateway:m2": adapter2},
    )

    batch = TranslationBatchRequest(
        job_id="b_pause",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(
        batch,
        candidates=[m1, m2],
        cancellation_token=cancel_token,
    )

    # Scheduler halts immediately after pause token is triggered
    assert len(adapter2.calls) == 0
    assert len(outcome.successes) == 0
    assert len(outcome.failures) == 1
    assert outcome.failures[0].kind == FailureKind.CANCELLATION


def test_no_checkpointed_retry_and_restart(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")
    adapter1 = MockAdapter("opencode", "m1")

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:m1": adapter1},
    )

    batch = TranslationBatchRequest(
        job_id="b_ckpt",
        items=(make_item("w1", "Already done"), make_item("w2", "Needs translation")),
        source_language="en",
        target_language="vi",
    )

    # w1 is already in checkpoint
    outcome = scheduler.schedule_batch(
        batch,
        candidates=[m1],
        completed_work_ids={"w1"},
    )

    # Adapter only saw w2
    assert len(adapter1.calls) == 1
    assert [i.work_id for i in adapter1.calls[0].items] == ["w2"]
    assert len(outcome.successes) == 1
    assert outcome.successes[0].work_id == "w2"


def test_adaptive_concurrency() -> None:
    cm = ConcurrencyManager(global_limit=2, default_route_limit=1)

    route = "opencode"
    # Route limit starts at 1
    assert cm.get_route_limit(route) == 1

    # Simulate 19 successes -> still 1
    for _ in range(19):
        cm.record_success(route)
    assert cm.get_route_limit(route) == 1

    # 20th success -> increases to 2
    cm.record_success(route)
    assert cm.get_route_limit(route) == 2

    # 429 rate limit reduces concurrency back to 1
    cm.record_rate_limit(route)
    assert cm.get_route_limit(route) == 1

    for _ in range(20):
        cm.record_success(route)
    assert cm.get_route_limit(route) == 2
    cm.record_failure(route)
    assert cm.get_route_limit(route) == 1


def test_adaptive_concurrency_can_limit_scaling_to_verified_routes() -> None:
    cm = ConcurrencyManager(
        global_limit=4,
        default_route_limit=1,
        max_route_limit=2,
        scalable_routes=frozenset({"opencode"}),
    )
    for _ in range(20):
        cm.record_success("opencode")
        cm.record_success("llm7")

    assert cm.get_route_limit("opencode") == 2
    assert cm.get_route_limit("llm7") == 1


def test_concurrency_bulkhead_enforces_global_and_per_route_limits() -> None:
    cm = ConcurrencyManager(global_limit=2, default_route_limit=1, max_route_limit=1)

    assert cm.try_acquire("route-a") is True
    assert cm.get_in_flight() == 1
    assert cm.get_in_flight("route-a") == 1
    assert cm.try_acquire("route-a") is False

    assert cm.try_acquire("route-b") is True
    assert cm.get_in_flight() == 2
    assert cm.try_acquire("route-c") is False

    cm.release("route-a")
    assert cm.get_in_flight() == 1
    assert cm.try_acquire("route-c") is True
    assert cm.get_in_flight() == 2

    cm.release("route-b")
    cm.release("route-c")
    assert cm.get_in_flight() == 0


def test_all_routes_fail_exhaustion(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")
    adapter1 = MockAdapter(
        "opencode",
        "m1",
        outcomes=[
            TranslationBatchOutcome(
                successes=(),
                failures=(
                    TranslationFailure(
                        work_id="w1",
                        kind=FailureKind.NETWORK,
                        retryable=True,
                        message="Connection refused",
                    ),
                ),
                attempts=(
                    RouteAttempt(
                        route_id="opencode",
                        model_id="m1",
                        duration_ms=10.0,
                        error_kind=FailureKind.NETWORK,
                    ),
                ),
            )
        ],
    )

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:m1": adapter1},
    )

    batch = TranslationBatchRequest(
        job_id="b_all_fail",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[m1])
    assert len(outcome.successes) == 0
    assert len(outcome.failures) == 1
    assert outcome.failures[0].work_id == "w1"
    assert outcome.failures[0].kind == FailureKind.NETWORK


def test_ranking_edge_cases() -> None:
    models = [
        DiscoveredModel(model_id="opencode/auto", display_name="Auto", route_id="opencode"),
        DiscoveredModel(model_id="custom/m1", display_name="Custom", route_id="unknown_provider"),
    ]
    ranked = rank_candidates(models, prefer_explicit_kilo=True)
    assert ranked[0].route_id == "opencode"
    assert ranked[1].route_id == "unknown_provider"

    ranked2 = rank_candidates(models, prefer_explicit_kilo=False)
    assert len(ranked2) == 2


def test_ranking_preserves_discovery_order_for_exact_ties() -> None:
    models = [
        DiscoveredModel(model_id="preferred-first", display_name="First", route_id="duckduckgo"),
        DiscoveredModel(model_id="alphabetically-earlier", display_name="Second", route_id="duckduckgo"),
    ]

    ranked = rank_candidates(models)

    assert [model.model_id for model in ranked] == ["preferred-first", "alphabetically-earlier"]


def test_empty_and_all_completed_batches(temp_db: KnowledgeDatabase) -> None:
    scheduler = RouterScheduler(health_tracker=RouteHealthTracker(temp_db))

    # Empty batch items
    b_empty = TranslationBatchRequest(
        job_id="b_empty",
        items=(),
        source_language="en",
        target_language="vi",
    )
    res_empty = scheduler.schedule_batch(b_empty)
    assert res_empty.successes == ()
    assert res_empty.failures == ()

    # All items already in completed
    b_full = TranslationBatchRequest(
        job_id="b_full",
        items=(make_item("w1"),),
        source_language="en",
        target_language="vi",
    )
    res_completed = scheduler.schedule_batch(b_full, completed_work_ids={"w1"})
    assert res_completed.successes == ()
    assert res_completed.failures == ()


def test_dynamic_discovery_service_integration(temp_db: KnowledgeDatabase) -> None:
    from crimson0020.translation_runtime.router.discovery import ModelDiscoveryService
    from crimson0020.translation_runtime.router.transport import SafeHttpTransport

    transport = SafeHttpTransport(injected_transport=lambda u, h, b, t: (200, {}, b'{"data": []}'))
    discovery = ModelDiscoveryService(database=temp_db, transport=transport)

    health_tracker = RouteHealthTracker(temp_db)
    unavailable = {}
    for route_id, model_id in (
        ("llm7", "default"),
        ("blockrun", "nvidia/gpt-oss-20b"),
        ("cehpoint", "cehpoint-ai"),
        ("vireonix", "auto"),
    ):
        unavailable[f"{route_id}:{model_id}"] = MockAdapter(
            route_id,
            model_id,
            outcomes=[TranslationBatchOutcome(
                successes=(),
                failures=(TranslationFailure(
                    work_id="w1",
                    kind=FailureKind.PROVIDER_INTERNAL,
                    retryable=True,
                    message="synthetic unavailable",
                ),),
                attempts=(RouteAttempt(
                    route_id=route_id,
                    model_id=model_id,
                    duration_ms=1.0,
                    error_kind=FailureKind.PROVIDER_INTERNAL,
                    status_code=500,
                ),),
            )],
        )
    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        discovery_service=discovery,
        adapters={
            **unavailable,
            "duckduckgo:gpt-4o-mini": MockAdapter("duckduckgo", "gpt-4o-mini"),
        },
    )

    # Static routes like duckduckgo are discovered by discovery_service
    batch = TranslationBatchRequest(
        job_id="b_disc",
        items=(make_item("w1", "Hi"),),
        source_language="en",
        target_language="vi",
    )
    outcome = scheduler.schedule_batch(batch, candidates=None)
    assert len(outcome.successes) == 1
    assert outcome.successes[0].route_id == "duckduckgo"


def test_discovery_candidates_are_cached_between_hot_path_batches() -> None:
    class CountingDiscovery:
        def __init__(self) -> None:
            self.calls = 0

        def discover_route_models(self, _route_id: str):
            self.calls += 1
            return []

    discovery = CountingDiscovery()
    scheduler = RouterScheduler(
        discovery_service=discovery,
        candidate_cache_ttl_seconds=120.0,
    )
    batch = TranslationBatchRequest(
        job_id="cached-discovery",
        items=(make_item("w1", "hello"),),
        source_language="en",
        target_language="vi",
    )

    scheduler.schedule_batch(batch, now=1000.0)
    first_calls = discovery.calls
    scheduler.schedule_batch(batch, now=1001.0)

    assert first_calls > 0
    assert discovery.calls == first_calls


def test_adapter_instantiation_and_factory(temp_db: KnowledgeDatabase) -> None:
    scheduler = RouterScheduler()
    # Adapter factory
    custom_adapter = MockAdapter("test_route", "test_m")
    scheduler.adapter_factory = lambda r, m: custom_adapter if r == "test_route" else None
    assert scheduler._get_adapter("test_route", "test_m") is custom_adapter

    # Concrete adapters instantiation paths
    assert scheduler._get_adapter("opencode", "m1") is not None
    assert scheduler._get_adapter("kilo_gateway", "m1") is not None
    assert scheduler._get_adapter("kilo_anonymous", "m1") is not None
    assert scheduler._get_adapter("unclose", "m1") is not None
    assert scheduler._get_adapter("duckduckgo", "m1") is not None
    assert scheduler._get_adapter("aihorde", "m1") is not None
    assert scheduler._get_adapter("llm7", "default") is not None
    assert scheduler._get_adapter("blockrun", "nvidia/gpt-oss-20b") is not None
    assert scheduler._get_adapter("vireonix", "auto") is not None
    assert scheduler._get_adapter("cehpoint", "cehpoint-ai") is not None
    # Cloudflare adapter instantiation
    cf_adapter = scheduler._get_adapter("cloudflare", "m1")
    assert cf_adapter is not None

    # Missing/unknown adapter
    assert scheduler._get_adapter("completely_unknown_route", "m1") is None


def test_concrete_adapter_instances_are_reused() -> None:
    scheduler = RouterScheduler()

    first = scheduler._get_adapter("llm7", "default")
    second = scheduler._get_adapter("llm7", "default")

    assert first is not None
    assert second is first


def test_dynamic_discovery_overlaps_route_lookups() -> None:
    class SlowDiscovery:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def discover_route_models(self, _route_id: str):
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self._lock:
                self.active -= 1
            return []

    discovery = SlowDiscovery()
    scheduler = RouterScheduler(discovery_service=discovery)
    batch = TranslationBatchRequest(
        job_id="parallel-discovery",
        items=(make_item("w1", "hello"),),
        source_language="en",
        target_language="vi",
    )

    scheduler.schedule_batch(batch)

    assert discovery.max_active > 1


def test_runtime_performance_deprioritizes_recent_failure() -> None:
    fast = DiscoveredModel(model_id="default", display_name="LLM7", route_id="llm7")
    fallback = DiscoveredModel(model_id="m1", display_name="OpenCode", route_id="opencode")
    fast_adapter = MockAdapter(
        "llm7",
        "default",
        outcomes=[
            TranslationBatchOutcome(
                successes=(),
                failures=(
                    TranslationFailure(
                        work_id="w1",
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=True,
                        message="temporary model failure",
                    ),
                ),
                attempts=(
                    RouteAttempt(
                        route_id="llm7",
                        model_id="default",
                        duration_ms=1200.0,
                        error_kind=FailureKind.PROVIDER_INTERNAL,
                        status_code=500,
                    ),
                ),
            )
        ],
    )
    fallback_adapter = MockAdapter("opencode", "m1")
    scheduler = RouterScheduler(
        adapters={
            "llm7:default": fast_adapter,
            "opencode:m1": fallback_adapter,
        },
    )
    scheduler.performance_tracker = RoutePerformanceTracker(
        success_alpha=1.0,
        latency_alpha=1.0,
        warmup_samples=1,
    )
    scheduler.performance_tracker.record(
        "llm7",
        "default",
        success=True,
        latency_ms=100.0,
    )
    batch = TranslationBatchRequest(
        job_id="adaptive-routing",
        items=(make_item("w1", "hello"),),
        source_language="en",
        target_language="vi",
    )

    first = scheduler.schedule_batch(batch, candidates=[fast, fallback])
    second = scheduler.schedule_batch(batch, candidates=[fast, fallback])

    assert first.successes[0].route_id == "opencode"
    assert second.successes[0].route_id == "opencode"
    assert len(fast_adapter.calls) == 1
    assert len(fallback_adapter.calls) == 2


def test_structural_bisection_has_shared_attempt_budget() -> None:
    model = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")

    def always_structural(req: TranslationBatchRequest) -> TranslationBatchOutcome:
        return TranslationBatchOutcome(
            successes=(),
            failures=tuple(
                TranslationFailure(
                    work_id=item.work_id,
                    kind=FailureKind.INVALID_STRUCTURE,
                    retryable=True,
                    message="malformed batch",
                )
                for item in req.items
            ),
            attempts=(
                RouteAttempt(
                    route_id="opencode",
                    model_id="m1",
                    duration_ms=10.0,
                    error_kind=FailureKind.INVALID_STRUCTURE,
                    status_code=200,
                ),
            ),
        )

    adapter = MockAdapter("opencode", "m1", side_effects=[always_structural] * 100)
    scheduler = RouterScheduler(
        adapters={"opencode:m1": adapter},
        max_bisection_attempts=6,
    )
    batch = TranslationBatchRequest(
        job_id="bounded-bisect",
        items=tuple(make_item(f"w{i}", f"row {i}") for i in range(32)),
        source_language="en",
        target_language="vi",
    )

    scheduler.schedule_batch(batch, candidates=[model])

    # One root request plus at most six bounded recovery requests.
    assert len(adapter.calls) <= 7


def test_bootstrap_ranking_prefers_opencode_before_llm7_when_no_performance_samples() -> None:
    opencode = DiscoveredModel(
        model_id="ling-3.0-flash-fin-free",
        display_name="OpenCode",
        route_id="opencode",
    )
    llm7 = DiscoveredModel(
        model_id="default",
        display_name="LLM7",
        route_id="llm7",
    )
    scheduler = RouterScheduler()

    ranked = scheduler._rank_for_execution([llm7, opencode])

    assert [model.route_id for model in ranked] == ["opencode", "llm7"]


def test_bootstrap_ranking_prefers_live_productive_fallbacks_before_dead_routes() -> None:
    candidates = [
        DiscoveredModel(model_id="m1", display_name="Unclose", route_id="unclose"),
        DiscoveredModel(model_id="m2", display_name="BlockRun", route_id="blockrun"),
        DiscoveredModel(model_id="m3", display_name="Cehpoint", route_id="cehpoint"),
        DiscoveredModel(model_id="m4", display_name="Vireonix", route_id="vireonix"),
        DiscoveredModel(model_id="m5", display_name="Kilo", route_id="kilo_gateway"),
        DiscoveredModel(model_id="m6", display_name="Horde", route_id="aihorde"),
    ]
    scheduler = RouterScheduler()

    ranked = scheduler._rank_for_execution(candidates)

    assert [model.route_id for model in ranked] == [
        "unclose",
        "kilo_gateway",
        "aihorde",
        "blockrun",
        "vireonix",
        "cehpoint",
    ]


def test_four_parallel_batches_claim_distinct_route_lanes_without_waiting() -> None:
    barrier = threading.Barrier(4)

    class BarrierAdapter(MockAdapter):
        def translate_batch(self, batch: TranslationBatchRequest) -> TranslationBatchOutcome:
            barrier.wait(timeout=2.0)
            return super().translate_batch(batch)

    candidates = [
        DiscoveredModel(model_id="open", display_name="OpenCode", route_id="opencode"),
        DiscoveredModel(model_id="default", display_name="LLM7", route_id="llm7"),
        DiscoveredModel(model_id="block", display_name="BlockRun", route_id="blockrun"),
        DiscoveredModel(model_id="auto", display_name="Vireonix", route_id="vireonix"),
    ]
    adapters = {
        "opencode:open": BarrierAdapter("opencode", "open"),
        "llm7:default": BarrierAdapter("llm7", "default"),
        "blockrun:block": BarrierAdapter("blockrun", "block"),
        "vireonix:auto": BarrierAdapter("vireonix", "auto"),
    }
    scheduler = RouterScheduler(
        adapters=adapters,
        concurrency_manager=ConcurrencyManager(
            global_limit=4,
            default_route_limit=1,
            max_route_limit=1,
        ),
        max_attempts_per_batch=6,
        max_attempts_per_route=2,
        primary_route_wait_seconds=0.0,
    )
    outcomes: list[TranslationBatchOutcome] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        try:
            outcome = scheduler.schedule_batch(
                TranslationBatchRequest(
                    job_id=f"lane-{index}",
                    items=(make_item(f"w{index}", f"row {index}"),),
                    source_language="en",
                    target_language="vi",
                ),
                candidates=candidates,
            )
            with lock:
                outcomes.append(outcome)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(outcomes) == 4
    assert {outcome.successes[0].route_id for outcome in outcomes} == {
        "opencode",
        "llm7",
        "blockrun",
        "vireonix",
    }


def test_route_sized_attempts_spread_one_logical_batch_across_routes() -> None:
    blockrun = DiscoveredModel(
        model_id="block",
        display_name="BlockRun",
        route_id="blockrun",
    )
    vireonix = DiscoveredModel(
        model_id="auto",
        display_name="Vireonix",
        route_id="vireonix",
    )
    blockrun_adapter = MockAdapter("blockrun", "block")
    blockrun_adapter.scheduler_batch_limits = (32, 6_000)
    vireonix_adapter = MockAdapter("vireonix", "auto")
    vireonix_adapter.scheduler_batch_limits = (16, 4_000)
    scheduler = RouterScheduler(
        adapters={
            "blockrun:block": blockrun_adapter,
            "vireonix:auto": vireonix_adapter,
        },
        max_attempts_per_batch=4,
        max_attempts_per_route=1,
    )
    batch = TranslationBatchRequest(
        job_id="route-sized",
        items=tuple(make_item(f"w{i}", f"row {i}") for i in range(48)),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[blockrun, vireonix])

    assert len(outcome.successes) == 48
    assert outcome.failures == ()
    assert [len(call.items) for call in blockrun_adapter.calls] == [32]
    assert [len(call.items) for call in vireonix_adapter.calls] == [16]


def test_successful_route_slice_is_recorded_as_health_success(
    temp_db: KnowledgeDatabase,
) -> None:
    blockrun = DiscoveredModel(
        model_id="block",
        display_name="BlockRun",
        route_id="blockrun",
    )
    vireonix = DiscoveredModel(
        model_id="auto",
        display_name="Vireonix",
        route_id="vireonix",
    )
    blockrun_adapter = MockAdapter("blockrun", "block")
    blockrun_adapter.scheduler_batch_limits = (1, 6_000)
    scheduler = RouterScheduler(
        health_tracker=RouteHealthTracker(temp_db),
        adapters={
            "blockrun:block": blockrun_adapter,
            "vireonix:auto": MockAdapter("vireonix", "auto"),
        },
        max_attempts_per_route=1,
    )
    batch = TranslationBatchRequest(
        job_id="slice-health",
        items=(make_item("w1"), make_item("w2")),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[blockrun, vireonix])

    assert len(outcome.successes) == 2
    health = scheduler.health_tracker.get_health("blockrun", "block")
    assert health.state == CircuitState.CLOSED
    assert health.consecutive_failures == 0


def test_structural_http_200_failure_does_not_open_route_circuit(
    temp_db: KnowledgeDatabase,
) -> None:
    model = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")

    def malformed(req: TranslationBatchRequest) -> TranslationBatchOutcome:
        return TranslationBatchOutcome(
            successes=(),
            failures=tuple(
                TranslationFailure(
                    work_id=item.work_id,
                    kind=FailureKind.INVALID_STRUCTURE,
                    retryable=False,
                    message="malformed response",
                )
                for item in req.items
            ),
            attempts=(
                RouteAttempt(
                    route_id="opencode",
                    model_id="m1",
                    duration_ms=10.0,
                    error_kind=FailureKind.INVALID_STRUCTURE,
                    status_code=200,
                ),
            ),
        )

    scheduler = RouterScheduler(
        health_tracker=RouteHealthTracker(temp_db),
        adapters={"opencode:m1": MockAdapter("opencode", "m1", side_effects=[malformed])},
        max_bisection_attempts=0,
    )
    scheduler.schedule_batch(
        TranslationBatchRequest(
            job_id="structural-health",
            items=(make_item("w1"),),
            source_language="en",
            target_language="vi",
        ),
        candidates=[model],
    )

    health = scheduler.health_tracker.get_health("opencode", "m1")
    assert health.state == CircuitState.CLOSED
    assert health.consecutive_failures == 0


def test_full_structural_miss_moves_to_another_route_without_same_route_repeat() -> None:
    first = DiscoveredModel(model_id="h1", display_name="H1", route_id="aihorde")
    second = DiscoveredModel(model_id="h2", display_name="H2", route_id="aihorde")
    fallback = DiscoveredModel(model_id="c1", display_name="C1", route_id="cehpoint")

    def malformed(req: TranslationBatchRequest) -> TranslationBatchOutcome:
        return TranslationBatchOutcome(
            successes=(),
            failures=tuple(
                TranslationFailure(
                    work_id=item.work_id,
                    kind=FailureKind.MISSING_ITEM,
                    retryable=True,
                    message="missing all items",
                )
                for item in req.items
            ),
            attempts=(
                RouteAttempt(
                    route_id="aihorde",
                    model_id="h1",
                    duration_ms=15_000.0,
                    error_kind=FailureKind.INVALID_STRUCTURE,
                    status_code=200,
                ),
            ),
        )

    first_adapter = MockAdapter("aihorde", "h1", side_effects=[malformed])
    first_adapter.allow_structural_bisection = False
    second_adapter = MockAdapter("aihorde", "h2")
    fallback_adapter = MockAdapter("cehpoint", "c1")
    scheduler = RouterScheduler(
        adapters={
            "aihorde:h1": first_adapter,
            "aihorde:h2": second_adapter,
            "cehpoint:c1": fallback_adapter,
        },
        max_attempts_per_batch=4,
        max_attempts_per_route=2,
    )
    outcome = scheduler.schedule_batch(
        TranslationBatchRequest(
            job_id="structural-route-hop",
            items=(make_item("w1"), make_item("w2")),
            source_language="en",
            target_language="vi",
        ),
        candidates=[first, second, fallback],
    )

    assert len(first_adapter.calls) == 1
    assert len(second_adapter.calls) == 0
    assert len(fallback_adapter.calls) == 1
    assert len(outcome.successes) == 2


def test_busy_primary_route_is_requeued_before_slow_fallback() -> None:
    opencode = DiscoveredModel(
        model_id="ling-3.0-flash-fin-free",
        display_name="OpenCode",
        route_id="opencode",
    )
    llm7 = DiscoveredModel(
        model_id="default",
        display_name="LLM7",
        route_id="llm7",
    )
    slow = DiscoveredModel(
        model_id="slow",
        display_name="Slow fallback",
        route_id="unclose",
    )
    llm7_adapter = MockAdapter(
        "llm7",
        "default",
        outcomes=[
            TranslationBatchOutcome(
                successes=(),
                failures=(
                    TranslationFailure(
                        work_id="w1",
                        kind=FailureKind.RATE_LIMIT,
                        retryable=True,
                        message="rate limited",
                        retry_after_seconds=30.0,
                    ),
                ),
                attempts=(
                    RouteAttempt(
                        route_id="llm7",
                        model_id="default",
                        duration_ms=1.0,
                        error_kind=FailureKind.RATE_LIMIT,
                        status_code=429,
                    ),
                ),
            )
        ],
    )
    opencode_adapter = MockAdapter("opencode", "ling-3.0-flash-fin-free")
    slow_adapter = MockAdapter("unclose", "slow")
    concurrency = ConcurrencyManager(global_limit=2, default_route_limit=1, max_route_limit=1)
    assert concurrency.try_acquire("opencode") is True
    scheduler = RouterScheduler(
        adapters={
            "opencode:ling-3.0-flash-fin-free": opencode_adapter,
            "llm7:default": llm7_adapter,
            "unclose:slow": slow_adapter,
        },
        concurrency_manager=concurrency,
        primary_route_wait_seconds=0.5,
    )
    timer = threading.Timer(0.05, lambda: concurrency.release("opencode"))
    timer.start()
    try:
        outcome = scheduler.schedule_batch(
            TranslationBatchRequest(
                job_id="requeue-primary",
                items=(make_item("w1", "hello"),),
                source_language="en",
                target_language="vi",
            ),
            candidates=[opencode, llm7, slow],
        )
    finally:
        timer.join(timeout=1.0)

    assert outcome.successes[0].route_id == "opencode"
    assert len(llm7_adapter.calls) == 1
    assert len(opencode_adapter.calls) == 1
    assert slow_adapter.calls == []


def test_time_budget_exhaustion_and_skipped_cooldown(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")
    m2 = DiscoveredModel(model_id="m2", display_name="M2", route_id="kilo_gateway")

    # Put m1 in cooldown
    health_tracker.record_failure("opencode", "m1", failure_kind=FailureKind.RATE_LIMIT, now=1000.0)

    # Scheduler with max_batch_duration_seconds=0.0
    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"kilo_gateway:m2": MockAdapter("kilo_gateway", "m2")},
        max_batch_duration_seconds=0.0,
    )

    batch = TranslationBatchRequest(
        job_id="b_timeout",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[m1, m2], now=1005.0)
    # m1 is skipped because in cooldown, and then duration expired before m2
    assert len(outcome.successes) == 0
    assert len(outcome.failures) == 1
    assert outcome.failures[0].kind == FailureKind.PROVIDER_INTERNAL


def test_adapter_exception_in_main_and_bisection(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    m1 = DiscoveredModel(model_id="m1", display_name="M1", route_id="opencode")

    call_count = [0]

    def exploding_handler(req: TranslationBatchRequest) -> TranslationBatchOutcome:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call returns structural error to trigger bisection
            return TranslationBatchOutcome(
                successes=(),
                failures=(
                    TranslationFailure(
                        work_id="w1",
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=True,
                        message="Bad batch",
                    ),
                    TranslationFailure(
                        work_id="w2",
                        kind=FailureKind.INVALID_STRUCTURE,
                        retryable=True,
                        message="Bad batch",
                    ),
                ),
                attempts=(),
            )
        # Inside bisection call, raise unexpected exception
        raise RuntimeError("Unexpected adapter crash")

    adapter = MockAdapter("opencode", "m1", side_effects=[exploding_handler] * 5)
    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={"opencode:m1": adapter},
    )

    batch = TranslationBatchRequest(
        job_id="b_crash",
        items=(make_item("w1", "One"), make_item("w2", "Two")),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[m1])
    assert len(outcome.successes) == 0
    assert len(outcome.failures) == 2
    assert outcome.failures[0].kind == FailureKind.INVALID_STRUCTURE



def test_provider_attempt_budget_falls_back_to_next_route_quickly(temp_db: KnowledgeDatabase) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    op_models = [
        DiscoveredModel(model_id=f"oc{i}", display_name=f"OC{i}", route_id="opencode")
        for i in range(4)
    ]
    kilo = DiscoveredModel(model_id="kilo-ok", display_name="Kilo", route_id="kilo_gateway")

    adapters = {}
    for model in op_models:
        adapters[f"opencode:{model.model_id}"] = MockAdapter(
            "opencode",
            model.model_id,
            outcomes=[TranslationBatchOutcome(
                successes=(),
                failures=(TranslationFailure(
                    work_id="w1",
                    kind=FailureKind.PROVIDER_INTERNAL,
                    retryable=True,
                    message="provider failure",
                ),),
                attempts=(RouteAttempt(
                    route_id="opencode",
                    model_id=model.model_id,
                    duration_ms=5.0,
                    error_kind=FailureKind.PROVIDER_INTERNAL,
                    status_code=500,
                ),),
            )],
        )
    adapters["kilo_gateway:kilo-ok"] = MockAdapter("kilo_gateway", "kilo-ok")

    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters=adapters,
        max_attempts_per_batch=6,
    )
    batch = TranslationBatchRequest(
        job_id="fast_provider_fallback",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[*op_models, kilo])

    assert len(outcome.successes) == 1
    assert outcome.successes[0].route_id == "kilo_gateway"
    assert [attempt.route_id for attempt in outcome.attempts] == [
        "opencode", "kilo_gateway"
    ]
    assert len(adapters["opencode:oc1"].calls) == 0
    assert len(adapters["opencode:oc2"].calls) == 0
    assert len(adapters["opencode:oc3"].calls) == 0


@pytest.mark.parametrize(
    ("failure_kind", "status_code"),
    [
        (FailureKind.RATE_LIMIT, 429),
        (FailureKind.TIMEOUT, None),
        (FailureKind.NETWORK, None),
        (FailureKind.UNAUTHORIZED, 403),
    ],
)
def test_route_wide_failure_skips_remaining_models_and_falls_back_immediately(
    temp_db: KnowledgeDatabase,
    failure_kind: FailureKind,
    status_code: int | None,
) -> None:
    health_tracker = RouteHealthTracker(temp_db)
    first = DiscoveredModel(model_id="oc-first", display_name="OC first", route_id="opencode")
    second = DiscoveredModel(model_id="oc-second", display_name="OC second", route_id="opencode")
    kilo = DiscoveredModel(model_id="kilo-ok", display_name="Kilo", route_id="kilo_gateway")

    first_adapter = MockAdapter(
        "opencode",
        "oc-first",
        outcomes=[TranslationBatchOutcome(
            successes=(),
            failures=(TranslationFailure(
                work_id="w1",
                kind=failure_kind,
                retryable=failure_kind != FailureKind.UNAUTHORIZED,
                message="route-wide failure",
            ),),
            attempts=(RouteAttempt(
                route_id="opencode",
                model_id="oc-first",
                duration_ms=5.0,
                error_kind=failure_kind,
                status_code=status_code,
            ),),
        )],
    )
    second_adapter = MockAdapter("opencode", "oc-second")
    kilo_adapter = MockAdapter("kilo_gateway", "kilo-ok")
    scheduler = RouterScheduler(
        health_tracker=health_tracker,
        adapters={
            "opencode:oc-first": first_adapter,
            "opencode:oc-second": second_adapter,
            "kilo_gateway:kilo-ok": kilo_adapter,
        },
        max_attempts_per_batch=6,
        max_attempts_per_route=2,
    )
    batch = TranslationBatchRequest(
        job_id="route_wide_fast_fallback",
        items=(make_item("w1", "Hello"),),
        source_language="en",
        target_language="vi",
    )

    outcome = scheduler.schedule_batch(batch, candidates=[first, second, kilo])

    assert len(outcome.successes) == 1
    assert outcome.successes[0].route_id == "kilo_gateway"
    assert [attempt.route_id for attempt in outcome.attempts] == ["opencode", "kilo_gateway"]
    assert len(first_adapter.calls) == 1
    assert len(second_adapter.calls) == 0


def test_live_verified_opencode_models_are_ranked_before_unverified_models() -> None:
    models = [
        DiscoveredModel(model_id="big-pickle", display_name="Big", route_id="opencode"),
        DiscoveredModel(model_id="muse-spark-1.2-contributor-free", display_name="Muse 1.2", route_id="opencode"),
        DiscoveredModel(model_id="ling-3.0-flash-fin-free", display_name="Ling", route_id="opencode"),
        DiscoveredModel(model_id="muse-spark-1.3-contributor-free", display_name="Muse 1.3", route_id="opencode"),
    ]

    ranked = rank_candidates(models)

    assert [model.model_id for model in ranked[:3]] == [
        "ling-3.0-flash-fin-free",
        "muse-spark-1.3-contributor-free",
        "muse-spark-1.2-contributor-free",
    ]
