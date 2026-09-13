"""Router scheduler orchestrating candidate priority, health, bisection, and bounded fallback (RTR-11)."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

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
from crimson0020.translation_runtime.router.catalog import DiscoveredModel
from crimson0020.translation_runtime.router.discovery import ModelDiscoveryService
from crimson0020.translation_runtime.router.health import RouteHealthTracker
from crimson0020.translation_runtime.router.ranking import (
    PROVIDER_PRIORITY,
    rank_candidates,
)

logger = logging.getLogger(__name__)

# Bootstrap order for a cold process.  The runtime quality tracker supersedes
# this as soon as real observations exist.  Keep same-route model ordering from
# rank_candidates(), including the OpenCode model fast path.
FAST_ROUTE_PRIORITY: tuple[str, ...] = (
    "opencode",
    "llm7",
    "unclose",
    "kilo_gateway",
    "kilo_anonymous",
    "aihorde",
    "blockrun",
    "vireonix",
    "cehpoint",
    "duckduckgo",
    "cloudflare",
)
PRIMARY_FAST_ROUTES: frozenset[str] = frozenset(FAST_ROUTE_PRIORITY[:2])


@dataclass(frozen=True, slots=True)
class RoutePerformance:
    samples: int
    success_ewma: float
    latency_ewma_ms: float
    rate_limit_ewma: float
    updated_at: float


class RoutePerformanceTracker:
    """Small in-process adaptive signal inspired by OmniRoute's quality EWMA.

    Circuit breakers remain the hard exclusion mechanism.  This tracker only
    changes soft ordering among candidates that are otherwise eligible.
    """

    def __init__(
        self,
        *,
        success_alpha: float = 0.2,
        latency_alpha: float = 0.1,
        warmup_samples: int = 5,
        latency_reference_ms: float = 1200.0,
    ) -> None:
        self.success_alpha = success_alpha
        self.latency_alpha = latency_alpha
        self.warmup_samples = max(1, warmup_samples)
        self.latency_reference_ms = max(1.0, latency_reference_ms)
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], RoutePerformance] = {}

    def record(
        self,
        route_id: str,
        model_id: str,
        *,
        success: bool,
        latency_ms: float,
        rate_limited: bool = False,
        now: float | None = None,
    ) -> RoutePerformance:
        key = (route_id, model_id)
        observed_success = 1.0 if success else 0.0
        observed_latency = max(0.0, float(latency_ms))
        ref_now = time.time() if now is None else now
        with self._lock:
            previous = self._records.get(key)
            if previous is None:
                record = RoutePerformance(
                    samples=1,
                    success_ewma=observed_success,
                    latency_ewma_ms=observed_latency,
                    rate_limit_ewma=1.0 if rate_limited else 0.0,
                    updated_at=ref_now,
                )
            else:
                record = RoutePerformance(
                    samples=previous.samples + 1,
                    success_ewma=(
                        self.success_alpha * observed_success
                        + (1.0 - self.success_alpha) * previous.success_ewma
                    ),
                    latency_ewma_ms=(
                        self.latency_alpha * observed_latency
                        + (1.0 - self.latency_alpha) * previous.latency_ewma_ms
                    ),
                    rate_limit_ewma=(
                        self.success_alpha * (1.0 if rate_limited else 0.0)
                        + (1.0 - self.success_alpha) * previous.rate_limit_ewma
                    ),
                    updated_at=ref_now,
                )
            self._records[key] = record
            return record

    def score(self, route_id: str, model_id: str) -> float:
        with self._lock:
            record = self._records.get((route_id, model_id))
        if record is None:
            return 0.5

        latency_score = 1.0 / (
            1.0 + (record.latency_ewma_ms / self.latency_reference_ms)
        )
        operational = (0.55 * record.success_ewma) + (0.45 * latency_score)
        operational *= 1.0 - (0.5 * record.rate_limit_ewma)
        confidence = min(1.0, record.samples / self.warmup_samples)
        return 0.5 + confidence * (operational - 0.5)

STRUCTURAL_FAILURE_KINDS: frozenset[FailureKind] = frozenset({
    FailureKind.INVALID_STRUCTURE,
    FailureKind.MISSING_ITEM,
    FailureKind.DUPLICATE_ITEM,
    FailureKind.EMPTY_TRANSLATION,
    FailureKind.PROTECTED_CONTENT_VIOLATION,
    FailureKind.GLOSSARY_VIOLATION,
})

ROUTE_WIDE_FAILURE_KINDS: frozenset[FailureKind] = frozenset({
    FailureKind.NETWORK,
    FailureKind.TIMEOUT,
    FailureKind.RATE_LIMIT,
    FailureKind.UNAUTHORIZED,
})

CIRCUIT_FAILURE_KINDS: frozenset[FailureKind] = frozenset({
    *ROUTE_WIDE_FAILURE_KINDS,
    FailureKind.UNSUPPORTED_MODEL,
})


class ConcurrencyManager:
    """Manages global and per-route adaptive concurrency limits."""

    def __init__(
        self,
        global_limit: int = 2,
        default_route_limit: int = 1,
        max_route_limit: int = 2,
        scalable_routes: frozenset[str] | None = None,
    ) -> None:
        if global_limit <= 0 or default_route_limit <= 0 or max_route_limit <= 0:
            raise ValueError("Concurrency limits must be positive.")
        if max_route_limit < default_route_limit:
            raise ValueError("Maximum route concurrency cannot be below the default.")
        self.global_limit = global_limit
        self.default_route_limit = default_route_limit
        self.max_route_limit = max_route_limit
        self.scalable_routes = scalable_routes
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._route_successes: dict[str, int] = {}
        self._route_limits: dict[str, int] = {}
        self._global_in_flight = 0
        self._route_in_flight: dict[str, int] = {}

    def get_route_limit(self, route_id: str) -> int:
        with self._lock:
            return self._route_limits.get(route_id, self.default_route_limit)

    def acquire(self, route_id: str, timeout_seconds: float | None = None) -> bool:
        """Acquire a global + per-route execution lease, optionally waiting for capacity."""
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("Concurrency wait timeout cannot be negative.")
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + timeout_seconds
        )
        with self._condition:
            while True:
                route_limit = self._route_limits.get(route_id, self.default_route_limit)
                route_in_flight = self._route_in_flight.get(route_id, 0)
                if (
                    self._global_in_flight < self.global_limit
                    and route_in_flight < route_limit
                ):
                    self._global_in_flight += 1
                    self._route_in_flight[route_id] = route_in_flight + 1
                    return True
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
                else:
                    self._condition.wait()

    def try_acquire(self, route_id: str) -> bool:
        """Acquire a non-blocking global + per-route execution lease."""
        return self.acquire(route_id, timeout_seconds=0.0)

    def release(self, route_id: str) -> None:
        """Release a previously acquired execution lease."""
        with self._condition:
            route_in_flight = self._route_in_flight.get(route_id, 0)
            if route_in_flight <= 1:
                self._route_in_flight.pop(route_id, None)
            else:
                self._route_in_flight[route_id] = route_in_flight - 1
            if self._global_in_flight > 0:
                self._global_in_flight -= 1
            self._condition.notify_all()

    def get_in_flight(self, route_id: str | None = None) -> int:
        """Returns current in-flight leases for diagnostics and tests."""
        with self._lock:
            if route_id is None:
                return self._global_in_flight
            return self._route_in_flight.get(route_id, 0)

    def record_success(self, route_id: str) -> None:
        """Increments consecutive successes and increases limit to 2 after 20 successes."""
        with self._lock:
            count = self._route_successes.get(route_id, 0) + 1
            self._route_successes[route_id] = count
            if count >= 20 and (
                self.scalable_routes is None or route_id in self.scalable_routes
            ):
                self._route_limits[route_id] = self.max_route_limit

    def record_rate_limit(self, route_id: str) -> None:
        """Reduces route concurrency back to default on 429 / rate limit."""
        with self._lock:
            self._route_successes[route_id] = 0
            self._route_limits[route_id] = self.default_route_limit

    def record_failure(self, route_id: str) -> None:
        with self._lock:
            self._route_successes[route_id] = 0
            self._route_limits[route_id] = self.default_route_limit


class RouterScheduler:
    """Orchestrates candidate fallback, health checks, batch bisection, and concurrency."""

    def __init__(
        self,
        health_tracker: RouteHealthTracker | None = None,
        discovery_service: ModelDiscoveryService | None = None,
        adapters: dict[str, Any] | None = None,
        adapter_factory: Callable[[str, str], Any] | None = None,
        concurrency_manager: ConcurrencyManager | None = None,
        max_attempts_per_batch: int = 6,
        max_attempts_per_route: int = 1,
        max_batch_duration_seconds: float = 180.0,
        candidate_cache_ttl_seconds: float = 120.0,
        max_bisection_attempts: int = 6,
        primary_route_wait_seconds: float = 30.0,
    ) -> None:
        self.health_tracker = health_tracker
        self.discovery_service = discovery_service
        self.adapters = dict(adapters or {})
        self.adapter_factory = adapter_factory
        self.concurrency_manager = concurrency_manager or ConcurrencyManager()
        self.max_attempts_per_batch = max(1, max_attempts_per_batch)
        self.max_attempts_per_route = max(1, max_attempts_per_route)
        self.max_batch_duration_seconds = max_batch_duration_seconds
        self.candidate_cache_ttl_seconds = max(0.0, candidate_cache_ttl_seconds)
        self.max_bisection_attempts = max(0, max_bisection_attempts)
        self.primary_route_wait_seconds = max(0.0, primary_route_wait_seconds)
        self._candidate_cache: tuple[DiscoveredModel, ...] | None = None
        self._candidate_cache_expires_at = 0.0
        self._adapter_cache: dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self.performance_tracker = RoutePerformanceTracker()

    def _wait_for_route_lease(
        self,
        route_id: str,
        *,
        monotonic_deadline: float,
        cancellation_token: CancellationToken | None,
    ) -> bool:
        wait_deadline = min(
            monotonic_deadline,
            time.monotonic() + self.primary_route_wait_seconds,
        )
        while True:
            if cancellation_token and cancellation_token.is_cancelled:
                return False
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self.concurrency_manager.acquire(route_id, min(0.1, remaining)):
                return True

    def _discover_candidates(self) -> tuple[DiscoveredModel, ...]:
        if self.discovery_service is None:
            return ()

        by_route: dict[str, tuple[DiscoveredModel, ...]] = {}

        def discover(route_id: str) -> tuple[str, tuple[DiscoveredModel, ...]]:
            try:
                models = tuple(self.discovery_service.discover_route_models(route_id))
            except Exception as exc:
                logger.debug("Router discovery failed for %s: %s", route_id, exc)
                models = ()
            return route_id, models

        worker_count = min(5, len(PROVIDER_PRIORITY))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(discover, route_id) for route_id in PROVIDER_PRIORITY]
            for future in concurrent.futures.as_completed(futures):
                route_id, models = future.result()
                by_route[route_id] = models

        return tuple(
            model
            for route_id in PROVIDER_PRIORITY
            for model in by_route.get(route_id, ())
        )

    def _rank_for_execution(
        self,
        candidates: Sequence[DiscoveredModel],
    ) -> list[DiscoveredModel]:
        baseline = rank_candidates(candidates)
        baseline_order = {
            (model.route_id, model.model_id): index
            for index, model in enumerate(baseline)
        }
        fast_order = {
            route_id: index for index, route_id in enumerate(FAST_ROUTE_PRIORITY)
        }

        def key(model: DiscoveredModel) -> tuple[float, int, int]:
            return (
                -self.performance_tracker.score(model.route_id, model.model_id),
                fast_order.get(model.route_id.lower(), len(FAST_ROUTE_PRIORITY)),
                baseline_order[(model.route_id, model.model_id)],
            )

        return sorted(baseline, key=key)

    def _get_adapter(self, route_id: str, model_id: str) -> Any:
        key = f"{route_id}:{model_id}"
        if key in self.adapters:
            return self.adapters[key]
        if route_id in self.adapters:
            return self.adapters[route_id]
        with self._cache_lock:
            if key in self._adapter_cache:
                return self._adapter_cache[key]
            if self.adapter_factory:
                adapter = self.adapter_factory(route_id, model_id)
                if adapter:
                    self._adapter_cache[key] = adapter
                    return adapter

            # Instantiate concrete adapter if available
            try:
                adapter = None
                if route_id == "opencode":
                    from crimson0020.translation_runtime.router.adapters.opencode import OpenCodeAdapter
                    adapter = OpenCodeAdapter(model_id=model_id)
                elif route_id in ("kilo_gateway", "kilo_anonymous"):
                    from crimson0020.translation_runtime.router.adapters.kilo import KiloAdapter
                    adapter = KiloAdapter(route_id=route_id, model_id=model_id)
                elif route_id == "unclose":
                    from crimson0020.translation_runtime.router.adapters.unclose import UncloseAdapter
                    adapter = UncloseAdapter(model_id=model_id)
                elif route_id == "duckduckgo":
                    from crimson0020.translation_runtime.router.adapters.duckduckgo import DuckDuckGoAdapter
                    adapter = DuckDuckGoAdapter(model_id=model_id)
                elif route_id == "cloudflare":
                    from crimson0020.translation_runtime.router.adapters.cloudflare import CloudflareAdapter
                    adapter = CloudflareAdapter(model_id=model_id)
                elif route_id == "aihorde":
                    from crimson0020.translation_runtime.router.adapters.aihorde import AIHordeAdapter
                    adapter = AIHordeAdapter(model_id=model_id)
                elif route_id in {"llm7", "blockrun", "vireonix", "cehpoint"}:
                    from crimson0020.translation_runtime.router.adapters.keyless_openai import KeylessOpenAIAdapter
                    adapter = KeylessOpenAIAdapter(route_id=route_id, model_id=model_id)
                if adapter is not None:
                    self._adapter_cache[key] = adapter
                    return adapter
            except Exception as e:
                logger.warning("Failed to initialize adapter for %s:%s: %s", route_id, model_id, e)

        return None

    @staticmethod
    def _items_for_attempt(
        adapter: Any,
        remaining_items: Sequence[TranslationRequestItem],
    ) -> tuple[TranslationRequestItem, ...]:
        """Bound one route attempt without shrinking the logical outer batch."""
        limits = getattr(adapter, "scheduler_batch_limits", None)
        if not (
            isinstance(limits, tuple)
            and len(limits) == 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in limits
            )
        ):
            return tuple(remaining_items)

        max_items, max_characters = limits
        selected: list[TranslationRequestItem] = []
        characters = 0
        for item in remaining_items:
            item_characters = len(item.masked_text)
            if selected and (
                len(selected) >= max_items
                or characters + item_characters > max_characters
            ):
                break
            selected.append(item)
            characters += item_characters
            if len(selected) >= max_items:
                break
        return tuple(selected)

    def schedule_batch(
        self,
        batch: TranslationBatchRequest,
        candidates: Sequence[DiscoveredModel] | None = None,
        completed_work_ids: set[str] | None = None,
        cancellation_token: CancellationToken | None = None,
        now: float | None = None,
    ) -> TranslationBatchOutcome:
        """Executes a translation batch across ranked candidates with fallback and bisection."""
        if not batch.items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        # Filter out already-checkpointed items (no checkpointed retry)
        completed = completed_work_ids or set()
        active_items = [item for item in batch.items if item.work_id not in completed]
        if not active_items:
            return TranslationBatchOutcome(successes=(), failures=(), attempts=())

        # Resolve candidate list
        if candidates is None:
            discovery_now = now if now is not None else time.time()
            with self._cache_lock:
                if (
                    self._candidate_cache is not None
                    and discovery_now < self._candidate_cache_expires_at
                ):
                    candidates = self._candidate_cache
                elif self.discovery_service:
                    self._candidate_cache = self._discover_candidates()
                    self._candidate_cache_expires_at = (
                        discovery_now + self.candidate_cache_ttl_seconds
                    )
                    candidates = self._candidate_cache
                else:
                    candidates = []

        ranked_models = self._rank_for_execution(candidates)

        accumulated_successes: list[TranslationItemResult] = []
        attempt_history: list[RouteAttempt] = []
        remaining_items: list[TranslationRequestItem] = list(active_items)
        last_failures: dict[str, TranslationFailure] = {}

        t_start = now if now is not None else time.time()
        monotonic_deadline = time.monotonic() + max(0.0, self.max_batch_duration_seconds)
        attempt_count = 0
        route_attempt_counts: dict[str, int] = {}
        blocked_routes: set[str] = set()
        next_retry_after: float | None = None

        pending_models = list(ranked_models)
        deferred_primary_models: list[DiscoveredModel] = []
        deferred_primary_routes: set[str] = set()

        while pending_models:
            model = pending_models.pop(0)
            if not remaining_items:
                break
            if attempt_count >= self.max_attempts_per_batch:
                break
            ref_now = now if now is not None else time.time()
            if (ref_now - t_start) >= self.max_batch_duration_seconds:
                break
            if time.monotonic() >= monotonic_deadline:
                break
            if cancellation_token and cancellation_token.is_cancelled:
                for item in remaining_items:
                    last_failures[item.work_id] = TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.CANCELLATION,
                        retryable=False,
                        message=cancellation_token.cancel_reason or "Operation cancelled",
                    )
                break

            preacquired_route = False
            if (
                model.route_id not in PRIMARY_FAST_ROUTES
                and deferred_primary_models
                and self.primary_route_wait_seconds > 0
            ):
                while deferred_primary_models:
                    deferred_model = deferred_primary_models.pop(0)
                    deferred_primary_routes.discard(deferred_model.route_id)
                    if deferred_model.route_id in blocked_routes:
                        continue
                    if (
                        route_attempt_counts.get(deferred_model.route_id, 0)
                        >= self.max_attempts_per_route
                    ):
                        continue
                    if self._wait_for_route_lease(
                        deferred_model.route_id,
                        monotonic_deadline=monotonic_deadline,
                        cancellation_token=cancellation_token,
                    ):
                        pending_models.insert(0, model)
                        model = deferred_model
                        preacquired_route = True
                        break

            route_id = model.route_id
            model_id = model.model_id

            if route_id in blocked_routes:
                continue

            # Do not let one unhealthy provider consume the whole batch budget.
            # Ranked candidates for later routes remain available for immediate fallback.
            if route_attempt_counts.get(route_id, 0) >= self.max_attempts_per_route:
                continue

            # Route health & cooldown probe
            if self.health_tracker:
                if not self.health_tracker.acquire_probe(route_id, model_id, now=ref_now):
                    if preacquired_route:
                        self.concurrency_manager.release(route_id)
                    retry_after = self.health_tracker.retry_after_seconds(
                        route_id, model_id, now=ref_now
                    )
                    if retry_after is not None:
                        next_retry_after = (
                            retry_after
                            if next_retry_after is None
                            else min(next_retry_after, retry_after)
                        )
                    continue

            adapter = self._get_adapter(route_id, model_id)
            if not adapter:
                if preacquired_route:
                    self.concurrency_manager.release(route_id)
                if self.health_tracker:
                    self.health_tracker.release_probe(route_id, model_id)
                continue

            if not preacquired_route and not self.concurrency_manager.try_acquire(route_id):
                if self.health_tracker:
                    self.health_tracker.release_probe(route_id, model_id)
                if (
                    route_id in PRIMARY_FAST_ROUTES
                    and route_id not in deferred_primary_routes
                ):
                    deferred_primary_models.append(model)
                    deferred_primary_routes.add(route_id)
                next_retry_after = (
                    0.05 if next_retry_after is None else min(next_retry_after, 0.05)
                )
                continue

            attempt_count += 1
            route_attempt_counts[route_id] = route_attempt_counts.get(route_id, 0) + 1
            attempt_items = self._items_for_attempt(adapter, remaining_items)
            current_sub_batch = TranslationBatchRequest(
                job_id=f"{batch.job_id}_try{attempt_count}",
                items=attempt_items,
                source_language=batch.source_language,
                target_language=batch.target_language,
            )

            attempt_started = time.monotonic()
            try:
                try:
                    outcome = adapter.translate_batch(current_sub_batch)
                except Exception as exc:
                    outcome = TranslationBatchOutcome(
                        successes=(),
                        failures=tuple(
                            TranslationFailure(
                                work_id=item.work_id,
                                kind=FailureKind.PROVIDER_INTERNAL,
                                retryable=True,
                                message=str(exc),
                            )
                            for item in current_sub_batch.items
                        ),
                        attempts=(
                            RouteAttempt(
                                route_id=route_id,
                                model_id=model_id,
                                duration_ms=0.0,
                                error_kind=FailureKind.PROVIDER_INTERNAL,
                            ),
                        ),
                    )
                observed_latency_ms = (time.monotonic() - attempt_started) * 1000.0
            finally:
                self.concurrency_manager.release(route_id)

            for att in outcome.attempts:
                attempt_history.append(att)

            # Check if any failure has 429 or status code
            rate_limited = any(
                f.kind == FailureKind.RATE_LIMIT for f in outcome.failures
            ) or any(
                getattr(att, "status_code", None) == 429 for att in outcome.attempts
            )
            if rate_limited:
                self.concurrency_manager.record_rate_limit(route_id)

            self.performance_tracker.record(
                route_id,
                model_id,
                success=(
                    len(outcome.successes) == len(current_sub_batch.items)
                    and not outcome.failures
                ),
                latency_ms=observed_latency_ms,
                rate_limited=rate_limited,
                now=ref_now,
            )

            route_wide_failure = rate_limited or any(
                failure.kind in ROUTE_WIDE_FAILURE_KINDS
                for failure in outcome.failures
            ) or any(
                attempt.error_kind in ROUTE_WIDE_FAILURE_KINDS
                for attempt in outcome.attempts
                if attempt.error_kind is not None
            )

            # Process successes
            succ_ids = set()
            for s in outcome.successes:
                accumulated_successes.append(s)
                succ_ids.add(s.work_id)

            if succ_ids:
                remaining_items = [i for i in remaining_items if i.work_id not in succ_ids]

            current_sub_batch_succeeded = (
                len(outcome.successes) == len(current_sub_batch.items)
                and not outcome.failures
            )

            if not remaining_items:
                # All items in batch succeeded!
                if self.health_tracker:
                    self.health_tracker.record_success(route_id, model_id, now=ref_now)
                self.concurrency_manager.record_success(route_id)
                break

            if current_sub_batch_succeeded:
                if self.health_tracker:
                    self.health_tracker.record_success(route_id, model_id, now=ref_now)
                self.concurrency_manager.record_success(route_id)
                continue

            # If there were failures, record in last_failures
            for f in outcome.failures:
                last_failures[f.work_id] = f

            # Check for structural failure bisection
            has_structural_error = any(
                f.kind in STRUCTURAL_FAILURE_KINDS for f in outcome.failures
            )
            failed_current_items = [
                item
                for item in current_sub_batch.items
                if item.work_id not in succ_ids
            ]
            bisection_allowed = getattr(adapter, "allow_structural_bisection", True)
            successes_before_bisection = len(accumulated_successes)
            if has_structural_error and len(failed_current_items) > 1 and bisection_allowed:
                # Bisect the remaining items
                self._bisect_and_execute(
                    items=failed_current_items,
                    adapter=adapter,
                    batch=batch,
                    accumulated_successes=accumulated_successes,
                    attempt_history=attempt_history,
                    last_failures=last_failures,
                    route_id=route_id,
                    model_id=model_id,
                    attempts_remaining=self.max_bisection_attempts,
                    monotonic_deadline=monotonic_deadline,
                )
                succ_ids_bisect = {s.work_id for s in accumulated_successes}
                remaining_items = [i for i in remaining_items if i.work_id not in succ_ids_bisect]
                if not remaining_items:
                    if self.health_tracker:
                        self.health_tracker.record_success(route_id, model_id, now=ref_now)
                    self.concurrency_manager.record_success(route_id)
                    break

            if (
                has_structural_error
                and not succ_ids
                and len(accumulated_successes) == successes_before_bisection
            ):
                # A full structural miss with no recovery is not evidence that a
                # second model on the same anonymous route will help this batch.
                # Move to a different route instead of repeating an expensive
                # malformed-response path in the same scheduler call.
                blocked_routes.add(route_id)

            # Update health tracker on failure
            if self.health_tracker:
                primary_kind = FailureKind.PROVIDER_INTERNAL
                status_code: int | None = None
                retry_after_sec: float | None = None

                if outcome.attempts and outcome.attempts[-1].error_kind:
                    primary_kind = outcome.attempts[-1].error_kind
                    status_code = outcome.attempts[-1].status_code
                if outcome.failures:
                    retry_after_sec = outcome.failures[0].retry_after_seconds
                    if not (outcome.attempts and outcome.attempts[-1].error_kind):
                        primary_kind = outcome.failures[0].kind

                should_open_circuit = (
                    primary_kind in CIRCUIT_FAILURE_KINDS
                    or (
                        primary_kind == FailureKind.PROVIDER_INTERNAL
                        and status_code is not None
                        and status_code >= 500
                    )
                )
                if should_open_circuit:
                    health_record = self.health_tracker.record_failure(
                        route_id=route_id,
                        model_id=model_id,
                        failure_kind=primary_kind,
                        status_code=status_code,
                        retry_after=retry_after_sec,
                        now=ref_now,
                    )
                    cooldown_remaining = max(0.0, health_record.cooldown_until - ref_now)
                    if cooldown_remaining > 0:
                        next_retry_after = (
                            cooldown_remaining
                            if next_retry_after is None
                            else min(next_retry_after, cooldown_remaining)
                        )
                        for failure in outcome.failures:
                            if failure.retryable and failure.work_id in last_failures:
                                last_failures[failure.work_id] = TranslationFailure(
                                    work_id=failure.work_id,
                                    kind=failure.kind,
                                    retryable=True,
                                    message=failure.message,
                                    retry_after_seconds=(
                                        failure.retry_after_seconds
                                        if failure.retry_after_seconds is not None
                                        else cooldown_remaining
                                    ),
                                )
                else:
                    self.health_tracker.release_probe(route_id, model_id)
            self.concurrency_manager.record_failure(route_id)
            if route_wide_failure:
                blocked_routes.add(route_id)

        # Assemble final failures for any remaining unresolved items
        final_failures: list[TranslationFailure] = []
        for item in remaining_items:
            if item.work_id in last_failures:
                final_failures.append(last_failures[item.work_id])
            else:
                final_failures.append(
                    TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=next_retry_after is not None,
                        message=(
                            "All eligible routes are temporarily busy or cooling down"
                            if next_retry_after is not None
                            else "Route attempts or time exhausted without resolution"
                        ),
                        retry_after_seconds=next_retry_after,
                    )
                )

        return TranslationBatchOutcome(
            successes=tuple(accumulated_successes),
            failures=tuple(final_failures),
            attempts=tuple(attempt_history),
        )

    def _bisect_and_execute(
        self,
        items: list[TranslationRequestItem],
        adapter: Any,
        batch: TranslationBatchRequest,
        accumulated_successes: list[TranslationItemResult],
        attempt_history: list[RouteAttempt],
        last_failures: dict[str, TranslationFailure],
        route_id: str,
        model_id: str,
        attempts_remaining: int,
        monotonic_deadline: float,
    ) -> int:
        """Recursively splits a batch encountering structural failure down to individual items."""
        if not items or attempts_remaining <= 0 or time.monotonic() >= monotonic_deadline:
            return attempts_remaining

        mid = len(items) // 2
        halves = [items[:mid], items[mid:]]

        for half in halves:
            if (
                not half
                or attempts_remaining <= 0
                or time.monotonic() >= monotonic_deadline
            ):
                continue
            attempts_remaining -= 1
            sub_batch = TranslationBatchRequest(
                job_id=f"{batch.job_id}_bisect_{len(half)}",
                items=tuple(half),
                source_language=batch.source_language,
                target_language=batch.target_language,
            )
            if not self.concurrency_manager.try_acquire(route_id):
                for item in half:
                    last_failures[item.work_id] = TranslationFailure(
                        work_id=item.work_id,
                        kind=FailureKind.PROVIDER_INTERNAL,
                        retryable=True,
                        message="Route is busy during structural retry",
                        retry_after_seconds=0.05,
                    )
                continue
            try:
                try:
                    outcome = adapter.translate_batch(sub_batch)
                except Exception as e:
                    outcome = TranslationBatchOutcome(
                        successes=(),
                        failures=tuple(
                            TranslationFailure(
                                work_id=it.work_id,
                                kind=FailureKind.INVALID_STRUCTURE,
                                retryable=False,
                                message=str(e),
                            )
                            for it in half
                        ),
                        attempts=(
                            RouteAttempt(
                                route_id=route_id,
                                model_id=model_id,
                                duration_ms=0.0,
                                error_kind=FailureKind.INVALID_STRUCTURE,
                            ),
                        ),
                    )
            finally:
                self.concurrency_manager.release(route_id)

            for att in outcome.attempts:
                attempt_history.append(att)

            succ_ids = set()
            for s in outcome.successes:
                accumulated_successes.append(s)
                succ_ids.add(s.work_id)

            for f in outcome.failures:
                last_failures[f.work_id] = f

            failed_half_items = [it for it in half if it.work_id not in succ_ids]
            if failed_half_items and len(failed_half_items) > 1:
                has_structural = any(
                    last_failures.get(it.work_id) is not None
                    and last_failures[it.work_id].kind in STRUCTURAL_FAILURE_KINDS
                    for it in failed_half_items
                )
                if has_structural:
                    attempts_remaining = self._bisect_and_execute(
                        items=failed_half_items,
                        adapter=adapter,
                        batch=batch,
                        accumulated_successes=accumulated_successes,
                        attempt_history=attempt_history,
                        last_failures=last_failures,
                        route_id=route_id,
                        model_id=model_id,
                        attempts_remaining=attempts_remaining,
                        monotonic_deadline=monotonic_deadline,
                    )
        return attempts_remaining
