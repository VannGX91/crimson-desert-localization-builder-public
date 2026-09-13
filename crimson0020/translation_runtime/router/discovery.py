"""Dynamic router model discovery and cache management."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from crimson0020.translation_runtime.errors import TranslationRuntimeError
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.router.catalog import DiscoveredModel, RouteCatalogEntry
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

logger = logging.getLogger(__name__)

CACHE_FRESH_SECONDS = 6 * 3600  # 6 hours fresh cache
CACHE_STALE_MAX_SECONDS = 7 * 86400  # 7 days max stale-if-error

OPENCODE_KNOWN_FREE_MODELS: frozenset[str] = frozenset({
    "big-pickle",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "hy3-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
})
KILO_EXPLICIT_FREE_MODELS: frozenset[str] = frozenset({"kilo-auto/free", "openrouter/free"})
APP_USER_AGENT = "Crimson0020-TranslationRuntime/3.0 (Autonomous Localizer; +https://github.com/crimson0020)"

# Fixed discovery endpoints for supported routes
ROUTE_DISCOVERY_ENDPOINTS: dict[str, str] = {
    "opencode": "https://opencode.ai/zen/v1/models",
    "kilo_gateway": "https://api.kilo.ai/api/gateway/models",
    "kilo_anonymous": "https://api.kilo.ai/api/openrouter/models",
    "unclose": "https://hermes.ai.unturf.com/v1/models",
    "aihorde": "https://stablehorde.net/api/v2/status/models?type=text",
}

# Fixed fallback static models for routes with static or probe-only endpoints
STATIC_ROUTE_MODELS: dict[str, list[DiscoveredModel]] = {
    "llm7": [
        DiscoveredModel(
            model_id="default",
            display_name="LLM7 default",
            route_id="llm7",
            context_limit=4096,
            capabilities=("chat",),
            is_free=True,
        )
    ],
    "blockrun": [
        DiscoveredModel(
            model_id="nvidia/gpt-oss-20b",
            display_name="NVIDIA GPT OSS 20B (BlockRun)",
            route_id="blockrun",
            context_limit=4096,
            capabilities=("chat",),
            is_free=True,
        )
    ],
    "vireonix": [
        DiscoveredModel(
            model_id="auto",
            display_name="Vireonix Auto",
            route_id="vireonix",
            context_limit=4096,
            capabilities=("chat",),
            is_free=True,
        )
    ],
    "cehpoint": [
        DiscoveredModel(
            model_id="cehpoint-ai",
            display_name="Cehpoint AI",
            route_id="cehpoint",
            context_limit=4096,
            capabilities=("chat",),
            is_free=True,
        )
    ],
    "duckduckgo": [
        DiscoveredModel(
            model_id="gpt-4o-mini",
            display_name="GPT-4o mini (DuckDuckGo)",
            route_id="duckduckgo",
            context_limit=8192,
            capabilities=("chat",),
            is_free=True,
        ),
        DiscoveredModel(
            model_id="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku (DuckDuckGo)",
            route_id="duckduckgo",
            context_limit=8192,
            capabilities=("chat",),
            is_free=True,
        ),
    ],
    "cloudflare": [
        DiscoveredModel(
            model_id="@cf/meta/llama-3.1-8b-instruct",
            display_name="Llama 3.1 8B Instruct (Cloudflare)",
            route_id="cloudflare",
            context_limit=8192,
            capabilities=("chat",),
            is_free=True,
        )
    ],
}


def _discovery_headers(route_id: str) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if route_id == "opencode":
        return {
            **headers,
            "User-Agent": APP_USER_AGENT,
            "x-opencode-client": "crimson0020",
            "x-opencode-project": "local",
        }
    return headers


class DiscoveryError(TranslationRuntimeError):
    """Raised when discovery fails and no fresh or stale cache is available."""


class ModelDiscoveryService:
    """Discovers, validates, and caches free anonymous AI models."""

    def __init__(
        self,
        database: KnowledgeDatabase,
        transport: SafeHttpTransport | None = None,
        now_fn: Any | None = None,
    ) -> None:
        self.database = database
        self.transport = transport or SafeHttpTransport()
        self.now_fn = now_fn or time.time

    def _get_cached_entries(self, route_id: str) -> list[RouteCatalogEntry]:
        entries: list[RouteCatalogEntry] = []
        with self.database.connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT route_id, model_id, display_name, is_free, context_limit, capabilities, discovered_at, expires_at
                FROM route_catalog
                WHERE route_id = ? AND is_free = 1
                ORDER BY model_id ASC
                """,
                (route_id,),
            )
            for row in cur.fetchall():
                entries.append(
                    RouteCatalogEntry(
                        route_id=row[0],
                        model_id=row[1],
                        display_name=row[2],
                        is_free=bool(row[3]),
                        context_limit=row[4],
                        capabilities=row[5],
                        discovered_at=row[6],
                        expires_at=row[7],
                    )
                )
        return entries

    def _save_cached_entries(self, route_id: str, models: list[DiscoveredModel], now: float) -> None:
        expires_at = now + CACHE_FRESH_SECONDS
        with self.database.transaction() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM route_catalog WHERE route_id = ?", (route_id,))
            for m in models:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO route_catalog
                    (route_id, model_id, display_name, is_free, context_limit, capabilities, discovered_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        route_id,
                        m.model_id,
                        m.display_name,
                        1 if m.is_free else 0,
                        m.context_limit,
                        json.dumps(list(m.capabilities)),
                        now,
                        expires_at,
                    ),
                )

    def expire_cache_for_test(self, route_id: str, age_seconds: float) -> None:
        """Utility for test suites to simulate aging cache entries."""
        now = self.now_fn()
        new_discovered_at = now - age_seconds
        new_expires_at = new_discovered_at + CACHE_FRESH_SECONDS
        with self.database.transaction() as con:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE route_catalog
                SET discovered_at = ?, expires_at = ?
                WHERE route_id = ?
                """,
                (new_discovered_at, new_expires_at, route_id),
            )

    def discover_route_models(self, route_id: str, force_refresh: bool = False) -> list[DiscoveredModel]:
        """Discovers and returns eligible models for the specified route."""
        now = self.now_fn()

        # Check static route definitions first if applicable
        if route_id in STATIC_ROUTE_MODELS:
            return list(STATIC_ROUTE_MODELS[route_id])

        cached_entries = self._get_cached_entries(route_id)

        # If we have cache and force_refresh is False, check if fresh (< 6 hours)
        if cached_entries and not force_refresh:
            oldest_age = min(now - e.discovered_at for e in cached_entries)
            if oldest_age < CACHE_FRESH_SECONDS:
                return [e.to_discovered_model() for e in cached_entries]

        endpoint = ROUTE_DISCOVERY_ENDPOINTS.get(route_id)
        if not endpoint:
            # If no discovery endpoint and no static models, check stale cache
            if cached_entries:
                return [e.to_discovered_model() for e in cached_entries]
            return []

        # Attempt remote discovery
        try:
            res = self.transport.execute(endpoint, headers=_discovery_headers(route_id))
            if not res.is_success or not res.response or res.response.status_code != 200:
                raise DiscoveryError(f"Discovery endpoint returned status {res.status_code}: {res.error_message}")

            models = self._parse_and_filter_models(route_id, res.response.body)
            if models:
                self._save_cached_entries(route_id, models, now)
                return models
            elif cached_entries:
                return [e.to_discovered_model() for e in cached_entries]
            return []
        except Exception as e:
            logger.warning("Remote discovery failed for route %s: %s", route_id, e)
            # Stale-if-error: accept cache up to 7 days
            if cached_entries:
                oldest_age = min(now - e.discovered_at for e in cached_entries)
                if oldest_age <= CACHE_STALE_MAX_SECONDS:
                    return [e.to_discovered_model() for e in cached_entries]
            return []

    def _parse_and_filter_models(self, route_id: str, raw_body: bytes) -> list[DiscoveredModel]:
        data = json.loads(raw_body.decode("utf-8"))
        items = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

        if route_id == "aihorde":
            def horde_capacity(item: Any) -> tuple[int, int, float]:
                if not isinstance(item, dict):
                    return (0, 0, 0.0)
                name = str(item.get("name") or "").lower()
                instruction_tuned = int(
                    "instruct" in name
                    or "-it-" in name
                    or name.endswith("-it")
                    or "/qwen" in name
                )
                try:
                    workers = int(item.get("count") or 0)
                except (TypeError, ValueError):
                    workers = 0
                try:
                    performance = float(item.get("performance") or 0.0)
                except (TypeError, ValueError):
                    performance = 0.0
                return (instruction_tuned, workers, performance)

            items = [item for item in items if horde_capacity(item)[1] > 0]
            items.sort(key=horde_capacity, reverse=True)

        discovered: list[DiscoveredModel] = []
        seen_ids: set[str] = set()

        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id") or item.get("name")
            if not model_id or not isinstance(model_id, str):
                continue
            if model_id in seen_ids:
                continue

            # Free-model filtering rules:
            # Route-specific proof of free eligibility. OpenCode does not expose
            # price metadata for Zen; accept only the frozen known-free set or
            # a live model id explicitly ending in -free. Kilo exposes isFree
            # and also documents explicit free router ids / :free models.
            if route_id == "opencode":
                if model_id not in OPENCODE_KNOWN_FREE_MODELS and not model_id.endswith("-free"):
                    continue
            elif route_id in {"kilo_gateway", "kilo_anonymous"}:
                kilo_free = item.get("isFree") is True
                explicit_free = model_id in KILO_EXPLICIT_FREE_MODELS or model_id.endswith(":free")
                if not (kilo_free or explicit_free):
                    continue

            # 1. Reject if pricing indicates non-zero cost
            pricing = item.get("pricing")
            if isinstance(pricing, dict):
                prompt_price = float(pricing.get("prompt", 0.0) or 0.0)
                completion_price = float(pricing.get("completion", 0.0) or 0.0)
                if prompt_price > 0.0 or completion_price > 0.0:
                    continue

            # 2. Reject if explicitly requires paid tier or auth
            if item.get("paid") is True or item.get("is_free") is False or item.get("isFree") is False:
                continue

            # 3. Capabilities filtering: reject video/image/audio/support/local
            capabilities = ["chat"]
            params = item.get("supported_parameters") or item.get("capabilities") or []
            if isinstance(params, list):
                has_video = any("video" in str(p).lower() for p in params)
                has_image = any("image" in str(p).lower() for p in params)
                has_audio = any("audio" in str(p).lower() for p in params)
                if has_video or has_image or has_audio:
                    # If it doesn't support chat or is strictly media, skip
                    if not any("chat" in str(p).lower() for p in params):
                        continue

            display_name = item.get("name") or model_id
            context_limit = int(item.get("context_length") or item.get("context_limit") or 4096)

            seen_ids.add(model_id)
            # Model IDs are strictly data; endpoint_url is never accepted from item
            discovered.append(
                DiscoveredModel(
                    model_id=model_id,
                    display_name=str(display_name),
                    route_id=route_id,
                    context_limit=context_limit,
                    capabilities=tuple(capabilities),
                    is_free=True,
                )
            )

        return discovered
