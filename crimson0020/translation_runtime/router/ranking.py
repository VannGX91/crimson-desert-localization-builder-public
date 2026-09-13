"""Candidate ranking and priority ordering according to PLAN.md router specification (RTR-11)."""

from __future__ import annotations

from typing import Sequence

from crimson0020.translation_runtime.router.catalog import DiscoveredModel

# Exact provider priority order from PLAN.md
PROVIDER_PRIORITY: tuple[str, ...] = (
    "opencode",
    "kilo_gateway",
    "kilo_anonymous",
    "llm7",
    "unclose",
    "blockrun",
    "cehpoint",
    "aihorde",
    "vireonix",
    "duckduckgo",
    "cloudflare",
)

# Routes strictly prohibited inside AI Router (standalone/manual only)
PROHIBITED_ROUTES: frozenset[str] = frozenset({
    "google",
    "google_translate",
    "mymemory",
    "mymemory_translated",
})

META_ROUTE_MARKERS: tuple[str, ...] = ("auto", "meta/", "router", "default", "any")

# Fast path from current sanitized live probes. Missing/stale entries are harmless:
# discovery still controls availability, and the health circuit breaker skips failures.
OPENCODE_FAST_PATH_MODEL_PRIORITY: dict[str, int] = {
    "ling-3.0-flash-fin-free": 0,
    "muse-spark-1.3-contributor-free": 1,
    "muse-spark-1.2-contributor-free": 2,
}


def is_meta_route(model_id: str) -> bool:
    """Returns True if the model identifier represents a meta/auto route rather than an explicit model."""
    lowered = model_id.lower()
    return any(marker in lowered for marker in META_ROUTE_MARKERS)


def rank_candidates(
    candidates: Sequence[DiscoveredModel],
    prefer_explicit_kilo: bool = True,
) -> list[DiscoveredModel]:
    """Ranks model candidates by provider priority, explicit model preference, and context capacity.

    Prohibited routes (Google, MyMemory) are strictly filtered out.
    """
    valid_candidates = [
        (index, model)
        for index, model in enumerate(candidates)
        if model.route_id.lower() not in PROHIBITED_ROUTES
    ]

    def sort_key(entry: tuple[int, DiscoveredModel]) -> tuple[int, int, int, int, int]:
        input_index, model = entry
        # 1. Provider priority index
        try:
            provider_idx = PROVIDER_PRIORITY.index(model.route_id.lower())
        except ValueError:
            provider_idx = 999

        # 2. Prefer OpenCode models that have a current real synthetic success.
        if model.route_id.lower() == "opencode":
            model_rank = OPENCODE_FAST_PATH_MODEL_PRIORITY.get(model.model_id, 100)
        else:
            model_rank = 0

        # 3. Explicit models before meta routes (especially for Kilo routes)
        if prefer_explicit_kilo and model.route_id.lower().startswith("kilo"):
            meta_rank = 1 if is_meta_route(model.model_id) else 0
        elif prefer_explicit_kilo and is_meta_route(model.model_id):
            meta_rank = 1
        else:
            meta_rank = 0

        # 4. Context limit descending (prefer higher context windows)
        context_rank = -model.context_limit

        # 5. Preserve deterministic discovery/catalog order on exact ties.
        return (provider_idx, model_rank, meta_rank, context_rank, input_index)

    return [model for _, model in sorted(valid_candidates, key=sort_key)]
