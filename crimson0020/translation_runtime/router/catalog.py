"""Discovered route and model data structures and catalog representations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """Represents a discovered model confirmed to be free and anonymous."""

    model_id: str
    display_name: str
    route_id: str
    context_limit: int = 4096
    capabilities: tuple[str, ...] = ("chat",)
    is_free: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "route_id": self.route_id,
            "context_limit": self.context_limit,
            "capabilities": list(self.capabilities),
            "is_free": self.is_free,
        }


@dataclass(frozen=True, slots=True)
class RouteCatalogEntry:
    """Catalog entry stored in SQLite route_catalog cache."""

    route_id: str
    model_id: str
    display_name: str
    is_free: bool
    context_limit: int
    capabilities: str  # JSON list of strings
    discovered_at: float
    expires_at: float

    def to_discovered_model(self) -> DiscoveredModel:
        try:
            caps = tuple(json.loads(self.capabilities))
        except Exception:
            caps = ("chat",)
        return DiscoveredModel(
            model_id=self.model_id,
            display_name=self.display_name,
            route_id=self.route_id,
            context_limit=self.context_limit,
            capabilities=caps,
            is_free=bool(self.is_free),
        )
