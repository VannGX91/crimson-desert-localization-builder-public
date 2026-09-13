import json
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.router.catalog import RouteCatalogEntry
from crimson0020.translation_runtime.router.discovery import (
    ModelDiscoveryService,
)
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

def test_model_discovery_filtering(tmp_path):
    db_path = tmp_path / "test-catalog.sqlite3"
    db = KnowledgeDatabase(db_path)

    # Fake catalog payload from OpenCode
    fake_catalog = {
        "data": [
            {
                "id": "mimo-v2.5-free",
                "name": "OpenCode Free 1",
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 8192,
                "supported_parameters": ["chat"]
            },
            {
                "id": "claude-sonnet-5",
                "name": "OpenCode Paid",
                "pricing": {"prompt": 0.001, "completion": 0.002},
                "context_length": 16384,
                "supported_parameters": ["chat"]
            },
            {
                "id": "media-preview-free",
                "name": "OpenCode Video",
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096,
                "supported_parameters": ["video_generate"]
            },
            {
                "id": "paid-flag-free",
                "name": "OpenCode Paid Flag",
                "paid": True,
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096
            },
            {
                "id": "not-free-explicit-free",
                "name": "OpenCode Not Free",
                "is_free": False,
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096
            },
            {
                "id": "north-mini-code-free",
                "name": "Poisoned Model",
                "endpoint_url": "https://malicious.attacker.com/v1/chat",
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 8192
            },
            "invalid_entry",
            {"id": None}
        ]
    }

    def fake_transport(url, headers, body, timeout):
        return 200, {"Content-Type": "application/json"}, json.dumps(fake_catalog).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    discovery = ModelDiscoveryService(database=db, transport=transport)

    models = discovery.discover_route_models("opencode")

    # Only free, chat-capable, non-poisoned model should be included
    model_ids = [m.model_id for m in models]
    assert "mimo-v2.5-free" in model_ids
    assert "claude-sonnet-5" not in model_ids
    assert "media-preview-free" not in model_ids
    assert "paid-flag-free" not in model_ids
    assert "not-free-explicit-free" not in model_ids
    # Model IDs are strictly data and endpoint_url is ignored
    for m in models:
        assert not hasattr(m, "endpoint_url") or getattr(m, "endpoint_url") is None
        d = m.to_dict()
        assert "model_id" in d

def test_cache_freshness_and_stale_fallback(tmp_path):
    db_path = tmp_path / "test-cache.sqlite3"
    db = KnowledgeDatabase(db_path)

    call_count = [0]
    def fake_transport(url, headers, body, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            catalog = {
                "data": [
                    {
                        "id": "fresh-model-free",
                        "name": "Model Fresh",
                        "pricing": {"prompt": 0.0, "completion": 0.0},
                        "context_length": 4096
                    }
                ]
            }
            return 200, {"Content-Type": "application/json"}, json.dumps(catalog).encode("utf-8")
        raise RuntimeError("Discovery endpoint unreachable")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    discovery = ModelDiscoveryService(database=db, transport=transport)

    # First call: discovers and caches
    models_1 = discovery.discover_route_models("opencode")
    assert len(models_1) == 1
    assert models_1[0].model_id == "fresh-model-free"
    assert call_count[0] == 1

    # Second call within 6 hours: served from fresh cache, no transport call
    models_2 = discovery.discover_route_models("opencode")
    assert len(models_2) == 1
    assert models_2[0].model_id == "fresh-model-free"
    assert call_count[0] == 1

    # Force refresh when remote fails returns stale
    models_refresh = discovery.discover_route_models("opencode", force_refresh=True)
    assert len(models_refresh) == 1
    assert models_refresh[0].model_id == "fresh-model-free"

    # Simulate cache aging past 6 hours but within 7 days
    discovery.expire_cache_for_test("opencode", age_seconds=7 * 3600)

    # Next call: transport fails, but stale cache (< 7 days) is returned
    models_3 = discovery.discover_route_models("opencode")
    assert len(models_3) == 1
    assert models_3[0].model_id == "fresh-model-free"

    # Simulate cache aging past 7 days
    discovery.expire_cache_for_test("opencode", age_seconds=8 * 86400)

    # Fourth call: cache expired, transport fails -> returns empty
    models_4 = discovery.discover_route_models("opencode")
    assert len(models_4) == 0

def test_duplicate_and_empty_discovery(tmp_path):
    db_path = tmp_path / "test-dedup.sqlite3"
    db = KnowledgeDatabase(db_path)

    fake_catalog = {
        "data": [
            {"id": "dup-1-free", "name": "Dup 1", "pricing": {"prompt": 0.0, "completion": 0.0}},
            {"id": "dup-1-free", "name": "Dup 1 Duplicate", "pricing": {"prompt": 0.0, "completion": 0.0}},
        ]
    }
    def fake_transport(url, headers, body, timeout):
        return 200, {"Content-Type": "application/json"}, json.dumps(fake_catalog).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    discovery = ModelDiscoveryService(database=db, transport=transport)
    models = discovery.discover_route_models("opencode")
    assert len(models) == 1
    assert models[0].model_id == "dup-1-free"

def test_static_routes_and_unknown_routes(tmp_path):
    db_path = tmp_path / "test-static.sqlite3"
    db = KnowledgeDatabase(db_path)
    transport = SafeHttpTransport(injected_transport=lambda u, h, b, t: (200, {}, b"{}"))
    discovery = ModelDiscoveryService(database=db, transport=transport)

    ddg_models = discovery.discover_route_models("duckduckgo")
    assert len(ddg_models) >= 2
    assert any("gpt-4o-mini" in m.model_id for m in ddg_models)

    cf_models = discovery.discover_route_models("cloudflare")
    assert len(cf_models) >= 1
    assert any("llama" in m.model_id.lower() for m in cf_models)

    assert [m.model_id for m in discovery.discover_route_models("llm7")] == ["default"]
    assert [m.model_id for m in discovery.discover_route_models("blockrun")] == ["nvidia/gpt-oss-20b"]
    assert [m.model_id for m in discovery.discover_route_models("vireonix")] == ["auto"]
    assert [m.model_id for m in discovery.discover_route_models("cehpoint")] == ["cehpoint-ai"]

    unknown = discovery.discover_route_models("unknown_route_id")
    assert unknown == []

def test_route_catalog_entry_corrupt_capabilities():
    entry = RouteCatalogEntry(
        route_id="test",
        model_id="m1",
        display_name="M1",
        is_free=True,
        context_limit=4096,
        capabilities="invalid_json",
        discovered_at=0.0,
        expires_at=0.0,
    )
    model = entry.to_discovered_model()
    assert model.capabilities == ("chat",)


def test_live_route_endpoints_and_explicit_free_filtering(tmp_path):
    from crimson0020.translation_runtime.router.discovery import ROUTE_DISCOVERY_ENDPOINTS

    assert ROUTE_DISCOVERY_ENDPOINTS["opencode"] == "https://opencode.ai/zen/v1/models"
    assert ROUTE_DISCOVERY_ENDPOINTS["kilo_gateway"] == "https://api.kilo.ai/api/gateway/models"
    assert ROUTE_DISCOVERY_ENDPOINTS["kilo_anonymous"] == "https://api.kilo.ai/api/openrouter/models"
    assert ROUTE_DISCOVERY_ENDPOINTS["unclose"] == "https://hermes.ai.unturf.com/v1/models"
    assert ROUTE_DISCOVERY_ENDPOINTS["aihorde"] == "https://stablehorde.net/api/v2/status/models?type=text"

    db = KnowledgeDatabase(tmp_path / "kilo-filter.sqlite3")
    payload = {"data": [
        {"id": "kilo-auto/efficient", "name": "Paid", "isFree": False, "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "kilo-auto/free", "name": "Free", "isFree": True, "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "mystery", "name": "Unknown", "pricing": {"prompt": "0", "completion": "0"}},
    ]}
    transport = SafeHttpTransport(injected_transport=lambda u,h,b,t: (200, {"Content-Type":"application/json"}, json.dumps(payload).encode()))
    models = ModelDiscoveryService(db, transport=transport).discover_route_models("kilo_gateway", force_refresh=True)
    assert [m.model_id for m in models] == ["kilo-auto/free"]


def test_aihorde_discovery_prefers_active_worker_capacity(tmp_path):
    db = KnowledgeDatabase(tmp_path / "horde-order.sqlite3")
    payload = [
        {"name": "few-workers", "count": 1, "performance": 10.0},
        {"name": "offline-model", "count": 0, "performance": 999.0},
        {"name": "many-workers", "count": 8, "performance": 5.0},
        {"name": "same-workers-faster", "count": 8, "performance": 20.0},
        {"name": "llama-3.2-instruct", "count": 2, "performance": 3.0},
    ]
    transport = SafeHttpTransport(
        injected_transport=lambda u, h, b, t: (
            200,
            {"Content-Type": "application/json"},
            json.dumps(payload).encode("utf-8"),
        )
    )

    models = ModelDiscoveryService(db, transport=transport).discover_route_models(
        "aihorde", force_refresh=True
    )

    assert [model.model_id for model in models] == [
        "llama-3.2-instruct",
        "same-workers-faster",
        "many-workers",
        "few-workers",
    ]


def test_opencode_discovery_sends_truthful_application_identity(tmp_path):
    db = KnowledgeDatabase(tmp_path / "opencode-headers.sqlite3")
    captured_headers = {}

    def fake_transport(url, headers, body, timeout):
        captured_headers.update(headers)
        payload = {
            "data": [
                {
                    "id": "mimo-v2.5-free",
                    "name": "OpenCode Free",
                    "pricing": {"prompt": 0.0, "completion": 0.0},
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    models = ModelDiscoveryService(db, transport=transport).discover_route_models(
        "opencode", force_refresh=True
    )

    assert [model.model_id for model in models] == ["mimo-v2.5-free"]
    assert captured_headers["Accept"] == "application/json"
    assert captured_headers["User-Agent"].startswith("Crimson0020-TranslationRuntime/")
    assert captured_headers["x-opencode-client"] == "crimson0020"
    assert captured_headers["x-opencode-project"] == "local"
