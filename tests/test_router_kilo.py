import json
import pytest
from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.kilo import KiloAdapter
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

def test_kilo_gateway_adapter_success():
    def fake_transport(url, headers, body, timeout):
        assert url == "https://api.kilo.ai/api/gateway/chat/completions"
        # Gateway free route is anonymous and needs no credential header.
        assert "Authorization" not in headers
        assert headers.get("User-Agent") != ""
        payload = json.loads(body.decode("utf-8"))
        assert payload["model"] == "kilo/free-model"

        resp = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "translations": [
                                {"id": "k1", "text": "Dịch 1"}
                            ]
                        })
                    }
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(resp).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = KiloAdapter(route_id="kilo_gateway", model_id="kilo/free-model", transport=transport)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="k1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.successes) == 1
    assert outcome.successes[0].translated_text == "Dịch 1"
    assert outcome.attempts[0].route_id == "kilo_gateway"

def test_kilo_anonymous_route_identity_and_separation():
    def fake_transport(url, headers, body, timeout):
        assert url == "https://api.kilo.ai/api/openrouter/chat/completions"
        assert headers.get("Authorization") == "Bearer anonymous"
        assert headers.get("X-KILOCODE-EDITORNAME") == "Crimson0020"
        resp = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "translations": [
                                {"id": "k1", "text": "Dịch Anon"}
                            ]
                        })
                    }
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(resp).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = KiloAdapter(route_id="kilo_anonymous", model_id="kilo/anon-model", transport=transport)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="k1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.successes) == 1
    assert outcome.successes[0].translated_text == "Dịch Anon"
    assert outcome.attempts[0].route_id == "kilo_anonymous"

def test_kilo_rate_limit_and_unauthorized():
    def fake_transport_429(url, headers, body, timeout):
        return 429, {"Retry-After": "45"}, b"Kilo limit reached"

    adapter = KiloAdapter(
        route_id="kilo_gateway",
        model_id="kilo/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_429),
    )
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="k1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert outcome.failures[0].kind == FailureKind.RATE_LIMIT
    assert outcome.failures[0].retry_after_seconds == 45.0

    def fake_transport_403(url, headers, body, timeout):
        return 403, {}, b"Forbidden"

    adapter_403 = KiloAdapter(
        route_id="kilo_gateway",
        model_id="kilo/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_403),
    )
    res_403 = adapter_403.translate_batch(batch)
    assert res_403.failures[0].kind == FailureKind.UNAUTHORIZED

    def fake_transport_404(url, headers, body, timeout):
        return 404, {}, b"Not found"

    adapter_404 = KiloAdapter(
        route_id="kilo_gateway",
        model_id="kilo/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_404),
    )
    res_404 = adapter_404.translate_batch(batch)
    assert res_404.failures[0].kind == FailureKind.UNSUPPORTED_MODEL

    def fake_transport_500(url, headers, body, timeout):
        return 500, {}, b"Server error"

    adapter_500 = KiloAdapter(
        route_id="kilo_gateway",
        model_id="kilo/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_500),
    )
    res_500 = adapter_500.translate_batch(batch)
    assert res_500.failures[0].kind == FailureKind.PROVIDER_INTERNAL

def test_kilo_empty_batch_invalid_route_and_malformed():
    with pytest.raises(ValueError):
        KiloAdapter(route_id="invalid_route")

    adapter = KiloAdapter()
    empty = TranslationBatchRequest(job_id="j", source_language="en", target_language="vi", items=())
    assert len(adapter.translate_batch(empty).successes) == 0

    def fake_bad_json(url, headers, body, timeout):
        return 200, {}, b"bad"
    adapter_bad = KiloAdapter(transport=SafeHttpTransport(injected_transport=fake_bad_json))
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="k1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter_bad.translate_batch(batch)
    assert outcome.failures[0].kind == FailureKind.INVALID_STRUCTURE
