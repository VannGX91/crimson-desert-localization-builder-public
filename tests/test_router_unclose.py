import json
from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.unclose import UncloseAdapter
from crimson0020.translation_runtime.router.transport import SafeHttpTransport


def _unclose_batch(count: int) -> TranslationBatchRequest:
    return TranslationBatchRequest(
        job_id=f"unclose-{count}",
        source_language="en",
        target_language="vi",
        items=tuple(
            TranslationRequestItem(
                work_id=f"u{i:02d}",
                masked_text=f"Synthetic row {i:02d} " + "x" * 80,
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            )
            for i in range(count)
        ),
    )


def _ids_from_unclose_prompt(body: bytes) -> list[str]:
    payload = json.loads(body.decode("utf-8"))
    prompt = payload["messages"][1]["content"]
    raw = prompt.split("<TRANSLATION_INPUT>\n", 1)[1].split("\n</TRANSLATION_INPUT>", 1)[0]
    request = json.loads(raw)
    return [item["id"] for item in request["items"]]

def test_unclose_adapter_anonymous_success():
    def fake_transport(url, headers, body, timeout):
        assert "hermes.ai.unturf.com" in url
        # Public app identifier only, no user secret
        assert headers.get("User-Agent") != ""
        assert "Authorization" not in headers
        payload = json.loads(body.decode("utf-8"))
        assert payload["model"] == "Lorbus/Qwen3.6-27B-int4-AutoRound"
        assert payload["response_format"] == {"type": "json_object"}

        resp = {
            "choices": [
                {"message": {"content": json.dumps({"translations": [{"id": "u1", "text": "Dịch Unclose"}]})}}
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(resp).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = UncloseAdapter(model_id="Lorbus/Qwen3.6-27B-int4-AutoRound", transport=transport)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="u1",
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
    assert outcome.successes[0].translated_text == "Dịch Unclose"
    assert outcome.attempts[0].route_id == "unclose"


def test_unclose_large_batch_is_split_into_timeout_bounded_chunks():
    seen_sizes: list[int] = []

    def fake_transport(url, headers, body, timeout):
        del url, headers, timeout
        ids = _ids_from_unclose_prompt(body)
        seen_sizes.append(len(ids))
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {"id": work_id, "text": f"VI:{work_id}"}
                                    for work_id in ids
                                ]
                            }
                        )
                    }
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response).encode("utf-8")

    adapter = UncloseAdapter(
        model_id="Lorbus/Qwen3.6-27B-int4-AutoRound",
        transport=SafeHttpTransport(injected_transport=fake_transport),
    )
    outcome = adapter.translate_batch(_unclose_batch(48))

    assert seen_sizes == [16, 16, 16]
    assert len(outcome.successes) == 48
    assert outcome.failures == ()
    assert len(outcome.attempts) == 3

def test_unclose_transition_to_key_required():
    def fake_transport_auth_required(url, headers, body, timeout):
        return 401, {}, b"Authentication required: API key missing"

    adapter = UncloseAdapter(
        model_id="unclose/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_auth_required),
    )
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="u1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert outcome.failures[0].kind == FailureKind.UNAUTHORIZED
    assert outcome.failures[0].retryable is False

def test_unclose_rate_limit_and_error_handling():
    def fake_transport_429(url, headers, body, timeout):
        return 429, {"Retry-After": "60"}, b"Too many requests"

    adapter = UncloseAdapter(
        model_id="unclose/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_429),
    )
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="u1",
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
    assert outcome.failures[0].retry_after_seconds == 60.0

    def fake_transport_404(url, headers, body, timeout):
        return 404, {}, b"Not found"

    adapter_404 = UncloseAdapter(
        model_id="unclose/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_404),
    )
    res_404 = adapter_404.translate_batch(batch)
    assert res_404.failures[0].kind == FailureKind.UNSUPPORTED_MODEL

    def fake_transport_500(url, headers, body, timeout):
        return 500, {}, b"Server error"

    adapter_500 = UncloseAdapter(
        model_id="unclose/free-model",
        transport=SafeHttpTransport(injected_transport=fake_transport_500),
    )
    res_500 = adapter_500.translate_batch(batch)
    assert res_500.failures[0].kind == FailureKind.PROVIDER_INTERNAL

def test_unclose_empty_batch_and_malformed():
    adapter = UncloseAdapter()
    empty = TranslationBatchRequest(job_id="j", source_language="en", target_language="vi", items=())
    assert len(adapter.translate_batch(empty).successes) == 0

    def fake_bad_json(url, headers, body, timeout):
        return 200, {}, b"bad"
    adapter_bad = UncloseAdapter(transport=SafeHttpTransport(injected_transport=fake_bad_json))
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="u1",
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


def test_unclose_default_transport_has_bounded_read_timeout():
    assert UncloseAdapter().transport.read_timeout_seconds == 12.0
