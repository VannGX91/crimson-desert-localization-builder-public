import json
from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.opencode import OpenCodeAdapter
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

def test_opencode_chat_completions_success():
    def fake_transport(url, headers, body, timeout):
        assert url == "https://opencode.ai/zen/v1/chat/completions"
        # Check truthful app headers and no credentials
        assert headers.get("User-Agent", "").startswith("Crimson0020-TranslationRuntime/")
        assert headers.get("x-opencode-client") == "crimson0020"
        assert headers.get("x-opencode-project") == "local"
        assert headers.get("x-opencode-request")
        assert headers.get("x-opencode-session")
        assert "Authorization" not in headers or "Bearer" not in headers.get("Authorization", "")
        payload = json.loads(body.decode("utf-8"))
        assert payload["model"] == "opencode/free-1"
        assert payload["temperature"] == 0.0

        response_data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "translations": [
                                {"id": "w1", "text": "Bản dịch 1"},
                                {"id": "w2", "text": "Bản dịch 2"}
                            ]
                        })
                    }
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = OpenCodeAdapter(model_id="opencode/free-1", transport=transport)

    batch = TranslationBatchRequest(
        job_id="job_1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
            TranslationRequestItem(
                work_id="w2",
                masked_text="Text 2",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.successes) == 2
    assert len(outcome.failures) == 0
    assert outcome.successes[0].translated_text == "Bản dịch 1"
    assert outcome.successes[1].translated_text == "Bản dịch 2"
    assert len(outcome.attempts) == 1

def test_opencode_responses_protocol_success():
    def fake_transport(url, headers, body, timeout):
        response_data = {
            "response": {
                "content": json.dumps({
                    "translations": [
                        {"id": "w1", "text": "Bản dịch 1"}
                    ]
                })
            }
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = OpenCodeAdapter(model_id="opencode/free-1", protocol="responses", transport=transport)
    batch = TranslationBatchRequest(
        job_id="job_1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
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
    assert outcome.successes[0].translated_text == "Bản dịch 1"


def test_opencode_responses_protocol_accepts_current_output_message_shape():
    def fake_transport(url, headers, body, timeout):
        response_data = {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({
                                "translations": [
                                    {"id": "w1", "text": "Bản dịch 1"}
                                ]
                            }),
                        }
                    ],
                },
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = OpenCodeAdapter(model_id="opencode/free-1", protocol="responses", transport=transport)
    batch = TranslationBatchRequest(
        job_id="job_1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
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
    assert outcome.successes[0].translated_text == "Bản dịch 1"

def test_opencode_rate_limit_and_unauthorized_errors():
    def fake_transport_429(url, headers, body, timeout):
        return 429, {"Retry-After": "30"}, b"Rate limit exceeded"

    transport_429 = SafeHttpTransport(injected_transport=fake_transport_429)
    adapter = OpenCodeAdapter(model_id="opencode/free-1", transport=transport_429)
    batch = TranslationBatchRequest(
        job_id="job_1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.failures) == 1
    assert outcome.failures[0].kind == FailureKind.RATE_LIMIT
    assert outcome.failures[0].retry_after_seconds == 30.0

    def fake_transport_403(url, headers, body, timeout):
        return 403, {}, b"Forbidden: key required"

    adapter_403 = OpenCodeAdapter(
        model_id="opencode/free-1",
        transport=SafeHttpTransport(injected_transport=fake_transport_403),
    )
    outcome_403 = adapter_403.translate_batch(batch)
    assert len(outcome_403.failures) == 1
    assert outcome_403.failures[0].kind == FailureKind.UNAUTHORIZED
    assert outcome_403.failures[0].retryable is False

    def fake_transport_404(url, headers, body, timeout):
        return 404, {}, b"Model not found"

    adapter_404 = OpenCodeAdapter(
        model_id="opencode/unknown",
        transport=SafeHttpTransport(injected_transport=fake_transport_404),
    )
    outcome_404 = adapter_404.translate_batch(batch)
    assert outcome_404.failures[0].kind == FailureKind.UNSUPPORTED_MODEL

    def fake_transport_500(url, headers, body, timeout):
        return 500, {}, b"Internal error"

    adapter_500 = OpenCodeAdapter(
        model_id="opencode/free-1",
        transport=SafeHttpTransport(injected_transport=fake_transport_500),
    )
    outcome_500 = adapter_500.translate_batch(batch)
    assert outcome_500.failures[0].kind == FailureKind.PROVIDER_INTERNAL

def test_opencode_partial_missing_and_rejected_items():
    def fake_transport_partial(url, headers, body, timeout):
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "translations": [
                                {"id": "w1", "text": "Bản dịch 1"},
                                {"id": "w3_rejected", "text": "Lạ lùng"}
                            ]
                        })
                    }
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode("utf-8")

    adapter = OpenCodeAdapter(
        model_id="opencode/free-1",
        transport=SafeHttpTransport(injected_transport=fake_transport_partial),
    )
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
            TranslationRequestItem(
                work_id="w2",
                masked_text="Text 2",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert len(outcome.successes) == 1
    assert outcome.successes[0].work_id == "w1"
    # w2 should be in failures as MISSING_ITEM
    assert any(f.work_id == "w2" and f.kind == FailureKind.MISSING_ITEM for f in outcome.failures)

def test_opencode_empty_batch_and_malformed_json():
    adapter = OpenCodeAdapter(model_id="opencode/free-1")
    empty_batch = TranslationBatchRequest(job_id="j", source_language="en", target_language="vi", items=())
    res_empty = adapter.translate_batch(empty_batch)
    assert len(res_empty.successes) == 0

    def fake_transport_bad_json(url, headers, body, timeout):
        return 200, {"Content-Type": "application/json"}, b"not a json"

    adapter_bad = OpenCodeAdapter(
        model_id="opencode/free-1",
        transport=SafeHttpTransport(injected_transport=fake_transport_bad_json),
    )
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
                masked_text="Text",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter_bad.translate_batch(batch)
    assert len(outcome.failures) == 1
    assert outcome.failures[0].kind == FailureKind.INVALID_STRUCTURE



def test_opencode_auto_uses_responses_hint_for_muse_model():
    calls = []

    def fake_transport(url, headers, body, timeout):
        calls.append(url)
        assert url == "https://opencode.ai/zen/v1/responses"
        response_data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"translations": [{"id": "w1", "text": "OK"}]}),
                        }
                    ],
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = OpenCodeAdapter(model_id="muse-spark-1.3-contributor-free", transport=transport)
    batch = TranslationBatchRequest(
        job_id="auto_muse",
        source_language="en",
        target_language="vi",
        items=(TranslationRequestItem(
            work_id="w1", masked_text="Hello", context=(), context_signature="",
            hard_terms=(), soft_terms=(),
        ),),
    )

    outcome = adapter.translate_batch(batch)

    assert len(outcome.successes) == 1
    assert calls == ["https://opencode.ai/zen/v1/responses"]


def test_opencode_auto_falls_back_to_responses_only_for_fast_protocol_mismatch():
    calls = []

    def fake_transport(url, headers, body, timeout):
        calls.append(url)
        if url.endswith("/chat/completions"):
            return 500, {}, b"protocol mismatch"
        response_data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"translations": [{"id": "w1", "text": "Recovered"}]}),
                        }
                    ],
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response_data).encode("utf-8")

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = OpenCodeAdapter(model_id="future-free-model", transport=transport)
    batch = TranslationBatchRequest(
        job_id="auto_fallback",
        source_language="en",
        target_language="vi",
        items=(TranslationRequestItem(
            work_id="w1", masked_text="Hello", context=(), context_signature="",
            hard_terms=(), soft_terms=(),
        ),),
    )

    outcome = adapter.translate_batch(batch)

    assert len(outcome.successes) == 1
    assert [a.status_code for a in outcome.attempts] == [500, 200]
    assert calls == [
        "https://opencode.ai/zen/v1/chat/completions",
        "https://opencode.ai/zen/v1/responses",
    ]


def test_opencode_auto_does_not_retry_alternate_protocol_on_rate_limit():
    calls = []

    def fake_transport(url, headers, body, timeout):
        calls.append(url)
        return 429, {"Retry-After": "30"}, b"rate limited"

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = OpenCodeAdapter(model_id="future-free-model", transport=transport)
    batch = TranslationBatchRequest(
        job_id="auto_429",
        source_language="en",
        target_language="vi",
        items=(TranslationRequestItem(
            work_id="w1", masked_text="Hello", context=(), context_signature="",
            hard_terms=(), soft_terms=(),
        ),),
    )

    outcome = adapter.translate_batch(batch)

    assert outcome.failures[0].kind == FailureKind.RATE_LIMIT
    assert calls == ["https://opencode.ai/zen/v1/chat/completions"]
