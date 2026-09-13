from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.duckduckgo import DuckDuckGoAdapter
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

def test_duckduckgo_session_and_chat_success():
    def fake_transport(url, headers, body, timeout):
        # 1. Status / token request
        if "duckchat/v1/status" in url:
            assert headers.get("x-vqd-accept") == "1"
            return 200, {"x-vqd-4": "test-vqd-token-123"}, b""

        # 2. Chat completion request
        if "duckchat/v1/chat" in url:
            assert headers.get("x-vqd-4") == "test-vqd-token-123"
            stream_body = (
                b'data: {"message": "{\\"translations\\": [{\\"id\\": \\"d1\\", \\"text\\": \\"Dich DDG\\"}]}"}\n\n'
                b'data: [DONE]\n\n'
            )
            return 200, {"Content-Type": "text/event-stream"}, stream_body

        return 404, {}, b"Not found"

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = DuckDuckGoAdapter(model_id="gpt-4o-mini", transport=transport)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="d1",
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
    assert outcome.successes[0].translated_text == "Dich DDG"

def test_duckduckgo_token_refresh_on_expiry():
    tokens = ["token-1", "token-2"]

    def fake_transport(url, headers, body, timeout):
        if "status" in url:
            return 200, {"x-vqd-4": tokens.pop(0)}, b""
        if "chat" in url:
            if headers.get("x-vqd-4") == "token-1":
                # Token expired
                return 403, {}, b"Header x-vqd-4 invalid or expired"
            # Second attempt with refreshed token succeeds
            stream_body = b'data: {"message": "{\\"translations\\": [{\\"id\\": \\"d1\\", \\"text\\": \\"Dich Success\\"}]}"}\n\ndata: [DONE]\n\n'
            return 200, {"Content-Type": "text/event-stream"}, stream_body
        return 404, {}, b""

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = DuckDuckGoAdapter(model_id="gpt-4o-mini", transport=transport)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="d1",
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
    assert outcome.successes[0].translated_text == "Dich Success"

def test_duckduckgo_challenge_or_unsupported():
    def fake_transport(url, headers, body, timeout):
        if "status" in url:
            return 403, {}, b"Bot challenge triggered"
        return 404, {}, b""

    adapter = DuckDuckGoAdapter(
        model_id="gpt-4o-mini",
        transport=SafeHttpTransport(injected_transport=fake_transport),
    )
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="d1",
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

def test_duckduckgo_empty_batch_and_rate_limit():
    adapter = DuckDuckGoAdapter()
    empty = TranslationBatchRequest(job_id="j", source_language="en", target_language="vi", items=())
    assert len(adapter.translate_batch(empty).successes) == 0

    def fake_429(url, headers, body, timeout):
        if "status" in url:
            return 200, {"x-vqd-4": "tok"}, b""
        return 429, {"Retry-After": "60"}, b"Rate limit"

    adapter_429 = DuckDuckGoAdapter(transport=SafeHttpTransport(injected_transport=fake_429))
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="d1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    res_429 = adapter_429.translate_batch(batch)
    assert res_429.failures[0].kind == FailureKind.RATE_LIMIT
