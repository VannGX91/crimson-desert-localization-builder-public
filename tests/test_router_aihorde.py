import json
from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.aihorde import AIHordeAdapter
from crimson0020.translation_runtime.router.transport import SafeHttpTransport

def test_aihorde_async_submit_poll_success():
    call_counts = [0]

    def fake_transport(url, headers, body, timeout):
        call_counts[0] += 1
        assert "stablehorde.net" in url
        # Check anonymous sentinel header (apikey: 0000000000)
        assert headers.get("apikey") == "0000000000"
        assert headers.get("Client-Agent") != ""
        assert headers.get("User-Agent", "").startswith("Crimson0020-TranslationRuntime/")

        # 1. Submit request
        if "/api/v2/generate/text/async" in url:
            payload = json.loads(body.decode("utf-8"))
            assert payload["params"]["max_length"] == 512
            assert payload["params"]["max_context_length"] == 4096
            assert payload["params"]["n"] == 1
            return 202, {}, json.dumps({"id": "job_nonce_123"}).encode("utf-8")

        # 2. Check status request
        if "/api/v2/generate/text/status/job_nonce_123" in url:
            resp_data = {
                "done": True,
                "faulted": False,
                "generations": [
                    {
                        "text": json.dumps({
                            "translations": [
                                {"id": "h1", "text": "Dịch Horde"}
                            ]
                        })
                    }
                ]
            }
            return 200, {}, json.dumps(resp_data).encode("utf-8")

        return 404, {}, b"Not found"

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = AIHordeAdapter(model_id="aphrodite/free", transport=transport, poll_interval=0.01)
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="h1",
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
    assert outcome.successes[0].translated_text == "Dịch Horde"
    assert outcome.attempts[0].route_id == "aihorde"

def test_aihorde_timeout():
    def fake_transport(url, headers, body, timeout):
        if "/api/v2/generate/text/async" in url:
            return 202, {}, json.dumps({"id": "job_nonce_timeout"}).encode("utf-8")
        if "/api/v2/generate/text/status/" in url:
            # Always still waiting
            return 200, {}, json.dumps({"done": False, "wait_time": 100}).encode("utf-8")
        return 404, {}, b""

    transport = SafeHttpTransport(injected_transport=fake_transport)
    adapter = AIHordeAdapter(
        model_id="aphrodite/free",
        transport=transport,
        poll_interval=0.01,
        total_deadline_seconds=0.05,
    )
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="h1",
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
    assert outcome.failures[0].kind == FailureKind.TIMEOUT
    assert outcome.failures[0].retryable is True

def test_aihorde_worker_faulted():
    def fake_transport(url, headers, body, timeout):
        if "/api/v2/generate/text/async" in url:
            return 202, {}, json.dumps({"id": "job_fault"}).encode("utf-8")
        if "/api/v2/generate/text/status/" in url:
            return 200, {}, json.dumps({"faulted": True}).encode("utf-8")
        return 404, {}, b""

    adapter = AIHordeAdapter(
        model_id="aphrodite/free",
        transport=SafeHttpTransport(injected_transport=fake_transport),
        poll_interval=0.01,
    )
    batch = TranslationBatchRequest(
        job_id="j1",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="h1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )
    outcome = adapter.translate_batch(batch)
    assert outcome.failures[0].kind == FailureKind.PROVIDER_INTERNAL

def test_aihorde_submit_errors():
    batch = TranslationBatchRequest(
        job_id="j",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="h1",
                masked_text="Text 1",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )

    def fake_429(url, headers, body, timeout):
        return 429, {}, b"Rate limited"
    adapter_429 = AIHordeAdapter(transport=SafeHttpTransport(injected_transport=fake_429))
    assert adapter_429.translate_batch(batch).failures[0].kind == FailureKind.RATE_LIMIT

    def fake_401(url, headers, body, timeout):
        return 401, {}, b"Unauthorized"
    adapter_401 = AIHordeAdapter(transport=SafeHttpTransport(injected_transport=fake_401))
    assert adapter_401.translate_batch(batch).failures[0].kind == FailureKind.UNAUTHORIZED

    def fake_404(url, headers, body, timeout):
        return 404, {}, b"Not found"
    adapter_404 = AIHordeAdapter(transport=SafeHttpTransport(injected_transport=fake_404))
    assert adapter_404.translate_batch(batch).failures[0].kind == FailureKind.UNSUPPORTED_MODEL

    def fake_500(url, headers, body, timeout):
        return 500, {}, b"Server error"
    adapter_500 = AIHordeAdapter(transport=SafeHttpTransport(injected_transport=fake_500))
    assert adapter_500.translate_batch(batch).failures[0].kind == FailureKind.PROVIDER_INTERNAL

    def fake_bad_submit(url, headers, body, timeout):
        return 200, {}, b"not json"
    adapter_bad = AIHordeAdapter(transport=SafeHttpTransport(injected_transport=fake_bad_submit))
    assert adapter_bad.translate_batch(batch).failures[0].kind == FailureKind.INVALID_STRUCTURE

def test_aihorde_empty_batch():
    adapter = AIHordeAdapter()
    empty = TranslationBatchRequest(job_id="j", source_language="en", target_language="vi", items=())
    assert len(adapter.translate_batch(empty).successes) == 0


def test_aihorde_exposes_bounded_scheduler_profile():
    adapter = AIHordeAdapter()
    assert adapter.scheduler_batch_limits == (8, 2_000)
    assert adapter.total_deadline_seconds == 15.0
