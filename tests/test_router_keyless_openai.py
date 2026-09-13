import json

import pytest

from crimson0020.translation_runtime.contracts import (
    FailureKind,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.adapters.keyless_openai import KeylessOpenAIAdapter
from crimson0020.translation_runtime.router.transport import SafeHttpTransport


def _batch() -> TranslationBatchRequest:
    return TranslationBatchRequest(
        job_id="keyless",
        source_language="en",
        target_language="vi",
        items=(
            TranslationRequestItem(
                work_id="w1",
                masked_text="Hello <x0> traveler.",
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            ),
        ),
    )


def _large_batch(count: int, *, chars_per_item: int = 80) -> TranslationBatchRequest:
    return TranslationBatchRequest(
        job_id=f"keyless-{count}",
        source_language="en",
        target_language="vi",
        items=tuple(
            TranslationRequestItem(
                work_id=f"w{i:02d}",
                masked_text=(f"row {i:02d} " + "x" * chars_per_item),
                context=(),
                context_signature="",
                hard_terms=(),
                soft_terms=(),
            )
            for i in range(count)
        ),
    )


def _ids_from_prompt(body: bytes) -> list[str]:
    payload = json.loads(body.decode("utf-8"))
    prompt = payload["messages"][1]["content"]
    raw = prompt.split("<TRANSLATION_INPUT>\n", 1)[1].split("\n</TRANSLATION_INPUT>", 1)[0]
    request = json.loads(raw)
    return [item["id"] for item in request["items"]]


def test_slow_keyless_fallbacks_use_bounded_read_timeout():
    assert KeylessOpenAIAdapter("blockrun").transport.read_timeout_seconds == 12.0
    assert KeylessOpenAIAdapter("cehpoint").transport.read_timeout_seconds == 12.0
    assert KeylessOpenAIAdapter("llm7").transport.read_timeout_seconds == 30.0


@pytest.mark.parametrize(
    ("route_id", "model_id", "expected_host", "expects_dummy_auth"),
    [
        ("llm7", "default", "api.llm7.io", True),
        ("blockrun", "nvidia/gpt-oss-20b", "blockrun.ai", False),
        ("vireonix", "auto", "vireonix.ai", False),
        ("cehpoint", "cehpoint-ai", "ai-api.cehpoint.co.in", False),
    ],
)
def test_keyless_openai_route_success(route_id, model_id, expected_host, expects_dummy_auth):
    def fake_transport(url, headers, body, timeout):
        assert expected_host in url
        if expects_dummy_auth:
            assert headers.get("Authorization") == "Bearer unused"
        else:
            assert "Authorization" not in headers
        payload = json.loads(body.decode("utf-8"))
        assert payload["model"] == model_id
        assert payload["temperature"] == 0.0
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"translations": [{"id": "w1", "text": "Xin chào <x0> lữ khách."}]},
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(response).encode("utf-8")

    adapter = KeylessOpenAIAdapter(
        route_id=route_id,
        model_id=model_id,
        transport=SafeHttpTransport(injected_transport=fake_transport),
    )
    outcome = adapter.translate_batch(_batch())

    assert len(outcome.successes) == 1
    assert outcome.successes[0].route_id == route_id
    assert outcome.successes[0].translated_text == "Xin chào <x0> lữ khách."
    assert outcome.failures == ()


def test_keyless_openai_rejects_unknown_route_and_classifies_rate_limit():
    with pytest.raises(ValueError):
        KeylessOpenAIAdapter(route_id="arbitrary", model_id="m1")

    def limited(url, headers, body, timeout):
        return 429, {"Retry-After": "30"}, b"rate limited"

    adapter = KeylessOpenAIAdapter(
        route_id="llm7",
        model_id="default",
        transport=SafeHttpTransport(injected_transport=limited),
    )
    outcome = adapter.translate_batch(_batch())
    assert outcome.failures[0].kind == FailureKind.RATE_LIMIT
    assert outcome.failures[0].retryable is True


@pytest.mark.parametrize(
    ("route_id", "model_id", "expected_sizes"),
    [
        ("llm7", "default", [24, 24]),
        ("blockrun", "nvidia/gpt-oss-20b", [32, 16]),
        ("vireonix", "auto", [16, 16, 16]),
    ],
)
def test_large_batches_are_split_to_route_latency_budget(route_id, model_id, expected_sizes):
    seen_ids: list[list[str]] = []

    def fake_transport(url, headers, body, timeout):
        del url, headers, timeout
        ids = _ids_from_prompt(body)
        seen_ids.append(ids)
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

    adapter = KeylessOpenAIAdapter(
        route_id=route_id,
        model_id=model_id,
        transport=SafeHttpTransport(injected_transport=fake_transport),
    )
    outcome = adapter.translate_batch(_large_batch(48))

    assert [len(ids) for ids in seen_ids] == expected_sizes
    assert [item.work_id for item in outcome.successes] == [f"w{i:02d}" for i in range(48)]
    assert outcome.failures == ()
    assert len(outcome.attempts) == len(expected_sizes)


def test_llm7_split_stops_after_route_wide_chunk_failure():
    calls = 0

    def limited(url, headers, body, timeout):
        nonlocal calls
        del url, headers, body, timeout
        calls += 1
        return 429, {"Retry-After": "30"}, b"rate limited"

    adapter = KeylessOpenAIAdapter(
        route_id="llm7",
        model_id="default",
        transport=SafeHttpTransport(injected_transport=limited),
    )
    outcome = adapter.translate_batch(_large_batch(48))

    assert calls == 1
    assert len(outcome.successes) == 0
    assert len(outcome.failures) == 48
    assert {failure.kind for failure in outcome.failures} == {FailureKind.RATE_LIMIT}
    assert all(failure.retryable for failure in outcome.failures)
