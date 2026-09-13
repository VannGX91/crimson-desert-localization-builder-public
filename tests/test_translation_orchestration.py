"""Specification tests for Auto-translate orchestration and facades (INT-03)."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from crimson0020.auto_translate import (
    failure_path_for,
    translate_delta,
)
from crimson0020.translation_providers import (
    FreeWebTranslationProvider,
    GoogleFreeProvider,
    LibreTranslateProvider,
    MyMemoryFreeProvider,
    OpenAICompatibleProvider,
    RetryableTranslationProviderError,
    TranslationProviderError,
    _BingWebTranslator,
    _normalized_endpoint,
    _normalized_openai_endpoint,
    _status_error,
    create_translation_provider,
    get_translation_provider_spec,
)
from crimson0020.translation_runtime.checkpoint import (
    CheckpointRecord,
    append_checkpoint_batch,
    initialize_checkpoint,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.service import KnowledgeService, ScopeIdentifier

DUMMY_FP = "A" * 64

def _make_delta_json(path: Path, entries: list[dict[str, object]]) -> Path:
    data = {
        "schema": "crimson.translation-delta",
        "schema_version": 1,
        "kind": "sparse-delta",
        "source": {
            "catalog_fingerprint": DUMMY_FP,
        },
        "metadata": {
            "format_version": 2,
            "layout": "multi-paloc",
            "package_folder": "0020",
            "source_fingerprint": DUMMY_FP,
            "target_fingerprint": "B" * 64,
            "palocs": [
                {
                    "path": "gamedata/ui.paloc",
                    "entry_count": len(entries),
                    "tail_bytes": "",
                }
            ],
        },
        "entries": entries,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _make_full_catalog_json(path: Path, entries: list[dict[str, object]]) -> Path:
    data = {
        "metadata": {
            "format_version": 2,
            "layout": "multi-paloc",
            "package_folder": "0020",
            "palocs": [
                {
                    "path": "gamedata/ui.paloc",
                    "entry_count": len(entries),
                    "tail_bytes": "",
                }
            ],
        },
        "entries": entries,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class MockProvider:
    def __init__(self, key: str = "ai_router_free", fail_all: bool = False) -> None:
        self.key = key
        self.fail_all = fail_all
        self.batches_received: list[list[str]] = []

    def translate(
        self, texts: Sequence[str], source_lang: str, target_lang: str
    ) -> tuple[str, ...]:
        self.batches_received.append(list(texts))
        if self.fail_all:
            raise RuntimeError("All routes unavailable")
        return tuple(f"VI:{t}" for t in texts)


class TestInt03Readiness(unittest.TestCase):
    def test_resume_uses_current_protection_contract_for_natural_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            source = _make_delta_json(
                temp / "delta.json",
                [
                    {
                        "marker": "0100000000000000",
                        "key": "bracket",
                        "original": "Read [Crafting Method] now",
                        "translation": "",
                        "translation_state": "pending",
                    }
                ],
            )
            output = temp / "translated.json"
            db_path = temp / "knowledge.sqlite3"

            class NaturalBracketProvider(MockProvider):
                def translate(self, texts, source_lang, target_lang):
                    self.batches_received.append(list(texts))
                    return tuple(
                        text.replace("Read", "Doc")
                        .replace("Crafting Method", "Phuong phap che tao")
                        .replace("now", "ngay")
                        for text in texts
                    )

            first = NaturalBracketProvider("google-web-free")
            translate_delta(source, output, first, "en", "vi", db_path=db_path)
            resumed_provider = MockProvider("google-web-free")

            resumed = translate_delta(
                source, output, resumed_provider, "en", "vi", db_path=db_path
            )
            translated = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual([], resumed_provider.batches_received)
        self.assertEqual(0, resumed.pending_entries)
        self.assertEqual("Doc [Phuong phap che tao] ngay", translated[0]["translation"])

    def test_production_step2_masks_hard_glossary_and_preserves_only_technical_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            source = _make_delta_json(
                temp / "delta.json",
                [
                    {"marker": "0100000000000000", "key": "abyss", "original": "Abyss Equipment", "translation": "", "translation_state": "pending"},
                    {"marker": "0200000000000000", "key": "money", "original": "Give Coin ({Money:Money_Copper:1})", "translation": "", "translation_state": "pending"},
                    {"marker": "0300000000000000", "key": "bracket", "original": "Read [Crafting Method] now", "translation": "", "translation_state": "pending"},
                    {"marker": "0400000000000000", "key": "mixed", "original": "Abyss {Money:Money_Copper:1}", "translation": "", "translation_state": "pending"},
                ],
            )
            output = temp / "translated.json"
            db_path = temp / "knowledge.sqlite3"
            database = KnowledgeDatabase(db_path)
            scope = ScopeIdentifier("Crimson0020", "0020", "en", "vi")
            KnowledgeService(database).set_glossary_override(scope, "Abyss", "Abyss")

            class GlossaryIgnoringProvider(MockProvider):
                def translate(self, texts, source_lang, target_lang):
                    self.batches_received.append(list(texts))
                    return tuple(
                        text.replace("Abyss", "Vực thẳm")
                        .replace("Equipment", "Trang bị")
                        .replace("Give Coin", "Đưa tiền")
                        .replace("Crafting Method", "Phương pháp chế tạo")
                        .replace("Read", "Đọc")
                        .replace("now", "ngay")
                        for text in texts
                    )

            provider = GlossaryIgnoringProvider("google-web-free")
            result = translate_delta(source, output, provider, "en", "vi", db_path=db_path)
            translated = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, result.pending_entries)
        sent = [text for batch in provider.batches_received for text in batch]
        self.assertFalse(any("Abyss" in text for text in sent))
        self.assertTrue(any("Crafting Method" in text for text in sent))
        self.assertFalse(any("Money_Copper" in text for text in sent))
        self.assertEqual("Abyss Trang bị", translated[0]["translation"])
        self.assertEqual("Đưa tiền ({Money:Money_Copper:1})", translated[1]["translation"])
        self.assertEqual("Đọc [Phương pháp chế tạo] ngay", translated[2]["translation"])
        self.assertEqual("Abyss {Money:Money_Copper:1}", translated[3]["translation"])

    def _delta(self, path: Path) -> Path:
        return _make_delta_json(
            path,
            [
                {
                    "marker": "0100000000000000",
                    "key": "k1",
                    "original": "First string",
                    "translation": "First string",
                    "translation_state": "pending",
                },
                {
                    "marker": "0200000000000000",
                    "key": "k2",
                    "original": "Second string",
                    "translation": "Second string",
                    "translation_state": "pending",
                },
            ],
        )

    def test_crash_injection_after_every_commit_boundary(self) -> None:
        for stage in ("checkpoint", "database", "output", "failure_state"):
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory() as raw_temp:
                    temp = Path(raw_temp)
                    source = self._delta(temp / "delta.json")
                    output = temp / "translated.json"
                    db_path = temp / "knowledge.sqlite3"
                    seen: list[str] = []

                    def crash_after(name: str) -> None:
                        seen.append(name)
                        if name == stage:
                            raise RuntimeError(f"injected crash after {name}")

                    provider: MockProvider
                    if stage == "failure_state":
                        provider = MockProvider(fail_all=True)
                    else:
                        provider = MockProvider()

                    with self.assertRaisesRegex(RuntimeError, "injected crash"):
                        translate_delta(
                            source,
                            output,
                            provider,
                            "en",
                            "vi",
                            on_commit_stage=crash_after,
                            db_path=db_path,
                        )
                    self.assertIn(stage, seen)
                    if stage in {"checkpoint", "database", "output"}:
                        self.assertTrue(
                            Path(f"{output}.autotranslate-checkpoint.jsonl").is_file()
                        )
                    if stage == "output":
                        self.assertTrue(output.is_file())
                    if stage == "failure_state":
                        self.assertTrue(
                            Path(f"{output}.autotranslate-failures.jsonl").is_file()
                        )

        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            source = self._delta(temp / "successful-delta.json")
            output = temp / "successful-output.json"
            successful_stages: list[str] = []
            translate_delta(
                source,
                output,
                MockProvider(),
                "en",
                "vi",
                on_commit_stage=successful_stages.append,
                db_path=temp / "successful.sqlite3",
            )
            self.assertEqual(
                successful_stages,
                ["checkpoint", "database", "output", "failure_state"],
            )

    def test_database_unavailable_does_not_abort_translation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            source = self._delta(temp / "delta.json")
            output = temp / "translated.json"
            unavailable_db = temp / "database-directory"
            unavailable_db.mkdir()

            result = translate_delta(
                source,
                output,
                MockProvider(),
                "en",
                "vi",
                db_path=unavailable_db,
            )

            self.assertEqual(result.translated_entries, 2)
            self.assertEqual(result.pending_entries, 0)
            self.assertTrue(output.is_file())

    def test_pending_completion_is_explicit_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = Path(raw_temp)
            source = self._delta(temp / "delta.json")
            output = temp / "translated.json"

            class PartiallyUnavailableProvider(MockProvider):
                def translate(self, texts, source_lang, target_lang):
                    if any(text == "Second string" for text in texts):
                        raise RuntimeError("second row unavailable")
                    return super().translate(texts, source_lang, target_lang)

            result = translate_delta(
                source,
                output,
                PartiallyUnavailableProvider(),
                "en",
                "vi",
            )

            self.assertEqual(result.translated_entries, 1)
            self.assertEqual(result.pending_entries, 1)
            output_data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                output_data["entries"][0]["translation_state"], "translated"
            )
            self.assertIn(
                output_data["entries"][1]["translation_state"], {"failed", "pending"}
            )

    def test_provider_facade_rejects_invalid_and_malformed_responses(self) -> None:
        with self.assertRaises(ValueError):
            LibreTranslateProvider(key="x", endpoint="https://host.test/bad")
        with self.assertRaises(TranslationProviderError):
            LibreTranslateProvider(
                key="x",
                endpoint="https://host.test",
                transport=lambda *_args: (200, b"not-json"),
            ).translate(("text",), "en", "vi")
        with self.assertRaises(TranslationProviderError):
            GoogleFreeProvider(
                transport=lambda *_args: (200, b"{}")
            ).translate(("text",), "en", "vi")
        with self.assertRaises(TranslationProviderError):
            MyMemoryFreeProvider(
                transport=lambda *_args: (
                    200,
                    json.dumps({"responseData": {}}).encode("utf-8"),
                )
            ).translate(("text",), "en", "vi")

    def test_provider_facade_validates_batches_and_error_classes(self) -> None:
        libre = LibreTranslateProvider(
            key="x",
            endpoint="https://host.test",
            transport=lambda *_args: (500, b'{"error":"down"}'),
        )
        with self.assertRaises(RetryableTranslationProviderError):
            libre.translate(("text",), "en", "vi")

        google = GoogleFreeProvider(
            transport=lambda *_args: (
                200,
                json.dumps(["translated", "extra"]).encode("utf-8"),
            )
        )
        with self.assertRaises(TranslationProviderError):
            google.translate(("text",), "en", "vi")

        for provider in (
            google,
            MyMemoryFreeProvider(
                transport=lambda *_args: (
                    200,
                    json.dumps(
                        {"responseData": {"translatedText": "translated"}}
                    ).encode("utf-8"),
                )
            ),
        ):
            with self.assertRaises(ValueError):
                provider.translate((), "en", "vi")

    def test_provider_facade_covers_web_and_openai_success_paths(self) -> None:
        free = FreeWebTranslationProvider(
            "synthetic",
            lambda text, _source, _target: f"vi:{text}",
            max_workers=1,
        )
        self.assertEqual(("vi:text",), free.translate(("text",), "en", "vi"))

        openai = OpenAICompatibleProvider(
            "https://gateway.test/v1",
            "model",
            "secret",
            transport=lambda *_args: (
                200,
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "```json\n{\"0\":\"vi:text\"}\n```"
                                }
                            }
                        ]
                    }
                ).encode("utf-8"),
            ),
        )
        self.assertEqual(("vi:text",), openai.translate(("text",), "en", "vi"))

        self.assertEqual(
            "google-web-free",
            create_translation_provider("google_free").key,
        )
        self.assertEqual(
            "mymemory-free",
            create_translation_provider("mymemory_free").key,
        )

    def test_provider_facade_covers_bing_session_and_expiry(self) -> None:
        translator = _BingWebTranslator(timeout=1)
        page = (
            'var params_AbusePreventionHelper = ["key","token"];'
            'IG:"ig-value" id="tta_outGDCont" data-iid="iid-value"'
        ).encode("utf-8")
        responses = [
            (200, page),
            (
                200,
                json.dumps(
                    [{"translations": [{"text": "vi:text"}]}]
                ).encode("utf-8"),
            ),
        ]
        translator._open = lambda _request, _limit: responses.pop(0)
        self.assertEqual("vi:text", translator("text", "en", "vi"))
        self.assertIsNotNone(translator._credentials)

        retry_responses = [
            (401, b'{"error":"expired"}'),
            (200, page),
            (
                200,
                json.dumps(
                    [{"translations": [{"text": "vi:text"}]}]
                ).encode("utf-8"),
            ),
        ]
        translator._open = lambda _request, _limit: retry_responses.pop(0)
        with self.assertRaises(TranslationProviderError):
            translator("text", "en", "vi")
        self.assertIsNone(translator._credentials)
        self.assertEqual("vi:text", translator("text", "en", "vi"))

    def test_provider_facade_covers_validation_and_factory_error_branches(self) -> None:
        with self.assertRaises(ValueError):
            get_translation_provider_spec(None)
        with self.assertRaises(ValueError):
            LibreTranslateProvider(
                key="x",
                endpoint="https://host.test",
            ).translate((), "en", "vi")
        for body in (b"{}", b'{"translatedText":[1]}', b"\xff"):
            with self.assertRaises(TranslationProviderError):
                LibreTranslateProvider(
                    key="x",
                    endpoint="https://host.test",
                    transport=lambda *_args, body=body: (200, body),
                ).translate(("text",), "en", "vi")
        with self.assertRaises(TranslationProviderError):
            LibreTranslateProvider(
                key="x",
                endpoint="https://host.test",
                transport=lambda *_args: (400, b"{}"),
            ).translate(("text",), "en", "vi")

        for translator in (
            lambda *_args: (_ for _ in ()).throw(RuntimeError("temporary")),
            lambda *_args: "",
        ):
            provider = FreeWebTranslationProvider("x", translator, max_workers=1)
            with self.assertRaises(TranslationProviderError):
                provider.translate(("text",), "en", "vi")
        with self.assertRaises(ValueError):
            FreeWebTranslationProvider("x", lambda *_args: "ok").translate(
                (), "en", "vi"
            )

        for status, body in ((404, b"{}"), (200, b"\xff")):
            with self.assertRaises(TranslationProviderError):
                MyMemoryFreeProvider(
                    transport=lambda *_args, status=status, body=body: (status, body)
                ).translate(("text",), "en", "vi")
        with self.assertRaises(TranslationProviderError):
            MyMemoryFreeProvider(
                transport=lambda *_args: (
                    200,
                    json.dumps(
                        {"responseData": {"translatedText": ""}}
                    ).encode("utf-8"),
                )
            ).translate(("text",), "en", "vi")

        for status, body in ((400, b"{}"), (200, b"\xff"), (200, b"{}")):
            with self.assertRaises(TranslationProviderError):
                GoogleFreeProvider(
                    transport=lambda *_args, status=status, body=body: (status, body)
                ).translate(("text",), "en", "vi")
        with self.assertRaises(ValueError):
            GoogleFreeProvider().translate(("text",), "", "vi")

        with self.assertRaises(ValueError):
            OpenAICompatibleProvider("http://remote.test", "m", "k")
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(
                "https://gateway.test",
                "m",
                "k",
                transport=lambda *_args: (200, b"{}"),
            ).translate((), "en", "vi")
        with self.assertRaises(TranslationProviderError):
            OpenAICompatibleProvider(
                "https://gateway.test",
                "m",
                "k",
                transport=lambda *_args: (500, b"{}"),
            ).translate(("text",), "en", "vi")
        with self.assertRaises(TranslationProviderError):
            OpenAICompatibleProvider(
                "https://gateway.test",
                "m",
                "k",
                transport=lambda *_args: (
                    200,
                    json.dumps(
                        {"choices": [{"message": {"content": "not-json"}}]}
                    ).encode("utf-8"),
                ),
            ).translate(("text",), "en", "vi")

        self.assertEqual(
            "libretranslate-cutie-free",
            create_translation_provider(
                "libretranslate-cutie-free"
            ).key,
        )
        self.assertEqual(
            "bing-web-free",
            create_translation_provider("bing-web-free").key,
        )

    def test_provider_facade_covers_endpoint_and_status_helpers(self) -> None:
        for endpoint in ("", "https://host.test?query=1", "http://remote.test"):
            with self.assertRaises(ValueError):
                _normalized_endpoint(endpoint)
        self.assertEqual(
            "http://localhost:1234/translate",
            _normalized_endpoint("http://localhost:1234"),
        )
        with self.assertRaises(ValueError):
            _normalized_openai_endpoint("https://user:pass@host.test")
        with self.assertRaises(ValueError):
            _normalized_openai_endpoint("https://host.test?query=1")
        with self.assertRaises(ValueError):
            _normalized_openai_endpoint("")
        self.assertEqual(
            "http://localhost:1234/chat/completions",
            _normalized_openai_endpoint("http://localhost:1234"),
        )
        retryable = _status_error(500, b'{"error":{"message":"temporary"}}')
        self.assertIsInstance(retryable, RetryableTranslationProviderError)
        non_retryable = _status_error(400, b"\xff")
        self.assertIsInstance(non_retryable, TranslationProviderError)

        captured: dict[str, object] = {}
        provider = LibreTranslateProvider(
            key="x",
            endpoint="https://host.test",
            api_key="session-key",
            transport=lambda url, payload, headers, timeout: (
                captured.update(
                    url=url, payload=payload, headers=headers, timeout=timeout
                )
                or (200, json.dumps({"translatedText": "ok"}).encode("utf-8"))
            ),
        )
        self.assertEqual(("ok",), provider.translate(("text",), "en", "vi"))
        self.assertIn("api_key", json.loads(captured["payload"]))
        with self.assertRaises(ValueError):
            provider.translate(("text",), "", "vi")

    def test_provider_facade_covers_transport_and_boundary_errors(self) -> None:
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _limit):
                return b"body"

        with patch(
            "crimson0020.translation_providers.urlopen",
            return_value=Response(),
        ):
            from crimson0020.translation_providers import (
                _default_get_transport,
                _default_transport,
            )

            self.assertEqual(
                (200, b"body"),
                _default_transport("https://host.test", b"{}", {}, 1),
            )
            self.assertEqual(
                (200, b"body"),
                _default_get_transport("https://host.test", {}, 1),
            )

        error_response = io.BytesIO(b"provider error")
        with patch(
            "crimson0020.translation_providers.urlopen",
            side_effect=HTTPError(
                "https://host.test", 503, "busy", {}, error_response
            ),
        ):
            from crimson0020.translation_providers import _default_transport

            self.assertEqual(
                (503, b"provider error"),
                _default_transport("https://host.test", b"{}", {}, 1),
            )

        with patch(
            "crimson0020.translation_providers.urlopen",
            side_effect=URLError("offline"),
        ):
            from crimson0020.translation_providers import _default_get_transport

            with self.assertRaises(RetryableTranslationProviderError):
                _default_get_transport("https://host.test", {}, 1)

        class OfflineOpener:
            def open(self, *_args, **_kwargs):
                raise URLError("offline")

        bing = _BingWebTranslator()
        bing._opener = OfflineOpener()
        with self.assertRaises(RetryableTranslationProviderError):
            bing._open(None, 100)

        with self.assertRaises(ValueError):
            LibreTranslateProvider("", "https://host.test")
        with self.assertRaises(ValueError):
            LibreTranslateProvider("x", "https://host.test", timeout=0)
        with self.assertRaises(ValueError):
            FreeWebTranslationProvider("", lambda *_args: "ok")
        with self.assertRaises(ValueError):
            FreeWebTranslationProvider("x", lambda *_args: "ok", max_workers=0)
        with self.assertRaises(ValueError):
            MyMemoryFreeProvider(timeout=0)
        with self.assertRaises(ValueError):
            GoogleFreeProvider(timeout=0)

        long_token = "x" * 700
        provider = MyMemoryFreeProvider(
            transport=lambda url, _headers, _timeout: (
                200,
                json.dumps(
                    {"responseData": {"translatedText": url.split("q=", 1)[1]}}
                ).encode("utf-8"),
            )
        )
        self.assertTrue(provider.translate((long_token,), "en", "vi")[0])

        for content in ('{"0":"ok","1":"extra"}', '{"0":1}'):
            provider = OpenAICompatibleProvider(
                "https://gateway.test",
                "m",
                "k",
                transport=lambda *_args, content=content: (
                    200,
                    json.dumps(
                        {"choices": [{"message": {"content": content}}]}
                    ).encode("utf-8"),
                ),
            )
            with self.assertRaises(TranslationProviderError):
                provider.translate(("text",), "en", "vi")


def test_provider_specs_contains_required_three() -> None:
    # Requirement: The UI/provider registry contains exactly:
    # - ai_router_free
    # - google_free
    # - mymemory_free
    for req_key in ("ai_router_free", "google_free", "mymemory_free"):
        spec = get_translation_provider_spec(req_key)
        assert spec is not None
        assert spec.key == req_key
        assert spec.free is True


def test_full_catalog_rejected_by_translate_delta(tmp_path: Path) -> None:
    full_cat = tmp_path / "full_cat.json"
    _make_full_catalog_json(
        full_cat,
        [
            {
                "marker": "0100000000000000",
                "key": "k1",
                "original": "Hello",
                "translation": "Hello",
            }
        ],
    )
    output = tmp_path / "out.json"
    provider = MockProvider()

    try:
        translate_delta(full_cat, output, provider, "en", "vi")
    except ValueError as error:
        assert "sparse translation delta" in str(error)
    else:
        raise AssertionError("full catalog was accepted as a translation delta")


def test_transaction_commit_order_and_resume_recovery(tmp_path: Path) -> None:
    delta_file = tmp_path / "delta.json"
    _make_delta_json(
        delta_file,
        [
            {
                "marker": "0100000000000000",
                "key": "k1",
                "original": "First string",
                "translation": "First string",
                "translation_state": "pending",
            },
            {
                "marker": "0200000000000000",
                "key": "k2",
                "original": "Second string",
                "translation": "Second string",
                "translation_state": "pending",
            },
        ],
    )
    output = tmp_path / "out.json"
    provider = MockProvider("ai_router_free")

    # Injected stage tracking for commit order:
    # 1. checkpoint -> 2. database -> 3. partial output -> 4. failure state
    commit_stages: list[str] = []

    def on_stage(stage_name: str) -> None:
        commit_stages.append(stage_name)

    result = translate_delta(
        delta_file,
        output,
        provider,
        "en",
        "vi",
        on_commit_stage=on_stage,
    )

    assert result.translated_entries == 2
    assert output.is_file()
    # Verify commit order occurred: checkpoint, database, output
    assert "checkpoint" in commit_stages
    assert "database" in commit_stages
    assert "output" in commit_stages
    assert commit_stages.index("checkpoint") < commit_stages.index("database")
    assert commit_stages.index("database") < commit_stages.index("output")


def test_resume_imports_checkpoint_rows_missing_from_database(tmp_path: Path) -> None:
    # Simulates a crash after checkpoint write before database write,
    # or existing checkpoint rows missing from knowledge database on resume.
    delta_file = tmp_path / "delta.json"
    _make_delta_json(
        delta_file,
        [
            {
                "marker": "0100000000000000",
                "key": "k1",
                "original": "First string",
                "translation": "First string",
                "translation_state": "pending",
            },
            {
                "marker": "0200000000000000",
                "key": "k2",
                "original": "Second string",
                "translation": "Second string",
                "translation_state": "pending",
            },
        ],
    )
    output = tmp_path / "out.json"
    from crimson0020.auto_translate import checkpoint_path_for
    checkpoint_file = checkpoint_path_for(output)

    initialize_checkpoint(
        checkpoint_file,
        source_fingerprint=DUMMY_FP,
        source_language="en",
        target_language="vi",
        provider="ai_router_free",
    )
    append_checkpoint_batch(
        checkpoint_file,
        (
            CheckpointRecord(
                marker="0100000000000000",
                key="k1",
                original_sha256=hashlib.sha256("First string".encode("utf-8")).hexdigest().upper(),
                translation="VI:First string",
                route_id="r1",
                model_id="m1",
            ),
        ),
    )

    provider = MockProvider("google_free")
    res = translate_delta(delta_file, output, provider, "en", "vi")

    # k1 was resumed from checkpoint, k2 translated
    assert res.resumed_entries == 1
    assert res.translated_entries == 1
    out_data = json.loads(output.read_text(encoding="utf-8"))
    assert out_data["entries"][0]["translation"] == "VI:First string"
    assert out_data["entries"][1]["translation"] == "VI:Second string"


def test_malformed_partial_result_isolated_not_global_stop(tmp_path: Path) -> None:
    delta_file = tmp_path / "delta.json"
    _make_delta_json(
        delta_file,
        [
            {
                "marker": "0100000000000000",
                "key": "k1",
                "original": "Good string",
                "translation": "Good string",
                "translation_state": "pending",
            },
            {
                "marker": "0200000000000000",
                "key": "k2",
                "original": "Bad string",
                "translation": "Bad string",
                "translation_state": "pending",
            },
        ],
    )
    output = tmp_path / "out.json"

    # Provider returns empty string or format error on Bad string
    class BisectionProvider:
        key = "ai_router_free"

        def translate(self, texts, src, tgt):
            if len(texts) > 1:
                # Return corrupted count or empty
                return ("VI:Good string", "")
            if texts[0] == "Good string":
                return ("VI:Good string",)
            # Bad string fails
            raise RuntimeError("Provider parse error on Bad string")

    res = translate_delta(delta_file, output, BisectionProvider(), "en", "vi")
    assert res.translated_entries == 1
    assert res.failure_count == 1
    assert res.pending_entries == 1

    out_data = json.loads(output.read_text(encoding="utf-8"))
    assert out_data["entries"][0]["translation_state"] == "translated"
    assert out_data["entries"][1]["translation_state"] in ("failed", "pending")


def test_all_routes_unavailable_leaves_pending(tmp_path: Path) -> None:
    delta_file = tmp_path / "delta.json"
    _make_delta_json(
        delta_file,
        [
            {
                "marker": "0100000000000000",
                "key": "k1",
                "original": "Need translation",
                "translation": "Need translation",
                "translation_state": "pending",
            }
        ],
    )
    output = tmp_path / "out.json"
    provider = MockProvider(fail_all=True)

    res = translate_delta(delta_file, output, provider, "en", "vi")
    assert res.translated_entries == 0
    assert res.pending_entries == 1
    assert res.failure_count == 1


def test_failure_sidecar_rewrites_across_resumes_and_clears_after_success(
    tmp_path: Path,
) -> None:
    delta_file = tmp_path / "delta.json"
    _make_delta_json(
        delta_file,
        [
            {
                "marker": "0100000000000000",
                "key": "k1",
                "original": "First string",
                "translation": "First string",
                "translation_state": "pending",
            },
            {
                "marker": "0200000000000000",
                "key": "k2",
                "original": "Second string",
                "translation": "Second string",
                "translation_state": "pending",
            },
        ],
    )
    output = tmp_path / "out.json"

    class PartiallyUnavailableProvider(MockProvider):
        def translate(self, texts, source_lang, target_lang):
            if any(text == "Second string" for text in texts):
                raise RuntimeError("second row unavailable")
            return super().translate(texts, source_lang, target_lang)

    first = translate_delta(
        delta_file,
        output,
        PartiallyUnavailableProvider("ai_router_free"),
        "en",
        "vi",
    )
    assert first.pending_entries == 1

    second = translate_delta(
        delta_file,
        output,
        PartiallyUnavailableProvider("google_free"),
        "en",
        "vi",
    )
    assert second.resumed_entries == 1
    assert second.pending_entries == 1

    failure_path = failure_path_for(output)
    failure_records = [
        json.loads(line)
        for line in failure_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(failure_records) == 1
    assert (failure_records[0]["marker"], failure_records[0]["key"]) == (
        "0200000000000000",
        "k2",
    )
    assert failure_records[0]["provider"] == "google_free"

    completed = translate_delta(
        delta_file,
        output,
        MockProvider("mymemory_free"),
        "en",
        "vi",
    )
    assert completed.resumed_entries == 1
    assert completed.pending_entries == 0
    assert not failure_path.exists()
