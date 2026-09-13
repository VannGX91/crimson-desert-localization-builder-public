import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from crimson0020.translation_providers import (
    GoogleFreeProvider,
    RetryableTranslationProviderError,
    TranslatorsEngineProvider,
    create_translation_provider,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.providers.google import GoogleFreeAdapter
from crimson0020.translation_runtime.knowledge.memory import (
    TranslationMemoryRepository,
    TranslationSegment,
)
from crimson0020.translation_work import prepare_translation_work


def _catalog(path: Path, *, old: bool) -> Path:
    rows = [
        {
            "marker": "0000000000000001",
            "key": "HELLO",
            "original": "Hello",
            "translation": "Xin chào" if old else "Hello",
        },
        {
            "marker": "0000000000000002",
            "key": "NEW",
            "original": "New line" if not old else "Old line",
            "translation": "Old translation" if old else "New line",
        },
    ]
    data = {
        "metadata": {
            "format_version": 2,
            "layout": "multi-paloc",
            "package_folder": "0020",
            "palocs": [{"path": "synthetic.paloc", "entry_count": len(rows), "tail_bytes": ""}],
        },
        "entries": rows,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ProviderAndSqliteRemediationTests(unittest.TestCase):
    def test_google_uses_large_safe_reference_batch_limits(self) -> None:
        self.assertEqual(2_000, GoogleFreeProvider.preferred_batch_items)
        self.assertEqual(64_000, GoogleFreeProvider.preferred_batch_characters)
        self.assertEqual(2_000, GoogleFreeAdapter.MAX_ITEMS)
        self.assertEqual(64_000, GoogleFreeAdapter.MAX_CHARACTERS)

    def test_google_pygtrans_request_is_single_attempt_and_uses_configured_timeout(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            status_code = 200
            content = b'["Xin chao"]'

            @staticmethod
            def json():
                return ["Xin chao"]

        class FakeSession:
            def post(self, url, *, params, data, timeout):
                calls.append(
                    {
                        "url": url,
                        "params": params,
                        "data": data,
                        "timeout": timeout,
                    }
                )
                return FakeResponse()

        class FakeTranslate:
            TRANSLATE_URL = "https://translate.google.com/translate_a/t"

            def __init__(self, *, fmt, timeout):
                calls.append({"constructor_timeout": timeout, "fmt": fmt})
                self.session = FakeSession()

        fake_module = SimpleNamespace(Translate=FakeTranslate)
        provider = GoogleFreeProvider(timeout=7.5)
        with patch.dict("sys.modules", {"pygtrans": fake_module}):
            result = provider.translate(("Hello",), "en", "vi")

        self.assertEqual(("Xin chao",), result)
        self.assertEqual(7.5, calls[0]["constructor_timeout"])
        self.assertEqual("text", calls[0]["fmt"])
        self.assertEqual(7.5, calls[1]["timeout"])
        self.assertEqual(["Hello"], calls[1]["data"]["q"])
        self.assertEqual(2, len(calls))

    def test_google_post_attempt_network_error_does_not_duplicate_via_native_transport(self) -> None:
        native_calls: list[object] = []

        class FakeSession:
            @staticmethod
            def post(*_args, **_kwargs):
                raise OSError("connection reset after request dispatch")

        class FakeTranslate:
            TRANSLATE_URL = "https://translate.google.com/translate_a/t"

            def __init__(self, *, fmt, timeout):
                self.session = FakeSession()

        def native_transport(*args):
            native_calls.append(args)
            return 200, b'["Xin chao"]'

        provider = GoogleFreeProvider(timeout=7.5, transport=None)
        provider._transport = None
        fake_module = SimpleNamespace(Translate=FakeTranslate)
        with (
            patch.dict("sys.modules", {"pygtrans": fake_module}),
            patch(
                "crimson0020.translation_providers._default_transport",
                side_effect=native_transport,
            ),
        ):
            with self.assertRaises(RetryableTranslationProviderError):
                provider.translate(("Hello",), "en", "vi")

        self.assertEqual([], native_calls)

    def test_bing_factory_uses_translators_engine_with_native_fallback(self) -> None:
        provider = create_translation_provider("bing-web-free")
        self.assertIsInstance(provider, TranslatorsEngineProvider)
        self.assertEqual("bing", provider.engine_name)
        self.assertEqual(32, provider._max_workers)
        self.assertIsNotNone(provider._fallback_translator)
        self.assertEqual(900, provider._pack_characters)
        self.assertEqual(2_000, provider.preferred_batch_items)
        self.assertEqual(64_000, provider.preferred_batch_characters)

    def test_translators_engine_packing_preserves_order_and_internal_newlines(self) -> None:
        def translator(text: str, _source: str, _target: str) -> str:
            return text.replace("One", "Một").replace("Two", "Hai").replace("Three", "Ba")

        provider = TranslatorsEngineProvider(
            "test",
            "bing",
            max_workers=1,
            translator_func=translator,
            pack_characters=900,
        )
        result = provider.translate(("One", "Two\nline", "Three"), "en", "vi")
        self.assertEqual(("Một", "Hai\nline", "Ba"), result)

    def test_translators_engine_reports_exception_type_when_detail_is_blank(self) -> None:
        def translate_text(*_args, **_kwargs):
            raise RuntimeError()

        provider = TranslatorsEngineProvider("test", "bing", max_workers=1)
        fake_module = SimpleNamespace(translate_text=translate_text)
        with patch.dict("sys.modules", {"translators": fake_module}):
            with self.assertRaises(RetryableTranslationProviderError) as caught:
                provider.translate(("Hello",), "en", "vi")

        self.assertIn("RuntimeError", str(caught.exception))

    def test_step1_same_catalog_pair_does_not_reingest_legacy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = _catalog(root / "old.json", old=True)
            new = _catalog(root / "new.json", old=False)
            db = KnowledgeDatabase(root / "knowledge.sqlite3")
            first = prepare_translation_work(old, new, root / "first.json", knowledge_db=db)
            self.assertEqual(2, first.imported_segments_count)
            with patch(
                "crimson0020.translation_work._legacy_term_rows",
                side_effect=AssertionError("redundant legacy ingestion"),
            ):
                second = prepare_translation_work(old, new, root / "second.json", knowledge_db=db)
            self.assertEqual(0, second.imported_segments_count)
            self.assertIsNone(second.knowledge_warning)

    def test_identical_tm_upsert_is_a_true_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = KnowledgeDatabase(Path(td) / "knowledge.sqlite3")
            with db.transaction() as con:
                con.execute(
                    "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("scope", "app", "0020", "en", "vi", 1.0),
                )
                con.execute(
                    "INSERT INTO lineages (lineage_id, scope_id, created_at) VALUES (?, ?, ?)",
                    ("lineage", "scope", 1.0),
                )
            repo = TranslationMemoryRepository(db)
            first = TranslationSegment(
                segment_id="seg_1", scope_id="scope", lineage_id="lineage",
                marker="m", key="k", source_hash="h", source_text="Hello",
                target_text="Xin chào", context_signature="ctx", hard_terms_hash="terms",
                glossary_revision=1, status="active", created_at=10.0, updated_at=10.0,
            )
            repo.save_segment(first)
            second = TranslationSegment(
                segment_id="seg_1", scope_id="scope", lineage_id="lineage",
                marker="m", key="k", source_hash="h", source_text="Hello",
                target_text="Xin chào", context_signature="ctx", hard_terms_hash="terms",
                glossary_revision=1, status="active", created_at=20.0, updated_at=20.0,
            )
            repo.save_segment(second)
            with db.connect() as con:
                updated_at = con.execute(
                    "SELECT updated_at FROM translation_segments WHERE segment_id='seg_1'"
                ).fetchone()[0]
            self.assertEqual(10.0, updated_at)

    def test_reference_provider_dependencies_are_pinned(self) -> None:
        requirements = Path("requirements.txt").read_text(encoding="utf-8")
        self.assertIn("pygtrans==1.6.2", requirements)
        self.assertIn("translators==6.0.4", requirements)
        self.assertNotRegex(requirements, r"(?mi)^js2py==")
        self.assertNotRegex(requirements, r"(?mi)^ai[_-]cloudscraper==")


if __name__ == "__main__":
    unittest.main()
