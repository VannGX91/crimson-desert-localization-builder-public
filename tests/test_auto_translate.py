import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.auto_translate import (
    DEFAULT_RETRY_ATTEMPTS,
    TranslationCheckpointError,
    checkpoint_path_for,
    failure_path_for,
    source_language_code,
    translate_catalog,
    translate_delta,
    translate_delta_with_provider,
)
from crimson0020.translation_providers import (
    RetryableTranslationProviderError,
    TranslationProviderError,
    TranslationResponseFormatError,
)
from crimson0020.translation_work import prepare_translation_work
from crimson0020.translation_runtime.progress import PauseRequested, TranslationControl


def _catalog(entries: list[dict[str, str]]) -> dict[str, object]:
    return {
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


def _entry(index: int, original: str, translation: str | None = None) -> dict[str, str]:
    return {
        "marker": f"{index:016x}",
        "key": f"KEY_{index}",
        "original": original,
        "translation": original if translation is None else translation,
    }


class _Provider:
    key = "test-provider"

    def __init__(self, transform=None) -> None:
        self.calls: list[tuple[tuple[str, ...], str, str]] = []
        self.transform = transform or (lambda text: f"VI:{text}")

    def translate(self, texts, source, target):
        texts = tuple(texts)
        self.calls.append((texts, source, target))
        return tuple(self.transform(text) for text in texts)


class AutoTranslateTests(unittest.TestCase):
    def _write_catalog(self, root: Path, entries: list[dict[str, str]]) -> Path:
        path = root / "source.json"
        path.write_text(
            json.dumps(_catalog(entries), ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        return path

    def test_translates_only_untranslated_rows_to_separate_atomic_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [
                    _entry(1, "Hello"),
                    _entry(2, "Already translated", "Đã dịch"),
                ],
            )
            source_before = source.read_bytes()
            output = root / "translated" / "vi.json"
            provider = _Provider()

            result = translate_catalog(source, output, provider, "en", "vi")
            translated = json.loads(output.read_text(encoding="utf-8"))
            source_after = source.read_bytes()

        self.assertEqual(source_before, source_after)
        self.assertEqual("VI:Hello", translated["entries"][0]["translation"])
        self.assertEqual("Đã dịch", translated["entries"][1]["translation"])
        self.assertEqual(
            _catalog([_entry(1, "Hello"), _entry(2, "Already translated", "Đã dịch")])[
                "metadata"
            ],
            translated["metadata"],
        )
        self.assertEqual(1, result.translated_entries)
        self.assertEqual(1, result.skipped_existing_entries)
        self.assertEqual(1, result.request_count)

    def test_empty_sparse_delta_completes_without_provider_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_catalog(root, [_entry(1, "Same", "Đã dịch")])
            new = root / "new.json"
            new.write_text(
                json.dumps(_catalog([_entry(1, "Same")])),
                encoding="utf-8",
            )
            delta = root / "delta.json"
            output = root / "translated.json"
            prepare_translation_work(old, new, delta)
            provider = _Provider()

            result = translate_catalog(delta, output, provider, "en", "vi")
            translated = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual([], provider.calls)
        self.assertEqual([], translated["entries"])
        self.assertEqual(0, result.pending_entries)

    def test_step_two_rejects_a_full_catalog_instead_of_translating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "Do not translate")])
            output = root / "translated.json"
            provider = _Provider()

            with self.assertRaisesRegex(ValueError, "sparse translation delta"):
                translate_delta(source, output, provider, "en", "vi")

        self.assertEqual([], provider.calls)
        self.assertFalse(output.exists())

    def test_token_only_sentinels_are_preserved_without_provider_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(1, "[EMPTY]"), _entry(2, "{Key:Key_Guard}")],
            )
            output = root / "vi.json"
            provider = _Provider()

            result = translate_catalog(source, output, provider, "en", "vi")
            translated = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual([], provider.calls)
        self.assertEqual("[EMPTY]", translated["entries"][0]["translation"])
        self.assertEqual("{Key:Key_Guard}", translated["entries"][1]["translation"])
        self.assertEqual(2, result.skipped_technical_entries)

    def test_exact_duplicate_sources_share_one_request_but_keep_each_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(1, "Same text"), _entry(2, "Same text")],
            )
            output = root / "vi.json"
            provider = _Provider()

            result = translate_catalog(source, output, provider, "en", "vi")
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]
            checkpoint_records = [
                json.loads(line)
                for line in checkpoint_path_for(output)
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(("Same text",), provider.calls[0][0])
        self.assertEqual(
            ["VI:Same text", "VI:Same text"], [row["translation"] for row in entries]
        )
        self.assertEqual(2, result.translated_entries)
        self.assertEqual(1, result.reused_duplicate_entries)
        self.assertEqual(2, len(checkpoint_records[1]["translations"]))

    def test_placeholders_markup_and_newlines_are_shielded_and_restored_exactly(
        self,
    ) -> None:
        original = "Hello {player}\n<color=#fff>World</color> [confirm] %s"

        def translate(text: str) -> str:
            self.assertNotIn("{player}", text)
            self.assertNotIn("<color=#fff>", text)
            self.assertNotIn("\n", text)
            return text.replace("Hello", "Xin chào").replace("World", "Thế giới")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, original)])
            output = root / "vi.json"

            translate_catalog(source, output, _Provider(translate), "en", "vi")
            value = json.loads(output.read_text(encoding="utf-8"))["entries"][0][
                "translation"
            ]

        self.assertEqual(
            "Xin chào {player}\n<color=#fff>Thế giới</color> [confirm] %s",
            value,
        )

    def test_placeholder_damage_is_rescued_without_changing_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "Hello {player}")])
            output = root / "vi.json"
            provider = _Provider(lambda text: text.replace("CRIMSON_KEEP", "BROKEN"))

            result = translate_catalog(source, output, provider, "en", "vi")

            self.assertTrue(output.exists())
            self.assertEqual(0, result.pending_entries)
            self.assertEqual(0, result.failure_count)
            translated = json.loads(output.read_text(encoding="utf-8"))["entries"][0][
                "translation"
            ]
            self.assertIn("{player}", translated)

    def test_response_count_damage_is_split_to_isolate_only_the_bad_row(self) -> None:
        class BadCountProvider(_Provider):
            def translate(self, texts, source, target):
                batch = tuple(texts)
                self.calls.append((batch, source, target))
                if len(batch) > 1 or batch == ("Bad",):
                    raise TranslationResponseFormatError("wrong result count")
                return (f"VI:{batch[0]}",)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(1, "Good one"), _entry(2, "Bad"), _entry(3, "Good two")],
            )
            output = root / "vi.json"

            result = translate_catalog(
                source,
                output,
                BadCountProvider(),
                "en",
                "vi",
                max_batch_items=3,
                retry_attempts=1,
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(1, result.pending_entries)
        self.assertEqual(
            ["VI:Good one", "Bad", "VI:Good two"],
            [entry["translation"] for entry in entries],
        )

    def test_provider_wide_nonretryable_error_does_not_split_every_row(self) -> None:
        class OfflineProvider(_Provider):
            preferred_batch_items = 2

            def translate(self, texts, source, target):
                batch = tuple(texts)
                self.calls.append((batch, source, target))
                raise TranslationProviderError("HTTP 401")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(index, f"Text {index}") for index in range(1, 5)],
            )
            output = root / "vi.json"
            provider = OfflineProvider()

            result = translate_catalog(
                source,
                output,
                provider,
                "en",
                "vi",
                retry_attempts=1,
            )

        self.assertEqual(4, result.pending_entries)
        self.assertEqual(2, result.request_count)
        self.assertEqual(2, len(provider.calls))

    def test_repeated_provider_outage_opens_bounded_circuit_and_returns_partial(self):
        class OfflineProvider(_Provider):
            preferred_batch_items = 2

            def translate(self, texts, source, target):
                batch = tuple(texts)
                self.calls.append((batch, source, target))
                raise RetryableTranslationProviderError("offline")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(index, f"Text {index}") for index in range(1, 11)],
            )
            output = root / "vi.json"
            provider = OfflineProvider()

            result = translate_catalog(
                source,
                output,
                provider,
                "en",
                "vi",
                retry_attempts=1,
            )

            self.assertTrue(output.exists())

        self.assertEqual(10, result.pending_entries)
        self.assertEqual(3, result.request_count)
        self.assertEqual(3, len(provider.calls))

    def test_oversized_row_is_reported_while_later_rows_still_translate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(1, "X" * 11), _entry(2, "Short")],
            )
            output = root / "vi.json"
            provider = _Provider()

            result = translate_catalog(
                source,
                output,
                provider,
                "en",
                "vi",
                max_batch_characters=10,
                retry_attempts=1,
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(1, result.pending_entries)
        self.assertEqual("X" * 11, entries[0]["translation"])
        self.assertEqual("VI:Short", entries[1]["translation"])

    def test_failure_journal_includes_every_identity_sharing_failed_text(self) -> None:
        class OfflineProvider(_Provider):
            def translate(self, texts, source, target):
                raise RetryableTranslationProviderError("offline")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(1, "Same"), _entry(2, "Same")],
            )
            output = root / "vi.json"

            result = translate_catalog(
                source,
                output,
                OfflineProvider(),
                "en",
                "vi",
                retry_attempts=1,
            )
            failures = [
                json.loads(line)
                for line in failure_path_for(output)
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(2, result.pending_entries)
        self.assertEqual(2, result.failure_count)
        self.assertEqual({"KEY_1", "KEY_2"}, {row["key"] for row in failures})

    def test_placeholder_only_provider_result_is_not_checkpointed(self) -> None:
        def discard_words(text: str) -> str:
            marker_start = text.index("__CRIMSON_KEEP_")
            return text[marker_start:]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "Hello {player}")])
            output = root / "vi.json"

            result = translate_catalog(
                source, output, _Provider(discard_words), "en", "vi"
            )

            checkpoint = checkpoint_path_for(output)
            self.assertEqual(
                1, len(checkpoint.read_text(encoding="utf-8").splitlines())
            )
            self.assertTrue(output.exists())
            self.assertEqual(1, result.pending_entries)

    def test_checkpoint_resumes_completed_rows_after_bounded_failure(self) -> None:
        class FailOnSecondBatch(_Provider):
            def translate(self, texts, source, target):
                if self.calls:
                    raise RetryableTranslationProviderError("temporary")
                return super().translate(texts, source, target)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "One"), _entry(2, "Two")])
            output = root / "vi.json"

            first = translate_catalog(
                source,
                output,
                FailOnSecondBatch(),
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )
            self.assertEqual(1, first.pending_entries)

            self.assertTrue(checkpoint_path_for(output).is_file())
            provider = _Provider()
            result = translate_catalog(
                source,
                output,
                provider,
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )

        self.assertEqual(1, result.resumed_entries)
        self.assertEqual(1, result.translated_entries)
        self.assertEqual(1, len(provider.calls))

    def test_pause_interrupts_retry_backoff_before_next_provider_attempt(self) -> None:
        from crimson0020.auto_translate import _translate_with_retry

        class TemporaryFailureProvider:
            key = "temporary-failure"

            def __init__(self) -> None:
                self.calls = 0

            def translate(self, texts, source, target):
                self.calls += 1
                raise RetryableTranslationProviderError("temporary")

        provider = TemporaryFailureProvider()
        control = TranslationControl()
        sleeps: list[float] = []

        def sleeper(delay: float) -> None:
            sleeps.append(delay)
            control.request_pause("test pause")

        with self.assertRaises(PauseRequested):
            _translate_with_retry(
                provider,
                ("Hello",),
                "en",
                "vi",
                attempts=3,
                base_delay=5.0,
                sleeper=sleeper,
                translation_control=control,
            )

        self.assertEqual(1, provider.calls)
        self.assertEqual([0.1], sleeps)

    def test_retry_honors_provider_retry_after_before_next_attempt(self) -> None:
        from crimson0020.auto_translate import _translate_with_retry

        class CooldownProvider:
            key = "cooldown-provider"

            def __init__(self) -> None:
                self.calls = 0

            def translate(self, texts, source, target):
                del source, target
                self.calls += 1
                if self.calls == 1:
                    raise RetryableTranslationProviderError(
                        "rate limited",
                        retry_after_seconds=0.35,
                    )
                return tuple(f"VI:{text}" for text in texts)

        provider = CooldownProvider()
        sleeps: list[float] = []

        result = _translate_with_retry(
            provider,
            ("Hello",),
            "en",
            "vi",
            attempts=2,
            base_delay=0.1,
            sleeper=sleeps.append,
        )

        self.assertEqual(("VI:Hello",), result)
        self.assertEqual(2, provider.calls)
        self.assertAlmostEqual(0.35, sum(sleeps), places=6)

    def test_checkpoint_resumes_with_a_different_provider(self) -> None:
        class FirstProvider(_Provider):
            key = "first-provider"

            def translate(self, texts, source, target):
                if self.calls:
                    raise RetryableTranslationProviderError("temporary")
                return super().translate(texts, source, target)

        class FallbackProvider(_Provider):
            key = "fallback-provider"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "One"), _entry(2, "Two")])
            output = root / "vi.json"

            first = translate_catalog(
                source,
                output,
                FirstProvider(),
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )
            self.assertEqual(1, first.pending_entries)

            fallback = FallbackProvider()
            result = translate_catalog(
                source,
                output,
                fallback,
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(1, result.resumed_entries)
        self.assertEqual(1, result.translated_entries)
        self.assertEqual(("Two",), fallback.calls[0][0])
        self.assertEqual(["VI:One", "VI:Two"], [row["translation"] for row in entries])

    def test_contract_failure_automatically_falls_back_for_failed_row(self) -> None:
        class MutatingProvider(_Provider):
            key = "mutating-provider"

            def translate(self, texts, source, target):
                texts = tuple(texts)
                self.calls.append((texts, source, target))
                return tuple(
                    text.replace("__CRIMSON_KEEP_", "__BROKEN_KEEP_")
                    for text in texts
                )

        class SafeProvider(_Provider):
            key = "safe-provider"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "Use <br/> now")])
            output = root / "vi.json"
            primary = MutatingProvider()
            fallback = SafeProvider()

            result = translate_catalog(
                source,
                output,
                primary,
                "en",
                "vi",
                retry_attempts=1,
                fallback_providers=(fallback,),
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(0, result.failure_count)
        self.assertEqual(1, len(primary.calls))
        self.assertEqual(1, len(fallback.calls))
        self.assertEqual("VI:Use <br/> now", entries[0]["translation"])

    def test_contract_failure_keeps_safe_siblings_and_falls_back_only_failed_subset(self) -> None:
        class SelectivelyMutatingProvider(_Provider):
            key = "mutating-provider"

            def translate(self, texts, source, target):
                texts = tuple(texts)
                self.calls.append((texts, source, target))
                values = []
                for text in texts:
                    translated = f"VI:{text}"
                    if "__CRIMSON_KEEP_" in text:
                        translated = translated.replace("__CRIMSON_KEEP_", "__BROKEN_KEEP_")
                    values.append(translated)
                return tuple(values)

        class SafeProvider(_Provider):
            key = "safe-provider"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [
                    _entry(1, "Plain one"),
                    _entry(2, "Use <br/> now"),
                    _entry(3, "Plain two"),
                ],
            )
            output = root / "vi.json"
            primary = SelectivelyMutatingProvider()
            fallback = SafeProvider()

            result = translate_catalog(
                source,
                output,
                primary,
                "en",
                "vi",
                retry_attempts=1,
                fallback_providers=(fallback,),
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(1, len(primary.calls))
        self.assertEqual(1, len(fallback.calls))
        self.assertEqual(1, len(fallback.calls[0][0]))
        self.assertEqual(
            ["VI:Plain one", "VI:Use <br/> now", "VI:Plain two"],
            [row["translation"] for row in entries],
        )

    def test_final_provider_uses_fragment_rescue_when_it_mutates_protected_tokens(self) -> None:
        class MutatesMarkersOnly(_Provider):
            key = "marker-mutator"

            def translate(self, texts, source, target):
                texts = tuple(texts)
                self.calls.append((texts, source, target))
                values = []
                for text in texts:
                    translated = f"VI:{text}"
                    if "__CRIMSON_KEEP_" in text:
                        translated = translated.replace("__CRIMSON_KEEP_", "__BROKEN_KEEP_")
                    values.append(translated)
                return tuple(values)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "Use <br/> now")])
            output = root / "vi.json"
            provider = MutatesMarkersOnly()

            result = translate_catalog(
                source,
                output,
                provider,
                "en",
                "vi",
                retry_attempts=1,
            )
            translated = json.loads(output.read_text(encoding="utf-8"))["entries"][0][
                "translation"
            ]

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(2, len(provider.calls))
        self.assertEqual(("Use", "now"), provider.calls[1][0])
        self.assertEqual("VI:Use <br/> VI:now", translated)

    def test_provider_error_automatically_falls_back_without_manual_resume(self) -> None:
        class OfflineProvider(_Provider):
            key = "offline-provider"

            def translate(self, texts, source, target):
                self.calls.append((tuple(texts), source, target))
                raise RetryableTranslationProviderError("temporary outage")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "One"), _entry(2, "Two")])
            output = root / "vi.json"
            primary = OfflineProvider()
            fallback = _Provider()

            result = translate_catalog(
                source,
                output,
                primary,
                "en",
                "vi",
                max_batch_items=2,
                retry_attempts=1,
                fallback_providers=(fallback,),
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(0, result.failure_count)
        self.assertEqual(1, len(primary.calls))
        self.assertEqual(1, len(fallback.calls))
        self.assertEqual(["VI:One", "VI:Two"], [row["translation"] for row in entries])

    def test_automatic_fallbacks_do_not_repeat_retryable_provider_failures(self) -> None:
        class OfflineProvider(_Provider):
            def translate(self, texts, source, target):
                self.calls.append((tuple(texts), source, target))
                raise RetryableTranslationProviderError(f"{self.key} unavailable")

        primary = OfflineProvider()
        primary.key = "primary"
        first_fallback = OfflineProvider()
        first_fallback.key = "fallback-offline"
        safe_fallback = _Provider()
        safe_fallback.key = "fallback-safe"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "One")])
            output = root / "vi.json"
            result = translate_catalog(
                source,
                output,
                primary,
                "en",
                "vi",
                retry_attempts=3,
                sleeper=lambda _seconds: None,
                fallback_providers=(first_fallback, safe_fallback),
            )

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(3, len(primary.calls))
        self.assertEqual(1, len(first_fallback.calls))
        self.assertEqual(1, len(safe_fallback.calls))

    def test_provider_facade_does_not_auto_fallback_between_free_providers(self) -> None:
        class OfflineGoogle(_Provider):
            key = "google-web-free"

            def translate(self, texts, source, target):
                self.calls.append((tuple(texts), source, target))
                raise RetryableTranslationProviderError("google unavailable")

        google = OfflineGoogle()
        bing = _Provider()
        bing.key = "bing-web-free"
        providers = {"google-web-free": google, "bing-web-free": bing}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_catalog(root, [])
            new = root / "new.json"
            new.write_text(json.dumps(_catalog([_entry(1, "One")])), encoding="utf-8")
            delta = root / "delta.json"
            output = root / "translated.json"
            prepare_translation_work(old, new, delta)

            with patch(
                "crimson0020.auto_translate.create_translation_provider",
                side_effect=lambda key, config=None: providers[key],
            ) as create_provider:
                result = translate_delta_with_provider(
                    delta,
                    output,
                    "google-web-free",
                    "en",
                    "vi",
                )

        # A standalone free endpoint must never silently switch to another one.
        self.assertEqual(DEFAULT_RETRY_ATTEMPTS, len(google.calls))
        self.assertEqual(0, len(bing.calls))
        self.assertEqual(1, result.pending_entries)
        self.assertEqual(
            ["google-web-free"],
            [call.args[0] for call in create_provider.call_args_list],
        )

    def test_provider_facade_does_not_initialize_unused_fallbacks(self) -> None:
        google = _Provider()
        google.key = "google-web-free"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_catalog(root, [])
            new = root / "new.json"
            new.write_text(json.dumps(_catalog([_entry(1, "One")])), encoding="utf-8")
            delta = root / "delta.json"
            output = root / "translated.json"
            prepare_translation_work(old, new, delta)

            with patch(
                "crimson0020.auto_translate.create_translation_provider",
                side_effect=lambda key, config=None: google,
            ) as create_provider:
                result = translate_delta_with_provider(
                    delta,
                    output,
                    "google-web-free",
                    "en",
                    "vi",
                )

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(["google-web-free"], [call.args[0] for call in create_provider.call_args_list])

    def test_provider_facade_constructs_only_the_selected_provider(self) -> None:
        class OfflineGoogle(_Provider):
            key = "google-web-free"

            def translate(self, texts, source, target):
                self.calls.append((tuple(texts), source, target))
                raise RetryableTranslationProviderError("google unavailable")

        google = OfflineGoogle()

        def factory(key, config=None):
            if key != "google-web-free":
                raise AssertionError(f"unexpected provider construction: {key}")
            return google

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_catalog(root, [])
            new = root / "new.json"
            new.write_text(
                json.dumps(_catalog([_entry(1, "One"), _entry(2, "Two")])),
                encoding="utf-8",
            )
            delta = root / "delta.json"
            output = root / "translated.json"
            prepare_translation_work(old, new, delta)

            with patch(
                "crimson0020.auto_translate.create_translation_provider",
                side_effect=factory,
            ) as create_provider:
                result = translate_delta_with_provider(
                    delta,
                    output,
                    "google-web-free",
                    "en",
                    "vi",
                )

        self.assertEqual(2, result.pending_entries)
        self.assertEqual(
            ["google-web-free"],
            [call.args[0] for call in create_provider.call_args_list],
        )

    def test_repaired_partial_delta_can_resume_against_existing_output_checkpoint(
        self,
    ) -> None:
        class FailOnSecondBatch(_Provider):
            def translate(self, texts, source, target):
                if self.calls:
                    raise RetryableTranslationProviderError("temporary")
                return super().translate(texts, source, target)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_catalog(root, [])
            new = root / "new.json"
            new.write_text(
                json.dumps(_catalog([_entry(1, "One"), _entry(2, "Two")])),
                encoding="utf-8",
            )
            template = root / "delta.json"
            output = root / "translated.json"
            repaired = root / "translated.fixed.json"
            prepare_translation_work(old, new, template)
            first = translate_catalog(
                template,
                output,
                FailOnSecondBatch(),
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )
            self.assertEqual(1, first.pending_entries)
            repaired.write_bytes(output.read_bytes())

            fallback = _Provider()
            resumed = translate_catalog(
                repaired,
                output,
                fallback,
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, resumed.pending_entries)
        self.assertEqual(1, len(fallback.calls))
        self.assertEqual(["VI:One", "VI:Two"], [row["translation"] for row in entries])

    def test_contextful_delta_does_not_migrate_contextless_legacy_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old = self._write_catalog(root, [])
            new = root / "new.json"
            new.write_text(
                json.dumps(_catalog([_entry(1, "One"), _entry(2, "Two")])),
                encoding="utf-8",
            )
            delta = root / "delta.json"
            output = root / "translated.json"
            legacy = root / "legacy-checkpoint.jsonl"
            prepare_translation_work(old, new, delta)
            delta_data = json.loads(delta.read_text(encoding="utf-8"))
            legacy_header = {
                "record": "header",
                "format_version": 1,
                "source_fingerprint": delta_data["source"]["catalog_fingerprint"],
                "provider": "google-web-free",
                "source_language": "en",
                "target_language": "vi",
            }
            legacy_batch = {
                "record": "batch",
                "translations": [
                    {
                        "marker": f"{1:016x}",
                        "key": "KEY_1",
                        "original_sha256": hashlib.sha256(b"One").hexdigest().upper(),
                        "translation": "VI cũ:One",
                    },
                    {
                        "marker": "ff" * 8,
                        "key": "UNKNOWN",
                        "original_sha256": hashlib.sha256(b"Unknown")
                        .hexdigest()
                        .upper(),
                        "translation": "Bỏ qua an toàn",
                    },
                ],
            }
            legacy.write_text(
                "\n".join(
                    json.dumps(value, ensure_ascii=False)
                    for value in (legacy_header, legacy_batch)
                )
                + "\n",
                encoding="utf-8",
            )
            provider = _Provider()

            result = translate_delta(
                delta,
                output,
                provider,
                "en",
                "vi",
                legacy_checkpoint=legacy,
            )
            entries = json.loads(output.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, result.migrated_entries)
        self.assertEqual(1, len(provider.calls))
        self.assertEqual(
            ["VI:One", "VI:Two"],
            [entry["translation"] for entry in entries],
        )

    def test_provider_batch_preferences_reduce_scheduler_requests(self) -> None:
        class BatchProvider(_Provider):
            preferred_batch_items = 3
            preferred_batch_characters = 100

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(index, f"Text {index}") for index in range(1, 8)],
            )
            output = root / "vi.json"
            provider = BatchProvider()

            result = translate_catalog(source, output, provider, "en", "vi")

        self.assertEqual([3, 3, 1], [len(call[0]) for call in provider.calls])
        self.assertEqual(3, result.request_count)

    def test_custom_openai_profile_can_run_independent_batches_concurrently(self) -> None:
        class ParallelCustomProvider:
            key = "openai-compatible-synthetic"
            preferred_batch_items = 1
            preferred_batch_characters = 100
            preferred_outer_concurrency = 3

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.active = 0
                self.peak = 0

            def translate(self, texts, source, target):
                with self._lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                try:
                    time.sleep(0.03)
                    return tuple(f"VI:{text}" for text in texts)
                finally:
                    with self._lock:
                        self.active -= 1

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(index, f"Text {index}") for index in range(1, 7)],
            )
            output = root / "vi.json"
            provider = ParallelCustomProvider()

            result = translate_catalog(source, output, provider, "en", "vi")

        self.assertEqual(6, result.request_count)
        self.assertGreaterEqual(provider.peak, 2)
        self.assertLessEqual(provider.peak, 3)

    def test_checkpoint_is_rejected_when_immutable_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "One")])
            output = root / "vi.json"
            translate_catalog(source, output, _Provider(), "en", "vi")
            source.write_text(
                json.dumps(_catalog([_entry(1, "Changed")]), indent=4),
                encoding="utf-8",
            )

            with self.assertRaises(TranslationCheckpointError):
                translate_catalog(source, output, _Provider(), "en", "vi")

    def test_checkpoint_is_rejected_when_saved_translation_breaks_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "Hello {player}")])
            output = root / "vi.json"
            translate_catalog(source, output, _Provider(), "en", "vi")
            checkpoint = checkpoint_path_for(output)
            lines = checkpoint.read_text(encoding="utf-8").splitlines()
            batch = json.loads(lines[1])
            batch["translations"][0]["translation"] = "Xin chào"
            checkpoint.write_text(
                "\n".join((lines[0], json.dumps(batch))) + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(TranslationCheckpointError):
                translate_catalog(source, output, _Provider(), "en", "vi")

    def test_checkpoint_journal_appends_only_each_new_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(
                root,
                [_entry(1, "One"), _entry(2, "Two"), _entry(3, "Three")],
            )
            output = root / "vi.json"

            translate_catalog(
                source,
                output,
                _Provider(),
                "en",
                "vi",
                max_batch_items=1,
            )
            records = [
                json.loads(line)
                for line in checkpoint_path_for(output)
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual("header", records[0]["record"])
        self.assertEqual([1, 1, 1], [len(row["translations"]) for row in records[1:]])

    def test_incomplete_final_checkpoint_record_is_ignored_on_resume(self) -> None:
        class FailOnSecondBatch(_Provider):
            def translate(self, texts, source, target):
                if self.calls:
                    raise RetryableTranslationProviderError("temporary")
                return super().translate(texts, source, target)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [_entry(1, "One"), _entry(2, "Two")])
            output = root / "vi.json"
            first = translate_catalog(
                source,
                output,
                FailOnSecondBatch(),
                "en",
                "vi",
                max_batch_items=1,
                retry_attempts=1,
            )
            self.assertEqual(1, first.pending_entries)
            checkpoint = checkpoint_path_for(output)
            with checkpoint.open("ab") as stream:
                stream.write(b'{"record":"batch"')

            provider = _Provider()
            result = translate_catalog(
                source,
                output,
                provider,
                "en",
                "vi",
                max_batch_items=1,
            )
            records = [
                json.loads(line)
                for line in checkpoint.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(1, result.resumed_entries)
        self.assertEqual(1, len(provider.calls))
        self.assertEqual(3, len(records))

    def test_duplicate_marker_key_is_rejected_before_provider_call(self) -> None:
        duplicate = _entry(1, "One")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_catalog(root, [duplicate, dict(duplicate)])
            output = root / "vi.json"
            provider = _Provider()

            with self.assertRaisesRegex(ValueError, r"Duplicate marker \+ key"):
                translate_catalog(source, output, provider, "en", "vi")

        self.assertEqual([], provider.calls)

    def test_source_language_mapping_covers_current_ui_languages(self) -> None:
        self.assertEqual("en", source_language_code("English"))
        self.assertEqual("pt", source_language_code("Portuguese - Brazil"))
        self.assertEqual("zh", source_language_code("Traditional Chinese"))


if __name__ == "__main__":
    unittest.main()
