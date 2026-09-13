from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.auto_translate import (
    AutoTranslateProgress,
    checkpoint_path_for,
    translate_delta,
)
from crimson0020.translation_merge import validate_translation_delta
from crimson0020.translation_providers import (
    PartialTranslationProviderError,
    RetryableTranslationProviderError,
    create_translation_provider,
)
from crimson0020.translation_runtime.contracts import (
    FailureKind,
    RouteAttempt,
    TranslationBatchOutcome,
    TranslationFailure,
    TranslationItemResult,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.terminology import TerminologyEngine
from crimson0020.translation_runtime.progress import ProgressReporter, TranslationControl
from crimson0020.translation_ui_service import TranslationUIService
from crimson0020.translation_work import prepare_translation_work
from crimson0020.webview_bridge import WebviewBridge


def _entry(index: int, text: str, *, context: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "marker": f"{index:016x}",
        "key": f"key_{index}",
        "original": text,
        "translation": "",
        "change": "new",
        "translation_state": "pending",
    }
    if context is not None:
        row["context"] = [context]
        row["context_signature"] = f"ctx-{context}"
    return row


def _delta(path: Path, entries: list[dict[str, object]], package: str = "0020") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "crimson.translation-delta",
                "schema_version": 1,
                "kind": "sparse-delta",
                "source": {
                    "catalog_fingerprint": "A" * 64,
                    "base_catalog_fingerprint": "B" * 64,
                    "entry_count": len(entries),
                    "package_folder": package,
                },
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _catalog(path: Path, text: str, package: str, context: str | None = None) -> Path:
    entry: dict[str, object] = {
        "marker": "0100000000000000",
        "key": "line",
        "original": text,
        "translation": text,
    }
    if context is not None:
        entry["context"] = [context]
        entry["context_signature"] = f"ctx-{context}"
    path.write_text(
        json.dumps(
            {"metadata": {"package_folder": package}, "entries": [entry]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


class EchoProvider:
    key = "echo"
    preferred_batch_items = 24

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, texts, source_language, target_language):
        self.calls += 1
        return tuple(f"VI:{text}" for text in texts)


class TestPostFinalHardening(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_hard_glossary_allows_reordering_but_rejects_loss_or_duplication(self) -> None:
        db = KnowledgeDatabase(self.root / "terms.sqlite3")
        engine = TerminologyEngine(db)
        from crimson0020.translation_runtime.knowledge.terminology import HardTermMaskMetadata

        meta = HardTermMaskMetadata(
            original_text="A B",
            token_to_target={"<_GT_0_>": "A_vi", "<_GT_1_>": "B_vi"},
        )
        self.assertEqual(
            "B_vi rồi A_vi",
            engine.restore_hard_terms("<_GT_1_> rồi <_GT_0_>", meta),
        )
        with self.assertRaises(ValueError):
            engine.restore_hard_terms("<_GT_0_>", meta)
        with self.assertRaises(ValueError):
            engine.restore_hard_terms("<_GT_0_> <_GT_0_> <_GT_1_>", meta)

    def test_step2_uses_delta_package_family_registered_by_step1(self) -> None:
        db_path = self.root / "knowledge.sqlite3"
        db = KnowledgeDatabase(db_path)
        old = _catalog(self.root / "old.json", "Old", "0021")
        new = _catalog(self.root / "new.json", "New", "0021")
        work = self.root / "work.json"
        prepare_translation_work(old, new, work, knowledge_db=db, source_language="ja")
        translate_delta(work, self.root / "out.json", EchoProvider(), "ja", "vi", db_path=db_path)
        with db.connect() as con:
            scopes = con.execute(
                "SELECT package_family, source_language, target_language FROM scopes ORDER BY package_family"
            ).fetchall()
        self.assertEqual([("0021", "ja", "vi")], scopes)

    def test_circuit_breaker_leaves_unsent_rows_pending(self) -> None:
        class Offline(EchoProvider):
            preferred_batch_items = 2

            def translate(self, texts, source_language, target_language):
                self.calls += 1
                raise RetryableTranslationProviderError("offline")

        source = _delta(self.root / "delta.json", [_entry(i, f"Text {i}") for i in range(1, 11)])
        output = self.root / "out.json"
        provider = Offline()
        result = translate_delta(source, output, provider, "en", "vi", retry_attempts=1)
        states = [row["translation_state"] for row in json.loads(output.read_text(encoding="utf-8"))["entries"]]
        self.assertEqual(3, provider.calls)
        self.assertEqual(6, result.failure_count)
        self.assertEqual(["failed"] * 6 + ["pending"] * 4, states)

    def test_progress_callback_reports_remaining_rows_not_failure_count(self) -> None:
        class OneAtATime(EchoProvider):
            preferred_batch_items = 1

        source = _delta(self.root / "delta.json", [_entry(i, f"Text {i}") for i in range(1, 4)])
        seen: list[AutoTranslateProgress] = []
        translate_delta(source, self.root / "out.json", OneAtATime(), "en", "vi", progress=seen.append)
        self.assertEqual([2, 1, 0], [snapshot.pending_entries for snapshot in seen])
        self.assertEqual([1, 2, 3], [snapshot.translated_entries for snapshot in seen])

    def test_progress_reporter_separates_resumed_from_new_translation(self) -> None:
        source = _delta(self.root / "delta.json", [_entry(1, "One"), _entry(2, "Two")])
        output = self.root / "out.json"
        control = TranslationControl()

        class Pausing(EchoProvider):
            preferred_batch_items = 1

            def translate(self, texts, source_language, target_language):
                value = super().translate(texts, source_language, target_language)
                control.request_pause("pause")
                return value

        translate_delta(source, output, Pausing(), "en", "vi", translation_control=control)
        control.clear_pause()
        reporter = ProgressReporter("resume", 2)
        result = translate_delta(
            source,
            output,
            EchoProvider(),
            "en",
            "vi",
            progress_reporter=reporter,
        )
        self.assertEqual(1, result.resumed_entries)
        self.assertEqual(1, result.translated_entries)
        self.assertEqual(1, reporter.snapshot.resumed_entries)
        self.assertEqual(1, reporter.snapshot.translated_entries)
        self.assertEqual(2, reporter.snapshot.saved_entries)

    def test_checkpoint_context_change_forces_retranslation(self) -> None:
        old = _catalog(self.root / "old.json", "Old", "0020")
        new_a = _catalog(self.root / "new-a.json", "I am ready.", "0020", "Speaker A")
        new_b = _catalog(self.root / "new-b.json", "I am ready.", "0020", "Speaker B")
        work_a = self.root / "work-a.json"
        work_b = self.root / "work-b.json"
        prepare_translation_work(old, new_a, work_a)
        prepare_translation_work(old, new_b, work_b)

        class ContextProvider:
            key = "context"

            def __init__(self) -> None:
                self.calls = 0

            def translate_items(self, items, source_language, target_language):
                self.calls += 1
                return tuple(
                    "A:I am ready." if "Speaker A" in item.context else "B:I am ready."
                    for item in items
                )

            def translate(self, texts, source_language, target_language):
                raise AssertionError("rich path expected")

        output = self.root / "out.json"
        first = ContextProvider()
        translate_delta(work_a, output, first, "en", "vi")
        second = ContextProvider()
        translate_delta(work_b, output, second, "en", "vi")
        final = json.loads(output.read_text(encoding="utf-8"))["entries"][0]
        checkpoint_lines = [json.loads(line) for line in (Path(str(output) + ".autotranslate-checkpoint.jsonl")).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(1, second.calls)
        self.assertEqual("B:I am ready.", final["translation"])
        self.assertEqual("ctx-Speaker B", checkpoint_lines[-1]["translations"][0]["context_signature"])

    def test_contextful_row_rejects_contextless_v2_checkpoint(self) -> None:
        source = _delta(
            self.root / "delta.json",
            [_entry(1, "I am ready.", context="Speaker A")],
        )
        output = self.root / "out.json"
        checkpoint = Path(str(output) + ".autotranslate-checkpoint.jsonl")
        data = json.loads(source.read_text(encoding="utf-8"))
        entry = data["entries"][0]
        checkpoint.write_text(
            "\n".join(
                (
                    json.dumps(
                        {
                            "record": "header",
                            "format_version": 2,
                            "source_fingerprint": data["source"]["catalog_fingerprint"],
                            "source_language": "en",
                            "target_language": "vi",
                        }
                    ),
                    json.dumps(
                        {
                            "record": "batch",
                            "translations": [
                                {
                                    "marker": entry["marker"],
                                    "key": entry["key"],
                                    "original_sha256": hashlib.sha256(
                                        entry["original"].encode("utf-8")
                                    ).hexdigest().upper(),
                                    "translation": "OLD:I am ready.",
                                }
                            ],
                        }
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        provider = EchoProvider()

        result = translate_delta(source, output, provider, "en", "vi")
        translated = json.loads(output.read_text(encoding="utf-8"))["entries"][0]

        self.assertEqual(1, provider.calls)
        self.assertEqual(0, result.resumed_entries)
        self.assertEqual(1, result.translated_entries)
        self.assertEqual("VI:I am ready.", translated["translation"])

    def test_contextless_row_still_reuses_contextless_v2_checkpoint(self) -> None:
        source = _delta(self.root / "delta.json", [_entry(1, "Menu")])
        output = self.root / "out.json"
        checkpoint = Path(str(output) + ".autotranslate-checkpoint.jsonl")
        data = json.loads(source.read_text(encoding="utf-8"))
        entry = data["entries"][0]
        checkpoint.write_text(
            "\n".join(
                (
                    json.dumps(
                        {
                            "record": "header",
                            "format_version": 2,
                            "source_fingerprint": data["source"]["catalog_fingerprint"],
                            "source_language": "en",
                            "target_language": "vi",
                        }
                    ),
                    json.dumps(
                        {
                            "record": "batch",
                            "translations": [
                                {
                                    "marker": entry["marker"],
                                    "key": entry["key"],
                                    "original_sha256": hashlib.sha256(
                                        entry["original"].encode("utf-8")
                                    ).hexdigest().upper(),
                                    "translation": "Trình đơn",
                                }
                            ],
                        }
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        provider = EchoProvider()

        result = translate_delta(source, output, provider, "en", "vi")
        translated = json.loads(output.read_text(encoding="utf-8"))["entries"][0]

        self.assertEqual(0, provider.calls)
        self.assertEqual(1, result.resumed_entries)
        self.assertEqual(0, result.translated_entries)
        self.assertEqual("Trình đơn", translated["translation"])

    def test_ai_router_partial_success_is_committed_without_resending_good_item(self) -> None:
        with patch.dict(os.environ, {"LOCALAPPDATA": str(self.root / "localapp")}, clear=False):
            provider = create_translation_provider("ai_router_free")

        class Scheduler:
            def __init__(self) -> None:
                self.calls = 0

            def schedule_batch(self, request):
                self.calls += 1
                good, bad = request.items
                return TranslationBatchOutcome(
                    successes=(TranslationItemResult(good.work_id, "VI:Good", "r1", "m1"),),
                    failures=(TranslationFailure(bad.work_id, FailureKind.NETWORK, True, "offline"),),
                    attempts=(RouteAttempt("r1", "m1"), RouteAttempt("r2", "m2", error_kind=FailureKind.NETWORK)),
                )

        scheduler = Scheduler()
        provider.scheduler = scheduler
        source = _delta(self.root / "delta.json", [_entry(1, "Good"), _entry(2, "Bad")])
        seen: list[AutoTranslateProgress] = []
        result = translate_delta(source, self.root / "out.json", provider, "en", "vi", progress=seen.append)
        rows = json.loads(result.output_json.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(1, scheduler.calls)
        self.assertEqual("translated", rows[0]["translation_state"])
        self.assertEqual("failed", rows[1]["translation_state"])
        self.assertEqual(1, result.translated_entries)
        self.assertEqual(1, result.failure_count)
        self.assertEqual("r1", seen[-1].current_route)
        self.assertEqual("m1", seen[-1].current_model)
        self.assertEqual(1, seen[-1].fallback_count)

    def test_ai_router_has_router_specific_throughput_retry_and_health_profile(self) -> None:
        with patch.dict(os.environ, {"LOCALAPPDATA": str(self.root / "localapp")}, clear=False):
            provider = create_translation_provider("ai_router_free")

        self.assertEqual(48, provider.preferred_batch_items)
        self.assertEqual(12_000, provider.preferred_batch_characters)
        self.assertEqual(1, provider.preferred_retry_attempts)
        self.assertEqual(4, provider.preferred_outer_concurrency)
        self.assertEqual(4, provider.preferred_cooldown_deferred_retries)
        self.assertEqual(5.0, provider.preferred_retryable_deferred_seconds)
        self.assertEqual(2, provider.preferred_deferred_outer_concurrency)
        self.assertEqual(4, provider.preferred_output_snapshot_batches)
        self.assertEqual(4, provider.scheduler.concurrency_manager.global_limit)
        self.assertEqual(1, provider.scheduler.concurrency_manager.get_route_limit("llm7"))
        self.assertEqual(4, provider.scheduler.max_attempts_per_batch)
        self.assertEqual(2, provider.scheduler.max_attempts_per_route)
        self.assertEqual(45.0, provider.scheduler.max_batch_duration_seconds)
        self.assertEqual(2, provider.scheduler.max_bisection_attempts)
        self.assertEqual(0.0, provider.scheduler.primary_route_wait_seconds)
        for _ in range(25):
            provider.scheduler.concurrency_manager.record_success("llm7")
        self.assertEqual(1, provider.scheduler.concurrency_manager.get_route_limit("llm7"))
        for _ in range(20):
            provider.scheduler.concurrency_manager.record_success("opencode")
        self.assertEqual(2, provider.scheduler.concurrency_manager.get_route_limit("opencode"))
        self.assertIsNotNone(provider.scheduler.health_tracker)

    def test_ai_router_parallel_batches_overlap_and_commit_sqlite_serially(self) -> None:
        class ParallelRouterProvider:
            key = "ai_router_free"
            preferred_batch_items = 1
            preferred_batch_characters = 8_000
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 2

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._barrier = threading.Barrier(2)
                self.active = 0
                self.max_active = 0

            def translate(self, texts, source_language, target_language):
                del source_language, target_language
                text = tuple(texts)[0]
                with self._lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    self._barrier.wait(timeout=2.0)
                    # Force completion order to differ from submission order.
                    if text == "One":
                        time.sleep(0.05)
                    return (f"VI:{text}",)
                finally:
                    with self._lock:
                        self.active -= 1

        source = _delta(
            self.root / "parallel-delta.json",
            [_entry(1, "One"), _entry(2, "Two")],
        )
        output = self.root / "parallel-out.json"
        db_path = self.root / "parallel.sqlite3"
        commit_stages: list[str] = []
        provider = ParallelRouterProvider()

        result = translate_delta(
            source,
            output,
            provider,
            "en",
            "vi",
            db_path=db_path,
            on_commit_stage=commit_stages.append,
        )

        rows = json.loads(output.read_text(encoding="utf-8"))["entries"]
        checkpoint_records = [
            json.loads(line)
            for line in checkpoint_path_for(output).read_text(encoding="utf-8").splitlines()
        ]
        checkpoint_rows = [
            row
            for record in checkpoint_records
            if record.get("record") == "batch"
            for row in record["translations"]
        ]
        db = KnowledgeDatabase(db_path)
        with db.connect() as con:
            segment_count = con.execute("SELECT COUNT(*) FROM translation_segments").fetchone()[0]

        self.assertEqual(2, provider.max_active)
        self.assertEqual(0, result.pending_entries)
        self.assertEqual(2, result.translated_entries)
        self.assertEqual(["VI:One", "VI:Two"], [row["translation"] for row in rows])
        self.assertEqual({"key_1", "key_2"}, {row["key"] for row in checkpoint_rows})
        self.assertEqual(2, len(checkpoint_rows))
        self.assertEqual(2, segment_count)
        self.assertEqual(
            ["checkpoint", "database", "output", "checkpoint", "database", "output"],
            commit_stages[:6],
        )

    def test_parallel_router_requeues_cooldown_without_blocking_workers(self) -> None:
        class DeferredRouterProvider:
            key = "ai_router_free"
            preferred_batch_items = 1
            preferred_batch_characters = 8_000
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 2
            preferred_cooldown_deferred_retries = 1
            preferred_retryable_deferred_seconds = 0.001

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.calls: list[str] = []
                self.counts: dict[str, int] = {}

            def translate(self, texts, source_language, target_language):
                del source_language, target_language
                text = tuple(texts)[0]
                with self._lock:
                    self.calls.append(text)
                    count = self.counts.get(text, 0) + 1
                    self.counts[text] = count
                if text == "One" and count == 1:
                    raise RetryableTranslationProviderError("transient structural failure")
                if text == "Two" and count == 1:
                    raise RetryableTranslationProviderError(
                        "cooling",
                        retry_after_seconds=0.001,
                    )
                if text == "Two" and count == 2:
                    time.sleep(0.02)
                return (f"VI:{text}",)

        source = _delta(
            self.root / "deferred-router-delta.json",
            [_entry(1, "One"), _entry(2, "Two"), _entry(3, "Three")],
        )
        provider = DeferredRouterProvider()

        result = translate_delta(
            source,
            self.root / "deferred-router-out.json",
            provider,
            "en",
            "vi",
        )

        self.assertEqual(3, result.translated_entries)
        self.assertEqual(0, result.failure_count)
        self.assertIn("Three", provider.calls)
        retry_positions = [
            index
            for index, text in enumerate(provider.calls)
            if text in {"One", "Two"}
            and provider.calls[: index + 1].count(text) == 2
        ]
        self.assertTrue(retry_positions)
        self.assertLess(provider.calls.index("Three"), min(retry_positions))

    def test_parallel_router_retries_only_partial_failure_tail(self) -> None:
        class PartialRouterProvider:
            key = "ai_router_free"
            preferred_batch_items = 2
            preferred_batch_characters = 8_000
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 2
            preferred_cooldown_deferred_retries = 2
            preferred_retryable_deferred_seconds = 0.001
            preferred_deferred_outer_concurrency = 1

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.calls: list[tuple[str, ...]] = []
                self.partial_sent = False

            def translate_items(self, items, source_language, target_language):
                del source_language, target_language
                texts = tuple(item.masked_text for item in items)
                with self._lock:
                    self.calls.append(texts)
                    should_partial = texts == ("One", "Two") and not self.partial_sent
                    if should_partial:
                        self.partial_sent = True
                if should_partial:
                    raise PartialTranslationProviderError(
                        "route returned only a safe subset",
                        successes={items[0].work_id: "VI:One"},
                        failures={items[1].work_id: "missing"},
                    )
                return tuple(f"VI:{text}" for text in texts)

            def translate(self, texts, source_language, target_language):
                raise AssertionError("rich path expected")

        source = _delta(
            self.root / "partial-router-delta.json",
            [_entry(index, text) for index, text in enumerate(("One", "Two", "Three", "Four"), 1)],
        )
        output = self.root / "partial-router-out.json"
        provider = PartialRouterProvider()

        result = translate_delta(source, output, provider, "en", "vi")

        rows = json.loads(output.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(4, result.translated_entries)
        self.assertEqual(0, result.failure_count)
        self.assertIn(("Two",), provider.calls)
        self.assertEqual(
            ["VI:One", "VI:Two", "VI:Three", "VI:Four"],
            [row["translation"] for row in rows],
        )

    def test_parallel_router_retries_zero_success_partial_batch(self) -> None:
        class EmptyPartialRouterProvider:
            key = "ai_router_free"
            preferred_batch_items = 1
            preferred_batch_characters = 8_000
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 2
            preferred_cooldown_deferred_retries = 1
            preferred_retryable_deferred_seconds = 0.001
            preferred_deferred_outer_concurrency = 1

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.calls: list[str] = []
                self.failed_once = False

            def translate_items(self, items, source_language, target_language):
                del source_language, target_language
                item = items[0]
                text = item.masked_text
                with self._lock:
                    self.calls.append(text)
                    should_fail = text == "One" and not self.failed_once
                    if should_fail:
                        self.failed_once = True
                if should_fail:
                    raise PartialTranslationProviderError(
                        "route returned no usable item",
                        successes={},
                        failures={item.work_id: "missing"},
                    )
                return (f"VI:{text}",)

            def translate(self, texts, source_language, target_language):
                raise AssertionError("rich path expected")

        source = _delta(
            self.root / "empty-partial-router-delta.json",
            [_entry(1, "One"), _entry(2, "Two")],
        )
        provider = EmptyPartialRouterProvider()

        result = translate_delta(
            source,
            self.root / "empty-partial-router-out.json",
            provider,
            "en",
            "vi",
        )

        self.assertEqual(2, result.translated_entries)
        self.assertEqual(0, result.failure_count)
        self.assertEqual(2, provider.calls.count("One"))

    def test_ai_router_can_throttle_full_output_snapshots_without_losing_batch_durability(self) -> None:
        class SnapshotRouterProvider:
            key = "ai_router_free"
            preferred_batch_items = 1
            preferred_batch_characters = 8_000
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 1
            preferred_output_snapshot_batches = 4

            def translate(self, texts, source_language, target_language):
                del source_language, target_language
                return tuple(f"VI:{text}" for text in texts)

        source = _delta(
            self.root / "snapshot-delta.json",
            [_entry(index, f"Row {index}") for index in range(1, 6)],
        )
        output = self.root / "snapshot-out.json"
        stages: list[str] = []

        result = translate_delta(
            source,
            output,
            SnapshotRouterProvider(),
            "en",
            "vi",
            on_commit_stage=stages.append,
        )

        self.assertEqual(5, result.translated_entries)
        self.assertEqual(5, stages.count("checkpoint"))
        self.assertEqual(5, stages.count("database"))
        self.assertEqual(2, stages.count("output"))
        rows = json.loads(output.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(
            [f"VI:Row {index}" for index in range(1, 6)],
            [row["translation"] for row in rows],
        )

    def test_ai_router_call_metadata_is_thread_local(self) -> None:
        with patch.dict(os.environ, {"LOCALAPPDATA": str(self.root / "localapp")}, clear=False):
            provider = create_translation_provider("ai_router_free")

        class Scheduler:
            def __init__(self) -> None:
                self.barrier = threading.Barrier(2)

            def schedule_batch(self, request):
                self.barrier.wait(timeout=2.0)
                item = request.items[0]
                route = "route-one" if item.masked_text == "One" else "route-two"
                model = "model-one" if item.masked_text == "One" else "model-two"
                if item.masked_text == "One":
                    time.sleep(0.05)
                return TranslationBatchOutcome(
                    successes=(
                        TranslationItemResult(item.work_id, f"VI:{item.masked_text}", route, model),
                    ),
                    failures=(),
                    attempts=(RouteAttempt(route, model),),
                )

        provider.scheduler = Scheduler()
        seen: dict[str, tuple[str | None, str | None]] = {}
        seen_lock = threading.Lock()

        def run(text: str) -> None:
            result = provider.translate((text,), "en", "vi")
            self.assertEqual((f"VI:{text}",), result)
            with seen_lock:
                seen[text] = (provider.last_route_id, provider.last_model_id)

        first = threading.Thread(target=run, args=("One",))
        second = threading.Thread(target=run, args=("Two",))
        first.start()
        second.start()
        first.join(timeout=3.0)
        second.join(timeout=3.0)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(("route-one", "model-one"), seen["One"])
        self.assertEqual(("route-two", "model-two"), seen["Two"])

    def test_ai_router_full_failure_is_not_replayed_by_outer_auto_translate_retry(self) -> None:
        with patch.dict(os.environ, {"LOCALAPPDATA": str(self.root / "localapp")}, clear=False):
            provider = create_translation_provider("ai_router_free")

        class Scheduler:
            def __init__(self) -> None:
                self.calls = 0

            def schedule_batch(self, request):
                self.calls += 1
                item = request.items[0]
                return TranslationBatchOutcome(
                    successes=(),
                    failures=(
                        TranslationFailure(
                            item.work_id,
                            FailureKind.NETWORK,
                            True,
                            "offline",
                        ),
                    ),
                    attempts=(RouteAttempt("r1", "m1", error_kind=FailureKind.NETWORK),),
                )

        scheduler = Scheduler()
        provider.scheduler = scheduler
        source = _delta(self.root / "delta.json", [_entry(1, "One")])
        result = translate_delta(source, self.root / "out.json", provider, "en", "vi")

        self.assertEqual(1, scheduler.calls)
        self.assertEqual(1, result.failure_count)

    def test_ai_router_verified_cooldown_retries_same_batch_once(self) -> None:
        class CooldownRouterProvider:
            key = "ai_router_free"
            preferred_batch_items = 48
            preferred_batch_characters = 12_000
            preferred_retry_attempts = 1
            preferred_outer_concurrency = 1

            def __init__(self) -> None:
                self.calls = 0

            @property
            def last_cooldown_seconds(self) -> float:
                return 0.2 if self.calls == 1 else 0.0

            def translate(self, texts, source_language, target_language):
                del source_language, target_language
                self.calls += 1
                if self.calls == 1:
                    raise RetryableTranslationProviderError(
                        "rate limited",
                        retry_after_seconds=0.2,
                    )
                return tuple(f"VI:{text}" for text in texts)

        provider = CooldownRouterProvider()
        sleeps: list[float] = []
        source = _delta(self.root / "cooldown-delta.json", [_entry(1, "One")])

        result = translate_delta(
            source,
            self.root / "cooldown-out.json",
            provider,
            "en",
            "vi",
            sleeper=sleeps.append,
        )

        self.assertEqual(2, provider.calls)
        self.assertEqual(1, result.translated_entries)
        self.assertEqual(0, result.failure_count)
        self.assertAlmostEqual(0.2, sum(sleeps), places=6)

    def test_tm_reuse_is_reported_separately_from_new_translation(self) -> None:
        db_path = self.root / "knowledge.sqlite3"
        db = KnowledgeDatabase(db_path)
        old = _catalog(self.root / "old.json", "Old", "0020")
        new = _catalog(self.root / "new.json", "Fresh", "0020")
        work = self.root / "work.json"
        prepare_translation_work(old, new, work, knowledge_db=db)
        translate_delta(work, self.root / "first.json", EchoProvider(), "en", "vi", db_path=db_path)
        provider = EchoProvider()
        result = translate_delta(work, self.root / "second.json", provider, "en", "vi", db_path=db_path)
        self.assertEqual(0, provider.calls)
        self.assertEqual(0, result.translated_entries)
        self.assertEqual(1, result.tm_reused_entries)

    def test_ui_start_uses_selected_language_and_delta_package_scope(self) -> None:
        source = _delta(self.root / "delta.json", [_entry(1, "Text")], package="0021")
        output = self.root / "out.json"
        captured: list[tuple[str, str]] = []

        def runner(src, out, provider, source_language, target_language, progress, control):
            captured.append((source_language, target_language))
            progress(AutoTranslateProgress(1, 1, 1, 0, translated_entries=1))
            return type("Result", (), {"pending_entries": 0, "translated_entries": 1, "resumed_entries": 0, "tm_reused_entries": 0, "failure_count": 0, "request_count": 1})()

        service = TranslationUIService(self.root, runner=runner, knowledge_database=KnowledgeDatabase(self.root / "ui.sqlite3"))
        bridge = WebviewBridge(self.root, translation_service=service)
        self.assertTrue(bridge.set_language("Japanese")["ok"])
        self.assertTrue(bridge.start_translation(source, output)["ok"])
        assert service._thread is not None
        service._thread.join(2)
        state = service.refresh()["state"]
        self.assertEqual([("ja", "vi")], captured)
        self.assertEqual("0021", state["knowledge"]["scope"]["package_family"])
        self.assertEqual("ja", state["knowledge"]["scope"]["source_language"])

    def test_ui_start_marks_busy_and_reports_translation_status(self) -> None:
        source = _delta(self.root / "delta.json", [_entry(1, "Text")])
        output = self.root / "out.json"
        release = threading.Event()
        started = threading.Event()

        def runner(src, out, provider, source_language, target_language, progress, control):
            progress(AutoTranslateProgress(0, 1, 0, 0))
            started.set()
            release.wait(5)
            progress(AutoTranslateProgress(1, 1, 1, 0, translated_entries=1))
            return type(
                "Result",
                (),
                {
                    "pending_entries": 0,
                    "translated_entries": 1,
                    "resumed_entries": 0,
                    "tm_reused_entries": 0,
                    "failure_count": 0,
                    "request_count": 1,
                },
            )()

        service = TranslationUIService(
            self.root,
            runner=runner,
            knowledge_database=KnowledgeDatabase(self.root / "status.sqlite3"),
        )
        bridge = WebviewBridge(self.root, translation_service=service)
        self.assertTrue(bridge.start_translation(source, output)["ok"])
        self.assertTrue(started.wait(5))

        running_state = bridge.get_state()
        self.assertTrue(running_state["busy"])
        self.assertEqual("working", running_state["status"]["level"])
        self.assertIn("DỊCH", running_state["status"]["title"].upper())

        release.set()
        deadline = time.time() + 5
        while bridge.get_state()["busy"] and time.time() < deadline:
            time.sleep(0.02)
        final_state = bridge.get_state()
        self.assertFalse(final_state["busy"])
        self.assertEqual("success", final_state["status"]["level"])

    def test_ui_safe_pause_releases_bridge_busy_state(self) -> None:
        source = _delta(self.root / "pause-delta.json", [_entry(1, "Text")])
        output = self.root / "pause-out.json"
        started = threading.Event()
        providers: list[str] = []

        def runner(src, out, provider, source_language, target_language, progress, control):
            providers.append(provider)
            if len(providers) > 1:
                return type(
                    "Result",
                    (),
                    {
                        "pending_entries": 0,
                        "translated_entries": 1,
                        "resumed_entries": 0,
                        "tm_reused_entries": 0,
                        "failure_count": 0,
                        "request_count": 1,
                    },
                )()
            started.set()
            deadline = time.time() + 5
            while not control.should_pause_after_request() and time.time() < deadline:
                time.sleep(0.01)
            return type(
                "Result",
                (),
                {
                    "pending_entries": 1,
                    "translated_entries": 0,
                    "resumed_entries": 0,
                    "tm_reused_entries": 0,
                    "failure_count": 0,
                    "request_count": 0,
                },
            )()

        service = TranslationUIService(
            self.root,
            runner=runner,
            knowledge_database=KnowledgeDatabase(self.root / "pause-status.sqlite3"),
        )
        bridge = WebviewBridge(self.root, translation_service=service)
        self.assertTrue(bridge.start_translation(source, output)["ok"])
        self.assertTrue(started.wait(5))
        self.assertTrue(bridge.pause_translation()["ok"])

        deadline = time.time() + 5
        state = bridge.get_state()
        while state["busy"] and time.time() < deadline:
            time.sleep(0.02)
            state = bridge.get_state()

        self.assertFalse(state["busy"])
        self.assertEqual("paused", state["translation_job"]["state"])
        self.assertIn("TẠM DỪNG", state["status"]["title"].upper())
        self.assertTrue(bridge.set_translation_provider("google_free")["ok"])
        self.assertTrue(bridge.resume_translation()["ok"])

        deadline = time.time() + 5
        state = bridge.get_state()
        while state["busy"] and time.time() < deadline:
            time.sleep(0.02)
            state = bridge.get_state()

        self.assertFalse(state["busy"])
        self.assertEqual("completed", state["translation_job"]["state"])
        self.assertEqual(["ai_router_free", "google_free"], providers)

    def test_ui_final_saved_count_includes_checkpoint_and_tm_reuse(self) -> None:
        source = _delta(
            self.root / "delta.json",
            [_entry(1, "One"), _entry(2, "Two")],
        )
        output = self.root / "out.json"

        def runner(src, out, provider, source_language, target_language, progress, control):
            return type(
                "Result",
                (),
                {
                    "pending_entries": 0,
                    "translated_entries": 0,
                    "resumed_entries": 1,
                    "tm_reused_entries": 1,
                    "failure_count": 0,
                    "request_count": 0,
                },
            )()

        service = TranslationUIService(
            self.root,
            runner=runner,
            knowledge_database=KnowledgeDatabase(self.root / "reuse-ui.sqlite3"),
        )
        self.assertTrue(service.start(source, output)["ok"])
        assert service._thread is not None
        service._thread.join(2)
        job = service.refresh()["state"]["job"]
        self.assertEqual(2, job["saved"])
        self.assertEqual(1, job["resumed"])
        self.assertEqual(1, job["tm_reused"])
        self.assertEqual(0, job["translated"])

    def test_step1_bridge_passes_selected_source_language(self) -> None:
        bridge = WebviewBridge(self.root, translation_service=TranslationUIService(self.root, knowledge_database=KnowledgeDatabase(self.root / "bridge.sqlite3")))
        old = _catalog(self.root / "old.json", "Old", "0021")
        new = _catalog(self.root / "new.json", "New", "0021")
        work = self.root / "work.json"
        self.assertTrue(
            bridge.set_fields(
                {
                    "auto_old": str(old),
                    "auto_new": str(new),
                    "auto_work": str(work),
                }
            )["ok"]
        )
        bridge.set_language("Japanese")
        prepared = bridge._prepare_operation("prepare_translation")
        self.assertEqual("Crimson0020", prepared.arguments[4])
        self.assertEqual("ja", prepared.arguments[5])
        self.assertEqual("vi", prepared.arguments[6])

    def test_step3_rejects_changed_context_metadata(self) -> None:
        old = _catalog(self.root / "old.json", "Old", "0020")
        new = _catalog(self.root / "new.json", "Fresh", "0020", "Speaker A")
        template = self.root / "work.json"
        prepare_translation_work(old, new, template)
        translated = json.loads(template.read_text(encoding="utf-8"))
        translated["entries"][0]["translation"] = "Bản dịch"
        translated["entries"][0]["translation_state"] = "translated"
        translated["entries"][0]["context"] = ["Speaker B"]
        translated["entries"][0]["context_signature"] = "ctx-Speaker B"
        translated_path = self.root / "translated.json"
        translated_path.write_text(json.dumps(translated, ensure_ascii=False), encoding="utf-8")
        result = validate_translation_delta(new, template, translated_path)
        self.assertFalse(result.valid)
        self.assertTrue(any("context" in issue for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
