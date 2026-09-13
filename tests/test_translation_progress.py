from __future__ import annotations

import threading
import json
import tempfile
import unittest
from pathlib import Path

from crimson0020.auto_translate import translate_delta
from crimson0020.translation_runtime.progress import (
    ProgressReporter,
    ProgressSnapshot,
    ProgressState,
    TranslationControl,
)


def _delta(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "crimson.translation-delta",
                "schema_version": 1,
                "kind": "sparse-delta",
                "source": {"catalog_fingerprint": "A" * 64},
                "metadata": {"format_version": 2},
                "entries": [
                    {
                        "marker": "0100000000000000",
                        "key": "one",
                        "original": "One",
                        "translation": "One",
                        "translation_state": "pending",
                    },
                    {
                        "marker": "0200000000000000",
                        "key": "two",
                        "original": "Two",
                        "translation": "Two",
                        "translation_state": "pending",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class TestTranslationProgress(unittest.TestCase):
    def test_snapshots_are_immutable_and_counts_are_monotonic(self) -> None:
        now = [0.0]
        seen: list[ProgressSnapshot] = []
        reporter = ProgressReporter(
            "job-1", 10, seen.append, clock=lambda: now[0]
        )
        reporter.publish(state=ProgressState.RUNNING)
        now[0] = 0.3
        reporter.publish(
            saved_entries=2,
            translated_entries=2,
            pending_entries=8,
            request_count=1,
            state=ProgressState.RUNNING,
        )
        snapshot = reporter.snapshot
        self.assertEqual(2, snapshot.saved_entries)
        with self.assertRaises(AttributeError):
            snapshot.saved_entries = 3  # type: ignore[misc]
        self.assertEqual(2, seen[-1].saved_entries)
        with self.assertRaisesRegex(ValueError, "saved_entries cannot decrease"):
            reporter.publish(saved_entries=1, pending_entries=9)

    def test_events_are_throttled_to_four_per_second_and_flushes_terminal_state(self) -> None:
        now = [0.0]
        seen: list[ProgressSnapshot] = []
        reporter = ProgressReporter("job-2", 5, seen.append, clock=lambda: now[0])
        reporter.publish(state=ProgressState.RUNNING)
        for value in range(1, 5):
            now[0] = value * 0.05
            reporter.publish(saved_entries=value, pending_entries=5 - value)
        self.assertEqual(1, len(seen))
        now[0] = 0.3
        reporter.publish(
            saved_entries=5,
            translated_entries=5,
            pending_entries=0,
            state=ProgressState.COMPLETED,
        )
        self.assertEqual(2, len(seen))

    def test_callback_errors_are_recorded_without_breaking_translation(self) -> None:
        calls = 0

        def callback(_snapshot: ProgressSnapshot) -> None:
            nonlocal calls
            calls += 1
            raise RuntimeError("ui callback failed")

        reporter = ProgressReporter("job-3", 1, callback)
        reporter.publish(state=ProgressState.RUNNING)
        self.assertEqual(1, calls)
        self.assertEqual(("ui callback failed",), reporter.callback_errors)

    def test_pause_control_is_thread_safe_and_resumeable(self) -> None:
        control = TranslationControl()
        started = threading.Event()

        def pause() -> None:
            started.wait(1)
            control.request_pause("safe pause")

        worker = threading.Thread(target=pause)
        worker.start()
        started.set()
        worker.join(1)
        self.assertTrue(control.pause_requested)
        self.assertTrue(control.should_pause_after_request())
        self.assertEqual("safe pause", control.reason)
        control.clear_pause()
        self.assertFalse(control.pause_requested)
        control.cancel("stop")
        self.assertTrue(control.cancelled)

    def test_pending_terminal_snapshot_is_explicit(self) -> None:
        reporter = ProgressReporter("job-4", 4)
        reporter.publish(
            saved_entries=2,
            translated_entries=2,
            pending_entries=2,
            failure_count=1,
            state=ProgressState.COMPLETED_WITH_PENDING,
            failure_summary="provider unavailable",
        )
        self.assertEqual(
            ProgressState.COMPLETED_WITH_PENDING, reporter.snapshot.state
        )
        self.assertEqual("provider unavailable", reporter.snapshot.failure_summary)

    def test_translation_pauses_after_current_request_and_resumes_with_provider_change(
        self,
    ) -> None:
        class PausingProvider:
            key = "ai_router_free"

            def __init__(self, control: TranslationControl) -> None:
                self.control = control

            def translate(self, texts, _source, _target):
                self.control.request_pause("safe pause")
                return tuple(f"vi:{text}" for text in texts)

        class ResumeProvider:
            key = "google_free"

            def translate(self, texts, _source, _target):
                return tuple(f"vi:{text}" for text in texts)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = _delta(root / "delta.json")
            output = root / "output.json"
            control = TranslationControl()
            reporter = ProgressReporter("job-pause", 2)
            paused = translate_delta(
                source,
                output,
                PausingProvider(control),
                "en",
                "vi",
                translation_control=control,
                progress_reporter=reporter,
                job_id="job-pause",
                max_batch_items=1,
            )
            self.assertEqual(1, paused.translated_entries)
            self.assertEqual(1, paused.pending_entries)
            self.assertEqual(ProgressState.PAUSED, reporter.snapshot.state)

            control.clear_pause()
            resumed = translate_delta(
                source,
                output,
                ResumeProvider(),
                "en",
                "vi",
                translation_control=control,
                progress_reporter=reporter,
                job_id="job-pause",
                max_batch_items=1,
            )
            self.assertEqual(1, resumed.translated_entries)
            self.assertEqual(0, resumed.pending_entries)
            self.assertEqual(ProgressState.COMPLETED, reporter.snapshot.state)


if __name__ == "__main__":
    unittest.main()
