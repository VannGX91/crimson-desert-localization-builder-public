import shutil
import tempfile
import unittest
from pathlib import Path

from crimson0020.translation_runtime.errors import TranslationRuntimeError
from crimson0020.translation_runtime.failure_state import (
    FailureRecord,
    FailureStateManager,
)
from crimson0020.translation_runtime.job_control import (
    CancellationToken,
    FileLock,
    JobController,
    JobSnapshot,
    JobState,
)


class TestTranslationJobControl(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lock_path = Path(self.temp_dir) / "job.lock"
        self.failure_path = Path(self.temp_dir) / "job.failures.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_job_state_transitions(self):
        controller = JobController("job-1", total_count=10)
        self.assertEqual(controller.state, JobState.IDLE)

        # Valid transitions
        controller.start()
        self.assertEqual(controller.state, JobState.RUNNING)

        controller.pause_requested()
        self.assertEqual(controller.state, JobState.PAUSING)

        controller.paused()
        self.assertEqual(controller.state, JobState.PAUSED)

        controller.resume()
        self.assertEqual(controller.state, JobState.RUNNING)

        controller.complete(pending_count=0)
        self.assertEqual(controller.state, JobState.COMPLETED)

        # Invalid transition from COMPLETED -> RUNNING
        with self.assertRaises(TranslationRuntimeError):
            controller.start()

    def test_completed_with_pending(self):
        controller = JobController("job-2", total_count=10)
        controller.start()
        controller.complete(pending_count=3)
        self.assertEqual(controller.state, JobState.COMPLETED_WITH_PENDING)

    def test_invalid_direct_transition_raises(self):
        controller = JobController("job-3", total_count=10)
        with self.assertRaises(TranslationRuntimeError):
            controller.paused()  # Cannot jump from IDLE to PAUSED

    def test_cancellation_token(self):
        token = CancellationToken()
        self.assertFalse(token.is_cancelled)
        token.cancel(reason="User request")
        self.assertTrue(token.is_cancelled)
        self.assertEqual(token.cancel_reason, "User request")

    def test_file_lock_mutual_exclusion(self):
        lock1 = FileLock(self.lock_path)
        self.assertTrue(lock1.acquire())

        # Second lock attempt must fail
        lock2 = FileLock(self.lock_path)
        self.assertFalse(lock2.acquire())

        # Release first lock
        lock1.release()

        # Now second lock can acquire
        self.assertTrue(lock2.acquire())
        lock2.release()

    def test_failure_state_atomic_rewrite_and_deduplication(self):
        fsm = FailureStateManager(self.failure_path)
        # Record failure for item 1
        rec1 = FailureRecord(marker="m1", key="k1", original_sha256="H1", message="Rate limited")
        fsm.record_failure(rec1)
        self.assertTrue(self.failure_path.is_file())

        failures = fsm.load_failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[("m1", "k1")].message, "Rate limited")

        # Update failure for item 1 (must overwrite, no duplicates in failure sidecar)
        rec1_update = FailureRecord(marker="m1", key="k1", original_sha256="H1", message="Timeout")
        fsm.record_failure(rec1_update)
        failures = fsm.load_failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[("m1", "k1")].message, "Timeout")

        # After success on item 1, failure entry is removed and sidecar removed if empty
        fsm.record_success("m1", "k1")
        self.assertFalse(self.failure_path.exists())

    def test_job_snapshot_immutability(self):
        controller = JobController("job-snap", total_count=5)
        controller.start()
        controller.record_progress(completed=2, pending=3, failures=0)
        snap = controller.snapshot()
        self.assertIsInstance(snap, JobSnapshot)
        self.assertEqual(snap.completed_count, 2)
        self.assertEqual(snap.pending_count, 3)
        self.assertEqual(snap.state, JobState.RUNNING)


if __name__ == "__main__":
    unittest.main()
