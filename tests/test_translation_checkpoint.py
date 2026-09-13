import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from crimson0020.translation_runtime.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointRecord,
    append_checkpoint_batch,
    initialize_checkpoint,
    load_checkpoint,
)
from crimson0020.translation_runtime.errors import CheckpointError


class TestTranslationCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = Path(self.temp_dir) / "test.jsonl"
        self.source_fingerprint = "fp_12345"
        self.source_lang = "en"
        self.target_lang = "vi"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()

    def test_initialize_and_load_empty(self):
        self.assertEqual(CHECKPOINT_FORMAT_VERSION, 2)
        initialize_checkpoint(
            self.checkpoint_path,
            source_fingerprint=self.source_fingerprint,
            source_language=self.source_lang,
            target_language=self.target_lang,
            provider="google_free",
        )
        self.assertTrue(self.checkpoint_path.is_file())
        completed = load_checkpoint(
            self.checkpoint_path,
            source_fingerprint=self.source_fingerprint,
            source_language=self.source_lang,
            target_language=self.target_lang,
            candidates=[],
        )
        self.assertEqual(len(completed), 0)

    def test_append_and_resume_with_optional_metadata(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        candidates = [
            {
                "marker": "m1",
                "key": "k1",
                "original": "Hello World",
                "context_signature": "sig_1",
            },
            {"marker": "m2", "key": "k2", "original": "Good Morning"},
        ]
        records = [
            CheckpointRecord(
                marker="m1",
                key="k1",
                original_sha256=self._hash("Hello World"),
                translation="Xin chào Thế giới",
                route_id="route_opencode",
                model_id="model_free",
                context_signature="sig_1",
                glossary_revision=2,
            )
        ]
        append_checkpoint_batch(self.checkpoint_path, records)

        # Provider switch: resuming under a different provider
        completed = load_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
            candidates=candidates,
        )
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[("m1", "k1")], "Xin chào Thế giới")

    def test_context_mismatch_or_missing_context_is_not_resumed(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        candidates = [
            {
                "marker": "m1",
                "key": "k1",
                "original": "Hello",
                "context_signature": "ctx-new",
            },
            {
                "marker": "m2",
                "key": "k2",
                "original": "World",
                "context_signature": "ctx-required",
            },
        ]
        append_checkpoint_batch(
            self.checkpoint_path,
            [
                CheckpointRecord(
                    "m1",
                    "k1",
                    self._hash("Hello"),
                    "Xin chào",
                    context_signature="ctx-old",
                ),
                CheckpointRecord("m2", "k2", self._hash("World"), "Thế giới"),
            ],
        )

        completed = load_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
            candidates=candidates,
        )

        self.assertEqual({}, completed)

    def test_unknown_checkpoint_identity_is_rejected(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        append_checkpoint_batch(
            self.checkpoint_path,
            [CheckpointRecord("m1", "k1", self._hash("Hello"), "Xin chào")],
        )

        with self.assertRaisesRegex(CheckpointError, "unknown"):
            load_checkpoint(
                self.checkpoint_path,
                self.source_fingerprint,
                self.source_lang,
                self.target_lang,
                candidates=[],
            )

    def test_wrong_language_or_fingerprint_rejected(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        # Wrong target language
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(
                self.checkpoint_path,
                self.source_fingerprint,
                self.source_lang,
                "fr",  # Expected vi, got fr
                candidates=[],
            )
        self.assertIn("match", str(ctx.exception).lower())

        # Wrong fingerprint
        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(
                self.checkpoint_path,
                "wrong_fp",
                self.source_lang,
                self.target_lang,
                candidates=[],
            )
        self.assertIn("match", str(ctx.exception).lower())

    def test_duplicate_identity_rejected(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        candidates = [{"marker": "m1", "key": "k1", "original": "Hello"}]
        rec1 = CheckpointRecord("m1", "k1", self._hash("Hello"), "Xin chào")
        rec2 = CheckpointRecord("m1", "k1", self._hash("Hello"), "Chào")
        append_checkpoint_batch(self.checkpoint_path, [rec1])
        append_checkpoint_batch(self.checkpoint_path, [rec2])

        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(
                self.checkpoint_path,
                self.source_fingerprint,
                self.source_lang,
                self.target_lang,
                candidates=candidates,
            )
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_truncated_tail_repair(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        candidates = [{"marker": "m1", "key": "k1", "original": "Hello"}]
        rec1 = CheckpointRecord("m1", "k1", self._hash("Hello"), "Xin chào")
        append_checkpoint_batch(self.checkpoint_path, [rec1])

        # Append truncated line at the end (missing newline and incomplete json)
        with self.checkpoint_path.open("a", encoding="utf-8") as f:
            f.write('{"record":"batch","translations":[{"marker":"m2')

        completed = load_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
            candidates=candidates,
        )
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[("m1", "k1")], "Xin chào")

        # Confirm file was repaired and ends cleanly with newline
        content = self.checkpoint_path.read_text(encoding="utf-8")
        self.assertTrue(content.endswith("\n"))
        self.assertNotIn("m2", content)

    def test_middle_corruption_rejected_with_line_number(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        candidates = [
            {"marker": "m1", "key": "k1", "original": "One"},
            {"marker": "m2", "key": "k2", "original": "Two"},
        ]
        rec1 = CheckpointRecord("m1", "k1", self._hash("One"), "Một")
        rec2 = CheckpointRecord("m2", "k2", self._hash("Two"), "Hai")
        append_checkpoint_batch(self.checkpoint_path, [rec1])
        append_checkpoint_batch(self.checkpoint_path, [rec2])

        # Corrupt line 2 (the first batch record)
        lines = self.checkpoint_path.read_text(encoding="utf-8").splitlines()
        lines[1] = "CORRUPTED_JSON_LINE"
        self.checkpoint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(
                self.checkpoint_path,
                self.source_fingerprint,
                self.source_lang,
                self.target_lang,
                candidates=candidates,
            )
        self.assertIn("line 2", str(ctx.exception).lower())

    def test_synthetic_scale_load(self):
        # 8,714-row synthetic load
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )
        count = 8714
        candidates = [
            {"marker": f"m_{i}", "key": f"k_{i}", "original": f"Text {i}"}
            for i in range(count)
        ]
        batch_size = 500
        for b_start in range(0, count, batch_size):
            b_end = min(b_start + batch_size, count)
            batch = [
                CheckpointRecord(
                    marker=f"m_{i}",
                    key=f"k_{i}",
                    original_sha256=self._hash(f"Text {i}"),
                    translation=f"Dịch {i}",
                )
                for i in range(b_start, b_end)
            ]
            append_checkpoint_batch(self.checkpoint_path, batch)

        completed = load_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
            candidates=candidates,
        )
        self.assertEqual(len(completed), 8714)
        self.assertEqual(completed[("m_0", "k_0")], "Dịch 0")
        self.assertEqual(completed[("m_8713", "k_8713")], "Dịch 8713")


    def test_checkpoint_record_serialization(self):
        r = CheckpointRecord(
            marker="m",
            key="k",
            original_sha256="HASH",
            translation="Dịch",
            route_id="r1",
            model_id="m1",
            context_signature="sig",
            glossary_revision=1,
            quality_flags=("fuzzy", "glossary_applied"),
        )
        d = r.to_dict()
        self.assertEqual(CheckpointRecord.from_dict(d), r)

    def test_load_non_existent_and_empty_append(self):
        res = load_checkpoint(
            Path(self.temp_dir) / "does_not_exist.jsonl",
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
            candidates=[],
        )
        self.assertEqual(res, {})

        # Empty append should do nothing
        append_checkpoint_batch(self.checkpoint_path, [])

    def test_candidate_as_object_and_hash_mismatch(self):
        initialize_checkpoint(
            self.checkpoint_path,
            self.source_fingerprint,
            self.source_lang,
            self.target_lang,
        )

        class CandidateObj:
            def __init__(self, marker, key, original):
                self.marker = marker
                self.key = key
                self.original = original

        candidates = [CandidateObj("m1", "k1", "Original Text")]
        # Wrong original hash
        rec = CheckpointRecord("m1", "k1", "WRONG_HASH", "Dịch")
        append_checkpoint_batch(self.checkpoint_path, [rec.to_dict()])

        with self.assertRaises(CheckpointError) as ctx:
            load_checkpoint(
                self.checkpoint_path,
                self.source_fingerprint,
                self.source_lang,
                self.target_lang,
                candidates=candidates,
            )
        self.assertIn("does not match candidate", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
