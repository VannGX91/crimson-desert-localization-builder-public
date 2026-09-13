import tempfile
import unittest
from pathlib import Path

from crimson0020.translation_runtime.checkpoint import CheckpointRecord
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.memory import (
    ExactLookupRequest,
    TranslationMemoryRepository,
    TranslationSegment,
)


class TestTranslationMemoryRepository(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_tm.sqlite3"
        self.db = KnowledgeDatabase(self.db_path)
        self.scope_id = "test_scope"
        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (self.scope_id, "app1", "pkg1", "en", "vi", 1000.0),
            )
            con.execute(
                "INSERT INTO lineages (lineage_id, scope_id, created_at) VALUES (?, ?, ?)",
                ("lineage_1", self.scope_id, 1000.0),
            )
            con.execute(
                "INSERT INTO lineages (lineage_id, scope_id, created_at) VALUES (?, ?, ?)",
                ("lineage_2", self.scope_id, 1000.0),
            )
        self.repo = TranslationMemoryRepository(self.db)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_exact_marker_key_reuse(self):
        # Insert a segment
        seg = TranslationSegment(
            segment_id="seg_1",
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            marker="loc_string",
            key="KEY_HELLO",
            source_hash="hash_hello",
            source_text="Hello world",
            target_text="Xin chào thế giới",
            context_signature="ctx_1",
            hard_terms_hash="terms_none",
            glossary_revision=1,
            status="active",
        )
        self.repo.save_segment(seg)

        # Exact lookup by marker + key + source_hash + lineage + scope
        req = ExactLookupRequest(
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            source_hash="hash_hello",
            marker="loc_string",
            key="KEY_HELLO",
        )
        found = self.repo.lookup_exact(req)
        self.assertIsNotNone(found)
        self.assertEqual(found.target_text, "Xin chào thế giới")

    def test_cross_key_contextual_reuse(self):
        # Insert segment under KEY_A
        seg = TranslationSegment(
            segment_id="seg_1",
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            marker="loc_string",
            key="KEY_A",
            source_hash="hash_agree",
            source_text="I agree.",
            target_text="Tôi đồng ý.",
            context_signature="scene_dialogue_accept",
            hard_terms_hash="terms_none",
            glossary_revision=1,
            status="active",
        )
        self.repo.save_segment(seg)

        # Lookup for different KEY_B but same source_hash + context_signature + hard_terms_hash + glossary_revision
        req = ExactLookupRequest(
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            source_hash="hash_agree",
            marker="loc_string",
            key="KEY_B",
            context_signature="scene_dialogue_accept",
            hard_terms_hash="terms_none",
            glossary_revision=1,
        )
        found = self.repo.lookup_exact(req)
        self.assertIsNotNone(found)
        self.assertEqual(found.target_text, "Tôi đồng ý.")

    def test_exact_many_preserves_rule_priority_order_and_misses(self):
        self.repo.save_segments(
            (
                TranslationSegment(
                    segment_id="seg_exact_batch",
                    scope_id=self.scope_id,
                    lineage_id="lineage_1",
                    marker="loc_string",
                    key="KEY_EXACT",
                    source_hash="hash_exact",
                    source_text="Exact",
                    target_text="Chính xác",
                    context_signature="ctx_exact",
                    hard_terms_hash="old_terms",
                    glossary_revision=7,
                    status="active",
                ),
                TranslationSegment(
                    segment_id="seg_context_batch",
                    scope_id=self.scope_id,
                    lineage_id="lineage_1",
                    marker="loc_string",
                    key="KEY_CONTEXT_A",
                    source_hash="hash_context",
                    source_text="Context",
                    target_text="Ngữ cảnh",
                    context_signature="ctx_shared",
                    hard_terms_hash="terms_shared",
                    glossary_revision=2,
                    status="active",
                ),
            )
        )
        requests = (
            ExactLookupRequest(
                scope_id=self.scope_id,
                lineage_id="lineage_1",
                marker="loc_string",
                key="KEY_EXACT",
                source_hash="hash_exact",
                context_signature="ctx_different",
                hard_terms_hash="different_terms",
                glossary_revision=99,
            ),
            ExactLookupRequest(
                scope_id=self.scope_id,
                lineage_id="lineage_1",
                marker="loc_string",
                key="KEY_CONTEXT_B",
                source_hash="hash_context",
                context_signature="ctx_shared",
                hard_terms_hash="terms_shared",
                glossary_revision=2,
            ),
            ExactLookupRequest(
                scope_id=self.scope_id,
                lineage_id="lineage_2",
                marker="loc_string",
                key="KEY_EXACT",
                source_hash="hash_exact",
            ),
        )

        found = self.repo.lookup_exact_many(requests)

        self.assertEqual(3, len(found))
        self.assertEqual("seg_exact_batch", found[0].segment_id)
        self.assertEqual("seg_context_batch", found[1].segment_id)
        self.assertIsNone(found[2])

    def test_context_mismatch_rejection(self):
        seg = TranslationSegment(
            segment_id="seg_1",
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            marker="loc_string",
            key="KEY_A",
            source_hash="hash_agree",
            source_text="I agree.",
            target_text="Tôi đồng ý.",
            context_signature="scene_dialogue_accept",
            hard_terms_hash="terms_none",
            glossary_revision=1,
            status="active",
        )
        self.repo.save_segment(seg)

        # Different context signature should NOT match across keys
        req = ExactLookupRequest(
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            source_hash="hash_agree",
            marker="loc_string",
            key="KEY_B",
            context_signature="scene_dialogue_reject",
            hard_terms_hash="terms_none",
            glossary_revision=1,
        )
        found = self.repo.lookup_exact(req)
        self.assertIsNone(found)

    def test_lineage_mismatch_rejection(self):
        seg = TranslationSegment(
            segment_id="seg_1",
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            marker="loc_string",
            key="KEY_A",
            source_hash="hash_hello",
            source_text="Hello",
            target_text="Xin chào",
            context_signature="ctx_1",
            hard_terms_hash="",
            glossary_revision=1,
            status="active",
        )
        self.repo.save_segment(seg)

        # Lookup in different lineage should NOT match
        req = ExactLookupRequest(
            scope_id=self.scope_id,
            lineage_id="lineage_2",
            source_hash="hash_hello",
            marker="loc_string",
            key="KEY_A",
        )
        self.assertIsNone(self.repo.lookup_exact(req))

    def test_fuzzy_prompt_only_not_returned_as_exact(self):
        seg = TranslationSegment(
            segment_id="seg_1",
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            marker="loc_string",
            key="KEY_A",
            source_hash="hash_sword_of_fire",
            source_text="The flaming sword of fury",
            target_text="Thanh kiếm rực lửa thịnh nộ",
            context_signature="",
            hard_terms_hash="",
            glossary_revision=1,
            status="active",
        )
        self.repo.save_segment(seg)

        # Exact lookup for similar text should return None
        req = ExactLookupRequest(
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            source_hash="hash_sword_of_ice",
            marker="loc_string",
            key="KEY_B",
        )
        self.assertIsNone(self.repo.lookup_exact(req))

        # But fuzzy search should return it as prompt example only
        fuzzy_matches = self.repo.search_fuzzy(self.scope_id, "flaming sword")
        self.assertEqual(len(fuzzy_matches), 1)
        self.assertEqual(fuzzy_matches[0].target_text, "Thanh kiếm rực lửa thịnh nộ")

    def test_checkpoint_import_idempotent_and_excludes_technical(self):
        records = [
            CheckpointRecord("loc", "K1", "h1", "Translated 1", context_signature="ctx1"),
            CheckpointRecord("loc", "K2", "h2", "K2", quality_flags=("source_equals_target",)),  # technical/source=target
            CheckpointRecord("loc", "K3", "h3", "[TECH_ID]", quality_flags=("technical_identifier",)),  # technical
        ]
        # First import
        count1 = self.repo.import_checkpoint_records(self.scope_id, "lineage_1", records)
        self.assertEqual(count1, 1)  # Only K1 is imported

        # Duplicate import should be idempotent
        count2 = self.repo.import_checkpoint_records(self.scope_id, "lineage_1", records)
        self.assertEqual(count2, 0)

    def test_glossary_revision_invalidation(self):
        seg = TranslationSegment(
            segment_id="seg_1",
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            marker="loc_string",
            key="KEY_A",
            source_hash="hash_term",
            source_text="Warrior shield",
            target_text="Khiên chiến binh",
            context_signature="ctx",
            hard_terms_hash="hash_term_old",
            glossary_revision=1,
            status="active",
        )
        self.repo.save_segment(seg)

        # Invalidate segments when glossary hard terms change
        invalidated = self.repo.invalidate_terms_hash(self.scope_id, "hash_term_old")
        self.assertEqual(invalidated, 1)

        # Inactive segment should no longer match exact lookup
        req = ExactLookupRequest(
            scope_id=self.scope_id,
            lineage_id="lineage_1",
            source_hash="hash_term",
            marker="loc_string",
            key="KEY_A",
        )
        self.assertIsNone(self.repo.lookup_exact(req))


if __name__ == "__main__":
    unittest.main()
