import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.terminology import (
    EvidenceRecord,
    TerminologyEngine,
    TermState,
)


class TestTranslationTerminology(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_terms.sqlite3"
        self.db = KnowledgeDatabase(self.db_path)
        self.scope_id = "scope_en_vi"
        with self.db.transaction() as con:
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (self.scope_id, "app1", "pkg1", "en", "vi", 1000.0),
            )
        self.engine = TerminologyEngine(self.db)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_two_consistent_legacy_identities_auto_locks(self):
        ev1 = EvidenceRecord(source_term="Excalibur", target_term="Gươm Excalibur", kind="standalone_legacy", provenance="cat1")
        ev2 = EvidenceRecord(source_term="Excalibur", target_term="Gươm Excalibur", kind="standalone_legacy", provenance="cat2")
        self.engine.record_evidence(self.scope_id, ev1)
        decision = self.engine.record_evidence(self.scope_id, ev2)

        self.assertEqual(decision.state, TermState.AUTO_LOCKED)
        self.assertGreaterEqual(decision.confidence, 0.92)
        self.assertEqual(decision.target_term, "Gươm Excalibur")

    def test_single_evidence_auto_suggested(self):
        ev = EvidenceRecord(source_term="Mana Potion", target_term="Bình năng lượng", kind="standalone_legacy", provenance="cat1")
        decision = self.engine.record_evidence(self.scope_id, ev)

        self.assertEqual(decision.state, TermState.AUTO_SUGGESTED)
        self.assertGreaterEqual(decision.confidence, 0.65)
        self.assertLess(decision.confidence, 0.92)

    def test_two_independent_route_consensus_auto_locks(self):
        ev1 = EvidenceRecord(source_term="Firestorm", target_term="Bão lửa", kind="route_consensus", provenance="route_kilo")
        ev2 = EvidenceRecord(source_term="Firestorm", target_term="Bão lửa", kind="route_consensus", provenance="route_opencode")
        self.engine.record_evidence(self.scope_id, ev1)
        decision = self.engine.record_evidence(self.scope_id, ev2)

        self.assertEqual(decision.state, TermState.AUTO_LOCKED)
        self.assertEqual(decision.target_term, "Bão lửa")

    def test_disagreement_and_margin_boundary_becomes_conflict(self):
        # Target A has 1 evidence, Target B has 1 evidence -> conflict
        ev1 = EvidenceRecord(source_term="Ring of Power", target_term="Nhẫn quyền năng", kind="standalone_legacy", provenance="cat1")
        ev2 = EvidenceRecord(source_term="Ring of Power", target_term="Chiếc nhẫn quyền lực", kind="standalone_legacy", provenance="cat2")
        self.engine.record_evidence(self.scope_id, ev1)
        decision = self.engine.record_evidence(self.scope_id, ev2)

        self.assertEqual(decision.state, TermState.CONFLICT)

    def test_user_override_priority(self):
        # Even if auto_locked or conflicting, user override sets user_locked
        ev1 = EvidenceRecord(source_term="Elixir", target_term="Thuốc tiên", kind="standalone_legacy", provenance="cat1")
        self.engine.record_evidence(self.scope_id, ev1)

        decision = self.engine.set_user_override(self.scope_id, "Elixir", "Thần dược")
        self.assertEqual(decision.state, TermState.USER_LOCKED)
        self.assertEqual(decision.target_term, "Thần dược")

        # Future automatic evidence cannot overwrite user_locked
        ev2 = EvidenceRecord(source_term="Elixir", target_term="Thuốc tiên", kind="standalone_legacy", provenance="cat2")
        decision2 = self.engine.record_evidence(self.scope_id, ev2)
        self.assertEqual(decision2.state, TermState.USER_LOCKED)
        self.assertEqual(decision2.target_term, "Thần dược")

    def test_disabled_term(self):
        self.engine.set_user_override(self.scope_id, "Gold", "Vàng")
        self.engine.disable_term(self.scope_id, "Gold")

        decision = self.engine.get_term_decision(self.scope_id, "Gold")
        self.assertIsNone(decision)

    def test_overlapping_terms_longest_match_first(self):
        self.engine.set_user_override(self.scope_id, "Knight", "Hiệp sĩ")
        self.engine.set_user_override(self.scope_id, "Dragon Knight", "Hiệp sĩ Rồng")

        # In "The brave Dragon Knight fought well", "Dragon Knight" should match first
        constraints = self.engine.extract_constraints(self.scope_id, "The brave Dragon Knight fought well")
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].source_term, "Dragon Knight")
        self.assertEqual(constraints[0].target_term, "Hiệp sĩ Rồng")

    def test_constraint_snapshot_is_reused_until_terms_change(self):
        self.engine.set_user_override(self.scope_id, "Knight", "Hiệp sĩ")
        original_connect = self.db.connect
        connect_calls = 0

        def counted_connect():
            nonlocal connect_calls
            connect_calls += 1
            return original_connect()

        with patch.object(self.db, "connect", side_effect=counted_connect):
            first = self.engine.extract_constraints(self.scope_id, "A Knight arrives")
            second = self.engine.extract_constraints(self.scope_id, "The Knight leaves")
            self.assertEqual(1, connect_calls)
            self.assertEqual("Knight", first[0].source_term)
            self.assertEqual("Knight", second[0].source_term)

            self.engine.set_user_override(self.scope_id, "Dragon", "Rồng")
            third = self.engine.extract_constraints(self.scope_id, "Dragon Knight")

        self.assertEqual(2, connect_calls)
        self.assertEqual({"Dragon", "Knight"}, {item.source_term for item in third})

    def test_hard_term_masking_and_restoration(self):
        self.engine.set_user_override(self.scope_id, "Iron Sword", "Kiếm Sắt")
        text = "Equip your Iron Sword now."
        masked, mask_meta = self.engine.mask_hard_terms(self.scope_id, text)
        self.assertIn("<_GT_0_>", masked)
        self.assertNotIn("Iron Sword", masked)

        # Provider translates around token: "Trang bị <_GT_0_> của bạn ngay."
        provider_output = "Trang bị <_GT_0_> của bạn ngay."
        restored = self.engine.restore_hard_terms(provider_output, mask_meta)
        self.assertEqual(restored, "Trang bị Kiếm Sắt của bạn ngay.")

    def test_prepared_hard_terms_reuses_one_match_plan(self):
        self.engine.set_user_override(self.scope_id, "Knight", "Hiệp sĩ")
        self.engine.set_user_override(self.scope_id, "Dragon Knight", "Hiệp sĩ Rồng")

        masked, metadata, constraints = self.engine.prepare_hard_terms(
            self.scope_id,
            "Knight greets the Dragon Knight.",
        )

        self.assertEqual(
            [item.source_term for item in constraints],
            ["Dragon Knight", "Knight"],
        )
        self.assertEqual("<_GT_1_> greets the <_GT_0_>.", masked)
        self.assertEqual(
            {"<_GT_0_>": "Hiệp sĩ Rồng", "<_GT_1_>": "Hiệp sĩ"},
            metadata.token_to_target,
        )

    def test_soft_suggestion_index_matches_case_insensitively_once(self):
        now = 1000.0
        with self.db.transaction() as con:
            con.executemany(
                """
                INSERT INTO terms (
                    term_id, scope_id, source_term, target_term, state,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'auto_suggested', ?, ?, ?)
                """,
                [
                    ("soft_castle", self.scope_id, "Castle", "Lâu đài", 0.75, now, now),
                    ("soft_gate", self.scope_id, "Gate", "Cổng", 0.70, now, now),
                    ("soft_unused", self.scope_id, "Potion", "Bình thuốc", 0.70, now, now),
                ],
            )

        matched = self.engine.match_soft_suggestions(
            self.scope_id,
            "CASTLE gate beside another castle gate",
        )

        self.assertEqual([item.source_term for item in matched], ["Castle", "Gate"])

    def test_soft_suggestion_snapshot_reused_for_many_texts(self):
        now = 1000.0
        with self.db.transaction() as con:
            con.executemany(
                """
                INSERT INTO terms (
                    term_id, scope_id, source_term, target_term, state,
                    confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'auto_suggested', ?, ?, ?)
                """,
                [
                    (
                        f"soft_{index}",
                        self.scope_id,
                        f"Term{index:04d}X",
                        f"Đích{index}",
                        0.70,
                        now,
                        now,
                    )
                    for index in range(2000)
                ],
            )

        original_connect = self.db.connect
        connect_calls = 0

        def counted_connect():
            nonlocal connect_calls
            connect_calls += 1
            return original_connect()

        with patch.object(self.db, "connect", side_effect=counted_connect):
            for index in range(50):
                matched = self.engine.match_soft_suggestions(
                    self.scope_id,
                    f"Text with Term{index:04d}X only.",
                )
                self.assertEqual(1, len(matched))

        self.assertEqual(1, connect_calls)

    def test_hard_term_unicode_lowercase_expansion_keeps_original_spans(self):
        self.engine.set_user_override(self.scope_id, "İtem", "Vật phẩm")

        masked, metadata, constraints = self.engine.prepare_hard_terms(
            self.scope_id,
            "Use İtem now",
        )

        self.assertEqual("Use <_GT_0_> now", masked)
        self.assertEqual("İtem", constraints[0].source_term)
        self.assertEqual({"<_GT_0_>": "Vật phẩm"}, metadata.token_to_target)


if __name__ == "__main__":
    unittest.main()
