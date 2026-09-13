import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.service import KnowledgeService, ScopeIdentifier
from crimson0020.translation_work import prepare_translation_work


def _catalog(entries: list[dict[str, str]]) -> dict[str, object]:
    return {
        "metadata": {
            "format_version": 2,
            "layout": "multi-paloc",
            "package_folder": "0020",
            "palocs": [
                {
                    "path": "game/localization/eng/a.paloc",
                    "entry_count": len(entries),
                    "tail_bytes": "",
                }
            ],
        },
        "entries": entries,
    }


def _entry(marker: str, key: str, original: str, translation: str) -> dict[str, str]:
    return {
        "marker": marker,
        "key": key,
        "original": original,
        "translation": translation,
    }


class TranslationWorkTests(unittest.TestCase):
    def test_legacy_glossary_ingestion_is_conservative_and_conflict_safe(self) -> None:
        old = _catalog(
            [
                _entry("0100000000000000", "abyss-1", "Abyss", "Abyss"),
                _entry("0200000000000000", "abyss-2", "Abyss", "Abyss"),
                _entry("0300000000000000", "gate-1", "Abyss Gate", "Cổng Abyss"),
                _entry("0400000000000000", "gate-2", "Abyss Gate", "Cổng Abyss"),
                _entry("0410000000000000", "gear-1", "Abyss Gear", "Trang bị Abyss"),
                _entry("0420000000000000", "gear-2", "Abyss Gear", "Trang bị Abyss"),
                _entry("0430000000000000", "beltran-1", "Beltran Plate Helm (+3)", "Mũ giáp tấm Beltran (+3)"),
                _entry("0440000000000000", "beltran-2", "Beltran Plate Helm (+3)", "Mũ giáp tấm Beltran (+3)"),
                _entry("0450000000000000", "aeserion-1", "Aeserion Sword (+3)", "Kiếm Aeserion (+3)"),
                _entry("0460000000000000", "aeserion-2", "Aeserion Sword (+3)", "Kiếm Aeserion (+3)"),
                _entry("0470000000000000", "atag-1", "A.T.A.G. Cannon", "Pháo A.T.A.G."),
                _entry("0480000000000000", "atag-2", "A.T.A.G. Cannon", "Pháo A.T.A.G."),
                _entry("0500000000000000", "key-1", "D", "D"),
                _entry("0600000000000000", "key-2", "D", "D"),
                _entry("0700000000000000", "sentence-1", "Enter the Abyss Gate.", "Tiến vào Cổng Abyss."),
                _entry("0800000000000000", "sentence-2", "Enter the Abyss Gate.", "Tiến vào Cổng Abyss."),
                _entry("0900000000000000", "placeholder-1", "Give Coin ({Money:Money_Copper:1})", "Đưa tiền ({Money:Money_Copper:1})"),
                _entry("0a00000000000000", "placeholder-2", "Give Coin ({Money:Money_Copper:1})", "Đưa tiền ({Money:Money_Copper:1})"),
                _entry("0b00000000000000", "animal-1", "Meerkat", "Chồn Meerkat"),
                _entry("0c00000000000000", "animal-2", "Meerkat", "Cầy Meerkat"),
            ]
        )
        new = _catalog([])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            database = KnowledgeDatabase(root / "knowledge.sqlite3")
            old_path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            new_path.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")

            prepare_translation_work(old_path, new_path, work_path, knowledge_db=database)

            scope = ScopeIdentifier("Crimson0020", "0020", "en", "vi").scope_id
            with database.connect() as connection:
                rows = connection.execute(
                    "SELECT source_term, target_term, state FROM terms WHERE scope_id = ? ORDER BY source_term, target_term",
                    (scope,),
                ).fetchall()

        self.assertIn(("Abyss", "Abyss", "auto_locked"), rows)
        self.assertIn(("Abyss Gate", "Cổng Abyss", "auto_locked"), rows)
        self.assertIn(("Abyss Gear", "Trang bị Abyss", "auto_locked"), rows)
        for unsafe_source in ("Beltran Plate Helm (+3)", "Aeserion Sword (+3)", "A.T.A.G. Cannon"):
            unsafe_rows = [row for row in rows if row[0] == unsafe_source]
            self.assertTrue(unsafe_rows)
            self.assertTrue(all(row[2] != "auto_locked" for row in unsafe_rows))
        self.assertFalse(any(row[0] == "D" for row in rows))
        self.assertFalse(any(row[0] == "Enter the Abyss Gate." for row in rows))
        self.assertFalse(any(row[0].startswith("Give Coin") for row in rows))
        meerkat = [row for row in rows if row[0] == "Meerkat"]
        self.assertTrue(meerkat)
        self.assertTrue(all(row[2] == "conflict" for row in meerkat))

    def test_legacy_glossary_rebuild_removes_buggy_generated_locks_but_preserves_user_override(self) -> None:
        old = _catalog(
            [
                _entry("0100000000000000", "beltran-1", "Beltran Plate Helm (+3)", "Mũ giáp tấm Beltran (+3)"),
                _entry("0200000000000000", "beltran-2", "Beltran Plate Helm (+3)", "Mũ giáp tấm Beltran (+3)"),
            ]
        )
        new = _catalog([])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            database = KnowledgeDatabase(root / "knowledge.sqlite3")
            scope = ScopeIdentifier("Crimson0020", "0020", "en", "vi")
            service = KnowledgeService(database)
            service.set_glossary_override(scope, "Abyss", "Vực Sâu Tùy Chỉnh")
            now = 1.0
            buggy_term_id = f"term_{scope.scope_id}_0123456789abcdef"
            with database.transaction() as connection:
                connection.execute(
                    "INSERT INTO terms (term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at) VALUES (?, ?, ?, ?, 'user_locked', 1.0, ?, ?)",
                    (buggy_term_id, scope.scope_id, "Beltran Plate Helm (+3)", "Mũ giáp tấm Beltran (+3)", now, now),
                )
            old_path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            new_path.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")

            prepare_translation_work(old_path, new_path, work_path, knowledge_db=database)

            with database.connect() as connection:
                rows = connection.execute(
                    "SELECT term_id, source_term, target_term, state FROM terms WHERE scope_id = ? ORDER BY source_term, target_term",
                    (scope.scope_id,),
                ).fetchall()

        self.assertNotIn(buggy_term_id, {row[0] for row in rows})
        self.assertIn(("Abyss", "Vực Sâu Tùy Chỉnh", "user_locked"), {(row[1], row[2], row[3]) for row in rows})
        beltran = [row for row in rows if row[1] == "Beltran Plate Helm (+3)"]
        self.assertTrue(beltran)
        self.assertTrue(all(row[3] != "auto_locked" for row in beltran))

    def test_compares_old_and_new_catalogs_into_repack_ready_work_json(self) -> None:
        old = _catalog(
            [
                _entry("0100000000000000", "same", "Hello", "Xin chào"),
                _entry("0200000000000000", "changed", "Old text", "Bản cũ"),
                _entry("0300000000000000", "removed", "Gone", "Đã mất"),
            ]
        )
        new = _catalog(
            [
                _entry("0100000000000000", "same", "Hello", "Hello"),
                _entry("0200000000000000", "changed", "New text", "New text"),
                _entry("0400000000000000", "new", "Brand new", "Brand new"),
            ]
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work" / "translation-work.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            new_path.write_text(json.dumps(new), encoding="utf-8")
            old_before = old_path.read_bytes()
            new_before = new_path.read_bytes()

            result = prepare_translation_work(old_path, new_path, work_path)

            work = json.loads(work_path.read_text(encoding="utf-8"))
            self.assertEqual("sparse-delta", work["kind"])
            self.assertRegex(work["source"]["catalog_fingerprint"], r"^[0-9A-F]{64}$")
            self.assertRegex(
                work["source"]["base_catalog_fingerprint"], r"^[0-9A-F]{64}$"
            )
            self.assertEqual(2, len(work["entries"]))
            self.assertEqual(
                ["changed", "new"], [entry["key"] for entry in work["entries"]]
            )
            self.assertEqual(
                ["", ""], [entry["translation"] for entry in work["entries"]]
            )
            self.assertEqual(old_before, old_path.read_bytes())
            self.assertEqual(new_before, new_path.read_bytes())
            self.assertEqual(1, result.unchanged_entries)
            self.assertEqual(1, result.new_entries)
            self.assertEqual(1, result.changed_originals)
            self.assertEqual(1, result.removed_entries)
            self.assertEqual(2, result.translation_candidates)

    def test_rejects_duplicate_identity_without_writing_work_file(self) -> None:
        old = _catalog([_entry("0900000000000000", "old", "Old", "Cũ")])
        duplicated = _catalog(
            [
                _entry("0100000000000000", "same", "A", "A"),
                _entry("0100000000000000", "same", "B", "B"),
            ]
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            new_path.write_text(json.dumps(duplicated), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"Duplicate marker \+ key"):
                prepare_translation_work(old_path, new_path, work_path)

            self.assertFalse(work_path.exists())

    def test_existing_work_file_is_backed_up_before_atomic_replacement(self) -> None:
        catalog = _catalog([_entry("0100000000000000", "a", "A", "A")])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            old_path.write_text(json.dumps(catalog), encoding="utf-8")
            new_path.write_text(json.dumps(catalog), encoding="utf-8")
            work_path.write_text("previous", encoding="utf-8")

            result = prepare_translation_work(old_path, new_path, work_path)

            self.assertIsNotNone(result.backup_json)
            assert result.backup_json is not None
            self.assertEqual("previous", result.backup_json.read_text(encoding="utf-8"))

    def test_new_token_only_rows_are_excluded_from_the_translation_delta(
        self,
    ) -> None:
        old = _catalog([_entry("0100000000000000", "old", "Old", "Cũ")])
        new = _catalog(
            [
                _entry("0200000000000000", "empty", "[EMPTY]", "[EMPTY]"),
                _entry("0300000000000000", "guard", "{Key:Guard}", "{Key:Guard}"),
            ]
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            new_path.write_text(json.dumps(new), encoding="utf-8")

            result = prepare_translation_work(old_path, new_path, work_path)
            work = json.loads(work_path.read_text(encoding="utf-8"))

        self.assertEqual([], work["entries"])
        self.assertEqual(0, result.translation_candidates)
        self.assertEqual(2, result.technical_entries)
        self.assertEqual(0, result.delta_entries)

    def test_existing_untranslated_row_is_included_in_candidate_count(self) -> None:
        old = _catalog([_entry("0100000000000000", "same", "Hello", "Hello")])
        new = _catalog([_entry("0100000000000000", "same", "Hello", "Hello")])
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            new_path.write_text(json.dumps(new), encoding="utf-8")

            result = prepare_translation_work(old_path, new_path, work_path)

        self.assertEqual(0, result.translation_candidates)

    def test_rejects_different_package_folders(self) -> None:
        old = _catalog([_entry("0100000000000000", "same", "Hello", "Xin chào")])
        new = _catalog([_entry("0100000000000000", "same", "Hello", "Hello")])
        new["metadata"]["package_folder"] = "0030"
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_path = root / "old.json"
            new_path = root / "new.json"
            work_path = root / "work.json"
            old_path.write_text(json.dumps(old), encoding="utf-8")
            new_path.write_text(json.dumps(new), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "package_folder"):
                prepare_translation_work(old_path, new_path, work_path)

            self.assertFalse(work_path.exists())


if __name__ == "__main__":
    unittest.main()
