"""Regression tests for Auto Translate release blockers and robustness fixes."""

import json
import tempfile
import unittest
from pathlib import Path

from crimson0020.auto_translate import (
    TranslationContractError,
    TranslationLockError,
    _acquire_process_lock,
    translate_delta,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.terminology import TerminologyEngine
from crimson0020.translation_work import (
    prepare_translation_work,
    read_translation_delta,
)


def _make_delta(path: Path, entries: list[dict[str, object]]) -> Path:
    data = {
        "schema": "crimson.translation-delta",
        "schema_version": 1,
        "kind": "sparse-delta",
        "source": {
            "catalog_fingerprint": "A" * 64,
            "base_catalog_fingerprint": "B" * 64,
            "entry_count": len(entries),
        },
        "entries": entries,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class EchoProvider:
    """Mock provider that translates text while preserving protected tags."""

    def __init__(self, key: str = "mock-echo"):
        self.key = key
        self.call_count = 0
        self.received_batches: list[tuple[str, ...]] = []

    def translate(
        self, texts: tuple[str, ...] | list[str], source_lang: str, target_lang: str
    ) -> tuple[str, ...]:
        self.call_count += 1
        self.received_batches.append(tuple(texts))
        results = []
        for t in texts:
            # Replace English word 'Equipment' with 'Trang bị', 'ready' with 'sẵn sàng'
            res = t.replace("Equipment", "Trang bị").replace("ready", "sẵn sàng")
            results.append(res)
        return tuple(results)


class BadTypeProvider:
    def __init__(self, bad_return: object, key: str = "mock-bad"):
        self.bad_return = bad_return
        self.key = key

    def translate(self, texts, src, tgt):
        return self.bad_return


class ContextAwareProvider:
    key = "mock-context-aware"

    def __init__(self):
        self.items = []
        self.call_count = 0

    def translate_items(self, items, source_lang, target_lang):
        self.call_count += 1
        self.items.extend(items)
        return tuple(
            "Kliff đã sẵn sàng."
            if item.context and "Kliff" in item.context[0]
            else "Darlene đã sẵn sàng."
            for item in items
        )

    def translate(self, texts, source_lang, target_lang):
        raise AssertionError("Context-aware provider must receive rich translation items.")


class TestAutoTranslateRegressionBlockers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _init_db(self, db_path: Path, scope_id: str = "Crimson0020_0020_en_vi") -> KnowledgeDatabase:
        db = KnowledgeDatabase(db_path)
        with db.transaction() as con:
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(scope_id) DO NOTHING",
                (scope_id, "Crimson0020", "0020", "en", "vi", 1000.0),
            )
        return db

    def test_placeholder_not_corrupted_by_hard_glossary(self):
        """Bug 1: Technical placeholders like {Money:1} must not be altered by glossary."""
        db_path = self.work_dir / "knowledge.sqlite3"
        db = self._init_db(db_path)
        engine = TerminologyEngine(db)
        engine.set_user_override("Crimson0020_0020_en_vi", "Money", "Tiền")

        delta_path = _make_delta(
            self.work_dir / "delta.json",
            [
                {
                    "marker": "0100000000000000",
                    "key": "msg_reward",
                    "original": "Reward: {Money:Money_Copper:1} Money",
                    "translation": "",
                    "change": "new",
                    "translation_state": "pending",
                }
            ],
        )
        output_path = self.work_dir / "translated.json"
        provider = EchoProvider()

        translate_delta(
            delta_path,
            output_path,
            provider,
            "en",
            "vi",
            db_path=db_path,
        )

        result_data = json.loads(output_path.read_text(encoding="utf-8"))
        entry = result_data["entries"][0]
        # The placeholder {Money:Money_Copper:1} must remain intact, and natural Money -> Tiền
        self.assertIn("{Money:Money_Copper:1}", entry["translation"])
        self.assertNotIn("{Tiền:Money_Copper:1}", entry["translation"])
        self.assertEqual(entry["translation"], "Reward: {Money:Money_Copper:1} Tiền")

    def test_glossary_token_isolation_no_namespace_collision(self):
        """Bug 2: Tag <x0> in source must not collide with glossary token namespace."""
        db_path = self.work_dir / "knowledge.sqlite3"
        db = self._init_db(db_path)
        engine = TerminologyEngine(db)
        engine.set_user_override("Crimson0020_0020_en_vi", "Abyss", "Vực")

        delta_path = _make_delta(
            self.work_dir / "delta.json",
            [
                {
                    "marker": "0200000000000000",
                    "key": "item_tag",
                    "original": "<x0> Abyss Equipment",
                    "translation": "",
                    "change": "new",
                    "translation_state": "pending",
                }
            ],
        )
        output_path = self.work_dir / "translated.json"
        provider = EchoProvider()

        res1 = translate_delta(
            delta_path,
            output_path,
            provider,
            "en",
            "vi",
            db_path=db_path,
        )
        self.assertEqual(res1.translated_entries, 1)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(data["entries"][0]["translation"], "<x0> Vực Trang bị")

        # Resume run must load checkpoint cleanly and not raise TranslationCheckpointError
        res2 = translate_delta(
            delta_path,
            output_path,
            provider,
            "en",
            "vi",
            db_path=db_path,
        )
        self.assertEqual(res2.resumed_entries, 1)
        self.assertEqual(res2.translated_entries, 0)

    def test_context_aware_deduplication(self):
        """Bug 3: Same string with different context must not be collapsed into 1 request."""
        delta_path = _make_delta(
            self.work_dir / "delta.json",
            [
                {
                    "marker": "0300000000000001",
                    "key": "kliff_dialog",
                    "original": "I am ready.",
                    "context": ["speaker: Kliff"],
                    "context_signature": "char:kliff",
                    "change": "new",
                    "translation_state": "pending",
                },
                {
                    "marker": "0300000000000002",
                    "key": "darlene_dialog",
                    "original": "I am ready.",
                    "context": ["speaker: Darlene"],
                    "context_signature": "char:darlene",
                    "change": "new",
                    "translation_state": "pending",
                },
                {
                    "marker": "0300000000000003",
                    "key": "kliff_dialog_repeat",
                    "original": "I am ready.",
                    "context": ["speaker: Kliff"],
                    "context_signature": "char:kliff",
                    "change": "new",
                    "translation_state": "pending",
                },
            ],
        )
        output_path = self.work_dir / "translated.json"
        provider = EchoProvider()

        res = translate_delta(
            delta_path,
            output_path,
            provider,
            "en",
            "vi",
            max_batch_items=1,  # Force 1 item per batch to verify exact request count
        )

        # Kliff and Darlene have different context: 2 provider requests.
        # kliff_dialog_repeat shares context with kliff_dialog: 1 reused duplicate.
        self.assertEqual(res.reused_duplicate_entries, 1)
        self.assertEqual(provider.call_count, 2)

    def test_step1_preserves_context_and_step2_passes_it_to_context_aware_provider(self):
        old_path = self.work_dir / "old.json"
        new_path = self.work_dir / "new.json"
        work_path = self.work_dir / "work.json"
        output_path = self.work_dir / "translated.json"
        old_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
        new_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "marker": "0310000000000001",
                            "key": "kliff_dialog",
                            "original": "I am ready.",
                            "translation": "I am ready.",
                            "context": ["speaker: Kliff"],
                            "context_signature": "char:kliff",
                        },
                        {
                            "marker": "0310000000000002",
                            "key": "darlene_dialog",
                            "original": "I am ready.",
                            "translation": "I am ready.",
                            "context": ["speaker: Darlene"],
                            "context_signature": "char:darlene",
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        prepare_translation_work(old_path, new_path, work_path)
        delta = json.loads(work_path.read_text(encoding="utf-8"))
        self.assertEqual(["speaker: Kliff"], delta["entries"][0]["context"])
        self.assertEqual("char:kliff", delta["entries"][0]["context_signature"])
        self.assertEqual(["speaker: Darlene"], delta["entries"][1]["context"])

        provider = ContextAwareProvider()
        result = translate_delta(
            work_path,
            output_path,
            provider,
            "en",
            "vi",
            max_batch_items=1,
        )
        translated = json.loads(output_path.read_text(encoding="utf-8"))["entries"]

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(2, provider.call_count)
        self.assertEqual(
            ["char:kliff", "char:darlene"],
            [item.context_signature for item in provider.items],
        )
        self.assertEqual("Kliff đã sẵn sàng.", translated[0]["translation"])
        self.assertEqual("Darlene đã sẵn sàng.", translated[1]["translation"])

    def test_step1_derives_distinct_neighbor_context_for_same_source(self):
        old_path = self.work_dir / "old.json"
        new_path = self.work_dir / "new.json"
        work_path = self.work_dir / "work.json"
        old_path.write_text(json.dumps({"entries": []}), encoding="utf-8")
        entries = [
            {"marker": f"032000000000000{i}", "key": f"k{i}", "original": text, "translation": text}
            for i, text in enumerate(
                (
                    "Kliff enters the yard.",
                    "I am ready.",
                    "The torch flickers.",
                    "A bell rings.",
                    "I am ready.",
                    "Darlene leaves the hall.",
                )
            )
        ]
        new_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "palocs": [{"path": "synthetic/a.paloc", "entry_count": len(entries)}]
                    },
                    "entries": entries,
                }
            ),
            encoding="utf-8",
        )

        prepare_translation_work(old_path, new_path, work_path)
        delta = json.loads(work_path.read_text(encoding="utf-8"))
        ready_rows = [entry for entry in delta["entries"] if entry["original"] == "I am ready."]

        self.assertEqual(2, len(ready_rows))
        self.assertTrue(all(row["context"] for row in ready_rows))
        self.assertNotEqual(
            ready_rows[0]["context_signature"], ready_rows[1]["context_signature"]
        )

    def test_process_lock_mutual_exclusion(self):
        """Bug 4: Process-level file lock prevents concurrent translate_catalog runs."""
        delta_path = _make_delta(
            self.work_dir / "delta.json",
            [
                {
                    "marker": "0400000000000000",
                    "key": "k1",
                    "original": "Test lock string",
                    "change": "new",
                    "translation_state": "pending",
                }
            ],
        )
        output_path = self.work_dir / "translated.json"
        lock_file = output_path.with_name(f"{output_path.name}.autotranslate.lock")

        # Manually hold the process lock
        with _acquire_process_lock(lock_file):
            provider = EchoProvider()
            with self.assertRaises(TranslationLockError):
                translate_delta(
                    delta_path,
                    output_path,
                    provider,
                    "en",
                    "vi",
                )

    def test_common_words_untranslated_not_auto_locked(self):
        """Bug 5: Words like Save, Load, Settings kept untranslated must not be auto-locked."""
        catalog_old = self.work_dir / "catalog_old.json"
        catalog_new = self.work_dir / "catalog_new.json"
        entries_old = [
            {"marker": f"050000000000000{i}", "key": f"ui_save_{i}", "original": "Save", "translation": "Save"}
            for i in range(3)
        ] + [
            {"marker": "0500000000000010", "key": "ui_load", "original": "Load", "translation": "Load"},
            {"marker": "0500000000000011", "key": "ui_load2", "original": "Load", "translation": "Load"},
            # A genuine proper noun should still be auto-locked
            {"marker": "0500000000000020", "key": "hero_1", "original": "Kliff", "translation": "Kliff"},
            {"marker": "0500000000000021", "key": "hero_2", "original": "Kliff", "translation": "Kliff"},
            # Normal translated row to provide a legacy segment
            {"marker": "0500000000000030", "key": "item_sword", "original": "Iron Sword", "translation": "Kiếm Sắt"},
        ]
        entries_new = entries_old + [
            {"marker": "0500000000000099", "key": "new_btn", "original": "Save and Exit"}
        ]
        for p, ents in ((catalog_old, entries_old), (catalog_new, entries_new)):
            p.write_text(json.dumps({"entries": ents}, ensure_ascii=False), encoding="utf-8")

        work_json = self.work_dir / "work.json"
        db_path = self.work_dir / "knowledge.sqlite3"
        db = self._init_db(db_path)
        prepare_translation_work(
            catalog_old,
            catalog_new,
            work_json,
            knowledge_db=db,
        )
        with db.connect() as con:
            cur = con.cursor()
            terms = {
                row[0]: (row[1], row[2])
                for row in cur.execute("SELECT source_term, target_term, state FROM terms").fetchall()
            }
        # "Save" and "Load" must NOT be auto_locked
        self.assertNotIn("Save", terms)
        self.assertNotIn("Load", terms)
        # "Kliff" (proper noun) SHOULD be auto_locked
        self.assertIn("Kliff", terms)
        self.assertEqual(terms["Kliff"][1], "auto_locked")

    def test_provider_strict_return_type_validation(self):
        """Bug 6: Provider returning bare string or dict must raise TranslationContractError."""
        from crimson0020.auto_translate import _translate_with_retry

        for bad_val in ("OK", {"text": "OK"}, ["valid", 123]):
            with self.subTest(bad_val=bad_val):
                provider = BadTypeProvider(bad_val)
                with self.assertRaises(TranslationContractError):
                    _translate_with_retry(
                        provider,
                        ("Hello", "World"),
                        "en",
                        "vi",
                        attempts=1,
                        base_delay=0.1,
                        sleeper=lambda _: None,
                    )

    def test_hard_glossary_token_loss_is_rescued_before_checkpoint(self):
        db_path = self.work_dir / "knowledge.sqlite3"
        db = self._init_db(db_path)
        TerminologyEngine(db).set_user_override(
            "Crimson0020_0020_en_vi", "Abyss", "Vực"
        )
        delta_path = _make_delta(
            self.work_dir / "delta.json",
            [
                {
                    "marker": "0610000000000001",
                    "key": "abyss",
                    "original": "Abyss Equipment",
                    "change": "new",
                    "translation_state": "pending",
                }
            ],
        )

        class DropsGlossaryToken(EchoProvider):
            def translate(self, texts, source_lang, target_lang):
                self.call_count += 1
                return tuple("Trang bị" for _ in texts)

        output_path = self.work_dir / "translated.json"
        result = translate_delta(
            delta_path,
            output_path,
            DropsGlossaryToken(),
            "en",
            "vi",
            db_path=db_path,
        )
        output = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, result.pending_entries)
        self.assertEqual(0, result.failure_count)
        self.assertEqual("translated", output["entries"][0]["translation_state"])
        self.assertIn("Vực", output["entries"][0]["translation"])
        self.assertNotIn("<_GT_", output["entries"][0]["translation"])
        checkpoint_text = result.checkpoint_json.read_text(encoding="utf-8")
        self.assertIn("Vực", checkpoint_text)

    def test_read_translation_delta_invalid_change_and_surrogates(self):
        """Malformed change type or lone surrogates must raise ValueError cleanly."""
        delta_path = self.work_dir / "bad_delta.json"
        _make_delta(
            delta_path,
            [
                {
                    "marker": "0700000000000001",
                    "key": "k1",
                    "original": "Test",
                    "change": ["unhashable", "list"],
                    "translation_state": "pending",
                }
            ],
        )
        with self.assertRaises(ValueError) as ctx:
            read_translation_delta(delta_path)
        self.assertIn("change is invalid", str(ctx.exception))

        # Test lone surrogates written via escaped JSON
        surrogate_json = json.dumps(
            {
                "schema": "crimson.translation-delta",
                "schema_version": 1,
                "kind": "sparse-delta",
                "source": {
                    "catalog_fingerprint": "A" * 64,
                    "base_catalog_fingerprint": "B" * 64,
                    "entry_count": 1,
                },
                "entries": [
                    {
                        "marker": "0700000000000002",
                        "key": "k2",
                        "original": "Invalid surrogate \ud800 string",
                        "change": "new",
                        "translation_state": "pending",
                    }
                ],
            },
            ensure_ascii=True,
        )
        delta_path.write_text(surrogate_json, encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            read_translation_delta(delta_path)
        self.assertIn("surrogates", str(ctx.exception))

    def test_step2_entrypoint_enforces_strict_delta_validation(self):
        output_path = self.work_dir / "translated.json"
        bad_change = _make_delta(
            self.work_dir / "bad-change.json",
            [
                {
                    "marker": "0710000000000001",
                    "key": "k1",
                    "original": "Test",
                    "change": ["bad"],
                    "translation_state": "pending",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "change is invalid"):
            translate_delta(bad_change, output_path, EchoProvider(), "en", "vi")

        bad_state = _make_delta(
            self.work_dir / "bad-state.json",
            [
                {
                    "marker": "0710000000000002",
                    "key": "k2",
                    "original": "Test",
                    "change": "new",
                    "translation_state": ["bad"],
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "translation state is invalid"):
            translate_delta(bad_state, output_path, EchoProvider(), "en", "vi")

    def test_step2_reuses_exact_tm_from_registered_lineage_without_provider_call(self):
        old_path = self.work_dir / "old.json"
        new_path = self.work_dir / "new.json"
        work_path = self.work_dir / "work.json"
        db_path = self.work_dir / "knowledge.sqlite3"
        db = KnowledgeDatabase(db_path)
        old_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "marker": "0720000000000001",
                            "key": "greeting",
                            "original": "Old greeting",
                            "translation": "Lời chào cũ",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        new_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "marker": "0720000000000001",
                            "key": "greeting",
                            "original": "Fresh greeting",
                            "translation": "Fresh greeting",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        prepared = prepare_translation_work(old_path, new_path, work_path, knowledge_db=db)
        self.assertIsNotNone(prepared.lineage_id)

        first_provider = EchoProvider()
        first_output = self.work_dir / "translated-first.json"
        translate_delta(
            work_path, first_output, first_provider, "en", "vi", db_path=db_path
        )
        self.assertEqual(1, first_provider.call_count)

        second_provider = EchoProvider()
        second_output = self.work_dir / "translated-second.json"
        second = translate_delta(
            work_path, second_output, second_provider, "en", "vi", db_path=db_path
        )
        first_text = json.loads(first_output.read_text(encoding="utf-8"))["entries"][0][
            "translation"
        ]
        second_text = json.loads(second_output.read_text(encoding="utf-8"))["entries"][0][
            "translation"
        ]

        self.assertEqual(0, second_provider.call_count)
        self.assertEqual(0, second.pending_entries)
        self.assertEqual(first_text, second_text)

    def test_repeated_fix_structure_generates_unique_output_path(self):
        """Fix structure should cleanly chain into unique filenames without colliding or nesting extensions."""
        import time
        from crimson0020.webview_bridge import WebviewBridge

        work_path = self.work_dir / "work.json"
        _make_delta(
            work_path,
            [
                {
                    "marker": "0800000000000001",
                    "key": "k1",
                    "original": "Test",
                    "change": "new",
                    "translation": "",
                    "translation_state": "pending",
                }
            ],
        )
        trans_path = self.work_dir / "trans.json"
        _make_delta(
            trans_path,
            [
                {
                    "marker": "0800000000000001",
                    "key": "k1",
                    "original": "Test",
                    "change": "new",
                    "translation": "",
                    "translation_state": "pending",
                }
            ],
        )

        bridge = WebviewBridge(base_directory=self.work_dir)
        bridge.set_fields({
            "auto_work": str(work_path),
            "auto_merge_input": str(trans_path),
            "auto_output": str(trans_path),
        })

        # Click 1
        res1 = bridge.start_operation("fix_translation_structure")
        self.assertTrue(res1["ok"])
        for _ in range(50):
            if not bridge.get_state()["busy"]:
                break
            time.sleep(0.05)
        self.assertEqual(bridge.get_state()["fields"]["auto_merge_input"], "trans.fixed.json")

        # Click 2
        res2 = bridge.start_operation("fix_translation_structure")
        self.assertTrue(res2["ok"])
        for _ in range(50):
            if not bridge.get_state()["busy"]:
                break
            time.sleep(0.05)
        self.assertEqual(bridge.get_state()["fields"]["auto_merge_input"], "trans.fixed-1.json")

        # Click 3
        res3 = bridge.start_operation("fix_translation_structure")
        self.assertTrue(res3["ok"])
        for _ in range(50):
            if not bridge.get_state()["busy"]:
                break
            time.sleep(0.05)
        self.assertEqual(bridge.get_state()["fields"]["auto_merge_input"], "trans.fixed-2.json")


if __name__ == "__main__":
    unittest.main()
