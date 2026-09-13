import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from crimson0020.auto_translate import translate_catalog
from crimson0020.translation_merge import (
    fix_translation_delta_structure,
    merge_translation_delta,
    validate_translation_delta,
)
from crimson0020.translation_work import prepare_translation_work


def _catalog(entries):
    return {"metadata": {"package_folder": "0020"}, "entries": entries}


def _entry(marker, key, original, translation):
    return {
        "marker": marker,
        "key": key,
        "original": original,
        "translation": translation,
    }


class _Provider:
    key = "test"

    def translate(self, texts, source, target):
        return tuple(f"VI:{text}" for text in texts)


class TranslationMergeTests(unittest.TestCase):
    def test_sparse_delta_validates_and_merges_over_new_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            final = root / "final.json"
            old.write_text(
                json.dumps(
                    _catalog([_entry("01" * 8, "same", "Same", "Cũ")]),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            new.write_text(
                json.dumps(
                    _catalog(
                        [
                            _entry("01" * 8, "same", "Same", "Same"),
                            _entry("02" * 8, "new", "Brand", "Brand"),
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            translate_catalog(template, translated, _Provider(), "en", "vi")

            validation = validate_translation_delta(new, template, translated)
            result = merge_translation_delta(old, new, template, translated, final)
            data = json.loads(final.read_text(encoding="utf-8"))

        self.assertTrue(validation.valid)
        self.assertTrue(validation.complete)
        self.assertEqual(2, result.merged_entries)
        self.assertEqual("Cũ", data["entries"][0]["translation"])
        self.assertEqual("VI:Brand", data["entries"][1]["translation"])
        self.assertEqual(1, result.reused_unchanged_entries)
        self.assertEqual(1, result.merged_delta_entries)

    def test_changed_old_translation_is_never_carried_and_removed_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            final = root / "final.json"
            old.write_text(
                json.dumps(
                    _catalog(
                        [
                            _entry("01" * 8, "changed", "Old source", "Old VI"),
                            _entry("02" * 8, "removed", "Gone", "Gone VI"),
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            new.write_text(
                json.dumps(
                    _catalog(
                        [
                            _entry("01" * 8, "changed", "New source", "New source"),
                            _entry("03" * 8, "technical", "{Guard}", "{Guard}"),
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            translate_catalog(template, translated, _Provider(), "en", "vi")
            result = merge_translation_delta(old, new, template, translated, final)
            data = json.loads(final.read_text(encoding="utf-8"))

        self.assertEqual(
            ["VI:New source", "{Guard}"],
            [entry["translation"] for entry in data["entries"]],
        )
        self.assertEqual(1, result.merged_delta_entries)
        self.assertEqual(1, result.source_fallback_entries)

    def test_structurally_valid_pending_delta_is_not_mergeable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            old.write_text(json.dumps(_catalog([])), encoding="utf-8")
            new.write_text(
                json.dumps(
                    _catalog([_entry("01" * 8, "new", "Brand", "Brand")]),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            translated.write_text(
                template.read_text(encoding="utf-8"), encoding="utf-8"
            )

            validation = validate_translation_delta(new, template, translated)
            with self.assertRaises(ValueError):
                merge_translation_delta(
                    old, new, template, translated, root / "final.json"
                )

        self.assertTrue(validation.valid)
        self.assertFalse(validation.complete)
        self.assertGreater(validation.pending_entries, 0)

    def test_fix_reconstructs_missing_rows_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            repaired = root / "repaired.json"
            old.write_text(json.dumps(_catalog([])), encoding="utf-8")
            new.write_text(
                json.dumps(
                    _catalog(
                        [
                            _entry("01" * 8, "a", "A", "A"),
                            _entry("02" * 8, "b", "B", "B"),
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            damaged = json.loads(template.read_text(encoding="utf-8"))
            damaged["entries"] = [damaged["entries"][0]]
            damaged["entries"][0]["translation"] = "VI:A"
            damaged["entries"][0]["translation_state"] = "translated"
            translated.write_text(json.dumps(damaged), encoding="utf-8")
            result = fix_translation_delta_structure(
                template, translated, None, repaired
            )
            data = json.loads(repaired.read_text(encoding="utf-8"))

        self.assertEqual(2, result.entry_count)
        self.assertEqual(1, result.repaired_entries)
        self.assertEqual(1, result.pending_entries)
        self.assertEqual(2, len(data["entries"]))
        self.assertEqual("", data["entries"][1]["translation"])

    def test_merge_rejects_a_different_old_catalog_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            wrong_old = root / "wrong-old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            final = root / "final.json"
            old.write_text(
                json.dumps(_catalog([_entry("01" * 8, "same", "Same", "Cũ")])),
                encoding="utf-8",
            )
            wrong_old.write_text(
                json.dumps(_catalog([_entry("01" * 8, "same", "Different", "Sai")])),
                encoding="utf-8",
            )
            new.write_text(
                json.dumps(_catalog([_entry("01" * 8, "same", "Same", "Same")])),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            translated.write_bytes(template.read_bytes())
            before = tuple(
                path.read_bytes() for path in (wrong_old, new, template, translated)
            )

            with self.assertRaisesRegex(ValueError, "old translated catalog"):
                merge_translation_delta(wrong_old, new, template, translated, final)

            self.assertFalse(final.exists())
            self.assertEqual(
                before,
                tuple(
                    path.read_bytes() for path in (wrong_old, new, template, translated)
                ),
            )

    def test_fix_and_merge_never_overwrite_their_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            old.write_text(json.dumps(_catalog([])), encoding="utf-8")
            new.write_text(
                json.dumps(_catalog([_entry("01" * 8, "new", "New", "New")])),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            translate_catalog(template, translated, _Provider(), "en", "vi")

            with self.assertRaisesRegex(ValueError, "separate"):
                fix_translation_delta_structure(template, translated, None, template)
            with self.assertRaisesRegex(ValueError, "separate"):
                merge_translation_delta(old, new, template, translated, new)

    def test_invalid_old_translation_contract_falls_back_to_new_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            translated = root / "translated.json"
            final = root / "final.json"
            old.write_text(
                json.dumps(
                    _catalog([_entry("01" * 8, "same", "Hello {name}", "Xin chào")]),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            new.write_text(
                json.dumps(
                    _catalog(
                        [
                            _entry(
                                "01" * 8,
                                "same",
                                "Hello {name}",
                                "Hello {name}",
                            )
                        ]
                    )
                ),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            translated.write_bytes(template.read_bytes())

            result = merge_translation_delta(old, new, template, translated, final)
            output = json.loads(final.read_text(encoding="utf-8"))

        self.assertEqual("Hello {name}", output["entries"][0]["translation"])
        self.assertEqual(0, result.reused_unchanged_entries)
        self.assertEqual(1, result.source_fallback_entries)

    def test_fix_rejects_checkpoint_from_the_wrong_language_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.json"
            new = root / "new.json"
            template = root / "delta.json"
            damaged = root / "damaged.json"
            checkpoint = root / "checkpoint.jsonl"
            repaired = root / "repaired.json"
            old.write_text(json.dumps(_catalog([])), encoding="utf-8")
            new.write_text(
                json.dumps(_catalog([_entry("01" * 8, "new", "Hello", "Hello")])),
                encoding="utf-8",
            )
            prepare_translation_work(old, new, template)
            template_data = json.loads(template.read_text(encoding="utf-8"))
            damaged.write_text("not json", encoding="utf-8")
            header = {
                "record": "header",
                "format_version": 2,
                "source_fingerprint": template_data["source"]["catalog_fingerprint"],
                "provider": "test",
                "source_language": "ja",
                "target_language": "fr",
            }
            batch = {
                "record": "batch",
                "translations": [
                    {
                        "marker": "01" * 8,
                        "key": "new",
                        "original_sha256": hashlib.sha256(b"Hello").hexdigest().upper(),
                        "translation": "Bonjour",
                    }
                ],
            }
            checkpoint.write_text(
                "\n".join(json.dumps(value) for value in (header, batch)) + "\n",
                encoding="utf-8",
            )

            result = fix_translation_delta_structure(
                template,
                damaged,
                checkpoint,
                repaired,
                "en",
                "vi",
            )
            output = json.loads(repaired.read_text(encoding="utf-8"))

        self.assertEqual(1, result.pending_entries)
        self.assertEqual("", output["entries"][0]["translation"])
        self.assertTrue(any("language" in issue.lower() for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
