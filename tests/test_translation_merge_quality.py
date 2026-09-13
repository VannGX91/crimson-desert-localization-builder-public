import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.translation_merge import (
    fix_translation_delta_structure,
    merge_translation_delta,
    validate_translation_delta,
)
from crimson0020.translation_work import prepare_translation_work


def _entry(marker: str, key: str, original: str, translation: str | None = None):
    return {
        "marker": marker,
        "key": key,
        "original": original,
        "translation": original if translation is None else translation,
    }


def _catalog(entries, *, metadata=None):
    return {
        "metadata": metadata or {"package_folder": "0020"},
        "entries": entries,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(
    path: Path,
    template_data,
    translations,
    *,
    source_fingerprint=None,
    source_language="en",
    target_language="vi",
) -> None:
    header = {
        "record": "header",
        "format_version": 2,
        "source_fingerprint": (
            source_fingerprint
            if source_fingerprint is not None
            else template_data["source"]["catalog_fingerprint"]
        ),
        "provider": "synthetic",
        "source_language": source_language,
        "target_language": target_language,
    }
    records = []
    for entry, translation in translations:
        records.append(
            {
                "marker": entry["marker"],
                "key": entry["key"],
                "original_sha256": hashlib.sha256(
                    entry["original"].encode("utf-8")
                ).hexdigest().upper(),
                "translation": translation,
            }
        )
    batch = {"record": "batch", "translations": records}
    path.write_text(
        "\n".join(json.dumps(value, ensure_ascii=False) for value in (header, batch))
        + "\n",
        encoding="utf-8",
    )


class TranslationMergeQualityTests(unittest.TestCase):
    def _case(self, root: Path):
        old = root / "old.json"
        new = root / "new.json"
        template = root / "delta.json"
        translated = root / "translated.json"
        final = root / "final.json"
        _write_json(old, _catalog([]))
        _write_json(
            new,
            _catalog(
                [
                    _entry("01" * 8, "alpha", "Hello {name}"),
                    _entry("02" * 8, "beta", "Second line"),
                ],
                metadata={
                    "package_folder": "0020",
                    "format_version": 9,
                    "nested": {"keep": [1, 2, 3]},
                },
            ),
        )
        prepare_translation_work(old, new, template)
        translated_data = _load(template)
        translated_data["entries"][0]["translation"] = "Xin chào {name}"
        translated_data["entries"][0]["translation_state"] = "translated"
        translated_data["entries"][1]["translation"] = "Dòng thứ hai"
        translated_data["entries"][1]["translation_state"] = "translated"
        _write_json(translated, translated_data)
        return old, new, template, translated, final

    def test_every_step3_blocker_prevents_merge(self):
        mutations = {
            "invalid_json": lambda data: "{broken",
            "invalid_schema": lambda data: {**data, "schema_version": 999},
            "missing_identity": lambda data: {
                **data,
                "entries": data["entries"][:-1],
            },
            "duplicate_identity": lambda data: {
                **data,
                "entries": [data["entries"][0], copy.deepcopy(data["entries"][0])],
            },
            "source_fingerprint": lambda data: {
                **data,
                "source": {**data["source"], "catalog_fingerprint": "WRONG"},
            },
            "empty_translation": lambda data: self._mutate_entry(
                data, 0, translation="", translation_state="translated"
            ),
            "missing_translation": lambda data: self._drop_translation(data, 0),
            "protected_content": lambda data: self._mutate_entry(
                data,
                0,
                translation="Xin chào",
                translation_state="translated",
            ),
            "metadata_mismatch": lambda data: {
                **data,
                "source": {**data["source"], "entry_count": 999},
            },
            "order_mismatch": lambda data: {
                **data,
                "entries": list(reversed(data["entries"])),
            },
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old, new, template, translated, final = self._case(root)
                data = _load(translated)
                mutated = mutate(data)
                if isinstance(mutated, str):
                    translated.write_text(mutated, encoding="utf-8")
                else:
                    _write_json(translated, mutated)

                validation = validate_translation_delta(new, template, translated)
                with self.assertRaises(ValueError):
                    merge_translation_delta(old, new, template, translated, final)

                self.assertFalse(validation.complete, label)
                self.assertFalse(final.exists(), label)

    @staticmethod
    def _mutate_entry(data, index, **changes):
        result = copy.deepcopy(data)
        result["entries"][index].update(changes)
        return result

    @staticmethod
    def _drop_translation(data, index):
        result = copy.deepcopy(data)
        result["entries"][index].pop("translation", None)
        return result

    def test_multiple_simultaneous_defects_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            damaged = _load(translated)
            damaged["source"]["entry_count"] = 999
            damaged["entries"].reverse()
            damaged["entries"][0]["original"] = "Mutated source"
            damaged["entries"][1]["translation"] = "Mất placeholder"
            _write_json(translated, damaged)

            validation = validate_translation_delta(new, template, translated)

        joined = "\n".join(validation.issues).lower()
        self.assertFalse(validation.valid)
        self.assertIn("metadata", joined)
        self.assertIn("order", joined)
        self.assertIn("immutable", joined)
        self.assertIn("protected", joined)

    def test_repair_salvages_good_rows_when_one_ai_row_has_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            damaged = _load(translated)
            damaged["entries"][0]["translation_state"] = "router_garbage"
            _write_json(translated, damaged)
            repaired = root / "repaired.json"

            validation = validate_translation_delta(new, template, translated)
            result = fix_translation_delta_structure(
                template, translated, None, repaired, "en", "vi"
            )
            output = _load(repaired)
            post = validate_translation_delta(new, template, repaired)

        self.assertFalse(validation.valid)
        self.assertTrue(validation.repairable)
        self.assertEqual(2, result.repaired_entries)
        self.assertEqual(0, result.pending_entries)
        self.assertEqual("Xin chào {name}", output["entries"][0]["translation"])
        self.assertEqual("translated", output["entries"][0]["translation_state"])
        self.assertEqual("Dòng thứ hai", output["entries"][1]["translation"])
        self.assertTrue(post.valid)
        self.assertTrue(post.complete)
        self.assertEqual(0, post.pending_entries)

    def test_repair_restores_structure_and_salvages_safe_text_from_damaged_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            trusted = _load(template)
            damaged = _load(translated)
            damaged["schema_version"] = 999
            damaged["source"]["entry_count"] = 999
            damaged["entries"].reverse()
            first = damaged["entries"][1]
            first["change"] = "original_changed"
            first["translation_state"] = "router_garbage"
            damaged["entries"].append(copy.deepcopy(first))
            _write_json(translated, damaged)
            repaired = root / "repaired.json"

            result = fix_translation_delta_structure(
                template, translated, None, repaired, "en", "vi"
            )
            output = _load(repaired)
            post = validate_translation_delta(new, template, repaired)

        self.assertEqual(2, result.repaired_entries)
        self.assertEqual(0, result.pending_entries)
        self.assertTrue(post.valid)
        self.assertTrue(post.complete)
        self.assertEqual(trusted["source"], output["source"])
        self.assertEqual(
            [(entry["marker"], entry["key"]) for entry in trusted["entries"]],
            [(entry["marker"], entry["key"]) for entry in output["entries"]],
        )
        self.assertEqual("Xin chào {name}", output["entries"][0]["translation"])
        joined = "\n".join(result.issues).lower()
        self.assertIn("immutable metadata", joined)
        self.assertIn("duplicate", joined)

    def test_repair_does_not_salvage_text_when_source_or_context_is_untrusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            damaged = _load(translated)
            damaged["entries"][0]["original"] = "Different source {name}"
            damaged["entries"][0]["context"] = ["different context"]
            damaged["entries"][0]["context_signature"] = "different-signature"
            _write_json(translated, damaged)
            repaired = root / "repaired.json"

            result = fix_translation_delta_structure(
                template, translated, None, repaired, "en", "vi"
            )
            output = _load(repaired)
            post = validate_translation_delta(new, template, repaired)

        self.assertEqual(1, result.repaired_entries)
        self.assertEqual(1, result.pending_entries)
        self.assertEqual("", output["entries"][0]["translation"])
        self.assertEqual("pending", output["entries"][0]["translation_state"])
        self.assertTrue(post.valid)
        self.assertFalse(post.complete)
        self.assertEqual(1, post.pending_entries)

    def test_repair_leaves_conflicting_duplicate_translations_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            damaged = _load(translated)
            duplicate = copy.deepcopy(damaged["entries"][0])
            duplicate["translation"] = "Chào {name}"
            damaged["entries"].append(duplicate)
            _write_json(translated, damaged)
            repaired = root / "repaired.json"

            result = fix_translation_delta_structure(
                template, translated, None, repaired, "en", "vi"
            )
            output = _load(repaired)
            post = validate_translation_delta(new, template, repaired)

        self.assertEqual(1, result.repaired_entries)
        self.assertEqual(1, result.pending_entries)
        self.assertEqual("", output["entries"][0]["translation"])
        self.assertEqual("pending", output["entries"][0]["translation_state"])
        self.assertTrue(post.valid)
        self.assertFalse(post.complete)
        self.assertEqual(1, post.pending_entries)
        self.assertTrue(any("conflicting valid translations" in issue for issue in result.issues))

    def test_repair_does_not_promote_glossary_violation_to_valid_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            damaged = _load(translated)
            damaged["entries"][0]["quality_flags"] = ["glossary_violation"]
            _write_json(translated, damaged)
            repaired = root / "repaired.json"

            result = fix_translation_delta_structure(
                template, translated, None, repaired, "en", "vi"
            )
            output = _load(repaired)
            post = validate_translation_delta(new, template, repaired)

        self.assertEqual(1, result.repaired_entries)
        self.assertEqual(1, result.pending_entries)
        self.assertEqual("", output["entries"][0]["translation"])
        self.assertEqual("pending", output["entries"][0]["translation_state"])
        self.assertTrue(post.valid)
        self.assertFalse(post.complete)

    def test_invalid_trusted_template_is_not_offered_as_repairable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, new, template, translated, _ = self._case(root)
            template_data = _load(template)
            template_data["schema_version"] = 999
            _write_json(template, template_data)

            validation = validate_translation_delta(new, template, translated)
            with self.assertRaisesRegex(ValueError, "Trusted translation template is invalid"):
                fix_translation_delta_structure(
                    template, translated, None, root / "repaired.json"
                )

        self.assertFalse(validation.valid)
        self.assertFalse(validation.repairable)
        self.assertTrue(any("trusted translation template" in issue.lower() for issue in validation.issues))

    def test_soft_glossary_warning_is_quality_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, template, translated, final = self._case(root)
            data = _load(translated)
            data["entries"][0]["quality_flags"] = ["soft_glossary_warning"]
            _write_json(translated, data)

            validation = validate_translation_delta(new, template, translated)
            merge_translation_delta(old, new, template, translated, final)

        self.assertTrue(validation.valid)
        self.assertTrue(validation.complete)

    def test_hard_glossary_violation_remains_pending_even_with_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, template, translated, final = self._case(root)
            data = _load(translated)
            data["entries"][0]["quality_flags"] = ["glossary_violation"]
            _write_json(translated, data)

            validation = validate_translation_delta(new, template, translated)
            with self.assertRaises(ValueError):
                merge_translation_delta(old, new, template, translated, final)

        self.assertTrue(validation.valid)
        self.assertFalse(validation.complete)
        self.assertEqual(1, validation.pending_entries)

    def test_repair_rejects_contextless_checkpoint_for_contextful_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, template, translated, _ = self._case(root)
            checkpoint = root / "checkpoint.jsonl"
            repaired = root / "repaired.json"
            template_data = _load(template)
            translated.write_text("damaged", encoding="utf-8")
            _write_checkpoint(
                checkpoint,
                template_data,
                [(template_data["entries"][0], "Checkpoint {name}")],
            )

            result = fix_translation_delta_structure(
                template, translated, checkpoint, repaired, "en", "vi"
            )
            output = _load(repaired)

        self.assertEqual(0, result.repaired_entries)
        self.assertEqual(2, result.pending_entries)
        self.assertTrue(all(entry["translation"] == "" for entry in output["entries"]))
        self.assertTrue(
            all(entry["translation_state"] == "pending" for entry in output["entries"])
        )

    def test_repair_rejects_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, template, translated, _ = self._case(root)
            checkpoint = root / "checkpoint.jsonl"
            repaired = root / "repaired.json"
            template_data = _load(template)
            translated.write_text("damaged", encoding="utf-8")
            _write_checkpoint(
                checkpoint,
                template_data,
                [(template_data["entries"][0], "Checkpoint {name}")],
                source_fingerprint="WRONG",
            )

            result = fix_translation_delta_structure(
                template, translated, checkpoint, repaired, "en", "vi"
            )
            output = _load(repaired)

        self.assertEqual(0, result.repaired_entries)
        self.assertEqual(2, result.pending_entries)
        self.assertTrue(any("checkpoint" in issue.lower() for issue in result.issues))
        self.assertTrue(all(entry["translation"] == "" for entry in output["entries"]))

    def test_repair_and_merge_reject_existing_output_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, template, translated, final = self._case(root)
            repaired = root / "repaired.json"
            repaired.write_text("KEEP REPAIR", encoding="utf-8")
            final.write_text("KEEP MERGE", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                fix_translation_delta_structure(template, translated, None, repaired)
            with self.assertRaises(FileExistsError):
                merge_translation_delta(old, new, template, translated, final)

            self.assertEqual("KEEP REPAIR", repaired.read_text(encoding="utf-8"))
            self.assertEqual("KEEP MERGE", final.read_text(encoding="utf-8"))

    def test_atomic_write_failure_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, template, translated, final = self._case(root)
            repaired = root / "repaired.json"

            with patch(
                "crimson0020.translation_merge.os.replace",
                side_effect=OSError("synthetic replace failure"),
            ):
                with self.assertRaises(OSError):
                    fix_translation_delta_structure(template, translated, None, repaired)
            self.assertFalse(repaired.exists())
            self.assertEqual([], list(root.glob(".*repaired.json.*.tmp")))

            with patch(
                "crimson0020.translation_merge.os.replace",
                side_effect=OSError("synthetic replace failure"),
            ):
                with self.assertRaises(OSError):
                    merge_translation_delta(old, new, template, translated, final)
            self.assertFalse(final.exists())
            self.assertEqual([], list(root.glob(".*final.json.*.tmp")))

    def test_merge_preserves_current_catalog_order_metadata_and_all_nontranslation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new, template, translated, final = self._case(root)
            source = _load(new)
            source["entries"][0]["routing"] = {"voice": "npc", "flags": [1, 3]}
            source["entries"][1]["unknown_future_field"] = "preserve-me"
            _write_json(new, source)
            prepare_translation_work(old, new, template)
            data = _load(template)
            data["entries"][0]["translation"] = "Xin chào {name}"
            data["entries"][0]["translation_state"] = "translated"
            data["entries"][1]["translation"] = "Dòng thứ hai"
            data["entries"][1]["translation_state"] = "translated"
            _write_json(translated, data)
            expected = copy.deepcopy(source)
            expected["entries"][0]["translation"] = "Xin chào {name}"
            expected["entries"][1]["translation"] = "Dòng thứ hai"

            merge_translation_delta(old, new, template, translated, final)
            output = _load(final)

        self.assertEqual(expected, output)


if __name__ == "__main__":
    unittest.main()
