import json
import struct
import tempfile
import unittest
from pathlib import Path

from crimson0020.archive import TARGET_ENTRY_PATH, extract_archive_entry, parse_pamt
from crimson0020.checksums import pa_checksum
from crimson0020.crypto import compress_lz4, decrypt, decompress_lz4, encrypt
from crimson0020.paloc import json_data_to_paloc, paloc_to_json_data
from crimson0020.languages import get_language_spec
from crimson0020.service import (
    build_0020_from_json,
    build_localization_from_json,
    export_0020_to_json,
    export_localization_to_json,
)

from package_fixtures import build_localization_package, build_multi_paloc_package


def _pack_named_node(parent: int, name: str) -> bytes:
    encoded = name.encode("utf-8")
    return struct.pack("<IB", parent, len(encoded)) + encoded


def _build_synthetic_0020(root: Path, translation: str = "Hello") -> Path:
    package = root / "0020"
    package.mkdir(parents=True)

    paloc_data = {
        "metadata": {"tail_bytes": ""},
        "entries": [
            {
                "marker": "0102030405060708",
                "key": "UI_HELLO",
                "original": translation,
                "translation": translation,
            }
        ],
    }
    plaintext = json_data_to_paloc(paloc_data)
    payload = encrypt(compress_lz4(plaintext), "localizationstring_eng.paloc")
    padding = b"\x00" * ((16 - len(payload) % 16) % 16)
    paz_data = payload + padding
    (package / "0.paz").write_bytes(paz_data)

    folder_section = _pack_named_node(0xFFFFFFFF, "gamedata")
    node_section = _pack_named_node(0xFFFFFFFF, "localizationstring_eng.paloc")

    pamt = bytearray()
    pamt += b"\x00\x00\x00\x00"
    pamt += struct.pack("<I", 1)
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack("<II", pa_checksum(paz_data), len(paz_data))
    pamt += struct.pack("<I", len(folder_section)) + folder_section
    pamt += struct.pack("<I", len(node_section)) + node_section
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack(
        "<IIIII",
        0,
        0,
        len(payload),
        len(plaintext),
        0x00020000,
    )
    struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
    (package / "0.pamt").write_bytes(pamt)
    return package


class ServiceTests(unittest.TestCase):
    def test_selected_language_controls_source_entry_json_and_output_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = get_language_spec("Korean")
            source = build_localization_package(
                root / "source",
                spec.package_folder,
                spec.paloc_filename,
                "안녕하세요",
            )
            json_path = root / "output_unpack" / spec.json_filename

            export_result = export_localization_to_json(source, json_path, spec)
            build_result = build_localization_from_json(
                source,
                json_path,
                root / "output_repack",
                spec,
            )

            self.assertEqual(spec.name, export_result.language)
            self.assertEqual(spec.name, build_result.language)
            self.assertEqual(
                (root / "output_repack" / spec.package_folder).resolve(),
                build_result.output_package.resolve(),
            )
            self.assertEqual(
                (source / "0.pamt").read_bytes(),
                (root / "source" / spec.package_folder / "0.pamt").read_bytes(),
            )
            rebuilt = extract_archive_entry(
                build_result.output_package,
                spec.paloc_entry_path,
            )
            self.assertEqual(
                "안녕하세요",
                paloc_to_json_data(rebuilt)["entries"][0]["translation"],
            )

    def test_package_selected_for_a_different_language_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            english = get_language_spec("English")
            korean = get_language_spec("Korean")
            source = build_localization_package(
                root,
                english.package_folder,
                english.paloc_filename,
            )

            with self.assertRaisesRegex(Exception, korean.package_folder):
                export_localization_to_json(source, root / "wrong.json", korean)

    def test_unpack_then_repack_creates_self_consistent_0020(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _build_synthetic_0020(root / "source")
            json_path = root / "translation.json"
            output_parent = root / "release"
            source_pamt_before = (source / "0.pamt").read_bytes()
            source_paz_before = (source / "0.paz").read_bytes()

            export_result = export_0020_to_json(source, json_path)
            json_data = json.loads(json_path.read_text(encoding="utf-8"))
            json_data["entries"][0]["translation"] = "Xin chào Crimson Desert"
            json_path.write_text(
                json.dumps(json_data, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )

            build_result = build_0020_from_json(source, json_path, output_parent)

            output = output_parent / "0020"
            self.assertEqual(1, export_result.entry_count)
            self.assertEqual(output.resolve(), build_result.output_0020.resolve())
            self.assertTrue((output / "0.pamt").is_file())
            self.assertTrue((output / "0.paz").is_file())
            self.assertFalse((output_parent / "meta").exists())
            self.assertEqual(source_pamt_before, (source / "0.pamt").read_bytes())
            self.assertEqual(source_paz_before, (source / "0.paz").read_bytes())

            index = parse_pamt(output / "0.pamt", output)
            entry = index.find_entry(TARGET_ENTRY_PATH)
            paz_data = entry.paz_path.read_bytes()
            stored_payload = paz_data[entry.offset : entry.offset + entry.comp_size]
            plaintext = decompress_lz4(
                decrypt(stored_payload, "localizationstring_eng.paloc"),
                entry.orig_size,
            )
            rebuilt_json = paloc_to_json_data(plaintext)

            self.assertEqual(
                "Xin chào Crimson Desert",
                rebuilt_json["entries"][0]["translation"],
            )
            self.assertEqual(0, len(paz_data) % 16)
            self.assertEqual(len(paz_data), index.paz_records[0].stored_size)
            self.assertEqual(pa_checksum(paz_data), index.paz_records[0].stored_checksum)
            output_pamt = (output / "0.pamt").read_bytes()
            self.assertEqual(
                struct.unpack_from("<I", output_pamt, 0)[0],
                pa_checksum(output_pamt[12:]),
            )

    def test_existing_output_is_backed_up_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _build_synthetic_0020(root / "source")
            json_path = root / "translation.json"
            output_parent = root / "release"
            export_0020_to_json(source, json_path)
            existing = output_parent / "0020"
            existing.mkdir(parents=True)
            (existing / "keep.txt").write_text("old", encoding="utf-8")

            result = build_0020_from_json(source, json_path, output_parent)

            self.assertIsNotNone(result.backup_0020)
            self.assertEqual(
                "old",
                (result.backup_0020 / "keep.txt").read_text(encoding="utf-8"),
            )

    def test_existing_json_is_backed_up_before_unpack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _build_synthetic_0020(root / "source")
            json_path = root / "translation.json"
            json_path.write_text("keep", encoding="utf-8")

            result = export_0020_to_json(source, json_path)

            self.assertIsNotNone(result.backup_json)
            self.assertEqual(
                "keep",
                result.backup_json.read_text(encoding="utf-8"),
            )
            self.assertTrue(json_path.is_file())

    def test_duplicate_archive_entry_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _build_synthetic_0020(root / "source")
            pamt = bytearray((source / "0.pamt").read_bytes())
            pamt += pamt[-20:]
            struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
            (source / "0.pamt").write_bytes(pamt)

            with self.assertRaisesRegex(ValueError, "duplicate entry paths"):
                export_0020_to_json(source, root / "translation.json")

    def test_multi_paloc_unpack_then_repack_preserves_current_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_multi_paloc_package(root / "source")
            source_pamt = (source / "0.pamt").read_bytes()
            source_paz = (source / "0.paz").read_bytes()
            json_path = root / "translation.json"

            export_result = export_0020_to_json(source, json_path)
            catalog = json.loads(json_path.read_text(encoding="utf-8"))

            self.assertEqual(2, catalog["metadata"]["format_version"])
            self.assertEqual(3, export_result.paloc_count)
            self.assertEqual(3, len(catalog["metadata"]["palocs"]))
            self.assertEqual(
                {
                    "gamedata/aidialog.paloc",
                    "gamedata/quest.paloc",
                    "gamedata/ui.paloc",
                },
                {paloc["path"] for paloc in catalog["metadata"]["palocs"]},
            )
            self.assertTrue(
                all(
                    set(entry) == {"marker", "key", "original", "translation"}
                    for entry in catalog["entries"]
                )
            )
            catalog["entries"][0]["translation"] = "Xin chào"
            json_path.write_text(
                json.dumps(catalog, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )

            result = build_0020_from_json(source, json_path, root / "output")
            output = result.output_0020
            output_index = parse_pamt(output / "0.pamt", output)

            self.assertEqual(3, result.paloc_count)
            self.assertEqual(3, len(output_index.entries))
            rebuilt = paloc_to_json_data(
                extract_archive_entry(output, "gamedata/aidialog.paloc")
            )
            self.assertEqual("Xin chào", rebuilt["entries"][0]["translation"])
            for item in catalog["metadata"]["palocs"]:
                rebuilt_paloc = paloc_to_json_data(
                    extract_archive_entry(output, item["path"])
                )
                self.assertEqual(
                    item["tail_bytes"],
                    rebuilt_paloc["metadata"]["tail_bytes"],
                )
            self.assertEqual(source_pamt, (source / "0.pamt").read_bytes())
            self.assertEqual(source_paz, (source / "0.paz").read_bytes())

    def test_legacy_json_migrates_by_identity_into_multi_paloc_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_multi_paloc_package(root / "source")
            json_path = root / "legacy.json"
            json_path.write_text(
                json.dumps(
                    {
                        "metadata": {"tail_bytes": "48cd0200"},
                        "entries": [
                            {
                                "marker": "0101010101010101",
                                "key": "DIALOG_HELLO",
                                "original": "Old hello",
                                "translation": "Xin chào",
                            },
                            {
                                "marker": "ffffffffffffffff",
                                "key": "REMOVED_KEY",
                                "original": "Removed",
                                "translation": "Đã xóa",
                            },
                        ],
                    },
                    ensure_ascii=False,
                    indent=4,
                ),
                encoding="utf-8",
            )

            result = build_0020_from_json(source, json_path, root / "output")
            output = result.output_0020
            dialog = paloc_to_json_data(
                extract_archive_entry(output, "gamedata/aidialog.paloc")
            )
            quest = paloc_to_json_data(
                extract_archive_entry(output, "gamedata/quest.paloc")
            )

            self.assertEqual("Xin chào", dialog["entries"][0]["translation"])
            self.assertEqual("Begin the quest", quest["entries"][0]["translation"])
            self.assertEqual(1, result.matched_entries)
            self.assertEqual(2, result.new_entries)
            self.assertEqual(1, result.removed_entries)
            self.assertEqual(1, result.changed_originals)

    def test_duplicate_identity_in_catalog_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_multi_paloc_package(root / "source")
            json_path = root / "translation.json"
            export_0020_to_json(source, json_path)
            catalog = json.loads(json_path.read_text(encoding="utf-8"))
            catalog["entries"].append(dict(catalog["entries"][0]))
            json_path.write_text(
                json.dumps(catalog, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"Duplicate marker \+ key"):
                build_0020_from_json(source, json_path, root / "output")

            self.assertFalse((root / "output").exists())

    def test_duplicate_identity_across_source_palocs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate = {
                "marker": "0101010101010101",
                "key": "SHARED_KEY",
                "original": "Shared",
                "translation": "Shared",
            }
            source = build_multi_paloc_package(
                root / "source",
                entries_by_paloc={
                    "aidialog.paloc": [dict(duplicate)],
                    "quest.paloc": [dict(duplicate)],
                },
            )

            with self.assertRaisesRegex(ValueError, r"Duplicate marker \+ key"):
                export_0020_to_json(source, root / "translation.json")

    def test_bad_source_pamt_checksum_is_rejected_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_multi_paloc_package(root / "source")
            pamt_path = source / "0.pamt"
            pamt = bytearray(pamt_path.read_bytes())
            pamt[0] ^= 0xFF
            pamt_path.write_bytes(pamt)

            with self.assertRaisesRegex(ValueError, "0.pamt header checksum"):
                export_0020_to_json(source, root / "translation.json")

    def test_bad_source_paz_checksum_is_rejected_before_repack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_multi_paloc_package(root / "source")
            json_path = root / "translation.json"
            export_0020_to_json(source, json_path)
            paz_path = source / "0.paz"
            paz = bytearray(paz_path.read_bytes())
            paz[-1] ^= 0xFF
            paz_path.write_bytes(paz)

            with self.assertRaisesRegex(ValueError, "PAZ checksum"):
                build_0020_from_json(source, json_path, root / "output")

            self.assertFalse((root / "output").exists())

    def test_overlapping_multi_paloc_entries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_multi_paloc_package(root / "source")
            pamt_path = source / "0.pamt"
            index = parse_pamt(pamt_path, source)
            pamt = bytearray(pamt_path.read_bytes())
            struct.pack_into(
                "<I",
                pamt,
                index.entries[1].record_offset + 4,
                index.entries[0].offset,
            )
            struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
            pamt_path.write_bytes(pamt)

            with self.assertRaisesRegex(ValueError, "overlapping entries"):
                export_0020_to_json(source, root / "translation.json")


if __name__ == "__main__":
    unittest.main()
