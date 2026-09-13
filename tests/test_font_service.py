import tempfile
import unittest
from pathlib import Path

from crimson0020.archive import extract_archive_entry, parse_pamt
from crimson0020.checksums import pa_checksum, pa_checksum_file
from crimson0020.font_service import (
    FontFormatError,
    build_font_package,
    export_fonts,
)
from crimson0020.languages import get_language_spec

from package_fixtures import build_font_package as build_source_font_package
from package_fixtures import valid_ttf


class FontServiceTests(unittest.TestCase):
    def test_font_export_refuses_any_path_overlapping_source_0012(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = get_language_spec("English")
            source, _ = build_source_font_package(root / "game", spec.font_filename)

            for unsafe_output in (source, source / "fonts", source.parent):
                with self.subTest(unsafe_output=unsafe_output):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        export_fonts(source, unsafe_output, spec)

    def test_unpack_then_repack_builds_verified_0012_patch_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = get_language_spec("English")
            source, original_fonts = build_source_font_package(
                root / "game",
                spec.font_filename,
            )
            source_pamt = (source / "0.pamt").read_bytes()
            source_last_paz = (source / "1.paz").read_bytes()
            font_folder = root / "output_unpack" / "fonts"

            export_result = export_fonts(source, font_folder, spec)
            replacement = valid_ttf("vietnamese-replacement")
            (font_folder / spec.font_filename).write_bytes(replacement)
            build_result = build_font_package(
                source,
                font_folder,
                root / "output_repack",
                spec,
            )

            output = root / "output_repack" / "0012"
            self.assertEqual(3, export_result.font_count)
            self.assertEqual(output.resolve(), build_result.output_0012.resolve())
            self.assertTrue((output / "0.pamt").is_file())
            self.assertTrue((output / "1.paz").is_file())
            self.assertFalse((output / "0.paz").exists())
            self.assertEqual(source_pamt, (source / "0.pamt").read_bytes())
            self.assertEqual(source_last_paz, (source / "1.paz").read_bytes())

            index = parse_pamt(output / "0.pamt", output)
            for entry_path in spec.font_entry_paths:
                entry = index.find_entry(entry_path)
                self.assertEqual(1, entry.paz_index)
            self.assertEqual(
                original_fonts["basefont.ttf"],
                extract_archive_entry(output, "ui/basefont.ttf"),
            )
            self.assertEqual(
                replacement,
                extract_archive_entry(output, spec.font_entry_path),
            )
            self.assertEqual(
                original_fonts["creditfont.ttf"],
                extract_archive_entry(output, "ui/creditfont.ttf"),
            )
            record = index.paz_records[1]
            self.assertEqual((output / "1.paz").stat().st_size, record.stored_size)
            self.assertEqual(pa_checksum_file(output / "1.paz"), record.stored_checksum)
            pamt_data = (output / "0.pamt").read_bytes()
            self.assertEqual(pa_checksum(pamt_data[12:]), int.from_bytes(pamt_data[:4], "little"))

    def test_missing_or_invalid_font_stops_before_output_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = get_language_spec("English")
            source, _ = build_source_font_package(root / "game", spec.font_filename)
            font_folder = root / "fonts"
            font_folder.mkdir()
            (font_folder / "basefont.ttf").write_bytes(valid_ttf("base"))
            (font_folder / "creditfont.ttf").write_bytes(valid_ttf("credit"))

            with self.assertRaisesRegex(FontFormatError, spec.font_filename):
                build_font_package(source, font_folder, root / "release", spec)

            (font_folder / spec.font_filename).write_bytes(b"not-a-font")
            with self.assertRaisesRegex(FontFormatError, "TrueType/OpenType"):
                build_font_package(source, font_folder, root / "release", spec)
            self.assertFalse((root / "release" / "0012").exists())

    def test_language_selection_controls_the_font_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            english = get_language_spec("English")
            korean = get_language_spec("Korean")
            source, _ = build_source_font_package(root, english.font_filename)

            with self.assertRaisesRegex(Exception, korean.font_filename):
                export_fonts(source, root / "fonts", korean)


if __name__ == "__main__":
    unittest.main()
