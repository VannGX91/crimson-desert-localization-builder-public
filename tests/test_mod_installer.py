import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.checksums import pa_checksum
from crimson0020.font_service import build_font_package, export_fonts
from crimson0020.languages import get_language_spec
from crimson0020.installer import BackupFormatError, install_0020
from crimson0020.metadata import parse_papgt, patch_folder_crc
from crimson0020.mod_installer import install_mod, uninstall_mod

from package_fixtures import (
    build_font_package as build_source_font_package,
    build_localization_package,
    build_multi_paloc_package,
    valid_ttf,
)
from test_metadata import build_papgt
from test_installer import build_game, build_synthetic_0020


def pamt_crc(package: Path) -> int:
    return pa_checksum((package / "0.pamt").read_bytes()[12:])


def build_multilanguage_game(root: Path) -> tuple[Path, object, bytes]:
    spec = get_language_spec("Korean")
    game_root = root / "Crimson Desert"
    language_package = build_localization_package(
        game_root,
        spec.package_folder,
        spec.paloc_filename,
        "Original Korean",
    )
    font_package, _ = build_source_font_package(
        game_root,
        spec.font_filename,
        "original",
    )
    metadata = build_papgt(
        [
            ("0012", pamt_crc(font_package)),
            ("0019", pamt_crc(language_package)),
            ("0020", 0x11223344),
            ("0099", 0x55667788),
        ]
    )
    (game_root / "meta").mkdir()
    (game_root / "meta" / "0.papgt").write_bytes(metadata)
    return game_root, spec, metadata


def build_mod(root: Path, game_root: Path, spec: object, include_font: bool = True) -> Path:
    mod_root = root / "output_repack"
    build_localization_package(
        mod_root,
        spec.package_folder,
        spec.paloc_filename,
        "Vietnamese translation",
    )
    if include_font:
        font_folder = root / "fonts"
        export_fonts(game_root / "0012", font_folder, spec)
        (font_folder / spec.font_filename).write_bytes(valid_ttf("vietnamese-font"))
        build_font_package(game_root / "0012", font_folder, mod_root, spec)
    return mod_root


class ModInstallerTests(unittest.TestCase):
    def test_install_accepts_verified_multi_paloc_localization_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_root, spec, _ = build_multilanguage_game(root / "game")
            mod_root = root / "mod"
            multi_package = build_multi_paloc_package(
                mod_root,
                package_folder=spec.package_folder,
            )

            result = install_mod(mod_root, game_root, spec)

            self.assertEqual((spec.package_folder,), result.installed_folders)
            self.assertEqual(
                (multi_package / "0.pamt").read_bytes(),
                (game_root / spec.package_folder / "0.pamt").read_bytes(),
            )
            self.assertEqual(
                (multi_package / "0.paz").read_bytes(),
                (game_root / spec.package_folder / "0.paz").read_bytes(),
            )

    def test_repeated_installs_restore_as_a_package_aware_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_root, spec, _ = build_multilanguage_game(root / "game")
            original_language = (
                game_root / spec.package_folder / "0.pamt"
            ).read_bytes()
            original_font = (game_root / "0012" / "0.pamt").read_bytes()
            first = build_mod(root / "first", game_root, spec, include_font=True)
            second = build_mod(root / "second", game_root, spec, include_font=False)

            install_mod(first, game_root, spec)
            first_language = (
                game_root / spec.package_folder / "0.pamt"
            ).read_bytes()
            first_font = (game_root / "0012" / "0.pamt").read_bytes()
            install_mod(second, game_root, spec)

            uninstall_mod(game_root)
            self.assertEqual(
                first_language,
                (game_root / spec.package_folder / "0.pamt").read_bytes(),
            )
            self.assertEqual(first_font, (game_root / "0012" / "0.pamt").read_bytes())

            uninstall_mod(game_root)
            self.assertEqual(
                original_language,
                (game_root / spec.package_folder / "0.pamt").read_bytes(),
            )
            self.assertEqual(
                original_font,
                (game_root / "0012" / "0.pamt").read_bytes(),
            )

    def test_uninstall_remains_compatible_with_legacy_0020_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_synthetic_0020(root / "release", "Installed")
            game_root, _ = build_game(root / "game")
            original_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            install_0020(source, game_root)

            result = uninstall_mod(game_root)

            self.assertEqual(("0020",), result.restored_folders)
            self.assertEqual(
                original_pamt,
                (game_root / "0020" / "0.pamt").read_bytes(),
            )

    def test_damaged_manifest_stops_uninstall_before_game_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_root, spec, _ = build_multilanguage_game(root / "game")
            mod_root = build_mod(root / "mod", game_root, spec)
            result = install_mod(mod_root, game_root, spec)
            installed_metadata = (game_root / "meta" / "0.papgt").read_bytes()
            installed_language = (
                game_root / spec.package_folder / "0.pamt"
            ).read_bytes()
            manifest_path = result.backup_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["folders"][0]["files"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(BackupFormatError):
                uninstall_mod(game_root)

            self.assertEqual(
                installed_metadata,
                (game_root / "meta" / "0.papgt").read_bytes(),
            )
            self.assertEqual(
                installed_language,
                (game_root / spec.package_folder / "0.pamt").read_bytes(),
            )

    def test_install_and_uninstall_selected_language_with_optional_font(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_root, spec, original_metadata = build_multilanguage_game(root / "game")
            original_metadata = patch_folder_crc(
                original_metadata,
                "0012",
                0x62DB8C1E,
            )
            (game_root / "meta" / "0.papgt").write_bytes(original_metadata)
            mod_root = build_mod(root / "mod", game_root, spec)
            original_files = {
                path.relative_to(game_root): path.read_bytes()
                for path in (
                    game_root / spec.package_folder / "0.pamt",
                    game_root / spec.package_folder / "0.paz",
                    game_root / "0012" / "0.pamt",
                    game_root / "0012" / "1.paz",
                )
            }

            install_result = install_mod(mod_root, game_root, spec)

            self.assertEqual(("0019", "0012"), install_result.installed_folders)
            for folder_name, filename in (
                (spec.package_folder, "0.pamt"),
                (spec.package_folder, "0.paz"),
                ("0012", "0.pamt"),
                ("0012", "1.paz"),
            ):
                self.assertEqual(
                    (mod_root / folder_name / filename).read_bytes(),
                    (game_root / folder_name / filename).read_bytes(),
                )
            installed_index = parse_papgt(
                (game_root / "meta" / "0.papgt").read_bytes()
            )
            self.assertEqual(
                pamt_crc(mod_root / spec.package_folder),
                installed_index.entry_for(spec.package_folder).crc,
            )
            self.assertEqual(
                pamt_crc(mod_root / "0012"),
                installed_index.entry_for("0012").crc,
            )
            original_index = parse_papgt(original_metadata)
            self.assertEqual(
                original_index.entry_for("0020").crc,
                installed_index.entry_for("0020").crc,
            )

            current_metadata = (game_root / "meta" / "0.papgt").read_bytes()
            newer_metadata = patch_folder_crc(current_metadata, "0099", 0xDEADBEEF)
            (game_root / "meta" / "0.papgt").write_bytes(newer_metadata)
            uninstall_result = uninstall_mod(game_root)

            self.assertEqual(("0019", "0012"), uninstall_result.restored_folders)
            for relative_path, expected in original_files.items():
                self.assertEqual(expected, (game_root / relative_path).read_bytes())
            restored_index = parse_papgt(
                (game_root / "meta" / "0.papgt").read_bytes()
            )
            self.assertEqual(0xDEADBEEF, restored_index.entry_for("0099").crc)
            self.assertEqual(
                original_index.entry_for("0019").crc,
                restored_index.entry_for("0019").crc,
            )
            self.assertEqual(
                pamt_crc(game_root / "0012"),
                restored_index.entry_for("0012").crc,
            )
            self.assertNotEqual(
                original_index.entry_for("0012").crc,
                restored_index.entry_for("0012").crc,
            )
            self.assertTrue(uninstall_result.restored_backup_directory.is_dir())

    def test_font_package_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_root, spec, _ = build_multilanguage_game(root / "game")
            original_font_pamt = (game_root / "0012" / "0.pamt").read_bytes()
            mod_root = build_mod(root / "mod", game_root, spec, include_font=False)

            result = install_mod(mod_root, game_root, spec)

            self.assertEqual((spec.package_folder,), result.installed_folders)
            self.assertEqual(
                original_font_pamt,
                (game_root / "0012" / "0.pamt").read_bytes(),
            )

    def test_failure_after_copy_rolls_back_every_selected_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            game_root, spec, original_metadata = build_multilanguage_game(root / "game")
            mod_root = build_mod(root / "mod", game_root, spec)
            original_files = {
                path: path.read_bytes()
                for path in (
                    game_root / spec.package_folder / "0.pamt",
                    game_root / spec.package_folder / "0.paz",
                    game_root / "0012" / "0.pamt",
                    game_root / "0012" / "1.paz",
                )
            }

            with patch(
                "crimson0020.mod_installer._write_metadata_atomically",
                side_effect=OSError("simulated metadata failure"),
            ):
                with self.assertRaises(Exception) as caught:
                    install_mod(mod_root, game_root, spec)

            self.assertTrue(caught.exception.rolled_back)
            for path, expected in original_files.items():
                self.assertEqual(expected, path.read_bytes())
            self.assertEqual(
                original_metadata,
                (game_root / "meta" / "0.papgt").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
