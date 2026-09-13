import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crimson0020.checksums import pa_checksum
from crimson0020.crypto import compress_lz4, encrypt
from crimson0020.installer import (
    BackupNotFoundError,
    InstallationError,
    UninstallationError,
    install_0020,
    uninstall_0020,
)
from crimson0020.metadata import parse_papgt, patch_folder_crc
from crimson0020.paloc import json_data_to_paloc

from test_metadata import build_papgt


def _pack_named_node(parent: int, name: str) -> bytes:
    encoded = name.encode("utf-8")
    return struct.pack("<IB", parent, len(encoded)) + encoded


def build_synthetic_0020(root: Path, translation: str) -> Path:
    package = root / "0020"
    package.mkdir(parents=True)
    plaintext = json_data_to_paloc(
        {
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
    )
    payload = encrypt(compress_lz4(plaintext), "localizationstring_eng.paloc")
    paz_data = payload + b"\x00" * ((16 - len(payload) % 16) % 16)
    (package / "0.paz").write_bytes(paz_data)

    folder_section = _pack_named_node(0xFFFFFFFF, "gamedata")
    node_section = _pack_named_node(0xFFFFFFFF, "localizationstring_eng.paloc")
    pamt = bytearray(b"\x00\x00\x00\x00")
    pamt += struct.pack("<I", 1)
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack("<II", pa_checksum(paz_data), len(paz_data))
    pamt += struct.pack("<I", len(folder_section)) + folder_section
    pamt += struct.pack("<I", len(node_section)) + node_section
    pamt += struct.pack("<II", 0, 0)
    pamt += struct.pack("<IIIII", 0, 0, len(payload), len(plaintext), 0x00020000)
    struct.pack_into("<I", pamt, 0, pa_checksum(pamt[12:]))
    (package / "0.pamt").write_bytes(pamt)
    return package


def build_game(root: Path, translation: str = "Original") -> tuple[Path, bytes]:
    game_root = root / "Crimson Desert"
    game_0020 = build_synthetic_0020(game_root, translation)
    metadata = build_papgt(
        [
            ("0019", 0xAABBCCDD),
            ("0020", pa_checksum((game_0020 / "0.pamt").read_bytes()[12:])),
            ("0021", 0x11223344),
        ]
    )
    meta = game_root / "meta"
    meta.mkdir()
    (meta / "0.papgt").write_bytes(metadata)
    return game_root, metadata


class InstallerTests(unittest.TestCase):
    def test_install_backs_up_files_and_patches_only_current_0020_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_synthetic_0020(root / "release", "New translation")
            game_root, original_metadata = build_game(root / "game")
            old_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            old_paz = (game_root / "0020" / "0.paz").read_bytes()

            result = install_0020(source, game_root)

            self.assertEqual(
                (source / "0.pamt").read_bytes(),
                (game_root / "0020" / "0.pamt").read_bytes(),
            )
            self.assertEqual(
                (source / "0.paz").read_bytes(),
                (game_root / "0020" / "0.paz").read_bytes(),
            )
            self.assertEqual(old_pamt, (result.backup_directory / "0020" / "0.pamt").read_bytes())
            self.assertEqual(old_paz, (result.backup_directory / "0020" / "0.paz").read_bytes())
            self.assertEqual(
                original_metadata,
                (result.backup_directory / "meta" / "0.papgt").read_bytes(),
            )

            installed_metadata = (game_root / "meta" / "0.papgt").read_bytes()
            original_index = parse_papgt(original_metadata)
            installed_index = parse_papgt(installed_metadata)
            self.assertEqual(
                original_index.entry_for("0019").crc,
                installed_index.entry_for("0019").crc,
            )
            self.assertEqual(
                original_index.entry_for("0021").crc,
                installed_index.entry_for("0021").crc,
            )
            self.assertEqual(result.pamt_crc, installed_index.entry_for("0020").crc)
            self.assertEqual(result.metadata_checksum, installed_index.stored_checksum)

    def test_failure_after_copy_rolls_back_all_original_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_synthetic_0020(root / "release", "New translation")
            game_root, original_metadata = build_game(root / "game")
            old_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            old_paz = (game_root / "0020" / "0.paz").read_bytes()

            with patch(
                "crimson0020.installer._write_metadata_atomically",
                side_effect=OSError("simulated metadata write failure"),
            ):
                with self.assertRaises(InstallationError) as caught:
                    install_0020(source, game_root)

            self.assertTrue(caught.exception.rolled_back)
            self.assertTrue(caught.exception.backup_directory.is_dir())
            self.assertEqual(old_pamt, (game_root / "0020" / "0.pamt").read_bytes())
            self.assertEqual(old_paz, (game_root / "0020" / "0.paz").read_bytes())
            self.assertEqual(original_metadata, (game_root / "meta" / "0.papgt").read_bytes())

    def test_game_metadata_without_0020_is_rejected_before_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_synthetic_0020(root / "release", "New translation")
            game_root, _ = build_game(root / "game")
            (game_root / "meta" / "0.papgt").write_bytes(
                build_papgt([("0019", 0xAABBCCDD)])
            )

            with self.assertRaisesRegex(Exception, "0020"):
                install_0020(source, game_root)

            self.assertFalse((game_root / "meta" / "crimson0020_backups").exists())

    def test_uninstall_restores_latest_0020_and_preserves_newer_mod_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_synthetic_0020(root / "release", "Installed")
            game_root, original_metadata = build_game(root / "game")
            original_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            original_paz = (game_root / "0020" / "0.paz").read_bytes()
            install_result = install_0020(source, game_root)
            installed_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            installed_paz = (game_root / "0020" / "0.paz").read_bytes()

            current_metadata = (game_root / "meta" / "0.papgt").read_bytes()
            newer_mod_metadata = patch_folder_crc(
                current_metadata,
                "0019",
                0xDEADBEEF,
            )
            (game_root / "meta" / "0.papgt").write_bytes(newer_mod_metadata)

            result = uninstall_0020(game_root)

            self.assertEqual(original_pamt, (game_root / "0020" / "0.pamt").read_bytes())
            self.assertEqual(original_paz, (game_root / "0020" / "0.paz").read_bytes())
            restored_metadata = (game_root / "meta" / "0.papgt").read_bytes()
            restored_index = parse_papgt(restored_metadata)
            original_index = parse_papgt(original_metadata)
            self.assertEqual(0xDEADBEEF, restored_index.entry_for("0019").crc)
            self.assertEqual(
                original_index.entry_for("0020").crc,
                restored_index.entry_for("0020").crc,
            )
            self.assertFalse(install_result.backup_directory.exists())
            self.assertTrue(result.restored_backup_directory.name.startswith("restored-"))
            self.assertEqual(
                installed_pamt,
                (result.safety_backup_directory / "0020" / "0.pamt").read_bytes(),
            )
            self.assertEqual(
                installed_paz,
                (result.safety_backup_directory / "0020" / "0.paz").read_bytes(),
            )
            self.assertEqual(
                newer_mod_metadata,
                (result.safety_backup_directory / "meta" / "0.papgt").read_bytes(),
            )

    def test_repeated_uninstall_walks_install_backups_as_a_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = build_synthetic_0020(root / "first", "First")
            second = build_synthetic_0020(root / "second", "Second")
            game_root, _ = build_game(root / "game")
            original_pamt = (game_root / "0020" / "0.pamt").read_bytes()

            install_0020(first, game_root)
            first_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            install_0020(second, game_root)

            uninstall_0020(game_root)
            self.assertEqual(first_pamt, (game_root / "0020" / "0.pamt").read_bytes())

            uninstall_0020(game_root)
            self.assertEqual(original_pamt, (game_root / "0020" / "0.pamt").read_bytes())

    def test_uninstall_failure_rolls_back_current_installed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = build_synthetic_0020(root / "release", "Installed")
            game_root, _ = build_game(root / "game")
            install_result = install_0020(source, game_root)
            installed_pamt = (game_root / "0020" / "0.pamt").read_bytes()
            installed_paz = (game_root / "0020" / "0.paz").read_bytes()
            installed_metadata = (game_root / "meta" / "0.papgt").read_bytes()

            with patch(
                "crimson0020.installer._write_metadata_atomically",
                side_effect=OSError("simulated uninstall metadata failure"),
            ):
                with self.assertRaises(UninstallationError) as caught:
                    uninstall_0020(game_root)

            self.assertTrue(caught.exception.rolled_back)
            self.assertTrue(install_result.backup_directory.is_dir())
            self.assertEqual(installed_pamt, (game_root / "0020" / "0.pamt").read_bytes())
            self.assertEqual(installed_paz, (game_root / "0020" / "0.paz").read_bytes())
            self.assertEqual(
                installed_metadata,
                (game_root / "meta" / "0.papgt").read_bytes(),
            )

    def test_uninstall_without_install_backup_stops_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            game_root, original_metadata = build_game(Path(temp_dir))

            with self.assertRaises(BackupNotFoundError):
                uninstall_0020(game_root)

            self.assertEqual(
                original_metadata,
                (game_root / "meta" / "0.papgt").read_bytes(),
            )
            self.assertFalse((game_root / "meta" / "crimson0020_backups").exists())


if __name__ == "__main__":
    unittest.main()
