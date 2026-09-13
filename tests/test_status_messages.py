import errno
import json
import unittest
from pathlib import Path

from crimson0020.archive import ArchiveFormatError
from crimson0020.auto_translate import (
    TranslationCheckpointError,
    TranslationContractError,
)
from crimson0020.font_service import FontFormatError
from crimson0020.installer import GamePathError, InstallationError
from crimson0020.installer import (
    BackupFormatError,
    BackupNotFoundError,
    UninstallationError,
)
from crimson0020.metadata import MetadataFormatError
from crimson0020.paloc import PalocFormatError
from crimson0020.status_messages import (
    error_status,
    ready_status,
    success_status,
    working_status,
)
from crimson0020.translation_providers import (
    RetryableTranslationProviderError,
    TranslationProviderError,
)


class StatusMessageTests(unittest.TestCase):
    def test_auto_translate_failures_have_actionable_statuses(self) -> None:
        cases = (
            (
                TranslationContractError("placeholder damaged"),
                "BẢN DỊCH LÀM HỎNG CẤU TRÚC",
            ),
            (
                TranslationCheckpointError("source changed"),
                "CHECKPOINT KHÔNG KHỚP",
            ),
            (
                RetryableTranslationProviderError("timeout"),
                "KHÔNG KẾT NỐI ĐƯỢC DỊCH VỤ DỊCH",
            ),
            (
                TranslationProviderError("bad response"),
                "DỊCH VỤ DỊCH TRẢ VỀ DỮ LIỆU LỖI",
            ),
        )

        for error, title in cases:
            with self.subTest(error=error):
                self.assertEqual(title, error_status(error).title)

    def test_provider_failure_invites_safe_resume_and_checkpoint_is_provider_portable(self) -> None:
        checkpoint = error_status(TranslationCheckpointError("source changed"))
        temporary = error_status(RetryableTranslationProviderError("timeout"))

        self.assertNotIn("provider", checkpoint.body.casefold())
        self.assertIn("dịch vụ khác", temporary.body.casefold())
        self.assertIn("tiếp tục", temporary.body.casefold())

    def test_font_errors_name_missing_invalid_and_wrong_0012_cases(self) -> None:
        missing = error_status(
            FontFormatError("Required font file was not found: basefont_kor.ttf")
        )
        invalid = error_status(
            FontFormatError("basefont_kor.ttf is not a valid TrueType/OpenType font file.")
        )
        wrong_package = error_status(
            FontFormatError("Please select the 0012 folder containing fonts.")
        )

        self.assertEqual("THIẾU FILE FONT", missing.title)
        self.assertEqual("FONT KHÔNG HỢP LỆ", invalid.title)
        self.assertEqual("GÓI FONT 0012 KHÔNG HỢP LỆ", wrong_package.title)

    def test_missing_source_files_have_specific_guidance(self) -> None:
        status = error_status(
            ArchiveFormatError("Folder 0020 must contain both 0.pamt and 0.paz.")
        )

        self.assertEqual("✕", status.icon)
        self.assertEqual("THIẾU FILE NGUỒN", status.title)
        self.assertIn("0.pamt", status.body)
        self.assertIn("0.paz", status.body)

    def test_invalid_translation_is_identified(self) -> None:
        error = PalocFormatError("entries[12].translation must be text.")

        status = error_status(error)

        self.assertEqual("TRANSLATION KHÔNG HỢP LỆ", status.title)
        self.assertIn("entries[12]", status.detail)

    def test_permission_and_missing_json_are_distinct(self) -> None:
        permission = error_status(PermissionError("Access is denied"))
        missing_json = error_status(
            ValueError(r"JSON file does not exist: D:\work\translation.json")
        )

        self.assertEqual("KHÔNG CÓ QUYỀN TRUY CẬP", permission.title)
        self.assertEqual("KHÔNG TÌM THẤY JSON", missing_json.title)

    def test_unknown_error_keeps_technical_detail(self) -> None:
        status = error_status(RuntimeError("unexpected state"))

        self.assertEqual("LỖI KHÔNG XÁC ĐỊNH", status.title)
        self.assertIn("RuntimeError: unexpected state", status.detail)

    def test_invalid_game_path_and_metadata_have_specific_guidance(self) -> None:
        game_path = error_status(
            GamePathError(r"The game folder must contain meta\0.papgt.")
        )
        metadata = error_status(
            MetadataFormatError("0.papgt self-checksum does not match.")
        )

        self.assertEqual("SAI THƯ MỤC GAME", game_path.title)
        self.assertIn(r"meta\0.papgt", game_path.body)
        self.assertEqual("METADATA GAME KHÔNG HỢP LỆ", metadata.title)

    def test_install_failure_reports_rollback_result_and_backup(self) -> None:
        backup = Path(r"D:\game\meta\crimson0020_backups\install-test")
        restored = error_status(
            InstallationError(
                "Installation failed; original files were restored.",
                rolled_back=True,
                backup_directory=backup,
            )
        )
        manual = error_status(
            InstallationError(
                "Installation and rollback failed.",
                rolled_back=False,
                backup_directory=backup,
            )
        )

        self.assertEqual("CÀI ĐẶT THẤT BẠI — ĐÃ HOÀN TÁC", restored.title)
        self.assertIn(str(backup), restored.detail)
        self.assertEqual("CÀI ĐẶT VÀ HOÀN TÁC THẤT BẠI", manual.title)
        self.assertIn("phục hồi thủ công", manual.body)

    def test_normal_status_factories_use_expected_icons(self) -> None:
        ready = ready_status()
        self.assertEqual("●", ready.icon)
        self.assertIn("Uninstall", ready.body)
        self.assertEqual("…", working_status("WORK", "body").icon)
        self.assertEqual("✓", success_status("DONE", "body", "detail").icon)

    def test_known_file_value_and_archive_errors_are_classified(self) -> None:
        cases = (
            (FileNotFoundError("missing"), "KHÔNG TÌM THẤY FILE"),
            (MemoryError("memory"), "KHÔNG ĐỦ BỘ NHỚ"),
            (OSError(errno.ENOSPC, "disk full"), "KHÔNG ĐỦ DUNG LƯỢNG"),
            (OSError(errno.EIO, "io"), "LỖI ĐỌC/GHI FILE"),
            (ValueError("Output must be outside source"), "THƯ MỤC OUTPUT KHÔNG HỢP LỆ"),
            (
                ValueError("Trusted translation template is invalid: bad schema"),
                "FILE MẪU DỊCH KHÔNG HỢP LỆ",
            ),
            (ValueError("Repaired delta output must be separate from its inputs."), "FILE ĐẦU RA BỊ TRÙNG"),
            (ValueError("Merged JSON output must be separate from every input."), "FILE ĐẦU RA BỊ TRÙNG"),
            (ValueError("Please select an output folder"), "CHƯA CHỌN NƠI LƯU"),
            (ValueError("Unexpected output contract"), "DỮ LIỆU KHÔNG HỢP LỆ"),
            (ValueError("bad value"), "DỮ LIỆU KHÔNG HỢP LỆ"),
            (
                ArchiveFormatError("Please select a folder named 0020"),
                "CHỌN SAI THƯ MỤC NGUỒN",
            ),
            (ArchiveFormatError("Cannot read 0.pamt"), "KHÔNG ĐỌC ĐƯỢC 0.PAMT"),
            (
                ArchiveFormatError("Cannot decompress localization"),
                "KHÔNG THỂ GIẢI NÉN PALOC",
            ),
            (ArchiveFormatError("checksum mismatch"), "XÁC MINH PACKAGE THẤT BẠI"),
            (
                ArchiveFormatError("unsupported layout"),
                "PACKAGE LOCALIZATION KHÔNG ĐƯỢC HỖ TRỢ",
            ),
        )

        for error, expected_title in cases:
            with self.subTest(error=error):
                self.assertEqual(expected_title, error_status(error).title)

    def test_uninstall_errors_report_backup_and_rollback_state(self) -> None:
        safety = Path(
            r"D:\game\meta\crimson0020_backups\uninstall-safety-test"
        )
        missing = error_status(BackupNotFoundError("No install backup exists"))
        invalid = error_status(BackupFormatError("Install backup is invalid"))
        restored = error_status(
            UninstallationError(
                "Uninstall failed; installed files were restored.",
                rolled_back=True,
                safety_backup_directory=safety,
            )
        )
        manual = error_status(
            UninstallationError(
                "Uninstall and rollback failed.",
                rolled_back=False,
                safety_backup_directory=safety,
            )
        )

        self.assertEqual("CHƯA CÓ BACKUP ĐỂ GỠ", missing.title)
        self.assertEqual("BACKUP GỠ CÀI ĐẶT KHÔNG HỢP LỆ", invalid.title)
        self.assertEqual("GỠ CÀI ĐẶT THẤT BẠI — ĐÃ HOÀN TÁC", restored.title)
        self.assertIn(str(safety), restored.detail)
        self.assertEqual("GỠ VÀ HOÀN TÁC THẤT BẠI", manual.title)
        self.assertIn("phục hồi thủ công", manual.body)

    def test_metadata_and_paloc_subtypes_are_classified(self) -> None:
        try:
            json.loads("{")
        except json.JSONDecodeError as cause:
            syntax_error = PalocFormatError("Cannot parse localization JSON")
            syntax_error.__cause__ = cause

        cases = (
            (
                MetadataFormatError("Source 0.paz checksum mismatch"),
                "PACKAGE NGUỒN KHÔNG HỢP LỆ",
            ),
            (
                MetadataFormatError("must contain exactly one 0020 entry"),
                "KHÔNG TÌM THẤY MỤC PACKAGE",
            ),
            (syntax_error, "JSON SAI CÚ PHÁP"),
            (PalocFormatError("Cannot read localization JSON"), "KHÔNG ĐỌC ĐƯỢC JSON"),
            (PalocFormatError("invalid UTF-8"), "ENCODING KHÔNG HỢP LỆ"),
            (PalocFormatError("invalid marker"), "CẤU TRÚC JSON/PALOC KHÔNG HỢP LỆ"),
        )

        for error, expected_title in cases:
            with self.subTest(error=error):
                self.assertEqual(expected_title, error_status(error).title)


if __name__ == "__main__":
    unittest.main()
