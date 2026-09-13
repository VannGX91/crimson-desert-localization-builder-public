import errno
import json
from dataclasses import dataclass

from .auto_translate import TranslationCheckpointError, TranslationContractError
from .archive import ArchiveFormatError
from .font_service import FontFormatError
from .installer import (
    BackupFormatError,
    BackupNotFoundError,
    GamePathError,
    InstallationError,
    UninstallationError,
)
from .metadata import MetadataFormatError
from .paloc import PalocFormatError
from .languages import LanguageSpec, get_language_spec
from .translation_providers import (
    RetryableTranslationProviderError,
    TranslationProviderError,
)


@dataclass(frozen=True)
class StatusMessage:
    level: str
    icon: str
    title: str
    body: str
    detail: str = ""


def ready_status(spec: LanguageSpec | None = None) -> StatusMessage:
    selected = spec or get_language_spec("English")
    return StatusMessage(
        level="ready",
        icon="●",
        title="SẴN SÀNG",
        body=(
            f"Ngôn ngữ nguồn: {selected.name} ({selected.package_folder}). Chọn "
            "localization/font nguồn, sau đó Unpack, Auto Translate, Repack, "
            "Install hoặc Uninstall."
        ),
    )


def working_status(title: str, body: str) -> StatusMessage:
    return StatusMessage(level="working", icon="…", title=title, body=body)


def success_status(title: str, body: str, detail: str = "") -> StatusMessage:
    return StatusMessage(
        level="success",
        icon="✓",
        title=title,
        body=body,
        detail=detail,
    )


def error_status(error: Exception) -> StatusMessage:
    detail = f"{type(error).__name__}: {error}"
    if isinstance(error, TranslationContractError):
        return _error(
            "BẢN DỊCH LÀM HỎNG CẤU TRÚC",
            "Provider đã thay đổi placeholder, tag, newline hoặc cấu trúc bất biến. "
            "Tool đã dừng trước khi ghi output.",
            detail,
        )
    if isinstance(error, TranslationCheckpointError):
        return _error(
            "CHECKPOINT KHÔNG KHỚP",
            "Checkpoint không thuộc đúng JSON nguồn hoặc cặp ngôn ngữ đang chọn. "
            "Hãy giữ file để kiểm tra hoặc chọn output mới.",
            detail,
        )
    if isinstance(error, RetryableTranslationProviderError):
        return _error(
            "KHÔNG KẾT NỐI ĐƯỢC DỊCH VỤ DỊCH",
            "Provider tạm thời không phản hồi sau các lần thử có giới hạn. "
            "Checkpoint đã lưu các lượt hoàn tất; bạn có thể thử lại hoặc chọn "
            "dịch vụ khác rồi tiếp tục.",
            detail,
        )
    if isinstance(error, TranslationProviderError):
        return _error(
            "DỊCH VỤ DỊCH TRẢ VỀ DỮ LIỆU LỖI",
            "Provider từ chối request hoặc trả về kết quả không đúng hợp đồng. "
            "Tool đã không chấp nhận batch lỗi.",
            detail,
        )
    if isinstance(error, UninstallationError):
        backup_detail = (
            f"{detail}\nSafety backup:\n{error.safety_backup_directory}"
        )
        if error.rolled_back:
            return _error(
                "GỠ CÀI ĐẶT THẤT BẠI — ĐÃ HOÀN TÁC",
                "Có lỗi trong khi gỡ, nhưng các package và metadata trước khi gỡ đã "
                "được phục hồi an toàn.",
                backup_detail,
            )
        return _error(
            "GỠ VÀ HOÀN TÁC THẤT BẠI",
            "Tool không thể phục hồi đầy đủ sau lỗi gỡ cài đặt. Không chạy game; "
            "hãy phục hồi thủ công ba file từ safety backup trong Chi tiết.",
            backup_detail,
        )
    if isinstance(error, InstallationError):
        backup_detail = f"{detail}\nBackup:\n{error.backup_directory}"
        if error.rolled_back:
            return _error(
                "CÀI ĐẶT THẤT BẠI — ĐÃ HOÀN TÁC",
                "Có lỗi trong khi cài, nhưng toàn bộ package và metadata ban đầu đã "
                "được phục hồi an toàn.",
                backup_detail,
            )
        return _error(
            "CÀI ĐẶT VÀ HOÀN TÁC THẤT BẠI",
            "Tool không thể phục hồi đầy đủ sau lỗi cài đặt. Không chạy game; hãy "
            "phục hồi thủ công ba file từ thư mục backup trong Chi tiết.",
            backup_detail,
        )
    if isinstance(error, BackupNotFoundError):
        return _error(
            "CHƯA CÓ BACKUP ĐỂ GỠ",
            "Không tìm thấy lần Install chưa được phục hồi. Nút Uninstall chỉ gỡ "
            "các gói đã được cài bằng tool này.",
            detail,
        )
    if isinstance(error, BackupFormatError):
        return _error(
            "BACKUP GỠ CÀI ĐẶT KHÔNG HỢP LỆ",
            "Backup Install mới nhất bị thiếu file hoặc không khớp checksum. Tool "
            "đã dừng trước khi thay đổi game.",
            detail,
        )
    if isinstance(error, GamePathError):
        return _error(
            "SAI THƯ MỤC GAME",
            "Hãy chọn thư mục Crimson Desert có meta\\0.papgt và các package "
            "ngôn ngữ/font cần cài.",
            detail,
        )
    if isinstance(error, FontFormatError):
        message = str(error).lower()
        if "not found" in message or "missing" in message:
            return _error(
                "THIẾU FILE FONT",
                "Cần đủ basefont.ttf, font của ngôn ngữ đã chọn và creditfont.ttf.",
                detail,
            )
        if "truetype/opentype" in message:
            return _error(
                "FONT KHÔNG HỢP LỆ",
                "File đã chọn không có header TrueType/OpenType hợp lệ.",
                detail,
            )
        return _error(
            "GÓI FONT 0012 KHÔNG HỢP LỆ",
            "Kiểm tra folder 0012, ngôn ngữ đã chọn và ba entry font cần thiết.",
            detail,
        )
    if isinstance(error, MetadataFormatError):
        return _metadata_error(error, detail)
    if isinstance(error, PermissionError):
        return _error(
            "KHÔNG CÓ QUYỀN TRUY CẬP",
            "Windows từ chối đọc hoặc ghi đường dẫn đã chọn. Hãy chọn thư mục khác "
            "hoặc chạy tool với quyền phù hợp.",
            detail,
        )
    if isinstance(error, FileNotFoundError):
        return _error(
            "KHÔNG TÌM THẤY FILE",
            "File hoặc thư mục đã chọn không còn tồn tại. Hãy kiểm tra lại đường dẫn.",
            detail,
        )
    if isinstance(error, MemoryError):
        return _error(
            "KHÔNG ĐỦ BỘ NHỚ",
            "Hệ thống không đủ RAM để hoàn tất thao tác. Hãy đóng bớt ứng dụng và thử lại.",
            detail,
        )
    if isinstance(error, ArchiveFormatError):
        return _archive_error(error, detail)
    if isinstance(error, PalocFormatError):
        return _paloc_error(error, detail)
    if isinstance(error, OSError):
        if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
            return _error(
                "KHÔNG ĐỦ DUNG LƯỢNG",
                "Ổ đĩa đích không còn đủ chỗ trống. Hãy giải phóng dung lượng hoặc chọn ổ khác.",
                detail,
            )
        return _error(
            "LỖI ĐỌC/GHI FILE",
            "Windows không thể đọc hoặc ghi một file cần thiết. Kiểm tra đường dẫn, "
            "dung lượng và chương trình đang khóa file.",
            detail,
        )
    if isinstance(error, ValueError):
        message = str(error).lower()
        if "trusted translation template is invalid" in message:
            return _error(
                "FILE MẪU DỊCH KHÔNG HỢP LỆ",
                "Fix cấu trúc cần file mẫu delta hợp lệ từ bước 1 làm chuẩn để dựng lại file dịch. "
                "Hãy chọn lại đúng file 'chỉ gồm câu cần dịch' rồi kiểm tra cấu trúc lại.",
                detail,
            )
        if "json file does not exist" in message:
            return _error(
                "KHÔNG TÌM THẤY JSON",
                "File JSON để Repack không tồn tại. Hãy chọn lại đúng file JSON.",
                detail,
            )
        if "output must be outside" in message:
            return _error(
                "THƯ MỤC OUTPUT KHÔNG HỢP LỆ",
                "Output phải nằm bên ngoài thư mục 0020 nguồn để bảo vệ dữ liệu gốc.",
                detail,
            )
        if "output must be separate" in message:
            return _error(
                "FILE ĐẦU RA BỊ TRÙNG",
                "File đầu ra phải khác toàn bộ file đầu vào. Hãy chọn một đường dẫn "
                "khác để tránh ghi đè dữ liệu nguồn.",
                detail,
            )
        if "please select" in message and "output" in message:
            return _error(
                "CHƯA CHỌN NƠI LƯU",
                "Hãy chọn một thư mục cha để tool tạo output_unpack hoặc output_repack.",
                detail,
            )
        return _error(
            "DỮ LIỆU KHÔNG HỢP LỆ",
            "Giá trị đầu vào không đúng định dạng mà tool yêu cầu.",
            detail,
        )
    return _error(
        "LỖI KHÔNG XÁC ĐỊNH",
        "Tool gặp lỗi ngoài các trường hợp đã biết. Hãy giữ lại phần Chi tiết để kiểm tra.",
        detail,
    )


def _archive_error(error: ArchiveFormatError, detail: str) -> StatusMessage:
    message = str(error).lower()
    if "must contain both" in message or "missing paz file" in message:
        return _error(
            "THIẾU FILE NGUỒN",
            "Package localization nguồn phải chứa đầy đủ cả 0.pamt và 0.paz.",
            detail,
        )
    if "please select a folder named" in message:
        return _error(
            "CHỌN SAI THƯ MỤC NGUỒN",
            "Hãy chọn đúng package của ngôn ngữ hiện tại có chứa 0.pamt và 0.paz.",
            detail,
        )
    if "cannot read 0.pamt" in message:
        return _error(
            "KHÔNG ĐỌC ĐƯỢC 0.PAMT",
            "File 0.pamt có thể bị khóa, hỏng hoặc không còn tồn tại.",
            detail,
        )
    if "cannot decompress" in message:
        return _error(
            "KHÔNG THỂ GIẢI NÉN PALOC",
            "Dữ liệu trong 0.paz không giải nén được hoặc không đúng định dạng hỗ trợ.",
            detail,
        )
    if any(token in message for token in ("checksum", "verification", "does not match")):
        return _error(
            "XÁC MINH PACKAGE THẤT BẠI",
            "Package vừa tạo không vượt qua kiểm tra kích thước hoặc checksum nên không được bàn giao.",
            detail,
        )
    return _error(
        "PACKAGE LOCALIZATION KHÔNG ĐƯỢC HỖ TRỢ",
        "Cấu trúc 0.pamt/0.paz không khớp định dạng localization mà tool hỗ trợ.",
        detail,
    )


def _metadata_error(error: MetadataFormatError, detail: str) -> StatusMessage:
    message = str(error).lower()
    if message.startswith("source"):
        return _error(
            "PACKAGE NGUỒN KHÔNG HỢP LỆ",
            "PAMT hoặc PAZ được chọn để cài không vượt qua kiểm tra checksum "
            "và kích thước.",
            detail,
        )
    if "exactly one" in message and "entry" in message:
        return _error(
            "KHÔNG TÌM THẤY MỤC PACKAGE",
            "Metadata hiện tại của game không có đúng một mục cho package cần xử "
            "lý nên tool dừng trước khi thay đổi file.",
            detail,
        )
    return _error(
        "METADATA GAME KHÔNG HỢP LỆ",
        "File meta\\0.papgt đang hỏng, bị cắt ngắn hoặc có checksum không khớp. "
        "Tool đã dừng trước khi cài.",
        detail,
    )


def _paloc_error(error: PalocFormatError, detail: str) -> StatusMessage:
    message = str(error).lower()
    cause = error.__cause__
    if "translation" in message:
        return _error(
            "TRANSLATION KHÔNG HỢP LỆ",
            "Trường translation được báo lỗi phải tồn tại và chứa văn bản.",
            detail,
        )
    if isinstance(cause, json.JSONDecodeError):
        return _error(
            "JSON SAI CÚ PHÁP",
            "JSON có dấu phẩy, dấu ngoặc hoặc dấu nháy không hợp lệ. Hãy kiểm tra vị trí trong Chi tiết.",
            detail,
        )
    if "cannot read localization json" in message:
        return _error(
            "KHÔNG ĐỌC ĐƯỢC JSON",
            "File JSON có thể bị khóa, sai UTF-8 hoặc không còn tồn tại.",
            detail,
        )
    if "utf-8" in message:
        return _error(
            "ENCODING KHÔNG HỢP LỆ",
            "Nội dung phải sử dụng UTF-8 hợp lệ.",
            detail,
        )
    return _error(
        "CẤU TRÚC JSON/PALOC KHÔNG HỢP LỆ",
        "Không thay đổi marker, key, metadata hoặc cấu trúc entries; chỉ chỉnh trường translation.",
        detail,
    )


def _error(title: str, body: str, detail: str) -> StatusMessage:
    return StatusMessage(
        level="error",
        icon="✕",
        title=title,
        body=body,
        detail=detail,
    )
