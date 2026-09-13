from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowPage:
    key: str
    label: str
    glyph: str
    title: str
    subtitle: str
    meta: str
    hint: str


WORKFLOW_PAGES = (
    WorkflowPage(
        key="source",
        label="Dữ liệu nguồn",
        glyph="◉",
        title="Dữ liệu nguồn",
        subtitle="Thiết lập package nguồn dùng xuyên suốt quá trình localization.",
        meta="SOURCE",
        hint="Các công cụ Unpack, Repack và Font sử dụng lại những đường dẫn này.",
    ),
    WorkflowPage(
        key="unpack",
        label="Unpack",
        glyph="⇣",
        title="Unpack",
        subtitle="Giải nén localization package thành JSON phẳng có thể chỉnh sửa.",
        meta="Package → JSON",
        hint="Output được giữ riêng để không ghi đè package nguồn.",
    ),
    WorkflowPage(
        key="repack",
        label="Repack",
        glyph="⟳",
        title="Repack",
        subtitle="Đóng gói JSON đã chỉnh sửa theo cấu trúc package hiện tại.",
        meta="JSON → Package",
        hint="Package đầu ra được xác minh lại trước khi dùng cho bước Install.",
    ),
    WorkflowPage(
        key="font",
        label="Font",
        glyph="A",
        title="Font",
        subtitle="Quản lý riêng package font 0012 để hỗ trợ ký tự tiếng Việt.",
        meta="Package 0012",
        hint="Font được tách khỏi localization để dễ kiểm soát và thay thế.",
    ),
    WorkflowPage(
        key="install",
        label="Install",
        glyph="⇩",
        title="Install",
        subtitle="Cài package đã build vào game với backup và kiểm tra metadata.",
        meta="Mod → Game",
        hint="Tool tạo backup riêng trước khi thay đổi package và metadata game.",
    ),
    WorkflowPage(
        key="uninstall",
        label="Uninstall / Restore",
        glyph="↶",
        title="Uninstall / Restore",
        subtitle="Gỡ bản dịch và phục hồi đúng dữ liệu từ lần Install gần nhất.",
        meta="RESTORE",
        hint="Restore chỉ dùng backup hợp lệ đã được chính tool tạo ra.",
    ),
)

_PAGE_BY_KEY = {page.key: page for page in WORKFLOW_PAGES}

AUTO_TRANSLATE_PAGE = WorkflowPage(
    key="translate",
    label="Auto Translate",
    glyph="文",
    title="Auto Translate",
    subtitle="Quy trình 3 bước: tạo file nhỏ, dịch phần còn thiếu, rồi ghép vào JSON chính.",
    meta="SO SÁNH → DỊCH → GHÉP",
    hint=(
        "File nhỏ chỉ chứa key mới hoặc nội dung đã đổi. JSON đầy đủ chỉ được "
        "tạo sau khi kiểm tra, sửa cấu trúc nếu cần và ghép ở Bước 3."
    ),
)
WEBVIEW_WORKFLOW_PAGES = (
    *WORKFLOW_PAGES[:3],
    AUTO_TRANSLATE_PAGE,
    *WORKFLOW_PAGES[3:],
)
_WEBVIEW_PAGE_BY_KEY = {page.key: page for page in WEBVIEW_WORKFLOW_PAGES}


def page_by_key(key: str) -> WorkflowPage:
    try:
        return _PAGE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unknown workflow page: {key}") from error


def webview_page_by_key(key: str) -> WorkflowPage:
    try:
        return _WEBVIEW_PAGE_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"Unknown WebView workflow page: {key}") from error
