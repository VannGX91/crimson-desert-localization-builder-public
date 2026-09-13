import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .webview_bridge import APPLICATION_VERSION, DialogRequest, WebviewBridge
from .windows_frame import apply_windows_frame_theme


WINDOW_TITLE = f"Crimson Desert Localization Builder v{APPLICATION_VERSION}"
WEBUI_DIRECTORY = Path(__file__).resolve().parent / "webui"
APPLICATION_ICON = (
    Path(__file__).resolve().parent
    / "assets"
    / "crimson-desert-localization.ico"
)


@dataclass(frozen=True)
class WebAssets:
    index: Path
    stylesheet: Path
    javascript: Path


def validate_web_assets(directory: Path = WEBUI_DIRECTORY) -> WebAssets:
    root = directory.resolve()
    assets = WebAssets(
        index=root / "index.html",
        stylesheet=root / "styles.css",
        javascript=root / "app.js",
    )
    for path in (assets.index, assets.stylesheet, assets.javascript):
        if not path.is_file():
            raise FileNotFoundError(f"Required WebView asset is missing: {path.name}")
    return assets


def initial_window_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    width = min(1460, max(720, screen_width - 32))
    height = min(820, max(520, screen_height - 48))
    return min(width, screen_width), min(height, screen_height)


def _screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        return 1280, 800


def _dialog_picker(
    webview_module: ModuleType,
    window: object,
    request: DialogRequest,
) -> str | None:
    dialog_type = (
        webview_module.FileDialog.OPEN
        if request.kind == "file"
        else webview_module.FileDialog.FOLDER
    )
    directory = request.initial_directory or ""
    result = window.create_file_dialog(
        dialog_type,
        directory=directory,
        allow_multiple=False,
        file_types=request.file_types,
    )
    if not result:
        return None
    if isinstance(result, (tuple, list)):
        return str(result[0]) if result else None
    return str(result)


def build_window(
    webview_module: ModuleType,
    bridge: WebviewBridge,
    webui_directory: Path = WEBUI_DIRECTORY,
    size: tuple[int, int] | None = None,
) -> object:
    assets = validate_web_assets(webui_directory)
    width, height = size or initial_window_size(*_screen_size())
    window = webview_module.create_window(
        WINDOW_TITLE,
        url=assets.index.as_uri(),
        js_api=bridge,
        width=width,
        height=height,
        min_size=(720, 520),
        background_color="#0f0a14",
        text_select=True,
    )
    bridge._set_dialog_picker(  # pywebview hides underscore-prefixed bridge methods
        lambda request: _dialog_picker(webview_module, window, request)
    )
    def frame_theme_applier(theme: str) -> bool:
        return apply_windows_frame_theme(window, theme)

    bridge._set_theme_applier(frame_theme_applier)
    window.events.closing += bridge.can_close
    window.events.loaded += lambda *_args: frame_theme_applier(
        str(bridge.get_state()["theme"])
    )
    return window


def run_app(webview_module: ModuleType | None = None) -> None:
    if webview_module is None:
        try:
            import webview as imported_webview
        except ImportError as error:
            raise RuntimeError(
                "Không thể tải pywebview. Hãy giải nén lại đầy đủ folder của tool."
            ) from error
        webview_module = imported_webview
    webview_module.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
    webview_module.settings["REMOTE_DEBUGGING_PORT"] = None
    bridge = WebviewBridge()
    build_window(webview_module, bridge)
    webview_module.start(
        gui="edgechromium",
        debug=False,
        private_mode=True,
        icon=str(APPLICATION_ICON),
    )
