import ctypes
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


LOGGER = logging.getLogger(__name__)

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36


@dataclass(frozen=True)
class WindowsFramePalette:
    caption: str
    text: str
    border: str
    dark: bool


WINDOWS_FRAME_PALETTES: Mapping[str, WindowsFramePalette] = MappingProxyType(
    {
        "crimson": WindowsFramePalette("#160d1a", "#f7eef8", "#59304f", True),
        "rose-teal": WindowsFramePalette("#0b1918", "#f7eef8", "#315b57", True),
        "pearl-light": WindowsFramePalette(
            "#e7dfe8", "#2c2331", "#cbbdce", False
        ),
        "crystal-clear": WindowsFramePalette(
            "#e5f1f2", "#1f3238", "#bcd4d8", False
        ),
    }
)


def colorref(value: str) -> int:
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None:
        raise ValueError("Windows frame color must use #RRGGBB format.")
    red = int(value[1:3], 16)
    green = int(value[3:5], 16)
    blue = int(value[5:7], 16)
    return red | (green << 8) | (blue << 16)


def _native_handle(window: object) -> int | None:
    native = getattr(window, "native", None)
    handle = getattr(native, "Handle", None)
    if handle is None:
        return None
    converter = getattr(handle, "ToInt64", None)
    value = converter() if callable(converter) else int(handle)
    return int(value) or None


def _dwm_set_window_attribute(handle: int, attribute: int, value: int) -> int:
    function = ctypes.windll.dwmapi.DwmSetWindowAttribute
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_long
    native_value = ctypes.c_int(value)
    return int(
        function(
            ctypes.c_void_p(handle),
            attribute,
            ctypes.byref(native_value),
            ctypes.sizeof(native_value),
        )
    )


def apply_windows_frame_theme(
    window: object,
    theme: str,
    *,
    setter: Callable[[int, int, int], int] = _dwm_set_window_attribute,
) -> bool:
    try:
        palette = WINDOWS_FRAME_PALETTES[theme]
    except KeyError as error:
        raise ValueError(f"Unknown Windows frame theme: {theme}") from error

    handle = _native_handle(window)
    if handle is None:
        return False

    attributes = (
        (DWMWA_USE_IMMERSIVE_DARK_MODE, int(palette.dark)),
        (DWMWA_BORDER_COLOR, colorref(palette.border)),
        (DWMWA_CAPTION_COLOR, colorref(palette.caption)),
        (DWMWA_TEXT_COLOR, colorref(palette.text)),
    )
    try:
        results = {
            attribute: setter(handle, attribute, value)
            for attribute, value in attributes
        }
    except Exception:
        LOGGER.warning("Windows frame theming is unavailable; using system frame.", exc_info=True)
        return False
    return (
        results[DWMWA_CAPTION_COLOR] == 0
        and results[DWMWA_TEXT_COLOR] == 0
    )
