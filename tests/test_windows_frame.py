import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from crimson0020.windows_frame import (
    DWMWA_BORDER_COLOR,
    DWMWA_CAPTION_COLOR,
    DWMWA_TEXT_COLOR,
    DWMWA_USE_IMMERSIVE_DARK_MODE,
    WINDOWS_FRAME_PALETTES,
    apply_windows_frame_theme,
    colorref,
)


class _Handle:
    def ToInt64(self) -> int:
        return 321


class WindowsFrameTests(unittest.TestCase):
    def test_colorref_converts_html_rgb_to_windows_bgr(self) -> None:
        self.assertEqual(0x563412, colorref("#123456"))
        with self.assertRaisesRegex(ValueError, "RRGGBB"):
            colorref("#123")

    def test_every_web_theme_has_a_native_frame_palette(self) -> None:
        self.assertEqual(
            {"crimson", "rose-teal", "pearl-light", "crystal-clear"},
            set(WINDOWS_FRAME_PALETTES),
        )
        self.assertTrue(WINDOWS_FRAME_PALETTES["crimson"].dark)
        self.assertFalse(WINDOWS_FRAME_PALETTES["pearl-light"].dark)

    def test_theme_applies_native_caption_text_border_and_dark_mode(self) -> None:
        window = SimpleNamespace(native=SimpleNamespace(Handle=_Handle()))
        setter = Mock(return_value=0)
        palette = WINDOWS_FRAME_PALETTES["pearl-light"]

        applied = apply_windows_frame_theme(window, "pearl-light", setter=setter)

        self.assertTrue(applied)
        self.assertEqual(
            [
                (321, DWMWA_USE_IMMERSIVE_DARK_MODE, 0),
                (321, DWMWA_BORDER_COLOR, colorref(palette.border)),
                (321, DWMWA_CAPTION_COLOR, colorref(palette.caption)),
                (321, DWMWA_TEXT_COLOR, colorref(palette.text)),
            ],
            [call.args for call in setter.call_args_list],
        )

    def test_unsupported_native_frame_falls_back_without_crashing(self) -> None:
        setter = Mock(side_effect=OSError("DWM unavailable"))

        self.assertFalse(
            apply_windows_frame_theme(SimpleNamespace(native=None), "crimson")
        )
        with self.assertLogs("crimson0020.windows_frame", level="WARNING"):
            self.assertFalse(
                apply_windows_frame_theme(
                    SimpleNamespace(native=SimpleNamespace(Handle=_Handle())),
                    "crimson",
                    setter=setter,
                )
            )


if __name__ == "__main__":
    unittest.main()
