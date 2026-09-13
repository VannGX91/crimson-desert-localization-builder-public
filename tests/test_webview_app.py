import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from crimson0020.webview_app import (
    APPLICATION_ICON,
    build_window,
    initial_window_size,
    run_app,
    validate_web_assets,
)
from crimson0020.webview_bridge import APPLICATION_VERSION, WebviewBridge


class _Event:
    def __init__(self) -> None:
        self.handlers: list[object] = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


class _Window:
    def __init__(self) -> None:
        self.events = type(
            "Events",
            (),
            {"closing": _Event(), "loaded": _Event()},
        )()

    def create_file_dialog(self, *_args, **_kwargs):
        return None


class _Webview:
    class FileDialog:
        OPEN = 10
        FOLDER = 20

    def __init__(self) -> None:
        self.window = _Window()
        self.create_window = Mock(return_value=self.window)
        self.settings: dict[str, object] = {}
        self.start = Mock()


class WebviewApplicationTests(unittest.TestCase):
    def test_application_uses_a_project_icon_instead_of_python(self) -> None:
        self.assertEqual(
            "crimson-desert-localization.ico",
            APPLICATION_ICON.name,
        )
        self.assertTrue(APPLICATION_ICON.is_file())

    def test_window_size_never_exceeds_small_screen(self) -> None:
        self.assertEqual((1460, 820), initial_window_size(1920, 1080))
        self.assertEqual((1334, 820), initial_window_size(1366, 900))
        self.assertEqual((992, 720), initial_window_size(1024, 768))
        self.assertEqual((736, 552), initial_window_size(768, 600))

    def test_web_assets_require_html_css_and_javascript(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("index.html", "styles.css", "app.js"):
                (root / name).write_text(name, encoding="utf-8")

            assets = validate_web_assets(root)
            (root / "app.js").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "app.js"):
                validate_web_assets(root)

        self.assertEqual((root / "index.html").resolve(), assets.index)

    def test_window_uses_local_assets_and_busy_close_guard(self) -> None:
        fake_webview = _Webview()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("index.html", "styles.css", "app.js"):
                (root / name).write_text(name, encoding="utf-8")
            bridge = WebviewBridge(base_directory=root)

            window = build_window(
                fake_webview,
                bridge,
                webui_directory=root,
                size=(1000, 700),
            )

        kwargs = fake_webview.create_window.call_args.kwargs
        title = fake_webview.create_window.call_args.args[0]
        self.assertEqual(
            f"Crimson Desert Localization Builder v{APPLICATION_VERSION}",
            title,
        )
        self.assertEqual((1000, 700), (kwargs["width"], kwargs["height"]))
        self.assertEqual((720, 520), kwargs["min_size"])
        self.assertTrue(kwargs["url"].startswith("file:///"))
        self.assertIs(bridge, kwargs["js_api"])
        self.assertEqual([bridge.can_close], window.events.closing.handlers)

    def test_window_frame_tracks_initial_and_changed_theme(self) -> None:
        fake_webview = _Webview()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("index.html", "styles.css", "app.js"):
                (root / name).write_text(name, encoding="utf-8")
            bridge = WebviewBridge(base_directory=root)
            with patch(
                "crimson0020.webview_app.apply_windows_frame_theme",
                return_value=True,
            ) as apply_theme:
                window = build_window(
                    fake_webview,
                    bridge,
                    webui_directory=root,
                    size=(1000, 700),
                )
                self.assertEqual(1, len(window.events.loaded.handlers))

                window.events.loaded.handlers[0]()
                response = bridge.set_theme("pearl-light")

        self.assertTrue(response["ok"])
        self.assertEqual(
            [(window, "crimson"), (window, "pearl-light")],
            [call.args for call in apply_theme.call_args_list],
        )

    def test_run_app_requires_edge_webview_and_private_local_session(self) -> None:
        fake_webview = _Webview()

        run_app(fake_webview)

        self.assertFalse(fake_webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"])
        self.assertIsNone(fake_webview.settings["REMOTE_DEBUGGING_PORT"])
        fake_webview.start.assert_called_once_with(
            gui="edgechromium",
            debug=False,
            private_mode=True,
            icon=str(APPLICATION_ICON),
        )


if __name__ == "__main__":
    unittest.main()
