import re
import unittest
from pathlib import Path

from crimson0020.ui_model import AUTO_TRANSLATE_PAGE


class WebviewUiAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.webui = cls.project_root / "crimson0020" / "webui"
        cls.html = (cls.webui / "index.html").read_text(encoding="utf-8")
        cls.css = (cls.webui / "styles.css").read_text(encoding="utf-8")
        cls.javascript = (cls.webui / "app.js").read_text(encoding="utf-8")

    def test_production_ui_uses_local_vanilla_assets(self) -> None:
        self.assertIn('<link rel="stylesheet" href="styles.css">', self.html)
        self.assertIn('<script src="app.js" defer></script>', self.html)
        self.assertNotIn("<style", self.html.casefold())
        self.assertNotIn("http://", self.html.casefold())
        self.assertNotIn("https://", self.html.casefold())
        self.assertNotIn("fetch(", self.javascript)
        self.assertNotIn("innerHTML", self.javascript)
        self.assertNotIn("eval(", self.javascript)

    def test_all_workflows_and_actions_are_semantically_addressable(self) -> None:
        pages = set(re.findall(r'data-page="([a-z]+)"', self.html))
        operations = set(re.findall(r'data-operation="([a-z_]+)"', self.html))

        self.assertEqual(
            {"source", "unpack", "repack", "translate", "font", "install", "uninstall"},
            pages,
        )
        self.assertEqual(
            {
                "unpack",
                "repack",
                "prepare_translation",
                "validate_translation",
                "fix_translation_structure",
                "merge_translation",
                "font_unpack",
                "font_repack",
                "install",
                "uninstall",
            },
            operations,
        )
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn('aria-label="Điều hướng công cụ"', self.html)
        self.assertIn('id="translationProvider"', self.html)
        self.assertIn('id="translationBaseUrl"', self.html)
        self.assertIn('id="translationModel"', self.html)
        self.assertIn('id="translationApiKey"', self.html)
        self.assertIn('id="translationBatchItems"', self.html)
        self.assertIn('id="translationBatchCharacters"', self.html)
        self.assertIn('id="translationConcurrency"', self.html)
        self.assertIn('id="translationTimeout"', self.html)
        self.assertIn('id="translationConfigOpen"', self.html)
        self.assertIn('id="translationConfigReload"', self.html)
        self.assertIn('id="translationAdvancedConfig"', self.html)
        self.assertIn('class="translation-advanced-card"', self.html)
        self.assertIn('class="translation-advanced-title">Nâng cao</span>', self.html)
        self.assertIn('id="translationSaveApiKey"', self.html)
        self.assertIn("Lưu API key vào file config", self.html)
        self.assertIn(">Mở thư mục config</button>", self.html)
        self.assertIn(">Nạp config…</button>", self.html)
        self.assertIn(">Lưu config</button>", self.html)
        self.assertIn("pywebview.api.set_translation_provider", self.javascript)
        self.assertIn("pywebview.api.set_translation_config", self.javascript)
        self.assertIn("pywebview.api.open_translation_config_file", self.javascript)
        self.assertIn("pywebview.api.reload_translation_config_file", self.javascript)
        self.assertIn("let customConfigDirty = false", self.javascript)
        self.assertIn("if (!customConfigDirty)", self.javascript)
        self.assertIn("readCustomConfigDraft", self.javascript)
        self.assertIn("persistApiKey", self.javascript)
        self.assertIn("job mới sẽ dùng chính các giá trị đang hiển thị này", self.javascript)
        self.assertIn("start_translation(null, null, readCustomConfigDraft())", self.javascript)
        self.assertIn("pywebview.api.start_translation", self.javascript)
        self.assertIn("pywebview.api.resume_translation", self.javascript)
        self.assertIn("pywebview.api.pause_translation", self.javascript)
        self.assertIn("fixButton.disabled = state.busy || !mergeReady", self.javascript)

    def test_auto_translate_page_summary_uses_plain_three_step_language(self) -> None:
        summary = " ".join(
            (
                AUTO_TRANSLATE_PAGE.subtitle,
                AUTO_TRANSLATE_PAGE.meta,
                AUTO_TRANSLATE_PAGE.hint,
            )
        ).casefold()

        self.assertIn("3 bước", summary)
        self.assertIn("file nhỏ", summary)
        self.assertIn("ghép", summary)
        self.assertNotIn("delta", summary)
        self.assertNotIn("merge", summary)
        self.assertNotIn("json công việc theo cấu trúc bản mới", summary)

    def test_responsive_layout_keeps_navigation_available(self) -> None:
        for query in (
            "@media (max-width: 1120px)",
            "@media (max-width: 900px)",
            "@media (max-width: 720px)",
            "@media (max-height: 700px)",
            "@media (prefers-reduced-motion: reduce)",
        ):
            self.assertIn(query, self.css)

        compact = self.css.split("@media (max-width: 900px)", 1)[1]
        compact = compact.split("@media (max-width: 720px)", 1)[0]
        self.assertIn("grid-auto-flow: column", compact)
        self.assertNotIn(".sidebar { display: none", compact)
        self.assertIn("overflow-x: auto", compact)

    def test_app_shell_fills_viewport_with_sixteen_pixel_margin(self) -> None:
        shell = self.css.split(".app-shell {", 1)[1].split("}", 1)[0]

        self.assertIn("width: calc(100vw - 32px)", shell)
        self.assertIn("height: calc(100vh - 32px)", shell)
        self.assertNotIn("min(1460px", shell)
        self.assertNotIn("min(820px", shell)

        compact = self.css.split("@media (max-width: 900px)", 1)[1]
        compact = compact.split("@media (max-width: 720px)", 1)[0]
        self.assertNotIn("body { display: block; padding: 0", compact)
        self.assertNotIn("height: 100vh", compact)

    def test_host_background_restores_the_theme_ambient_field(self) -> None:
        body = self.css.split("\nbody {", 1)[1].split("}", 1)[0]
        ambient = self.css.split(".ambient {", 1)[1].split("}", 1)[0]

        self.assertIn(
            "radial-gradient(850px 600px at -5% -8%, var(--ambient-a)",
            body,
        )
        self.assertIn(
            "radial-gradient(850px 620px at 105% -10%, var(--ambient-b)",
            body,
        )
        self.assertIn("linear-gradient(145deg, var(--bg-a), var(--bg-b))", body)
        self.assertNotIn("background: #fff", body)
        self.assertIn("position: fixed", ambient)
        self.assertNotIn("display: none", ambient)

    def test_status_glyph_is_not_rendered_inside_a_dot(self) -> None:
        self.assertIn('class="state-icon" id="statusIcon"', self.html)
        self.assertNotIn('class="state-dot" id="statusIcon"', self.html)
        self.assertIn(".state-icon {", self.css)

    def test_light_shells_match_the_preview_ambient_palettes(self) -> None:
        self.assertIn(
            "linear-gradient(145deg, #e9e1e8 0%, #e1e5ed 48%, #e7e2ed 100%)",
            self.css,
        )
        self.assertIn(
            "linear-gradient(145deg, #e5f1f2 0%, #eeedf6 49%, #f5eef3 100%)",
            self.css,
        )
        self.assertIn("rgba(85, 204, 197, .34)", self.css)
        self.assertIn("rgba(150, 132, 228, .31)", self.css)

    def test_light_theme_hints_keep_subtle_surface_definition(self) -> None:
        pearl = self.css.split('html[data-theme="pearl-light"] {', 1)[1]
        pearl = pearl.split("}", 1)[0]
        crystal = self.css.split('html[data-theme="crystal-clear"] {', 1)[1]
        crystal = crystal.split("}", 1)[0]
        hint = self.css.split(".hint {", 1)[1].split("}", 1)[0]

        self.assertIn("--hint-surface: rgba(76, 55, 84, .04)", pearl)
        self.assertIn("--hint-border: rgba(76, 55, 84, .12)", pearl)
        self.assertIn("--hint-text: rgba(44, 35, 49, .78)", pearl)
        self.assertIn("--hint-surface: rgba(31, 50, 56, .035)", crystal)
        self.assertIn("--hint-border: rgba(31, 50, 56, .11)", crystal)
        self.assertIn("--hint-text: rgba(31, 50, 56, .78)", crystal)
        self.assertIn("background: var(--hint-surface)", hint)
        self.assertIn("border: 1px solid var(--hint-border)", hint)
        self.assertIn("color: var(--hint-text)", hint)
        self.assertIn('html[data-theme="pearl-light"] .hint {', self.css)
        self.assertIn('html[data-theme="crystal-clear"] .hint {', self.css)

    def test_native_language_options_stay_legible_in_dark_themes(self) -> None:
        self.assertIn(
            'html[data-theme="crimson"] .language-picker option',
            self.css,
        )
        self.assertIn(
            'html[data-theme="rose-teal"] .language-picker option',
            self.css,
        )
        self.assertIn("color: #211827", self.css)
        self.assertIn("background: #f7f4f8", self.css)
        self.assertIn("font-weight: 400", self.css)
        self.assertNotIn("font-weight: 500", self.css)

    def test_javascript_uses_pywebview_ready_and_polling_contract(self) -> None:
        for required in (
            "pywebviewready",
            "pywebview.api.get_state",
            "pywebview.api.start_operation",
            "pywebview.api.browse",
            "pywebview.api.set_language",
            "pywebview.api.set_theme",
            "pywebview.api.set_page",
        ):
            self.assertIn(required, self.javascript)

    def test_auto_translate_explains_sparse_three_step_workflow(self) -> None:
        self.assertIn("File chỉ gồm câu cần dịch", self.html)
        self.assertIn("không chứa các câu không đổi", self.html)
        self.assertIn("Chọn file cần dịch", self.html)
        self.assertIn("Kiểm tra, sửa cấu trúc và ghép", self.html)
        self.assertIn("FIX CẤU TRÚC", self.html)
        self.assertIn("GHÉP VÀO JSON CHÍNH", self.html)
        self.assertNotIn("Bản làm việc đầy đủ", self.html)
        self.assertNotIn("giữ toàn bộ cấu trúc", self.html)
        self.assertNotIn("BẮT ĐẦU / TIẾP TỤC DỊCH", self.html)
        self.assertIn('id="autoTranslateAction"', self.html)
        self.assertIn('id="translationWorkReadiness"', self.html)
        self.assertIn("translation_input_ready", self.javascript)
        self.assertIn("translation_merge_input_ready", self.javascript)
        self.assertIn('jobState === "running"', self.javascript)
        self.assertIn(
            'button.disabled = !ready || jobState === "pausing" || (busy && jobState !== "running")',
            self.javascript,
        )

    def test_theme_menu_has_complete_keyboard_navigation(self) -> None:
        self.assertIn('aria-controls="themeMenu"', self.html)
        for key in ('"ArrowDown"', '"ArrowUp"', '"Home"', '"End"'):
            self.assertIn(key, self.javascript)
        self.assertIn("focusThemeOption", self.javascript)
        self.assertIn("requestAnimationFrame", self.javascript)
        self.assertIn('byId("themeTrigger").focus()', self.javascript)

    def test_native_frame_styling_keeps_the_complete_inner_header(self) -> None:
        titlebar = self.html.split('<header class="titlebar">', 1)[1]
        titlebar = titlebar.split("</header>", 1)[0]

        self.assertIn('class="title-copy"', titlebar)
        self.assertIn('id="versionLabel"', titlebar)
        self.assertIn('id="languageSelect"', titlebar)
        self.assertIn('id="themeTrigger"', titlebar)


if __name__ == "__main__":
    unittest.main()
