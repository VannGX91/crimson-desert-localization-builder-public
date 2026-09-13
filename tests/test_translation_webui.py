import re
import unittest
from pathlib import Path


class TranslationWebUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.webui = cls.project_root / "crimson0020" / "webui"
        cls.html = (cls.webui / "index.html").read_text(encoding="utf-8")
        cls.javascript = (cls.webui / "app.js").read_text(encoding="utf-8")
        cls.css = (cls.webui / "styles.css").read_text(encoding="utf-8")

    def test_provider_contract_is_exact_and_ai_router_is_default(self) -> None:
        self.assertIn('const DEFAULT_TRANSLATION_PROVIDER = "ai_router_free";', self.javascript)
        for key, label in (
            ("ai_router_free", "AI Router (Free)"),
            ("google_free", "Google-free"),
            ('"bing-web-free"', "Bing-free"),
            ('"alibaba-web-free"', "Alibaba-free"),
            ('"modernmt-web-free"', "ModernMT-free"),
            ('"youdao-web-free"', "YouDao-free"),
            ('"openai-compatible"', "Custom (OpenAI-compatible)"),
        ):
            self.assertIn(f'{key}: "{label}"', self.javascript)
        self.assertEqual(1, self.javascript.count('"AI Router (Free)"'))
        self.assertIn("TRANSLATION_PROVIDER_KEYS.includes(provider.key)", self.javascript)

    def test_current_route_uses_friendly_ui_labels_without_changing_backend_keys(self) -> None:
        self.assertIn('"google-web-free": "google_free"', self.javascript)
        self.assertIn(
            "TRANSLATION_PROVIDER_LABELS[providerKey] || routeKey",
            self.javascript,
        )
        self.assertIn("function translationRouteLabel(routeKey)", self.javascript)
        self.assertIn(
            'translationRouteLabel(job.current_route)',
            self.javascript,
        )

    def test_step_two_has_distinct_picker_and_custom_provider_config(self) -> None:
        step_two = self.html.split('id="translateStepTitle"', 1)[1].split(
            'id="mergeStepTitle"', 1
        )[0]
        self.assertIn('id="field-auto_translate_input"', step_two)
        self.assertIn('data-browse="auto_translate_input"', step_two)
        self.assertNotIn('data-browse="auto_work"', step_two)
        self.assertNotIn("providerConfig", self.html)
        for field_id in (
            "translationBaseUrl",
            "translationModel",
            "translationApiKey",
            "translationConfigSave",
        ):
            self.assertIn(f'id="{field_id}"', self.html)
        self.assertIn("Bing-free", self.javascript)
        self.assertIn("Alibaba-free", self.javascript)
        self.assertIn("ModernMT-free", self.javascript)
        self.assertIn("YouDao-free", self.javascript)
        self.assertIn("Custom (OpenAI-compatible)", self.javascript)
        self.assertIn("pywebview.api.set_translation_config", self.javascript)

    def test_translation_action_is_stateful_and_uses_ui01_bridge_methods(self) -> None:
        for label in ("Start translation", "Resume translation", "Pause safely"):
            self.assertIn(f'"{label}"', self.javascript)
        for method in (
            "pywebview.api.start_translation",
            "pywebview.api.resume_translation",
            "pywebview.api.pause_translation",
        ):
            self.assertIn(method, self.javascript)
        self.assertIn('jobState === "running"', self.javascript)
        self.assertIn('["paused", "completed_with_pending"].includes(jobState)', self.javascript)
        self.assertNotIn('data-operation="auto_translate"', self.html)

    def test_running_translation_keeps_safe_pause_action_enabled(self) -> None:
        render_job = self.javascript.split(
            "function renderTranslationJob(job, ready, busy) {", 1
        )[1].split("\n}", 1)[0]
        self.assertIn(
            'button.disabled = !ready || jobState === "pausing" || (busy && jobState !== "running");',
            render_job,
        )

    def test_progress_surface_exposes_approved_job_state_in_plain_language(self) -> None:
        for element_id in (
            "translationTotal",
            "translationSaved",
            "translationCheckpointReuse",
            "translationTmReuse",
            "translationNewlyTranslated",
            "translationPending",
            "translationCurrentRoute",
            "translationCurrentModel",
            "translationFallbacks",
            "translationCooldown",
            "translationFailureReason",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        for field in (
            "job.total",
            "job.saved",
            "job.resumed",
            "job.translated",
            "job.pending",
            "job.current_route",
            "job.current_model",
            "job.fallbacks",
            "job.cooldown_seconds",
            "job.failure_summary",
        ):
            self.assertIn(field, self.javascript)
        for label in (
            "Tổng cần dịch",
            "Đã lưu",
            "Dùng lại checkpoint",
            "Dùng lại Translation Knowledge",
            "Mới dịch",
            "Còn chờ",
            "Route hiện tại",
            "Model hiện tại",
            "Số lần fallback",
            "Cooldown",
            "Lý do lỗi",
        ):
            self.assertIn(label, self.html)

    def test_fallback_cooldown_failure_and_offline_states_are_visible(self) -> None:
        self.assertIn("renderTranslationJob", self.javascript)
        self.assertIn("renderTranslationOffline", self.javascript)
        self.assertIn("Không thể đọc trạng thái dịch", self.javascript)
        self.assertIn('id="translationConnectionStatus"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn("cooldown_seconds", self.javascript)
        self.assertIn("failure_summary", self.javascript)
        self.assertIn("fallbacks", self.javascript)

    def test_translation_knowledge_panel_is_accessible_paginated_and_editable(self) -> None:
        self.assertIn('id="translationKnowledgePanel"', self.html)
        self.assertIn('id="knowledgeSearch"', self.html)
        self.assertIn('for="knowledgeSearch"', self.html)
        self.assertIn('id="knowledgePrev"', self.html)
        self.assertIn('id="knowledgeNext"', self.html)
        self.assertIn('id="knowledgeResults"', self.html)
        for method in (
            "pywebview.api.get_translation_knowledge",
            "pywebview.api.search_translation_knowledge",
            "pywebview.api.correct_translation_memory",
            "pywebview.api.disable_translation_memory",
            "pywebview.api.import_translation_knowledge",
            "pywebview.api.export_translation_knowledge",
        ):
            self.assertIn(method, self.javascript)
        self.assertIn("knowledgePage", self.javascript)
        self.assertIn("knowledgePageSize", self.javascript)

    def test_translation_knowledge_import_export_uses_accessible_app_dialog(self) -> None:
        self.assertNotIn("window.prompt", self.javascript)
        self.assertIn('id="knowledgePathDialog"', self.html)
        self.assertIn('aria-labelledby="knowledgePathDialogTitle"', self.html)
        self.assertIn('aria-describedby="knowledgePathDialogHelp"', self.html)
        self.assertIn('for="knowledgePathInput"', self.html)
        self.assertIn('id="knowledgePathError"', self.html)
        self.assertIn('role="alert"', self.html)
        self.assertIn("knowledgePathDialog.showModal()", self.javascript)
        self.assertIn('knowledgePathInput.focus()', self.javascript)
        self.assertIn('knowledgePathTrigger.focus()', self.javascript)
        self.assertIn('knowledgePathDialog.addEventListener("cancel"', self.javascript)
        self.assertIn('path.toLowerCase().endsWith(".json")', self.javascript)
        self.assertIn("import_translation_knowledge(path)", self.javascript)
        self.assertIn("export_translation_knowledge(path)", self.javascript)

    def test_state_poll_does_not_clear_paginated_translation_knowledge(self) -> None:
        self.assertIn("hasPaginatedTranslationKnowledge", self.javascript)
        self.assertIn("Array.isArray(knowledge.items)", self.javascript)
        self.assertIn(
            "if (hasPaginatedTranslationKnowledge(state.translation_knowledge))",
            self.javascript,
        )
        self.assertNotIn(
            "renderTranslationKnowledge(state.translation_knowledge);",
            self.javascript,
        )

    def test_busy_idle_poll_restores_translation_knowledge_pagination_boundaries(self) -> None:
        self.assertIn("knowledgeLastPage", self.javascript)
        self.assertIn("syncKnowledgePaginationDisabled", self.javascript)
        set_busy = self.javascript.split("function setBusy(busy) {", 1)[1].split("\n}", 1)[0]
        self.assertIn("syncKnowledgePaginationDisabled(busy);", set_busy)
        render_knowledge = self.javascript.split(
            "function renderTranslationKnowledge(knowledge) {", 1
        )[1].split("\n}", 1)[0]
        self.assertIn("knowledgeLastPage =", render_knowledge)
        self.assertIn("syncKnowledgePaginationDisabled", render_knowledge)

    def test_keyboard_focus_order_keeps_step_two_action_predictable(self) -> None:
        ordered_tokens = (
            'id="field-auto_translate_input"',
            'data-browse="auto_translate_input"',
            'id="translationProvider"',
            'id="field-auto_output"',
            'id="autoTranslateAction"',
            'id="translationKnowledgePanel"',
        )
        positions = [self.html.index(token) for token in ordered_tokens]
        self.assertEqual(sorted(positions), positions)
        self.assertIn('aria-describedby="providerDescription"', self.html)
        self.assertIn('role="status"', self.html)

    def test_long_vietnamese_text_and_narrow_layout_can_wrap_without_overflow(self) -> None:
        self.assertIn("Nếu một phản hồi bị lỗi", self.html)
        self.assertIn("overflow-wrap: anywhere", self.css)
        self.assertIn(".translation-progress", self.css)
        self.assertIn(".knowledge-item", self.css)
        compact = self.css.split("@media (max-width: 900px)", 1)[1]
        self.assertIn(".translation-progress", compact)
        self.assertIn(".knowledge-toolbar", compact)
        self.assertNotRegex(
            self.css,
            re.compile(r"\.translation-(?:progress|knowledge)[^{]*\{[^}]*min-width:\s*[4-9]\d{2}px", re.S),
        )

    def test_dynamic_translation_content_is_rendered_without_html_injection(self) -> None:
        lowered = self.javascript.casefold()
        self.assertNotIn("innerhtml", lowered)
        self.assertNotIn("insertadjacenthtml", lowered)
        self.assertNotIn("eval(", lowered)
        self.assertIn("textContent", self.javascript)
        self.assertIn("replaceChildren", self.javascript)
        self.assertIn("item.source_text", self.javascript)
        self.assertIn("item.target_text", self.javascript)


if __name__ == "__main__":
    unittest.main()
