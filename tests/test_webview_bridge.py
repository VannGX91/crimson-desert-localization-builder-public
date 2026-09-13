import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

from crimson0020.auto_translate import AutoTranslateResult
from crimson0020.font_service import FontBuildResult, FontExportResult
from crimson0020.mod_installer import ModInstallResult, ModUninstallResult
from crimson0020.service import BuildResult, ExportResult
from crimson0020.translation_work import TranslationWorkResult
from crimson0020.translation_merge import (
    TranslationDeltaFixResult,
    TranslationDeltaValidationResult,
    TranslationMergeResult,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_ui_service import TranslationUIService
from crimson0020.webview_bridge import BackendServices, WebviewBridge


def _wait_until_idle(bridge: WebviewBridge, timeout: float = 2.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    state = bridge.get_state()
    while state["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
        state = bridge.get_state()
    return state


class WebviewBridgeTests(unittest.TestCase):
    def test_valid_theme_notifies_native_frame_but_invalid_theme_does_not(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(base_directory=Path(temp_dir))
            apply_theme = Mock()
            bridge._set_theme_applier(apply_theme)

            accepted = bridge.set_theme("rose-teal")
            rejected = bridge.set_theme("remote-theme")

        self.assertTrue(accepted["ok"])
        self.assertFalse(rejected["ok"])
        apply_theme.assert_called_once_with("rose-teal")

    def _services(self, export) -> BackendServices:
        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        return BackendServices(
            export_localization=export,
            build_localization=unexpected,
            export_fonts=unexpected,
            build_fonts=unexpected,
            install=unexpected,
            uninstall=unexpected,
            prepare_translation=unexpected,
            auto_translate=unexpected,
        )

    def test_initial_state_contains_workflows_and_portable_defaults(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(
                base_directory=Path(temp_dir),
                clock=lambda: "12:34:56",
            )

            state = bridge.get_state()

        self.assertEqual(
            ["source", "unpack", "repack", "translate", "font", "install", "uninstall"],
            [page["key"] for page in state["pages"]],
        )
        self.assertEqual("source", state["active_page"])
        self.assertEqual("crimson", state["theme"])
        self.assertEqual("English", state["language"])
        self.assertEqual("output_unpack", state["fields"]["unpack_output"])
        self.assertEqual("output_repack", state["fields"]["repack_output"])
        self.assertEqual("", state["fields"]["auto_old"])
        self.assertTrue(
            state["fields"]["auto_new"].endswith("localizationstring_eng.paloc.json")
        )
        self.assertTrue(
            state["fields"]["auto_translate_input"].endswith(
                "localizationstring_eng.paloc.translation-delta.json"
            )
        )
        self.assertFalse(state["translation_work_ready"])
        self.assertFalse(state["translation_input_ready"])
        self.assertFalse(state["translation_merge_input_ready"])
        self.assertFalse(state["translation_structure_valid"])
        self.assertEqual("ai_router_free", state["translation_provider"])
        self.assertEqual(
            [
                "ai_router_free",
                "google_free",
                "bing-web-free",
                "alibaba-web-free",
                "modernmt-web-free",
                "youdao-web-free",
                "openai-compatible",
            ],
            [provider["key"] for provider in state["translation_providers"]],
        )
        self.assertEqual(
            {
                "base_url": "",
                "model": "",
                "api_key_configured": False,
            },
            state["translation_config"],
        )
        json.dumps(state["translation_config"], ensure_ascii=False)
        self.assertFalse(state["busy"])
        self.assertIn("[12:34:56]", state["monitor"][0])

    def test_translation_readiness_tracks_the_explicit_selected_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bridge = WebviewBridge(base_directory=base)
            work = base / "prepared.translation-work.json"

            missing = bridge.set_fields({"auto_translate_input": str(work)})["state"]
            work.write_text('{"metadata":{},"entries":[]}', encoding="utf-8")
            wrong_format = bridge.get_state()
            work.write_text(
                '{"schema":"crimson.translation-delta",'
                '"schema_version":1,"kind":"sparse-delta"}',
                encoding="utf-8",
            )
            ready = bridge.get_state()

        self.assertFalse(missing["translation_work_ready"])
        self.assertFalse(wrong_format["translation_input_ready"])
        self.assertTrue(ready["translation_input_ready"])

    def test_unknown_page_theme_language_and_field_fail_closed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(base_directory=Path(temp_dir))

            responses = (
                bridge.set_page("unknown"),
                bridge.set_theme("remote-theme"),
                bridge.set_language("Klingon"),
                bridge.set_fields({"__class__": "unsafe"}),
            )

        self.assertTrue(all(not response["ok"] for response in responses))
        self.assertTrue(
            all(response["error"]["code"] == "invalid_input" for response in responses)
        )

    def test_language_change_updates_known_package_and_json_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(base_directory=Path(temp_dir))
            bridge.set_fields(
                {
                    "source": r"C:\Game\0020",
                    "json": r"output_unpack\localizationstring_eng.paloc.json",
                    "auto_new": r"output_unpack\localizationstring_eng.paloc.json",
                }
            )

            response = bridge.set_language("Japanese")

        self.assertTrue(response["ok"])
        fields = response["state"]["fields"]
        self.assertTrue(fields["source"].endswith("0021"))
        self.assertTrue(fields["json"].endswith("localizationstring_jpn.paloc.json"))
        self.assertTrue(
            fields["auto_new"].endswith("localizationstring_jpn.paloc.json")
        )
        self.assertTrue(
            fields["auto_work"].endswith(
                "localizationstring_jpn.paloc.translation-delta.json"
            )
        )
        self.assertTrue(
            fields["auto_output"].endswith(
                "localizationstring_jpn.paloc.translation-delta.vi.json"
            )
        )
        self.assertTrue(
            fields["auto_merge_output"].endswith("localizationstring_jpn.paloc.vi.json")
        )

    def test_prepare_translation_compares_old_and_new_then_updates_work_source(
        self,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        def prepare(old, new, work):
            calls.append((old, new, work))
            return TranslationWorkResult(
                work_json=Path(work),
                backup_json=None,
                unchanged_entries=8,
                new_entries=2,
                changed_originals=1,
                removed_entries=3,
                translation_candidates=3,
                technical_entries=1,
            )

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=prepare,
                    auto_translate=unexpected,
                ),
            )
            bridge.set_fields(
                {
                    "auto_old": str(base / "old.json"),
                    "auto_new": str(base / "new.json"),
                    "auto_work": str(base / "work.json"),
                }
            )

            response = bridge.start_operation("prepare_translation")
            state = _wait_until_idle(bridge)

        self.assertTrue(response["ok"])
        self.assertEqual("FILE CẦN DỊCH ĐÃ SẴN SÀNG", state["status"]["title"])
        self.assertIn("File delta nhỏ", state["status"]["body"])
        self.assertEqual(1, len(calls))
        self.assertEqual(r"work.json", state["fields"]["auto_work"])
        self.assertEqual(r"work.json", state["fields"]["auto_translate_input"])

    def test_auto_translate_runs_behind_bridge_and_prepares_repack_json(self) -> None:
        calls: list[tuple[object, ...]] = []

        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        def auto_translate(
            source,
            output,
            provider_key,
            source_language,
            target_language,
            provider_config,
            progress,
            legacy_checkpoint,
            on_commit_stage=None,
            db_path=None,
        ):
            calls.append(
                (
                    source,
                    output,
                    provider_key,
                    source_language,
                    target_language,
                    provider_config,
                    legacy_checkpoint,
                    on_commit_stage,
                    db_path,
                )
            )
            progress(
                type(
                    "Progress",
                    (),
                    {"completed_entries": 1, "total_entries": 2, "request_count": 1},
                )()
            )
            return AutoTranslateResult(
                output_json=Path(output),
                backup_json=None,
                checkpoint_json=Path(str(output) + ".autotranslate-checkpoint.json"),
                translated_entries=2,
                resumed_entries=0,
                skipped_existing_entries=3,
                skipped_empty_entries=1,
                skipped_technical_entries=1,
                reused_duplicate_entries=1,
                request_count=1,
                provider_key=provider_key,
                tm_reused_entries=4,
            )

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=unexpected,
                    auto_translate=auto_translate,
                ),
            )
            bridge.set_fields(
                {
                    "auto_translate_input": str(base / "source.json"),
                    "auto_output": str(base / "translated" / "vi.json"),
                }
            )
            self.assertTrue(bridge.set_translation_provider("google_free")["ok"])
            empty_config = bridge.set_translation_config({})
            self.assertTrue(empty_config["ok"])
            self.assertNotIn("secret-shaped", str(empty_config))

            response = bridge.start_operation("auto_translate")
            state = _wait_until_idle(bridge)

        self.assertTrue(response["ok"])
        self.assertEqual("AUTO TRANSLATE HOÀN TẤT", state["status"]["title"])
        self.assertIn("Dòng dùng lại Translation Memory: 4", state["status"]["body"])
        self.assertEqual("google_free", calls[0][2])
        self.assertEqual("en", calls[0][3])
        self.assertEqual("vi", calls[0][4])
        self.assertEqual({}, calls[0][5])
        self.assertIsNone(calls[0][6])
        self.assertIsNone(calls[0][7])
        self.assertEqual(bridge._translation_ui._database.db_path, calls[0][8])
        self.assertEqual(r"translated\vi.json", state["fields"]["auto_merge_input"])
        self.assertNotEqual(r"translated\vi.json", state["fields"]["json"])

    def test_unknown_translation_provider_fails_closed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(base_directory=Path(temp_dir))

            response = bridge.set_translation_provider("custom-python")

        self.assertFalse(response["ok"])
        self.assertEqual("invalid_input", response["error"]["code"])

    def test_legacy_translation_config_is_inert_and_rejects_nonempty_values_without_echo(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(base_directory=Path(temp_dir))

            empty_object = bridge.set_translation_config({})
            empty_legacy = bridge.set_translation_config(
                {"base_url": "", "model": "", "api_key": ""}
            )
            endpoint = bridge.set_translation_config(
                {"base_url": "https://obsolete.invalid", "model": "", "api_key": ""}
            )
            model = bridge.set_translation_config(
                {"base_url": "", "model": "obsolete-model", "api_key": ""}
            )
            secret = bridge.set_translation_config(
                {"base_url": "", "model": "", "api_key": "secret-shaped"}
            )
            invalid_shape = bridge.set_translation_config(None)

        self.assertTrue(empty_object["ok"])
        self.assertTrue(empty_legacy["ok"])
        self.assertEqual(
            {"base_url": "", "model": "", "api_key_configured": False},
            empty_legacy["state"]["translation_config"],
        )
        for response in (endpoint, model, secret, invalid_shape):
            self.assertFalse(response["ok"])
        self.assertNotIn("https://obsolete.invalid", json.dumps(endpoint, ensure_ascii=False))
        self.assertNotIn("obsolete-model", json.dumps(model, ensure_ascii=False))
        self.assertNotIn("secret-shaped", json.dumps(secret, ensure_ascii=False))

    def test_custom_translation_config_is_accepted_without_echoing_secret(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bridge = WebviewBridge(base_directory=base)
            self.assertTrue(bridge.set_translation_provider("openai-compatible")["ok"])

            response = bridge.set_translation_config(
                {
                    "base_url": "https://gateway.example.test/v1",
                    "model": "synthetic-model",
                    "api_key": "session-secret",
                    "max_batch_items": 72,
                    "max_batch_characters": 16000,
                    "concurrency": 2,
                    "timeout_seconds": 150,
                }
            )
            expected_config_path = str(
                (base / "custom_openai_config.json").resolve()
            )

        self.assertTrue(response["ok"])
        self.assertEqual(
            {
                "base_url": "https://gateway.example.test/v1",
                "model": "synthetic-model",
                "api_key_configured": True,
                "max_batch_items": 72,
                "max_batch_characters": 16000,
                "concurrency": 2,
                "timeout_seconds": 150.0,
                "config_path": expected_config_path,
            },
            response["state"]["translation_config"],
        )
        self.assertNotIn("session-secret", json.dumps(response, ensure_ascii=False))

    def test_custom_config_file_can_be_opened_and_reloaded(self) -> None:
        opened: list[Path] = []
        picked = []
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "configs" / "chosen-custom.json"
            config_path.parent.mkdir()

            def pick(request):
                picked.append(request)
                return str(config_path)

            bridge = WebviewBridge(
                base_directory=base,
                dialog_picker=pick,
                directory_opener=lambda path: opened.append(path),
            )
            self.assertTrue(bridge.set_translation_provider("openai-compatible")["ok"])
            self.assertTrue(bridge.open_translation_config_file()["ok"])
            data = {
                "base_url": "https://gateway.example.test/v1",
                "model": "edited-model",
                "max_batch_items": 128,
                "max_batch_characters": 32000,
                "concurrency": 4,
                "timeout_seconds": 210,
                "api_key": "loaded-secret",
            }
            config_path.write_text(json.dumps(data), encoding="utf-8")
            reloaded = bridge.reload_translation_config_file()
            expected_base = base.resolve()
            expected_config_path = config_path.resolve()

        self.assertEqual([expected_base], opened)
        self.assertEqual("translation_config", picked[0].field)
        self.assertEqual("file", picked[0].kind)
        self.assertEqual(str(expected_base), picked[0].initial_directory)
        self.assertTrue(reloaded["ok"])
        self.assertEqual(str(expected_config_path), reloaded["config_path"])
        config = reloaded["state"]["translation_config"]
        self.assertEqual("edited-model", config["model"])
        self.assertEqual(128, config["max_batch_items"])
        self.assertEqual(32000, config["max_batch_characters"])
        self.assertEqual(4, config["concurrency"])
        self.assertEqual(210.0, config["timeout_seconds"])
        self.assertTrue(config["api_key_configured"])
        self.assertNotIn("loaded-secret", json.dumps(reloaded, ensure_ascii=False))

    def test_custom_start_uses_current_form_config_without_persisting_it(self) -> None:
        started = threading.Event()

        def runner(*_args):
            started.set()
            return SimpleNamespace(pending_entries=0)

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "delta.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": "crimson.translation-delta",
                        "entries": [],
                    }
                ),
                encoding="utf-8",
            )
            service = TranslationUIService(
                base,
                runner=runner,
                knowledge_database=KnowledgeDatabase(base / "test-knowledge.sqlite3"),
            )
            bridge = WebviewBridge(base_directory=base, translation_service=service)
            self.assertTrue(bridge.set_translation_provider("openai-compatible")["ok"])

            response = bridge.start_translation(
                source,
                base / "translated.json",
                {
                    "base_url": "https://gateway.example.test/v1",
                    "model": "screen-model",
                    "api_key": "screen-secret",
                    "max_batch_items": 64,
                    "max_batch_characters": 12000,
                    "concurrency": 3,
                    "timeout_seconds": 100,
                },
            )
            self.assertTrue(started.wait(1))
            runtime = service._runtime_provider_config("openai-compatible")
            config_written = (base / "custom_openai_config.json").exists()

        self.assertTrue(response["ok"])
        self.assertEqual("screen-model", runtime["model"])
        self.assertEqual("screen-secret", runtime["api_key"])
        self.assertEqual(3, runtime["concurrency"])
        self.assertFalse(config_written)

    def test_translation_bridge_delegates_state_knowledge_and_separate_pickers(self) -> None:
        selected = {
            "auto_translate_input": Path("input.json"),
            "auto_output": Path("output.json"),
        }

        def pick(request):
            return str(selected[request.field])

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "input.json").write_text("{}", encoding="utf-8")
            bridge = WebviewBridge(base_directory=base, dialog_picker=pick)
            state = bridge.get_translation_state()
            input_response = bridge.pick_translation_input()
            output_response = bridge.pick_translation_output()
            invalid_page = bridge.get_translation_knowledge(page=0)

        self.assertTrue(state["ok"])
        self.assertEqual("ai_router_free", state["state"]["translation_provider"])
        self.assertTrue(input_response["ok"])
        self.assertTrue(output_response["ok"])
        self.assertFalse(invalid_page["ok"])
        self.assertRegex(invalid_page["error"]["summary"], "[À-ỹ]")

    def test_translation_bridge_exposes_knowledge_and_job_controls_without_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "delta.json"
            source.write_text(
                '{"schema":"crimson.translation-delta","entries":[]}',
                encoding="utf-8",
            )
            bridge = WebviewBridge(
                base_directory=base,
                translation_service=TranslationUIService(
                    base,
                    knowledge_database=KnowledgeDatabase(base / "test-knowledge.sqlite3"),
                ),
            )

            selected_input = bridge.set_translation_input(source)
            selected_output = bridge.set_translation_output(base / "translated.json")
            knowledge = bridge.get_translation_knowledge()
            search = bridge.search_translation_knowledge("synthetic")
            exported = bridge.export_translation_knowledge(base / "knowledge.json")
            imported = bridge.import_translation_knowledge(base / "knowledge.json")
            corrected = bridge.correct_translation_memory("missing", "Bản dịch")
            disabled = bridge.disable_translation_memory("missing")
            glossary = bridge.set_translation_glossary("Synthetic", "Tổng hợp")
            glossary_disabled = bridge.disable_translation_glossary("Synthetic")
            rejected_config = bridge.set_translation_config({"api_key": "secret-shaped"})
            paused = bridge.pause_translation()
            resumed = bridge.resume_translation()

        self.assertTrue(selected_input["ok"])
        self.assertTrue(selected_output["ok"])
        self.assertTrue(knowledge["ok"])
        self.assertTrue(search["ok"])
        self.assertTrue(exported["ok"])
        self.assertTrue(imported["ok"])
        self.assertFalse(corrected["ok"])
        self.assertFalse(disabled["ok"])
        self.assertTrue(glossary["ok"])
        self.assertTrue(glossary_disabled["ok"])
        self.assertFalse(rejected_config["ok"])
        self.assertFalse(paused["ok"])
        self.assertFalse(resumed["ok"])
        self.assertNotIn("secret-shaped", json.dumps(rejected_config, ensure_ascii=False))

    def test_unpack_runs_on_worker_and_preserves_backend_arguments(self) -> None:
        calls: list[tuple[object, ...]] = []
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            def export(source, output, spec):
                calls.append((source, output, spec))
                return ExportResult(
                    json_path=Path(output),
                    backup_json=None,
                    entry_count=39,
                    paloc_size=1024,
                    language=spec.name,
                    paloc_count=2,
                )

            bridge = WebviewBridge(
                base_directory=base,
                services=self._services(export),
                clock=lambda: "01:02:03",
            )
            bridge.set_fields({"source": str(base / "0020")})

            response = bridge.start_operation("unpack")
            state = _wait_until_idle(bridge)
            expected_source = (base / "0020").resolve()
            expected_output = (
                base / "output_unpack" / "localizationstring_eng.paloc.json"
            ).resolve()

        self.assertTrue(response["ok"])
        self.assertFalse(state["busy"])
        self.assertEqual("success", state["status"]["level"])
        self.assertEqual("UNPACK THÀNH CÔNG", state["status"]["title"])
        self.assertEqual(1, len(calls))
        self.assertEqual(expected_source, Path(calls[0][0]).resolve())
        self.assertEqual(expected_output, Path(calls[0][1]).resolve())
        self.assertEqual("English", calls[0][2].name)

    def test_second_operation_and_close_are_rejected_while_busy(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def export(_source, output, spec):
            started.set()
            release.wait(2)
            return ExportResult(
                json_path=Path(output),
                backup_json=None,
                entry_count=1,
                paloc_size=1,
                language=spec.name,
            )

        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(
                base_directory=Path(temp_dir),
                services=self._services(export),
            )
            bridge.set_fields({"source": str(Path(temp_dir) / "0020")})

            first = bridge.start_operation("unpack")
            self.assertTrue(started.wait(1))
            second = bridge.start_operation("unpack")
            can_close = bridge.can_close()
            release.set()
            state = _wait_until_idle(bridge)

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual("busy", second["error"]["code"])
        self.assertFalse(can_close)
        self.assertFalse(state["busy"])

    def test_backend_error_is_returned_as_typed_status(self) -> None:
        def export(*_args):
            raise FileNotFoundError("missing 0.pamt")

        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(
                base_directory=Path(temp_dir),
                services=self._services(export),
            )
            bridge.set_fields({"source": str(Path(temp_dir) / "0020")})

            bridge.start_operation("unpack")
            state = _wait_until_idle(bridge)

        self.assertEqual("error", state["status"]["level"])
        self.assertEqual("KHÔNG TÌM THẤY FILE", state["status"]["title"])
        self.assertIn("FileNotFoundError", state["status"]["detail"])

    def test_every_write_workflow_maps_typed_result_to_web_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            def unexpected_export(*_args):
                raise AssertionError("Unpack is covered by a dedicated test")

            def build_localization(_source, _json, output, spec):
                return BuildResult(
                    output_package=Path(output) / spec.package_folder,
                    backup_package=None,
                    entry_count=10,
                    paloc_size=20,
                    compressed_size=15,
                    paz_size=30,
                    language=spec.name,
                    paloc_count=2,
                    matched_entries=8,
                    new_entries=2,
                )

            def export_font(_source, output, _spec):
                return FontExportResult(Path(output), None, 3, 120)

            def build_font(_source, _input, output, _spec):
                return FontBuildResult(
                    Path(output) / "0012",
                    None,
                    3,
                    "42.paz",
                    450,
                    0x12345678,
                )

            def install(_source, game, spec):
                game_root = Path(game)
                return ModInstallResult(
                    game_root,
                    game_root / "meta" / "0.papgt",
                    game_root / "backup-install",
                    (spec.package_folder,),
                    ((spec.package_folder, 0xABCD),),
                    0x23456789,
                )

            def uninstall(game):
                game_root = Path(game)
                return ModUninstallResult(
                    game_root,
                    game_root / "meta" / "0.papgt",
                    game_root / "backup-install",
                    game_root / "backup-safety",
                    ("0020",),
                    (("0020", 0xBCDE),),
                    0x3456789A,
                )

            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected_export,
                    build_localization=build_localization,
                    export_fonts=export_font,
                    build_fonts=build_font,
                    install=install,
                    uninstall=uninstall,
                ),
            )
            bridge.set_fields(
                {
                    "source": str(base / "0020"),
                    "font_source": str(base / "0012"),
                    "json": str(base / "catalog.json"),
                    "font_repack_input": str(base / "fonts"),
                    "install_source": str(base / "output_repack"),
                    "game": str(base / "game"),
                }
            )

            expected_titles = {
                "repack": "REPACK THÀNH CÔNG",
                "font_unpack": "UNPACK FONT THÀNH CÔNG",
                "font_repack": "REPACK FONT THÀNH CÔNG",
                "install": "CÀI ĐẶT THÀNH CÔNG",
                "uninstall": "GỠ CÀI ĐẶT THÀNH CÔNG",
            }
            for operation, title in expected_titles.items():
                with self.subTest(operation=operation):
                    response = bridge.start_operation(operation)
                    state = _wait_until_idle(bridge)
                    self.assertTrue(response["ok"])
                    self.assertEqual("success", state["status"]["level"])
                    self.assertEqual(title, state["status"]["title"])

    def test_dialog_and_open_path_use_only_allowlisted_fields(self) -> None:
        chosen = Path(r"C:\Selected\0020")
        opened: list[Path] = []
        requests: list[object] = []

        def pick(request):
            requests.append(request)
            return str(chosen)

        with TemporaryDirectory() as temp_dir:
            bridge = WebviewBridge(
                base_directory=Path(temp_dir),
                dialog_picker=pick,
                directory_opener=opened.append,
            )

            selected = bridge.browse("source")
            opened_response = bridge.open_path("source")
            rejected = bridge.open_path("__dict__")

        self.assertTrue(selected["ok"])
        self.assertEqual(str(chosen), selected["state"]["fields"]["source"])
        self.assertTrue(opened_response["ok"])
        self.assertEqual([chosen], opened)
        self.assertEqual("folder", requests[0].kind)
        self.assertFalse(rejected["ok"])

    def test_auto_translate_output_picker_starts_at_parent_and_derives_json_name(
        self,
    ) -> None:
        requests: list[object] = []

        def pick(request):
            requests.append(request)
            return str(base / "chosen")

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bridge = WebviewBridge(base_directory=base, dialog_picker=pick)
            bridge.set_fields(
                {
                    "auto_work": str(base / "source.paloc.translation-work.json"),
                    "auto_translate_input": str(
                        base / "source.paloc.translation-delta.json"
                    ),
                    "auto_output": str(base / "existing" / "old.vi.json"),
                }
            )

            response = bridge.browse("auto_output")
            expected_initial_directory = str((base / "existing").resolve())

        self.assertTrue(response["ok"])
        self.assertEqual("folder", requests[0].kind)
        self.assertEqual(expected_initial_directory, requests[0].initial_directory)
        self.assertEqual(
            r"chosen\source.paloc.translation-delta.vi.json",
            response["state"]["fields"]["auto_output"],
        )

    def test_translation_step_three_validates_fixes_then_merges(self) -> None:
        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        validation_results: list[TranslationDeltaValidationResult] = []
        calls: list[tuple[str, tuple[object, ...]]] = []

        def validate(new_source, template, translated):
            calls.append(("validate", (new_source, template, translated)))
            return validation_results.pop(0)

        def fix(
            template,
            translated,
            checkpoint,
            repaired,
            source_language,
            target_language,
        ):
            calls.append(
                (
                    "fix",
                    (
                        template,
                        translated,
                        checkpoint,
                        repaired,
                        source_language,
                        target_language,
                    ),
                )
            )
            return TranslationDeltaFixResult(
                output_json=Path(repaired),
                entry_count=2,
                repaired_entries=2,
                pending_entries=0,
                issues=("Đã loại một row lạ.",),
            )

        def merge(old_translated, new_source, template, translated, final):
            calls.append(
                (
                    "merge",
                    (old_translated, new_source, template, translated, final),
                )
            )
            return TranslationMergeResult(
                output_json=Path(final),
                backup_json=None,
                merged_entries=2,
                unchanged_entries=8,
                reused_unchanged_entries=8,
                merged_delta_entries=2,
                source_fallback_entries=0,
            )

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            template = base / "delta.json"
            damaged = base / "translated.json"
            final = base / "full.vi.json"
            validation_results.extend(
                [
                    TranslationDeltaValidationResult(
                        valid=False,
                        complete=False,
                        template_json=template,
                        translated_json=damaged,
                        entry_count=2,
                        translated_entries=1,
                        pending_entries=1,
                        issues=("Thiếu một row.",),
                        repairable=True,
                    ),
                    TranslationDeltaValidationResult(
                        valid=True,
                        complete=True,
                        template_json=template,
                        translated_json=base / "translated.fixed.json",
                        entry_count=2,
                        translated_entries=2,
                        pending_entries=0,
                    ),
                ]
            )
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=unexpected,
                    auto_translate=unexpected,
                    validate_translation=validate,
                    fix_translation_structure=fix,
                    merge_translation=merge,
                ),
            )
            bridge.set_fields(
                {
                    "auto_old": str(base / "old.json"),
                    "auto_new": str(base / "new.json"),
                    "auto_work": str(template),
                    "auto_output": str(damaged),
                    "auto_merge_input": str(damaged),
                    "auto_merge_output": str(final),
                }
            )

            self.assertTrue(bridge.start_operation("validate_translation")["ok"])
            invalid = _wait_until_idle(bridge)
            self.assertEqual("CHƯA THỂ GHÉP", invalid["status"]["title"])
            self.assertFalse(invalid["translation_structure_valid"])
            self.assertTrue(invalid["translation_fix_available"])

            self.assertTrue(bridge.start_operation("fix_translation_structure")["ok"])
            fixed = _wait_until_idle(bridge)
            self.assertTrue(
                fixed["fields"]["auto_merge_input"].endswith("translated.fixed.json")
            )
            self.assertEqual(
                fixed["fields"]["auto_output"],
                fixed["fields"]["auto_merge_input"],
            )
            self.assertFalse(fixed["translation_structure_valid"])

            self.assertTrue(bridge.start_operation("validate_translation")["ok"])
            valid = _wait_until_idle(bridge)
            self.assertTrue(valid["translation_structure_valid"])
            self.assertEqual("CẤU TRÚC HỢP LỆ — CÓ THỂ GHÉP", valid["status"]["title"])

            self.assertTrue(bridge.start_operation("merge_translation")["ok"])
            merged = _wait_until_idle(bridge)

        self.assertEqual("GHÉP BẢN DỊCH THÀNH CÔNG", merged["status"]["title"])
        self.assertEqual(r"full.vi.json", merged["fields"]["json"])
        self.assertEqual(
            ["validate", "fix", "validate", "merge"], [x[0] for x in calls]
        )

    def test_pending_translations_require_another_provider_not_structure_fix(
        self,
    ) -> None:
        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            pending = TranslationDeltaValidationResult(
                valid=True,
                complete=False,
                template_json=base / "delta.json",
                translated_json=base / "translated.json",
                entry_count=3,
                translated_entries=1,
                pending_entries=2,
                issues=("Translated delta still has 2 pending/invalid row(s).",),
            )
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=unexpected,
                    auto_translate=unexpected,
                    validate_translation=lambda *_args: pending,
                    fix_translation_structure=unexpected,
                    merge_translation=unexpected,
                ),
            )
            bridge.set_fields(
                {
                    "auto_new": str(base / "new.json"),
                    "auto_work": str(base / "delta.json"),
                    "auto_merge_input": str(base / "translated.json"),
                }
            )

            self.assertTrue(bridge.start_operation("validate_translation")["ok"])
            state = _wait_until_idle(bridge)

        self.assertFalse(state["translation_structure_valid"])
        self.assertFalse(state["translation_fix_available"])
        self.assertEqual(2, state["translation_pending_count"])

    def test_nonrepairable_validation_error_does_not_enable_structure_fix(self) -> None:
        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            invalid = TranslationDeltaValidationResult(
                valid=False,
                complete=False,
                template_json=base / "delta.json",
                translated_json=base / "translated.json",
                entry_count=0,
                translated_entries=0,
                pending_entries=0,
                issues=("Trusted translation template is invalid.",),
                repairable=False,
            )
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=unexpected,
                    auto_translate=unexpected,
                    validate_translation=lambda *_args: invalid,
                    fix_translation_structure=unexpected,
                    merge_translation=unexpected,
                ),
            )
            bridge.set_fields(
                {
                    "auto_new": str(base / "new.json"),
                    "auto_work": str(base / "delta.json"),
                    "auto_merge_input": str(base / "translated.json"),
                }
            )

            self.assertTrue(bridge.start_operation("validate_translation")["ok"])
            state = _wait_until_idle(bridge)

        self.assertFalse(state["translation_structure_valid"])
        self.assertFalse(state["translation_fix_available"])

    def test_fix_structure_uses_selected_merge_file_without_step2_output_dependency(
        self,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        def fix(*args):
            calls.append(args)
            return TranslationDeltaFixResult(
                output_json=Path(args[3]),
                entry_count=2,
                repaired_entries=2,
                pending_entries=0,
            )

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            template = base / "delta.json"
            translated = base / "translated.json"
            translated.write_text("{}", encoding="utf-8")
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=unexpected,
                    auto_translate=unexpected,
                    validate_translation=unexpected,
                    fix_translation_structure=fix,
                    merge_translation=unexpected,
                ),
            )
            bridge.set_fields(
                {
                    "auto_work": str(template),
                    "auto_output": "",
                    "auto_merge_input": str(translated),
                }
            )

            response = bridge.start_operation("fix_translation_structure")
            self.assertTrue(response["ok"])
            state = _wait_until_idle(bridge)

        self.assertEqual(1, len(calls))
        self.assertIsNone(calls[0][2])
        self.assertTrue(state["fields"]["auto_output"].endswith("translated.fixed.json"))
        self.assertEqual(
            state["fields"]["auto_output"],
            state["fields"]["auto_merge_input"],
        )

    def test_fix_structure_checkpoint_follows_selected_merge_file_not_stale_output(
        self,
    ) -> None:
        calls: list[tuple[object, ...]] = []

        def unexpected(*_args, **_kwargs):
            raise AssertionError("Unexpected backend service call")

        def fix(*args):
            calls.append(args)
            return TranslationDeltaFixResult(
                output_json=Path(args[3]),
                entry_count=1,
                repaired_entries=1,
                pending_entries=0,
            )

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            selected = base / "selected.json"
            selected.write_text("{}", encoding="utf-8")
            selected_checkpoint = Path(f"{selected}.autotranslate-checkpoint.jsonl")
            selected_checkpoint.write_text("checkpoint", encoding="utf-8")
            stale = base / "stale.json"
            stale.write_text("{}", encoding="utf-8")
            Path(f"{stale}.autotranslate-checkpoint.jsonl").write_text(
                "stale checkpoint", encoding="utf-8"
            )
            bridge = WebviewBridge(
                base_directory=base,
                services=BackendServices(
                    export_localization=unexpected,
                    build_localization=unexpected,
                    export_fonts=unexpected,
                    build_fonts=unexpected,
                    install=unexpected,
                    uninstall=unexpected,
                    prepare_translation=unexpected,
                    auto_translate=unexpected,
                    validate_translation=unexpected,
                    fix_translation_structure=fix,
                    merge_translation=unexpected,
                ),
            )
            bridge.set_fields(
                {
                    "auto_work": str(base / "delta.json"),
                    "auto_output": str(stale),
                    "auto_merge_input": str(selected),
                }
            )

            self.assertTrue(bridge.start_operation("fix_translation_structure")["ok"])
            _wait_until_idle(bridge)

        self.assertEqual(selected_checkpoint.resolve(), Path(calls[0][2]).resolve())


if __name__ == "__main__":
    unittest.main()
