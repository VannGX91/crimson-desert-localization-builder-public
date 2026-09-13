import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from crimson0020.auto_translate import AutoTranslateProgress
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.service import (
    CatalogRegistration,
    KnowledgeService,
    ScopeIdentifier,
)
from crimson0020.translation_ui_service import (
    TRANSLATION_PROVIDER_KEYS,
    TranslationUIService,
)


def _delta(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "crimson.translation-delta",
                "schema_version": 1,
                "kind": "sparse-delta",
                "source": {"catalog_fingerprint": "A" * 64},
                "entries": [
                    {
                        "marker": "0011223344556677",
                        "key": "ui.hello",
                        "original": "Hello",
                        "translation": "",
                        "translation_state": "pending",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class TranslationUIServiceTests(TestCase):
    def test_default_scope_matches_runtime_0020_knowledge_scope(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            database = KnowledgeDatabase(base / "knowledge.sqlite3")
            service = TranslationUIService(base, knowledge_database=database)

            self.assertEqual("Crimson0020_0020_en_vi", service._scope.scope_id)

    def test_provider_boundary_and_snapshot_are_serializable_and_immutable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            service = TranslationUIService(Path(temp_dir))

            first = service.refresh()
            self.assertTrue(first["ok"])
            self.assertEqual("ai_router_free", first["state"]["provider"])
            self.assertEqual(
                TRANSLATION_PROVIDER_KEYS,
                tuple(item["key"] for item in first["state"]["providers"]),
            )
            self.assertEqual(
                {"base_url": "", "model": "", "api_key_configured": False},
                first["state"]["configuration"],
            )
            self.assertNotIn("session-secret", json.dumps(first, ensure_ascii=False))
            json.dumps(first, ensure_ascii=False)

            first["state"]["providers"].clear()
            second = service.refresh()

        self.assertEqual(7, len(second["state"]["providers"]))

    def test_user_requested_provider_families_are_exposed(self) -> None:
        self.assertEqual(
            (
                "ai_router_free",
                "google_free",
                "bing-web-free",
                "alibaba-web-free",
                "modernmt-web-free",
                "youdao-web-free",
                "openai-compatible",
            ),
            TRANSLATION_PROVIDER_KEYS,
        )

    def test_custom_openai_config_persists_nonsecret_settings_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom_openai_config.json"
            service = TranslationUIService(Path(temp_dir))
            self.assertTrue(service.set_provider("openai-compatible")["ok"])
            configured = service.set_configuration(
                {
                    "base_url": "https://gateway.example.test/v1",
                    "model": "synthetic-model",
                    "api_key": "session-secret",
                    "max_batch_items": 96,
                    "max_batch_characters": 24000,
                    "concurrency": 3,
                    "timeout_seconds": 180,
                }
            )
            self.assertTrue(configured["ok"])
            snapshot = service.refresh()

            self.assertEqual(
                {
                    "base_url": "https://gateway.example.test/v1",
                    "model": "synthetic-model",
                    "api_key_configured": True,
                    "max_batch_items": 96,
                    "max_batch_characters": 24000,
                    "concurrency": 3,
                    "timeout_seconds": 180.0,
                    "config_path": str(config_path.resolve()),
                },
                snapshot["state"]["configuration"],
            )
            self.assertNotIn("session-secret", json.dumps(snapshot, ensure_ascii=False))
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertNotIn("api_key", persisted)
            self.assertNotIn("session-secret", config_path.read_text(encoding="utf-8"))
            self.assertEqual(96, persisted["max_batch_items"])
            self.assertEqual(3, persisted["concurrency"])

            with patch("crimson0020.translation_ui_service.translate_delta_with_provider") as translate:
                service._default_runner(
                    Path(temp_dir) / "source.json",
                    Path(temp_dir) / "output.json",
                    "openai-compatible",
                    "en",
                    "vi",
                    lambda _progress: None,
                    object(),
                )

            self.assertEqual("openai-compatible", translate.call_args.args[2])
            self.assertEqual(
                {
                    "base_url": "https://gateway.example.test/v1",
                    "model": "synthetic-model",
                    "api_key": "session-secret",
                    "max_batch_items": 96,
                    "max_batch_characters": 24000,
                    "concurrency": 3,
                    "timeout_seconds": 180.0,
                },
                translate.call_args.args[5],
            )
            self.assertEqual(
                service._database.db_path,
                translate.call_args.kwargs["db_path"],
            )

            restarted = TranslationUIService(Path(temp_dir))
            self.assertTrue(restarted.set_provider("openai-compatible")["ok"])
            restarted_config = restarted.refresh()["state"]["configuration"]
            self.assertEqual("synthetic-model", restarted_config["model"])
            self.assertEqual(96, restarted_config["max_batch_items"])
            self.assertEqual(3, restarted_config["concurrency"])
            self.assertFalse(restarted_config["api_key_configured"])

    def test_custom_openai_config_can_persist_api_key_only_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            config_path = base / "custom_openai_config.json"
            service = TranslationUIService(base)
            self.assertTrue(service.set_provider("openai-compatible")["ok"])
            config = {
                "base_url": "https://gateway.example.test/v1",
                "model": "synthetic-model",
                "api_key": "session-secret",
                "max_batch_items": 48,
                "max_batch_characters": 8000,
                "concurrency": 2,
                "timeout_seconds": 90,
            }

            self.assertTrue(
                service.set_configuration(config, persist_api_key=True)["ok"]
            )
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual("session-secret", persisted["api_key"])

            restarted = TranslationUIService(base)
            self.assertTrue(restarted.set_provider("openai-compatible")["ok"])
            self.assertTrue(
                restarted.refresh()["state"]["configuration"]["api_key_configured"]
            )

            self.assertTrue(
                restarted.set_configuration({"api_key": ""}, persist_api_key=False)["ok"]
            )
            persisted_without_key = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertNotIn("api_key", persisted_without_key)

    def test_apply_custom_config_updates_runtime_without_writing_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            service = TranslationUIService(base)
            self.assertTrue(service.set_provider("openai-compatible")["ok"])

            applied = service.apply_configuration(
                {
                    "base_url": "https://gateway.example.test/v1",
                    "model": "runtime-model",
                    "api_key": "runtime-secret",
                    "max_batch_items": 32,
                    "max_batch_characters": 6000,
                    "concurrency": 2,
                    "timeout_seconds": 75,
                }
            )

            self.assertTrue(applied["ok"])
            self.assertFalse((base / "custom_openai_config.json").exists())
            runtime = service._runtime_provider_config("openai-compatible")
            self.assertEqual("runtime-model", runtime["model"])
            self.assertEqual("runtime-secret", runtime["api_key"])

    def test_invalid_provider_page_path_and_state_have_vietnamese_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            service = TranslationUIService(Path(temp_dir))

            responses = (
                service.set_provider("legacy_provider"),
                service.knowledge(page=0),
                service.knowledge(page_size=1000),
                service.select_input(Path(temp_dir) / "missing.json"),
                service.pause(),
            )

        for response in responses:
            self.assertFalse(response["ok"])
            self.assertTrue(response["error"]["summary"])
            self.assertRegex(response["error"]["summary"], "[À-ỹ]")
            self.assertTrue(response["error"]["detail"])

    def test_refresh_is_read_only_and_safe_concurrently(self) -> None:
        with TemporaryDirectory() as temp_dir:
            service = TranslationUIService(Path(temp_dir))
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda _: service.refresh(), range(32)))

        self.assertTrue(all(response["ok"] for response in responses))
        self.assertEqual(1, len({json.dumps(response["state"], sort_keys=True) for response in responses}))

    def test_pause_and_resume_keep_job_state_serializable(self) -> None:
        started = threading.Event()
        calls = 0

        def runner(source, output, provider, source_language, target_language, progress, control):
            nonlocal calls
            calls += 1
            started.set()
            if calls == 1:
                while not control.pause_requested:
                    time.sleep(0.005)
                progress(AutoTranslateProgress(0, 1, 1, 1))
                return None
            progress(AutoTranslateProgress(1, 1, 1, 0))
            return {"output_json": str(output), "provider": provider}

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "input.translation-delta.json"
            output = base / "output.json"
            _delta(source)
            service = TranslationUIService(base, runner=runner)
            self.assertTrue(service.select_input(source)["ok"])
            self.assertTrue(service.select_output(output)["ok"])
            self.assertTrue(service.start()["ok"])
            self.assertTrue(started.wait(1))
            self.assertFalse(service.set_provider("google_free")["ok"])
            self.assertFalse(service.select_input(source)["ok"])
            self.assertFalse(service.select_output(output)["ok"])
            self.assertTrue(service.pause()["ok"])
            deadline = time.monotonic() + 1
            while service.refresh()["state"]["state"] != "paused" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual("paused", service.refresh()["state"]["state"])
            self.assertTrue(service.resume()["ok"])
            deadline = time.monotonic() + 1
            while service.refresh()["state"]["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.01)
            final = service.refresh()

        self.assertEqual("completed", final["state"]["state"])
        json.dumps(final)

    def test_path_delta_language_and_job_failure_validation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            service = TranslationUIService(base)
            self.assertFalse(service.select_input(None)["ok"])
            self.assertFalse(service.select_input(base / "wrong.txt")["ok"])
            self.assertFalse(service.select_output(base / "missing" / "out.json")["ok"])
            self.assertFalse(service.start()["ok"])

            invalid = base / "invalid.json"
            invalid.write_text("{}", encoding="utf-8")
            self.assertTrue(service.select_input(invalid)["ok"])
            self.assertTrue(service.select_output(base / "out.json")["ok"])
            self.assertFalse(service.start(source_language="")["ok"])
            self.assertFalse(service.start(target_language="")["ok"])
            self.assertFalse(service.start()["ok"])

            valid = base / "valid.translation-delta.json"
            _delta(valid)

            def failing_runner(*_args):
                raise RuntimeError("synthetic runner failure")

            failed = TranslationUIService(base, runner=failing_runner)
            self.assertTrue(failed.select_input(valid)["ok"])
            self.assertTrue(failed.select_output(base / "failed.json")["ok"])
            self.assertTrue(failed.start()["ok"])
            deadline = time.monotonic() + 1
            while failed.refresh()["state"]["state"] != "failed" and time.monotonic() < deadline:
                time.sleep(0.01)
            state = failed.refresh()

        self.assertEqual("failed", state["state"]["state"])
        self.assertEqual("job_failed", state["state"]["error"]["code"])

    def test_job_failure_state_and_error_publish_atomically(self) -> None:
        error_entered = threading.Event()
        release_error = threading.Event()

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            valid = base / "valid.translation-delta.json"
            _delta(valid)

            def failing_runner(*_args):
                raise RuntimeError("synthetic runner failure")

            failed = TranslationUIService(base, runner=failing_runner)
            self.assertTrue(failed.select_input(valid)["ok"])
            self.assertTrue(failed.select_output(base / "failed.json")["ok"])
            original_error = failed._error

            def delayed_error(code, summary, detail, **kwargs):
                error_entered.set()
                self.assertTrue(release_error.wait(5))
                return original_error(code, summary, detail, **kwargs)

            with patch.object(failed, "_error", side_effect=delayed_error):
                self.assertTrue(failed.start()["ok"])
                self.assertTrue(error_entered.wait(5))
                in_transition = failed.refresh()["state"]
                release_error.set()
                worker = failed._thread
                self.assertIsNotNone(worker)
                worker.join(5)

            self.assertFalse(worker.is_alive())
            final = failed.refresh()["state"]

        self.assertEqual("running", in_transition["state"])
        self.assertIsNone(in_transition["error"])
        self.assertEqual("failed", final["state"])
        self.assertEqual("job_failed", final["error"]["code"])


    def test_knowledge_pagination_search_and_import_export_are_bounded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            database = KnowledgeDatabase(base / "knowledge.sqlite3")
            knowledge = KnowledgeService(database)
            scope = ScopeIdentifier("crimson0020", "synthetic", "en", "vi")
            lineage = knowledge.register_catalog(
                CatalogRegistration(scope, "B" * 64, 2)
            )
            for index in range(2):
                knowledge.save_translation_segment(
                    scope,
                    lineage,
                    f"{index:016x}",
                    f"ui.item.{index}",
                    f"hash-{index}",
                    f"Synthetic source {index}",
                    f"Bản dịch {index}",
                )
            service = TranslationUIService(base, knowledge_database=database, scope=scope)

            page = service.knowledge(page=1, page_size=1)
            search = service.search_knowledge("Synthetic", page=1, page_size=1)
            package_path = base / "knowledge-export.json"
            exported = service.export_knowledge_file(package_path)
            imported = service.import_knowledge_file(package_path)
            segment_id = page["state"]["items"][0]["segment_id"]
            corrected = service.correct_memory(segment_id, "Bản dịch sửa")
            disabled = service.disable_memory(segment_id)
            glossary = service.correct_glossary("Synthetic", "Tổng hợp")
            glossary_disabled = service.disable_glossary("Synthetic")
            invalid_search = service.search_knowledge(" ")
            invalid_import = service.import_knowledge_file(base / "missing.json")
            invalid_export = service.export_knowledge_file(base / "missing" / "out.json")

        self.assertTrue(page["ok"])
        self.assertEqual(1, len(page["state"]["items"]))
        self.assertEqual(2, page["state"]["total_count"])
        self.assertTrue(search["ok"])
        self.assertLessEqual(len(search["state"]["items"]), 1)
        self.assertTrue(exported["ok"])
        self.assertTrue(imported["ok"])
        self.assertLessEqual(exported["state"]["bytes"], 50 * 1024 * 1024)
        self.assertTrue(corrected["ok"])
        self.assertTrue(disabled["ok"])
        self.assertTrue(glossary["ok"])
        self.assertTrue(glossary_disabled["ok"])
        self.assertFalse(invalid_search["ok"])
        self.assertFalse(invalid_import["ok"])
        self.assertFalse(invalid_export["ok"])

    def test_completed_with_pending_can_be_resumed(self) -> None:
        calls = 0

        def runner(*args):
            nonlocal calls
            calls += 1
            args[5](SimpleNamespace(completed_entries=0, total_entries=1, request_count=1, pending_entries=1))
            return SimpleNamespace(pending_entries=1 if calls == 1 else 0)

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "input.json"
            _delta(source)
            service = TranslationUIService(base, runner=runner)
            self.assertTrue(service.start(source, base / "output.json")["ok"])
            deadline = time.monotonic() + 1
            while service.refresh()["state"]["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual("completed_with_pending", service.refresh()["state"]["state"])
            self.assertTrue(service.resume()["ok"])
            deadline = time.monotonic() + 1
            while service.refresh()["state"]["state"] == "running" and time.monotonic() < deadline:
                time.sleep(0.01)
            final = service.refresh()

        self.assertEqual("completed", final["state"]["state"])
        self.assertEqual(2, calls)

    def test_database_recovery_warning_is_visible_without_losing_service(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "knowledge.sqlite3"
            db_path.write_bytes(b"synthetic damaged sqlite")
            service = TranslationUIService(
                Path(temp_dir),
                knowledge_database=KnowledgeDatabase(db_path),
            )

            state = service.refresh()

        self.assertTrue(state["ok"])
        self.assertEqual("warning", state["state"]["knowledge"]["recovery"]["level"])
        self.assertIn("khôi phục", state["state"]["knowledge"]["recovery"]["summary"].lower())
        self.assertTrue(state["state"]["knowledge"]["recovery"]["detail"])

    def test_database_falls_back_to_local_appdata_when_base_directory_unwritable(self) -> None:
        from tempfile import NamedTemporaryFile
        from crimson0020.translation_runtime.knowledge.database import resolve_default_database_path
        with NamedTemporaryFile() as fake_file:
            # Using a file path as base_directory causes path / 'db.sqlite3' to fail mkdir/create
            service = TranslationUIService(Path(fake_file.name))
            self.assertEqual(
                service._database.db_path.resolve(),
                resolve_default_database_path().resolve(),
            )
