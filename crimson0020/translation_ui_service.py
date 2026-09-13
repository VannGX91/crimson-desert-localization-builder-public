"""Small, serializable UI boundary for translation jobs and Translation Knowledge."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .auto_translate import AutoTranslateProgress, translate_delta_with_provider
from .translation_runtime.knowledge.database import KnowledgeDatabase, RecoveryResult
from .translation_runtime.knowledge.service import KnowledgeService, ScopeIdentifier
from .translation_runtime.progress import TranslationControl
from .translation_providers import (
    TRANSLATION_PROVIDER_SPECS,
    create_translation_provider,
)


TRANSLATION_PROVIDER_KEYS = (
    "ai_router_free",
    "google_free",
    "bing-web-free",
    "alibaba-web-free",
    "modernmt-web-free",
    "youdao-web-free",
    "openai-compatible",
)
CUSTOM_TRANSLATION_PROVIDER = "openai-compatible"
CUSTOM_TRANSLATION_CONFIG_FILENAME = "custom_openai_config.json"
CUSTOM_TRANSLATION_DEFAULTS: dict[str, object] = {
    "base_url": TRANSLATION_PROVIDER_SPECS[CUSTOM_TRANSLATION_PROVIDER].endpoint,
    "model": "",
    "max_batch_items": 24,
    "max_batch_characters": 4_000,
    "concurrency": 1,
    "timeout_seconds": 120.0,
}
CUSTOM_TRANSLATION_PERSISTED_KEYS = frozenset(CUSTOM_TRANSLATION_DEFAULTS)
TRANSLATION_CONFIG_KEYS = CUSTOM_TRANSLATION_PERSISTED_KEYS | {"api_key"}
CUSTOM_TRANSLATION_FILE_KEYS = TRANSLATION_CONFIG_KEYS
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
MAX_PAGE = 1000
MAX_SEARCH_LENGTH = 200
MAX_PATH_LENGTH = 4096
MAX_IMPORT_BYTES = 50 * 1024 * 1024

Runner = Callable[..., object]


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-shaped snapshot so callers cannot mutate service state."""

    return copy.deepcopy(dict(value))


def _provider_state() -> list[dict[str, object]]:
    providers: list[dict[str, object]] = []
    for key in TRANSLATION_PROVIDER_KEYS:
        spec = TRANSLATION_PROVIDER_SPECS[key]
        providers.append(
            {
                "key": key,
                "name": spec.name,
                "description": spec.description,
                "requires_api_key": spec.requires_api_key,
                "requires_model": spec.requires_model,
                "configurable_endpoint": spec.configurable_endpoint,
            }
        )
    return providers


class TranslationUIService:
    """Owns translation UI state while exposing only JSON-compatible responses."""

    def __init__(
        self,
        base_directory: Path,
        *,
        runner: Runner | None = None,
        knowledge_database: KnowledgeDatabase | None = None,
        scope: ScopeIdentifier | None = None,
    ) -> None:
        self.base_directory = Path(base_directory).expanduser().resolve()
        self._lock = threading.RLock()
        self._provider = TRANSLATION_PROVIDER_KEYS[0]
        self._custom_config_path = self.base_directory / CUSTOM_TRANSLATION_CONFIG_FILENAME
        self._provider_config: dict[str, object] = {
            **CUSTOM_TRANSLATION_DEFAULTS,
            "api_key": "",
        }
        self._load_custom_config_if_present()
        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._runner = runner or self._default_runner
        self._observer: Callable[[AutoTranslateProgress], None] | None = None
        self._control: TranslationControl | None = None
        self._thread: threading.Thread | None = None
        self._job_id: str | None = None
        self._job: dict[str, object] = self._empty_job()
        self._last_error: dict[str, str] | None = None
        self._scope = scope or ScopeIdentifier("Crimson0020", "0020", "en", "vi")
        self._source_language = self._scope.source_language
        self._target_language = self._scope.target_language
        if knowledge_database is not None:
            self._database = knowledge_database
        else:
            self._database = KnowledgeDatabase()
        self._knowledge = KnowledgeService(self._database)
        self._recovery = self._database.last_recovery_result

    @staticmethod
    def _empty_job() -> dict[str, object]:
        return {
            "job_id": None,
            "state": "idle",
            "total": 0,
            "saved": 0,
            "translated": 0,
            "resumed": 0,
            "tm_reused": 0,
            "pending": 0,
            "failures": 0,
            "requests": 0,
            "current_route": None,
            "current_model": None,
            "fallbacks": 0,
            "cooldown_seconds": 0.0,
            "failure_summary": None,
        }

    @property
    def knowledge_service(self) -> KnowledgeService:
        return self._knowledge

    def set_progress_observer(
        self, observer: Callable[[AutoTranslateProgress], None] | None
    ) -> None:
        """Register a UI observer that is notified outside the internal lock."""

        with self._lock:
            self._observer = observer

    def job_state(self) -> str:
        with self._lock:
            return str(self._job["state"])

    def _default_runner(
        self,
        source: Path,
        output: Path,
        provider: str,
        source_language: str,
        target_language: str,
        progress: Callable[[AutoTranslateProgress], None],
        control: TranslationControl,
    ) -> object:
        provider_config = self._runtime_provider_config(provider)
        return translate_delta_with_provider(
            source,
            output,
            provider,
            source_language,
            target_language,
            provider_config,
            progress,
            db_path=self._database.db_path,
            translation_control=control,
        )

    @property
    def custom_config_path(self) -> Path:
        return self._custom_config_path

    @staticmethod
    def _integer_config_value(value: object, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{label} phải là số nguyên.")
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} phải là số nguyên.") from error

    @staticmethod
    def _float_config_value(value: object, label: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{label} phải là số.")
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} phải là số.") from error

    def _normalized_custom_config(
        self,
        values: Mapping[str, object],
        current: Mapping[str, object],
        *,
        validate_runtime: bool,
    ) -> dict[str, object]:
        base_url = str(values.get("base_url", current["base_url"])).strip()
        model = str(values.get("model", current["model"])).strip()
        supplied_key = values.get("api_key", "")
        if not isinstance(supplied_key, str):
            raise ValueError("API key phải là chuỗi.")
        api_key = supplied_key.strip() or str(current.get("api_key", ""))
        candidate: dict[str, object] = {
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "max_batch_items": self._integer_config_value(
                values.get("max_batch_items", current["max_batch_items"]),
                "Số dòng mỗi batch",
            ),
            "max_batch_characters": self._integer_config_value(
                values.get("max_batch_characters", current["max_batch_characters"]),
                "Số ký tự mỗi batch",
            ),
            "concurrency": self._integer_config_value(
                values.get("concurrency", current["concurrency"]),
                "Số request đồng thời",
            ),
            "timeout_seconds": self._float_config_value(
                values.get("timeout_seconds", current["timeout_seconds"]),
                "Timeout",
            ),
        }
        validation_candidate = dict(candidate)
        if not validation_candidate["model"]:
            validation_candidate["model"] = "config-validation-model"
        if not validation_candidate["api_key"]:
            validation_candidate["api_key"] = "config-validation-key"
        create_translation_provider(CUSTOM_TRANSLATION_PROVIDER, validation_candidate)
        if validate_runtime:
            create_translation_provider(CUSTOM_TRANSLATION_PROVIDER, candidate)
        return candidate

    def _persisted_custom_config(
        self,
        config: Mapping[str, object],
        *,
        persist_api_key: bool,
    ) -> dict[str, object]:
        persisted = {key: config[key] for key in CUSTOM_TRANSLATION_PERSISTED_KEYS}
        if persist_api_key:
            persisted["api_key"] = config["api_key"]
        return persisted

    def _write_custom_config_file(
        self,
        config: Mapping[str, object],
        *,
        persist_api_key: bool = False,
    ) -> None:
        self._custom_config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._custom_config_path.with_name(
            f".{self._custom_config_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(
                    self._persisted_custom_config(
                        config,
                        persist_api_key=persist_api_key,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(temporary, self._custom_config_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_custom_config_if_present(self) -> None:
        if not self._custom_config_path.is_file():
            return
        try:
            raw = json.loads(self._custom_config_path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or not set(raw) <= CUSTOM_TRANSLATION_FILE_KEYS:
                raise ValueError("Custom config file contains unsupported fields.")
            self._provider_config = self._normalized_custom_config(
                raw,
                self._provider_config,
                validate_runtime=False,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            # A malformed optional config must not prevent the application from starting.
            return

    def _runtime_provider_config(self, provider: str | None = None) -> dict[str, object]:
        with self._lock:
            selected = provider or self._provider
            if selected != CUSTOM_TRANSLATION_PROVIDER:
                return {}
            return dict(self._provider_config)

    def _configuration_state_locked(self) -> dict[str, object]:
        if self._provider != CUSTOM_TRANSLATION_PROVIDER:
            return {
                "base_url": "",
                "model": "",
                "api_key_configured": False,
            }
        return {
            "base_url": self._provider_config["base_url"],
            "model": self._provider_config["model"],
            "api_key_configured": bool(self._provider_config["api_key"]),
            "max_batch_items": self._provider_config["max_batch_items"],
            "max_batch_characters": self._provider_config["max_batch_characters"],
            "concurrency": self._provider_config["concurrency"],
            "timeout_seconds": self._provider_config["timeout_seconds"],
            "config_path": str(self._custom_config_path),
        }

    def _state_locked(self) -> dict[str, object]:
        recovery = self._recovery_state(self._recovery)
        return {
            "provider": self._provider,
            "providers": _provider_state(),
            "configuration": self._configuration_state_locked(),
            "input_path": str(self._input_path) if self._input_path else "",
            "output_path": str(self._output_path) if self._output_path else "",
            "state": self._job["state"],
            "job": dict(self._job),
            "knowledge": {
                "scope": self._scope.to_dict(),
                "recovery": recovery,
            },
            "error": copy.deepcopy(self._last_error),
        }

    @staticmethod
    def _recovery_state(result: RecoveryResult) -> dict[str, str]:
        if not result.recovered:
            return {
                "level": "ok",
                "summary": "Translation Knowledge đã sẵn sàng.",
                "detail": result.message,
            }
        return {
            "level": "warning",
            "summary": "Translation Knowledge đã được khôi phục; dữ liệu mới vẫn có thể tiếp tục.",
            "detail": result.message,
        }

    def refresh(self) -> dict[str, object]:
        """Read current state without changing paths, job state, or database data."""

        with self._lock:
            return {"ok": True, "state": _copy_json(self._state_locked())}

    get_state = refresh

    def _response(
        self,
        ok: bool,
        *,
        error: dict[str, str] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            state = _copy_json(self._state_locked())
        response: dict[str, object] = {"ok": ok, "state": state}
        if error is not None:
            response["error"] = copy.deepcopy(error)
        if extra:
            response.update(copy.deepcopy(dict(extra)))
        return response

    def _error(
        self,
        code: str,
        summary: str,
        detail: object,
        *,
        job_state: str | None = None,
    ) -> dict[str, object]:
        diagnostic = f"{type(detail).__name__}: {detail}"
        error = {"code": code, "summary": summary, "detail": diagnostic, "message": summary}
        with self._lock:
            if job_state is not None:
                self._job = {**self._job, "state": job_state}
            self._last_error = {key: str(value) for key, value in error.items()}
        return self._response(False, error=error)

    def set_provider(self, key: str) -> dict[str, object]:
        if not isinstance(key, str) or key not in TRANSLATION_PROVIDER_KEYS:
            return self._error("invalid_provider", "Nhà cung cấp bản dịch không hợp lệ.", key)
        with self._lock:
            if self._job["state"] in {"running", "pausing"}:
                return self._error("invalid_state", "Không thể đổi nhà cung cấp khi job đang chạy.", self._job["state"])
            self._provider = key
            self._last_error = None
        return self._response(True)

    def _configuration_candidate(
        self,
        config: object,
        *,
        validate_runtime: bool,
    ) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        if not isinstance(config, Mapping):
            return None, self._error(
                "invalid_config",
                "Cấu hình Custom không hợp lệ.",
                "configuration must be an object",
            )
        keys = set(config)
        if not keys <= TRANSLATION_CONFIG_KEYS:
            return None, self._error(
                "invalid_config",
                "Cấu hình Custom không hợp lệ.",
                "unsupported configuration key",
            )
        with self._lock:
            if self._job["state"] in {"running", "pausing"}:
                return None, self._error(
                    "invalid_state",
                    "Không thể đổi cấu hình khi job đang chạy.",
                    self._job["state"],
                )
            if self._provider != CUSTOM_TRANSLATION_PROVIDER:
                if all(value == "" for value in config.values()):
                    return None, self._response(True)
                return None, self._error(
                    "config_not_supported",
                    "Nhà cung cấp hiện tại không cần cấu hình API key/model.",
                    self._provider,
                )
            current = dict(self._provider_config)

        try:
            candidate = self._normalized_custom_config(
                config,
                current,
                validate_runtime=validate_runtime,
            )
        except ValueError as error:
            return None, self._error(
                "invalid_config",
                "Cấu hình Custom không hợp lệ.",
                error,
            )
        return candidate, None

    def apply_configuration(self, config: object) -> dict[str, object]:
        candidate, error_response = self._configuration_candidate(
            config,
            validate_runtime=True,
        )
        if error_response is not None:
            return error_response
        if candidate is None:
            return self._response(True)
        with self._lock:
            self._provider_config = candidate
            self._last_error = None
        return self._response(True)

    def set_configuration(
        self,
        config: object,
        *,
        persist_api_key: bool = False,
    ) -> dict[str, object]:
        candidate, error_response = self._configuration_candidate(
            config,
            validate_runtime=False,
        )
        if error_response is not None:
            return error_response
        if candidate is None:
            return self._response(True)
        try:
            self._write_custom_config_file(
                candidate,
                persist_api_key=persist_api_key,
            )
        except (OSError, UnicodeError) as error:
            return self._error(
                "invalid_config",
                "Không thể lưu file config Custom.",
                error,
            )
        with self._lock:
            self._provider_config = candidate
            self._last_error = None
        return self._response(True)

    def ensure_configuration_file(self) -> dict[str, object]:
        with self._lock:
            current = dict(self._provider_config)
        try:
            if not self._custom_config_path.is_file():
                self._write_custom_config_file(current)
        except (OSError, UnicodeError) as error:
            return self._error(
                "config_file_error",
                "Không thể tạo file config Custom.",
                error,
            )
        return self._response(
            True,
            extra={"config_path": str(self._custom_config_path)},
        )

    def reload_configuration(self, path: object | None = None) -> dict[str, object]:
        with self._lock:
            if self._job["state"] in {"running", "pausing"}:
                return self._error(
                    "invalid_state",
                    "Không thể nạp lại cấu hình khi job đang chạy.",
                    self._job["state"],
                )
            current = dict(self._provider_config)
            current_path = self._custom_config_path
        selected_path = current_path
        if path is not None:
            try:
                selected_path = self._validate_path(
                    path,
                    must_exist=True,
                    label="File config Custom",
                )
            except (OSError, ValueError) as error:
                return self._error(
                    "invalid_config_file",
                    "File config Custom không hợp lệ.",
                    error,
                )
        if not selected_path.is_file():
            return self._error(
                "config_file_missing",
                "Chưa có file config Custom để nạp lại.",
                selected_path,
            )
        try:
            raw = json.loads(selected_path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or not set(raw) <= CUSTOM_TRANSLATION_FILE_KEYS:
                raise ValueError("Custom config file contains unsupported fields.")
            load_base = {**current, "api_key": ""}
            candidate = self._normalized_custom_config(
                raw,
                load_base,
                validate_runtime=False,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            return self._error(
                "invalid_config_file",
                "File config Custom không hợp lệ.",
                error,
            )
        with self._lock:
            self._custom_config_path = selected_path
            self._provider_config = candidate
            self._last_error = None
        return self._response(True)

    def reject_configuration(self) -> dict[str, object]:
        return self._error(
            "config_not_supported",
            "Nhà cung cấp hiện tại không hỗ trợ cấu hình tùy chỉnh.",
            self._provider,
        )

    def _validate_path(self, value: object, *, must_exist: bool, label: str) -> Path:
        if not isinstance(value, (str, Path)) or isinstance(value, bool):
            raise ValueError(f"{label} phải là đường dẫn hợp lệ.")
        text = str(value).strip()
        if not text or len(text) > MAX_PATH_LENGTH:
            raise ValueError(f"{label} phải là đường dẫn hợp lệ.")
        path = Path(text).expanduser().resolve()
        if path.suffix.casefold() != ".json":
            raise ValueError(f"{label} phải là file JSON.")
        if must_exist and not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {label.lower()}.")
        return path

    def select_input(self, path: object) -> dict[str, object]:
        try:
            selected = self._validate_path(path, must_exist=True, label="File đầu vào")
        except (OSError, ValueError) as error:
            return self._error("invalid_path", "File đầu vào không hợp lệ.", error)
        with self._lock:
            if self._job["state"] in {"running", "pausing"}:
                return self._error("invalid_state", "Không thể đổi file khi job đang chạy.", self._job["state"])
            self._input_path = selected
            self._last_error = None
        return self._response(True)

    def select_output(self, path: object) -> dict[str, object]:
        try:
            selected = self._validate_path(path, must_exist=False, label="File đầu ra")
            if selected.exists() and selected.is_dir():
                raise ValueError("File đầu ra không được là thư mục.")
            if not selected.parent.is_dir():
                raise FileNotFoundError("Thư mục chứa file đầu ra không tồn tại.")
        except (OSError, ValueError) as error:
            return self._error("invalid_path", "File đầu ra không hợp lệ.", error)
        with self._lock:
            if self._job["state"] in {"running", "pausing"}:
                return self._error("invalid_state", "Không thể đổi file khi job đang chạy.", self._job["state"])
            self._output_path = selected
            self._last_error = None
        return self._response(True)

    def _delta_summary(self, path: Path) -> tuple[int, str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("File đầu vào không phải JSON hợp lệ.") from error
        if not isinstance(data, dict) or data.get("schema") != "crimson.translation-delta":
            raise ValueError("Chỉ nhận file sparse translation delta của Step 1.")
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Translation delta thiếu danh sách dòng.")
        source = data.get("source")
        package_family = "0020"
        if isinstance(source, dict):
            raw_package = source.get("package_folder")
            if isinstance(raw_package, str) and raw_package.strip():
                package_family = raw_package.strip()
        return len(entries), package_family

    def _count_delta(self, path: Path) -> int:
        return self._delta_summary(path)[0]

    def start(
        self,
        input_path: object | None = None,
        output_path: object | None = None,
        *,
        source_language: str = "en",
        target_language: str = "vi",
    ) -> dict[str, object]:
        if input_path is not None:
            selected = self.select_input(input_path)
            if not selected["ok"]:
                return selected
        if output_path is not None:
            selected = self.select_output(output_path)
            if not selected["ok"]:
                return selected
        if not isinstance(source_language, str) or not source_language.strip():
            return self._error("invalid_input", "Mã ngôn ngữ nguồn không hợp lệ.", source_language)
        if not isinstance(target_language, str) or not target_language.strip():
            return self._error("invalid_input", "Mã ngôn ngữ đích không hợp lệ.", target_language)
        with self._lock:
            if self._job["state"] in {"running", "pausing"}:
                return self._error("invalid_state", "Một job bản dịch khác đang chạy.", self._job["state"])
            source = self._input_path
            output = self._output_path
            provider = self._provider
            custom_config = dict(self._provider_config)
        if provider == CUSTOM_TRANSLATION_PROVIDER:
            try:
                create_translation_provider(CUSTOM_TRANSLATION_PROVIDER, custom_config)
            except ValueError as error:
                return self._error(
                    "missing_config",
                    "Hãy kiểm tra Base URL, model và API key của Custom trước khi dịch.",
                    error,
                )
        if source is None or output is None:
            return self._error("invalid_path", "Hãy chọn file đầu vào và file đầu ra.", "translation paths missing")
        try:
            total, package_family = self._delta_summary(source)
        except (OSError, ValueError) as error:
            return self._error("invalid_input", "Không thể bắt đầu bản dịch với file này.", error)
        control = TranslationControl()
        job_id = f"translation-{time.time_ns()}"
        with self._lock:
            self._control = control
            self._job_id = job_id
            self._source_language = source_language.strip()
            self._target_language = target_language.strip()
            self._scope = ScopeIdentifier(
                self._scope.app_id,
                package_family,
                self._source_language,
                self._target_language,
            )
            self._job = {
                **self._empty_job(),
                "job_id": job_id,
                "state": "running",
                "total": total,
                "pending": total,
            }
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                args=(source, output, provider, source_language, target_language, control),
                daemon=True,
                name="crimson-translation-ui",
            )
            self._thread.start()
        return self._response(True)

    def _update_progress(self, progress: AutoTranslateProgress) -> None:
        with self._lock:
            completed = max(0, min(progress.completed_entries, int(self._job["total"])))
            pending = max(0, min(progress.pending_entries, int(self._job["total"] - completed)))
            self._job = {
                **self._job,
                "saved": completed,
                "translated": max(
                    int(self._job["translated"]),
                    int(getattr(progress, "translated_entries", 0)),
                ),
                "resumed": max(
                    int(self._job["resumed"]),
                    int(getattr(progress, "resumed_entries", 0)),
                ),
                "tm_reused": max(
                    int(self._job["tm_reused"]),
                    int(getattr(progress, "tm_reused_entries", 0)),
                ),
                "pending": pending,
                "requests": max(int(self._job["requests"]), progress.request_count),
                "current_route": getattr(progress, "current_route", self._job["current_route"]),
                "current_model": getattr(progress, "current_model", self._job["current_model"]),
                "fallbacks": max(
                    int(self._job["fallbacks"]),
                    int(getattr(progress, "fallback_count", self._job["fallbacks"])),
                ),
                "cooldown_seconds": float(
                    getattr(progress, "cooldown_seconds", self._job["cooldown_seconds"])
                ),
                "failure_summary": getattr(
                    progress, "failure_summary", self._job["failure_summary"]
                ),
            }
            observer = self._observer
        if observer is not None:
            observer(progress)

    def _run(
        self,
        source: Path,
        output: Path,
        provider: str,
        source_language: str,
        target_language: str,
        control: TranslationControl,
    ) -> None:
        try:
            result = self._runner(
                source,
                output,
                provider,
                source_language,
                target_language,
                self._update_progress,
                control,
            )
        except Exception as error:
            self._error(
                "job_failed",
                "Bản dịch gặp lỗi; các dòng đã lưu vẫn được giữ lại.",
                error,
                job_state="failed",
            )
            return
        with self._lock:
            paused = control.should_pause_after_request()
            pending = int(getattr(result, "pending_entries", self._job["pending"])) if result is not None else int(self._job["pending"])
            state = "paused" if paused else ("completed_with_pending" if pending else "completed")
            self._job = {
                **self._job,
                "state": state,
                "pending": pending,
                "translated": int(getattr(result, "translated_entries", self._job["translated"])) if result is not None else int(self._job["translated"]),
                "resumed": int(getattr(result, "resumed_entries", self._job["resumed"])) if result is not None else int(self._job["resumed"]),
                "tm_reused": int(getattr(result, "tm_reused_entries", self._job["tm_reused"])) if result is not None else int(self._job["tm_reused"]),
                "failures": int(getattr(result, "failure_count", self._job["failures"])) if result is not None else int(self._job["failures"]),
                "requests": int(getattr(result, "request_count", self._job["requests"])) if result is not None else int(self._job["requests"]),
                "saved": max(
                    int(self._job["saved"]),
                    max(0, int(self._job["total"]) - pending),
                ),
            }

    def pause(self) -> dict[str, object]:
        with self._lock:
            if self._job["state"] != "running" or self._control is None:
                return self._error("invalid_state", "Job hiện tại không thể tạm dừng an toàn.", self._job["state"])
            self._job = {**self._job, "state": "pausing"}
            self._control.request_pause("Người dùng yêu cầu tạm dừng an toàn.")
        return self._response(True)

    def resume(self) -> dict[str, object]:
        with self._lock:
            state = self._job["state"]
        if state not in {"paused", "completed_with_pending"}:
            return self._error("invalid_state", "Job hiện tại chưa ở trạng thái có thể tiếp tục.", state)
        return self.start(
            source_language=self._source_language,
            target_language=self._target_language,
        )

    @staticmethod
    def _validate_page(page: object, page_size: object) -> tuple[int, int]:
        if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= MAX_PAGE:
            raise ValueError("Trang phải nằm trong giới hạn cho phép.")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("Số dòng mỗi trang phải nằm trong giới hạn cho phép.")
        return page, page_size

    @staticmethod
    def _segment_dict(segment: object) -> dict[str, object]:
        return {
            "segment_id": segment.segment_id,
            "lineage_id": segment.lineage_id,
            "marker": segment.marker,
            "key": segment.key,
            "source_hash": segment.source_hash,
            "source_text": segment.source_text,
            "target_text": segment.target_text,
            "context_signature": segment.context_signature,
            "glossary_revision": segment.glossary_revision,
            "status": segment.status,
        }

    def _knowledge_totals(self) -> dict[str, int]:
        page = self._knowledge.list_segments(self._scope, limit=1, offset=0)
        with self._database.connect() as connection:
            terms = connection.execute(
                "SELECT COUNT(*) FROM terms WHERE scope_id = ?", (self._scope.scope_id,)
            ).fetchone()[0]
        return {"segments": page.total_count, "terms": int(terms)}

    def knowledge(self, *, page: object = 1, page_size: object = DEFAULT_PAGE_SIZE) -> dict[str, object]:
        try:
            page_number, size = self._validate_page(page, page_size)
            result = self._knowledge.list_segments(
                self._scope, limit=size, offset=(page_number - 1) * size
            )
            items = tuple(self._segment_dict(item) for item in result.items)
            state = {
                "items": items,
                "page": page_number,
                "page_size": size,
                "total_count": result.total_count,
                "totals": self._knowledge_totals(),
            }
        except Exception as error:
            return self._error("knowledge_error", "Không thể đọc Translation Knowledge.", error)
        response = self._response(True, extra={"knowledge": _copy_json(state)})
        response["state"].update(_copy_json(state))
        response["state"]["knowledge"] = {
            **response["state"]["knowledge"],
            **_copy_json(state),
        }
        return response

    def search_knowledge(
        self, query: object, *, page: object = 1, page_size: object = DEFAULT_PAGE_SIZE
    ) -> dict[str, object]:
        try:
            page_number, size = self._validate_page(page, page_size)
            if not isinstance(query, str) or not query.strip() or len(query) > MAX_SEARCH_LENGTH:
                raise ValueError("Từ khóa tìm kiếm không hợp lệ hoặc quá dài.")
            bounded = min(MAX_PAGE_SIZE * MAX_PAGE, page_number * size)
            matches = tuple(self._knowledge.memory.search_fuzzy(self._scope.scope_id, query, limit=bounded))
            offset = (page_number - 1) * size
            state = {
                "items": tuple(self._segment_dict(item) for item in matches[offset : offset + size]),
                "page": page_number,
                "page_size": size,
                "total_count": len(matches),
                "query": query.strip(),
            }
        except Exception as error:
            return self._error("knowledge_search_error", "Không thể tìm trong Translation Knowledge.", error)
        response = self._response(True, extra={"knowledge": _copy_json(state)})
        response["state"].update(_copy_json(state))
        response["state"]["knowledge"] = {
            **response["state"]["knowledge"],
            **_copy_json(state),
        }
        return response

    def export_knowledge_file(self, path: object) -> dict[str, object]:
        try:
            destination = self._validate_path(path, must_exist=False, label="File xuất")
            if not destination.parent.is_dir():
                raise FileNotFoundError("Thư mục chứa file xuất không tồn tại.")
            package = self._knowledge.export_package(self._scope)
            payload = json.dumps(package, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            temporary = destination.with_name(
                f".{destination.name}.tmp-{threading.get_ident()}-{time.time_ns()}"
            )
            try:
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        except Exception as error:
            return self._error("knowledge_export_error", "Không thể xuất Translation Knowledge.", error)
        export_state = {"path": str(destination), "bytes": len(payload.encode("utf-8"))}
        response = self._response(True, extra={"export": export_state})
        response["state"].update(export_state)
        return response

    def import_knowledge_file(self, path: object) -> dict[str, object]:
        try:
            source = self._validate_path(path, must_exist=True, label="File nhập")
            if source.stat().st_size > MAX_IMPORT_BYTES:
                raise ValueError("File nhập vượt quá giới hạn cho phép.")
            package = json.loads(source.read_text(encoding="utf-8-sig"))
            if not isinstance(package, dict):
                raise ValueError("Gói Translation Knowledge phải là object JSON.")
            self._knowledge.import_package(package)
        except Exception as error:
            return self._error("knowledge_import_error", "Không thể nhập Translation Knowledge.", error)
        return self._response(True, extra={"import": {"path": str(source)}})

    def correct_memory(self, segment_id: object, target_text: object) -> dict[str, object]:
        if not isinstance(segment_id, str) or not segment_id.strip() or not isinstance(target_text, str) or not target_text.strip() or len(target_text) > 10000:
            return self._error("invalid_input", "Bản sửa Translation Memory không hợp lệ.", "segment or target text")
        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    "UPDATE translation_segments SET target_text = ?, updated_at = ? WHERE scope_id = ? AND segment_id = ?",
                    (target_text, time.time(), self._scope.scope_id, segment_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Không tìm thấy dòng Translation Memory.")
        except Exception as error:
            return self._error("knowledge_update_error", "Không thể sửa Translation Memory.", error)
        return self._response(True)

    def disable_memory(self, segment_id: object) -> dict[str, object]:
        if not isinstance(segment_id, str) or not segment_id.strip():
            return self._error("invalid_input", "Dòng Translation Memory không hợp lệ.", segment_id)
        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    "UPDATE translation_segments SET status = 'disabled', updated_at = ? WHERE scope_id = ? AND segment_id = ?",
                    (time.time(), self._scope.scope_id, segment_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Không tìm thấy dòng Translation Memory.")
        except Exception as error:
            return self._error("knowledge_update_error", "Không thể vô hiệu hóa Translation Memory.", error)
        return self._response(True)

    def correct_glossary(self, source_term: object, target_term: object) -> dict[str, object]:
        if not isinstance(source_term, str) or not source_term.strip() or not isinstance(target_term, str) or not target_term.strip():
            return self._error("invalid_input", "Mục từ điển thuật ngữ không hợp lệ.", "glossary term")
        try:
            self._knowledge.set_glossary_override(self._scope, source_term, target_term)
        except Exception as error:
            return self._error("knowledge_update_error", "Không thể cập nhật thuật ngữ.", error)
        return self._response(True)

    def disable_glossary(self, source_term: object) -> dict[str, object]:
        if not isinstance(source_term, str) or not source_term.strip():
            return self._error("invalid_input", "Mục từ điển thuật ngữ không hợp lệ.", source_term)
        try:
            self._knowledge.terminology.disable_term(self._scope.scope_id, source_term)
        except Exception as error:
            return self._error("knowledge_update_error", "Không thể vô hiệu hóa thuật ngữ.", error)
        return self._response(True)
