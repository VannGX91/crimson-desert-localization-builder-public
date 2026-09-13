import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from .auto_translate import (
    AutoTranslateProgress,
    AutoTranslateResult,
    checkpoint_path_for,
    source_language_code,
    translate_delta_with_provider,
)
from .font_service import (
    FontBuildResult,
    FontExportResult,
    build_font_package,
    export_fonts,
)
from .gui_paths import (
    application_directory,
    default_font_source,
    default_game,
    default_install_source_directory,
    default_localization_source,
    default_output_directories,
    ensure_named_output_directory,
    font_package_output_path,
    font_unpack_output_path,
    portable_path_text,
    resolve_portable_path,
    unpack_json_output_path,
)
from .languages import LANGUAGE_MAP, LanguageSpec, get_language_spec
from .mod_installer import (
    ModInstallResult,
    ModUninstallResult,
    install_mod,
    uninstall_mod,
)
from .service import (
    BuildResult,
    ExportResult,
    build_localization_from_json,
    export_localization_to_json,
)
from .status_messages import (
    StatusMessage,
    error_status,
    ready_status,
    success_status,
    working_status,
)
from .translation_ui_service import TranslationUIService
from .translation_work import TranslationWorkResult, prepare_translation_work
from .translation_merge import (
    TranslationDeltaFixResult,
    TranslationDeltaValidationResult,
    TranslationMergeResult,
    fix_translation_delta_structure,
    merge_translation_delta,
    validate_translation_delta,
)
from .ui_model import WEBVIEW_WORKFLOW_PAGES, webview_page_by_key


APPLICATION_VERSION = "2.0.0"
LOGGER = logging.getLogger(__name__)
WEB_THEMES = (
    {"key": "crimson", "name": "Crimson Glass", "description": "plum / violet"},
    {
        "key": "rose-teal",
        "name": "Rose–Teal Split",
        "description": "teal glass · rose accent",
    },
    {
        "key": "pearl-light",
        "name": "Pearl Light",
        "description": "light glass · lilac pearl",
    },
    {
        "key": "crystal-clear",
        "name": "Crystal Clear",
        "description": "near-transparent · aurora glass",
    },
)
THEME_KEYS = frozenset(theme["key"] for theme in WEB_THEMES)
FIELD_NAMES = frozenset(
    {
        "source",
        "font_source",
        "json",
        "auto_old",
        "auto_new",
        "auto_work",
        "auto_translate_input",
        "auto_output",
        "auto_merge_input",
        "auto_merge_output",
        "unpack_output",
        "repack_output",
        "font_unpack_output",
        "font_repack_input",
        "font_repack_output",
        "install_source",
        "game",
    }
)
LEGACY_TRANSLATION_CONFIG_KEYS = frozenset({"base_url", "model", "api_key"})
OPERATION_NAMES = frozenset(
    {
        "unpack",
        "repack",
        "prepare_translation",
        "auto_translate",
        "validate_translation",
        "fix_translation_structure",
        "merge_translation",
        "font_unpack",
        "font_repack",
        "install",
        "uninstall",
    }
)


@dataclass(frozen=True)
class DialogRequest:
    field: str
    kind: str
    title: str
    initial_directory: str
    file_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendServices:
    export_localization: Callable[..., object] = export_localization_to_json
    build_localization: Callable[..., object] = build_localization_from_json
    export_fonts: Callable[..., object] = export_fonts
    build_fonts: Callable[..., object] = build_font_package
    install: Callable[..., object] = install_mod
    uninstall: Callable[..., object] = uninstall_mod
    prepare_translation: Callable[..., object] = prepare_translation_work
    auto_translate: Callable[..., object] = translate_delta_with_provider
    validate_translation: Callable[..., object] = validate_translation_delta
    fix_translation_structure: Callable[..., object] = fix_translation_delta_structure
    merge_translation: Callable[..., object] = merge_translation_delta


@dataclass(frozen=True)
class PreparedOperation:
    name: str
    arguments: tuple[object, ...]
    spec: LanguageSpec
    field_updates: Mapping[str, str]


def _default_clock() -> str:
    from datetime import datetime

    return f"{datetime.now():%H:%M:%S}"


def _default_directory_opener(path: Path) -> None:
    if os.name != "nt":
        raise OSError("Opening folders is supported only on Windows.")
    os.startfile(path)  # type: ignore[attr-defined]


def _looks_like_translation_delta(path: Path) -> bool:
    """Cheap UI gate; the backend performs complete schema validation."""

    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            header = stream.read(4096)
    except (OSError, UnicodeError):
        return False
    return all(
        token in header
        for token in (
            '"schema"',
            '"crimson.translation-delta"',
            '"schema_version"',
            '"sparse-delta"',
        )
    )


class WebviewBridge:
    def __init__(
        self,
        base_directory: Path | None = None,
        services: BackendServices | None = None,
        clock: Callable[[], str] | None = None,
        dialog_picker: Callable[[DialogRequest], str | None] | None = None,
        directory_opener: Callable[[Path], None] | None = None,
        translation_service: TranslationUIService | None = None,
    ) -> None:
        self.base_directory = (base_directory or application_directory()).resolve()
        self._services = services or BackendServices()
        self._clock = clock or _default_clock
        self._dialog_picker = dialog_picker
        self._theme_applier: Callable[[str], object] | None = None
        self._directory_opener = directory_opener or _default_directory_opener
        self._lock = threading.RLock()
        self._busy = False
        self._language = "English"
        self._translation_provider = "ai_router_free"
        self._translation_ui = translation_service or TranslationUIService(
            self.base_directory
        )
        self._translation_structure_valid = False
        self._translation_fix_available = False
        self._translation_pending_count = 0
        self._theme = "crimson"
        self._active_page = "source"
        self._status = ready_status(get_language_spec(self._language))
        self._monitor = (f"[{self._clock()}] Hệ thống khởi động hoàn tất.",)
        unpack_output, repack_output = default_output_directories(self.base_directory)
        spec = get_language_spec(self._language)
        auto_work = (
            unpack_output
            / "auto_translate"
            / (f"{Path(spec.json_filename).stem}.translation-delta.json")
        )
        auto_output = (
            unpack_output
            / "auto_translate"
            / (f"{Path(spec.json_filename).stem}.translation-delta.vi.json")
        )
        auto_merge_output = (
            unpack_output
            / "auto_translate"
            / (f"{Path(spec.json_filename).stem}.vi.json")
        )
        fields = {
            "source": default_localization_source(spec),
            "font_source": default_font_source(),
            "json": portable_path_text(
                unpack_output / spec.json_filename,
                self.base_directory,
            ),
            "auto_old": "",
            "auto_new": portable_path_text(
                unpack_output / spec.json_filename,
                self.base_directory,
            ),
            "auto_work": portable_path_text(auto_work, self.base_directory),
            "auto_translate_input": portable_path_text(auto_work, self.base_directory),
            "auto_output": portable_path_text(auto_output, self.base_directory),
            "auto_merge_input": portable_path_text(auto_output, self.base_directory),
            "auto_merge_output": portable_path_text(
                auto_merge_output, self.base_directory
            ),
            "unpack_output": portable_path_text(unpack_output, self.base_directory),
            "repack_output": portable_path_text(repack_output, self.base_directory),
            "font_unpack_output": portable_path_text(
                unpack_output / "fonts",
                self.base_directory,
            ),
            "font_repack_input": portable_path_text(
                unpack_output / "fonts",
                self.base_directory,
            ),
            "font_repack_output": portable_path_text(
                repack_output / "0012",
                self.base_directory,
            ),
            "install_source": portable_path_text(
                default_install_source_directory(self.base_directory),
                self.base_directory,
            ),
            "game": default_game(),
        }
        self._fields: Mapping[str, str] = MappingProxyType(fields)

    def _set_dialog_picker(
        self,
        picker: Callable[[DialogRequest], str | None],
    ) -> None:
        self._dialog_picker = picker

    def _set_theme_applier(self, applier: Callable[[str], object]) -> None:
        self._theme_applier = applier

    def get_state(self) -> dict[str, object]:
        with self._lock:
            translation_ui_state = self._translation_ui.refresh()["state"]
            try:
                translation_input_ready = _looks_like_translation_delta(
                    resolve_portable_path(
                        self._fields["auto_translate_input"],
                        self.base_directory,
                    )
                )
            except (OSError, ValueError):
                translation_input_ready = False
            try:
                translation_merge_input_ready = resolve_portable_path(
                    self._fields["auto_merge_input"],
                    self.base_directory,
                ).is_file()
            except (OSError, ValueError):
                translation_merge_input_ready = False
            return {
                "active_page": self._active_page,
                "busy": self._busy,
                "fields": dict(self._fields),
                "language": self._language,
                "languages": list(LANGUAGE_MAP),
                "monitor": list(self._monitor),
                "pages": [asdict(page) for page in WEBVIEW_WORKFLOW_PAGES],
                "status": asdict(self._status),
                "theme": self._theme,
                "themes": [dict(theme) for theme in WEB_THEMES],
                "translation_provider": translation_ui_state["provider"],
                "translation_providers": translation_ui_state["providers"],
                "translation_config": dict(translation_ui_state["configuration"]),
                "translation_state": translation_ui_state,
                "translation_job": translation_ui_state["job"],
                "translation_knowledge": translation_ui_state["knowledge"],
                "translation_work_ready": translation_input_ready,
                "translation_input_ready": translation_input_ready,
                "translation_merge_input_ready": translation_merge_input_ready,
                "translation_structure_valid": self._translation_structure_valid,
                "translation_fix_available": self._translation_fix_available,
                "translation_pending_count": self._translation_pending_count,
                "version": APPLICATION_VERSION,
            }

    def set_page(self, key: str) -> dict[str, object]:
        try:
            webview_page_by_key(key)
        except (TypeError, ValueError) as error:
            return self._invalid(error)
        with self._lock:
            if self._busy:
                return self._busy_response()
            self._active_page = key
            return self._success_response()

    def set_theme(self, key: str) -> dict[str, object]:
        if not isinstance(key, str) or key not in THEME_KEYS:
            return self._invalid(ValueError(f"Unknown theme: {key}"))
        with self._lock:
            if self._busy:
                return self._busy_response()
            self._theme = key
        if self._theme_applier is not None:
            try:
                self._theme_applier(key)
            except Exception:
                LOGGER.warning(
                    "Native frame theme update failed; the web theme remains active.",
                    exc_info=True,
                )
        return self._success_response()

    def set_language(self, name: str) -> dict[str, object]:
        try:
            selected = get_language_spec(name)
        except (TypeError, ValueError) as error:
            return self._invalid(error)
        with self._lock:
            if self._busy:
                return self._busy_response()
            previous = get_language_spec(self._language)
            fields = dict(self._fields)
            package_names = {values[0] for values in LANGUAGE_MAP.values()}
            source = fields["source"].strip()
            if source and Path(source).name in package_names:
                fields["source"] = str(Path(source).parent / selected.package_folder)
            elif not source:
                fields["source"] = default_localization_source(selected)

            json_names = {f"{values[1]}.json" for values in LANGUAGE_MAP.values()}
            json_path = fields["json"].strip()
            if json_path and Path(json_path).name in json_names:
                resolved = resolve_portable_path(json_path, self.base_directory)
                fields["json"] = portable_path_text(
                    resolved.with_name(selected.json_filename),
                    self.base_directory,
                )
            for auto_field in ("auto_old", "auto_new"):
                auto_json = fields[auto_field].strip()
                if auto_json and Path(auto_json).name in json_names:
                    resolved = resolve_portable_path(auto_json, self.base_directory)
                    fields[auto_field] = portable_path_text(
                        resolved.with_name(selected.json_filename),
                        self.base_directory,
                    )
                elif not auto_json and auto_field == "auto_new":
                    fields[auto_field] = fields["json"]

            work_names = {
                f"{Path(values[1] + '.json').stem}.translation-delta.json"
                for values in LANGUAGE_MAP.values()
            }
            auto_work = fields["auto_work"].strip()
            selected_work_name = (
                f"{Path(selected.json_filename).stem}.translation-delta.json"
            )
            if auto_work and Path(auto_work).name in work_names:
                resolved = resolve_portable_path(auto_work, self.base_directory)
                fields["auto_work"] = portable_path_text(
                    resolved.with_name(selected_work_name), self.base_directory
                )

            auto_translate_input = fields["auto_translate_input"].strip()
            if auto_translate_input and Path(auto_translate_input).name in work_names:
                resolved = resolve_portable_path(
                    auto_translate_input, self.base_directory
                )
                fields["auto_translate_input"] = portable_path_text(
                    resolved.with_name(selected_work_name), self.base_directory
                )

            auto_output_names = {
                f"{Path(values[1] + '.json').stem}.translation-delta.vi.json"
                for values in LANGUAGE_MAP.values()
            }
            auto_output = fields["auto_output"].strip()
            selected_auto_name = (
                f"{Path(selected.json_filename).stem}.translation-delta.vi.json"
            )
            if auto_output and Path(auto_output).name in auto_output_names:
                resolved = resolve_portable_path(auto_output, self.base_directory)
                fields["auto_output"] = portable_path_text(
                    resolved.with_name(selected_auto_name),
                    self.base_directory,
                )
            auto_merge_input = fields["auto_merge_input"].strip()
            if auto_merge_input and Path(auto_merge_input).name in auto_output_names:
                resolved = resolve_portable_path(auto_merge_input, self.base_directory)
                fields["auto_merge_input"] = portable_path_text(
                    resolved.with_name(selected_auto_name), self.base_directory
                )

            merge_output_names = {
                f"{Path(values[1] + '.json').stem}.vi.json"
                for values in LANGUAGE_MAP.values()
            }
            auto_merge_output = fields["auto_merge_output"].strip()
            selected_merge_name = f"{Path(selected.json_filename).stem}.vi.json"
            if auto_merge_output and Path(auto_merge_output).name in merge_output_names:
                resolved = resolve_portable_path(auto_merge_output, self.base_directory)
                fields["auto_merge_output"] = portable_path_text(
                    resolved.with_name(selected_merge_name), self.base_directory
                )
            self._language = selected.name
            self._fields = MappingProxyType(fields)
            self._reset_translation_validation_locked()
            if previous != selected:
                self._status = ready_status(selected)
                self._append_monitor_locked(f"Đã đổi ngôn ngữ nguồn: {selected.name}.")
            return self._success_response()

    def set_translation_provider(self, key: str) -> dict[str, object]:
        with self._lock:
            if self._busy:
                return self._busy_response()
        response = self._translation_ui.set_provider(key)
        if response["ok"]:
            with self._lock:
                self._translation_provider = key
        return self._translation_response(response)

    def set_translation_config(
        self,
        config: object,
        persist_api_key: object = False,
    ) -> dict[str, object]:
        if not isinstance(persist_api_key, bool):
            return self._invalid(ValueError("Tùy chọn lưu API key phải là true/false."))
        return self._translation_response(
            self._translation_ui.set_configuration(
                config,
                persist_api_key=persist_api_key,
            )
        )

    def apply_translation_config(self, config: object) -> dict[str, object]:
        """Apply the current Custom form to runtime without writing a config file."""

        return self._translation_response(self._translation_ui.apply_configuration(config))

    def open_translation_config_file(self) -> dict[str, object]:
        with self._lock:
            if self._busy:
                return self._busy_response()
        response = self._translation_ui.ensure_configuration_file()
        if not response["ok"]:
            return self._translation_response(response)
        try:
            self._directory_opener(self._translation_ui.custom_config_path.parent)
        except Exception as error:
            return self._operation_error(error)
        return self._translation_response(response)

    def reload_translation_config_file(self) -> dict[str, object]:
        with self._lock:
            if self._busy:
                return self._busy_response()
            picker = self._dialog_picker
        if picker is None:
            return self._invalid(RuntimeError("File dialog is not initialized."))
        current_path = self._translation_ui.custom_config_path
        request = DialogRequest(
            field="translation_config",
            kind="file",
            title="Chọn file config Custom",
            initial_directory=str(current_path.parent),
            file_types=("JSON files (*.json)", "All files (*.*)"),
        )
        try:
            selected = picker(request)
        except Exception as error:
            return self._operation_error(error)
        if not selected:
            return {"ok": True, "cancelled": True, "state": self.get_state()}
        response = self._translation_ui.reload_configuration(selected)
        payload = self._translation_response(response)
        if response["ok"]:
            payload["config_path"] = str(self._translation_ui.custom_config_path)
        return payload

    def get_translation_state(self) -> dict[str, object]:
        return self._translation_response(self._translation_ui.refresh())

    def refresh_translation(self) -> dict[str, object]:
        return self.get_translation_state()

    def set_translation_input(self, path: object) -> dict[str, object]:
        with self._lock:
            if self._busy:
                return self._busy_response()
        response = self._translation_ui.select_input(path)
        if response["ok"]:
            response = self.set_fields({"auto_translate_input": str(path)})
        return self._translation_response(response)

    def set_translation_output(self, path: object) -> dict[str, object]:
        with self._lock:
            if self._busy:
                return self._busy_response()
        response = self._translation_ui.select_output(path)
        if response["ok"]:
            response = self.set_fields({"auto_output": str(path)})
        return self._translation_response(response)

    def pick_translation_input(self) -> dict[str, object]:
        return self.browse("auto_translate_input")

    def pick_translation_output(self) -> dict[str, object]:
        return self.browse("auto_output")

    def start_translation(
        self,
        input_path: object | None = None,
        output_path: object | None = None,
        custom_config: object | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self._busy:
                return self._busy_response()
        if custom_config is not None:
            configured = self._translation_ui.apply_configuration(custom_config)
            if not configured["ok"]:
                return self._translation_response(configured)
        if input_path is None:
            input_path = self._fields["auto_translate_input"]
        if output_path is None:
            output_path = self._fields["auto_output"]
        with self._lock:
            source_language = source_language_code(get_language_spec(self._language).name)
        self._translation_ui.set_progress_observer(self._auto_translate_progress)
        response = self._translation_ui.start(
            input_path,
            output_path,
            source_language=source_language,
            target_language="vi",
        )
        return self._track_translation(response)

    def pause_translation(self) -> dict[str, object]:
        return self._translation_response(self._translation_ui.pause())

    def resume_translation(self) -> dict[str, object]:
        self._translation_ui.set_progress_observer(self._auto_translate_progress)
        return self._track_translation(self._translation_ui.resume())

    def get_translation_knowledge(
        self, page: object = 1, page_size: object = 25
    ) -> dict[str, object]:
        return self._translation_response(
            self._translation_ui.knowledge(page=page, page_size=page_size)
        )

    def search_translation_knowledge(
        self, query: object, page: object = 1, page_size: object = 25
    ) -> dict[str, object]:
        return self._translation_response(
            self._translation_ui.search_knowledge(query, page=page, page_size=page_size)
        )

    def import_translation_knowledge(self, path: object) -> dict[str, object]:
        return self._translation_response(
            self._translation_ui.import_knowledge_file(path)
        )

    def export_translation_knowledge(self, path: object) -> dict[str, object]:
        return self._translation_response(
            self._translation_ui.export_knowledge_file(path)
        )

    def correct_translation_memory(
        self, segment_id: object, target_text: object
    ) -> dict[str, object]:
        return self._translation_response(
            self._translation_ui.correct_memory(segment_id, target_text)
        )

    def disable_translation_memory(self, segment_id: object) -> dict[str, object]:
        return self._translation_response(self._translation_ui.disable_memory(segment_id))

    def set_translation_glossary(
        self, source_term: object, target_term: object
    ) -> dict[str, object]:
        return self._translation_response(
            self._translation_ui.correct_glossary(source_term, target_term)
        )

    def disable_translation_glossary(self, source_term: object) -> dict[str, object]:
        return self._translation_response(self._translation_ui.disable_glossary(source_term))

    def set_fields(self, changes: object) -> dict[str, object]:
        if not isinstance(changes, dict):
            return self._invalid(ValueError("Field changes must be an object."))
        if not changes.keys() <= FIELD_NAMES:
            return self._invalid(ValueError("Unknown or unsafe field name."))
        if any(not isinstance(value, str) for value in changes.values()):
            return self._invalid(ValueError("All field values must be text."))
        with self._lock:
            if self._busy:
                return self._busy_response()
            self._fields = MappingProxyType(
                {
                    **self._fields,
                    **{key: value.strip() for key, value in changes.items()},
                }
            )
            if changes.keys() & {
                "auto_new",
                "auto_work",
                "auto_translate_input",
                "auto_output",
                "auto_merge_input",
            }:
                self._reset_translation_validation_locked()
            return self._success_response()

    def browse(self, field: str) -> dict[str, object]:
        request = self._dialog_request(field)
        if isinstance(request, dict):
            return request
        with self._lock:
            if self._busy:
                return self._busy_response()
            picker = self._dialog_picker
        if picker is None:
            return self._invalid(RuntimeError("File dialog is not initialized."))
        try:
            selected = picker(request)
        except Exception as error:
            return self._operation_error(error)
        if not selected:
            return self._success_response()

        selected_path = Path(selected).expanduser()
        value = str(selected_path)
        if field == "unpack_output":
            output = self._named_child(selected_path, "output_unpack")
            value = portable_path_text(output, self.base_directory)
            return self.set_fields(
                {
                    field: value,
                    "json": portable_path_text(
                        output / get_language_spec(self._language).json_filename,
                        self.base_directory,
                    ),
                }
            )
        if field == "repack_output":
            output = self._named_child(selected_path, "output_repack")
            value = portable_path_text(output, self.base_directory)
            return self.set_fields({field: value, "install_source": value})
        if field in {"auto_work", "auto_output", "auto_merge_output"}:
            source_field = {
                "auto_work": "auto_new",
                "auto_output": "auto_translate_input",
                "auto_merge_output": "auto_new",
            }[field]
            source_stem = Path(self._fields[source_field]).stem
            output_name = {
                "auto_work": f"{source_stem}.translation-delta.json",
                "auto_output": f"{source_stem}.vi.json",
                "auto_merge_output": f"{source_stem}.vi.json",
            }[field]
            value = portable_path_text(
                selected_path / output_name,
                self.base_directory,
            )
            return self.set_fields({field: value})
        if field == "font_unpack_output":
            value = portable_path_text(
                font_unpack_output_path(selected_path),
                self.base_directory,
            )
            return self.set_fields({field: value, "font_repack_input": value})
        if field == "font_repack_output":
            value = portable_path_text(
                font_package_output_path(selected_path),
                self.base_directory,
            )
        elif field not in {
            "json",
            "auto_old",
            "auto_new",
            "auto_translate_input",
            "auto_merge_input",
        }:
            value = portable_path_text(selected_path, self.base_directory)
        return self.set_fields({field: value})

    def open_path(self, field: str) -> dict[str, object]:
        if not isinstance(field, str) or field not in FIELD_NAMES:
            return self._invalid(ValueError("Unknown or unsafe field name."))
        with self._lock:
            if self._busy:
                return self._busy_response()
            value = self._fields[field]
        try:
            selected = resolve_portable_path(value, self.base_directory)
            target = (
                selected.parent
                if selected.suffix and not selected.is_dir()
                else selected
            )
            self._directory_opener(target)
        except Exception as error:
            return self._operation_error(error)
        return self._success_response()

    def start_operation(self, name: str) -> dict[str, object]:
        if not isinstance(name, str) or name not in OPERATION_NAMES:
            return self._invalid(ValueError(f"Unknown operation: {name}"))
        try:
            with self._lock:
                if self._busy:
                    return self._busy_response()
                prepared = self._prepare_operation(name)
                self._busy = True
                self._fields = MappingProxyType(
                    {**self._fields, **prepared.field_updates}
                )
                self._status = self._working_status(prepared)
                self._append_monitor_locked(self._status.title)
        except Exception as error:
            return self._operation_error(error)
        threading.Thread(
            target=self._run_operation,
            args=(prepared,),
            daemon=True,
            name=f"crimson-{name}",
        ).start()
        return self._success_response()

    def can_close(self) -> bool:
        with self._lock:
            if self._busy:
                self._append_monitor_locked(
                    "Không thể đóng tool khi tác vụ đang chạy; hãy đợi hoàn tất."
                )
                return False
            return True

    def _run_operation(self, prepared: PreparedOperation) -> None:
        try:
            operation = getattr(self._services, self._service_attribute(prepared.name))
            try:
                result = operation(*prepared.arguments)
            except TypeError:
                # If mock/custom backend takes only (old, new, work)
                if prepared.name == "prepare_translation" and len(prepared.arguments) > 3:
                    result = operation(*prepared.arguments[:3])
                else:
                    raise
            status, updates = self._success_status(prepared, result)
        except Exception as error:
            with self._lock:
                self._busy = False
                self._status = error_status(error)
                self._append_monitor_locked(self._status.title)
            return
        with self._lock:
            self._busy = False
            self._status = status
            self._fields = MappingProxyType({**self._fields, **updates})
            self._apply_translation_result_state_locked(prepared.name, result)
            self._append_monitor_locked(status.title)

    @staticmethod
    def _service_attribute(name: str) -> str:
        return {
            "unpack": "export_localization",
            "repack": "build_localization",
            "font_unpack": "export_fonts",
            "font_repack": "build_fonts",
            "install": "install",
            "uninstall": "uninstall",
            "prepare_translation": "prepare_translation",
            "auto_translate": "auto_translate",
            "validate_translation": "validate_translation",
            "fix_translation_structure": "fix_translation_structure",
            "merge_translation": "merge_translation",
        }[name]

    def _prepare_operation(self, name: str) -> PreparedOperation:
        with self._lock:
            fields = dict(self._fields)
            spec = get_language_spec(self._language)
            provider_key = self._translation_provider
            provider_config = self._translation_ui._runtime_provider_config(provider_key)
        updates: dict[str, str] = {}
        if name == "unpack":
            output = ensure_named_output_directory(
                resolve_portable_path(fields["unpack_output"], self.base_directory),
                "output_unpack",
            )
            json_path = unpack_json_output_path(output, spec.json_filename)
            updates.update(
                unpack_output=portable_path_text(output, self.base_directory),
                json=portable_path_text(json_path, self.base_directory),
            )
            arguments = (fields["source"], json_path, spec)
        elif name == "repack":
            output = ensure_named_output_directory(
                resolve_portable_path(fields["repack_output"], self.base_directory),
                "output_repack",
            )
            updates.update(
                repack_output=portable_path_text(output, self.base_directory),
                install_source=portable_path_text(output, self.base_directory),
            )
            arguments = (
                fields["source"],
                resolve_portable_path(fields["json"], self.base_directory),
                str(output),
                spec,
            )
        elif name == "prepare_translation":
            arguments = (
                resolve_portable_path(fields["auto_old"], self.base_directory),
                resolve_portable_path(fields["auto_new"], self.base_directory),
                resolve_portable_path(fields["auto_work"], self.base_directory),
                self._translation_ui._database,
                "Crimson0020",
                source_language_code(spec.name),
                "vi",
            )
        elif name == "auto_translate":
            new_source = resolve_portable_path(fields["auto_new"], self.base_directory)
            legacy_output = new_source.with_name(f"{new_source.stem}.vi.json")
            legacy_checkpoint = checkpoint_path_for(legacy_output)
            arguments = (
                resolve_portable_path(
                    fields["auto_translate_input"], self.base_directory
                ),
                resolve_portable_path(fields["auto_output"], self.base_directory),
                provider_key,
                source_language_code(spec.name),
                "vi",
                provider_config,
                self._auto_translate_progress,
                legacy_checkpoint if legacy_checkpoint.is_file() else None,
                None,
                self._translation_ui._database.db_path,
            )
        elif name == "validate_translation":
            arguments = (
                resolve_portable_path(fields["auto_new"], self.base_directory),
                resolve_portable_path(fields["auto_work"], self.base_directory),
                resolve_portable_path(fields["auto_merge_input"], self.base_directory),
            )
        elif name == "fix_translation_structure":
            translated = resolve_portable_path(
                fields["auto_merge_input"], self.base_directory
            )
            # The repair step belongs to the file selected in step 3.  Do not
            # depend on auto_output from step 2: that field may legitimately be
            # blank, stale, or point at an older translation run when the user
            # selects an existing translated delta for validation/repair.
            checkpoint = checkpoint_path_for(translated)
            base_stem = re.sub(r"(\.fixed(-\d+)?)+$", "", translated.stem)
            candidate = translated.with_name(f"{base_stem}.fixed.json")
            suffix = 1
            while candidate.exists() or candidate == translated:
                candidate = translated.with_name(f"{base_stem}.fixed-{suffix}.json")
                suffix += 1
            repaired = candidate
            arguments = (
                resolve_portable_path(fields["auto_work"], self.base_directory),
                translated,
                checkpoint if checkpoint.is_file() else None,
                repaired,
                source_language_code(spec.name),
                "vi",
            )
        elif name == "merge_translation":
            arguments = (
                resolve_portable_path(fields["auto_old"], self.base_directory),
                resolve_portable_path(fields["auto_new"], self.base_directory),
                resolve_portable_path(fields["auto_work"], self.base_directory),
                resolve_portable_path(fields["auto_merge_input"], self.base_directory),
                resolve_portable_path(fields["auto_merge_output"], self.base_directory),
            )
        elif name == "font_unpack":
            output = font_unpack_output_path(
                resolve_portable_path(fields["font_unpack_output"], self.base_directory)
            )
            updates["font_unpack_output"] = portable_path_text(
                output, self.base_directory
            )
            arguments = (
                resolve_portable_path(fields["font_source"], self.base_directory),
                output,
                spec,
            )
        elif name == "font_repack":
            output_0012 = font_package_output_path(
                resolve_portable_path(fields["font_repack_output"], self.base_directory)
            )
            updates.update(
                font_repack_output=portable_path_text(output_0012, self.base_directory),
                install_source=portable_path_text(
                    output_0012.parent, self.base_directory
                ),
            )
            arguments = (
                resolve_portable_path(fields["font_source"], self.base_directory),
                resolve_portable_path(fields["font_repack_input"], self.base_directory),
                output_0012.parent,
                spec,
            )
        elif name == "install":
            arguments = (
                resolve_portable_path(fields["install_source"], self.base_directory),
                fields["game"],
                spec,
            )
        else:
            arguments = (fields["game"],)
        return PreparedOperation(
            name=name,
            arguments=arguments,
            spec=spec,
            field_updates=MappingProxyType(updates),
        )

    @staticmethod
    def _working_status(prepared: PreparedOperation) -> StatusMessage:
        spec = prepared.spec
        return {
            "unpack": working_status(
                "ĐANG UNPACK",
                f"Đang đọc {spec.package_folder}, giải nén PALOC và tạo JSON phẳng.",
            ),
            "repack": working_status(
                "ĐANG REPACK",
                "Đang ánh xạ marker + key, cập nhật PAZ/PAMT và xác minh checksum.",
            ),
            "prepare_translation": working_status(
                "ĐANG SO SÁNH JSON",
                "Đang tách riêng key mới và nội dung nguồn đã thay đổi.",
            ),
            "auto_translate": working_status(
                "ĐANG DỊCH TỰ ĐỘNG",
                "Đang dịch toàn bộ delta; dòng lỗi được cô lập thay vì dừng job.",
            ),
            "validate_translation": working_status(
                "ĐANG KIỂM TRA CẤU TRÚC",
                "Đang đối chiếu file đã dịch với delta mẫu và JSON nguồn mới.",
            ),
            "fix_translation_structure": working_status(
                "ĐANG FIX CẤU TRÚC",
                "Đang dựng file sửa riêng từ delta mẫu và checkpoint hợp lệ.",
            ),
            "merge_translation": working_status(
                "ĐANG GHÉP BẢN DỊCH",
                "Đang ghép delta đã xác minh vào JSON nguồn mới đầy đủ.",
            ),
            "font_unpack": working_status(
                "ĐANG UNPACK FONT", "Đang trích xuất ba font từ package 0012."
            ),
            "font_repack": working_status(
                "ĐANG REPACK FONT", "Đang đóng gói ba font và xác minh package 0012."
            ),
            "install": working_status(
                "ĐANG CÀI ĐẶT", "Đang xác minh mod, tạo backup và cập nhật game."
            ),
            "uninstall": working_status(
                "ĐANG GỠ CÀI ĐẶT",
                "Đang tạo safety backup và phục hồi lần Install gần nhất.",
            ),
        }[prepared.name]

    def _success_status(
        self,
        prepared: PreparedOperation,
        result: object,
    ) -> tuple[StatusMessage, dict[str, str]]:
        name = prepared.name
        updates: dict[str, str] = {}
        if name == "unpack":
            if not isinstance(result, ExportResult):
                raise TypeError("Unexpected unpack result.")
            body = (
                f"Ngôn ngữ nguồn: {result.language}\nSố dòng: {result.entry_count:,}\n"
                f"Số PALOC: {result.paloc_count:,}\n"
                f"Tổng dữ liệu PALOC: {result.paloc_size:,} byte\n"
                f"JSON: {portable_path_text(result.json_path, self.base_directory)}"
            )
            detail = self._backup_detail("Backup JSON cũ", result.backup_json)
            updates["json"] = portable_path_text(
                result.json_path,
                self.base_directory,
            )
            return success_status("UNPACK THÀNH CÔNG", body, detail), updates
        if name == "repack":
            if not isinstance(result, BuildResult):
                raise TypeError("Unexpected repack result.")
            migration = (
                f"\nKhớp bản dịch cũ: {result.matched_entries:,}"
                f"\nDòng mới dùng tiếng nguồn: {result.new_entries:,}"
                f"\nDòng cũ không còn: {result.removed_entries:,}"
                f"\nTiếng nguồn đã thay đổi: {result.changed_originals:,}"
                if result.paloc_count > 1
                else ""
            )
            body = (
                f"Ngôn ngữ nguồn: {result.language}\nSố dòng: {result.entry_count:,}\n"
                f"Số PALOC: {result.paloc_count:,}\n"
                f"Tổng dữ liệu PALOC: {result.paloc_size:,} byte\n"
                f"Dữ liệu nén: {result.compressed_size:,} byte\n"
                f"PAZ hoàn chỉnh: {result.paz_size:,} byte\n"
                f"Output: {portable_path_text(result.output_package, self.base_directory)}"
                f"{migration}"
            )
            detail = self._backup_detail(
                f"Backup {result.output_package.name} cũ", result.backup_package
            )
            updates["install_source"] = portable_path_text(
                result.output_package.parent, self.base_directory
            )
            return success_status("REPACK THÀNH CÔNG", body, detail), updates
        if name == "prepare_translation":
            if not isinstance(result, TranslationWorkResult):
                raise TypeError("Unexpected translation-work result.")
            output_text = portable_path_text(result.work_json, self.base_directory)
            body = (
                f"Dòng không đổi (không đưa vào file): {result.unchanged_entries:,}\n"
                f"Key mới phát hiện: {result.new_entries:,}\n"
                f"Key cũ có nội dung nguồn thay đổi: {result.changed_originals:,}\n"
                f"Key cũ không còn: {result.removed_entries:,}\n"
                f"Dòng kỹ thuật không cần dịch: {result.technical_entries:,}\n"
                f"Tổng dòng chờ dịch: {result.translation_candidates:,}\n"
                f"File delta nhỏ: {output_text}"
            )
            detail = self._backup_detail("Backup delta cũ", result.backup_json)
            updates.update(
                auto_work=output_text,
                auto_translate_input=output_text,
            )
            return success_status("FILE CẦN DỊCH ĐÃ SẴN SÀNG", body, detail), updates
        if name == "auto_translate":
            if not isinstance(result, AutoTranslateResult):
                raise TypeError("Unexpected Auto Translate result.")
            output_text = portable_path_text(result.output_json, self.base_directory)
            body = (
                f"Provider: {result.provider_key}\n"
                f"Dòng mới dịch: {result.translated_entries:,}\n"
                f"Dòng tiếp tục từ checkpoint: {result.resumed_entries:,}\n"
                f"Dòng dùng lại Translation Memory: {result.tm_reused_entries:,}\n"
                f"Dòng nhập an toàn từ checkpoint cũ: "
                f"{result.migrated_entries:,}\n"
                f"Dòng đã có bản dịch được giữ nguyên: "
                f"{result.skipped_existing_entries:,}\n"
                f"Dòng kỹ thuật không cần dịch: "
                f"{result.skipped_technical_entries:,}\n"
                f"Dòng trùng dùng lại kết quả exact-match: "
                f"{result.reused_duplicate_entries:,}\n"
                f"Dòng còn chờ / lỗi: {result.pending_entries:,}\n"
                f"Số lượt dịch: {result.request_count:,}\n"
                f"File delta đã dịch: {output_text}"
            )
            detail_parts = [
                "Checkpoint:\n"
                f"{portable_path_text(result.checkpoint_json, self.base_directory)}"
            ]
            backup = self._backup_detail("Backup output cũ", result.backup_json)
            if backup:
                detail_parts.append(backup)
            if result.failures_json is not None:
                detail_parts.append(
                    "Báo cáo dòng lỗi:\n"
                    f"{portable_path_text(result.failures_json, self.base_directory)}"
                )
            updates.update(auto_output=output_text, auto_merge_input=output_text)
            title = (
                "AUTO TRANSLATE HOÀN TẤT"
                if result.pending_entries == 0
                else "AUTO TRANSLATE CHƯA HOÀN TẤT"
            )
            return (
                success_status(
                    title,
                    body,
                    "\n\n".join(detail_parts),
                ),
                updates,
            )
        if name == "validate_translation":
            if not isinstance(result, TranslationDeltaValidationResult):
                raise TypeError("Unexpected translation validation result.")
            body = (
                f"Tổng dòng delta: {result.entry_count:,}\n"
                f"Đã dịch hợp lệ: {result.translated_entries:,}\n"
                f"Còn thiếu: {result.pending_entries:,}\n"
                f"Vấn đề cấu trúc: {len(result.issues):,}"
            )
            title = (
                "CẤU TRÚC HỢP LỆ — CÓ THỂ GHÉP"
                if result.valid and result.complete
                else "CHƯA THỂ GHÉP"
            )
            return success_status(title, body, "\n".join(result.issues)), updates
        if name == "fix_translation_structure":
            if not isinstance(result, TranslationDeltaFixResult):
                raise TypeError("Unexpected translation fix result.")
            output_text = portable_path_text(result.output_json, self.base_directory)
            body = (
                f"Tổng dòng đã dựng lại: {result.entry_count:,}\n"
                f"Bản dịch hợp lệ phục hồi: {result.repaired_entries:,}\n"
                f"Còn chờ dịch: {result.pending_entries:,}\n"
                f"File đã sửa riêng: {output_text}"
            )
            # The repaired delta becomes the canonical translated working file
            # for both continued translation and validation/merge.  Keeping only
            # auto_merge_input updated leaves step 2 writing the damaged file
            # while step 3 reads the repaired file.
            updates.update(auto_output=output_text, auto_merge_input=output_text)
            return (
                success_status(
                    "ĐÃ FIX CẤU TRÚC — HÃY KIỂM TRA LẠI",
                    body,
                    "\n".join(result.issues),
                ),
                updates,
            )
        if name == "merge_translation":
            if not isinstance(result, TranslationMergeResult):
                raise TypeError("Unexpected translation merge result.")
            output_text = portable_path_text(result.output_json, self.base_directory)
            body = (
                f"Bản dịch cũ không đổi được giữ lại: "
                f"{result.reused_unchanged_entries:,}\n"
                f"Bản dịch mới từ delta đã ghép: {result.merged_delta_entries:,}\n"
                f"Dòng còn dùng nội dung nguồn: {result.source_fallback_entries:,}\n"
                f"JSON tiếng Việt hoàn chỉnh: {output_text}"
            )
            detail = self._backup_detail(
                "Backup JSON hoàn chỉnh cũ", result.backup_json
            )
            updates.update(json=output_text, auto_merge_output=output_text)
            return success_status("GHÉP BẢN DỊCH THÀNH CÔNG", body, detail), updates
        if name == "font_unpack":
            if not isinstance(result, FontExportResult):
                raise TypeError("Unexpected font unpack result.")
            body = (
                f"Ngôn ngữ nguồn: {prepared.spec.name}\nSố font: {result.font_count}\n"
                f"Tổng dữ liệu: {result.total_size:,} byte\n"
                f"Output: {portable_path_text(result.output_folder, self.base_directory)}"
            )
            updates["font_repack_input"] = portable_path_text(
                result.output_folder, self.base_directory
            )
            detail = self._backup_detail("Backup thư mục Font cũ", result.backup_folder)
            return success_status("UNPACK FONT THÀNH CÔNG", body, detail), updates
        if name == "font_repack":
            if not isinstance(result, FontBuildResult):
                raise TypeError("Unexpected font repack result.")
            body = (
                f"Font ngôn ngữ: {prepared.spec.font_filename}\n"
                f"Số font: {result.font_count}\n"
                f"PAZ thay đổi: {result.paz_filename} ({result.paz_size:,} byte)\n"
                f"Checksum PAMT: {result.pamt_checksum:08X}\n"
                f"Output: {portable_path_text(result.output_0012, self.base_directory)}"
            )
            updates.update(
                font_repack_output=portable_path_text(
                    result.output_0012, self.base_directory
                ),
                install_source=portable_path_text(
                    result.output_0012.parent, self.base_directory
                ),
            )
            detail = self._backup_detail("Backup 0012 cũ", result.backup_0012)
            return success_status("REPACK FONT THÀNH CÔNG", body, detail), updates
        if name == "install":
            if not isinstance(result, ModInstallResult):
                raise TypeError("Unexpected install result.")
            crc_lines = "\n".join(
                f"CRC {folder}: {crc:08X}" for folder, crc in result.folder_crcs
            )
            body = (
                f"Game: {result.game_root}\nPackage đã cài: "
                f"{', '.join(result.installed_folders)}\n{crc_lines}\n"
                f"Checksum metadata: {result.metadata_checksum:08X}\n"
                "Các mục metadata khác được giữ nguyên."
            )
            return (
                success_status(
                    "CÀI ĐẶT THÀNH CÔNG",
                    body,
                    f"Backup trước khi cài:\n{result.backup_directory}",
                ),
                updates,
            )
        if not isinstance(result, ModUninstallResult):
            raise TypeError("Unexpected uninstall result.")
        crc_lines = "\n".join(
            f"CRC {folder} đã phục hồi: {crc:08X}" for folder, crc in result.folder_crcs
        )
        body = (
            f"Game: {result.game_root}\nPackage đã phục hồi: "
            f"{', '.join(result.restored_folders)}\n{crc_lines}\n"
            f"Checksum metadata: {result.metadata_checksum:08X}\n"
            "Metadata của các mục khác được giữ nguyên."
        )
        detail = (
            f"Backup Install đã dùng:\n{result.restored_backup_directory}\n\n"
            f"Safety backup trước khi gỡ:\n{result.safety_backup_directory}"
        )
        return success_status("GỠ CÀI ĐẶT THÀNH CÔNG", body, detail), updates

    def _backup_detail(self, label: str, path: Path | None) -> str:
        if path is None:
            return ""
        return f"{label}:\n{portable_path_text(path, self.base_directory)}"

    def _reset_translation_validation_locked(self) -> None:
        self._translation_structure_valid = False
        self._translation_fix_available = False
        self._translation_pending_count = 0

    def _apply_translation_result_state_locked(
        self,
        operation: str,
        result: object,
    ) -> None:
        if operation == "validate_translation" and isinstance(
            result, TranslationDeltaValidationResult
        ):
            self._translation_structure_valid = result.valid and result.complete
            self._translation_fix_available = result.repairable
            self._translation_pending_count = result.pending_entries
        elif operation == "auto_translate" and isinstance(result, AutoTranslateResult):
            self._translation_structure_valid = False
            self._translation_fix_available = False
            self._translation_pending_count = result.pending_entries
        elif operation == "fix_translation_structure" and isinstance(
            result, TranslationDeltaFixResult
        ):
            self._translation_structure_valid = False
            self._translation_fix_available = False
            self._translation_pending_count = result.pending_entries
        elif operation == "prepare_translation":
            self._reset_translation_validation_locked()
        elif operation == "merge_translation":
            self._translation_structure_valid = True
            self._translation_fix_available = False
            self._translation_pending_count = 0

    def _auto_translate_progress(self, progress: AutoTranslateProgress) -> None:
        with self._lock:
            self._status = working_status(
                "ĐANG DỊCH TỰ ĐỘNG",
                f"Đã xử lý {progress.completed_entries:,}/{progress.total_entries:,} "
                f"dòng qua {progress.request_count:,} lượt dịch.",
            )

    def _track_translation(self, response: Mapping[str, object]) -> dict[str, object]:
        if response.get("ok"):
            with self._lock:
                self._busy = True
                self._status = working_status(
                    "ĐANG DỊCH TỰ ĐỘNG",
                    "Đang khởi động bản dịch; tiến độ sẽ được cập nhật liên tục.",
                )
                self._append_monitor_locked(self._status.title)
            threading.Thread(
                target=self._watch_translation,
                daemon=True,
                name="crimson-translation-watch",
            ).start()
        return self._translation_response(response)

    def _watch_translation(self) -> None:
        service = self._translation_ui
        terminal = {"idle", "paused", "completed", "completed_with_pending", "failed"}
        state = service.job_state()
        while state not in terminal:
            time.sleep(0.05)
            state = service.job_state()
        job = service.refresh()["state"]["job"]
        with self._lock:
            self._busy = False
            if state == "failed":
                self._status = error_status(
                    RuntimeError(str(job.get("failure_summary") or "Bản dịch gặp lỗi."))
                )
            elif state == "paused":
                self._status = success_status(
                    "ĐÃ TẠM DỪNG AN TOÀN",
                    f"Đã lưu {int(job.get('saved') or 0):,}/"
                    f"{int(job.get('total') or 0):,} dòng; có thể đổi provider rồi tiếp tục.",
                )
            elif state == "completed_with_pending":
                self._status = working_status(
                    "DỊCH CÒN DÒNG CHỜ",
                    f"Còn {int(job.get('pending') or 0):,} dòng chưa dịch. "
                    "Chọn provider khác rồi Tiếp tục.",
                )
            else:
                self._status = success_status(
                    "DỊCH HOÀN TẤT",
                    f"Đã lưu {int(job.get('saved') or 0):,}/"
                    f"{int(job.get('total') or 0):,} dòng.",
                )
            self._append_monitor_locked(self._status.title)

    def _dialog_request(self, field: str) -> DialogRequest | dict[str, object]:
        if not isinstance(field, str) or field not in FIELD_NAMES:
            return self._invalid(ValueError("Unknown or unsafe field name."))
        file_field = field in {
            "json",
            "auto_old",
            "auto_new",
            "auto_translate_input",
            "auto_merge_input",
        }
        titles = {
            "source": "Chọn package localization nguồn",
            "font_source": "Chọn package Font 0012 nguồn",
            "json": "Chọn localization JSON",
            "auto_old": "Chọn JSON tiếng Việt của bản cũ",
            "auto_new": "Chọn JSON nguồn của phiên bản mới",
            "auto_work": "Chọn nơi lưu file chỉ gồm câu cần dịch",
            "auto_translate_input": "Chọn file delta cần dịch",
            "auto_output": "Chọn nơi lưu file delta đã dịch",
            "auto_merge_input": "Chọn file delta đã dịch để ghép",
            "auto_merge_output": "Chọn nơi lưu JSON tiếng Việt hoàn chỉnh",
            "unpack_output": "Chọn nơi lưu kết quả Unpack",
            "repack_output": "Chọn nơi lưu kết quả Repack",
            "font_unpack_output": "Chọn nơi lưu Font sau khi Unpack",
            "font_repack_input": "Chọn thư mục Font dùng để Repack",
            "font_repack_output": "Chọn nơi lưu package Font 0012",
            "install_source": "Chọn thư mục mod để cài",
            "game": "Chọn thư mục game Crimson Desert",
        }
        with self._lock:
            current = self._fields[field]
        if field in {"auto_work", "auto_output", "auto_merge_output"} and current:
            current = str(resolve_portable_path(current, self.base_directory).parent)
        return DialogRequest(
            field=field,
            kind="file" if file_field else "folder",
            title=titles[field],
            initial_directory=current,
            file_types=("JSON files (*.json)", "All files (*.*)"),
        )

    @staticmethod
    def _named_child(path: Path, name: str) -> Path:
        return path if path.name.casefold() == name.casefold() else path / name

    def _append_monitor_locked(self, text: str) -> None:
        self._monitor = (*self._monitor, f"[{self._clock()}] {text}")[-100:]

    def _success_response(self) -> dict[str, object]:
        return {"ok": True, "state": self.get_state()}

    def _translation_response(self, response: Mapping[str, object]) -> dict[str, object]:
        payload: dict[str, object] = {
            "ok": bool(response.get("ok")),
            "state": self.get_state(),
        }
        if "error" in response:
            error = dict(response["error"])
            if error.get("code") == "invalid_provider":
                error["code"] = "invalid_input"
            payload["error"] = error
        for key in ("knowledge", "export", "import"):
            if key in response:
                payload[key] = response[key]
        if "knowledge" in response:
            payload["state"]["translation_knowledge"] = response["knowledge"]
        return payload

    def _busy_response(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {"code": "busy", "message": "Một tác vụ khác đang chạy."},
            "state": self.get_state(),
        }

    def _invalid(self, error: Exception) -> dict[str, object]:
        return {
            "ok": False,
            "error": {"code": "invalid_input", "message": str(error)},
            "state": self.get_state(),
        }

    def _operation_error(self, error: Exception) -> dict[str, object]:
        with self._lock:
            self._status = error_status(error)
            self._append_monitor_locked(self._status.title)
        return {
            "ok": False,
            "error": {"code": "operation_error", "message": str(error)},
            "state": self.get_state(),
        }
