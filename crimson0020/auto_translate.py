"""Provider orchestration for sparse localization translation deltas."""

import concurrent.futures
import copy
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .paloc import PalocFormatError, read_paloc_json
from .translation_providers import (
    PartialTranslationProviderError,
    RetryableTranslationProviderError,
    TranslationProvider,
    TranslationProviderError,
    TranslationResponseFormatError,
    create_translation_provider,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.memory import (
    TranslationMemoryRepository,
    TranslationSegment,
)
from crimson0020.translation_runtime.knowledge.service import (
    CatalogRegistration,
    KnowledgeService,
    ScopeIdentifier,
)
from crimson0020.translation_runtime.failure_state import (
    FailureRecord,
    FailureStateManager,
)
from crimson0020.translation_runtime.progress import (
    PauseRequested,
    ProgressReporter,
    ProgressState,
    TranslationControl,
)
from crimson0020.translation_runtime.protection import mask_protected_content
from crimson0020.translation_runtime.contracts import TranslationRequestItem
from crimson0020.translation_runtime.knowledge.terminology import TerminologyEngine
from crimson0020.translation_runtime.unit_builder import (
    RawTranslationInput,
    TranslationUnitBuilder,
)

CHECKPOINT_FORMAT_VERSION = 2
DEFAULT_MAX_BATCH_ITEMS = 24
DEFAULT_MAX_BATCH_CHARACTERS = 4000
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_PROVIDER_FAILURE_CIRCUIT = 3
_SOURCE_LANGUAGE_CODES = {
    "English": "en",
    "Korean": "ko",
    "Japanese": "ja",
    "Russian": "ru",
    "Turkish": "tr",
    "Spanish": "es",
    "Mexican Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Polish": "pl",
    "Portuguese - Brazil": "pt",
    "Traditional Chinese": "zh",
    "Simplified Chinese": "zh",
}
_PROTECTED_MARKER = re.compile(r"__CRIMSON_KEEP_\d{4}_[0-9A-F]{8}__")
_LOGGER = logging.getLogger(__name__)
def has_translatable_content(value: str) -> bool:
    masked = mask_protected_content(value).masked_text
    return bool(re.sub(r"<x\d+>", "", masked).strip())


def _contract_tokens(value: str) -> tuple[str, ...]:
    """Return only tokens that the current protected-content contract freezes."""
    return mask_protected_content(value).tokens


class AutoTranslateError(RuntimeError):
    """Base error for the Auto Translate workflow."""


class TranslationContractError(AutoTranslateError):
    """Raised for an invalid protected-text contract."""


class TranslationCheckpointError(AutoTranslateError):
    """Raised when durable resume data does not match the selected delta."""


class TranslationLockError(AutoTranslateError):
    """Raised when another auto-translate process holds the lock."""


@contextmanager
def _acquire_process_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except (OSError, PermissionError) as exc:
                raise TranslationLockError(
                    f"Another translation process is already running for {lock_path.name}."
                ) from exc
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, IOError) as exc:
                raise TranslationLockError(
                    f"Another translation process is already running for {lock_path.name}."
                ) from exc
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass(frozen=True)
class AutoTranslateProgress:
    completed_entries: int
    total_entries: int
    request_count: int
    pending_entries: int = 0
    translated_entries: int = 0
    resumed_entries: int = 0
    tm_reused_entries: int = 0
    current_route: str | None = None
    current_model: str | None = None
    fallback_count: int = 0
    cooldown_seconds: float = 0.0
    failure_summary: str | None = None


@dataclass(frozen=True)
class AutoTranslateResult:
    output_json: Path
    backup_json: Path | None
    checkpoint_json: Path
    translated_entries: int
    resumed_entries: int
    skipped_existing_entries: int
    skipped_empty_entries: int
    skipped_technical_entries: int
    reused_duplicate_entries: int
    request_count: int
    provider_key: str
    pending_entries: int = 0
    failure_count: int = 0
    failures_json: Path | None = None
    migrated_entries: int = 0
    tm_reused_entries: int = 0


@dataclass(frozen=True)
class _CatalogRow:
    index: int
    marker: str
    key: str
    original: str
    translation: str
    context: tuple[str, ...] = ()
    context_signature: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        return self.marker, self.key

    @property
    def dedup_key(self) -> tuple[str, str | tuple[str, ...]]:
        return (self.original, self.context_signature or self.context)


@dataclass(frozen=True)
class _ProtectedText:
    value: str
    tokens: tuple[str, ...]
    markers: tuple[str, ...]


def source_language_code(language_name: str) -> str:
    try:
        return _SOURCE_LANGUAGE_CODES[language_name]
    except KeyError as error:
        raise ValueError(
            f"Auto Translate does not support source language: {language_name}"
        ) from error


def _create_automatic_provider_chain(
    provider_key: str,
    provider_config: Mapping[str, str] | None,
) -> tuple[TranslationProvider, ...]:
    """Create only the selected provider.

    Standalone free endpoints (Google, Bing, ...) must never switch to a
    different provider on their own; the user changes provider explicitly.
    Only the AI Router may fall back internally between its own routes/models.
    """

    return (create_translation_provider(provider_key, provider_config),)


def checkpoint_path_for(output_json: Path | str) -> Path:
    output = Path(output_json).expanduser().resolve()
    return output.with_name(f"{output.name}.autotranslate-checkpoint.jsonl")


def failure_path_for(output_json: Path | str) -> Path:
    output = Path(output_json).expanduser().resolve()
    return output.with_name(f"{output.name}.autotranslate-failures.jsonl")


def _catalog_rows(data: dict[str, object]) -> tuple[_CatalogRow, ...]:
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise PalocFormatError("JSON must contain an entries list.")
    rows: list[_CatalogRow] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise PalocFormatError(f"entries[{index}] must be an object.")
        marker, key, original, translation = (
            raw.get("marker"),
            raw.get("key"),
            raw.get("original"),
            raw.get("translation", ""),
        )
        if not isinstance(marker, str):
            raise PalocFormatError(f"entries[{index}].marker must be hexadecimal text.")
        try:
            marker_bytes = bytes.fromhex(marker)
        except ValueError as error:
            raise PalocFormatError(
                f"entries[{index}].marker is not valid hexadecimal."
            ) from error
        if len(marker_bytes) != 8:
            raise PalocFormatError(
                f"entries[{index}].marker must contain exactly 8 bytes."
            )
        if (
            not isinstance(key, str)
            or not isinstance(original, str)
            or not isinstance(translation, str)
        ):
            raise PalocFormatError(
                f"entries[{index}] key, original and translation must be text."
            )
        identity = (marker_bytes.hex(), key)
        if identity in seen:
            raise ValueError(f"Duplicate marker + key identity in entries[{index}].")
        seen.add(identity)
        raw_context = raw.get("context")
        if raw_context is None:
            context = ()
        elif isinstance(raw_context, list) and all(
            isinstance(value, str) for value in raw_context
        ):
            context = tuple(raw_context)
        elif isinstance(raw_context, str):
            context = (raw_context,)
        else:
            raise PalocFormatError(f"entries[{index}].context must contain text only.")
        raw_sig = raw.get("context_signature")
        if raw_sig is None:
            context_signature = ""
        elif isinstance(raw_sig, str):
            context_signature = raw_sig
        else:
            raise PalocFormatError(f"entries[{index}].context_signature must be text.")
        rows.append(
            _CatalogRow(
                index,
                identity[0],
                key,
                original,
                translation,
                context,
                context_signature,
            )
        )
    return tuple(rows)


def _source_fingerprint(data: dict[str, object], rows: Sequence[_CatalogRow]) -> str:
    immutable = {
        "metadata": data.get("metadata"),
        "entries": [
            {"marker": row.marker, "key": row.key, "original": row.original}
            for row in rows
        ],
    }
    return (
        hashlib.sha256(
            json.dumps(
                immutable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def _is_delta(data: dict[str, object]) -> bool:
    return (
        data.get("schema") == "crimson.translation-delta"
        and data.get("schema_version") == 1
        and data.get("kind") == "sparse-delta"
    )


def _delta_fingerprint(data: dict[str, object], rows: Sequence[_CatalogRow]) -> str:
    source = data.get("source")
    if not isinstance(source, dict) or not isinstance(
        source.get("catalog_fingerprint"), str
    ):
        raise TranslationCheckpointError(
            "Translation delta is missing its source fingerprint."
        )
    value = source["catalog_fingerprint"]
    if len(value) != 64 or any(
        character not in "0123456789ABCDEFabcdef" for character in value
    ):
        raise TranslationCheckpointError(
            "Translation delta source fingerprint is invalid."
        )
    return value.upper()


def _package_family(data: dict[str, object], *, is_delta: bool) -> str:
    container = data.get("source") if is_delta else data.get("metadata")
    if isinstance(container, dict):
        value = container.get("package_folder")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "0020"


def _effective_context_signature(row: _CatalogRow) -> str:
    if row.context_signature:
        return row.context_signature
    if row.context:
        return hashlib.sha256(
            ("context\0" + "\0".join(row.context)).encode("utf-8")
        ).hexdigest()
    return ""


def _protect_text(value: str) -> _ProtectedText:
    masked = mask_protected_content(value)
    tokens = list(masked.tokens)
    markers: list[str] = []
    protected_value = masked.masked_text
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8].upper()
        marker = f"__CRIMSON_KEEP_{index:04d}_{digest}__"
        markers.append(marker)
        protected_value = protected_value.replace(f"<x{index}>", marker, 1)
    return _ProtectedText(protected_value, tuple(tokens), tuple(markers))


def _restore_text(translated: str, protected: _ProtectedText) -> str:
    if not isinstance(translated, str):
        raise TranslationContractError("Provider output must be text.")
    if tuple(_PROTECTED_MARKER.findall(translated)) != protected.markers:
        raise TranslationContractError(
            "Provider changed, removed, duplicated, or reordered a protected token."
        )
    restored = translated
    for marker, token in zip(protected.markers, protected.tokens, strict=True):
        restored = restored.replace(marker, token, 1)
    if _contract_tokens(restored) != protected.tokens:
        raise TranslationContractError(
            "Provider output does not preserve placeholders, markup, or newlines."
        )
    return restored


def _batches(
    rows: Sequence[_CatalogRow], max_items: int, max_characters: int
) -> tuple[tuple[_CatalogRow, ...], ...]:
    if max_items <= 0 or max_characters <= 0:
        raise ValueError("Auto Translate batch limits must be positive.")
    batches: list[tuple[_CatalogRow, ...]] = []
    current: list[_CatalogRow] = []
    size = 0
    for row in rows:
        characters = len(row.original)
        if characters > max_characters:
            raise TranslationContractError(
                f"Text for marker + key {row.marker} + {row.key!r} exceeds batch limit."
            )
        if current and (
            len(current) >= max_items or size + characters > max_characters
        ):
            batches.append(tuple(current))
            current, size = [], 0
        current.append(row)
        size += characters
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_header(
    source_fingerprint: str,
    provider_key: str,
    source_language: str,
    target_language: str,
) -> dict[str, object]:
    return {
        "record": "header",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "source_fingerprint": source_fingerprint,
        "provider": provider_key,
        "source_language": source_language,
        "target_language": target_language,
    }


def _translation_record(row: _CatalogRow, translation: str) -> dict[str, str]:
    return {
        "marker": row.marker,
        "key": row.key,
        "original_sha256": hashlib.sha256(row.original.encode("utf-8"))
        .hexdigest()
        .upper(),
        "context_signature": _effective_context_signature(row),
        "translation": translation,
    }


def _initialize_checkpoint(path: Path, header: dict[str, object]) -> None:
    try:
        _atomic_text(
            path, json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    except OSError as error:
        raise TranslationCheckpointError(
            f"Cannot initialize translation checkpoint: {error}"
        ) from error


def _append_checkpoint_batch(
    path: Path, rows: Sequence[_CatalogRow], translations: Sequence[str]
) -> None:
    record = {
        "record": "batch",
        "translations": [
            _translation_record(row, translation)
            for row, translation in zip(rows, translations, strict=True)
        ],
    }
    try:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise TranslationCheckpointError(
            f"Cannot append completed translation batch: {error}"
        ) from error


def _rewrite_failures(
    path: Path,
    rows: Sequence[_CatalogRow],
    provider_key: Mapping[tuple[str, str], str] | str,
    messages: Mapping[tuple[str, str], str] | str,
) -> None:
    current = {
        row.identity: FailureRecord(
            marker=row.marker,
            key=row.key,
            original_sha256=hashlib.sha256(row.original.encode("utf-8"))
            .hexdigest()
            .upper(),
            provider=(
                provider_key.get(row.identity, "unknown-provider")
                if isinstance(provider_key, Mapping)
                else provider_key
            ),
            message=(
                messages.get(row.identity, "translation failed")
                if isinstance(messages, Mapping)
                else messages
            ),
        )
        for row in rows
    }
    FailureStateManager(path).save_failures(current)


def _load_checkpoint(
    path: Path,
    source_fingerprint: str,
    source_language: str,
    target_language: str,
    candidates: Sequence[_CatalogRow],
) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        header = json.loads(lines[0]) if lines else None
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError) as error:
        raise TranslationCheckpointError(f"Cannot read checkpoint: {error}") from error
    expected = {
        "record": "header",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "source_fingerprint": source_fingerprint,
        "source_language": source_language,
        "target_language": target_language,
    }
    if not isinstance(header, dict) or any(
        header.get(key) != value for key, value in expected.items()
    ):
        raise TranslationCheckpointError(
            "Checkpoint does not match the selected source or language pair."
        )
    by_identity = {row.identity: row for row in candidates}
    completed: dict[tuple[str, str], str] = {}
    incomplete_tail = not text.endswith(("\n", "\r"))
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines) and not text.endswith(("\n", "\r")):
                break
            raise TranslationCheckpointError(
                f"Checkpoint record on line {line_number} is invalid JSON."
            )
        if (
            not isinstance(record, dict)
            or record.get("record") != "batch"
            or not isinstance(record.get("translations"), list)
        ):
            raise TranslationCheckpointError(
                f"Checkpoint record on line {line_number} is invalid."
            )
        for raw in record["translations"]:
            if not isinstance(raw, dict):
                raise TranslationCheckpointError(
                    f"Checkpoint record on line {line_number} has an invalid translation."
                )
            identity = (raw.get("marker"), raw.get("key"))
            row = by_identity.get(identity)
            translation = raw.get("translation")
            if row is None or not isinstance(translation, str):
                raise TranslationCheckpointError(
                    f"Checkpoint record on line {line_number} has an unknown or duplicate identity."
                )
            if "context_signature" in raw:
                raw_context_signature = raw.get("context_signature")
                if not isinstance(raw_context_signature, str):
                    raise TranslationCheckpointError(
                        f"Checkpoint record on line {line_number} has an invalid context signature."
                    )
                if raw_context_signature != _effective_context_signature(row):
                    continue
            elif _effective_context_signature(row):
                continue
            if identity in completed:
                raise TranslationCheckpointError(
                    f"Checkpoint record on line {line_number} has an unknown or duplicate identity."
                )
            expected_hash = (
                hashlib.sha256(row.original.encode("utf-8")).hexdigest().upper()
            )
            raw_hash = raw.get("original_sha256")
            if isinstance(raw_hash, str):
                raw_hash = raw_hash.upper()
            if (
                raw_hash != expected_hash
                or _contract_tokens(translation) != _contract_tokens(row.original)
                or not has_translatable_content(translation)
            ):
                raise TranslationCheckpointError(
                    f"Checkpoint record on line {line_number} breaks the translation contract."
                )
            completed[identity] = translation
    if incomplete_tail:
        try:
            _atomic_text(path, "\n".join(lines[:-1]) + "\n")
        except OSError as error:
            raise TranslationCheckpointError(
                f"Cannot repair incomplete checkpoint tail: {error}"
            ) from error
    return completed


def _load_legacy_checkpoint(
    path: Path | None,
    source_fingerprint: str,
    source_language: str,
    target_language: str,
    candidates: Sequence[_CatalogRow],
) -> dict[tuple[str, str], str]:
    """Import only exact, contract-valid rows from the former v1 journal."""

    if path is None or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        header = json.loads(lines[0]) if lines else None
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError):
        return {}
    expected_header = {
        "record": "header",
        "format_version": 1,
        "source_fingerprint": source_fingerprint,
        "source_language": source_language,
        "target_language": target_language,
    }
    if not isinstance(header, dict) or any(
        header.get(key) != value for key, value in expected_header.items()
    ):
        return {}
    by_identity = {row.identity: row for row in candidates}
    completed: dict[tuple[str, str], str] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines) and not text.endswith(("\n", "\r")):
                break
            return {}
        if not isinstance(record, dict) or record.get("record") != "batch":
            return {}
        translations = record.get("translations")
        if not isinstance(translations, list):
            return {}
        for raw in translations:
            if not isinstance(raw, dict):
                continue
            identity = (raw.get("marker"), raw.get("key"))
            row = by_identity.get(identity)
            translation = raw.get("translation")
            if row is None or identity in completed or not isinstance(translation, str):
                continue
            if _effective_context_signature(row):
                continue
            expected_hash = (
                hashlib.sha256(row.original.encode("utf-8")).hexdigest().upper()
            )
            if (
                raw.get("original_sha256") == expected_hash
                and _contract_tokens(translation) == _contract_tokens(row.original)
                and has_translatable_content(translation)
            ):
                completed[row.identity] = translation
    return completed


def _translate_with_retry(
    provider: TranslationProvider,
    texts: Sequence[str],
    source_language: str,
    target_language: str,
    attempts: int,
    base_delay: float,
    sleeper: Callable[[float], None],
    request_items: Sequence[TranslationRequestItem] | None = None,
    translation_control: TranslationControl | None = None,
) -> tuple[str, ...]:
    if attempts <= 0 or base_delay < 0:
        raise ValueError("Auto Translate retry settings are invalid.")
    for attempt in range(attempts):
        if translation_control is not None and translation_control.should_pause_after_request():
            raise PauseRequested(translation_control.reason or "Translation paused")
        try:
            rich_translate = getattr(provider, "translate_items", None)
            if request_items is not None and callable(rich_translate):
                if tuple(item.masked_text for item in request_items) != tuple(texts):
                    raise TranslationContractError(
                        "Rich provider items do not match the protected translation batch."
                    )
                result = rich_translate(request_items, source_language, target_language)
            else:
                result = provider.translate(texts, source_language, target_language)
            if not isinstance(result, (list, tuple)) or isinstance(result, (str, bytes, dict)):
                raise TranslationContractError(
                    f"Provider returned {type(result).__name__} instead of a sequence of strings."
                )
            if any(not isinstance(item, str) for item in result):
                raise TranslationContractError(
                    "Provider returned non-string elements in translation sequence."
                )
            return tuple(result)
        except RetryableTranslationProviderError as error:
            if attempt + 1 >= attempts:
                raise
            retry_after = getattr(error, "retry_after_seconds", None)
            remaining = max(
                base_delay * (2**attempt),
                float(retry_after or 0.0),
            )
            while remaining > 0:
                if (
                    translation_control is not None
                    and translation_control.should_pause_after_request()
                ):
                    raise PauseRequested(translation_control.reason or "Translation paused")
                interval = min(0.1, remaining)
                sleeper(interval)
                remaining -= interval
            if translation_control is not None and translation_control.should_pause_after_request():
                raise PauseRequested(translation_control.reason or "Translation paused")
    raise AssertionError("unreachable")


def _backup_path(output: Path) -> Path:
    candidate = output.with_name(f"{output.name}.backup")
    suffix = 1
    while candidate.exists():
        candidate = output.with_name(f"{output.name}.backup-{suffix}")
        suffix += 1
    return candidate


def translate_catalog(
    source_json: Path | str,
    output_json: Path | str,
    provider: TranslationProvider,
    source_language: str,
    target_language: str = "vi",
    *,
    max_batch_items: int | None = None,
    max_batch_characters: int | None = None,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[AutoTranslateProgress], None] | None = None,
    legacy_checkpoint: Path | str | None = None,
    on_commit_stage: Callable[[str], None] | None = None,
    db_path: Path | str | None = None,
    translation_control: TranslationControl | None = None,
    progress_reporter: ProgressReporter | None = None,
    job_id: str | None = None,
    fallback_providers: Sequence[TranslationProvider] = (),
) -> AutoTranslateResult:
    source = Path(source_json).expanduser().resolve()
    output = Path(output_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Auto Translate source JSON does not exist: {source}")
    if source == output:
        raise ValueError("Auto Translate output must be separate from the source JSON.")
    if not getattr(provider, "key", ""):
        raise ValueError("Translation provider must expose a stable key.")
    provider_chain = (provider, *tuple(fallback_providers))
    if any(not getattr(item, "key", "") for item in provider_chain):
        raise ValueError("Every fallback translation provider must expose a stable key.")
    lock_path = output.with_name(f"{output.name}.autotranslate.lock")
    with _acquire_process_lock(lock_path):
        return _translate_catalog_core(
            source=source,
            output=output,
            provider=provider,
            source_language=source_language,
            target_language=target_language,
            max_batch_items=max_batch_items,
            max_batch_characters=max_batch_characters,
            retry_attempts=retry_attempts,
            retry_base_delay=retry_base_delay,
            sleeper=sleeper,
            progress=progress,
            legacy_checkpoint=legacy_checkpoint,
            on_commit_stage=on_commit_stage,
            db_path=db_path,
            translation_control=translation_control,
            progress_reporter=progress_reporter,
            job_id=job_id,
            fallback_providers=provider_chain[1:],
        )


def _translate_catalog_core(
    source: Path,
    output: Path,
    provider: TranslationProvider,
    source_language: str,
    target_language: str = "vi",
    *,
    max_batch_items: int | None = None,
    max_batch_characters: int | None = None,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[AutoTranslateProgress], None] | None = None,
    legacy_checkpoint: Path | str | None = None,
    on_commit_stage: Callable[[str], None] | None = None,
    db_path: Path | str | None = None,
    translation_control: TranslationControl | None = None,
    progress_reporter: ProgressReporter | None = None,
    job_id: str | None = None,
    fallback_providers: Sequence[TranslationProvider] = (),
) -> AutoTranslateResult:
    data = read_paloc_json(source)
    is_delta = _is_delta(data)
    if is_delta:
        # Reuse the Step 1 boundary validator so every Step 2 entrypoint enforces
        # the same strict types and Unicode contract before any checkpoint/provider work.
        from .translation_work import read_translation_delta

        data = read_translation_delta(source)
    rows = _catalog_rows(data)
    if is_delta:
        source_fingerprint = _delta_fingerprint(data, rows)
        checkpoint_rows = tuple(
            row
            for row in rows
            if row.original and has_translatable_content(row.original)
        )
        candidates = tuple(
            row
            for row in rows
            if row.original
            and data["entries"][row.index].get("translation_state", "pending")
            not in {"translated", "skipped_technical"}
            and has_translatable_content(row.original)
        )
    else:
        source_fingerprint = _source_fingerprint(data, rows)
        candidates = tuple(
            row
            for row in rows
            if row.original
            and (not row.translation.strip() or row.translation == row.original)
            and has_translatable_content(row.original)
        )
        checkpoint_rows = candidates
    batch_items = (
        getattr(provider, "preferred_batch_items", DEFAULT_MAX_BATCH_ITEMS)
        if max_batch_items is None
        else max_batch_items
    )
    batch_characters = (
        getattr(provider, "preferred_batch_characters", DEFAULT_MAX_BATCH_CHARACTERS)
        if max_batch_characters is None
        else max_batch_characters
    )
    base_delay = (
        getattr(provider, "preferred_retry_base_delay", 1.0)
        if retry_base_delay is None
        else retry_base_delay
    )
    checkpoint = checkpoint_path_for(output)
    migrated: dict[tuple[str, str], str] = {}
    if checkpoint.exists():
        completed = _load_checkpoint(
            checkpoint,
            source_fingerprint,
            source_language,
            target_language,
            checkpoint_rows,
        )
    else:
        legacy_path = (
            Path(legacy_checkpoint).expanduser().resolve()
            if legacy_checkpoint is not None
            else None
        )
        migrated = _load_legacy_checkpoint(
            legacy_path,
            source_fingerprint,
            source_language,
            target_language,
            checkpoint_rows,
        )
        _initialize_checkpoint(
            checkpoint,
            _checkpoint_header(
                source_fingerprint, provider.key, source_language, target_language
            ),
        )
        if migrated:
            migrated_rows = tuple(
                row for row in checkpoint_rows if row.identity in migrated
            )
            _append_checkpoint_batch(
                checkpoint,
                migrated_rows,
                tuple(migrated[row.identity] for row in migrated_rows),
            )
        completed = dict(migrated)

    scope = ScopeIdentifier(
        app_id="Crimson0020",
        package_family=_package_family(data, is_delta=is_delta),
        source_language=source_language,
        target_language=target_language,
    )
    knowledge_db: KnowledgeDatabase | None = None
    terminology: TerminologyEngine | None = None
    unit_builder: TranslationUnitBuilder | None = None
    lineage_id: str | None = None

    def hard_terms_hash_for(row: _CatalogRow) -> str:
        if terminology is None:
            return ""
        protected = _protect_text(row.original)
        constraints = terminology.extract_constraints(scope.scope_id, protected.value)
        if not constraints:
            return ""
        serialized = "|".join(
            f"{constraint.source_term}:{constraint.target_term}"
            for constraint in constraints
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    if db_path is not None:
        try:
            knowledge_db = KnowledgeDatabase(database_path=db_path)
            terminology = TerminologyEngine(knowledge_db)
            knowledge_service = KnowledgeService(knowledge_db)
            lineage_id = knowledge_service.get_lineage_for_fingerprint(
                scope, source_fingerprint
            )
            if lineage_id is None:
                source_info = data.get("source") if is_delta else None
                prior_fingerprint = (
                    source_info.get("base_catalog_fingerprint")
                    if isinstance(source_info, dict)
                    and isinstance(source_info.get("base_catalog_fingerprint"), str)
                    else None
                )
                declared_count = (
                    source_info.get("entry_count")
                    if isinstance(source_info, dict)
                    else None
                )
                item_count = (
                    declared_count
                    if isinstance(declared_count, int)
                    and not isinstance(declared_count, bool)
                    and declared_count >= 0
                    else len(rows)
                )
                lineage_id = knowledge_service.register_catalog(
                    CatalogRegistration(
                        scope=scope,
                        fingerprint=source_fingerprint,
                        item_count=item_count,
                        prior_fingerprint=prior_fingerprint,
                    )
                )
            unit_builder = TranslationUnitBuilder(
                db=knowledge_db,
                scope=scope,
                lineage_id=lineage_id,
            )

            now_time = time.time()
            memory_repo = TranslationMemoryRepository(knowledge_db)
            by_ident = {row.identity: row for row in checkpoint_rows}
            initial_segments = []
            for ident, trans_text in completed.items():
                r = by_ident.get(ident)
                if r is not None:
                    shash = hashlib.sha256(r.original.encode("utf-8")).hexdigest()
                    initial_segments.append(
                        TranslationSegment(
                            segment_id=(
                                f"seg_{scope.scope_id}_{lineage_id}_{r.marker}_{r.key}_{shash}"
                            ),
                            scope_id=scope.scope_id,
                            lineage_id=lineage_id,
                            marker=r.marker,
                            key=r.key,
                            source_hash=shash,
                            source_text=r.original,
                            target_text=trans_text,
                            context_signature=r.context_signature,
                            hard_terms_hash=hard_terms_hash_for(r),
                            created_at=now_time,
                            updated_at=now_time,
                        )
                    )
            if initial_segments:
                memory_repo.save_segments(initial_segments)
        except Exception as error:
            _LOGGER.warning(
                "Translation knowledge is unavailable: %s", type(error).__name__
            )
            knowledge_db = None
            terminology = None
            unit_builder = None
            lineage_id = None

    resumed = len(completed) - len(migrated)
    pending = tuple(row for row in candidates if row.identity not in completed)
    if progress_reporter is not None:
        progress_reporter.publish(
            state=ProgressState.RUNNING,
            saved_entries=len(completed),
            resumed_entries=resumed,
            pending_entries=len(pending),
        )
    source_cache = {
        row.dedup_key: value
        for row in candidates
        if row.identity in completed
        for value in (completed[row.identity],)
    }
    reused_rows = tuple(row for row in pending if row.dedup_key in source_cache)
    if reused_rows:
        reused_values = tuple(source_cache[row.dedup_key] for row in reused_rows)
        _append_checkpoint_batch(checkpoint, reused_rows, reused_values)
        completed = {
            **completed,
            **{
                row.identity: value
                for row, value in zip(reused_rows, reused_values, strict=True)
            },
        }
    pending = tuple(row for row in pending if row.identity not in completed)
    groups: dict[tuple[str, str | tuple[str, ...]], list[_CatalogRow]] = {}
    for row in pending:
        groups.setdefault(row.dedup_key, []).append(row)
    representatives = tuple(group[0] for group in groups.values())
    planned_items: dict[tuple[str, str], TranslationRequestItem] = {}
    tm_reused_rows: list[_CatalogRow] = []
    tm_reused_values: list[str] = []
    if unit_builder is not None and representatives:
        try:
            protected_inputs = tuple(
                RawTranslationInput(
                    marker=row.marker,
                    key=row.key,
                    original=_protect_text(row.original).value,
                    source_hash=hashlib.sha256(row.original.encode("utf-8")).hexdigest(),
                    context=row.context,
                    context_signature=row.context_signature,
                )
                for row in representatives
            )
            unit_plan = unit_builder.build_units(protected_inputs)
            for request_item in unit_plan.outbound_items:
                identity = unit_plan.work_id_map[request_item.work_id]
                planned_items[identity] = request_item

            representative_by_identity = {
                row.identity: row for row in representatives
            }
            for identity, memory_item in unit_plan.exact_matches.items():
                row = representative_by_identity.get(identity)
                if row is None:
                    continue
                translated_text = memory_item.translated_text
                if (
                    _contract_tokens(translated_text) != _contract_tokens(row.original)
                    or not has_translatable_content(translated_text)
                ):
                    continue
                protected = _protect_text(row.original)
                hard_constraints = terminology.extract_constraints(
                    scope.scope_id, protected.value
                ) if terminology is not None else ()
                if any(
                    constraint.target_term.casefold() not in translated_text.casefold()
                    for constraint in hard_constraints
                ):
                    continue
                for grouped in groups[row.dedup_key]:
                    tm_reused_rows.append(grouped)
                    tm_reused_values.append(translated_text)
        except Exception as error:
            _LOGGER.warning(
                "Translation-unit planning is unavailable: %s", type(error).__name__
            )
            planned_items.clear()

    if tm_reused_rows:
        _append_checkpoint_batch(
            checkpoint, tuple(tm_reused_rows), tuple(tm_reused_values)
        )
        completed = {
            **completed,
            **{
                row.identity: value
                for row, value in zip(
                    tm_reused_rows, tm_reused_values, strict=True
                )
            },
        }
        representatives = tuple(
            row for row in representatives if row.identity not in completed
        )
    reused_duplicates = len(reused_rows) + sum(
        len(group) - 1 for group in groups.values()
    )
    requests = 0
    translated_now = 0
    failures: list[_CatalogRow] = []
    failure_identities: set[tuple[str, str]] = set()
    failure_messages: dict[tuple[str, str], str] = {}
    failure_providers: dict[tuple[str, str], str] = {}
    provider_error_events = 0
    fallback_count = 0
    current_route: str | None = None
    current_model: str | None = None
    cooldown_seconds = 0.0
    failure_summary: str | None = None
    state_lock = threading.RLock()
    output_snapshot_interval = getattr(provider, "preferred_output_snapshot_batches", 1)
    if (
        not isinstance(output_snapshot_interval, int)
        or isinstance(output_snapshot_interval, bool)
        or output_snapshot_interval <= 0
    ):
        output_snapshot_interval = 1
    successful_batch_commits = 0

    provider_chain = (provider, *tuple(fallback_providers))

    class _RouterCooldownDeferred(RuntimeError):
        def __init__(
            self,
            deferred_batch: tuple[_CatalogRow, ...],
            retries_remaining: int,
            retry_after_seconds: float,
        ) -> None:
            super().__init__("AI Router batch deferred until route cooldown expires")
            self.deferred_batch = deferred_batch
            self.retries_remaining = retries_remaining
            self.retry_after_seconds = max(0.0, float(retry_after_seconds))

    def mark_failed(
        batch: tuple[_CatalogRow, ...], error: Exception, provider_key: str
    ) -> None:
        message = str(error).strip() or type(error).__name__
        with state_lock:
            for representative in batch:
                for row in groups[representative.dedup_key]:
                    if row.identity not in failure_identities:
                        failures.append(row)
                        failure_identities.add(row.identity)
                    failure_messages[row.identity] = message
                    failure_providers[row.identity] = provider_key

    def run_batch(
        batch: tuple[_CatalogRow, ...],
        provider_index: int = 0,
        router_cooldown_retries: int = 1,
        defer_router_cooldown: bool = False,
    ) -> None:
        nonlocal requests, translated_now, completed, provider_error_events
        nonlocal fallback_count, current_route, current_model, cooldown_seconds
        nonlocal failure_summary, successful_batch_commits
        active_provider = provider_chain[provider_index]
        next_provider_index = provider_index + 1
        has_fallback = next_provider_index < len(provider_chain)
        protected = tuple(_protect_text(row.original) for row in batch)
        term_masks = []
        texts_to_send = []
        request_items: list[TranslationRequestItem] = []
        for item, row in zip(protected, batch, strict=True):
            if terminology is not None:
                term_text, term_meta, hard_terms = terminology.prepare_hard_terms(
                    scope.scope_id, item.value
                )
            else:
                term_text, term_meta = item.value, None
                hard_terms = ()
            term_masks.append(term_meta)
            texts_to_send.append(term_text)
            planned = planned_items.get(row.identity)
            request_items.append(
                TranslationRequestItem(
                    work_id=planned.work_id if planned is not None else f"w_{uuid.uuid4().hex[:12]}",
                    masked_text=term_text,
                    context=row.context,
                    context_signature=row.context_signature,
                    hard_terms=planned.hard_terms if planned is not None else hard_terms,
                    soft_terms=planned.soft_terms if planned is not None else (),
                )
            )
        with state_lock:
            requests += 1
        deferred_fallback: tuple[_CatalogRow, ...] = ()
        deferred_router_batch: tuple[_CatalogRow, ...] = ()
        try:
            try:
                provider_retry_attempts = retry_attempts if provider_index == 0 else 1
                preferred_retry_attempts = getattr(
                    active_provider, "preferred_retry_attempts", provider_retry_attempts
                )
                if (
                    isinstance(preferred_retry_attempts, int)
                    and not isinstance(preferred_retry_attempts, bool)
                    and preferred_retry_attempts > 0
                ):
                    provider_retry_attempts = min(
                        provider_retry_attempts, preferred_retry_attempts
                    )
                translated = _translate_with_retry(
                    active_provider,
                    tuple(texts_to_send),
                    source_language,
                    target_language,
                    provider_retry_attempts,
                    (
                        base_delay
                        if retry_base_delay is not None or provider_index == 0
                        else getattr(active_provider, "preferred_retry_base_delay", 1.0)
                    ),
                    sleeper,
                    tuple(request_items),
                    translation_control,
                )
            except PartialTranslationProviderError as error:
                with state_lock:
                    provider_error_events += 1
                successful_indexes = tuple(
                    index
                    for index, request_item in enumerate(request_items)
                    if request_item.work_id in error.successes
                )
                failed_indexes = tuple(
                    index
                    for index, request_item in enumerate(request_items)
                    if request_item.work_id not in error.successes
                )
                failed_batch = tuple(batch[index] for index in failed_indexes)
                if failed_batch and has_fallback:
                    deferred_fallback = failed_batch
                elif (
                    failed_batch
                    and active_provider.key == "ai_router_free"
                    and provider_index == 0
                    and defer_router_cooldown
                    and router_cooldown_retries > 0
                ):
                    deferred_router_batch = failed_batch
                else:
                    for index in failed_indexes:
                        request_item = request_items[index]
                        mark_failed(
                            (batch[index],),
                            TranslationProviderError(
                                error.failures.get(request_item.work_id, str(error))
                            ),
                            active_provider.key,
                        )
                if not successful_indexes:
                    if deferred_fallback:
                        with state_lock:
                            fallback_count += 1
                        run_batch(deferred_fallback, next_provider_index)
                    elif deferred_router_batch:
                        retry_delay = float(
                            getattr(
                                active_provider,
                                "preferred_retryable_deferred_seconds",
                                0.0,
                            )
                            or 0.0
                        )
                        if retry_delay > 0:
                            raise _RouterCooldownDeferred(
                                deferred_router_batch,
                                router_cooldown_retries - 1,
                                retry_delay,
                            )
                    return
                translated = tuple(
                    error.successes[request_items[index].work_id]
                    for index in successful_indexes
                )
                batch = tuple(batch[index] for index in successful_indexes)
                protected = tuple(protected[index] for index in successful_indexes)
                term_masks = [term_masks[index] for index in successful_indexes]
                request_items = [request_items[index] for index in successful_indexes]
            with state_lock:
                current_route = getattr(active_provider, "last_route_id", active_provider.key)
                current_model = getattr(active_provider, "last_model_id", current_model)
                fallback_count += int(
                    getattr(active_provider, "last_fallback_count", 0) or 0
                )
                cooldown_seconds = float(
                    getattr(active_provider, "last_cooldown_seconds", cooldown_seconds) or 0.0
                )
                failure_summary = getattr(
                    active_provider, "last_failure_summary", None
                )
            if len(translated) != len(batch):
                raise TranslationContractError(
                    "Provider result count does not match the requested batch."
                )
            valid_rows: list[_CatalogRow] = []
            restored_values: list[str] = []
            contract_failures: list[
                tuple[int, _CatalogRow, TranslationContractError]
            ] = []
            for index, (row, value, meta, protected_item) in enumerate(
                zip(batch, translated, term_masks, protected, strict=True)
            ):
                try:
                    restored_term_value = (
                        terminology.restore_hard_terms(value, meta)
                        if terminology is not None and meta is not None
                        else value
                    )
                except ValueError as error:
                    contract_failures.append(
                        (
                            index,
                            row,
                            TranslationContractError(
                                f"Hard glossary contract violation: {error}"
                            ),
                        )
                    )
                    continue
                try:
                    restored_value = _restore_text(restored_term_value, protected_item)
                    if not has_translatable_content(restored_value):
                        raise TranslationContractError(
                            "Translation provider returned no translated text beyond protected tokens."
                        )
                except TranslationContractError as error:
                    contract_failures.append((index, row, error))
                    continue
                valid_rows.append(row)
                restored_values.append(restored_value)

            if contract_failures:
                with state_lock:
                    provider_error_events += 1
                failed_rows = tuple(row for _index, row, _error in contract_failures)
                if has_fallback:
                    deferred_fallback = (*deferred_fallback, *failed_rows)
                else:
                    for failed_index, row, error in contract_failures:
                        term_text = texts_to_send[failed_index]
                        term_meta = term_masks[failed_index]
                        protected_item = protected[failed_index]
                        separators = list(protected_item.markers)
                        token_to_target = getattr(term_meta, "token_to_target", None)
                        if isinstance(token_to_target, Mapping):
                            separators.extend(str(token) for token in token_to_target)
                        separators = sorted(set(separators), key=len, reverse=True)
                        try:
                            if not separators:
                                raise error
                            splitter = re.compile(
                                "(" + "|".join(re.escape(token) for token in separators) + ")"
                            )
                            parts = splitter.split(term_text)
                            fragment_specs: list[tuple[int, str, str]] = []
                            fragments_list: list[str] = []
                            for part_index, part in enumerate(parts):
                                if not part or part in separators:
                                    continue
                                match = re.fullmatch(r"(\s*)(.*?)(\s*)", part, flags=re.DOTALL)
                                if match is None:
                                    continue
                                leading, core, trailing = match.groups()
                                if not core or not has_translatable_content(core):
                                    continue
                                fragment_specs.append((part_index, leading, trailing))
                                fragments_list.append(core)
                            fragments = tuple(fragments_list)
                            if not fragments:
                                raise error
                            with state_lock:
                                requests += 1
                            fragment_values = _translate_with_retry(
                                active_provider,
                                fragments,
                                source_language,
                                target_language,
                                1,
                                base_delay,
                                sleeper,
                                translation_control=translation_control,
                            )
                            if len(fragment_values) != len(fragment_specs):
                                raise TranslationContractError(
                                    "Fragment rescue result count does not match the requested fragments."
                                )
                            for (part_index, leading, trailing), fragment_value in zip(
                                fragment_specs, fragment_values, strict=True
                            ):
                                parts[part_index] = f"{leading}{fragment_value}{trailing}"
                            rescued = "".join(parts)
                            rescued = (
                                terminology.restore_hard_terms(rescued, term_meta)
                                if terminology is not None and term_meta is not None
                                else rescued
                            )
                            rescued = _restore_text(rescued, protected_item)
                            if not has_translatable_content(rescued):
                                raise TranslationContractError(
                                    "Fragment rescue returned no translated text beyond protected tokens."
                                )
                        except Exception as rescue_error:
                            mark_failed((row,), rescue_error, active_provider.key)
                            continue
                        valid_rows.append(row)
                        restored_values.append(rescued)
                if not valid_rows:
                    if deferred_fallback:
                        with state_lock:
                            fallback_count += 1
                        run_batch(deferred_fallback, next_provider_index)
                    return

            batch = tuple(valid_rows)
            restored = tuple(restored_values)
        except _RouterCooldownDeferred:
            raise
        except PauseRequested:
            return
        except RetryableTranslationProviderError as error:
            with state_lock:
                provider_error_events += 1
                cooldown_seconds = float(
                    getattr(error, "retry_after_seconds", None)
                    or getattr(active_provider, "last_cooldown_seconds", 0.0)
                    or 0.0
                )
                failure_summary = str(error).strip() or type(error).__name__
            if (
                active_provider.key == "ai_router_free"
                and provider_index == 0
                and router_cooldown_retries > 0
            ):
                if defer_router_cooldown:
                    retry_delay = cooldown_seconds or float(
                        getattr(
                            active_provider,
                            "preferred_retryable_deferred_seconds",
                            0.0,
                        )
                        or 0.0
                    )
                    if retry_delay > 0:
                        raise _RouterCooldownDeferred(
                            batch,
                            router_cooldown_retries - 1,
                            retry_delay,
                        )
                elif cooldown_seconds > 0:
                    remaining = cooldown_seconds
                    while remaining > 0:
                        if (
                            translation_control is not None
                            and translation_control.should_pause_after_request()
                        ):
                            return
                        interval = min(0.1, remaining)
                        sleeper(interval)
                        remaining -= interval
                    run_batch(
                        batch,
                        provider_index,
                        router_cooldown_retries=router_cooldown_retries - 1,
                        defer_router_cooldown=defer_router_cooldown,
                    )
                    return
            if has_fallback:
                with state_lock:
                    fallback_count += 1
                run_batch(batch, next_provider_index)
                return
            mark_failed(batch, error, active_provider.key)
            return
        except (TranslationResponseFormatError, TranslationContractError) as error:
            if len(batch) > 1:
                middle = len(batch) // 2
                run_batch(batch[:middle], provider_index)
                run_batch(batch[middle:], provider_index)
                return
            with state_lock:
                provider_error_events += 1
            if has_fallback:
                with state_lock:
                    fallback_count += 1
                run_batch(batch, next_provider_index)
                return
            mark_failed(batch, error, active_provider.key)
            return
        except TranslationProviderError as error:
            with state_lock:
                provider_error_events += 1
            if has_fallback:
                with state_lock:
                    fallback_count += 1
                run_batch(batch, next_provider_index)
                return
            if (
                active_provider.key == "ai_router_free"
                and provider_index == 0
                and defer_router_cooldown
                and router_cooldown_retries > 0
            ):
                retry_delay = float(
                    getattr(
                        active_provider,
                        "preferred_retryable_deferred_seconds",
                        0.0,
                    )
                    or 0.0
                )
                if retry_delay > 0:
                    raise _RouterCooldownDeferred(
                        batch,
                        router_cooldown_retries - 1,
                        retry_delay,
                    )
            mark_failed(batch, error, active_provider.key)
            return
        except (ValueError, TypeError) as error:
            with state_lock:
                provider_error_events += 1
            if has_fallback:
                with state_lock:
                    fallback_count += 1
                run_batch(batch, next_provider_index)
                return
            mark_failed(batch, error, active_provider.key)
            return
        except Exception as error:
            if len(batch) > 1:
                middle = len(batch) // 2
                run_batch(batch[:middle], provider_index)
                run_batch(batch[middle:], provider_index)
                return
            with state_lock:
                provider_error_events += 1
            if has_fallback:
                with state_lock:
                    fallback_count += 1
                run_batch(batch, next_provider_index)
                return
            mark_failed(batch, error, active_provider.key)
            return
        translated_rows: list[_CatalogRow] = []
        translated_values: list[str] = []
        for row, value in zip(batch, restored, strict=True):
            for grouped in groups.get(row.dedup_key, [row]):
                translated_rows.append(grouped)
                translated_values.append(value)

        # Network work may finish out of order, but durable state commits must be
        # serialized so checkpoint, SQLite TM, partial output, and progress remain
        # one coherent monotonic stream.
        with state_lock:
            # Commit Stage 1: Checkpoint
            _append_checkpoint_batch(checkpoint, translated_rows, translated_values)
            if on_commit_stage is not None:
                on_commit_stage("checkpoint")

            # Commit Stage 2: Database
            if knowledge_db is not None and lineage_id is not None:
                try:
                    now_t = time.time()
                    mem = TranslationMemoryRepository(knowledge_db)
                    segments_to_save = [
                        TranslationSegment(
                            segment_id=(
                                f"seg_{scope.scope_id}_{lineage_id}_{r.marker}_{r.key}_"
                                f"{hashlib.sha256(r.original.encode('utf-8')).hexdigest()}"
                            ),
                            scope_id=scope.scope_id,
                            lineage_id=lineage_id,
                            marker=r.marker,
                            key=r.key,
                            source_hash=hashlib.sha256(r.original.encode("utf-8")).hexdigest(),
                            source_text=r.original,
                            target_text=val,
                            context_signature=r.context_signature,
                            hard_terms_hash=hard_terms_hash_for(r),
                            created_at=now_t,
                            updated_at=now_t,
                        )
                        for r, val in zip(translated_rows, translated_values, strict=True)
                    ]
                    mem.save_segments(segments_to_save)
                except Exception as error:
                    _LOGGER.warning(
                        "Translation memory save failed: %s", type(error).__name__
                    )
            if on_commit_stage is not None:
                on_commit_stage("database")

            completed.update(
                {
                    row.identity: value
                    for row, value in zip(translated_rows, translated_values, strict=True)
                }
            )
            translated_now += len(translated_rows)
            successful_batch_commits += 1

            # Commit Stage 3: Partial output
            should_snapshot_output = (
                successful_batch_commits == 1
                or successful_batch_commits % output_snapshot_interval == 0
                or (
                    translation_control is not None
                    and translation_control.should_pause_after_request()
                )
            )
            if should_snapshot_output:
                partial_data = copy.deepcopy(data)
                partial_entries = partial_data["entries"]
                for r_item in rows:
                    if r_item.identity in completed:
                        partial_entries[r_item.index]["translation"] = completed[r_item.identity]
                        if is_delta:
                            partial_entries[r_item.index]["translation_state"] = "translated"
                try:
                    _atomic_text(
                        output, json.dumps(partial_data, ensure_ascii=False, indent=2) + "\n"
                    )
                except Exception:
                    pass
                if on_commit_stage is not None:
                    on_commit_stage("output")

            if progress is not None:
                progress(
                    AutoTranslateProgress(
                        completed_entries=len(completed),
                        total_entries=len(candidates),
                        request_count=requests,
                        pending_entries=max(0, len(candidates) - len(completed)),
                        translated_entries=translated_now,
                        resumed_entries=resumed,
                        tm_reused_entries=len(tm_reused_rows),
                        current_route=current_route,
                        current_model=current_model,
                        fallback_count=fallback_count,
                        cooldown_seconds=cooldown_seconds,
                        failure_summary=failure_summary,
                    )
                )
            if progress_reporter is not None:
                progress_reporter.publish(
                    state=ProgressState.RUNNING,
                    saved_entries=len(completed),
                    translated_entries=translated_now,
                    resumed_entries=resumed,
                    pending_entries=max(0, len(candidates) - len(completed)),
                    failure_count=len(failures),
                    request_count=requests,
                    current_route=current_route,
                    current_model=current_model,
                    fallback_count=fallback_count,
                    cooldown_seconds=cooldown_seconds,
                    failure_summary=failure_summary,
                )
        if deferred_router_batch:
            retry_delay = float(
                getattr(
                    active_provider,
                    "preferred_retryable_deferred_seconds",
                    0.0,
                )
                or 0.0
            )
            if retry_delay > 0:
                raise _RouterCooldownDeferred(
                    deferred_router_batch,
                    router_cooldown_retries - 1,
                    retry_delay,
                )
        if deferred_fallback:
            with state_lock:
                fallback_count += 1
            run_batch(deferred_fallback, next_provider_index)

    if batch_items <= 0 or batch_characters <= 0:
        raise ValueError("Auto Translate batch limits must be positive.")
    batchable: list[_CatalogRow] = []
    for row in representatives:
        if len(row.original) > batch_characters:
            mark_failed(
                (row,),
                TranslationContractError(
                    f"Text for marker + key {row.marker} + {row.key!r} exceeds "
                    f"the {batch_characters}-character provider limit."
                ),
                provider.key,
            )
        else:
            batchable.append(row)
    scheduled_batches = _batches(tuple(batchable), batch_items, batch_characters)
    preferred_outer_concurrency = getattr(provider, "preferred_outer_concurrency", 1)
    provider_key = getattr(provider, "key", "")
    use_parallel_router = (
        provider_key == "ai_router_free"
        and isinstance(preferred_outer_concurrency, int)
        and not isinstance(preferred_outer_concurrency, bool)
        and preferred_outer_concurrency > 1
        and len(scheduled_batches) > 1
    )
    use_parallel_custom = (
        isinstance(provider_key, str)
        and provider_key.startswith("openai-compatible-")
        and isinstance(preferred_outer_concurrency, int)
        and not isinstance(preferred_outer_concurrency, bool)
        and preferred_outer_concurrency > 1
        and len(scheduled_batches) > 1
    )
    if use_parallel_router:
        worker_count = min(preferred_outer_concurrency, len(scheduled_batches))
        fresh_batches = list(scheduled_batches)
        fresh_index = 0
        deferred_batches: list[
            tuple[float, int, tuple[_CatalogRow, ...], int]
        ] = []
        deferred_sequence = 0
        configured_deferred_retries = getattr(
            provider, "preferred_cooldown_deferred_retries", 1
        )
        if (
            not isinstance(configured_deferred_retries, int)
            or isinstance(configured_deferred_retries, bool)
            or configured_deferred_retries < 0
        ):
            configured_deferred_retries = 1
        configured_deferred_concurrency = getattr(
            provider, "preferred_deferred_outer_concurrency", worker_count
        )
        if (
            not isinstance(configured_deferred_concurrency, int)
            or isinstance(configured_deferred_concurrency, bool)
            or configured_deferred_concurrency <= 0
        ):
            configured_deferred_concurrency = worker_count
        deferred_worker_count = min(worker_count, configured_deferred_concurrency)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ai-router-batch",
        ) as executor:
            in_flight: set[concurrent.futures.Future[None]] = set()

            def submit_router_batch(
                queued_batch: tuple[_CatalogRow, ...],
                retries_remaining: int,
            ) -> None:
                in_flight.add(
                    executor.submit(
                        run_batch,
                        queued_batch,
                        0,
                        retries_remaining,
                        True,
                    )
                )

            def fill_router_workers() -> None:
                nonlocal fresh_index
                target_workers = (
                    worker_count
                    if fresh_index < len(fresh_batches)
                    else deferred_worker_count
                )
                while len(in_flight) < target_workers:
                    if (
                        translation_control is not None
                        and translation_control.should_pause_after_request()
                    ):
                        return
                    if fresh_index < len(fresh_batches):
                        queued_batch = fresh_batches[fresh_index]
                        fresh_index += 1
                        submit_router_batch(
                            queued_batch,
                            configured_deferred_retries,
                        )
                        continue
                    ref_now = time.monotonic()
                    ready_index = next(
                        (
                            index
                            for index, item in enumerate(deferred_batches)
                            if item[0] <= ref_now
                        ),
                        None,
                    )
                    if ready_index is not None:
                        _ready_at, _sequence, queued_batch, retries = deferred_batches.pop(
                            ready_index
                        )
                        submit_router_batch(queued_batch, retries)
                        continue
                    return

            fill_router_workers()
            while in_flight or fresh_index < len(fresh_batches) or deferred_batches:
                fill_router_workers()
                if not in_flight:
                    if (
                        translation_control is not None
                        and translation_control.should_pause_after_request()
                    ):
                        break
                    if deferred_batches:
                        next_ready = min(item[0] for item in deferred_batches)
                        remaining = max(0.0, next_ready - time.monotonic())
                        if remaining > 0:
                            sleeper(min(0.1, remaining))
                        continue
                    break

                done, pending = concurrent.futures.wait(
                    in_flight,
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                in_flight = set(pending)
                for future in done:
                    try:
                        future.result()
                    except _RouterCooldownDeferred as deferred:
                        deferred_sequence += 1
                        deferred_batches.append(
                            (
                                time.monotonic() + deferred.retry_after_seconds,
                                deferred_sequence,
                                deferred.deferred_batch,
                                deferred.retries_remaining,
                            )
                        )
                fill_router_workers()
    elif use_parallel_custom:
        worker_count = min(preferred_outer_concurrency, len(scheduled_batches))
        pending_batches = iter(scheduled_batches)
        exhausted = False
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="custom-openai-batch",
        ) as executor:
            in_flight: set[concurrent.futures.Future[None]] = set()

            def fill_custom_workers() -> None:
                nonlocal exhausted
                while not exhausted and len(in_flight) < worker_count:
                    if (
                        translation_control is not None
                        and translation_control.should_pause_after_request()
                    ):
                        return
                    try:
                        queued_batch = next(pending_batches)
                    except StopIteration:
                        exhausted = True
                        return
                    in_flight.add(executor.submit(run_batch, queued_batch))

            fill_custom_workers()
            while in_flight:
                done, pending = concurrent.futures.wait(
                    in_flight,
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                in_flight = set(pending)
                for future in done:
                    future.result()
                fill_custom_workers()
    else:
        consecutive_provider_failures = 0
        for batch in scheduled_batches:
            completed_before = len(completed)
            provider_errors_before = provider_error_events
            run_batch(batch)
            if translation_control is not None and translation_control.should_pause_after_request():
                break
            if len(completed) > completed_before:
                consecutive_provider_failures = 0
            elif provider_error_events > provider_errors_before:
                consecutive_provider_failures += 1
            else:
                consecutive_provider_failures = 0
            if consecutive_provider_failures >= DEFAULT_PROVIDER_FAILURE_CIRCUIT:
                break
    failure_path = failure_path_for(output)
    _rewrite_failures(failure_path, failures, failure_providers, failure_messages)
    if on_commit_stage is not None:
        on_commit_stage("failure_state")
    output_data = copy.deepcopy(data)
    output_entries = output_data["entries"]
    for row in rows:
        if row.identity in completed:
            output_entries[row.index]["translation"] = completed[row.identity]
            if is_delta:
                output_entries[row.index]["translation_state"] = "translated"
        elif is_delta and output_entries[row.index].get("translation_state") not in {
            "skipped_technical",
            "translated",
        }:
            output_entries[row.index]["translation_state"] = (
                "failed" if row.identity in failure_identities else "pending"
            )
    backup: Path | None = None
    if output.is_file():
        backup = _backup_path(output)
        os.replace(output, backup)
    try:
        _atomic_text(
            output, json.dumps(output_data, ensure_ascii=False, indent=2) + "\n"
        )
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
            backup = None
        raise
    pending_count = sum(1 for row in candidates if row.identity not in completed)
    if progress_reporter is not None:
        if translation_control is not None and translation_control.pause_requested:
            terminal_state = ProgressState.PAUSED
        elif pending_count:
            terminal_state = ProgressState.COMPLETED_WITH_PENDING
        else:
            terminal_state = ProgressState.COMPLETED
        progress_reporter.publish(
            state=terminal_state,
            saved_entries=len(completed),
            translated_entries=translated_now,
            resumed_entries=resumed,
            pending_entries=pending_count,
            failure_count=len(failures),
            request_count=requests,
            current_route=current_route,
            current_model=current_model,
            fallback_count=fallback_count,
            cooldown_seconds=cooldown_seconds,
            failure_summary=(
                failure_summary
                or (next(iter(failure_messages.values()), None) if failures else None)
            ),
        )
    return AutoTranslateResult(
        output,
        backup,
        checkpoint,
        translated_now,
        resumed,
        max(0, len(rows) - len(candidates)),
        sum(1 for row in rows if not row.original),
        sum(
            1
            for row in rows
            if row.original and not has_translatable_content(row.original)
        ),
        reused_duplicates,
        requests,
        provider.key,
        pending_count,
        len(failures),
        failure_path if failures else None,
        len(migrated),
        len(tm_reused_rows),
    )


def translate_delta(
    source_json: Path | str,
    output_json: Path | str,
    provider: TranslationProvider,
    source_language: str,
    target_language: str = "vi",
    *,
    max_batch_items: int | None = None,
    max_batch_characters: int | None = None,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[AutoTranslateProgress], None] | None = None,
    legacy_checkpoint: Path | str | None = None,
    on_commit_stage: Callable[[str], None] | None = None,
    db_path: Path | str | None = None,
    translation_control: TranslationControl | None = None,
    progress_reporter: ProgressReporter | None = None,
    job_id: str | None = None,
    fallback_providers: Sequence[TranslationProvider] = (),
) -> AutoTranslateResult:
    """Translate only a sparse delta created by the compare step."""

    source = Path(source_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Auto Translate source JSON does not exist: {source}")
    if not _is_delta(read_paloc_json(source)):
        raise ValueError(
            "Step 2 requires a sparse translation delta created by Step 1."
        )
    return translate_catalog(
        source,
        output_json,
        provider,
        source_language,
        target_language,
        max_batch_items=max_batch_items,
        max_batch_characters=max_batch_characters,
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        sleeper=sleeper,
        progress=progress,
        legacy_checkpoint=legacy_checkpoint,
        on_commit_stage=on_commit_stage,
        db_path=db_path,
        translation_control=translation_control,
        progress_reporter=progress_reporter,
        job_id=job_id,
        fallback_providers=fallback_providers,
    )


def translate_catalog_with_provider(
    source_json: Path | str,
    output_json: Path | str,
    provider_key: str,
    source_language: str,
    target_language: str,
    provider_config: Mapping[str, str] | None = None,
    progress: Callable[[AutoTranslateProgress], None] | None = None,
    translation_control: TranslationControl | None = None,
    progress_reporter: ProgressReporter | None = None,
    job_id: str | None = None,
) -> AutoTranslateResult:
    provider_chain = _create_automatic_provider_chain(provider_key, provider_config)
    return translate_catalog(
        source_json,
        output_json,
        provider_chain[0],
        source_language,
        target_language,
        progress=progress,
        translation_control=translation_control,
        progress_reporter=progress_reporter,
        job_id=job_id,
        fallback_providers=provider_chain[1:],
    )


def translate_delta_with_provider(
    source_json: Path | str,
    output_json: Path | str,
    provider_key: str,
    source_language: str,
    target_language: str,
    provider_config: Mapping[str, str] | None = None,
    progress: Callable[[AutoTranslateProgress], None] | None = None,
    legacy_checkpoint: Path | str | None = None,
    on_commit_stage: Callable[[str], None] | None = None,
    db_path: Path | str | None = None,
    translation_control: TranslationControl | None = None,
    progress_reporter: ProgressReporter | None = None,
    job_id: str | None = None,
) -> AutoTranslateResult:
    """Translate a sparse delta, rejecting legacy full-catalog work files."""

    source = Path(source_json).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Auto Translate source JSON does not exist: {source}")
    if not _is_delta(read_paloc_json(source)):
        raise ValueError(
            "Step 2 requires a sparse translation delta created by Step 1."
        )
    provider_chain = _create_automatic_provider_chain(provider_key, provider_config)
    return translate_catalog(
        source,
        output_json,
        provider_chain[0],
        source_language,
        target_language,
        progress=progress,
        legacy_checkpoint=legacy_checkpoint,
        on_commit_stage=on_commit_stage,
        db_path=db_path,
        translation_control=translation_control,
        progress_reporter=progress_reporter,
        job_id=job_id,
        fallback_providers=provider_chain[1:],
    )
