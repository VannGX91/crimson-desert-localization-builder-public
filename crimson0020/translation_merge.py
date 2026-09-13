"""Validation, deterministic repair, and merge for translation deltas."""

import copy
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auto_translate import (
    _catalog_rows,
    _contract_tokens,
    _source_fingerprint,
    has_translatable_content,
)
from .paloc import read_paloc_json
from .translation_work import read_translation_delta


@dataclass(frozen=True)
class TranslationDeltaValidationResult:
    valid: bool
    complete: bool
    template_json: Path
    translated_json: Path
    entry_count: int
    translated_entries: int
    pending_entries: int
    issues: tuple[str, ...] = ()
    repairable: bool = False


@dataclass(frozen=True)
class TranslationDeltaFixResult:
    output_json: Path
    entry_count: int
    repaired_entries: int
    pending_entries: int
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationMergeResult:
    output_json: Path
    backup_json: Path | None
    merged_entries: int
    unchanged_entries: int
    reused_unchanged_entries: int = 0
    merged_delta_entries: int = 0
    source_fallback_entries: int = 0


def _path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def _read(path: Path | str) -> dict[str, Any]:
    return read_translation_delta(_path(path))


def _translation_is_valid(original: str, translation: object) -> bool:
    return (
        isinstance(translation, str)
        and bool(translation.strip())
        and has_translatable_content(translation)
        and _contract_tokens(original) == _contract_tokens(translation)
    )


def _quality_flags(entry: dict[str, Any]) -> tuple[str, ...]:
    flags = entry.get("quality_flags", ())
    if isinstance(flags, str):
        return (flags,)
    if isinstance(flags, (list, tuple, set, frozenset)):
        return tuple(flag for flag in flags if isinstance(flag, str))
    return ()


def _protected_content_matches(original: str, translation: object) -> bool:
    return isinstance(translation, str) and _contract_tokens(original) == _contract_tokens(translation)


def _old_translation_is_reusable(original: str, translation: object) -> bool:
    if not isinstance(translation, str):
        return False
    if _contract_tokens(original) != _contract_tokens(translation):
        return False
    if not has_translatable_content(original):
        return translation == original
    return bool(translation.strip()) and translation != original


def _source_matches(new_source: Path, template: dict[str, Any]) -> bool:
    new_data = read_paloc_json(new_source)
    new_rows = _catalog_rows(new_data)
    source = template.get("source")
    return isinstance(source, dict) and source.get(
        "catalog_fingerprint"
    ) == _source_fingerprint(new_data, new_rows)


def _validate_internal(
    new_source: Path, template_path: Path, translated_path: Path
) -> TranslationDeltaValidationResult:
    issues: list[str] = []
    try:
        new_data = read_paloc_json(new_source)
        new_rows = _catalog_rows(new_data)
    except Exception as error:
        return TranslationDeltaValidationResult(
            False, False, template_path, translated_path, 0, 0, 0, (str(error),)
        )
    try:
        template = _read(template_path)
    except Exception as error:
        return TranslationDeltaValidationResult(
            False,
            False,
            template_path,
            translated_path,
            0,
            0,
            0,
            (f"Trusted translation template is invalid: {error}",),
            False,
        )
    expected_fingerprint = _source_fingerprint(new_data, new_rows)
    template_source = template.get("source", {})
    template_entries = template["entries"]
    template_matches_source = (
        template_source.get("catalog_fingerprint") == expected_fingerprint
    )
    if not template_matches_source:
        issues.append("Template does not match the selected new source catalog.")
    try:
        translated = _read(translated_path)
    except Exception as error:
        issues.append(str(error))
        return TranslationDeltaValidationResult(
            False,
            False,
            template_path,
            translated_path,
            len(template_entries),
            0,
            len(template_entries),
            tuple(issues),
            template_matches_source,
        )
    translated_source = translated.get("source", {})
    if translated_source.get("catalog_fingerprint") != expected_fingerprint:
        issues.append(
            "Translated delta does not match the selected new source catalog."
        )
    if translated_source != template_source:
        issues.append("Translated delta source metadata does not match the delta template.")
    translated_entries = translated["entries"]
    expected_order = [
        (entry["marker"], entry["key"])
        for entry in template_entries
    ]
    translated_order = [
        (entry["marker"], entry["key"])
        for entry in translated_entries
    ]
    if translated_order != expected_order:
        issues.append("Translated delta identity order does not match the delta template.")
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for index, entry in enumerate(translated_entries):
        identity = (entry["marker"], entry["key"])
        if identity in by_identity:
            issues.append(
                f"Translated delta contains duplicate identity at row {index}."
            )
        by_identity[identity] = entry
    expected: dict[tuple[str, str], dict[str, Any]] = {
        (entry["marker"], entry["key"]): entry for entry in template_entries
    }
    missing = [identity for identity in expected if identity not in by_identity]
    extra = [identity for identity in by_identity if identity not in expected]
    if missing:
        issues.append(f"Translated delta is missing {len(missing)} template row(s).")
    if extra:
        issues.append(f"Translated delta contains {len(extra)} unknown row(s).")
    valid_translated = 0
    pending = 0
    for identity, template_entry in expected.items():
        entry = by_identity.get(identity)
        if entry is None:
            pending += 1
            continue
        for field in ("original", "change", "context", "context_signature"):
            if entry.get(field) != template_entry.get(field):
                issues.append(
                    f"Immutable field {field!r} changed for {identity[0]} + {identity[1]!r}."
                )
        state = entry.get("translation_state", "pending")
        if "glossary_violation" in _quality_flags(entry):
            pending += 1
            continue
        if state == "skipped_technical":
            if entry.get("translation") != template_entry.get("original"):
                issues.append(
                    f"Technical row {identity[0]} + {identity[1]!r} was modified."
                )
            valid_translated += 1
        elif state == "translated":
            translation = entry.get("translation")
            if _translation_is_valid(template_entry["original"], translation):
                valid_translated += 1
            else:
                pending += 1
                if (
                    isinstance(translation, str)
                    and bool(translation.strip())
                    and not _protected_content_matches(
                        template_entry["original"], translation
                    )
                ):
                    issues.append(
                        f"Protected content changed for {identity[0]} + {identity[1]!r}."
                    )
        else:
            pending += 1
    structure_valid = not issues
    if pending:
        issues.append(f"Translated delta still has {pending} pending/invalid row(s).")
    return TranslationDeltaValidationResult(
        structure_valid,
        structure_valid and pending == 0 and valid_translated == len(expected),
        template_path,
        translated_path,
        len(expected),
        valid_translated,
        pending,
        tuple(issues),
        bool(not structure_valid and template_matches_source),
    )


def validate_translation_delta(
    new_source: Path | str, template: Path | str, translated: Path | str
) -> TranslationDeltaValidationResult:
    """Validate identity, immutable source text, completeness and contracts."""
    return _validate_internal(_path(new_source), _path(template), _path(translated))


def _valid_checkpoint_rows(
    path: Path | None,
    template_entries: list[dict[str, Any]],
    expected_fingerprint: str,
    source_language: str | None,
    target_language: str,
) -> tuple[dict[tuple[str, str], str], str | None]:
    if path is None or not path.is_file():
        return {}, None
    expected = {(entry["marker"], entry["key"]): entry for entry in template_entries}
    found: dict[tuple[str, str], str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        header = json.loads(lines[0]) if lines else None
        language_mismatch = not isinstance(header, dict) or (
            header.get("target_language") != target_language
            or (
                source_language is not None
                and header.get("source_language") != source_language
            )
        )
        if (
            not isinstance(header, dict)
            or header.get("record") != "header"
            or header.get("format_version") != 2
            or header.get("source_fingerprint") != expected_fingerprint
            or language_mismatch
        ):
            return (
                {},
                "Checkpoint header or language pair does not match the trusted "
                "translation job.",
            )
        for line in lines[1:]:
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("record") != "batch":
                continue
            for raw in record.get("translations", []):
                if not isinstance(raw, dict):
                    continue
                identity = (raw.get("marker"), raw.get("key"))
                entry = expected.get(identity)
                translation = raw.get("translation")
                if (
                    entry is None
                    or identity in found
                    or not _translation_is_valid(entry["original"], translation)
                ):
                    continue
                if "context_signature" in raw:
                    raw_context_signature = raw.get("context_signature")
                    expected_context_signature = entry.get("context_signature", "")
                    if (
                        not isinstance(raw_context_signature, str)
                        or raw_context_signature != expected_context_signature
                    ):
                        continue
                elif entry.get("context_signature") or entry.get("context"):
                    continue
                digest = (
                    hashlib.sha256(entry["original"].encode("utf-8"))
                    .hexdigest()
                    .upper()
                )
                if raw.get("original_sha256") == digest:
                    found[identity] = translation
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return found, "Checkpoint could not be read safely."
    return found, None


def _atomic_write(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalized_row_identity(entry: object) -> tuple[str, str] | None:
    if not isinstance(entry, dict):
        return None
    marker = entry.get("marker")
    key = entry.get("key")
    if not isinstance(marker, str) or not isinstance(key, str):
        return None
    try:
        marker_bytes = bytes.fromhex(marker)
    except ValueError:
        return None
    if len(marker_bytes) != 8:
        return None
    return marker_bytes.hex(), key


def _read_repair_candidates(path: Path) -> tuple[list[object], str | None]:
    """Read a damaged delta permissively so valid rows can still be salvaged."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [], f"Damaged translated delta could not be parsed: {error}"
    if not isinstance(raw, dict):
        return [], "Damaged translated delta root is not an object."
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return [], "Damaged translated delta does not contain an entries list."
    return entries, None


def _repair_translation_candidate(
    expected: dict[str, Any], raw_entry: object
) -> str | None:
    """Return text that is safe to transplant into the trusted template row."""
    if not isinstance(raw_entry, dict):
        return None
    if "glossary_violation" in _quality_flags(raw_entry):
        return None
    if raw_entry.get("original") != expected.get("original"):
        return None
    if raw_entry.get("context") != expected.get("context"):
        return None
    if raw_entry.get("context_signature") != expected.get("context_signature"):
        return None
    translation = raw_entry.get("translation")
    original = expected["original"]
    if not has_translatable_content(original):
        return original if translation == original else None
    return translation if _translation_is_valid(original, translation) else None


def fix_translation_delta_structure(
    template: Path | str,
    translated: Path | str,
    checkpoint: Path | str | None,
    repaired: Path | str,
    source_language: str | None = None,
    target_language: str = "vi",
) -> TranslationDeltaFixResult:
    """Reconstruct a damaged delta from immutable template rows and valid text only."""
    template_path, translated_path, output = (
        _path(template),
        _path(translated),
        _path(repaired),
    )
    checkpoint_path = _path(checkpoint) if checkpoint else None
    input_paths = {template_path, translated_path}
    if checkpoint_path is not None:
        input_paths.add(checkpoint_path)
    if output in input_paths:
        raise ValueError("Repaired delta output must be separate from its inputs.")
    if output.exists():
        raise FileExistsError(f"Repaired delta output already exists: {output}")
    try:
        template_data = _read(template_path)
    except ValueError as error:
        raise ValueError(f"Trusted translation template is invalid: {error}") from error
    template_entries = template_data["entries"]
    expected_by_identity = {
        _normalized_row_identity(entry): entry for entry in template_entries
    }
    valid: dict[tuple[str, str], str] = {}
    issues: list[str] = []
    damaged_entries, damaged_issue = _read_repair_candidates(translated_path)
    if damaged_issue:
        issues.append(damaged_issue)
    candidates_by_identity: dict[tuple[str, str], list[str]] = {}
    ignored_rows = 0
    structurally_damaged_rows = 0
    for raw_entry in damaged_entries:
        identity = _normalized_row_identity(raw_entry)
        expected = expected_by_identity.get(identity) if identity is not None else None
        if expected is None:
            ignored_rows += 1
            continue
        if isinstance(raw_entry, dict) and any(
            raw_entry.get(field) != expected.get(field)
            for field in ("original", "change", "context", "context_signature")
        ):
            structurally_damaged_rows += 1
        candidate = _repair_translation_candidate(expected, raw_entry)
        if candidate is None:
            ignored_rows += 1
            continue
        candidates_by_identity.setdefault(identity, []).append(candidate)
    conflicting_rows = 0
    duplicate_rows = 0
    for identity, candidates in candidates_by_identity.items():
        unique_candidates = tuple(dict.fromkeys(candidates))
        if len(candidates) > 1:
            duplicate_rows += len(candidates) - 1
        if len(unique_candidates) == 1:
            valid[identity] = unique_candidates[0]
        else:
            conflicting_rows += 1
    if structurally_damaged_rows:
        issues.append(
            f"Restored trusted immutable metadata for {structurally_damaged_rows} row(s)."
        )
    if duplicate_rows:
        issues.append(
            f"Collapsed {duplicate_rows} duplicate row(s) while rebuilding the delta."
        )
    if conflicting_rows:
        issues.append(
            f"Left {conflicting_rows} identity with conflicting valid translations pending."
        )
    if ignored_rows:
        issues.append(
            f"Ignored {ignored_rows} unsafe/unknown row(s) while rebuilding the delta."
        )
    checkpoint_rows, checkpoint_issue = _valid_checkpoint_rows(
        checkpoint_path,
        template_entries,
        template_data["source"]["catalog_fingerprint"],
        source_language,
        target_language,
    )
    if checkpoint_issue:
        issues.append(checkpoint_issue)
    valid.update(
        {
            identity: value
            for identity, value in checkpoint_rows.items()
            if identity not in valid
        }
    )
    repaired_data = copy.deepcopy(template_data)
    repaired_count = 0
    pending = 0
    for entry in repaired_data["entries"]:
        identity = _normalized_row_identity(entry)
        value = valid.get(identity)
        if value is not None:
            entry["translation"] = value
            entry["translation_state"] = (
                "skipped_technical"
                if value == entry["original"] and not has_translatable_content(value)
                else "translated"
            )
            repaired_count += 1
        else:
            entry["translation"] = ""
            entry["translation_state"] = "pending"
            pending += 1
    _atomic_write(repaired_data, output)
    return TranslationDeltaFixResult(
        output, len(template_entries), repaired_count, pending, tuple(issues)
    )


def merge_translation_delta(
    old_translated: Path | str,
    new_source: Path | str,
    template: Path | str,
    translated: Path | str,
    final: Path | str,
) -> TranslationMergeResult:
    """Apply a complete sparse delta while carrying forward unchanged translations.

    The new catalog owns structure, identity order, metadata, and source text.
    The old catalog is consulted only for an identity whose source text is
    byte-for-byte equal in both catalogs.  New and source-changed rows can only
    be supplied by the validated sparse delta.
    """
    old_path = _path(old_translated)
    new_path = _path(new_source)
    template_path = _path(template)
    translated_path = _path(translated)
    output = _path(final)
    if output in {old_path, new_path, template_path, translated_path}:
        raise ValueError("Merged JSON output must be separate from every input.")
    if output.exists():
        raise FileExistsError(f"Merged JSON output already exists: {output}")
    result = validate_translation_delta(new_path, template_path, translated_path)
    if not result.valid or not result.complete:
        raise ValueError(
            "Cannot merge an invalid or incomplete translation delta: "
            + "; ".join(result.issues)
        )
    old_data = read_paloc_json(old_path)
    source_data = read_paloc_json(new_path)
    old_rows = _catalog_rows(old_data)
    source_rows = _catalog_rows(source_data)
    template_data = _read(template_path)
    expected_old_fingerprint = template_data["source"].get("base_catalog_fingerprint")
    actual_old_fingerprint = _source_fingerprint(old_data, old_rows)
    if expected_old_fingerprint != actual_old_fingerprint:
        raise ValueError(
            "The selected old translated catalog does not match the one used "
            "to create this translation delta."
        )
    old_by_identity = {row.identity: row for row in old_rows}
    translated_data = _read(translated_path)
    translations = {
        (entry["marker"], entry["key"]): entry["translation"]
        for entry in translated_data["entries"]
    }
    output_data = copy.deepcopy(source_data)
    changed = 0
    reused = 0
    delta_merged = 0
    fallback = 0
    for row in source_rows:
        delta_value = translations.get(row.identity)
        old_row = old_by_identity.get(row.identity)
        if delta_value is not None:
            output_data["entries"][row.index]["translation"] = delta_value
            changed += 1
            delta_merged += 1
        elif (
            old_row is not None
            and old_row.original == row.original
            and _old_translation_is_reusable(old_row.original, old_row.translation)
        ):
            output_data["entries"][row.index]["translation"] = old_row.translation
            changed += 1
            reused += 1
        else:
            fallback += 1
    _atomic_write(output_data, output)
    return TranslationMergeResult(
        output,
        None,
        changed,
        len(source_rows) - changed,
        reused,
        delta_merged,
        fallback,
    )
