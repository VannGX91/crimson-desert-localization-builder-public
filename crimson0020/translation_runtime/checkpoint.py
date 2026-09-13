"""Checkpoint version 2 module for durable, provider-portable resume."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from crimson0020.translation_runtime.errors import CheckpointError

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Individual translation item record stored in a checkpoint batch."""

    marker: str
    key: str
    original_sha256: str
    translation: str
    route_id: str | None = None
    model_id: str | None = None
    context_signature: str | None = None
    glossary_revision: int | None = None
    quality_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "marker": self.marker,
            "key": self.key,
            "original_sha256": self.original_sha256,
            "translation": self.translation,
        }
        if self.route_id is not None:
            data["route_id"] = self.route_id
        if self.model_id is not None:
            data["model_id"] = self.model_id
        if self.context_signature is not None:
            data["context_signature"] = self.context_signature
        if self.glossary_revision is not None:
            data["glossary_revision"] = self.glossary_revision
        if self.quality_flags:
            data["quality_flags"] = list(self.quality_flags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointRecord:
        return cls(
            marker=data["marker"],
            key=data["key"],
            original_sha256=data["original_sha256"],
            translation=data["translation"],
            route_id=data.get("route_id"),
            model_id=data.get("model_id"),
            context_signature=data.get("context_signature"),
            glossary_revision=data.get("glossary_revision"),
            quality_flags=tuple(data.get("quality_flags", ())),
        )


def _atomic_replace_text(path: Path, text: str) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        temp.replace(path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def initialize_checkpoint(
    path: Path,
    source_fingerprint: str,
    source_language: str,
    target_language: str,
    provider: str = "ai_router_free",
) -> None:
    """Initializes a new checkpoint file with version 2 header."""
    header = {
        "record": "header",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "source_fingerprint": source_fingerprint,
        "provider": provider,
        "source_language": source_language,
        "target_language": target_language,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_replace_text(
            path, json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    except OSError as error:
        raise CheckpointError(
            f"Cannot initialize translation checkpoint: {error}"
        ) from error


def append_checkpoint_batch(
    path: Path, records: Sequence[CheckpointRecord | dict[str, Any]]
) -> None:
    """Appends a batch of translated records to the checkpoint file, flushing and fsyncing."""
    if not records:
        return

    serialized_translations = []
    for r in records:
        if isinstance(r, CheckpointRecord):
            serialized_translations.append(r.to_dict())
        elif isinstance(r, dict):
            serialized_translations.append(r)
        else:
            raise CheckpointError(f"Unsupported record type: {type(r)}")

    batch_entry = {
        "record": "batch",
        "translations": serialized_translations,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(batch_entry, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise CheckpointError(
            f"Cannot append completed translation batch: {error}"
        ) from error


def _candidate_context_signature(candidate: Any) -> str:
    if isinstance(candidate, dict):
        raw_signature = candidate.get("context_signature", "")
        raw_context = candidate.get("context", ())
    else:
        raw_signature = getattr(candidate, "context_signature", "")
        raw_context = getattr(candidate, "context", ())
    if raw_signature is None:
        raw_signature = ""
    if not isinstance(raw_signature, str):
        raise CheckpointError("Candidate context_signature must be text.")
    if raw_signature:
        return raw_signature
    if raw_context is None:
        return ""
    if isinstance(raw_context, str):
        context = (raw_context,)
    elif isinstance(raw_context, (list, tuple)) and all(
        isinstance(part, str) for part in raw_context
    ):
        context = tuple(raw_context)
    else:
        raise CheckpointError("Candidate context must be text or a sequence of text.")
    if not context:
        return ""
    return hashlib.sha256(
        ("context\0" + "\0".join(context)).encode("utf-8")
    ).hexdigest()


def load_checkpoint(
    path: Path,
    source_fingerprint: str,
    source_language: str,
    target_language: str,
    candidates: Sequence[Any],
) -> dict[tuple[str, str], str]:
    """Loads and validates a checkpoint file for provider-portable resume.

    Validates header, line integrity, identity uniqueness, and contracts.
    Repairs truncated tails automatically.
    """
    if not path.is_file():
        return {}

    try:
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        header = json.loads(lines[0]) if lines else None
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError) as error:
        raise CheckpointError(f"Cannot read checkpoint: {error}") from error

    expected_header = {
        "record": "header",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "source_fingerprint": source_fingerprint,
        "source_language": source_language,
        "target_language": target_language,
    }

    if not isinstance(header, dict) or any(
        header.get(k) != v for k, v in expected_header.items()
    ):
        raise CheckpointError(
            "Checkpoint does not match the selected source or language pair."
        )

    # Build candidates lookup
    by_identity: dict[tuple[str, str], Any] = {}
    for item in candidates:
        if isinstance(item, dict):
            m = item.get("marker", "")
            k = item.get("key", "")
            by_identity[(m, k)] = item
        else:
            m = getattr(item, "marker", "")
            k = getattr(item, "key", "")
            by_identity[(m, k)] = item

    completed: dict[tuple[str, str], str] = {}
    has_trailing_newline = text.endswith(("\n", "\r"))
    valid_lines = [lines[0]]

    for line_number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # If it's the last line and no trailing newline, it's an incomplete tail
            if line_number == len(lines) and not has_trailing_newline:
                break
            raise CheckpointError(
                f"Checkpoint record on line {line_number} is invalid JSON."
            )

        if (
            not isinstance(record, dict)
            or record.get("record") != "batch"
            or not isinstance(record.get("translations"), list)
        ):
            raise CheckpointError(
                f"Checkpoint record on line {line_number} is invalid."
            )

        for raw in record["translations"]:
            if not isinstance(raw, dict):
                raise CheckpointError(
                    f"Checkpoint record on line {line_number} has an invalid translation."
                )

            m = raw.get("marker")
            k = raw.get("key")
            if not isinstance(m, str) or not isinstance(k, str):
                raise CheckpointError(
                    f"Checkpoint record on line {line_number} has invalid marker/key."
                )

            identity = (m, k)
            if identity in completed:
                raise CheckpointError(
                    f"Checkpoint record on line {line_number} has an unknown or duplicate identity: {identity}"
                )

            candidate = by_identity.get(identity)
            translation = raw.get("translation")
            if not isinstance(translation, str):
                raise CheckpointError(
                    f"Checkpoint record on line {line_number} translation is not a string."
                )

            if candidate is None:
                raise CheckpointError(
                    f"Checkpoint record on line {line_number} has an unknown identity: {identity}"
                )

            original = (
                candidate["original"]
                if isinstance(candidate, dict)
                else getattr(candidate, "original", "")
            )
            expected_hash = hashlib.sha256(original.encode("utf-8")).hexdigest().upper()
            if raw.get("original_sha256") != expected_hash:
                raise CheckpointError(
                    f"Checkpoint record on line {line_number} original_sha256 does not match candidate."
                )

            expected_context_signature = _candidate_context_signature(candidate)
            if "context_signature" in raw:
                raw_context_signature = raw.get("context_signature")
                if not isinstance(raw_context_signature, str):
                    raise CheckpointError(
                        f"Checkpoint record on line {line_number} context_signature is not text."
                    )
                if raw_context_signature != expected_context_signature:
                    continue
            elif expected_context_signature:
                continue

            completed[identity] = translation
        valid_lines.append(line)

    # Check if tail was truncated and needs repair
    if not has_trailing_newline or len(valid_lines) < len(lines):
        try:
            _atomic_replace_text(path, "\n".join(valid_lines) + "\n")
        except OSError as error:
            raise CheckpointError(
                f"Cannot repair incomplete checkpoint tail: {error}"
            ) from error

    return completed
