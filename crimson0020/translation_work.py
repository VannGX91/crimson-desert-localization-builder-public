"""Build the small, reviewable translation delta used by Auto Translate."""

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auto_translate import _catalog_rows, _source_fingerprint, has_translatable_content
from .paloc import read_paloc_json
from .translation_runtime.knowledge.context import ContextEngine, ContextItem, ContextKind

DELTA_SCHEMA = "crimson.translation-delta"
DELTA_SCHEMA_VERSION = 1
_LEGACY_HARD_TERM_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9'’-]*(?: [A-Za-z][A-Za-z0-9'’-]*)?$"
)
_COMMON_SYSTEM_WORDS = frozenset(
    {
        "save", "load", "settings", "setting", "option", "options", "menu", "back", "exit", "quit",
        "play", "game", "new game", "continue", "retry", "game over", "yes", "no", "ok", "on", "off",
        "resume", "pause", "sound", "graphics", "display", "audio", "music", "video", "controls", "control",
        "system", "status", "level", "equipment", "equip", "skill", "skills", "party", "inventory", "item",
        "items", "quest", "quests", "mission", "map", "help", "info", "information", "shop", "buy", "sell",
        "remove", "accept", "decline", "cancel", "confirm", "close", "open", "start", "select", "delete",
        "create", "details", "search", "filter", "sort", "total", "price", "cost", "rank", "stage", "wave",
        "round", "victory", "defeat", "target", "score", "result", "reward", "rewards", "bonus", "time",
        "speed", "normal", "hard", "easy", "auto", "manual", "default", "reset", "clear", "apply", "action",
        "command", "attack", "defend", "guard", "escape", "flee", "run", "weapon", "armor", "shield", "helmet",
        "accessory", "boots", "gloves", "ring", "necklace", "potion", "gold", "silver", "copper", "money",
    }
)


def _legacy_term_candidate(source: str, target: str) -> bool:
    """Return whether a legacy row is safe enough to contribute term evidence."""
    source = source.strip()
    target = target.strip()
    if not source or not target or len(source) <= 2 or len(source) > 50:
        return False
    if source.startswith(("{", "<")) or source[0].islower():
        return False
    if any(char in source for char in "\r\n{}<>[]"):
        return False
    if source.endswith((".", "!", "?", ":", ";")) or "," in source:
        return False
    words = source.split()
    if not 1 <= len(words) <= 4:
        return False
    if any(source.startswith(prefix) for prefix in ("PHM_", "PTM_", "PHW_", "Animal_", "Mon_")):
        return False
    if source == target:
        if source.casefold() in _COMMON_SYSTEM_WORDS:
            return False
        return True
    return any(len(word) > 3 and word.casefold() in target.casefold() for word in words)


def _legacy_hard_lock_eligible(source: str) -> bool:
    """Allow automatic hard locks only for short, clean standalone terms."""
    source = source.strip()
    return len(source) > 2 and _LEGACY_HARD_TERM_RE.fullmatch(source) is not None


def _is_buggy_legacy_term_id(term_id: str, scope_id: str) -> bool:
    """Identify only IDs emitted by the removed hash-only legacy heuristic."""
    prefix = f"term_{scope_id}_"
    if not term_id.startswith(prefix):
        return False
    suffix = term_id[len(prefix) :]
    return re.fullmatch(r"[0-9a-f]{16}", suffix) is not None


def _legacy_term_rows(scope_id: str, rows: tuple[Any, ...], now: float) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """Build deterministic term/evidence/conflict rows from standalone legacy labels."""
    evidence: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        source = row.original.strip()
        target = getattr(row, "translation", "").strip()
        if not _legacy_term_candidate(source, target):
            continue
        provenance = f"{row.marker}:{row.key}"
        evidence.setdefault(source, {}).setdefault(target, set()).add(provenance)

    terms: list[tuple[Any, ...]] = []
    evidence_rows: list[tuple[Any, ...]] = []
    conflicts: list[tuple[Any, ...]] = []
    for source, targets in sorted(evidence.items()):
        conflicting = len(targets) > 1
        for target, provenances in sorted(targets.items()):
            count = len(provenances)
            if conflicting:
                state, confidence = "conflict", 0.5
            elif count >= 2 and _legacy_hard_lock_eligible(source):
                state, confidence = "auto_locked", 0.95
            else:
                state, confidence = "auto_suggested", 0.75
            digest = hashlib.sha256(f"{scope_id}\0{source}\0{target}".encode("utf-8")).hexdigest()[:20]
            term_id = f"legacy_term_{scope_id}_{digest}"
            terms.append((term_id, scope_id, source, target, state, confidence, now, now))
            for provenance in sorted(provenances):
                ev_digest = hashlib.sha256(f"{term_id}\0{provenance}".encode("utf-8")).hexdigest()[:20]
                evidence_rows.append((f"legacy_ev_{ev_digest}", term_id, "standalone_legacy", provenance, now))
            if conflicting:
                conflict_digest = hashlib.sha256(f"{scope_id}\0{source}\0{target}".encode("utf-8")).hexdigest()[:20]
                conflicts.append((f"legacy_conflict_{conflict_digest}", scope_id, source, target, "Competing legacy targets; not enforced.", now))
    return terms, evidence_rows, conflicts


@dataclass(frozen=True)
class TranslationWorkResult:
    work_json: Path
    backup_json: Path | None
    unchanged_entries: int
    new_entries: int
    changed_originals: int
    removed_entries: int
    translation_candidates: int
    technical_entries: int = 0
    knowledge_warning: str | None = None
    lineage_id: str | None = None
    imported_segments_count: int = 0

    @property
    def delta_entries(self) -> int:
        return self.translation_candidates


def _backup_path(output: Path) -> Path:
    candidate = output.with_name(f"{output.name}.backup")
    suffix = 1
    while candidate.exists():
        candidate = output.with_name(f"{output.name}.backup-{suffix}")
        suffix += 1
    return candidate


def _atomic_json_write(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _legacy_knowledge_already_ingested(
    knowledge_db: Any,
    scope_id: str,
    lineage_id: str,
) -> bool:
    """Return True after the atomic legacy bootstrap has committed once."""

    prefix = f"legacy_{scope_id}_{lineage_id}_"
    with knowledge_db.connect() as con:
        row = con.execute(
            """
            SELECT 1
            FROM translation_segments
            WHERE scope_id = ?
              AND lineage_id = ?
              AND segment_id GLOB ?
            LIMIT 1
            """,
            (scope_id, lineage_id, f"{prefix}*"),
        ).fetchone()
    return row is not None


def _package_folder(data: dict[str, object]) -> str | None:
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("package_folder"), str):
        return metadata["package_folder"]
    return None


def _catalog_context_ranges(
    data: dict[str, object], row_count: int
) -> tuple[tuple[int, int], ...]:
    """Return contiguous PALOC ranges without exposing PALOC paths as context."""
    metadata = data.get("metadata")
    palocs = metadata.get("palocs") if isinstance(metadata, dict) else None
    if isinstance(palocs, list):
        ranges: list[tuple[int, int]] = []
        cursor = 0
        valid = True
        for paloc in palocs:
            count = paloc.get("entry_count") if isinstance(paloc, dict) else None
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                valid = False
                break
            if count:
                ranges.append((cursor, cursor + count))
            cursor += count
        if valid and cursor == row_count:
            return tuple(ranges)
    return ((0, row_count),) if row_count else ()


def _catalog_contexts(
    data: dict[str, object], rows: tuple[Any, ...]
) -> dict[int, tuple[tuple[str, ...], str]]:
    """Preserve explicit context or derive bounded neighbor context per PALOC range."""
    engine = ContextEngine()
    resolved: dict[int, tuple[tuple[str, ...], str]] = {}
    for start, end in _catalog_context_ranges(data, len(rows)):
        window_rows = rows[start:end]
        items = tuple(
            ContextItem(index=index, key=row.key, text=row.original)
            for index, row in enumerate(window_rows)
        )
        for local_index, row in enumerate(window_rows):
            if row.context or row.context_signature:
                context = tuple(row.context)
                signature = row.context_signature
                if context and not signature:
                    signature = hashlib.sha256(
                        ("explicit\0" + "\0".join(context)).encode("utf-8")
                    ).hexdigest()
                resolved[row.index] = (context, signature)
                continue
            assembled = engine.build_context(
                items, target_index=local_index, kind=ContextKind.SCENE
            )
            context = tuple(
                part for part in assembled.context_text.splitlines() if part.strip()
            )
            resolved[row.index] = (
                context,
                assembled.signature if context else "",
            )
    return resolved


def _delta_source(
    new_data: dict[str, Any],
    new_rows: tuple[Any, ...],
    old_data: dict[str, Any],
    old_rows: tuple[Any, ...],
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "catalog_fingerprint": _source_fingerprint(new_data, new_rows),
        "base_catalog_fingerprint": _source_fingerprint(old_data, old_rows),
        "entry_count": len(new_rows),
    }
    package = _package_folder(new_data)
    if package is not None:
        source["package_folder"] = package
    return source


def prepare_translation_work(
    old_json: Path | str,
    new_json: Path | str,
    work_json: Path | str,
    knowledge_db: Any | None = None,
    app_id: str = "Crimson0020",
    source_language: str = "en",
    target_language: str = "vi",
) -> TranslationWorkResult:
    """Create a sparse delta containing only new or source-changed identities."""
    old_path = Path(old_json).expanduser().resolve()
    new_path = Path(new_json).expanduser().resolve()
    output = Path(work_json).expanduser().resolve()
    if not old_path.is_file():
        raise FileNotFoundError(f"Old localization JSON does not exist: {old_path}")
    if not new_path.is_file():
        raise FileNotFoundError(f"New localization JSON does not exist: {new_path}")
    if output in {old_path, new_path}:
        raise ValueError("Translation delta JSON must be separate from both inputs.")
    old_data = read_paloc_json(old_path)
    new_data = read_paloc_json(new_path)
    old_package = _package_folder(old_data)
    new_package = _package_folder(new_data)
    if (
        old_package is not None
        and new_package is not None
        and old_package != new_package
    ):
        raise ValueError(
            "Old and new localization JSON use different package_folder values."
        )
    old_rows = _catalog_rows(old_data)
    new_rows = _catalog_rows(new_data)
    new_contexts = _catalog_contexts(new_data, new_rows)
    old_by_identity = {row.identity: row for row in old_rows}
    new_identities = {row.identity for row in new_rows}
    entries: list[dict[str, Any]] = []
    unchanged = new_count = changed = technical = candidates = 0
    for row in new_rows:
        previous = old_by_identity.get(row.identity)
        if previous is None:
            change = "new"
            new_count += 1
        elif previous.original != row.original:
            change = "original_changed"
            changed += 1
        else:
            unchanged += 1
            continue
        if not has_translatable_content(row.original):
            technical += 1
            continue
        candidates += 1
        entry: dict[str, Any] = {
            "marker": row.marker,
            "key": row.key,
            "original": row.original,
            "change": change,
            "translation": "",
            "translation_state": "pending",
        }
        context, context_signature = new_contexts.get(row.index, ((), ""))
        if context:
            entry["context"] = list(context)
        if context_signature:
            entry["context_signature"] = context_signature
        entries.append(entry)
    delta = {
        "schema": DELTA_SCHEMA,
        "schema_version": DELTA_SCHEMA_VERSION,
        "kind": "sparse-delta",
        "source": _delta_source(new_data, new_rows, old_data, old_rows),
        "entries": entries,
    }
    backup: Path | None = None
    if output.is_file():
        backup = _backup_path(output)
        os.replace(output, backup)
    try:
        _atomic_json_write(delta, output)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
            backup = None
        raise

    knowledge_warning: str | None = None
    lineage_id: str | None = None
    imported_segments_count = 0

    # Step 1 knowledge ingestion: register old and new catalogs and import legacy knowledge
    if knowledge_db is not None:
        try:
            from crimson0020.translation_runtime.knowledge.service import (
                CatalogRegistration,
                KnowledgeService,
                ScopeIdentifier,
            )

            service = KnowledgeService(knowledge_db)
            package_family = new_package or old_package or "0020"
            scope = ScopeIdentifier(
                app_id=app_id,
                package_family=package_family,
                source_language=source_language,
                target_language=target_language,
            )
            old_fp = _source_fingerprint(old_data, old_rows)
            service.register_catalog(
                CatalogRegistration(
                    scope=scope,
                    fingerprint=old_fp,
                    item_count=len(old_rows),
                )
            )
            new_fp = _source_fingerprint(new_data, new_rows)
            lineage_id = service.register_catalog(
                CatalogRegistration(
                    scope=scope,
                    fingerprint=new_fp,
                    item_count=len(new_rows),
                    prior_fingerprint=old_fp,
                )
            )

            knowledge_old_rows = (
                ()
                if _legacy_knowledge_already_ingested(
                    knowledge_db,
                    scope.scope_id,
                    lineage_id,
                )
                else old_rows
            )
            now_ts = time.time()
            legacy_segments = []
            for r in knowledge_old_rows:
                tr = getattr(r, "translation", "")
                orig = r.original
                if (
                    tr
                    and tr != orig
                    and has_translatable_content(orig)
                    and not tr.startswith("[TECH_")
                ):
                    src_hash = hashlib.sha256(orig.encode("utf-8")).hexdigest()
                    seg_id = f"legacy_{scope.scope_id}_{lineage_id}_{r.marker}_{r.key}_{src_hash}"
                    legacy_segments.append(
                        (
                            seg_id,
                            scope.scope_id,
                            lineage_id,
                            r.marker,
                            r.key,
                            src_hash,
                            orig,
                            tr,
                            "",
                            "",
                            1,
                            "active",
                            now_ts,
                            now_ts,
                        )
                    )

            if legacy_segments:
                with knowledge_db.transaction() as con:
                    cur = con.cursor()
                    cur.executemany(
                        """
                        INSERT INTO translation_segments (
                            segment_id, scope_id, lineage_id, marker, key, source_hash,
                            source_text, target_text, context_signature, hard_terms_hash,
                            glossary_revision, status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(segment_id) DO UPDATE SET
                            target_text = excluded.target_text,
                            updated_at = excluded.updated_at
                        WHERE translation_segments.target_text IS NOT excluded.target_text
                        """,
                        legacy_segments,
                    )

                    # Remove only rows created by the buggy post-final heuristic. Explicit
                    # user overrides use a different term-id shape and are preserved.
                    legacy_bug_ids = [
                        row[0]
                        for row in cur.execute(
                            "SELECT term_id FROM terms WHERE scope_id = ? AND state = 'user_locked'",
                            (scope.scope_id,),
                        ).fetchall()
                        if _is_buggy_legacy_term_id(row[0], scope.scope_id)
                    ]
                    if legacy_bug_ids:
                        cur.executemany("DELETE FROM terms WHERE term_id = ?", ((term_id,) for term_id in legacy_bug_ids))

                    # Rebuild generated legacy evidence deterministically. User-locked
                    # sources always win and are not shadowed by automatic candidates.
                    cur.execute("DELETE FROM term_conflicts WHERE conflict_id LIKE 'legacy_conflict_%'")
                    cur.execute("DELETE FROM terms WHERE scope_id = ? AND term_id LIKE 'legacy_term_%'", (scope.scope_id,))
                    explicit_sources = {
                        row[0]
                        for row in cur.execute(
                            "SELECT source_term FROM terms WHERE scope_id = ? AND state = 'user_locked'",
                            (scope.scope_id,),
                        ).fetchall()
                    }
                    terms_rows, evidence_rows, conflict_rows = _legacy_term_rows(
                        scope.scope_id,
                        knowledge_old_rows,
                        now_ts,
                    )
                    terms_rows = [row for row in terms_rows if row[2] not in explicit_sources]
                    allowed_ids = {row[0] for row in terms_rows}
                    evidence_rows = [row for row in evidence_rows if row[1] in allowed_ids]
                    conflict_rows = [row for row in conflict_rows if row[2] not in explicit_sources]
                    if terms_rows:
                        cur.executemany(
                            """
                            INSERT INTO terms (
                                term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(scope_id, source_term, target_term) DO UPDATE SET
                                state = CASE WHEN terms.state = 'user_locked' THEN terms.state ELSE excluded.state END,
                                confidence = CASE WHEN terms.state = 'user_locked' THEN terms.confidence ELSE excluded.confidence END,
                                updated_at = excluded.updated_at
                            """,
                            terms_rows,
                        )
                    if evidence_rows:
                        cur.executemany(
                            "INSERT OR IGNORE INTO term_evidence (evidence_id, term_id, evidence_kind, provenance, created_at) VALUES (?, ?, ?, ?, ?)",
                            evidence_rows,
                        )
                    if conflict_rows:
                        cur.executemany(
                            "INSERT OR REPLACE INTO term_conflicts (conflict_id, scope_id, source_term, competing_target, details, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                            conflict_rows,
                        )

                imported_segments_count = len(legacy_segments)
        except Exception as exc:
            knowledge_warning = f"Knowledge ingestion warning: {exc}"

    return TranslationWorkResult(
        output,
        backup,
        unchanged,
        new_count,
        changed,
        len(set(old_by_identity) - new_identities),
        candidates,
        technical,
        knowledge_warning,
        lineage_id,
        imported_segments_count,
    )


def read_translation_delta(path: Path | str) -> dict[str, Any]:
    """Read and validate the sparse delta envelope and immutable row fields."""
    source = Path(path).expanduser().resolve()
    try:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read translation delta: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Translation delta root must be an object.")
    if (
        data.get("schema") != DELTA_SCHEMA
        or data.get("schema_version") != DELTA_SCHEMA_VERSION
        or data.get("kind") != "sparse-delta"
    ):
        raise ValueError("Selected file is not a supported translation delta.")
    source_info = data.get("source")
    entries = data.get("entries")
    if (
        not isinstance(source_info, dict)
        or not isinstance(entries, list)
        or not isinstance(source_info.get("catalog_fingerprint"), str)
    ):
        raise ValueError(
            "Translation delta must contain source fingerprint and entries."
        )
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Translation delta entries[{index}] must be an object.")
        marker, key, original = (
            entry.get("marker"),
            entry.get("key"),
            entry.get("original"),
        )
        if (
            not isinstance(marker, str)
            or not isinstance(key, str)
            or not isinstance(original, str)
        ):
            raise ValueError(
                f"Translation delta entries[{index}] identity/source is invalid."
            )
        try:
            marker_bytes = bytes.fromhex(marker)
        except ValueError as error:
            raise ValueError(
                f"Translation delta entries[{index}].marker is invalid."
            ) from error
        identity = (marker_bytes.hex(), key)
        if len(marker_bytes) != 8:
            raise ValueError(
                f"Translation delta entries[{index}].marker must contain 8 bytes."
            )
        if identity in seen:
            raise ValueError(
                f"Duplicate marker + key identity in translation delta entries[{index}]."
            )
        seen.add(identity)
        if "change" in entry:
            change_val = entry["change"]
            if not isinstance(change_val, str) or change_val not in {
                "new",
                "original_changed",
            }:
                raise ValueError(
                    f"Translation delta entries[{index}].change is invalid."
                )
        translation_val = entry.get("translation", "")
        translation_state = entry.get("translation_state", "pending")
        if (
            not isinstance(translation_val, str)
            or not isinstance(translation_state, str)
            or translation_state not in {"pending", "translated", "failed", "skipped_technical"}
        ):
            raise ValueError(
                f"Translation delta entries[{index}] translation state is invalid."
            )
        raw_context = entry.get("context", [])
        if isinstance(raw_context, str):
            raw_context = [raw_context]
        if not isinstance(raw_context, list) or any(
            not isinstance(value, str) for value in raw_context
        ):
            raise ValueError(f"Translation delta entries[{index}].context is invalid.")
        context_signature = entry.get("context_signature", "")
        if not isinstance(context_signature, str):
            raise ValueError(
                f"Translation delta entries[{index}].context_signature is invalid."
            )
        for text_val, field_name in ((original, "original"), (translation_val, "translation")):
            try:
                text_val.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    f"Translation delta entries[{index}].{field_name} contains invalid Unicode surrogates."
                ) from error
    return data
