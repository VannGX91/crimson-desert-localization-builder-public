"""Translation memory repository for exact and fuzzy retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from crimson0020.translation_runtime.checkpoint import CheckpointRecord
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase


@dataclass(frozen=True, slots=True)
class TranslationSegment:
    """Represents a stored translation unit segment in Translation Memory."""

    segment_id: str
    scope_id: str
    lineage_id: str
    marker: str
    key: str
    source_hash: str
    source_text: str
    target_text: str
    context_signature: str = ""
    hard_terms_hash: str = ""
    glossary_revision: int = 1
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class ExactLookupRequest:
    """Parameters for looking up an exact translation memory candidate."""

    scope_id: str
    lineage_id: str
    source_hash: str
    marker: str
    key: str
    context_signature: str = ""
    hard_terms_hash: str = ""
    glossary_revision: int = 1


class TranslationMemoryRepository:
    """Repository managing Translation Memory segments with exact & fuzzy retrieval."""

    def __init__(self, db: KnowledgeDatabase) -> None:
        self.db = db

    def save_segment(self, segment: TranslationSegment) -> None:
        """Saves or replaces an active translation segment."""
        now = time.time()
        created = segment.created_at or now
        updated = segment.updated_at or now
        with self.db.transaction() as con:
            con.execute(
                """
                INSERT INTO translation_segments (
                    segment_id, scope_id, lineage_id, marker, key, source_hash,
                    source_text, target_text, context_signature, hard_terms_hash,
                    glossary_revision, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    target_text = excluded.target_text,
                    context_signature = excluded.context_signature,
                    hard_terms_hash = excluded.hard_terms_hash,
                    glossary_revision = excluded.glossary_revision,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                WHERE translation_segments.target_text IS NOT excluded.target_text
                   OR translation_segments.context_signature IS NOT excluded.context_signature
                   OR translation_segments.hard_terms_hash IS NOT excluded.hard_terms_hash
                   OR translation_segments.glossary_revision IS NOT excluded.glossary_revision
                   OR translation_segments.status IS NOT excluded.status
                """,
                (
                    segment.segment_id,
                    segment.scope_id,
                    segment.lineage_id,
                    segment.marker,
                    segment.key,
                    segment.source_hash,
                    segment.source_text,
                    segment.target_text,
                    segment.context_signature,
                    segment.hard_terms_hash,
                    segment.glossary_revision,
                    segment.status,
                    created,
                    updated,
                ),
            )

    def save_segments(self, segments: Sequence[TranslationSegment]) -> None:
        """Saves or replaces a sequence of active translation segments in a single transaction."""
        if not segments:
            return
        now = time.time()
        rows = [
            (
                segment.segment_id,
                segment.scope_id,
                segment.lineage_id,
                segment.marker,
                segment.key,
                segment.source_hash,
                segment.source_text,
                segment.target_text,
                segment.context_signature,
                segment.hard_terms_hash,
                segment.glossary_revision,
                segment.status,
                segment.created_at or now,
                segment.updated_at or now,
            )
            for segment in segments
        ]
        with self.db.transaction() as con:
            con.executemany(
                """
                INSERT INTO translation_segments (
                    segment_id, scope_id, lineage_id, marker, key, source_hash,
                    source_text, target_text, context_signature, hard_terms_hash,
                    glossary_revision, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    target_text = excluded.target_text,
                    context_signature = excluded.context_signature,
                    hard_terms_hash = excluded.hard_terms_hash,
                    glossary_revision = excluded.glossary_revision,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                WHERE translation_segments.target_text IS NOT excluded.target_text
                   OR translation_segments.context_signature IS NOT excluded.context_signature
                   OR translation_segments.hard_terms_hash IS NOT excluded.hard_terms_hash
                   OR translation_segments.glossary_revision IS NOT excluded.glossary_revision
                   OR translation_segments.status IS NOT excluded.status
                """,
                rows,
            )

    def lookup_exact(self, req: ExactLookupRequest) -> TranslationSegment | None:
        """Looks up an exact reusable translation segment obeying isolation rules."""
        with self.db.connect() as con:
            cur = con.cursor()
            # Rule 1: exact marker + key + source_hash + lineage + scope
            cur.execute(
                """
                SELECT segment_id, scope_id, lineage_id, marker, key, source_hash,
                       source_text, target_text, context_signature, hard_terms_hash,
                       glossary_revision, status, created_at, updated_at
                FROM translation_segments
                WHERE scope_id = ?
                  AND lineage_id = ?
                  AND marker = ?
                  AND key = ?
                  AND source_hash = ?
                  AND status = 'active'
                LIMIT 1
                """,
                (req.scope_id, req.lineage_id, req.marker, req.key, req.source_hash),
            )
            row = cur.fetchone()
            if row:
                return TranslationSegment(*row)

            # Rule 2: same source_hash + context_signature + hard_terms_hash + glossary_revision + lineage + scope
            if req.context_signature:
                cur.execute(
                    """
                    SELECT segment_id, scope_id, lineage_id, marker, key, source_hash,
                           source_text, target_text, context_signature, hard_terms_hash,
                       glossary_revision, status, created_at, updated_at
                    FROM translation_segments
                    WHERE scope_id = ?
                      AND lineage_id = ?
                      AND source_hash = ?
                      AND context_signature = ?
                      AND hard_terms_hash = ?
                      AND glossary_revision = ?
                      AND status = 'active'
                    LIMIT 1
                    """,
                    (
                        req.scope_id,
                        req.lineage_id,
                        req.source_hash,
                        req.context_signature,
                        req.hard_terms_hash,
                        req.glossary_revision,
                    ),
                )
                row = cur.fetchone()
                if row:
                    return TranslationSegment(*row)

        return None

    def lookup_exact_many(
        self, requests: Sequence[ExactLookupRequest]
    ) -> tuple[TranslationSegment | None, ...]:
        """Resolve many exact TM requests with one SQLite connection.

        The temporary request table is connection-local, so batching lookups does
        not add persistent rows or grow the user's knowledge database.
        """

        batch = tuple(requests)
        if not batch:
            return ()

        results: list[TranslationSegment | None] = [None] * len(batch)
        with self.db.connect() as con:
            con.execute(
                """
                CREATE TEMP TABLE tm_lookup_requests (
                    req_index INTEGER PRIMARY KEY,
                    scope_id TEXT NOT NULL,
                    lineage_id TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    marker TEXT NOT NULL,
                    key TEXT NOT NULL,
                    context_signature TEXT NOT NULL,
                    hard_terms_hash TEXT NOT NULL,
                    glossary_revision INTEGER NOT NULL
                )
                """
            )
            con.executemany(
                """
                INSERT INTO tm_lookup_requests (
                    req_index, scope_id, lineage_id, source_hash, marker, key,
                    context_signature, hard_terms_hash, glossary_revision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        index,
                        req.scope_id,
                        req.lineage_id,
                        req.source_hash,
                        req.marker,
                        req.key,
                        req.context_signature,
                        req.hard_terms_hash,
                        req.glossary_revision,
                    )
                    for index, req in enumerate(batch)
                ),
            )

            segment_columns = """
                s.segment_id, s.scope_id, s.lineage_id, s.marker, s.key,
                s.source_hash, s.source_text, s.target_text,
                s.context_signature, s.hard_terms_hash, s.glossary_revision,
                s.status, s.created_at, s.updated_at
            """
            exact_rows = con.execute(
                f"""
                SELECT r.req_index, {segment_columns}
                FROM tm_lookup_requests AS r
                JOIN translation_segments AS s
                  ON s.scope_id = r.scope_id
                 AND s.lineage_id = r.lineage_id
                 AND s.marker = r.marker
                 AND s.key = r.key
                 AND s.source_hash = r.source_hash
                 AND s.status = 'active'
                ORDER BY r.req_index, s.segment_id
                """
            ).fetchall()
            for row in exact_rows:
                request_index = int(row[0])
                if results[request_index] is None:
                    results[request_index] = TranslationSegment(*row[1:])

            contextual_rows = con.execute(
                f"""
                SELECT r.req_index, {segment_columns}
                FROM tm_lookup_requests AS r
                JOIN translation_segments AS s
                  ON s.scope_id = r.scope_id
                 AND s.lineage_id = r.lineage_id
                 AND s.source_hash = r.source_hash
                 AND s.context_signature = r.context_signature
                 AND s.hard_terms_hash = r.hard_terms_hash
                 AND s.glossary_revision = r.glossary_revision
                 AND s.status = 'active'
                WHERE r.context_signature != ''
                ORDER BY r.req_index, s.segment_id
                """
            ).fetchall()
            for row in contextual_rows:
                request_index = int(row[0])
                if results[request_index] is None:
                    results[request_index] = TranslationSegment(*row[1:])

        return tuple(results)

    def search_fuzzy(
        self, scope_id: str, query: str, limit: int = 5
    ) -> Sequence[TranslationSegment]:
        """Performs FTS5 search across translation segments for prompt context examples only."""
        clean_query = "".join(c if c.isalnum() or c.isspace() else " " for c in query).strip()
        if not clean_query:
            return ()

        with self.db.connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT s.segment_id, s.scope_id, s.lineage_id, s.marker, s.key, s.source_hash,
                       s.source_text, s.target_text, s.context_signature, s.hard_terms_hash,
                       s.glossary_revision, s.status, s.created_at, s.updated_at
                FROM segments_fts f
                JOIN translation_segments s ON f.segment_id = s.segment_id
                WHERE segments_fts MATCH ?
                  AND s.scope_id = ?
                  AND s.status = 'active'
                LIMIT ?
                """,
                (clean_query, scope_id, limit),
            )
            rows = cur.fetchall()
            return tuple(TranslationSegment(*r) for r in rows)

    def import_checkpoint_records(
        self,
        scope_id: str,
        lineage_id: str,
        records: Sequence[CheckpointRecord],
    ) -> int:
        """Imports checkpoint records idempotently, skipping technical or source-equals-target items."""
        imported_count = 0
        now = time.time()
        with self.db.transaction() as con:
            for rec in records:
                # Exclude technical/source=target
                if "source_equals_target" in rec.quality_flags or "technical_identifier" in rec.quality_flags:
                    continue
                if rec.translation == rec.key or rec.translation.startswith("[TECH_"):
                    continue

                segment_id = f"ckpt_{scope_id}_{lineage_id}_{rec.marker}_{rec.key}_{rec.original_sha256}"
                cur = con.cursor()
                cur.execute("SELECT 1 FROM translation_segments WHERE segment_id = ?", (segment_id,))
                if cur.fetchone():
                    continue

                cur.execute(
                    """
                    INSERT INTO translation_segments (
                        segment_id, scope_id, lineage_id, marker, key, source_hash,
                        source_text, target_text, context_signature, hard_terms_hash,
                        glossary_revision, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        segment_id,
                        scope_id,
                        lineage_id,
                        rec.marker,
                        rec.key,
                        rec.original_sha256,
                        rec.key,  # Checkpoint stores key/source hash
                        rec.translation,
                        rec.context_signature or "",
                        "",
                        rec.glossary_revision or 1,
                        now,
                        now,
                    ),
                )
                imported_count += 1

        return imported_count

    def invalidate_terms_hash(self, scope_id: str, hard_terms_hash: str) -> int:
        """Invalidates all cached active translation segments that depend on an outdated hard terms hash."""
        now = time.time()
        with self.db.transaction() as con:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE translation_segments
                SET status = 'invalidated', updated_at = ?
                WHERE scope_id = ?
                  AND hard_terms_hash = ?
                  AND status = 'active'
                """,
                (now, scope_id, hard_terms_hash),
            )
            return cur.rowcount
