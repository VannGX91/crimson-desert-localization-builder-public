"""Knowledge service orchestrating scopes, catalog lineages, paginated queries, and overrides."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from crimson0020.translation_runtime.errors import TranslationRuntimeError
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.import_export import (
    export_knowledge_package,
    import_knowledge_package,
)
from crimson0020.translation_runtime.knowledge.memory import (
    TranslationMemoryRepository,
    TranslationSegment,
)
from crimson0020.translation_runtime.knowledge.terminology import (
    GlossaryDecision,
    TerminologyEngine,
)


class LineageChainError(TranslationRuntimeError):
    """Raised when catalog lineage chaining is invalid or conflicting."""


@dataclass(frozen=True, slots=True)
class ScopeIdentifier:
    """Uniquely identifies an isolated translation knowledge scope."""

    app_id: str
    package_family: str
    source_language: str
    target_language: str

    @property
    def scope_id(self) -> str:
        return f"{self.app_id}_{self.package_family}_{self.source_language}_{self.target_language}"

    def to_dict(self) -> dict[str, str]:
        return {
            "app_id": self.app_id,
            "package_family": self.package_family,
            "source_language": self.source_language,
            "target_language": self.target_language,
        }


@dataclass(frozen=True, slots=True)
class CatalogRegistration:
    """Catalog registration parameters."""

    scope: ScopeIdentifier
    fingerprint: str
    item_count: int
    prior_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class PaginatedSegments:
    """Paginated segment query result."""

    items: tuple[TranslationSegment, ...]
    total_count: int
    offset: int
    limit: int


class KnowledgeService:
    """High-level service orchestrating knowledge storage, catalog lineages, and queries."""

    def __init__(self, db: KnowledgeDatabase) -> None:
        self.db = db
        self.memory = TranslationMemoryRepository(db)
        self.terminology = TerminologyEngine(db)

    def _ensure_scope(self, scope: ScopeIdentifier) -> None:
        now = time.time()
        with self.db.transaction() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id) DO NOTHING
                """,
                (scope.scope_id, scope.app_id, scope.package_family, scope.source_language, scope.target_language, now),
            )

    def register_catalog(self, reg: CatalogRegistration) -> str:
        """Registers a catalog fingerprint, continuing known lineage or spawning new."""
        self._ensure_scope(reg.scope)
        now = time.time()

        with self.db.transaction() as con:
            cur = con.cursor()
            # Check if this fingerprint already exists
            cur.execute(
                "SELECT lineage_id FROM catalog_fingerprints WHERE fingerprint = ?",
                (reg.fingerprint,),
            )
            existing = cur.fetchone()
            if existing:
                return existing[0]

            lineage_id = None
            if reg.prior_fingerprint:
                # Find lineage of prior fingerprint
                cur.execute(
                    "SELECT lineage_id FROM catalog_fingerprints WHERE fingerprint = ?",
                    (reg.prior_fingerprint,),
                )
                prior_row = cur.fetchone()
                if prior_row:
                    lineage_id = prior_row[0]

            if not lineage_id:
                # Spawn new lineage
                lineage_id = f"lin_{reg.scope.scope_id}_{reg.fingerprint}_{int(now)}"
                cur.execute(
                    "INSERT INTO lineages (lineage_id, scope_id, parent_lineage_id, created_at) VALUES (?, ?, ?, ?)",
                    (lineage_id, reg.scope.scope_id, None, now),
                )

            # Record catalog fingerprint
            cur.execute(
                "INSERT INTO catalog_fingerprints (fingerprint, lineage_id, item_count, created_at) VALUES (?, ?, ?, ?)",
                (reg.fingerprint, lineage_id, reg.item_count, now),
            )

            return lineage_id

    def get_lineage_for_fingerprint(self, scope: ScopeIdentifier, fingerprint: str) -> str | None:
        """Looks up existing lineage ID for catalog fingerprint."""
        with self.db.connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT lineage_id FROM catalog_fingerprints WHERE fingerprint = ?",
                (fingerprint,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def save_translation_segment(
        self,
        scope: ScopeIdentifier,
        lineage_id: str,
        marker: str,
        key: str,
        source_hash: str,
        source_text: str,
        target_text: str,
        context_signature: str = "",
        hard_terms_hash: str = "",
        glossary_revision: int = 1,
    ) -> None:
        self._ensure_scope(scope)
        seg_id = f"seg_{scope.scope_id}_{lineage_id}_{marker}_{key}_{source_hash}"
        seg = TranslationSegment(
            segment_id=seg_id,
            scope_id=scope.scope_id,
            lineage_id=lineage_id,
            marker=marker,
            key=key,
            source_hash=source_hash,
            source_text=source_text,
            target_text=target_text,
            context_signature=context_signature,
            hard_terms_hash=hard_terms_hash,
            glossary_revision=glossary_revision,
            status="active",
        )
        self.memory.save_segment(seg)

    def set_glossary_override(
        self, scope: ScopeIdentifier, source_term: str, target_term: str
    ) -> GlossaryDecision:
        self._ensure_scope(scope)
        return self.terminology.set_user_override(scope.scope_id, source_term, target_term)

    def list_segments(
        self, scope: ScopeIdentifier, limit: int = 50, offset: int = 0
    ) -> PaginatedSegments:
        """Returns paginated segments within a scope."""
        with self.db.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM translation_segments WHERE scope_id = ?", (scope.scope_id,))
            total = cur.fetchone()[0]

            cur.execute(
                """
                SELECT segment_id, scope_id, lineage_id, marker, key, source_hash,
                       source_text, target_text, context_signature, hard_terms_hash,
                       glossary_revision, status, created_at, updated_at
                FROM translation_segments
                WHERE scope_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
                """,
                (scope.scope_id, limit, offset),
            )
            rows = cur.fetchall()
            items = tuple(TranslationSegment(*r) for r in rows)
            return PaginatedSegments(items=items, total_count=total, offset=offset, limit=limit)

    def export_package(self, scope: ScopeIdentifier) -> dict[str, Any]:
        """Exports all segments and terms within a scope into a portable dict."""
        with self.db.connect() as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT segment_id, lineage_id, marker, key, source_hash, source_text,
                       target_text, context_signature, hard_terms_hash, glossary_revision, status
                FROM translation_segments WHERE scope_id = ?
                """,
                (scope.scope_id,),
            )
            seg_rows = cur.fetchall()
            segments = [
                {
                    "segment_id": r[0],
                    "lineage_id": r[1],
                    "marker": r[2],
                    "key": r[3],
                    "source_hash": r[4],
                    "source_text": r[5],
                    "target_text": r[6],
                    "context_signature": r[7],
                    "hard_terms_hash": r[8],
                    "glossary_revision": r[9],
                    "status": r[10],
                }
                for r in seg_rows
            ]

            cur.execute(
                "SELECT source_term, target_term, state, confidence FROM terms WHERE scope_id = ?",
                (scope.scope_id,),
            )
            term_rows = cur.fetchall()
            terms = [
                {
                    "source_term": r[0],
                    "target_term": r[1],
                    "state": r[2],
                    "confidence": r[3],
                }
                for r in term_rows
            ]

        return export_knowledge_package(
            scope=scope.to_dict(),
            segments=segments,
            terms=terms,
        )

    def import_package(self, package_data: dict[str, Any]) -> None:
        """Imports and merges knowledge package atomically."""
        pkg = import_knowledge_package(package_data)
        scope = ScopeIdentifier(**pkg.scope)
        self._ensure_scope(scope)
        now = time.time()

        with self.db.transaction() as con:
            cur = con.cursor()
            for s in pkg.segments:
                # Ensure lineage exists
                cur.execute(
                    """
                    INSERT INTO lineages (lineage_id, scope_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(lineage_id) DO NOTHING
                    """,
                    (s["lineage_id"], scope.scope_id, now),
                )
                cur.execute(
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
                    (
                        s["segment_id"],
                        scope.scope_id,
                        s["lineage_id"],
                        s["marker"],
                        s["key"],
                        s["source_hash"],
                        s["source_text"],
                        s["target_text"],
                        s["context_signature"],
                        s["hard_terms_hash"],
                        s["glossary_revision"],
                        s["status"],
                        now,
                        now,
                    ),
                )

            for t in pkg.terms:
                term_id = f"term_{scope.scope_id}_{t['source_term']}_{t['target_term']}"
                cur.execute(
                    """
                    INSERT INTO terms (term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(scope_id, source_term, target_term) DO UPDATE SET
                        state = excluded.state,
                        confidence = excluded.confidence,
                        updated_at = excluded.updated_at
                    WHERE terms.state IS NOT excluded.state
                       OR terms.confidence IS NOT excluded.confidence
                    """,
                    (term_id, scope.scope_id, t["source_term"], t["target_term"], t["state"], t["confidence"], now, now),
                )
