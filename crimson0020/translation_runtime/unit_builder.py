"""Translation-unit builder creating privacy-safe translation items and resolving exact TM (INT-02)."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Sequence

from crimson0020.translation_runtime.contracts import TranslationRequestItem
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.memory import (
    ExactLookupRequest,
    TranslationMemoryRepository,
)
from crimson0020.translation_runtime.knowledge.service import ScopeIdentifier
from crimson0020.translation_runtime.knowledge.terminology import (
    TerminologyEngine,
)


@dataclass(frozen=True, slots=True)
class RawTranslationInput:
    """Internal input representing a raw string to be translated."""

    marker: str
    key: str
    original: str
    source_hash: str | None = None
    context: tuple[str, ...] = ()
    context_signature: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedTMItem:
    """Translation item resolved immediately from Translation Memory without provider invocation."""

    marker: str
    key: str
    source_text: str
    translated_text: str
    segment_id: str


@dataclass(frozen=True, slots=True)
class UnitBuildOutcome:
    """Result of translation unit building process."""

    outbound_items: tuple[TranslationRequestItem, ...]
    exact_matches: dict[tuple[str, str], ResolvedTMItem]
    work_id_map: dict[str, tuple[str, str]]


class TranslationUnitBuilder:
    """Builds privacy-safe TranslationRequestItems and resolves exact TM matches."""

    def __init__(
        self,
        db: KnowledgeDatabase,
        scope: ScopeIdentifier,
        lineage_id: str,
        glossary_revision: int = 1,
    ) -> None:
        self.db = db
        self.scope = scope
        self.lineage_id = lineage_id
        self.glossary_revision = glossary_revision
        self.memory = TranslationMemoryRepository(db)
        self.terminology = TerminologyEngine(db)
        self._ensure_scope()

    def _ensure_scope(self) -> None:
        now = time.time()
        with self.db.transaction() as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id) DO NOTHING
                """,
                (
                    self.scope.scope_id,
                    self.scope.app_id,
                    self.scope.package_family,
                    self.scope.source_language,
                    self.scope.target_language,
                    now,
                ),
            )

    def build_units(self, inputs: Sequence[RawTranslationInput]) -> UnitBuildOutcome:
        """Processes raw translation inputs into outbound items and exact TM matches."""
        outbound: list[TranslationRequestItem] = []
        exact_matches: dict[tuple[str, str], ResolvedTMItem] = {}
        work_id_map: dict[str, tuple[str, str]] = {}

        scope_id = self.scope.scope_id
        prepared: list[
            tuple[
                RawTranslationInput,
                str,
                str,
                tuple,
                ExactLookupRequest,
            ]
        ] = []
        lookup_requests: list[ExactLookupRequest] = []

        for item in inputs:
            source_text = item.original
            source_hash = item.source_hash or hashlib.sha256(source_text.encode("utf-8")).hexdigest()

            # Hard terms extraction and masking
            masked_text, _hard_meta, hard_constraints = self.terminology.prepare_hard_terms(
                scope_id, source_text
            )
            hard_terms_repr = "|".join(f"{c.source_term}:{c.target_term}" for c in hard_constraints)
            hard_terms_hash = hashlib.sha256(hard_terms_repr.encode("utf-8")).hexdigest() if hard_terms_repr else ""

            # Check exact TM hit
            lookup_req = ExactLookupRequest(
                scope_id=scope_id,
                lineage_id=self.lineage_id,
                source_hash=source_hash,
                marker=item.marker,
                key=item.key,
                context_signature=item.context_signature,
                hard_terms_hash=hard_terms_hash,
                glossary_revision=self.glossary_revision,
            )
            prepared.append(
                (item, source_text, masked_text, tuple(hard_constraints), lookup_req)
            )
            lookup_requests.append(lookup_req)

        exact_segments = self.memory.lookup_exact_many(lookup_requests)
        for prepared_item, exact_seg in zip(prepared, exact_segments, strict=True):
            item, source_text, masked_text, hard_constraints, _lookup_req = prepared_item
            if exact_seg is not None and exact_seg.status == "active":
                exact_matches[(item.marker, item.key)] = ResolvedTMItem(
                    marker=item.marker,
                    key=item.key,
                    source_text=exact_seg.source_text,
                    translated_text=exact_seg.target_text,
                    segment_id=exact_seg.segment_id,
                )
                continue

            matched_soft = self.terminology.match_soft_suggestions(scope_id, source_text)

            # Generate privacy-safe opaque work ID
            work_id = f"w_{uuid.uuid4().hex[:12]}"
            work_id_map[work_id] = (item.marker, item.key)

            req_item = TranslationRequestItem(
                work_id=work_id,
                masked_text=masked_text,
                context=item.context,
                context_signature=item.context_signature,
                hard_terms=tuple(hard_constraints),
                soft_terms=tuple(matched_soft),
            )
            outbound.append(req_item)

        return UnitBuildOutcome(
            outbound_items=tuple(outbound),
            exact_matches=exact_matches,
            work_id_map=work_id_map,
        )
