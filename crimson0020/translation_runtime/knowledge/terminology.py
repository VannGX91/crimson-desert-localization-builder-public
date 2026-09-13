"""Terminology engine for poison-resistant glossary decisions and hard-term masking."""

from __future__ import annotations

import enum
import re
from collections import Counter
import time
from dataclasses import dataclass
from typing import Sequence

from crimson0020.translation_runtime.contracts import TermConstraint, TermSuggestion
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase


class TermState(str, enum.Enum):
    """Glossary term states defined by PLAN.md."""

    USER_LOCKED = "user_locked"
    AUTO_LOCKED = "auto_locked"
    AUTO_SUGGESTED = "auto_suggested"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Evidence supporting a term translation."""

    source_term: str
    target_term: str
    kind: str  # standalone_legacy, route_consensus, user_override
    provenance: str


@dataclass(frozen=True, slots=True)
class GlossaryDecision:
    """Calculated glossary decision for a term."""

    source_term: str
    target_term: str
    state: TermState
    confidence: float


@dataclass(frozen=True, slots=True)
class HardTermMaskMetadata:
    """Metadata tracking masked hard terms for restoration."""

    original_text: str
    token_to_target: dict[str, str]


@dataclass(frozen=True, slots=True)
class _SubstringTrie:
    transitions: tuple[dict[str, int], ...]
    outputs: tuple[tuple[int, ...], ...]
    terms: tuple[str, ...]

    @classmethod
    def build(cls, terms: Sequence[str]) -> "_SubstringTrie":
        transitions: list[dict[str, int]] = [{}]
        outputs: list[list[int]] = [[]]
        normalized_terms = tuple(term.lower() for term in terms)
        for term_index, term in enumerate(normalized_terms):
            if not term:
                continue
            node = 0
            for character in term:
                next_node = transitions[node].get(character)
                if next_node is None:
                    next_node = len(transitions)
                    transitions[node][character] = next_node
                    transitions.append({})
                    outputs.append([])
                node = next_node
            outputs[node].append(term_index)
        return cls(
            transitions=tuple(dict(edges) for edges in transitions),
            outputs=tuple(tuple(items) for items in outputs),
            terms=normalized_terms,
        )

    def find(self, text: str) -> tuple[tuple[int, int, int], ...]:
        normalized = text.lower()
        if len(normalized) != len(text):
            fallback: list[tuple[int, int, int]] = []
            for term_index, term in enumerate(self.terms):
                start = normalized.find(term)
                while start >= 0:
                    fallback.append((term_index, start, start + len(term)))
                    start = normalized.find(term, start + 1)
            return tuple(fallback)

        matches: list[tuple[int, int, int]] = []
        for start in range(len(normalized)):
            node = 0
            position = start
            while position < len(normalized):
                next_node = self.transitions[node].get(normalized[position])
                if next_node is None:
                    break
                node = next_node
                position += 1
                matches.extend(
                    (term_index, start, position)
                    for term_index in self.outputs[node]
                )
        return tuple(matches)


@dataclass(frozen=True, slots=True)
class _HardTermSnapshot:
    rows: tuple[tuple[str, str, bool, bool], ...]
    matcher: _SubstringTrie


@dataclass(frozen=True, slots=True)
class _SoftTermSnapshot:
    suggestions: tuple[TermSuggestion, ...]
    matcher: _SubstringTrie


class TerminologyEngine:
    """Engine managing glossary confidence, consensus, overrides, and hard-term masking."""

    def __init__(self, db: KnowledgeDatabase) -> None:
        self.db = db
        self._constraint_cache: dict[str, _HardTermSnapshot] = {}
        self._soft_suggestion_cache: dict[str, _SoftTermSnapshot] = {}

    def _invalidate_constraint_cache(self, scope_id: str) -> None:
        self._constraint_cache = {
            key: value
            for key, value in self._constraint_cache.items()
            if key != scope_id
        }
        self._soft_suggestion_cache = {
            key: value
            for key, value in self._soft_suggestion_cache.items()
            if key != scope_id
        }

    def record_evidence(
        self, scope_id: str, evidence: EvidenceRecord
    ) -> GlossaryDecision:
        """Records term evidence and updates term state according to confidence rules."""
        source = evidence.source_term.strip()
        target = evidence.target_term.strip()
        now = time.time()

        with self.db.transaction() as con:
            cur = con.cursor()
            # Check if term is user_locked or disabled
            cur.execute(
                "SELECT state, target_term, confidence FROM terms WHERE scope_id = ? AND source_term = ?",
                (scope_id, source),
            )
            existing = cur.fetchone()
            if existing:
                state, existing_target, conf = existing
                if state == TermState.USER_LOCKED.value:
                    return GlossaryDecision(
                        source_term=source,
                        target_term=existing_target,
                        state=TermState.USER_LOCKED,
                        confidence=1.0,
                    )
                if state == "disabled":
                    return None

            # Ensure term exists for foreign key constraint
            term_id = f"term_{scope_id}_{source}_{target}"
            cur.execute(
                """
                INSERT INTO terms (term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(term_id) DO NOTHING
                """,
                (term_id, scope_id, source, target, TermState.AUTO_SUGGESTED.value, 0.5, now, now),
            )

            # Insert evidence
            ev_id = f"ev_{scope_id}_{source}_{target}_{evidence.provenance}_{now}"
            cur.execute(
                """
                INSERT INTO term_evidence (evidence_id, term_id, evidence_kind, provenance, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ev_id, term_id, evidence.kind, evidence.provenance, now),
            )

            # Query all evidence for this source_term across targets
            cur.execute(
                """
                SELECT t.target_term, e.evidence_kind, e.provenance
                FROM term_evidence e
                JOIN terms t ON e.term_id = t.term_id
                WHERE t.scope_id = ? AND t.source_term = ?
                """,
                (scope_id, source),
            )
            rows = cur.fetchall()

            # Include current evidence if term row doesn't exist yet
            targets_evidence: dict[str, list[tuple[str, str]]] = {}
            for r_target, r_kind, r_prov in rows:
                targets_evidence.setdefault(r_target, []).append((r_kind, r_prov))
            targets_evidence.setdefault(target, []).append((evidence.kind, evidence.provenance))

            # Calculate confidence per target
            target_scores: dict[str, float] = {}
            target_state: dict[str, TermState] = {}
            for t_term, ev_list in targets_evidence.items():
                legacy_count = len({prov for kind, prov in ev_list if kind == "standalone_legacy"})
                route_count = len({prov for kind, prov in ev_list if kind == "route_consensus"})
                if legacy_count >= 2 or route_count >= 2:
                    score = 0.95
                    st = TermState.AUTO_LOCKED
                elif legacy_count >= 1 or route_count >= 1:
                    score = 0.75
                    st = TermState.AUTO_SUGGESTED
                else:
                    score = 0.50
                    st = TermState.CONFLICT
                target_scores[t_term] = score
                target_state[t_term] = st

            # Sort targets by score
            sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
            best_target, best_score = sorted_targets[0]

            # Check competing targets margin (must be >= 0.25 margin)
            final_state = target_state[best_target]
            if len(sorted_targets) > 1:
                second_target, second_score = sorted_targets[1]
                if best_score - second_score < 0.25:
                    final_state = TermState.CONFLICT

            # Upsert into terms table for best_target
            best_term_id = f"term_{scope_id}_{source}_{best_target}"
            cur.execute(
                """
                INSERT INTO terms (term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, source_term, target_term) DO UPDATE SET
                    state = excluded.state,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (best_term_id, scope_id, source, best_target, final_state.value, best_score, now, now),
            )
            self._invalidate_constraint_cache(scope_id)

            return GlossaryDecision(
                source_term=source,
                target_term=best_target,
                state=final_state,
                confidence=best_score,
            )

    def set_user_override(
        self, scope_id: str, source_term: str, target_term: str
    ) -> GlossaryDecision:
        """Explicitly locks a glossary term to user's desired target."""
        source = source_term.strip()
        target = target_term.strip()
        now = time.time()
        term_id = f"term_{scope_id}_{source}_{target}"

        with self.db.transaction() as con:
            cur = con.cursor()
            # Remove any non-user terms for this source
            cur.execute(
                "DELETE FROM terms WHERE scope_id = ? AND source_term = ? AND term_id != ?",
                (scope_id, source, term_id),
            )
            cur.execute(
                """
                INSERT INTO terms (term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(term_id) DO UPDATE SET
                    target_term = excluded.target_term,
                    state = excluded.state,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (term_id, scope_id, source, target, TermState.USER_LOCKED.value, 1.0, now, now),
            )
        self._invalidate_constraint_cache(scope_id)

        return GlossaryDecision(
            source_term=source,
            target_term=target,
            state=TermState.USER_LOCKED,
            confidence=1.0,
        )

    def disable_term(self, scope_id: str, source_term: str) -> None:
        """Disables a term from matching or locking."""
        source = source_term.strip()
        now = time.time()
        with self.db.transaction() as con:
            cur = con.cursor()
            cur.execute(
                "UPDATE terms SET state = 'disabled', updated_at = ? WHERE scope_id = ? AND source_term = ?",
                (now, scope_id, source),
            )
        self._invalidate_constraint_cache(scope_id)

    def get_term_decision(self, scope_id: str, source_term: str) -> GlossaryDecision | None:
        """Retrieves current glossary decision for a source term."""
        source = source_term.strip()
        with self.db.connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT target_term, state, confidence FROM terms WHERE scope_id = ? AND source_term = ? AND state != 'disabled'",
                (scope_id, source),
            )
            row = cur.fetchone()
            if not row:
                return None
            target, state_str, conf = row
            return GlossaryDecision(
                source_term=source,
                target_term=target,
                state=TermState(state_str),
                confidence=conf,
            )

    def extract_constraints(self, scope_id: str, text: str) -> Sequence[TermConstraint]:
        """Extracts hard locked term constraints for a given text, longest match first."""
        return tuple(
            constraint
            for constraint, _start, _end in self._constraint_matches(scope_id, text)
        )

    def _hard_snapshot(self, scope_id: str) -> _HardTermSnapshot:
        cached = self._constraint_cache.get(scope_id)
        if cached is None:
            with self.db.connect() as con:
                cur = con.cursor()
                cur.execute(
                    """
                    SELECT source_term, target_term
                    FROM terms
                    WHERE scope_id = ?
                      AND state IN ('user_locked', 'auto_locked')
                    """,
                    (scope_id,),
                )
                rows = cur.fetchall()
            ordered = tuple(
                (source, target)
                for source, target in sorted(
                    rows, key=lambda row: len(row[0]), reverse=True
                )
                if source
            )
            cached = _HardTermSnapshot(
                rows=tuple(
                    (
                        source,
                        target,
                        bool(re.match(r"^\w", source)),
                        bool(re.match(r".*\w$", source)),
                    )
                    for source, target in ordered
                ),
                matcher=_SubstringTrie.build(
                    tuple(source for source, _target in ordered)
                ),
            )
            self._constraint_cache = {
                **self._constraint_cache,
                scope_id: cached,
            }
        return cached

    @staticmethod
    def _word_character(character: str) -> bool:
        return character == "_" or character.isalnum()

    def _constraint_matches(
        self, scope_id: str, text: str
    ) -> tuple[tuple[TermConstraint, int, int], ...]:
        snapshot = self._hard_snapshot(scope_id)
        if len(text.lower()) != len(text):
            legacy_matches: list[tuple[TermConstraint, int, int]] = []
            matched_spans: list[tuple[int, int]] = []
            for source, target, _prefix, _suffix in snapshot.rows:
                for match in self._term_pattern(source).finditer(text):
                    start, end = match.span()
                    if any(
                        start < span_end and end > span_start
                        for span_start, span_end in matched_spans
                    ):
                        continue
                    matched_spans.append((start, end))
                    legacy_matches.append(
                        (TermConstraint(source_term=source, target_term=target), start, end)
                    )
            return tuple(legacy_matches)

        candidates: list[tuple[int, int, int]] = []
        for term_index, start, end in snapshot.matcher.find(text):
            _source, _target, prefix_boundary, suffix_boundary = snapshot.rows[term_index]
            if prefix_boundary and start > 0 and self._word_character(text[start - 1]):
                continue
            if suffix_boundary and end < len(text) and self._word_character(text[end]):
                continue
            candidates.append((term_index, start, end))

        # term_index preserves the old global longest-first priority.
        candidates.sort(key=lambda item: (item[0], item[1]))

        matches: list[tuple[TermConstraint, int, int]] = []
        matched_spans: list[tuple[int, int]] = []
        for term_index, start, end in candidates:
            if any(start < s_end and end > s_start for s_start, s_end in matched_spans):
                continue
            source, target, _prefix, _suffix = snapshot.rows[term_index]
            matched_spans.append((start, end))
            matches.append(
                (TermConstraint(source_term=source, target_term=target), start, end)
            )
        return tuple(matches)

    def _soft_snapshot(self, scope_id: str) -> _SoftTermSnapshot:
        cached = self._soft_suggestion_cache.get(scope_id)
        if cached is None:
            with self.db.connect() as con:
                rows = con.execute(
                    """
                    SELECT source_term, target_term, confidence
                    FROM terms
                    WHERE scope_id = ? AND state = 'auto_suggested'
                    """,
                    (scope_id,),
                ).fetchall()
            suggestions = tuple(
                TermSuggestion(
                    source_term=source,
                    suggested_target=target,
                    confidence=confidence,
                )
                for source, target, confidence in rows
                if source
            )
            cached = _SoftTermSnapshot(
                suggestions=suggestions,
                matcher=_SubstringTrie.build(
                    tuple(item.source_term for item in suggestions)
                ),
            )
            self._soft_suggestion_cache = {
                **self._soft_suggestion_cache,
                scope_id: cached,
            }
        return cached

    def match_soft_suggestions(
        self, scope_id: str, text: str
    ) -> Sequence[TermSuggestion]:
        """Returns each soft suggestion present in text once."""

        snapshot = self._soft_snapshot(scope_id)
        matched_indices = {
            term_index for term_index, _start, _end in snapshot.matcher.find(text)
        }
        return tuple(
            suggestion
            for index, suggestion in enumerate(snapshot.suggestions)
            if index in matched_indices
        )

    @staticmethod
    def _term_pattern(source_term: str) -> re.Pattern[str]:
        prefix = r"\b" if re.match(r"^\w", source_term) else ""
        suffix = r"\b" if re.match(r".*\w$", source_term) else ""
        return re.compile(f"{prefix}{re.escape(source_term)}{suffix}", re.IGNORECASE)

    def mask_hard_terms(
        self, scope_id: str, text: str
    ) -> tuple[str, HardTermMaskMetadata]:
        """Masks hard-locked terms with <_GT_0_>, <_GT_1_>... tokens before sending to providers."""
        masked, metadata, _constraints = self.prepare_hard_terms(scope_id, text)
        return masked, metadata

    def prepare_hard_terms(
        self, scope_id: str, text: str
    ) -> tuple[str, HardTermMaskMetadata, tuple[TermConstraint, ...]]:
        """Matches and masks hard terms with a single indexed scan."""

        matches = self._constraint_matches(scope_id, text)
        if not matches:
            return text, HardTermMaskMetadata(original_text=text, token_to_target={}), ()

        token_to_target: dict[str, str] = {}
        replacements: list[tuple[int, int, str]] = []
        constraints: list[TermConstraint] = []
        for index, (constraint, start, end) in enumerate(matches):
            token = f"<_GT_{index}_>"
            token_to_target[token] = constraint.target_term
            replacements.append((start, end, token))
            constraints.append(constraint)

        pieces: list[str] = []
        cursor = 0
        for start, end, token in sorted(replacements, key=lambda item: item[0]):
            pieces.append(text[cursor:start])
            pieces.append(token)
            cursor = end
        pieces.append(text[cursor:])
        return (
            "".join(pieces),
            HardTermMaskMetadata(original_text=text, token_to_target=token_to_target),
            tuple(constraints),
        )

    def restore_hard_terms(
        self, translated_text: str, meta: HardTermMaskMetadata
    ) -> str:
        """Restores masked hard terms with their locked target translations."""
        if not isinstance(translated_text, str):
            raise ValueError("Hard glossary output must be text.")
        expected_tokens = tuple(meta.token_to_target)
        token_pattern = re.compile(r"<_GT_(\d+)_>")
        found_tokens = tuple(match.group(0) for match in token_pattern.finditer(translated_text))
        if Counter(found_tokens) != Counter(expected_tokens):
            raise ValueError(
                "Provider changed, removed, duplicated, or invented a hard glossary token."
            )
        result = translated_text
        for token, target in meta.token_to_target.items():
            result = result.replace(token, target, 1)
        return result
