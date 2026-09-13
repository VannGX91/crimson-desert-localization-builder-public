"""Specification tests for Translation-Unit Builder and Batch Planner (INT-02)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from crimson0020.translation_runtime.contracts import (
    TermConstraint,
    TermSuggestion,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.memory import (
    TranslationMemoryRepository,
    TranslationSegment,
)
from crimson0020.translation_runtime.knowledge.service import ScopeIdentifier
from crimson0020.translation_runtime.knowledge.terminology import (
    EvidenceRecord,
    TerminologyEngine,
)
from crimson0020.translation_runtime.batch_planner import BatchPlanner, BatchingLimits
from crimson0020.translation_runtime.unit_builder import (
    RawTranslationInput,
    TranslationUnitBuilder,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> KnowledgeDatabase:
    db_file = tmp_path / "test_unit_builder.sqlite3"
    return KnowledgeDatabase(database_path=db_file)


@pytest.fixture
def scope(temp_db: KnowledgeDatabase) -> ScopeIdentifier:
    sc = ScopeIdentifier(
        app_id="Crimson0020",
        package_family="0020",
        source_language="eng",
        target_language="vie",
    )
    now = time.time()
    with temp_db.transaction() as con:
        con.execute(
            """
            INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_id) DO NOTHING
            """,
            (sc.scope_id, sc.app_id, sc.package_family, sc.source_language, sc.target_language, now),
        )
    return sc


def test_identical_source_different_context_not_collapsed(temp_db: KnowledgeDatabase, scope: ScopeIdentifier) -> None:
    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")

    inputs = [
        RawTranslationInput(
            marker="0100000000000000",
            key="btn_open_door",
            original="Open",
            context=("Door interaction menu",),
            context_signature="sig_door",
        ),
        RawTranslationInput(
            marker="0200000000000000",
            key="status_shop_open",
            original="Open",
            context=("Shop business hours",),
            context_signature="sig_shop",
        ),
    ]

    outcome = builder.build_units(inputs)

    # Neither string may be collapsed away (source-only global dedupe is prohibited)
    assert len(outcome.outbound_items) == 2
    assert outcome.outbound_items[0].work_id != outcome.outbound_items[1].work_id
    assert outcome.outbound_items[0].context_signature == "sig_door"
    assert outcome.outbound_items[1].context_signature == "sig_shop"
    assert len(outcome.exact_matches) == 0


def test_exact_tm_hit_skips_provider_batch(temp_db: KnowledgeDatabase, scope: ScopeIdentifier) -> None:
    memory = TranslationMemoryRepository(temp_db)
    # Insert exact segment in TM
    seg = TranslationSegment(
        segment_id="seg_exact_1",
        scope_id=scope.scope_id,
        lineage_id="lineage_1",
        marker="0100000000000000",
        key="potion_hp",
        source_hash="hash_hp",
        source_text="Health Potion",
        target_text="Bình máu",
        context_signature="sig_item",
        hard_terms_hash="",
        glossary_revision=1,
        status="active",
        created_at=1000.0,
        updated_at=1000.0,
    )
    memory.save_segment(seg)

    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")

    inputs = [
        RawTranslationInput(
            marker="0100000000000000",
            key="potion_hp",
            original="Health Potion",
            source_hash="hash_hp",
            context=("Item menu",),
            context_signature="sig_item",
        ),
        RawTranslationInput(
            marker="0200000000000000",
            key="potion_mp",
            original="Mana Potion",
            source_hash="hash_mp",
            context=("Item menu",),
            context_signature="sig_item",
        ),
    ]

    outcome = builder.build_units(inputs)

    # Exactly one item resolved via TM, zero provider request for it
    assert len(outcome.exact_matches) == 1
    assert outcome.exact_matches[("0100000000000000", "potion_hp")].translated_text == "Bình máu"

    # Only Mana Potion is outbound to provider
    assert len(outcome.outbound_items) == 1
    assert outcome.outbound_items[0].masked_text == "Mana Potion"


def test_fuzzy_tm_as_examples_only(temp_db: KnowledgeDatabase, scope: ScopeIdentifier) -> None:
    memory = TranslationMemoryRepository(temp_db)
    # Insert segment that matches similar text but has different marker/context
    seg = TranslationSegment(
        segment_id="seg_fuzzy_1",
        scope_id=scope.scope_id,
        lineage_id="lineage_1",
        marker="9900000000000000",
        key="other_key",
        source_hash="hash_other",
        source_text="Great Health Potion",
        target_text="Bình máu lớn",
        context_signature="sig_other",
        hard_terms_hash="",
        glossary_revision=1,
        status="active",
        created_at=1000.0,
        updated_at=1000.0,
    )
    memory.save_segment(seg)

    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")

    inputs = [
        RawTranslationInput(
            marker="0100000000000000",
            key="potion_hp",
            original="Health Potion",
            source_hash="hash_hp",
            context=("Item menu",),
            context_signature="sig_item",
        )
    ]

    outcome = builder.build_units(inputs)

    # Fuzzy match must NOT auto-resolve as exact match
    assert len(outcome.exact_matches) == 0
    assert len(outcome.outbound_items) == 1


def test_changed_glossary_and_overlapping_hard_terms(temp_db: KnowledgeDatabase, scope: ScopeIdentifier) -> None:
    terminology = TerminologyEngine(temp_db)
    # Insert 2 legacy evidence records for terms to become auto_locked
    terminology.record_evidence(scope.scope_id, EvidenceRecord("Fire Ball", "Quả cầu lửa", "standalone_legacy", "c1"))
    terminology.record_evidence(scope.scope_id, EvidenceRecord("Fire Ball", "Quả cầu lửa", "standalone_legacy", "c2"))

    terminology.record_evidence(scope.scope_id, EvidenceRecord("Fire", "Lửa", "standalone_legacy", "c1"))
    terminology.record_evidence(scope.scope_id, EvidenceRecord("Fire", "Lửa", "standalone_legacy", "c2"))

    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")

    inputs = [
        RawTranslationInput(
            marker="0100000000000000",
            key="spell_1",
            original="Cast Fire Ball now!",
        )
    ]

    outcome = builder.build_units(inputs)
    assert len(outcome.outbound_items) == 1
    item = outcome.outbound_items[0]

    # Overlapping hard term: longer term 'Fire Ball' must be masked first as <_GT_0_>, not 'Fire'
    assert "<_GT_0_>" in item.masked_text
    assert "Cast <_GT_0_> now!" == item.masked_text
    assert len(item.hard_terms) == 1
    assert item.hard_terms[0].source_term == "Fire Ball"
    assert item.hard_terms[0].target_term == "Quả cầu lửa"


def test_batching_at_each_limit() -> None:
    # 1. Item count limit (max 48)
    planner = BatchPlanner(limits=BatchingLimits(max_items=48, max_chars=100000, max_bytes=1000000))
    items_50 = [
        TranslationRequestItem(
            work_id=f"w_{i}",
            masked_text=f"Item {i}",
            context=(),
            context_signature="",
            hard_terms=(),
            soft_terms=(),
        )
        for i in range(50)
    ]
    batches_item = planner.plan_batches("job_1", "eng", "vie", items_50)
    assert len(batches_item) == 2
    assert len(batches_item[0].items) == 48
    assert len(batches_item[1].items) == 2

    # 2. Character limit (max 12,000 chars)
    planner_char = BatchPlanner(limits=BatchingLimits(max_items=100, max_chars=500, max_bytes=1000000))
    long_items = [
        TranslationRequestItem(
            work_id=f"w_long_{i}",
            masked_text="A" * 200,
            context=(),
            context_signature="",
            hard_terms=(),
            soft_terms=(),
        )
        for i in range(4)
    ]
    # Each item 200 chars, limit 500 chars -> items 0 and 1 (400), item 2 (200), item 3 (200) -> 3 batches
    batches_char = planner_char.plan_batches("job_2", "eng", "vie", long_items)
    assert len(batches_char) == 2
    assert len(batches_char[0].items) == 2  # 400 chars
    assert len(batches_char[1].items) == 2  # 400 chars

    # 3. Byte envelope limit (max 64 KiB = 65,536 bytes)
    planner_bytes = BatchPlanner(limits=BatchingLimits(max_items=100, max_chars=100000, max_bytes=1000))
    byte_items = [
        TranslationRequestItem(
            work_id=f"w_b_{i}",
            masked_text="X" * 300,
            context=(),
            context_signature="",
            hard_terms=(),
            soft_terms=(),
        )
        for i in range(5)
    ]
    batches_byte = planner_bytes.plan_batches("job_3", "eng", "vie", byte_items)
    assert len(batches_byte) >= 2

    # Empty items returns empty batches
    assert planner.plan_batches("job_empty", "eng", "vie", []) == []


def test_sensitive_field_absence() -> None:
    item = TranslationRequestItem(
        work_id="opaque_id_123",
        masked_text="Hello world",
        context=("Scene A",),
        context_signature="sig_123",
        hard_terms=(TermConstraint(source_term="Hello", target_term="Xin chao"),),
        soft_terms=(TermSuggestion(source_term="world", suggested_target="the gioi", confidence=0.8),),
    )

    item_dict = item.to_dict()
    # Sensitive identity & catalog fields must be strictly absent
    for forbidden in ("marker", "key", "file_path", "path", "catalog_fingerprint", "lineage_id"):
        assert forbidden not in item_dict
        assert not hasattr(item, forbidden)

    batch = TranslationBatchRequest(
        job_id="job_clean",
        source_language="eng",
        target_language="vie",
        items=(item,),
    )
    batch_dict = batch.to_dict()
    for forbidden in ("marker", "key", "file_path", "path", "catalog_fingerprint", "lineage_id"):
        assert forbidden not in batch_dict


def test_deterministic_unit_order(temp_db: KnowledgeDatabase, scope: ScopeIdentifier) -> None:
    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")

    inputs = [
        RawTranslationInput(marker=f"{i:016x}", key=f"k_{i}", original=f"Text {i}")
        for i in reversed(range(10))
    ]

    outcome1 = builder.build_units(inputs)
    outcome2 = builder.build_units(inputs)

    texts1 = [it.masked_text for it in outcome1.outbound_items]
    texts2 = [it.masked_text for it in outcome2.outbound_items]
    assert texts1 == texts2


def test_soft_terms_matching(temp_db: KnowledgeDatabase, scope: ScopeIdentifier) -> None:
    # Insert soft suggested term
    now = time.time()
    with temp_db.transaction() as con:
        con.execute(
            """
            INSERT INTO terms (term_id, scope_id, source_term, target_term, state, confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("term_soft_1", scope.scope_id, "Castle", "Lâu đài", "auto_suggested", 0.6, now, now),
        )

    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")
    inputs = [
        RawTranslationInput(marker="01", key="k1", original="Explore the Castle gate")
    ]
    outcome = builder.build_units(inputs)
    assert len(outcome.outbound_items) == 1
    assert len(outcome.outbound_items[0].soft_terms) == 1
    assert outcome.outbound_items[0].soft_terms[0].source_term == "Castle"
    assert outcome.outbound_items[0].soft_terms[0].suggested_target == "Lâu đài"


def test_builder_uses_single_hard_term_plan_per_outbound_item(
    temp_db: KnowledgeDatabase, scope: ScopeIdentifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminology = TerminologyEngine(temp_db)
    terminology.set_user_override(scope.scope_id, "Castle", "Lâu đài")
    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")

    prepare_calls = 0
    original_prepare = builder.terminology.prepare_hard_terms

    def counted_prepare(scope_id: str, text: str):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(scope_id, text)

    monkeypatch.setattr(builder.terminology, "prepare_hard_terms", counted_prepare)
    outcome = builder.build_units(
        [
            RawTranslationInput(marker="01", key="k1", original="Castle gate"),
            RawTranslationInput(marker="02", key="k2", original="Castle wall"),
        ]
    )

    assert len(outcome.outbound_items) == 2
    assert prepare_calls == 2


def test_builder_batches_exact_memory_lookup(
    temp_db: KnowledgeDatabase, scope: ScopeIdentifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = TranslationUnitBuilder(db=temp_db, scope=scope, lineage_id="lineage_1")
    batch_calls = 0
    original_many = builder.memory.lookup_exact_many

    def counted_many(requests):
        nonlocal batch_calls
        batch_calls += 1
        return original_many(requests)

    monkeypatch.setattr(builder.memory, "lookup_exact_many", counted_many)
    monkeypatch.setattr(
        builder.memory,
        "lookup_exact",
        lambda _request: (_ for _ in ()).throw(AssertionError("per-row lookup used")),
    )

    outcome = builder.build_units(
        [
            RawTranslationInput(marker=f"{index:016x}", key=f"k{index}", original=f"Text {index}")
            for index in range(20)
        ]
    )

    assert len(outcome.outbound_items) == 20
    assert batch_calls == 1
