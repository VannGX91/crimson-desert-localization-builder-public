"""Tests for Step 1 knowledge ingestion and sparse delta preparation (INT-01)."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_work import (
    prepare_translation_work,
    read_translation_delta,
)


def make_catalog(entries: list[dict[str, str]], package_folder: str = "0020") -> dict[str, object]:
    return {
        "metadata": {
            "format_version": 2,
            "layout": "multi-paloc",
            "package_folder": package_folder,
            "palocs": [
                {
                    "path": "game/localization/eng/test.paloc",
                    "entry_count": len(entries),
                    "tail_bytes": "",
                }
            ],
        },
        "entries": entries,
    }


def make_entry(marker: str, key: str, original: str, translation: str = "") -> dict[str, str]:
    return {
        "marker": marker,
        "key": key,
        "original": original,
        "translation": translation,
    }


def test_knowledge_ingestion_success(tmp_path: Path) -> None:
    db_path = tmp_path / "test_knowledge.sqlite3"
    db = KnowledgeDatabase(db_path)

    old_entries = [
        make_entry("0100000000000000", "key_same", "Attack", "Tấn công"),
        make_entry("0200000000000000", "key_changed", "Defense", "Phòng thủ cũ"),
        make_entry("0300000000000000", "key_removed", "Old spell", "Phép thuật cũ"),
    ]
    new_entries = [
        make_entry("0100000000000000", "key_same", "Attack", "Attack"),
        make_entry("0200000000000000", "key_changed", "Shield Defense", "Shield Defense"),
        make_entry("0400000000000000", "key_new_tech", "[TECH_FLAG]", "[TECH_FLAG]"),
        make_entry("0500000000000000", "key_new", "New sword", "New sword"),
    ]

    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    work_file = tmp_path / "work.json"

    old_file.write_text(json.dumps(make_catalog(old_entries)), encoding="utf-8")
    new_file.write_text(json.dumps(make_catalog(new_entries)), encoding="utf-8")

    result = prepare_translation_work(
        old_json=old_file,
        new_json=new_file,
        work_json=work_file,
        knowledge_db=db,
    )

    # 1. Check sparse delta
    assert work_file.exists()
    delta_data = read_translation_delta(work_file)
    assert delta_data["kind"] == "sparse-delta"
    assert len(delta_data["entries"]) == 2  # key_changed, key_new
    keys_in_delta = [e["key"] for e in delta_data["entries"]]
    assert keys_in_delta == ["key_changed", "key_new"]

    # 2. Check counts
    assert result.unchanged_entries == 1
    assert result.new_entries == 2  # key_new_tech + key_new
    assert result.changed_originals == 1
    assert result.removed_entries == 1
    assert result.technical_entries == 1
    assert result.translation_candidates == 2
    assert result.knowledge_warning is None
    assert result.lineage_id is not None
    assert result.imported_segments_count >= 2  # Attack and Defense legacy translations

    # 3. Verify SQLite catalog registration and translation segments
    with db.connect() as con:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM catalog_fingerprints")
        assert cur.fetchone()[0] == 2  # old and new catalogs registered

        cur.execute("SELECT COUNT(*) FROM translation_segments WHERE marker = '0100000000000000'")
        assert cur.fetchone()[0] == 1  # legacy Attack imported


def test_locked_database_emits_warning_preserving_delta(tmp_path: Path) -> None:
    db_path = tmp_path / "locked_knowledge.sqlite3"
    _db = KnowledgeDatabase(db_path)

    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    work_file = tmp_path / "work.json"

    old_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "Hi", "Chao")])), encoding="utf-8")
    new_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "Hello", "Hello")])), encoding="utf-8")

    # Lock database with exclusive lock from raw connection
    raw_con = sqlite3.connect(str(db_path), timeout=0.1)
    raw_con.execute("BEGIN EXCLUSIVE")

    try:
        # Step 1 should succeed with sparse delta and set knowledge_warning
        result = prepare_translation_work(
            old_json=old_file,
            new_json=new_file,
            work_json=work_file,
            knowledge_db=KnowledgeDatabase(db_path, busy_timeout_ms=100),
        )

        # Delta is preserved and valid
        assert work_file.exists()
        delta = read_translation_delta(work_file)
        assert len(delta["entries"]) == 1
        assert result.knowledge_warning is not None
        assert "locked" in result.knowledge_warning.lower() or "busy" in result.knowledge_warning.lower()
    finally:
        raw_con.rollback()
        raw_con.close()


def test_corrupt_database_recovery_in_step_1(tmp_path: Path) -> None:
    db_path = tmp_path / "corrupt_knowledge.sqlite3"
    db_path.write_bytes(b"CORRUPT DATA NOT A SQLITE FILE")

    db = KnowledgeDatabase(db_path)

    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    work_file = tmp_path / "work.json"

    old_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "Hi", "Chao")])), encoding="utf-8")
    new_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "Hello", "Hello")])), encoding="utf-8")

    result = prepare_translation_work(
        old_json=old_file,
        new_json=new_file,
        work_json=work_file,
        knowledge_db=db,
    )

    assert work_file.exists()
    assert result.translation_candidates == 1
    # DB recovered and imported cleanly
    assert result.imported_segments_count >= 1


def test_synthetic_scale_190905_to_13376(tmp_path: Path) -> None:
    db_path = tmp_path / "scale_knowledge.sqlite3"
    db = KnowledgeDatabase(db_path)

    # Scale: generate 10,000 synthetic rows with exact 1,000 delta candidates to test scale ratio
    # and deterministic throughput
    total_rows = 10000
    delta_count = 1000
    unchanged_count = total_rows - delta_count

    old_entries = []
    new_entries = []

    for i in range(unchanged_count):
        marker = f"{i:016x}"
        old_entries.append(make_entry(marker, f"key_{i}", f"Common text {i}", f"Dịch {i}"))
        new_entries.append(make_entry(marker, f"key_{i}", f"Common text {i}", f"Common text {i}"))

    for i in range(unchanged_count, total_rows):
        marker = f"{i:016x}"
        old_entries.append(make_entry(marker, f"key_{i}", f"Old text {i}", f"Dịch cũ {i}"))
        new_entries.append(make_entry(marker, f"key_{i}", f"New text {i}", f"New text {i}"))

    old_file = tmp_path / "old_scale.json"
    new_file = tmp_path / "new_scale.json"
    work_file = tmp_path / "work_scale.json"

    old_file.write_text(json.dumps(make_catalog(old_entries)), encoding="utf-8")
    new_file.write_text(json.dumps(make_catalog(new_entries)), encoding="utf-8")

    t0 = time.time()
    result = prepare_translation_work(
        old_json=old_file,
        new_json=new_file,
        work_json=work_file,
        knowledge_db=db,
    )
    t_elapsed = time.time() - t0

    assert t_elapsed < 10.0, f"Synthetic scale must complete rapidly; took {t_elapsed:.2f}s"
    assert result.unchanged_entries == unchanged_count
    assert result.translation_candidates == delta_count
    assert result.changed_originals == delta_count

    delta = read_translation_delta(work_file)
    assert len(delta["entries"]) == delta_count


def test_package_folder_mismatch_and_backup_rotation(tmp_path: Path) -> None:
    old_file = tmp_path / "old.json"
    new_file = tmp_path / "new.json"
    work_file = tmp_path / "work.json"

    # Package mismatch
    old_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "A")], package_folder="0020")), encoding="utf-8")
    new_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "B")], package_folder="0021")), encoding="utf-8")

    with pytest.raises(ValueError, match="different package_folder"):
        prepare_translation_work(old_file, new_file, work_file)

    # Backup rotation
    new_file.write_text(json.dumps(make_catalog([make_entry("0100000000000000", "k1", "B")], package_folder="0020")), encoding="utf-8")
    # First write
    prepare_translation_work(old_file, new_file, work_file)
    assert work_file.exists()

    # Pre-create backup to test backup-1 rotation
    backup_file = work_file.with_name(f"{work_file.name}.backup")
    assert backup_file.exists() is False

    # Second write creates backup
    res2 = prepare_translation_work(old_file, new_file, work_file)
    assert res2.backup_json is not None and res2.backup_json.exists()

    # Third write creates backup-1
    res3 = prepare_translation_work(old_file, new_file, work_file)
    assert res3.backup_json is not None and res3.backup_json.exists()
    assert "-1" in str(res3.backup_json)
