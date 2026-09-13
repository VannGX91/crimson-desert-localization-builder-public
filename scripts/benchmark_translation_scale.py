"""Deterministic offline benchmark for translation scale and safety gates."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crimson0020.translation_runtime.batch_planner import BatchPlanner  # noqa: E402
from crimson0020.translation_runtime.checkpoint import (  # noqa: E402
    CheckpointRecord,
    append_checkpoint_batch,
    initialize_checkpoint,
    load_checkpoint,
)
from crimson0020.translation_runtime.contracts import TranslationRequestItem  # noqa: E402
from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase  # noqa: E402
from crimson0020.translation_runtime.knowledge.memory import (  # noqa: E402
    TranslationMemoryRepository,
)
from crimson0020.translation_runtime.knowledge.service import (  # noqa: E402
    KnowledgeService,
    ScopeIdentifier,
)
from crimson0020.translation_work import (  # noqa: E402
    prepare_translation_work,
    read_translation_delta,
)


GIB = 1024**3
SCHEMA_VERSION = 1


class ScaleProfile(NamedTuple):
    catalog_rows: int
    delta_rows: int
    checkpoint_rows: int
    term_rows: int
    context_rows: int
    search_iterations: int = 100


FROZEN_PROFILE = ScaleProfile(
    catalog_rows=190_905,
    delta_rows=13_376,
    checkpoint_rows=8_714,
    term_rows=24_334,
    context_rows=56_485,
    search_iterations=100,
)

SMALL_PROFILE = ScaleProfile(
    catalog_rows=128,
    delta_rows=29,
    checkpoint_rows=17,
    term_rows=31,
    context_rows=43,
    search_iterations=12,
)


def _validate_profile(profile: ScaleProfile) -> None:
    values = tuple(profile)
    if any(value < 0 for value in values):
        raise ValueError("Scale counts must be non-negative.")
    if profile.delta_rows > profile.catalog_rows:
        raise ValueError("Delta rows cannot exceed catalog rows.")
    if profile.checkpoint_rows > profile.delta_rows:
        raise ValueError("Checkpoint rows cannot exceed delta rows.")
    if profile.search_iterations < 1:
        raise ValueError("Search iterations must be at least one.")


def _catalog_metadata(row_count: int) -> dict[str, object]:
    return {
        "format_version": 2,
        "layout": "multi-paloc",
        "package_folder": "0020",
        "palocs": [
            {
                "path": "synthetic/localization/eng/scale.paloc",
                "entry_count": row_count,
                "tail_bytes": "",
            }
        ],
    }


def _entry(index: int, *, changed: bool, old_catalog: bool, seed: int) -> dict[str, str]:
    marker = f"{index + 1:016x}"
    key = f"synthetic_key_{index:06d}"
    if changed:
        version = "old" if old_catalog else "new"
        original = f"Synthetic {version} source {seed}-{index} contextword"
    else:
        original = f"Synthetic stable source {seed}-{index} contextword"
    translation = f"Ban dich tong hop {seed}-{index}" if old_catalog else original
    return {
        "marker": marker,
        "key": key,
        "original": original,
        "translation": translation,
    }


def _write_catalog(path: Path, profile: ScaleProfile, *, old_catalog: bool, seed: int) -> None:
    changed_from = profile.catalog_rows - profile.delta_rows
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('{"metadata":')
        json.dump(_catalog_metadata(profile.catalog_rows), stream, ensure_ascii=False, separators=(",", ":"))
        stream.write(',"entries":[')
        for index in range(profile.catalog_rows):
            if index:
                stream.write(",")
            json.dump(
                _entry(
                    index,
                    changed=index >= changed_from,
                    old_catalog=old_catalog,
                    seed=seed,
                ),
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        stream.write("]}\n")


def _knowledge_package(
    profile: ScaleProfile,
    scope: ScopeIdentifier,
    lineage_id: str,
    seed: int,
) -> dict[str, object]:
    segments: list[dict[str, object]] = []
    for index in range(profile.context_rows):
        source = f"Synthetic memory contextword {seed}-{index}"
        segments.append(
            {
                "segment_id": f"scale_ctx_{index:06d}",
                "lineage_id": lineage_id,
                "marker": f"{index + 1:016x}",
                "key": f"context_key_{index:06d}",
                "source_hash": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "source_text": source,
                "target_text": f"Ngu canh tong hop {seed}-{index}",
                "context_signature": hashlib.sha256(f"ctx-{seed}-{index}".encode()).hexdigest(),
                "hard_terms_hash": "",
                "glossary_revision": 1,
                "status": "active",
            }
        )

    terms = [
        {
            "source_term": f"SyntheticTerm{index:05d}",
            "target_term": f"ThuatNgu{index:05d}",
            "state": "auto_suggested",
            "confidence": 0.75,
        }
        for index in range(profile.term_rows)
    ]
    return {
        "schema_version": 1,
        "scope": scope.to_dict(),
        "segments": segments,
        "terms": terms,
    }


def _peak_rss_bytes() -> int:
    """Return the process peak resident set without adding benchmark dependencies."""
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _database_size_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def _percentile_95(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _make_checkpoint(
    path: Path,
    candidates: Sequence[dict[str, object]],
    source_fingerprint: str,
) -> None:
    initialize_checkpoint(path, source_fingerprint, "eng", "vie")
    records: list[CheckpointRecord] = []
    for item in candidates:
        context_signature = item.get("context_signature", "")
        if not isinstance(context_signature, str):
            raise TypeError("Synthetic checkpoint context_signature must be text.")
        records.append(
            CheckpointRecord(
                marker=str(item["marker"]),
                key=str(item["key"]),
                original_sha256=hashlib.sha256(str(item["original"]).encode("utf-8")).hexdigest().upper(),
                translation=f"Da dich {item['key']}",
                route_id="synthetic",
                model_id="offline",
                context_signature=context_signature,
                glossary_revision=1,
            )
        )
    for start in range(0, len(records), 128):
        append_checkpoint_batch(path, records[start : start + 128])


def _plan_scheduler_rows(delta_entries: Sequence[dict[str, str]]) -> tuple[float, int]:
    items = tuple(
        TranslationRequestItem(
            work_id=f"scale_{index:06d}",
            masked_text=entry["original"],
            context=(),
            context_signature="",
            hard_terms=(),
            soft_terms=(),
        )
        for index, entry in enumerate(delta_entries)
    )
    start = time.perf_counter()
    batches = BatchPlanner().plan_batches("scale_job", "eng", "vie", items)
    elapsed = time.perf_counter() - start
    planned_rows = sum(len(batch.items) for batch in batches)
    return elapsed, planned_rows


def evaluate_gates(metrics: dict[str, float | int], *, exact_counts: bool) -> dict[str, bool]:
    return {
        "step1_import_time": float(metrics["step1_import_seconds"]) <= 90.0,
        "peak_rss": int(metrics["peak_rss_bytes"]) <= int(1.5 * GIB),
        "checkpoint_load": float(metrics["checkpoint_load_seconds"]) <= 10.0,
        "scheduler_planning": float(metrics["scheduler_planning_seconds"]) <= 15.0,
        "search_p95": float(metrics["search_p95_ms"]) < 100.0,
        "exact_counts": exact_counts,
    }


def _is_temporary_root(root: Path) -> bool:
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        resolved = root.resolve()
        return os.path.commonpath((str(temp_root), str(resolved))) == str(temp_root)
    except (OSError, ValueError):
        return False


def run_scale_benchmark(
    *,
    profile: ScaleProfile = FROZEN_PROFILE,
    root: Path,
    seed: int = 20260910,
) -> dict[str, object]:
    _validate_profile(profile)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    old_path = root / "old.json"
    new_path = root / "new.json"
    delta_path = root / "delta.json"
    db_path = root / "knowledge.sqlite3"
    checkpoint_path = root / "checkpoint.jsonl"

    _write_catalog(old_path, profile, old_catalog=True, seed=seed)
    _write_catalog(new_path, profile, old_catalog=False, seed=seed)

    db = KnowledgeDatabase(db_path)
    step1_start = time.perf_counter()
    work_result = prepare_translation_work(
        old_json=old_path,
        new_json=new_path,
        work_json=delta_path,
        knowledge_db=db,
    )
    if not work_result.lineage_id:
        raise RuntimeError("Step 1 did not return a knowledge lineage.")
    scope = ScopeIdentifier("Crimson0020", "0020", "eng", "vie")
    KnowledgeService(db).import_package(
        _knowledge_package(profile, scope, work_result.lineage_id, seed)
    )
    step1_import_seconds = time.perf_counter() - step1_start
    peak_rss_bytes = _peak_rss_bytes()

    delta = read_translation_delta(delta_path)
    delta_entries = delta["entries"]
    checkpoint_candidates = delta_entries[: profile.checkpoint_rows]
    source_fingerprint = str(delta["source"]["catalog_fingerprint"])
    _make_checkpoint(checkpoint_path, checkpoint_candidates, source_fingerprint)

    checkpoint_start = time.perf_counter()
    completed = load_checkpoint(
        checkpoint_path,
        source_fingerprint,
        "eng",
        "vie",
        checkpoint_candidates,
    )
    checkpoint_load_seconds = time.perf_counter() - checkpoint_start

    scheduler_planning_seconds, planned_rows = _plan_scheduler_rows(delta_entries)

    memory = TranslationMemoryRepository(db)
    search_samples_ms: list[float] = []
    for _ in range(profile.search_iterations):
        started = time.perf_counter()
        memory.search_fuzzy(scope.scope_id, "contextword", limit=5)
        search_samples_ms.append((time.perf_counter() - started) * 1000.0)

    with db.connect() as con:
        term_rows = int(
            con.execute("SELECT COUNT(*) FROM terms WHERE scope_id = ?", (scope.scope_id,)).fetchone()[0]
        )
        context_rows = int(
            con.execute(
                "SELECT COUNT(*) FROM translation_segments WHERE scope_id = ? AND context_signature != ''",
                (scope.scope_id,),
            ).fetchone()[0]
        )

    counts = {
        "catalog_rows": profile.catalog_rows,
        "delta_rows": len(delta_entries),
        "checkpoint_rows": len(completed),
        "pending_rows": len(delta_entries) - len(completed),
        "term_rows": term_rows,
        "context_rows": context_rows,
    }
    expected_counts = {
        "catalog_rows": profile.catalog_rows,
        "delta_rows": profile.delta_rows,
        "checkpoint_rows": profile.checkpoint_rows,
        "pending_rows": profile.delta_rows - profile.checkpoint_rows,
        "term_rows": profile.term_rows,
        "context_rows": profile.context_rows,
    }
    exact_counts = counts == expected_counts and planned_rows == profile.delta_rows

    metrics: dict[str, float | int] = {
        "step1_import_seconds": round(step1_import_seconds, 6),
        "peak_rss_bytes": peak_rss_bytes,
        "database_size_bytes": _database_size_bytes(db_path),
        "checkpoint_load_seconds": round(checkpoint_load_seconds, 6),
        "scheduler_planning_seconds": round(scheduler_planning_seconds, 6),
        "search_p95_ms": round(_percentile_95(search_samples_ms), 6),
    }
    gates = evaluate_gates(metrics, exact_counts=exact_counts)
    status = "success" if all(gates.values()) else "error"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "summary": "Synthetic translation scale gates passed." if status == "success" else "Synthetic translation scale gate failed.",
        "seed": seed,
        "profile": profile._asdict(),
        "counts": counts,
        "metrics": metrics,
        "gates": gates,
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "network_calls": [],
        "temporary_paths_only": _is_temporary_root(root),
        "next_actions": [] if status == "success" else ["Resolve failed performance gates and rerun the scale benchmark."],
        "artifacts": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--small", action="store_true", help="Run a fast contract smoke profile.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = SMALL_PROFILE if args.small else FROZEN_PROFILE
    with tempfile.TemporaryDirectory(prefix="crimson-eval03-") as td:
        result = run_scale_benchmark(profile=profile, root=Path(td), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
