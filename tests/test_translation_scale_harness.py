from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_translation_scale.py"
spec = importlib.util.spec_from_file_location("benchmark_translation_scale", SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(bench)


class TranslationScaleHarnessTests(unittest.TestCase):
    def test_frozen_profile_matches_scale_contract(self) -> None:
        profile = bench.FROZEN_PROFILE
        self.assertEqual(190_905, profile.catalog_rows)
        self.assertEqual(13_376, profile.delta_rows)
        self.assertEqual(8_714, profile.checkpoint_rows)
        self.assertEqual(24_334, profile.term_rows)
        self.assertEqual(56_485, profile.context_rows)

    def test_small_profile_runs_offline_with_exact_counts(self) -> None:
        profile = bench.ScaleProfile(
            catalog_rows=64,
            delta_rows=13,
            checkpoint_rows=8,
            term_rows=11,
            context_rows=17,
            search_iterations=12,
        )
        with tempfile.TemporaryDirectory() as td:
            result = bench.run_scale_benchmark(profile=profile, root=Path(td), seed=20260910)

        self.assertEqual("success", result["status"])
        self.assertEqual(
            {
                "catalog_rows": 64,
                "delta_rows": 13,
                "checkpoint_rows": 8,
                "pending_rows": 5,
                "term_rows": 11,
                "context_rows": 17,
            },
            result["counts"],
        )
        self.assertEqual([], result["network_calls"])
        self.assertTrue(result["temporary_paths_only"])
        self.assertGreater(result["metrics"]["database_size_bytes"], 0)
        self.assertGreaterEqual(result["metrics"]["peak_rss_bytes"], 1)
        self.assertGreaterEqual(result["metrics"]["search_p95_ms"], 0.0)
        self.assertIn("python", result["runtime"])
        self.assertIn("platform", result["machine"])

    def test_acceptance_evaluation_is_strict(self) -> None:
        metrics = {
            "step1_import_seconds": 90.0,
            "peak_rss_bytes": int(1.5 * 1024**3),
            "checkpoint_load_seconds": 10.0,
            "scheduler_planning_seconds": 15.0,
            "search_p95_ms": 99.999,
        }
        gates = bench.evaluate_gates(metrics, exact_counts=True)
        self.assertTrue(all(gates.values()))

        metrics["search_p95_ms"] = 100.0
        self.assertFalse(bench.evaluate_gates(metrics, exact_counts=True)["search_p95"])
        self.assertFalse(bench.evaluate_gates(metrics, exact_counts=False)["exact_counts"])

    def test_cli_writes_reproducible_json_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = root / "report.json"
            exit_code = bench.main(
                [
                    "--output",
                    str(report),
                    "--small",
                    "--seed",
                    "7",
                ]
            )
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual(7, payload["seed"])
        self.assertEqual("success", payload["status"])
        self.assertNotIn("old.json", json.dumps(payload).lower())

    def test_direct_script_entrypoint_resolves_project_imports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            report = Path(td) / "direct-report.json"
            completed = subprocess.run(
                [sys.executable, "-B", str(SCRIPT), "--output", str(report), "--small"],
                cwd=SCRIPT.parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(0, completed.returncode, completed.stderr)


if __name__ == "__main__":
    unittest.main()
