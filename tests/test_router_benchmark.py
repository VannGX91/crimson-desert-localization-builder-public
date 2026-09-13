from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_free_router.py"
spec = importlib.util.spec_from_file_location("benchmark_free_router", SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(bench)


class RouterBenchmarkTests(unittest.TestCase):
    def _cases(self):
        return [
            {
                "id": "a",
                "source": "Open panel",
                "reference_vi": "Mở bảng",
                "structural_gold": {"protected_tokens": []},
            },
            {
                "id": "b",
                "source": "Hello {name}",
                "reference_vi": "Xin chào {name}",
                "structural_gold": {"protected_tokens": ["{name}"]},
            },
        ]

    def test_offline_fake_benchmark_ranks_only_structural_passes(self):
        def runner(model, case):
            if model == "bad":
                return ""
            return case["reference_vi"]

        out = bench.run_benchmark(["good", "bad"], self._cases(), runner)
        self.assertEqual(["good"], out["preferred_ranking"])
        self.assertEqual(1.0, out["models"][0]["structural_pass_rate"])

    def test_resume_redaction_and_evaluator_disagreement(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state.json"
            first = bench.run_benchmark(
                ["m1"], self._cases()[:1], lambda _m, c: c["reference_vi"], state_path=state
            )
            second = bench.run_benchmark(
                ["m1"], self._cases(), lambda _m, c: c["reference_vi"], state_path=state
            )
            self.assertEqual(1, first["models"][0]["case_count"])
            self.assertEqual(2, second["models"][0]["case_count"])
            raw = json.dumps(second)
            self.assertNotIn("authorization", raw.lower())
            self.assertNotIn("prompt_body", raw.lower())
            self.assertEqual(5.0, bench.semantic_score_a("Mở bảng", "Mở bảng"))
            self.assertGreaterEqual(bench.semantic_score_b("Xin chào bạn", "Chào bạn"), 4.0)
            self.assertLess(bench.semantic_score_a("unrelated", "Mở bảng"), 3.0)

    def test_live_explicit_models_skip_discovery(self):
        models = bench.resolve_models(
            live=True,
            models_arg="kilo_gateway:dated-free-model",
            routes_arg="opencode,kilo_gateway",
            state_path=Path("state.json"),
            max_models_per_route=2,
            discoverer=lambda *_args, **_kwargs: self.fail("discovery should be skipped"),
        )
        self.assertEqual(["kilo_gateway:dated-free-model"], models)

    def test_scoring_and_structure_boundaries(self):
        self.assertEqual(1.0, bench.semantic_score_a("", "reference"))
        self.assertEqual(4.5, bench.semantic_score_a("abcdef", "abcdeg"))
        self.assertEqual(4.0, bench.semantic_score_a("ab", "abcdef"))
        self.assertEqual(3.0, bench.semantic_score_a("a", "abcdef"))
        self.assertEqual(2.0, bench.semantic_score_a("a", "abcdefghij"))
        self.assertEqual(1.0, bench.semantic_score_a("x", "abcdefghij"))
        self.assertEqual(1.0, bench.semantic_score_b("", "reference"))
        self.assertEqual(4.5, bench.semantic_score_b("a b", "a b c d"))
        self.assertEqual(4.0, bench.semantic_score_b("a b", "a b c d e f"))
        self.assertEqual(3.0, bench.semantic_score_b("a", "a b c d e"))
        self.assertEqual(1.5, bench.semantic_score_b("x", "a b c"))
        self.assertFalse(bench._structural_pass("", self._cases()[0]))
        self.assertFalse(bench._structural_pass("Xin chào", self._cases()[1]))

    def test_exception_rate_limit_invalid_state_and_empty_model(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state.json"
            state.write_text('{"schema_version": 999}', encoding="utf-8")
            with patch.object(bench.time, "sleep") as sleeper:
                out = bench.run_benchmark(
                    ["broken"],
                    self._cases()[:1],
                    lambda _model, _case: (_ for _ in ()).throw(RuntimeError("offline")),
                    state_path=state,
                    rate_limit_seconds=0.01,
                )
            self.assertEqual([], out["preferred_ranking"])
            self.assertEqual("RuntimeError", json.loads(state.read_text(encoding="utf-8"))["results"]["broken|a"]["error_class"])
            sleeper.assert_called_once_with(0.01)
        empty = bench.run_benchmark(["empty"], [], lambda _model, _case: "unused")
        self.assertEqual(0, empty["models"][0]["case_count"])

    def test_load_gold_model_resolution_and_adapter_factory(self):
        with tempfile.TemporaryDirectory() as td:
            gold = Path(td) / "gold.json"
            gold.write_text(json.dumps({"cases": self._cases()}), encoding="utf-8")
            self.assertEqual(1, len(bench.load_gold(gold, 1)))
            self.assertEqual(2, len(bench.load_gold(gold)))
        discovered = bench.resolve_models(
            live=True,
            models_arg="",
            routes_arg="opencode,kilo_gateway",
            state_path=Path("state.json"),
            max_models_per_route=1,
            discoverer=lambda routes, db, limit: [f"{routes[0]}:{db.stem}:{limit}"],
        )
        self.assertEqual(["opencode:state:1"], discovered)
        self.assertEqual(
            ["offline-one", "offline-two"],
            bench.resolve_models(
                live=False,
                models_arg="offline-one,offline-two",
                routes_arg="",
                state_path=Path("state.json"),
                max_models_per_route=1,
            ),
        )
        self.assertEqual(
            ["reference-fake"],
            bench.resolve_models(
                live=False,
                models_arg="",
                routes_arg="",
                state_path=Path("state.json"),
                max_models_per_route=1,
            ),
        )
        for route in ("opencode", "kilo_gateway", "kilo_anonymous", "unclose", "duckduckgo", "cloudflare", "aihorde"):
            self.assertIsNotNone(bench._make_adapter(route, "synthetic-model"))
        with self.assertRaises(ValueError):
            bench._make_adapter("unknown", "synthetic-model")

    def test_default_live_sample_covers_one_full_category_cycle(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "translation_gold_v1.json"
        cases = bench.load_gold(fixture, bench.DEFAULT_MAX_CASES)
        self.assertEqual(19, len(cases))
        categories = {case["category"] for case in cases}
        self.assertTrue(
            {
                "placeholders",
                "repeated_tags",
                "technical_brackets",
                "unicode",
                "long_input",
                "prompt_injection",
            }.issubset(categories)
        )

    def test_live_runner_success_and_failure_boundaries(self):
        success = SimpleNamespace(
            successes=(SimpleNamespace(translated_text="Bản dịch"),),
            failures=(),
        )
        adapter = SimpleNamespace(translate_batch=lambda _request: success)
        with patch.object(bench, "_make_adapter", return_value=adapter):
            self.assertEqual("Bản dịch", bench.live_runner("kilo_gateway:model", self._cases()[0]))
        failure = SimpleNamespace(
            successes=(),
            failures=(SimpleNamespace(kind=SimpleNamespace(value="timeout")),),
        )
        with patch.object(
            bench,
            "_make_adapter",
            return_value=SimpleNamespace(translate_batch=lambda _request: failure),
        ):
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                bench.live_runner("kilo_gateway:model", self._cases()[0])
        no_detail = SimpleNamespace(successes=(), failures=())
        with patch.object(
            bench,
            "_make_adapter",
            return_value=SimpleNamespace(translate_batch=lambda _request: no_detail),
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                bench.live_runner("kilo_gateway:model", self._cases()[0])

    def test_main_offline_writes_reproducible_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gold = root / "gold.json"
            output = root / "result.json"
            state = root / "state.json"
            gold.write_text(json.dumps({"cases": self._cases()}), encoding="utf-8")
            argv = [
                "benchmark_free_router.py",
                "--gold",
                str(gold),
                "--output",
                str(output),
                "--state",
                str(state),
                "--max-cases",
                "1",
            ]
            with patch.object(bench.sys, "argv", argv):
                self.assertEqual(0, bench.main())
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(["reference-fake"], result["preferred_ranking"])

if __name__ == "__main__":
    unittest.main()
