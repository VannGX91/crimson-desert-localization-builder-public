from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MAX_CASES = 19


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def semantic_score_a(output: str, reference: str) -> float:
    """Reference-aware character similarity, mapped to the 1-5 quality rubric."""
    a, b = _norm(output), _norm(reference)
    if not a or not b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.90:
        return 5.0
    if ratio >= 0.60:
        return 4.5
    if ratio >= 0.35:
        return 4.0
    if ratio >= 0.20:
        return 3.0
    if ratio >= 0.10:
        return 2.0
    return 1.0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", _norm(text), flags=re.UNICODE))


def semantic_score_b(output: str, reference: str) -> float:
    """Independent token-overlap evaluator with a separate scoring function."""
    out_tokens, ref_tokens = _tokens(output), _tokens(reference)
    if not out_tokens or not ref_tokens:
        return 1.0
    overlap = len(out_tokens & ref_tokens) / len(ref_tokens)
    if overlap >= 0.80:
        return 5.0
    if overlap >= 0.50:
        return 4.5
    if overlap >= 0.30:
        return 4.0
    if overlap >= 0.15:
        return 3.0
    return 1.5


def _structural_pass(output: str, case: dict[str, Any]) -> bool:
    if not isinstance(output, str) or not output.strip():
        return False
    structural = case.get("structural_gold", {})
    for token in structural.get("protected_tokens", []):
        if output.count(token) != case.get("source", "").count(token):
            return False
    return True


def _load_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"schema_version": 1, "results": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) and data.get("schema_version") == 1 else {"schema_version": 1, "results": {}}


def _save_state(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_benchmark(
    models: Iterable[str],
    cases: list[dict[str, Any]],
    runner: Callable[[str, dict[str, Any]], str],
    *,
    state_path: Path | None = None,
    evaluator_a: Callable[[str, str], float] = semantic_score_a,
    evaluator_b: Callable[[str, str], float] = semantic_score_b,
    rate_limit_seconds: float = 0.0,
) -> dict[str, Any]:
    state = _load_state(state_path)
    stored: dict[str, Any] = state.setdefault("results", {})
    summaries: list[dict[str, Any]] = []
    for model in models:
        model_results: list[dict[str, Any]] = []
        for case in cases:
            key = f"{model}|{case['id']}"
            if key in stored:
                record = stored[key]
            else:
                started = time.monotonic()
                try:
                    output = runner(model, case)
                    structural = _structural_pass(output, case)
                    score_a = evaluator_a(output, case["reference_vi"]) if structural else 1.0
                    score_b = evaluator_b(output, case["reference_vi"]) if structural else 1.0
                    error = None
                except Exception as exc:
                    output, structural, score_a, score_b = "", False, 1.0, 1.0
                    error = type(exc).__name__
                record = {
                    "model": model,
                    "case_id": case["id"],
                    "structural_pass": structural,
                    "evaluator_a": score_a,
                    "evaluator_b": score_b,
                    "severe_error": (not structural) or ((score_a + score_b) / 2.0 < 2.0),
                    "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                    "error_class": error,
                    "synthetic_output": output,
                }
                stored[key] = record
                _save_state(state_path, state)
                if rate_limit_seconds > 0:
                    time.sleep(rate_limit_seconds)
            model_results.append(record)
        count = len(model_results)
        structural_rate = sum(1 for r in model_results if r["structural_pass"]) / count if count else 0.0
        a_mean = sum(float(r["evaluator_a"]) for r in model_results) / count if count else 0.0
        b_mean = sum(float(r["evaluator_b"]) for r in model_results) / count if count else 0.0
        average = (a_mean + b_mean) / 2.0
        severe_rate = sum(1 for r in model_results if r["severe_error"]) / count if count else 1.0
        summaries.append({
            "model_id": model,
            "case_count": count,
            "structural_pass_rate": structural_rate,
            "evaluator_a_mean": round(a_mean, 4),
            "evaluator_b_mean": round(b_mean, 4),
            "average_score": round(average, 4),
            "severe_error_rate": severe_rate,
            "eligible": structural_rate == 1.0 and average >= 4.0 and severe_rate <= 0.02,
        })
    ranking = [r["model_id"] for r in sorted(summaries, key=lambda r: (-r["eligible"], -r["average_score"], r["model_id"])) if r["eligible"]]
    return {
        "schema_version": 1,
        "status": "success",
        "summary": "Synthetic free-router benchmark completed.",
        "models": summaries,
        "preferred_ranking": ranking,
        "evaluator_contract": ["character_similarity", "token_overlap"],
        "sensitive_fields_present": [],
    }


def load_gold(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = list(data["cases"])
    return cases[:limit] if limit else cases


def _make_adapter(route_id: str, model_id: str):
    if route_id == "opencode":
        from crimson0020.translation_runtime.router.adapters.opencode import OpenCodeAdapter
        return OpenCodeAdapter(model_id)
    if route_id in {"kilo_gateway", "kilo_anonymous"}:
        from crimson0020.translation_runtime.router.adapters.kilo import KiloAdapter
        return KiloAdapter(route_id, model_id)
    if route_id == "unclose":
        from crimson0020.translation_runtime.router.adapters.unclose import UncloseAdapter
        return UncloseAdapter(model_id)
    if route_id == "duckduckgo":
        from crimson0020.translation_runtime.router.adapters.duckduckgo import DuckDuckGoAdapter
        return DuckDuckGoAdapter(model_id)
    if route_id == "cloudflare":
        from crimson0020.translation_runtime.router.adapters.cloudflare import CloudflareAdapter
        return CloudflareAdapter(model_id)
    if route_id == "aihorde":
        from crimson0020.translation_runtime.router.adapters.aihorde import AIHordeAdapter
        return AIHordeAdapter(model_id)
    raise ValueError("unsupported route")


def live_runner(model: str, case: dict[str, Any]) -> str:
    from crimson0020.translation_runtime.contracts import TranslationBatchRequest, TranslationRequestItem
    route_id, model_id = model.split(":", 1)
    item = TranslationRequestItem(
        work_id="bench1",
        masked_text=case["source"],
        context=(case.get("context", ""),),
        context_signature="synthetic-benchmark",
        hard_terms=(), soft_terms=(),
    )
    req = TranslationBatchRequest(job_id="synthetic-quality-benchmark", source_language="en", target_language="vi", items=(item,))
    outcome = _make_adapter(route_id, model_id).translate_batch(req)
    if len(outcome.successes) != 1:
        kind = outcome.failures[0].kind.value if outcome.failures else "unavailable"
        raise RuntimeError(kind)
    return outcome.successes[0].translated_text


def discover_models(routes: list[str], db_path: Path, max_per_route: int = 2) -> list[str]:
    from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
    from crimson0020.translation_runtime.router.discovery import ModelDiscoveryService
    service = ModelDiscoveryService(KnowledgeDatabase(db_path))
    found: list[str] = []
    for route in routes:
        route_models = [m for m in service.discover_route_models(route, force_refresh=True) if m.is_free]
        for model in route_models[:max_per_route]:
            found.append(f"{route}:{model.model_id}")
    return found


def resolve_models(
    *,
    live: bool,
    models_arg: str,
    routes_arg: str,
    state_path: Path,
    max_models_per_route: int,
    discoverer: Callable[[list[str], Path, int], list[str]] | None = None,
) -> list[str]:
    explicit = [value for value in models_arg.split(",") if value]
    if explicit:
        return explicit
    if live:
        discover = discoverer or discover_models
        routes = [value for value in routes_arg.split(",") if value]
        return discover(routes, state_path.with_suffix(".sqlite3"), max_models_per_route)
    return ["reference-fake"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic quality benchmark for free AI router models")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--models", default="")
    parser.add_argument("--routes", default="opencode,kilo_gateway,kilo_anonymous,unclose,duckduckgo,cloudflare,aihorde")
    parser.add_argument("--max-models-per-route", type=int, default=2)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    cases = load_gold(args.gold, args.max_cases)
    models = resolve_models(
        live=args.live,
        models_arg=args.models,
        routes_arg=args.routes,
        state_path=args.state,
        max_models_per_route=args.max_models_per_route,
    )
    runner = live_runner if args.live else (lambda _model, case: case["reference_vi"])
    result = run_benchmark(models, cases, runner, state_path=args.state, rate_limit_seconds=0.25 if args.live else 0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["preferred_ranking"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
