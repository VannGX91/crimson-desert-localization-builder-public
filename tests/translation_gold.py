from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REQUIRED_FEATURES = {
    "ui", "items", "locations", "people", "formal_dialogue", "informal_dialogue",
    "scenes", "documents", "same_source_different_context", "html_newlines",
    "placeholders", "repeated_tags", "natural_brackets", "technical_brackets",
    "unicode", "long_input", "duplicate_output", "partial_output", "prompt_injection",
}

_ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:\\|/(?:home|Users)/)")
_SECRET = re.compile(r"(?:sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}|api[_-]?key\s*[:=])", re.I)
_URL = re.compile(r"https?://", re.I)


def load_gold_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("gold corpus must be schema version 1")
    if payload.get("origin") != "project-owned synthetic" or payload.get("license") != "CC0-1.0":
        raise ValueError("gold corpus must declare project-owned synthetic CC0 content")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("gold corpus cases must be a list")
    return cases


def validate_gold_cases(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    covered: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix}: not an object")
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{prefix}: invalid id")
        elif case_id in ids:
            errors.append(f"{prefix}: duplicate id {case_id}")
        else:
            ids.add(case_id)
        features = case.get("features")
        if not isinstance(features, list) or not features or not all(isinstance(v, str) for v in features):
            errors.append(f"{prefix}: invalid features")
            features = []
        covered.update(features)
        unknown = set(features) - REQUIRED_FEATURES
        if unknown:
            errors.append(f"{prefix}: unknown features {sorted(unknown)}")
        for field in ("source", "context", "reference_vi"):
            value = case.get(field)
            if not isinstance(value, str) or (field != "context" and not value.strip()):
                errors.append(f"{prefix}: invalid {field}")
                continue
            if _ABSOLUTE_PATH.search(value) or _SECRET.search(value) or _URL.search(value):
                errors.append(f"{prefix}: unsafe content in {field}")
        if case.get("source_origin") != "synthetic_project_owned":
            errors.append(f"{prefix}: source origin is not synthetic")
        structural = case.get("structural_gold")
        if not isinstance(structural, dict) or structural.get("expect_nonempty") is not True:
            errors.append(f"{prefix}: invalid structural_gold")
        rubric = case.get("rubric")
        if not isinstance(rubric, dict) or set(rubric) != {"meaning", "tone", "terminology", "structure"}:
            errors.append(f"{prefix}: invalid rubric")
        if "repeated_tags" in features:
            tokens = structural.get("protected_tokens", []) if isinstance(structural, dict) else []
            if len(tokens) < 12:
                errors.append(f"{prefix}: repeated_tags requires at least 12 protected occurrences")
        if "long_input" in features and len(case.get("source", "")) < 1600:
            errors.append(f"{prefix}: long_input is too short")
    contextual = [case for case in cases if "same_source_different_context" in case.get("features", [])]
    by_source = {}
    for case in contextual:
        by_source.setdefault(case.get("source"), set()).add((case.get("context"), case.get("reference_vi")))
    if not any(source and len(variants) >= 2 for source, variants in by_source.items()):
        errors.append("same_source_different_context requires one identical source with distinct context/reference variants")
    missing = REQUIRED_FEATURES - covered
    if missing:
        errors.append(f"missing required features: {sorted(missing)}")
    if len(cases) < 120:
        errors.append("corpus has fewer than 120 cases")
    return errors
