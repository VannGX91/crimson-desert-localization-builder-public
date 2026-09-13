"""Robust LLM response parser recovering valid translations and classifying errors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ParsedResponseResult:
    """Outcome of parsing an LLM translation response."""

    is_success: bool
    translations: dict[str, str] = field(default_factory=dict)
    missing_ids: tuple[str, ...] = ()
    rejected_ids: tuple[str, ...] = ()
    error_message: str = ""


class RouterResponseParser:
    """Parses JSON translations, strips fences, deduplicates IDs, and isolates partials."""

    def parse_response(self, raw_text: str, expected_ids: Sequence[str]) -> ParsedResponseResult:
        if not raw_text or not raw_text.strip():
            return ParsedResponseResult(
                is_success=False,
                missing_ids=tuple(expected_ids),
                error_message="Empty response body from model",
            )

        cleaned = raw_text.strip()
        # Remove at most one markdown code fence
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Fallback attempt: find first { ... } or [ ... ]
            bracket_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if bracket_match:
                try:
                    data = json.loads(bracket_match.group(1))
                except Exception:
                    return ParsedResponseResult(
                        is_success=False,
                        missing_ids=tuple(expected_ids),
                        error_message=f"Invalid JSON format: {exc}",
                    )
            else:
                return ParsedResponseResult(
                    is_success=False,
                    missing_ids=tuple(expected_ids),
                    error_message=f"Invalid JSON format: {exc}",
                )

        items_list: list[dict] = []
        if isinstance(data, dict):
            if "translations" in data and isinstance(data["translations"], list):
                items_list = data["translations"]
            elif "items" in data and isinstance(data["items"], list):
                items_list = data["items"]
            else:
                # Could be a direct map { "id1": "text1", "id2": "text2" }
                items_list = [{"id": k, "text": v} for k, v in data.items() if isinstance(v, str)]
        elif isinstance(data, list):
            items_list = data

        translations: dict[str, str] = {}
        rejected_ids: list[str] = []
        seen_counts: dict[str, int] = {}

        expected_set = set(expected_ids)

        for item in items_list:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id or not isinstance(item_id, str):
                continue
            text = item.get("text")
            if text is None:
                text = item.get("translation") or item.get("translated")
            if not isinstance(text, str):
                rejected_ids.append(item_id)
                continue

            seen_counts[item_id] = seen_counts.get(item_id, 0) + 1
            if seen_counts[item_id] > 1:
                # Duplicate ID detected: reject it to avoid ambiguous mapping
                rejected_ids.append(item_id)
                if item_id in translations:
                    del translations[item_id]
                continue

            if item_id not in expected_set:
                # Unknown ID not requested in prompt
                rejected_ids.append(item_id)
                continue

            translations[item_id] = text

        missing_ids = [eid for eid in expected_ids if eid not in translations and eid not in rejected_ids]
        is_success = len(translations) == len(expected_ids) and not rejected_ids

        return ParsedResponseResult(
            is_success=is_success,
            translations=translations,
            missing_ids=tuple(missing_ids),
            rejected_ids=tuple(set(rejected_ids)),
            error_message="Partial response or ambiguous IDs" if not is_success else "",
        )
