"""Batch planning and envelope chunking conforming to provider constraints (INT-02)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from crimson0020.translation_runtime.contracts import (
    TranslationBatchRequest,
    TranslationRequestItem,
)


@dataclass(frozen=True, slots=True)
class BatchingLimits:
    """Configurable batch size constraints."""

    max_items: int = 48
    max_chars: int = 12000
    max_bytes: int = 65536  # 64 KiB


class BatchPlanner:
    """Plans translation items into bounded, provider-safe batch requests."""

    def __init__(self, limits: BatchingLimits | None = None) -> None:
        self.limits = limits or BatchingLimits()

    def plan_batches(
        self,
        job_id: str,
        source_language: str,
        target_language: str,
        items: Sequence[TranslationRequestItem],
    ) -> list[TranslationBatchRequest]:
        """Groups items into chunks adhering to item count, character, and byte size limits."""
        if not items:
            return []

        batches: list[TranslationBatchRequest] = []
        current_chunk: list[TranslationRequestItem] = []
        current_chars = 0
        batch_idx = 1

        for item in items:
            item_chars = len(item.masked_text)

            # Check if adding this item violates constraints
            new_item_count = len(current_chunk) + 1
            new_chars = current_chars + item_chars

            # Estimate envelope byte size
            estimated_bytes = 0
            if current_chunk:
                test_items = current_chunk + [item]
                test_dict = {
                    "job_id": f"{job_id}_b{batch_idx}",
                    "source_language": source_language,
                    "target_language": target_language,
                    "items": [it.to_dict() for it in test_items],
                }
                estimated_bytes = len(json.dumps(test_dict, ensure_ascii=False).encode("utf-8"))

            should_split = False
            if current_chunk:
                if new_item_count > self.limits.max_items:
                    should_split = True
                elif new_chars > self.limits.max_chars:
                    should_split = True
                elif estimated_bytes > self.limits.max_bytes:
                    should_split = True

            if should_split:
                batches.append(
                    TranslationBatchRequest(
                        job_id=f"{job_id}_b{batch_idx}",
                        source_language=source_language,
                        target_language=target_language,
                        items=tuple(current_chunk),
                    )
                )
                batch_idx += 1
                current_chunk = [item]
                current_chars = item_chars
            else:
                current_chunk.append(item)
                current_chars = new_chars

        if current_chunk:
            batches.append(
                TranslationBatchRequest(
                    job_id=f"{job_id}_b{batch_idx}",
                    source_language=source_language,
                    target_language=target_language,
                    items=tuple(current_chunk),
                )
            )

        return batches
