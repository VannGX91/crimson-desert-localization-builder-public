"""Prompt envelope and builder for translation router models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from crimson0020.translation_runtime.contracts import (
    TranslationBatchRequest,
    TranslationRequestItem,
)

MAX_PROMPT_ITEMS = 48
MAX_TOTAL_CHARS = 12000
MAX_ENVELOPE_BYTES = 64 * 1024  # 64 KiB

SYSTEM_INSTRUCTION = (
    "You are a precise, deterministic game localization translation engine. "
    "Translate the given text segments from the source language to the target language. "
    "Rules:\n"
    "1. You MUST preserve all protected markup tags and placeholders (such as <x0>, <x1>, etc.) exactly as they appear.\n"
    "2. You MUST respect any provided terminology / glossary translations.\n"
    "3. Respond ONLY with a valid JSON object of the form: {\"translations\": [{\"id\": \"<item_id>\", \"text\": \"<translated_text>\"}]}.\n"
    "4. Do NOT include markdown code blocks, explanatory prose, conversational greetings, or additional keys.\n"
    "5. Do NOT alter, omit, or reorder item IDs."
)


@dataclass(frozen=True, slots=True)
class RouterPromptEnvelope:
    """Represents a bounded prompt ready for submission to an LLM provider."""

    system_instruction: str
    user_prompt: str
    item_ids: tuple[str, ...]
    temperature: float = 0.0


class PromptBuilder:
    """Constructs prompt envelopes conforming to size, delimiter, and security bounds."""

    def __init__(self, system_instruction: str = SYSTEM_INSTRUCTION) -> None:
        self.system_instruction = system_instruction

    def build_batch_prompt(
        self,
        batch: TranslationBatchRequest,
        extra_glossary: dict[str, str] | None = None,
    ) -> RouterPromptEnvelope:
        return self.build_prompt(
            items=batch.items,
            source_language=batch.source_language,
            target_language=batch.target_language,
            extra_glossary=extra_glossary,
        )

    def build_prompt(
        self,
        items: Sequence[TranslationRequestItem],
        source_language: str = "en",
        target_language: str = "vi",
        extra_glossary: dict[str, str] | None = None,
    ) -> RouterPromptEnvelope:
        if len(items) > MAX_PROMPT_ITEMS:
            raise ValueError(f"Batch size {len(items)} exceeds maximum allowed items {MAX_PROMPT_ITEMS}")

        total_chars = sum(len(item.masked_text) for item in items)
        if total_chars > MAX_TOTAL_CHARS:
            raise ValueError(f"Batch text length {total_chars} exceeds maximum allowed characters {MAX_TOTAL_CHARS}")

        # Collect hard and soft terms
        combined_glossary: dict[str, str] = dict(extra_glossary or {})
        for item in items:
            for ht in item.hard_terms:
                combined_glossary[ht.source_term] = ht.target_term
            for st in item.soft_terms:
                if st.source_term not in combined_glossary:
                    combined_glossary[st.source_term] = st.suggested_target

        payload_obj = {
            "source_language": source_language,
            "target_language": target_language,
            "glossary": combined_glossary,
            "items": [
                {
                    "id": item.work_id,
                    "source": item.masked_text,
                    "context": list(item.context),
                }
                for item in items
            ],
        }

        user_content = (
            f"Please translate the following items from {source_language} to {target_language}.\n"
            f"<TRANSLATION_INPUT>\n"
            f"{json.dumps(payload_obj, ensure_ascii=False, indent=2)}\n"
            f"</TRANSLATION_INPUT>"
        )

        envelope_bytes = len(user_content.encode("utf-8")) + len(self.system_instruction.encode("utf-8"))
        if envelope_bytes > MAX_ENVELOPE_BYTES:
            raise ValueError(f"Prompt envelope size {envelope_bytes} bytes exceeds {MAX_ENVELOPE_BYTES} bytes")

        return RouterPromptEnvelope(
            system_instruction=self.system_instruction,
            user_prompt=user_content,
            item_ids=tuple(item.work_id for item in items),
            temperature=0.0,
        )
