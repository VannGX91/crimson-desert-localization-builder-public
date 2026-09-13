"""Protected-content engine for occurrence-safe token masking and restoration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from crimson0020.translation_runtime.contracts import FailureKind, TranslationFailure

_RAW_TOKEN_PATTERN = re.compile(
    r"\r\n|\r|\n|\\[nrt]"
    r"|<[^<>\r\n]+>"
    r"|\{[^{}\r\n]*\}"
    r"|\[[^\[\]\r\n]*\]"
    r"|%(?:\([^)]+\))?[#0\- +]?\d*(?:\.\d+)?[diouxXeEfFgGcrsa%]"
    r"|\$[A-Za-z_][A-Za-z0-9_]*"
    r"|&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);"
)

_MUTATED_TOKEN_PATTERN = re.compile(r"<[Xx]\s+\d+>|<X\d+>")
_PROTECTED_TAG_PATTERN = re.compile(r"<x(\d+)>")


def is_technical_bracket(text: str) -> bool:
    """Classifies whether a bracketed token is technical according to PLAN.md."""
    if not (text.startswith("[") and text.endswith("]")):
        return False
    inner = text[1:-1]
    if not inner:
        return False
    # Numeric / index token: [0], [12]
    if inner.isdigit():
        return True
    # Uppercase / digit / underscore identifier: [KEY_A], [ITEM_01]
    if re.match(r"^[A-Z0-9_]+$", inner):
        return True
    # If it contains spaces and is not a strict attribute, treat as natural language
    if " " in inner:
        if not re.match(r"^[a-zA-Z0-9_-]+\s*=\s*['\"]?[^'\"\s]+['\"]?$", inner):
            return False
    # Variable / control syntax
    if re.match(r"^%[a-zA-Z0-9_.*+-]+$", inner) or re.match(r"^\$[a-zA-Z0-9_]+$", inner):
        return True
    if inner.startswith("/") and re.match(r"^/[a-zA-Z0-9_=-]+$", inner):
        return True
    if re.match(r"^[a-zA-Z0-9_.-]+=[^\]\s]+$", inner):
        return True
    if re.match(r"^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$", inner):
        return True
    return False


@dataclass(frozen=True, slots=True)
class MaskResult:
    """Result of masking protected content occurrences in a text string."""

    original_text: str
    masked_text: str
    tokens: tuple[str, ...]


class ProtectionEngine:
    """Occurrence-safe protection engine."""

    @staticmethod
    def mask(text: str) -> MaskResult:
        return mask_protected_content(text)

    @staticmethod
    def restore(
        masked_translation: str,
        mask_result: MaskResult,
        work_id: str = "protected_item",
    ) -> tuple[str, TranslationFailure | None]:
        return restore_protected_content(masked_translation, mask_result, work_id=work_id)


def mask_protected_content(text: str) -> MaskResult:
    """Extracts protected tokens and replaces each occurrence with a short unique marker <xN>."""
    if not text:
        return MaskResult(original_text="", masked_text="", tokens=())

    tokens: list[str] = []
    output_parts: list[str] = []
    last_idx = 0

    for match in _RAW_TOKEN_PATTERN.finditer(text):
        raw_token = match.group(0)
        # Bracket tokens must be checked for technical status
        if raw_token.startswith("[") and raw_token.endswith("]"):
            if not is_technical_bracket(raw_token):
                continue

        start, end = match.span()
        output_parts.append(text[last_idx:start])
        marker = f"<x{len(tokens)}>"
        output_parts.append(marker)
        tokens.append(raw_token)
        last_idx = end

    output_parts.append(text[last_idx:])
    return MaskResult(
        original_text=text,
        masked_text="".join(output_parts),
        tokens=tuple(tokens),
    )


def restore_protected_content(
    masked_translation: str,
    mask_result: MaskResult,
    work_id: str = "protected_item",
) -> tuple[str, TranslationFailure | None]:
    """Restores protected tokens into the translated text in a single pass.

    Validates that no tokens are missing, duplicated, reordered, mutated, or hallucinated.
    """
    if _MUTATED_TOKEN_PATTERN.search(masked_translation):
        return masked_translation, TranslationFailure(
            work_id=work_id,
            kind=FailureKind.PROTECTED_CONTENT_VIOLATION,
            retryable=False,
            message="Mutated or malformed protected token detected in translation.",
        )

    found_matches = list(_PROTECTED_TAG_PATTERN.finditer(masked_translation))
    found_indices = [int(m.group(1)) for m in found_matches]
    expected_count = len(mask_result.tokens)

    # Check for unexpected/hallucinated tokens
    for idx in found_indices:
        if idx >= expected_count:
            return masked_translation, TranslationFailure(
                work_id=work_id,
                kind=FailureKind.PROTECTED_CONTENT_VIOLATION,
                retryable=False,
                message=f"Unexpected or fake protected token <x{idx}> found in translation.",
            )

    # Check for missing tokens
    found_set = set(found_indices)
    missing = [i for i in range(expected_count) if i not in found_set]
    if missing:
        return masked_translation, TranslationFailure(
            work_id=work_id,
            kind=FailureKind.PROTECTED_CONTENT_VIOLATION,
            retryable=False,
            message=f"Missing protected tokens: {', '.join(f'<x{i}>' for i in missing)}",
        )

    # Check for duplicate tokens
    if len(found_indices) > len(found_set):
        duplicates = [idx for idx in found_indices if found_indices.count(idx) > 1]
        return masked_translation, TranslationFailure(
            work_id=work_id,
            kind=FailureKind.PROTECTED_CONTENT_VIOLATION,
            retryable=False,
            message=f"Duplicate protected tokens found: {', '.join(f'<x{i}>' for i in set(duplicates))}",
        )

    # Check for reordering
    expected_indices = list(range(expected_count))
    if found_indices != expected_indices:
        return masked_translation, TranslationFailure(
            work_id=work_id,
            kind=FailureKind.PROTECTED_CONTENT_VIOLATION,
            retryable=False,
            message="Reordered protected tokens detected in translation.",
        )

    # Single-pass restoration
    def _replace_token(match: re.Match) -> str:
        token_index = int(match.group(1))
        return mask_result.tokens[token_index]

    restored = _PROTECTED_TAG_PATTERN.sub(_replace_token, masked_translation)
    return restored, None
