"""Context engine for deterministic scene, document, and register context assembly."""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass
from typing import Sequence


class ContextKind(str, enum.Enum):
    """Kinds of context windows."""

    SCENE = "scene"
    DOCUMENT = "document"


@dataclass(frozen=True, slots=True)
class ContextItem:
    """Individual item available to context window assembly."""

    index: int
    key: str
    text: str
    scene_id: str | None = None
    title: str | None = None
    document_path: str | None = None


@dataclass(frozen=True, slots=True)
class AssembledContext:
    """Bounded, assembled context text and deterministic signature."""

    context_text: str
    surrounding_text: str
    signature: str


class ContextEngine:
    """Engine providing deterministic, bounded context assembly without invented roles."""

    MAX_CONTEXT_CHARS = 1500

    def _truncate_safely(self, text: str, max_chars: int) -> str:
        """Truncates text at whitespace boundary within max_chars limit."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated.strip()

    def build_context(
        self,
        items: Sequence[ContextItem],
        target_index: int,
        kind: ContextKind = ContextKind.SCENE,
    ) -> AssembledContext:
        """Builds bounded context text and deterministic signature for target item."""
        if not items:
            return AssembledContext("", "", hashlib.sha256(b"").hexdigest())

        target_item = items[target_index]
        parts: list[str] = []
        surrounding_parts: list[str] = []

        if kind == ContextKind.DOCUMENT:
            if target_item.title:
                parts.append(f"Title: {target_item.title.strip()}")
            # Window +-1
            for i in range(max(0, target_index - 1), min(len(items), target_index + 2)):
                if i != target_index:
                    parts.append(items[i].text.strip())
                    surrounding_parts.append(items[i].text.strip())
        else:
            # Scene window +-2
            for i in range(max(0, target_index - 2), min(len(items), target_index + 3)):
                if i != target_index:
                    parts.append(items[i].text.strip())
                    surrounding_parts.append(items[i].text.strip())

        full_context_text = "\n".join(parts)
        truncated_context = self._truncate_safely(full_context_text, self.MAX_CONTEXT_CHARS)
        surrounding_text = "\n".join(surrounding_parts)

        # Deterministic context signature (excluding paths, IDs, mutable order)
        sig_basis = f"{kind.value}|{target_item.scene_id or ''}|{target_item.title or ''}|{truncated_context}"
        signature = hashlib.sha256(sig_basis.encode("utf-8")).hexdigest()

        return AssembledContext(
            context_text=truncated_context,
            surrounding_text=surrounding_text,
            signature=signature,
        )
