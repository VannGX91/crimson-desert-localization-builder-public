"""Translation providers package."""

from __future__ import annotations

from crimson0020.translation_runtime.providers.base import (
    BaseTranslationAdapter,
    TransportCallable,
)
from crimson0020.translation_runtime.providers.google import GoogleFreeAdapter
from crimson0020.translation_runtime.providers.mymemory import MyMemoryFreeAdapter

__all__ = [
    "BaseTranslationAdapter",
    "GoogleFreeAdapter",
    "MyMemoryFreeAdapter",
    "TransportCallable",
]
