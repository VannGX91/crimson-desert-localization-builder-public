"""Translation runtime error definitions."""

from __future__ import annotations


class TranslationRuntimeError(Exception):
    """Base exception for all translation runtime errors."""


class ContractValidationError(TranslationRuntimeError):
    """Raised when a translation contract or request invariant is violated."""


class ProviderError(TranslationRuntimeError):
    """Raised when an external translation provider fails."""


class CheckpointError(TranslationRuntimeError):
    """Raised when a checkpoint file is corrupt or cannot be appended."""


class GlossaryError(TranslationRuntimeError):
    """Raised when a glossary constraint is violated or conflicted."""


class ProtectedContentError(TranslationRuntimeError):
    """Raised when protected content tokens are corrupted or modified."""
