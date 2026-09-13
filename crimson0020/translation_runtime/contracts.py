"""Provider-neutral immutable contracts for the translation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from crimson0020.translation_runtime.errors import ContractValidationError


class FailureKind(str, Enum):
    """Classified failure kinds covering all failure modes specified in PLAN.md."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNAUTHORIZED = "unauthorized"
    UNSUPPORTED_MODEL = "unsupported_model"
    INVALID_STRUCTURE = "invalid_structure"
    MISSING_ITEM = "missing_item"
    DUPLICATE_ITEM = "duplicate_item"
    EMPTY_TRANSLATION = "empty_translation"
    PROTECTED_CONTENT_VIOLATION = "protected_content_violation"
    GLOSSARY_VIOLATION = "glossary_violation"
    CANCELLATION = "cancellation"
    PROVIDER_INTERNAL = "provider_internal"


@dataclass(frozen=True, slots=True)
class TermConstraint:
    """Hard glossary constraint mapping source term to required target term."""

    source_term: str
    target_term: str

    def __post_init__(self) -> None:
        if not self.source_term or not isinstance(self.source_term, str):
            raise ContractValidationError("source_term must be a non-empty string")
        if not self.target_term or not isinstance(self.target_term, str):
            raise ContractValidationError("target_term must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"source_term": self.source_term, "target_term": self.target_term}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TermConstraint:
        return cls(
            source_term=data["source_term"],
            target_term=data["target_term"],
        )


@dataclass(frozen=True, slots=True)
class TermSuggestion:
    """Soft glossary suggestion for prompting context."""

    source_term: str
    suggested_target: str
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_term or not isinstance(self.source_term, str):
            raise ContractValidationError("source_term must be a non-empty string")
        if not self.suggested_target or not isinstance(self.suggested_target, str):
            raise ContractValidationError("suggested_target must be a non-empty string")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ContractValidationError("confidence must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_term": self.source_term,
            "suggested_target": self.suggested_target,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TermSuggestion:
        return cls(
            source_term=data["source_term"],
            suggested_target=data["suggested_target"],
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class TranslationRequestItem:
    """Individual item in a batch translation request.

    Guaranteed not to expose sensitive identity (marker, key, file path).
    """

    work_id: str
    masked_text: str
    context: tuple[str, ...]
    context_signature: str
    hard_terms: tuple[TermConstraint, ...]
    soft_terms: tuple[TermSuggestion, ...]

    def __post_init__(self) -> None:
        if not self.work_id or not isinstance(self.work_id, str):
            raise ContractValidationError("work_id must be a non-empty string")
        if not isinstance(self.masked_text, str):
            raise ContractValidationError("masked_text must be a string")
        if not isinstance(self.context, tuple):
            object.__setattr__(self, "context", tuple(self.context))
        if not isinstance(self.context_signature, str):
            raise ContractValidationError("context_signature must be a string")
        if not isinstance(self.hard_terms, tuple):
            object.__setattr__(self, "hard_terms", tuple(self.hard_terms))
        if not isinstance(self.soft_terms, tuple):
            object.__setattr__(self, "soft_terms", tuple(self.soft_terms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "masked_text": self.masked_text,
            "context": list(self.context),
            "context_signature": self.context_signature,
            "hard_terms": [t.to_dict() for t in self.hard_terms],
            "soft_terms": [t.to_dict() for t in self.soft_terms],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationRequestItem:
        return cls(
            work_id=data["work_id"],
            masked_text=data["masked_text"],
            context=tuple(data.get("context", ())),
            context_signature=data.get("context_signature", ""),
            hard_terms=tuple(
                TermConstraint.from_dict(t) for t in data.get("hard_terms", ())
            ),
            soft_terms=tuple(
                TermSuggestion.from_dict(t) for t in data.get("soft_terms", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class TranslationBatchRequest:
    """Batch of items submitted to a translation provider."""

    job_id: str
    source_language: str
    target_language: str
    items: tuple[TranslationRequestItem, ...]

    def __post_init__(self) -> None:
        if not self.job_id or not isinstance(self.job_id, str):
            raise ContractValidationError("job_id must be a non-empty string")
        if not self.source_language or not isinstance(self.source_language, str):
            raise ContractValidationError("source_language must be a non-empty string")
        if not self.target_language or not isinstance(self.target_language, str):
            raise ContractValidationError("target_language must be a non-empty string")
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationBatchRequest:
        return cls(
            job_id=data["job_id"],
            source_language=data["source_language"],
            target_language=data["target_language"],
            items=tuple(
                TranslationRequestItem.from_dict(item)
                for item in data.get("items", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class TranslationItemResult:
    """Successfully translated item result."""

    work_id: str
    translated_text: str
    route_id: str
    model_id: str

    def __post_init__(self) -> None:
        if not self.work_id or not isinstance(self.work_id, str):
            raise ContractValidationError("work_id must be a non-empty string")
        if not isinstance(self.translated_text, str):
            raise ContractValidationError("translated_text must be a string")
        if not isinstance(self.route_id, str):
            raise ContractValidationError("route_id must be a string")
        if not isinstance(self.model_id, str):
            raise ContractValidationError("model_id must be a string")

    def to_dict(self) -> dict[str, str]:
        return {
            "work_id": self.work_id,
            "translated_text": self.translated_text,
            "route_id": self.route_id,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationItemResult:
        return cls(
            work_id=data["work_id"],
            translated_text=data["translated_text"],
            route_id=data["route_id"],
            model_id=data["model_id"],
        )


@dataclass(frozen=True, slots=True)
class TranslationFailure:
    """Classified failure for an item or request."""

    work_id: str
    kind: FailureKind
    retryable: bool
    message: str
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.work_id or not isinstance(self.work_id, str):
            raise ContractValidationError("work_id must be a non-empty string")
        if not isinstance(self.kind, FailureKind):
            if isinstance(self.kind, str):
                object.__setattr__(self, "kind", FailureKind(self.kind))
            else:
                raise ContractValidationError(f"Invalid failure kind: {self.kind}")
        if not isinstance(self.message, str):
            raise ContractValidationError("message must be a string")
        if self.retry_after_seconds is not None:
            if not isinstance(self.retry_after_seconds, (int, float)) or self.retry_after_seconds < 0:
                raise ContractValidationError("retry_after_seconds must be a non-negative float")

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "kind": self.kind.value,
            "retryable": self.retryable,
            "message": self.message,
            "retry_after_seconds": self.retry_after_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationFailure:
        return cls(
            work_id=data["work_id"],
            kind=FailureKind(data["kind"]),
            retryable=bool(data["retryable"]),
            message=data["message"],
            retry_after_seconds=(
                float(data["retry_after_seconds"])
                if data.get("retry_after_seconds") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    """Audit log entry for an attempted route and model during request execution."""

    route_id: str
    model_id: str
    duration_ms: float = 0.0
    error_kind: FailureKind | None = None
    status_code: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str):
            raise ContractValidationError("route_id must be a string")
        if not isinstance(self.model_id, str):
            raise ContractValidationError("model_id must be a string")
        if self.error_kind is not None and not isinstance(self.error_kind, FailureKind):
            if isinstance(self.error_kind, str):
                object.__setattr__(self, "error_kind", FailureKind(self.error_kind))
            else:
                raise ContractValidationError(f"Invalid error_kind: {self.error_kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "model_id": self.model_id,
            "duration_ms": self.duration_ms,
            "error_kind": self.error_kind.value if self.error_kind else None,
            "status_code": self.status_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteAttempt:
        return cls(
            route_id=data["route_id"],
            model_id=data["model_id"],
            duration_ms=float(data.get("duration_ms", 0.0)),
            error_kind=(
                FailureKind(data["error_kind"]) if data.get("error_kind") else None
            ),
            status_code=data.get("status_code"),
        )


@dataclass(frozen=True, slots=True)
class TranslationBatchOutcome:
    """Complete batch outcome containing successes, failures, and routing attempts."""

    successes: tuple[TranslationItemResult, ...]
    failures: tuple[TranslationFailure, ...]
    attempts: tuple[RouteAttempt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.successes, tuple):
            object.__setattr__(self, "successes", tuple(self.successes))
        if not isinstance(self.failures, tuple):
            object.__setattr__(self, "failures", tuple(self.failures))
        if not isinstance(self.attempts, tuple):
            object.__setattr__(self, "attempts", tuple(self.attempts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": [s.to_dict() for s in self.successes],
            "failures": [f.to_dict() for f in self.failures],
            "attempts": [a.to_dict() for a in self.attempts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationBatchOutcome:
        return cls(
            successes=tuple(
                TranslationItemResult.from_dict(s)
                for s in data.get("successes", ())
            ),
            failures=tuple(
                TranslationFailure.from_dict(f) for f in data.get("failures", ())
            ),
            attempts=tuple(
                RouteAttempt.from_dict(a) for a in data.get("attempts", ())
            ),
        )
