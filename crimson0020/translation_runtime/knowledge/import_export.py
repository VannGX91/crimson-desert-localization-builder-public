"""Knowledge export and import serialization structures and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from crimson0020.translation_runtime.errors import TranslationRuntimeError

CURRENT_EXPORT_VERSION = 1
MAX_PACKAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MiB limit


class KnowledgeImportError(TranslationRuntimeError):
    """Raised when imported package is malformed or invalid."""


@dataclass(frozen=True, slots=True)
class KnowledgeExportPackage:
    """Represents a portable, serialized knowledge package."""

    schema_version: int
    scope: dict[str, str]
    segments: list[dict[str, Any]]
    terms: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "segments": self.segments,
            "terms": self.terms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeExportPackage:
        if not isinstance(data, dict):
            raise KnowledgeImportError("Package data must be a dictionary.")
        version = data.get("schema_version")
        if version != CURRENT_EXPORT_VERSION:
            raise KnowledgeImportError(
                f"Unsupported package schema_version: {version}. Expected: {CURRENT_EXPORT_VERSION}"
            )
        scope = data.get("scope")
        if not isinstance(scope, dict):
            raise KnowledgeImportError("Package missing valid 'scope' object.")
        segments = data.get("segments", [])
        if not isinstance(segments, list):
            raise KnowledgeImportError("Field 'segments' must be a list.")
        terms = data.get("terms", [])
        if not isinstance(terms, list):
            raise KnowledgeImportError("Field 'terms' must be a list.")

        return cls(
            schema_version=version,
            scope=scope,
            segments=segments,
            terms=terms,
        )


def export_knowledge_package(
    scope: dict[str, str],
    segments: list[dict[str, Any]],
    terms: list[dict[str, Any]],
) -> dict[str, Any]:
    """Serializes knowledge data into validated dictionary format."""
    pkg = KnowledgeExportPackage(
        schema_version=CURRENT_EXPORT_VERSION,
        scope=scope,
        segments=segments,
        terms=terms,
    )
    return pkg.to_dict()


def import_knowledge_package(data: dict[str, Any]) -> KnowledgeExportPackage:
    """Validates and deserializes knowledge export package."""
    # Check JSON serializability & approximate size
    serialized = json.dumps(data)
    if len(serialized.encode("utf-8")) > MAX_PACKAGE_SIZE_BYTES:
        raise KnowledgeImportError("Package exceeds 50 MiB limit.")
    return KnowledgeExportPackage.from_dict(data)
