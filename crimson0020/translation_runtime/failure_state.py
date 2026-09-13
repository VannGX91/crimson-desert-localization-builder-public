"""Failure state sidecar management with atomic rewriting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Classified failure record for an unresolved translation identity."""

    marker: str
    key: str
    original_sha256: str
    message: str
    provider: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": "failure",
            "marker": self.marker,
            "key": self.key,
            "original_sha256": self.original_sha256,
            "provider": self.provider,
            "message": self.message[:500],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureRecord:
        return cls(
            marker=data["marker"],
            key=data["key"],
            original_sha256=data.get("original_sha256", ""),
            message=data.get("message", ""),
            provider=data.get("provider", ""),
        )


def _atomic_replace_text(path: Path, text: str) -> None:
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        temp.replace(path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


class FailureStateManager:
    """Manages the atomic rewriting and resolution of translation failure state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load_failures(self) -> dict[tuple[str, str], FailureRecord]:
        """Loads unresolved failures keyed by (marker, key)."""
        if not self.path.is_file():
            return {}

        results: dict[tuple[str, str], FailureRecord] = {}
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if isinstance(data, dict) and data.get("record") == "failure":
                    rec = FailureRecord.from_dict(data)
                    results[(rec.marker, rec.key)] = rec
        except (OSError, json.JSONDecodeError):
            return {}

        return results

    def save_failures(self, failures: Mapping[tuple[str, str], FailureRecord]) -> None:
        """Atomically rewrites the failure sidecar with current unresolved failures.

        If failures mapping is empty, the file is deleted.
        """
        if not failures:
            if self.path.exists():
                try:
                    self.path.unlink()
                except OSError:
                    pass
            return

        lines = [
            json.dumps(rec.to_dict(), ensure_ascii=False, separators=(",", ":"))
            for rec in failures.values()
        ]
        text = "\n".join(lines) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_replace_text(self.path, text)

    def record_failure(self, record: FailureRecord) -> None:
        """Records or updates an unresolved failure record."""
        current = self.load_failures()
        current[(record.marker, record.key)] = record
        self.save_failures(current)

    def record_success(self, marker: str, key: str) -> None:
        """Removes a resolved identity from the failure sidecar."""
        current = self.load_failures()
        if (marker, key) in current:
            del current[(marker, key)]
            self.save_failures(current)
