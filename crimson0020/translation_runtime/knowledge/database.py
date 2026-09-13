"""Knowledge store database connection, migration, backup, and recovery manager."""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from crimson0020.translation_runtime.errors import TranslationRuntimeError
from crimson0020.translation_runtime.knowledge.schema import (
    INIT_SCHEMA_SQL,
    SCHEMA_VERSION,
)


class KnowledgeDatabaseError(TranslationRuntimeError):
    """Raised when an unrecoverable knowledge database error occurs."""


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Outcome of database opening/recovery."""

    recovered: bool
    backup_path: str | None = None
    message: str = "Database initialized or opened cleanly."


def resolve_default_database_path() -> Path:
    """Resolves default path to translation-knowledge-v1.sqlite3 under local application data."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        base_dir = Path(local_app_data) / "Crimson0020"
    else:
        base_dir = Path.home() / ".crimson0020"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "translation-knowledge-v1.sqlite3"


class KnowledgeDatabase:
    """Manages SQLite storage with WAL, transactions, backup, and corrupt recovery."""

    def __init__(self, database_path: str | Path | None = None, busy_timeout_ms: int = 5000) -> None:
        if database_path is None:
            self.db_path = resolve_default_database_path()
        else:
            self.db_path = Path(database_path).expanduser().resolve()
        self.busy_timeout_ms = busy_timeout_ms
        self.last_recovery_result = self.ensure_database()

    def _open_raw_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(
            str(self.db_path),
            timeout=self.busy_timeout_ms / 1000.0,
            check_same_thread=False,
        )

    def _open_connection(self) -> sqlite3.Connection:
        con = self._open_raw_connection()
        con.execute("PRAGMA journal_mode = WAL")
        con.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def check_integrity(self, con: sqlite3.Connection) -> bool:
        """Executes PRAGMA quick_check to verify database health."""
        try:
            cur = con.cursor()
            cur.execute("PRAGMA quick_check")
            rows = cur.fetchall()
            return len(rows) == 1 and rows[0][0] == "ok"
        except Exception:
            return False

    def create_backup(self, reason: str = "backup") -> Path:
        """Creates a timestamped copy of the database before migration or on corruption."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.db_path.stem}_{reason}_{timestamp}_{time.time_ns()}{self.db_path.suffix}"
        backup_path = self.db_path.parent / backup_name
        if self.db_path.exists():
            shutil.copy2(self.db_path, backup_path)
        return backup_path

    def ensure_database(self) -> RecoveryResult:
        """Ensures database is initialized, migrated, and intact."""
        is_corrupt = False

        if self.db_path.exists():
            con = None
            try:
                con = self._open_raw_connection()
                if not self.check_integrity(con):
                    is_corrupt = True
            except Exception:
                is_corrupt = True
            finally:
                if con is not None:
                    con.close()
                    del con

        if is_corrupt:
            backup_path = self.create_backup(reason="corrupt_recovery")
            # Remove corrupt database and sidecars
            for ext in ("", "-wal", "-shm"):
                p = Path(f"{self.db_path}{ext}")
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            # Initialize fresh
            self._init_schema()
            return RecoveryResult(
                recovered=True,
                backup_path=str(backup_path),
                message=f"Damaged database preserved at {backup_path.name}; fresh store created.",
            )

        self._init_schema()
        return RecoveryResult(recovered=False)

    def _init_schema(self) -> None:
        con = self._open_connection()
        try:
            with con:
                con.executescript(INIT_SCHEMA_SQL)
                cur = con.cursor()
                cur.execute("SELECT schema_version FROM schema_metadata WHERE schema_version = ?", (SCHEMA_VERSION,))
                if not cur.fetchone():
                    cur.execute(
                        "INSERT INTO schema_metadata (schema_version, applied_at, description) VALUES (?, ?, ?)",
                        (SCHEMA_VERSION, time.time(), "Initial translation knowledge schema"),
                    )
        finally:
            con.close()

    @contextlib.contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an active connection."""
        con = self._open_connection()
        try:
            yield con
        finally:
            con.close()

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager wrapping block in an atomic transaction."""
        con = self._open_connection()
        try:
            with con:
                yield con
        finally:
            con.close()
