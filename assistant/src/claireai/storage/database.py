"""SQLite connection handling and versioned migrations.

Migrations are plain ``.sql`` files named ``NNNN_name.sql``. Each applied file
is recorded with its checksum so that editing an already-applied migration is
detected instead of silently producing divergent databases.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterator

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


class MigrationError(RuntimeError):
    """Raised when migrations cannot be applied safely."""


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block in one explicit transaction.

    Connections are opened with ``isolation_level=None`` so that DDL inside a
    migration is committed atomically; callers therefore opt in explicitly.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def split_statements(script: str) -> list[str]:
    """Split a migration script into statements the driver can execute one by one.

    ``sqlite3.complete_statement`` is used instead of splitting on ``;`` because
    trigger bodies contain semicolons.
    """
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = _drop_comment_only(buffer)
            if statement:
                statements.append(statement)
            buffer = ""
    trailing = _drop_comment_only(buffer)
    if trailing:
        raise MigrationError(f"unterminated SQL statement in migration: {buffer.strip()[:120]!r}")
    return statements


def _drop_comment_only(fragment: str) -> str:
    stripped_lines = [
        line for line in fragment.splitlines() if line.strip() and not line.strip().startswith("--")
    ]
    statement = "\n".join(stripped_lines).strip()
    return "" if statement in {"", ";"} else statement


class Database:
    """Owns the SQLite file and knows how to apply migrations to it."""

    def __init__(self, path: Path | str, migrations_dir: Path | None = None) -> None:
        self.path = Path(path)
        self.migrations_dir = Path(migrations_dir) if migrations_dir else MIGRATIONS_DIR

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def migrate(self) -> list[str]:
        """Apply pending migrations and return the versions applied in this call."""
        conn = self.connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version TEXT PRIMARY KEY,"
                " applied_at TEXT NOT NULL,"
                " checksum TEXT NOT NULL"
                ")"
            )
            applied = {
                row["version"]: row["checksum"]
                for row in conn.execute("SELECT version, checksum FROM schema_migrations")
            }
            newly_applied: list[str] = []
            for path in sorted(self.migrations_dir.glob("*.sql")):
                version = path.stem
                script = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(script.encode("utf-8")).hexdigest()
                if version in applied:
                    if applied[version] != checksum:
                        raise MigrationError(
                            f"migration {version} changed after it was applied; "
                            "add a new migration instead of editing history"
                        )
                    continue
                self._apply(conn, version, script, checksum)
                newly_applied.append(version)
            return newly_applied
        finally:
            conn.close()

    @staticmethod
    def _apply(conn: sqlite3.Connection, version: str, script: str, checksum: str) -> None:
        statements = split_statements(script)
        with transaction(conn):
            for index, statement in enumerate(statements, start=1):
                try:
                    conn.execute(statement)
                except sqlite3.Error as exc:
                    raise MigrationError(f"migration {version} failed at statement {index}: {exc}") from exc
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at, checksum) VALUES (?, datetime('now'), ?)",
                (version, checksum),
            )
