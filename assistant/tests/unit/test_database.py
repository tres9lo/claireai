from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from claireai.storage import Database, MigrationError
from claireai.storage.database import MIGRATIONS_DIR, split_statements

EXPECTED_TABLES = {
    "tasks", "task_log", "summaries", "notes", "schema_migrations",
    "conversations", "conversation_messages", "user_directives", "user_preferences",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def test_migration_creates_expected_tables(database: Database) -> None:
    conn = database.connect()
    try:
        assert EXPECTED_TABLES <= _table_names(conn)
        applied = conn.execute("SELECT version FROM schema_migrations").fetchall()
        assert [row["version"] for row in applied] == ["0001_initial", "0002_conversations_directives_preferences"]
    finally:
        conn.close()


def test_migrate_is_idempotent(database: Database) -> None:
    assert database.migrate() == []


def test_connection_enables_foreign_keys_and_busy_timeout(database: Database) -> None:
    conn = database.connect()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_foreign_key_violation_is_rejected(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO task_log (created_at, device, task_id, actor, action, status)"
            " VALUES ('2026-10-29T10:00:00+00:00', 'desktop', 'missing-task', 'user', 'test', 'done')"
        )


def test_audit_rows_cannot_be_updated_or_deleted(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO task_log (created_at, device, actor, action, status)"
        " VALUES ('2026-10-29T10:00:00+00:00', 'desktop', 'user', 'test.action', 'done')"
    )
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE task_log SET status = 'failed'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM task_log")


def test_editing_an_applied_migration_is_detected(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations)
    db = Database(tmp_path / "db.sqlite3", migrations_dir=migrations)
    db.migrate()

    target = next(migrations.glob("0001_initial.sql"))
    target.write_text(target.read_text(encoding="utf-8") + "\n-- edited later\n", encoding="utf-8")

    with pytest.raises(MigrationError, match="changed after it was applied"):
        db.migrate()


def test_split_statements_keeps_trigger_bodies_intact() -> None:
    script = (MIGRATIONS_DIR / "0001_initial.sql").read_text(encoding="utf-8")
    statements = split_statements(script)
    triggers = [statement for statement in statements if statement.startswith("CREATE TRIGGER")]
    assert len(statements) == 9
    assert len(triggers) == 2
    assert all("BEGIN" in trigger and trigger.rstrip().endswith("END;") for trigger in triggers)


def test_unterminated_statement_is_rejected() -> None:
    with pytest.raises(MigrationError, match="unterminated"):
        split_statements("CREATE TABLE broken (id TEXT)\n")
