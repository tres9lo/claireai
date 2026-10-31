"""The read-only today page: escaped, read-only, and honest about missing data."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from claireai.audit import AuditLog
from claireai.config import Settings
from claireai.storage import Database, TaskRepository, transaction
from claireai.summary import build, save
from claireai.today import MAX_REQUESTS_SHOWN, open_read_only, render_today, render_unavailable


def _seed(conn: sqlite3.Connection, request: str, *, tool: str = "list_folder", status: str = "done"):
    tasks = TaskRepository(conn)
    with transaction(conn):
        task = tasks.create(request=request, kind="agent")
        tasks.update_status(task.id, "running")
        tasks.update_status(task.id, "succeeded")
    with transaction(conn):
        AuditLog(conn).record(task_id=task.id, tool=tool, action="tool.call", status=status, detail={})
    return task


def test_the_page_shows_the_summary_and_the_tasks(conn: sqlite3.Connection, settings: Settings) -> None:
    _seed(conn, "List the biggest files in my Downloads folder", tool="find_largest_files")
    with transaction(conn):
        save(conn, build(conn), data_dir=settings.resolved_data_dir)

    page = render_today(conn)

    assert page.startswith("<!doctype html>")
    assert "Daily summary" in page
    assert "## Done" in page  # the stored markdown, verbatim inside <pre>
    assert "List the biggest files" in page
    assert "find_largest_files" in page
    assert "1 done" in page


def test_there_is_no_script_or_form_anywhere(conn: sqlite3.Connection) -> None:
    _seed(conn, "say hello")
    page = render_today(conn)
    assert "<script" not in page.lower()
    assert "<form" not in page.lower()
    assert "<button" not in page.lower()


def test_stored_text_cannot_inject_html(conn: sqlite3.Connection, settings: Settings) -> None:
    _seed(conn, 'create a file called "<script>alert(1)</script>.txt"')
    with transaction(conn):
        save(conn, build(conn), data_dir=settings.resolved_data_dir)

    page = render_today(conn)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_a_day_without_a_summary_says_how_to_make_one(conn: sqlite3.Connection) -> None:
    page = render_today(conn, day=date(2001, 1, 1))
    assert "No summary stored for this day" in page
    assert "claireai --summary" in page
    assert "No tasks recorded" in page


def test_long_lists_are_truncated_not_dumped(conn: sqlite3.Connection) -> None:
    for index in range(MAX_REQUESTS_SHOWN + 3):
        _seed(conn, f"task number {index}")
    page = render_today(conn)
    assert f"Showing {MAX_REQUESTS_SHOWN} of {MAX_REQUESTS_SHOWN + 3} tasks." in page


def test_the_unavailable_page_explains_itself() -> None:
    page = render_unavailable("No database at /tmp/x.sqlite3 yet.")
    assert "No task log to show" in page
    assert "/tmp/x.sqlite3" in page
    assert "<script" not in page.lower()


def test_the_connection_really_is_read_only(database: Database) -> None:
    conn = open_read_only(database.path)
    assert conn is not None
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM tasks")
    finally:
        conn.close()


def test_a_missing_file_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope.sqlite3"
    assert open_read_only(missing) is None
    assert not missing.exists()


def test_the_page_uses_local_time(conn: sqlite3.Connection) -> None:
    """Stored times are UTC; the page must show the owner's clock.

    The first version of this test asserted "00:12" was absent, which passed only
    because this machine is UTC+2 — CI (UTC) failed it, correctly.
    """
    task = _seed(conn, "Say OK.")
    with transaction(conn):
        conn.execute(
            "UPDATE tasks SET created_at = ? WHERE id = ?",
            ("2026-10-31T00:12:00+00:00", task.id),
        )

    page = render_today(conn, day=date(2026, 10, 31), tz=timezone(timedelta(hours=3)))

    assert "03:12" in page
    assert "00:12" not in page
