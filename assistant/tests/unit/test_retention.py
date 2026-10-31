"""Retention: planned before it is done, and the audit log is never touched."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claireai.audit import AuditLog
from claireai.config import Settings
from claireai.retention import (
    AUDIT_NOTE,
    RetentionError,
    apply,
    describe,
    human_size,
    plan,
)

NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)


def _batch(data_dir: Path, days_old: int, name: str = "file.txt", size: int = 100) -> Path:
    stamp = (NOW - timedelta(days=days_old)).strftime("%Y%m%dT%H%M%SZ")
    batch = data_dir / "trash" / stamp
    batch.mkdir(parents=True, exist_ok=True)
    (batch / name).write_bytes(b"z" * size)
    return batch


def _summary(data_dir: Path, days_old: int) -> Path:
    day = (NOW - timedelta(days=days_old)).date().isoformat()
    path = data_dir / "summaries" / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# ClaireAI\n", encoding="utf-8")
    return path


def _row(conn: sqlite3.Connection, days_old: int) -> str:
    day = (NOW - timedelta(days=days_old)).date().isoformat()
    conn.execute(
        "INSERT INTO summaries (summary_date, created_at, content_md) VALUES (?, ?, ?)",
        (day, NOW.isoformat(timespec="seconds"), "# old"),
    )
    return day


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_old_trash_is_planned_and_new_trash_is_kept(conn: sqlite3.Connection, data_dir: Path) -> None:
    old = _batch(data_dir, 40)
    fresh = _batch(data_dir, 2, name="other.txt")

    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)

    assert [item.target for item in result.removals] == [str(old)]
    assert result.removals[0].kind == "trash_batch"
    assert result.removals[0].size_bytes == 100
    assert result.kept_trash == 1
    assert fresh.exists()


def test_an_unparseable_batch_name_is_refused_not_guessed(conn: sqlite3.Connection, data_dir: Path) -> None:
    weird = data_dir / "trash" / "important-stuff"
    weird.mkdir(parents=True)
    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)
    assert result.removals == ()
    assert any("not a timestamp" in refusal for refusal in result.refusals)
    assert weird.exists()


def test_a_symlinked_batch_is_refused(conn: sqlite3.Connection, data_dir: Path, tmp_path: Path, make_symlink) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete", encoding="utf-8")
    (data_dir / "trash").mkdir(parents=True)
    stamp = (NOW - timedelta(days=400)).strftime("%Y%m%dT%H%M%SZ")
    make_symlink(data_dir / "trash" / stamp, outside)

    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)

    assert result.removals == ()
    assert any("symlink" in refusal for refusal in result.refusals)


def test_old_summaries_are_planned_in_both_places(conn: sqlite3.Connection, data_dir: Path) -> None:
    old_file = _summary(data_dir, 45)
    fresh_file = _summary(data_dir, 1)
    old_day = _row(conn, 45)
    _row(conn, 1)

    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)

    kinds = {(item.kind, item.target) for item in result.removals}
    assert ("summary_file", str(old_file)) in kinds
    assert ("summary_row", old_day) in kinds
    assert result.kept_summaries == 1
    assert fresh_file.exists()


def test_the_audit_log_is_never_planned_for_removal(conn: sqlite3.Connection, data_dir: Path) -> None:
    with conn:
        AuditLog(conn).record(action="tool.call", status="done", tool="list_folder", detail={"a": "b"})

    result = plan(conn=conn, data_dir=data_dir, days=1, now=NOW)

    assert result.removals == ()
    assert result.audit_rows == 1
    assert "append-only" in AUDIT_NOTE
    assert conn.execute("SELECT COUNT(*) AS n FROM task_log").fetchone()["n"] == 1


def test_a_window_below_one_day_is_rejected(conn: sqlite3.Connection, data_dir: Path) -> None:
    with pytest.raises(RetentionError, match="at least 1 day"):
        plan(conn=conn, data_dir=data_dir, days=0, now=NOW)


def test_apply_removes_exactly_what_was_planned_and_logs_it(
    conn: sqlite3.Connection, data_dir: Path
) -> None:
    old = _batch(data_dir, 40)
    old_summary = _summary(data_dir, 45)
    old_day = _row(conn, 45)
    fresh = _batch(data_dir, 2)
    summary_size = old_summary.stat().st_size

    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)
    report = apply(result, conn=conn, data_dir=data_dir)

    assert not old.exists()
    assert not old_summary.exists()
    assert conn.execute("SELECT COUNT(*) AS n FROM summaries WHERE summary_date = ?", (old_day,)).fetchone()["n"] == 0
    assert fresh.exists()
    assert report.removed_count == 3
    assert report.freed_bytes == 100 + summary_size

    entry = AuditLog(conn).recent(1)[0]
    assert entry.action == "retention.prune"
    assert entry.status == "done"


def test_a_dry_run_touches_nothing(conn: sqlite3.Connection, data_dir: Path) -> None:
    old = _batch(data_dir, 40)
    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)

    assert result.removals  # there is something to do
    assert old.exists()  # ...but planning alone removed nothing
    assert conn.execute("SELECT COUNT(*) AS n FROM task_log").fetchone()["n"] == 0


def test_nothing_to_do_is_reported_plainly(conn: sqlite3.Connection, data_dir: Path) -> None:
    result = plan(conn=conn, data_dir=data_dir, days=30, now=NOW)
    assert result.is_empty
    assert "nothing to remove" in describe(result)


def test_the_plan_is_readable_before_it_runs(conn: sqlite3.Connection, data_dir: Path) -> None:
    _batch(data_dir, 40, size=2048)
    text = describe(plan(conn=conn, data_dir=data_dir, days=30, now=NOW))
    assert "would remove 1 item(s)" in text
    assert "40 day(s) old" in text
    assert "trash_batch" in text


def test_human_size_is_not_misleading() -> None:
    assert human_size(0) == "0 B"
    assert human_size(100) == "100 B"
    assert human_size(2048) == "2.0 KiB"
    assert human_size(5 * 1024 * 1024) == "5.0 MiB"
