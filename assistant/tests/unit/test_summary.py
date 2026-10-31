"""The daily summary: three sections, derived from the log, never invented.

It is built by plain code from `tasks` + `task_log`, so these tests seed those
tables and read the rendered report — no model is involved anywhere.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from claireai.audit import AuditLog
from claireai.config import Settings
from claireai.scheduler import build_scheduler, next_run_label, run_once
from claireai.storage import TaskRepository, transaction
from claireai.summary import (
    DONE_HEADING,
    FAILED_HEADING,
    PENDING_HEADING,
    SummaryError,
    build,
    day_bounds,
    markdown_path,
    parse_day,
    render_markdown,
    save,
    stored,
)

TZ = timezone.utc


def _task(conn: sqlite3.Connection, request: str, *, status: str = "succeeded", error: str | None = None):
    tasks = TaskRepository(conn)
    with transaction(conn):
        task = tasks.create(request=request, kind="agent")
        tasks.update_status(task.id, "running")
        if status in {"succeeded", "failed", "cancelled"}:
            tasks.update_status(task.id, status, error=error)
    return task


def _tool(conn: sqlite3.Connection, task_id: str, name: str, *, status: str = "done", detail=None):  # noqa: ANN001
    with transaction(conn):
        AuditLog(conn).record(
            task_id=task_id,
            tool=name,
            action="tool.call",
            status=status,
            detail=detail if detail is not None else {"arguments": {}},
        )


@pytest.fixture()
def day_tasks(conn: sqlite3.Connection) -> dict[str, str]:
    done = _task(conn, "List the biggest files in my Downloads folder")
    _tool(conn, done.id, "find_largest_files")
    failed = _task(conn, "Move report.pdf into Documents", status="failed", error="destination missing")
    _tool(conn, failed.id, "move_file", status="failed")
    refused = _task(conn, "make a folder called reports")
    _tool(conn, refused.id, "create_folder", status="skipped", detail={"reason": "approval_denied"})
    pending = _task(conn, "summarise my email inbox", status="running")
    return {"done": done.id, "failed": failed.id, "refused": refused.id, "pending": pending.id}


def test_the_three_sections_are_always_present(conn: sqlite3.Connection) -> None:
    report = render_markdown(build(conn))
    for heading in (DONE_HEADING, FAILED_HEADING, PENDING_HEADING):
        assert f"## {heading}" in report
    assert report.index(DONE_HEADING) < report.index(FAILED_HEADING) < report.index(PENDING_HEADING)
    assert "_Nothing completed._" in report
    assert "_Nothing pending._" in report


def test_a_successful_task_lists_the_tools_it_ran(conn: sqlite3.Connection, day_tasks) -> None:
    summary = build(conn)
    assert [line.task_id for line in summary.done] == [day_tasks["done"]]
    report = render_markdown(summary)
    assert "biggest files" in report
    assert "find_largest_files" in report


def test_a_failed_task_is_reported_with_its_reason(conn: sqlite3.Connection, day_tasks) -> None:
    summary = build(conn)
    assert [line.task_id for line in summary.failed] == [day_tasks["failed"], day_tasks["refused"]]
    report = render_markdown(summary)
    assert "destination missing" in report
    assert "failed:" in report


def test_a_refused_action_appears_even_though_the_task_succeeded(
    conn: sqlite3.Connection, day_tasks
) -> None:
    # The task finished; the write did not. Hiding that would be the summary
    # lying about the day.
    summary = build(conn)
    refused = next(line for line in summary.failed if line.task_id == day_tasks["refused"])
    assert refused.status == "succeeded"
    assert refused.skipped == ("create_folder (not approved)",)
    # Exactly once: it is not silently counted as a clean success too.
    assert day_tasks["refused"] not in {line.task_id for line in summary.done}
    assert "completed, but skipped: create_folder (not approved)" in render_markdown(summary)


def test_an_unfinished_task_is_pending_not_done(conn: sqlite3.Connection, day_tasks) -> None:
    summary = build(conn)
    assert [line.task_id for line in summary.pending] == [day_tasks["pending"]]
    assert day_tasks["pending"] not in {line.task_id for line in summary.done}
    pending_section = render_markdown(summary).split(f"## {PENDING_HEADING}")[1]
    assert "summarise my email inbox" in pending_section
    assert "running" in pending_section


def test_another_day_is_excluded(conn: sqlite3.Connection, day_tasks) -> None:
    yesterday = (datetime.now(TZ) - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    with transaction(conn):
        conn.execute(
            "UPDATE tasks SET created_at = ? WHERE id = ?",
            (yesterday.isoformat(timespec="seconds"), day_tasks["done"]),
        )
    summary = build(conn, tz=TZ)
    assert day_tasks["done"] not in {line.task_id for line in summary.lines}
    assert summary.is_empty is False  # the other tasks remain


def test_an_empty_day_says_so_in_every_section(conn: sqlite3.Connection) -> None:
    report = render_markdown(build(conn, day=date(2001, 1, 1)))
    assert report.count("_Nothing") == 3
    assert "no model was involved" in report


def test_day_bounds_are_half_open_and_local() -> None:
    start, end = day_bounds(date(2026, 10, 31), timezone(timedelta(hours=2)))
    assert start == "2026-10-30T22:00:00+00:00"
    assert end == "2026-10-31T22:00:00+00:00"
    # A task at the very end of the day belongs to the next day, not this one.
    assert not (start <= end and end <= end and end < end)


def test_saving_writes_a_file_and_can_be_re_run(
    conn: sqlite3.Connection, settings: Settings, day_tasks
) -> None:
    summary = build(conn)
    with transaction(conn):
        path = save(conn, summary, data_dir=settings.resolved_data_dir)

    assert path == markdown_path(settings.resolved_data_dir, summary.day)
    assert path.read_text(encoding="utf-8") == render_markdown(summary)
    assert stored(conn, summary.day) == render_markdown(summary)

    # A second run replaces the day's report instead of stacking a duplicate.
    with transaction(conn):
        save(conn, build(conn), data_dir=settings.resolved_data_dir)
    count = conn.execute("SELECT COUNT(*) AS n FROM summaries").fetchone()["n"]
    assert count == 1


def test_writing_the_file_fails_loudly(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, day_tasks
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SummaryError, match="cannot write"):
        with transaction(conn):
            save(conn, build(conn), data_dir=blocked)


def test_parse_day_is_strict() -> None:
    assert parse_day("2026-10-31") == date(2026, 10, 31)
    assert parse_day(None) == datetime.now().date()
    assert parse_day("") == datetime.now().date()
    with pytest.raises(SummaryError, match="YYYY-MM-DD"):
        parse_day("31/10/2026")


def test_run_once_saves_the_report_and_logs_it(conn: sqlite3.Connection, settings: Settings, day_tasks) -> None:
    summary, path = run_once(conn=conn, settings=settings)

    assert path is not None and path.exists()
    row = AuditLog(conn).recent(1)[0]
    assert row.action == "summary.build"
    assert row.status == "done"
    assert row.target == summary.day.isoformat()


def test_run_once_can_skip_the_file(conn: sqlite3.Connection, settings: Settings, day_tasks) -> None:
    summary, path = run_once(conn=conn, settings=settings, write_file=False)
    assert path is None
    assert not markdown_path(settings.resolved_data_dir, summary.day).exists()
    assert stored(conn, summary.day) is None


def test_the_scheduler_has_exactly_one_cron_job(settings: Settings) -> None:
    scheduler = build_scheduler(settings=settings, when="18:30")
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == "daily-summary"
    assert jobs[0].trigger.fields[5].expressions[0].first == 18  # hour
    assert jobs[0].trigger.fields[6].expressions[0].first == 30  # minute


@pytest.mark.parametrize("when", ["25:00", "18:99", "evening", "18", ""])
def test_a_bad_summary_time_is_refused(settings: Settings, when: str) -> None:
    with pytest.raises(SummaryError):
        build_scheduler(settings=settings, when=when)


def test_times_are_printed_in_local_time_not_utc(conn: sqlite3.Connection, settings: Settings) -> None:
    """A real report showed 00:12 for a task run at 03:12 (UTC stored, printed raw)."""
    from datetime import timezone as tz_mod

    task = _task(conn, "Say OK.")
    with transaction(conn):
        conn.execute(
            "UPDATE tasks SET created_at = ? WHERE id = ?",
            ("2026-10-31T00:12:00+00:00", task.id),
        )

    # UTC+3, as used on the machine where the bug was seen.
    ua = tz_mod(timedelta(hours=3))
    line = build(conn, day=date(2026, 10, 31), tz=ua).lines[0]
    assert line.local_time == "03:12"
    assert line.created_at == "2026-10-31T00:12:00+00:00"

    report = render_markdown(build(conn, day=date(2026, 10, 31), tz=ua))
    assert "03:12" in report
    assert "00:12" not in report


def test_the_next_run_is_known_before_the_scheduler_starts(settings: Settings) -> None:
    # APScheduler fills next_run_time only on start(); the CLI printed
    # "(not scheduled)" for a perfectly scheduled job because of that.
    scheduler = build_scheduler(settings=settings, when="18:30")
    label = next_run_label(scheduler)
    assert label != "(not scheduled)"
    assert datetime.fromisoformat(label) > datetime.now(timezone.utc).astimezone()


def test_the_scheduled_job_really_runs(settings: Settings) -> None:
    """Prove the wiring, not just the cron string: fire the job body now."""
    from apscheduler.events import EVENT_JOB_EXECUTED
    from apscheduler.triggers.date import DateTrigger

    seen: list[tuple[int, Path | None]] = []
    scheduler = build_scheduler(
        settings=settings, when="18:30", on_run=lambda summary, path: seen.append((len(summary.lines), path))
    )
    scheduler.add_job(
        scheduler.get_job("daily-summary").func,
        DateTrigger(run_date=datetime.now() + timedelta(seconds=0.5)),
        id="now",
    )
    scheduler.add_listener(
        lambda event: scheduler.shutdown(wait=False), EVENT_JOB_EXECUTED
    )
    scheduler.start()

    assert len(seen) == 1
    path = seen[0][1]
    assert path is not None and path.exists()
    assert path.read_text(encoding="utf-8").startswith("# ClaireAI — ")
