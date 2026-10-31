"""The evening summary job (Month 1, Week 4).

APScheduler is already a declared dependency, so this uses it rather than a
hand-rolled loop: a cron trigger is the thing the roadmap asked for, and it
handles the "what time is it" arithmetic honestly (local time, DST included).

The scheduler owns no state and no model. It opens a connection, asks
:mod:`claireai.summary` for the day, saves it, and records one audit row so the
log shows the report was produced. A failure is logged and swallowed: a bad
evening must not kill the process that is supposed to try again tomorrow.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from claireai.audit import AuditLog
from claireai.config import Settings
from claireai.storage import Database, transaction
from claireai.summary import DailySummary, SummaryError, build, render_markdown, save

DEFAULT_SUMMARY_TIME = "18:00"
JOB_ID = "daily-summary"


def run_once(
    *,
    conn: sqlite3.Connection,
    settings: Settings,
    day: date | None = None,
    write_file: bool = True,
) -> tuple[DailySummary, Path | None]:
    """Build (and by default save) one day's summary, recording an audit row."""
    audit = AuditLog(conn)
    try:
        summary = build(conn, day=day)
    except (sqlite3.Error, SummaryError) as exc:
        with transaction(conn):
            audit.record(
                action="summary.build",
                status="failed",
                target=day.isoformat() if day else "today",
                detail={"error": str(exc)},
            )
        raise

    path: Path | None = None
    with transaction(conn):
        if write_file:
            path = save(conn, summary, data_dir=settings.resolved_data_dir)
        audit.record(
            action="summary.build",
            status="done",
            target=summary.day.isoformat(),
            detail={
                "tasks": len(summary.lines),
                "done": len(summary.done),
                "failed": len(summary.failed),
                "pending": len(summary.pending),
                "file": str(path) if path else None,
            },
        )
    return summary, path


def build_scheduler(
    *,
    settings: Settings,
    when: str = DEFAULT_SUMMARY_TIME,
    on_run: Callable[[DailySummary, Path | None], None] | None = None,
) -> BlockingScheduler:
    """A blocking scheduler with one cron job at ``when`` (local ``HH:MM``).

    ``on_run`` is the CLI's printing hook; the scheduler itself prints nothing.
    """
    hour, minute = _parse_when(when)
    scheduler = BlockingScheduler()

    def job() -> None:
        database = Database(settings.database_path)
        database.migrate()
        conn = database.connect()
        try:
            summary, path = run_once(conn=conn, settings=settings)
        except (sqlite3.Error, SummaryError) as exc:
            print(f"[summary] failed: {exc}", flush=True)
            return
        finally:
            conn.close()
        if on_run:
            on_run(summary, path)

    scheduler.add_job(job, CronTrigger(hour=hour, minute=minute), id=JOB_ID, replace_existing=True)
    return scheduler


def _parse_when(when: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = when.strip().split(":")
        hour, minute = int(hour_text), int(minute_text)
    except (ValueError, AttributeError) as exc:
        raise SummaryError(f"{when!r} is not a time in HH:MM form") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise SummaryError(f"{when!r} is not a valid time of day")
    return hour, minute


def next_run_label(scheduler: BlockingScheduler) -> str:
    """When the job will next fire, workable *before* ``start()``.

    APScheduler only fills ``job.next_run_time`` once the scheduler runs, so a
    pre-start call asks the cron trigger instead — otherwise the CLI printed
    "(not scheduled)" for a job that was scheduled correctly.
    """
    job = scheduler.get_job(JOB_ID)
    if job is None:
        return "(not scheduled)"
    when = getattr(job, "next_run_time", None) or job.trigger.get_next_fire_time(
        None, datetime.now().astimezone()
    )
    return when.isoformat(timespec="seconds") if when else "(not scheduled)"


__all__ = [
    "DEFAULT_SUMMARY_TIME",
    "JOB_ID",
    "build_scheduler",
    "next_run_label",
    "render_markdown",
    "run_once",
]
