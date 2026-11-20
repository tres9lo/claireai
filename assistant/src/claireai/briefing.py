"""Daily briefing: news, calendar and pending work in one local report.

This is deliberately assembled by code. The news comes from a public RSS feed,
the calendar from SQLite, and pending work from the task table. No model is asked
to remember or invent any of it.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Callable, Iterable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from claireai.news import NewsError, brief
from claireai.storage.appointments import AppointmentRepository
from claireai.summary import PENDING_STATUSES

MAX_COUNTRIES = 5


def _pending(conn: sqlite3.Connection, limit: int = 8) -> list[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in PENDING_STATUSES)
    return conn.execute(
        "SELECT request, status, created_at FROM tasks"
        f" WHERE status IN ({placeholders})"
        " ORDER BY created_at DESC LIMIT ?",
        (*sorted(PENDING_STATUSES), limit),
    ).fetchall()


def build_briefing(
    conn: sqlite3.Connection,
    *,
    countries: Iterable[str] = (),
    day: date | None = None,
    headlines: int = 3,
    news_enabled: bool = True,
) -> str:
    """Render a source-backed briefing without calling the model."""
    day = day or datetime.now().date()
    names = [str(item).strip() for item in countries if str(item).strip()][:MAX_COUNTRIES]
    lines = [f"# ClaireAI briefing — {day.isoformat()}", "", "## News"]
    if not names:
        lines.append("_No countries selected._")
    elif not news_enabled:
        lines.append("_News is off because browser access is disabled._")
    else:
        try:
            lines.append(brief(names, limit=max(1, min(10, headlines))))
        except NewsError as exc:
            lines.append(f"_News unavailable: {exc}_")

    lines += ["", "## Calendar"]
    appointments = AppointmentRepository(conn).for_day(day)
    if not appointments:
        lines.append("_Nothing scheduled._")
    else:
        lines.extend(f"- {item.as_line()}" for item in appointments)

    lines += ["", "## Needs attention"]
    pending = _pending(conn)
    if not pending:
        lines.append("_No pending tasks._")
    else:
        for row in pending:
            lines.append(f"- {row['request']} — {row['status']}")

    lines += [
        "",
        "_News items are links to their publishers; calendar and task items are "
        "local records. No model was involved._",
        "",
    ]
    return "\n".join(lines)


def build_scheduler(
    conn: sqlite3.Connection,
    *,
    countries: Iterable[str],
    when: str,
    headlines: int = 3,
    news_enabled: bool | Callable[[], bool] = True,
) -> BlockingScheduler:
    """A foreground daily scheduler. The process must stay running."""
    try:
        hour_text, minute_text = when.strip().split(":")
        hour, minute = int(hour_text), int(minute_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{when!r} is not a time in HH:MM form") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{when!r} is not a valid time of day")

    scheduler = BlockingScheduler()

    def job() -> None:
        try:
            # The browser switch can change while the scheduler is running, so
            # re-read it for every briefing instead of trusting the value at
            # start-up.
            enabled = news_enabled() if callable(news_enabled) else news_enabled
            print(
                build_briefing(
                    conn,
                    countries=countries,
                    headlines=headlines,
                    news_enabled=enabled,
                ),
                flush=True,
            )
        except (sqlite3.Error, OSError) as exc:
            print(f"[briefing] failed: {exc}", flush=True)

    scheduler.add_job(
        job,
        CronTrigger(hour=hour, minute=minute),
        id="daily-briefing",
        replace_existing=True,
    )
    return scheduler


__all__ = ["build_briefing", "build_scheduler"]
