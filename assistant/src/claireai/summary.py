"""The daily summary (Month 1, Week 4).

Built from `tasks` and `task_log` alone, by ordinary code — no model call. The
audit log is the source of truth for what happened, so the summary must be a
mechanical read of it: a model asked to "summarise the day" can quietly invent a
task, and a report that lies about what the assistant did is worse than no
report. (The evaluation case `summary-three-sections` shows the local model does
follow the three-heading shape; that is a nice-to-have, not the mechanism.)

Sections, as the roadmap specifies:

* **Done**            — tasks that finished successfully, with the tools they ran
* **Failed or skipped** — failed tasks, and actions the owner or a policy refused
* **Still pending**   — anything not in a terminal state when the summary runs
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path

from claireai.dates import parse_when
from claireai.storage.tasks import utc_now

PENDING_STATUSES = frozenset({"queued", "planning", "awaiting_approval", "running", "interrupted"})
DONE_HEADING = "Done"
FAILED_HEADING = "Failed or skipped"
PENDING_HEADING = "Still pending"


class SummaryError(RuntimeError):
    """The day could not be summarised (bad date, unwritable file)."""


@dataclass(frozen=True)
class TaskLine:
    """One task as it appears in the summary.

    ``local_time`` is computed in :func:`build`, not here: timestamps are stored
    in UTC, and printing them raw showed 00:12 for a task the owner ran at 03:12
    (found by reading a real report, not by a test).
    """

    task_id: str
    created_at: str
    local_time: str
    request: str
    status: str
    tools: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class DailySummary:
    day: date
    lines: tuple[TaskLine, ...] = field(default_factory=tuple)

    @property
    def done(self) -> tuple[TaskLine, ...]:
        # A task that succeeded *with a refused action* is not "done": it is
        # reported under Failed or skipped, so every task appears exactly once.
        return tuple(
            line for line in self.lines if line.status == "succeeded" and not line.skipped
        )

    @property
    def failed(self) -> tuple[TaskLine, ...]:
        # A task that succeeded but had an action refused still belongs here: the
        # owner needs to see the step that did not happen, not only the tasks
        # that crashed.
        return tuple(
            line for line in self.lines if line.status == "failed" or line.skipped
        )

    @property
    def pending(self) -> tuple[TaskLine, ...]:
        return tuple(line for line in self.lines if line.status in PENDING_STATUSES)

    @property
    def is_empty(self) -> bool:
        return not self.lines


def day_bounds(day: date, tz: tzinfo | None = None) -> tuple[str, str]:
    """UTC ISO bounds for a local calendar day, half-open ``[start, end)``.

    Local, not UTC: "today" means the owner's day. Stored timestamps are UTC with
    an explicit offset, so string comparison inside SQLite is safe once both
    bounds use the same format.
    """
    tz = tz or datetime.now().astimezone().tzinfo or timezone.utc
    start = datetime.combine(day, time.min, tzinfo=tz).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    return (
        start.isoformat(timespec="seconds"),
        end.isoformat(timespec="seconds"),
    )


def _tool_activity(conn: sqlite3.Connection, task_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Distinct tools that ran, and distinct reasons an action was skipped."""
    rows = conn.execute(
        "SELECT tool, status, detail FROM task_log WHERE task_id = ? AND action = 'tool.call' ORDER BY id",
        (task_id,),
    ).fetchall()
    tools: list[str] = []
    skipped: list[str] = []
    for row in rows:
        name = row["tool"] or "?"
        if row["status"] == "skipped":
            reason = _reason(row["detail"])
            skipped.append(f"{name} ({reason})" if reason else name)
        elif row["status"] == "done" and name not in tools:
            tools.append(name)
    return tuple(tools), tuple(skipped)


def _reason(detail: str | None) -> str:
    if not detail:
        return ""
    try:
        payload = json.loads(detail)
    except ValueError:
        return ""
    reason = payload.get("reason")
    if reason == "approval_denied":
        return "not approved"
    if reason == "permission_denied":
        return "outside the allowed folders"
    if reason == "dry_run":
        return "dry run"
    return str(reason) if reason else ""


def build(conn: sqlite3.Connection, *, day: date | None = None, tz: tzinfo | None = None) -> DailySummary:
    """Read one day's tasks and their tool activity. Never writes."""
    if day is None:
        day = datetime.now(tz).date()
    start, end = day_bounds(day, tz)
    rows = conn.execute(
        "SELECT id, created_at, request, status, error FROM tasks"
        " WHERE created_at >= ? AND created_at < ? ORDER BY created_at",
        (start, end),
    ).fetchall()

    lines: list[TaskLine] = []
    for row in rows:
        tools, skipped = _tool_activity(conn, row["id"])
        created = datetime.fromisoformat(row["created_at"])
        local = created.astimezone(tz) if tz else created.astimezone()
        lines.append(
            TaskLine(
                task_id=row["id"],
                created_at=row["created_at"],
                local_time=local.strftime("%H:%M"),
                request=row["request"],
                status=row["status"],
                tools=tools,
                skipped=skipped,
                error=row["error"],
            )
        )
    return DailySummary(day=day, lines=tuple(lines))


def _clip(text: str, limit: int = 110) -> str:
    single = " ".join(text.split())
    return single if len(single) <= limit else single[: limit - 1] + "…"


def render_markdown(summary: DailySummary) -> str:
    """The report: three fixed sections, always present even when empty."""
    out: list[str] = [f"# ClaireAI — {summary.day.isoformat()}", ""]

    out.append(f"## {DONE_HEADING}")
    if not summary.done:
        out.append("_Nothing completed._")
    for line in summary.done:
        tools = f" — {', '.join(line.tools)}" if line.tools else ""
        out.append(f"- {line.local_time} {_clip(line.request)}{tools}")
    out.append("")

    out.append(f"## {FAILED_HEADING}")
    if not summary.failed:
        out.append("_Nothing failed or was refused._")
    for line in summary.failed:
        if line.status == "failed":
            out.append(f"- {line.local_time} {_clip(line.request)} — failed: {_clip(line.error or 'no reason recorded', 80)}")
        else:
            skipped = "; ".join(line.skipped)
            lead = "completed, but skipped" if line.status == "succeeded" else "skipped"
            out.append(f"- {line.local_time} {_clip(line.request)} — {lead}: {skipped}")
    out.append("")

    out.append(f"## {PENDING_HEADING}")
    if not summary.pending:
        out.append("_Nothing pending._")
    for line in summary.pending:
        out.append(f"- {line.local_time} {_clip(line.request)} — {line.status}")
    out.append("")

    out.append("_Generated from the task log by ClaireAI; no model was involved._")
    out.append("")
    return "\n".join(out)


def summaries_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "summaries"


def markdown_path(data_dir: Path, day: date) -> Path:
    return summaries_dir(data_dir) / f"{day.isoformat()}.md"


def save(conn: sqlite3.Connection, summary: DailySummary, *, data_dir: Path) -> Path:
    """Upsert the day's row and write the markdown file. Returns the file path.

    Re-running a day replaces that day's report in both places: the summary is
    derived data, so recomputing it must not leave two versions behind.
    """
    content = render_markdown(summary)
    day = summary.day.isoformat()
    conn.execute(
        "INSERT INTO summaries (summary_date, created_at, content_md) VALUES (?, ?, ?)"
        " ON CONFLICT (summary_date) DO UPDATE SET created_at = excluded.created_at,"
        " content_md = excluded.content_md",
        (day, utc_now(), content),
    )

    path = markdown_path(data_dir, summary.day)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise SummaryError(f"cannot write the summary to {path}: {exc}") from exc
    return path


def stored(conn: sqlite3.Connection, day: date) -> str | None:
    """The summary already saved for ``day``, if any."""
    row = conn.execute(
        "SELECT content_md FROM summaries WHERE summary_date = ?", (day.isoformat(),)
    ).fetchone()
    return row["content_md"] if row else None


def parse_day(value: str | None) -> date:
    """A date phrase or ``YYYY-MM-DD``, or today. A bad date is an error, not today.

    The phrases ("yesterday", "last friday", "3 Nov 2026") come from
    :mod:`claireai.dates`, so ``--summary yesterday`` works and stays testable.
    """
    if not value:
        return datetime.now().date()
    try:
        return parse_when(value)
    except ValueError as exc:
        raise SummaryError(str(exc)) from exc


__all__ = [
    "DONE_HEADING",
    "FAILED_HEADING",
    "PENDING_HEADING",
    "PENDING_STATUSES",
    "DailySummary",
    "SummaryError",
    "TaskLine",
    "build",
    "day_bounds",
    "markdown_path",
    "parse_day",
    "render_markdown",
    "save",
    "stored",
    "summaries_dir",
]
