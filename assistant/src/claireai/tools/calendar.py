"""Workspace calendar tools.

The day argument is a phrase — "today", "tomorrow", "last friday", "2026-11-07" —
resolved by :func:`claireai.dates.parse_when`. That is deliberate: the same words
mean the same day in chat, in the summary and here, and the resolution happens in
code. A date a model inferred is a date nobody can check.

Reading is ``safe``. Adding writes to the local database, so it is
``needs_approval`` like every other change the assistant can make.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from claireai.dates import describe_day, parse_when
from claireai.storage.appointments import AppointmentRepository
from claireai.storage.database import Database
from claireai.tools.base import ToolContext, ToolError, ToolResult


class _Args(BaseModel):
    """Reject unknown keys: a tool must not be handed arguments it ignores."""

    model_config = ConfigDict(extra="forbid")


class ListAppointmentsArgs(_Args):
    day: str = Field(
        default="today",
        description="Which day: 'today', 'tomorrow', 'yesterday', '2026-11-07' or 'last friday'.",
    )


class AddAppointmentArgs(_Args):
    title: str = Field(description="What the appointment is.")
    day: str = Field(
        default="today",
        description="Which day: 'today', 'tomorrow', '2026-11-07' or a weekday name.",
    )
    starts_at: str = Field(default="09:00", description="Start time, 24-hour HH:MM.")
    ends_at: str | None = Field(default=None, description="Optional end time, 24-hour HH:MM.")
    location: str | None = Field(default=None, description="Optional place.")


def _connect(context: ToolContext) -> sqlite3.Connection:
    if context.data_dir is None:
        raise ToolError("the local database is not available in this run")
    path = Path(context.data_dir) / "db" / "claireai.sqlite3"
    if not path.is_file():
        raise ToolError(f"the local database does not exist at {path}")
    try:
        return Database(path).connect()
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        raise ToolError(f"the local database could not be opened: {exc}") from exc


def _resolve_day(text: str) -> date:
    try:
        return parse_when(text)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _agenda_text(rows: list, day: date) -> str:
    if not rows:
        return f"Nothing is scheduled for {describe_day(day)}."
    return "\n".join([f"{describe_day(day)}:"] + [f"- {row.as_line()}" for row in rows])


def list_appointments(context: ToolContext, args: ListAppointmentsArgs) -> ToolResult:
    day = _resolve_day(args.day)
    conn = _connect(context)
    try:
        rows = AppointmentRepository(conn).for_day(day)
    except sqlite3.Error as exc:
        raise ToolError(
            "the appointments table is missing; start ClaireAI once so migrations run"
        ) from exc
    finally:
        conn.close()
    return ToolResult(_agenda_text(rows, day), {"day": day.isoformat(), "count": len(rows)})


def add_appointment(context: ToolContext, args: AddAppointmentArgs) -> ToolResult:
    day = _resolve_day(args.day)
    conn = _connect(context)
    try:
        appointment = AppointmentRepository(conn).add(
            title=args.title,
            day=day,
            starts_at=args.starts_at,
            ends_at=args.ends_at,
            location=args.location,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise ToolError(
            "the appointments table is missing; start ClaireAI once so migrations run"
        ) from exc
    finally:
        conn.close()
    return ToolResult(
        f"Added for {describe_day(day)}: {appointment.as_line()}.",
        {"day": day.isoformat(), "id": appointment.id},
    )


_POLITE_PREFIX = re.compile(r"^(?:hey|hi|hello|ok|okay|claire|claireai|please)[ ,]+")
_AGENDA_QUESTION = re.compile(
    r"^(?:what(?:'s| is) on|what do i have(?: got)?|what have i got|"
    r"(?:show|list|tell) me (?:my )?(?:schedule|agenda|appointments)|"
    r"my (?:schedule|agenda|appointments)|(?:schedule|agenda|appointments))"
    r"(?: (?:for|on))?(?: (?P<when>.+))?$"
)


def answer_agenda(conn: sqlite3.Connection, text: str) -> str | None:
    """Answer "what's on tomorrow" from the table, or ``None`` to let the model reply.

    The model cannot know the calendar, and an invented meeting is worse than no
    answer, so the runtime answers this itself — the same reasoning as dates.
    Only a phrase whose day :func:`parse_when` accepts is claimed: "what's on the
    news" falls through to the model instead of being forced onto today.
    """
    value = " ".join(text.strip().rstrip("?!.").lower().split())
    for _ in range(4):
        stripped = _POLITE_PREFIX.sub("", value)
        if stripped == value:
            break
        value = stripped
    match = _AGENDA_QUESTION.match(value)
    if match is None:
        return None
    when = match.group("when")
    try:
        day = parse_when(when) if when else date.today()
    except ValueError:
        return None
    try:
        rows = AppointmentRepository(conn).for_day(day)
    except sqlite3.Error:
        return None
    return _agenda_text(rows, day)


__all__ = [
    "AddAppointmentArgs",
    "ListAppointmentsArgs",
    "add_appointment",
    "answer_agenda",
    "list_appointments",
]
