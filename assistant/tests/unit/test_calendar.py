"""The workspace calendar: local rows, dates resolved by code.

The point of these tests is that "tomorrow" is resolved by
:mod:`claireai.dates` and not by the model, and that adding an appointment is
gated while reading it is not.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from claireai.dates import describe_day
from claireai.permissions import Sandbox
from claireai.routing import route as choose_route
from claireai.storage.appointments import AppointmentRepository
from claireai.storage.database import Database
from claireai.tools import default_registry
from claireai.tools.base import ToolContext, ToolError
from claireai.tools.calendar import (
    AddAppointmentArgs,
    ListAppointmentsArgs,
    add_appointment,
    answer_agenda,
    list_appointments,
)

DB = Path("db") / "claireai.sqlite3"


def _context(tmp_path: Path) -> ToolContext:
    Database(tmp_path / DB).migrate()
    return ToolContext(sandbox=Sandbox(roots=()), data_dir=tmp_path)


def _conn(tmp_path: Path):
    database = Database(tmp_path / DB)
    database.migrate()
    return database.connect()


def test_an_empty_day_says_so(tmp_path: Path) -> None:
    result = list_appointments(_context(tmp_path), ListAppointmentsArgs(day="2026-11-07"))
    assert result.text == "Nothing is scheduled for Saturday, 7 November 2026."
    assert result.detail == {"day": "2026-11-07", "count": 0}


def test_appointments_are_listed_in_time_order_with_their_place(tmp_path: Path) -> None:
    context = _context(tmp_path)
    add_appointment(
        context,
        AddAppointmentArgs(title="Standup", day="2026-11-07", starts_at="09:30", location="Kigali"),
    )
    add_appointment(
        context,
        AddAppointmentArgs(title="Dentist", day="2026-11-07", starts_at="14:00", ends_at="14:45"),
    )

    result = list_appointments(context, ListAppointmentsArgs(day="2026-11-07"))

    lines = result.text.splitlines()
    assert lines[0] == "Saturday, 7 November 2026:"
    assert "- 09:30 Standup at Kigali" in lines
    assert "- 14:00–14:45 Dentist" in lines
    assert result.text.index("Standup") < result.text.index("Dentist")
    assert result.detail["count"] == 2


def test_a_phrase_is_resolved_through_the_date_module(tmp_path: Path) -> None:
    context = _context(tmp_path)
    add_appointment(context, AddAppointmentArgs(title="Review", day="tomorrow"))

    result = list_appointments(context, ListAppointmentsArgs(day="tomorrow"))

    assert "Review" in result.text
    assert result.detail["day"] == (date.today() + timedelta(days=1)).isoformat()


def test_an_unreadable_day_is_an_error_and_not_today(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not a date I understand"):
        list_appointments(_context(tmp_path), ListAppointmentsArgs(day="someday"))


def test_a_bad_clock_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="24-hour"):
        add_appointment(
            _context(tmp_path),
            AddAppointmentArgs(title="Plan", day="today", starts_at="9am"),
        )


def test_the_week_view_covers_only_its_own_days(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        repo = AppointmentRepository(conn)
        repo.add(title="Today", day=date(2026, 11, 6))
        repo.add(title="Next week", day=date(2026, 11, 20))
        titles = [appointment.title for appointment in repo.upcoming(date(2026, 11, 6), days=7)]
    finally:
        conn.close()
    assert titles == ["Today"]


def test_reading_is_free_and_writing_needs_approval() -> None:
    registry = default_registry()
    assert registry.get("list_appointments").risk == "safe"
    assert registry.get("add_appointment").risk == "needs_approval"


def test_what_is_on_tomorrow_routes_to_the_calendar() -> None:
    route = choose_route("what's on tomorrow", default_registry())
    assert route.intent == "calendar"
    assert "list_appointments" in route.tools


def test_an_agenda_question_is_answered_from_the_table(tmp_path: Path) -> None:
    add_appointment(
        _context(tmp_path),
        AddAppointmentArgs(title="Dentist", day="tomorrow", starts_at="14:00", location="Kigali"),
    )

    conn = _conn(tmp_path)
    try:
        answer = answer_agenda(conn, "hey claire, what's on tomorrow?")
        empty = answer_agenda(conn, "what's on today")
    finally:
        conn.close()

    assert answer is not None
    assert "- 14:00 Dentist at Kigali" in answer
    assert empty == f"Nothing is scheduled for {describe_day(date.today())}."


@pytest.mark.parametrize(
    "text",
    ["hello", "what's on the news", "list my Downloads folder", "what is the biggest file"],
)
def test_other_requests_are_left_to_the_model(tmp_path: Path, text: str) -> None:
    conn = _conn(tmp_path)
    try:
        assert answer_agenda(conn, text) is None
    finally:
        conn.close()
