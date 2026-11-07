"""Appointments, stored locally and keyed by local calendar day.

The day is a local ``YYYY-MM-DD`` string rather than a UTC timestamp, because the
question the owner asks is "what is on tomorrow", and tomorrow is a local day.
Times are ``HH:MM`` on that day. Sorting is therefore plain text sorting, which is
correct for zero-padded 24-hour times and needs no timezone maths.

Nothing here is model-driven: the caller resolves a phrase like "tomorrow" with
:mod:`claireai.dates` and passes a date.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date

from claireai.storage.tasks import utc_now

_CLOCK = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _clock(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not _CLOCK.match(text):
        raise ValueError(f"{field} must be a 24-hour time like 09:30, got {value!r}")
    return text


@dataclass(frozen=True)
class Appointment:
    id: str
    day: str
    starts_at: str
    ends_at: str | None
    title: str
    location: str | None

    def as_line(self) -> str:
        span = f"{self.starts_at}–{self.ends_at}" if self.ends_at else self.starts_at
        where = f" at {self.location}" if self.location else ""
        return f"{span} {self.title}{where}"


class AppointmentRepository:
    """Create and read appointments. One row per appointment."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(
        self,
        *,
        title: str,
        day: date,
        starts_at: str = "09:00",
        ends_at: str | None = None,
        location: str | None = None,
    ) -> Appointment:
        name = " ".join(str(title or "").split())
        if not name:
            raise ValueError("an appointment needs a title")
        start = _clock(starts_at, field="starts_at")
        end = _clock(ends_at, field="ends_at") if ends_at else None
        if end and end <= start:
            raise ValueError("ends_at must be after starts_at")
        appointment = Appointment(
            id=uuid.uuid4().hex,
            day=day.isoformat(),
            starts_at=start,
            ends_at=end,
            title=name,
            location=" ".join(str(location).split()) if location else None,
        )
        self.conn.execute(
            "INSERT INTO appointments (id, created_at, day, starts_at, ends_at, title, location)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                appointment.id,
                utc_now(),
                appointment.day,
                appointment.starts_at,
                appointment.ends_at,
                appointment.title,
                appointment.location,
            ),
        )
        return appointment

    def for_day(self, day: date) -> tuple[Appointment, ...]:
        rows = self.conn.execute(
            "SELECT id, day, starts_at, ends_at, title, location"
            " FROM appointments WHERE day = ? ORDER BY starts_at, title",
            (day.isoformat(),),
        ).fetchall()
        return tuple(
            Appointment(
                id=row["id"],
                day=row["day"],
                starts_at=row["starts_at"],
                ends_at=row["ends_at"],
                title=row["title"],
                location=row["location"],
            )
            for row in rows
        )

    def upcoming(self, start: date, *, days: int = 7) -> tuple[Appointment, ...]:
        """Appointments in the next ``days`` local days, inclusive of ``start``."""
        rows = self.conn.execute(
            "SELECT id, day, starts_at, ends_at, title, location"
            " FROM appointments WHERE day BETWEEN ? AND ?"
            " ORDER BY day, starts_at, title",
            (start.isoformat(), date.fromordinal(start.toordinal() + max(0, days - 1)).isoformat()),
        ).fetchall()
        return tuple(
            Appointment(
                id=row["id"],
                day=row["day"],
                starts_at=row["starts_at"],
                ends_at=row["ends_at"],
                title=row["title"],
                location=row["location"],
            )
            for row in rows
        )


__all__ = ["Appointment", "AppointmentRepository"]
