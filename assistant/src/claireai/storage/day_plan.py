"""Owner-assigned tasks for one local day.

A day-plan item is a request plus a local start time. It is deliberately separate
from ``appointments``: an appointment is a calendar fact, while a plan item is
work the assistant may run. It is also separate from ``tasks`` because a plan can
exist before any task starts, and a task can be retried without recreating the
plan entry.

The repository enforces a small state machine. In particular, an
``awaiting_approval`` item is never picked up by the runner again; only an
explicit owner action may move it back to ``running``.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

from claireai.dates import describe_day
from claireai.storage.tasks import utc_now

_CLOCK = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

PLAN_STATUSES = frozenset(
    {
        "planned",
        "running",
        "awaiting_approval",
        "succeeded",
        "failed",
        "cancelled",
        "interrupted",
    }
)
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "planned": frozenset({"running", "cancelled", "failed"}),
    "running": frozenset(
        {"awaiting_approval", "succeeded", "failed", "cancelled", "interrupted"}
    ),
    # The runner must not silently retry a paused write. A human may resume it.
    "awaiting_approval": frozenset({"running", "cancelled", "failed"}),
    "interrupted": frozenset({"running", "cancelled", "failed"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def _clock(value: str, *, field: str = "starts_at") -> str:
    text = str(value or "").strip()
    if not _CLOCK.match(text):
        raise ValueError(f"{field} must be a 24-hour time like 09:30, got {value!r}")
    return text


def _clock_from(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    return _clock(value)


@dataclass(frozen=True)
class DayPlanItem:
    id: str
    day: str
    starts_at: str
    request: str
    status: str
    task_id: str | None = None
    approval_ref: str | None = None
    error: str | None = None

    def as_line(self) -> str:
        suffix = f" [{self.status}]" if self.status != "planned" else ""
        return f"{self.starts_at} {self.request}{suffix}"


def _row_to_item(row: sqlite3.Row) -> DayPlanItem:
    return DayPlanItem(
        id=row["id"],
        day=row["day"],
        starts_at=row["starts_at"],
        request=row["request"],
        status=row["status"],
        task_id=row["task_id"],
        approval_ref=row["approval_ref"],
        error=row["error"],
    )


class DayPlanRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, *, request: str, day: date, starts_at: str) -> DayPlanItem:
        text = " ".join(str(request or "").split())
        if not text:
            raise ValueError("a day-plan item needs a request")
        if len(text) > 2000:
            raise ValueError("a day-plan request must be 2000 characters or less")
        start = _clock(starts_at)
        item = DayPlanItem(
            id=uuid.uuid4().hex,
            day=day.isoformat(),
            starts_at=start,
            request=text,
            status="planned",
        )
        now = utc_now()
        self.conn.execute(
            "INSERT INTO day_plan_items"
            " (id, created_at, updated_at, day, starts_at, request, status)"
            " VALUES (?, ?, ?, ?, ?, ?, 'planned')",
            (item.id, now, now, item.day, item.starts_at, item.request),
        )
        return item

    def add_many(
        self, *, day: date, items: Iterable[tuple[str, str]]
    ) -> tuple[DayPlanItem, ...]:
        return tuple(self.add(request=request, day=day, starts_at=starts_at) for starts_at, request in items)

    def get(self, item_id: str) -> DayPlanItem | None:
        row = self.conn.execute(
            "SELECT * FROM day_plan_items WHERE id = ?", (item_id,)
        ).fetchone()
        return _row_to_item(row) if row else None

    def for_day(self, day: date) -> tuple[DayPlanItem, ...]:
        rows = self.conn.execute(
            "SELECT * FROM day_plan_items WHERE day = ?"
            " ORDER BY starts_at, created_at",
            (day.isoformat(),),
        ).fetchall()
        return tuple(_row_to_item(row) for row in rows)

    def due(
        self, day: date, *, at: datetime | str | None = None
    ) -> tuple[DayPlanItem, ...]:
        """Planned items whose local start time has arrived.

        Only ``planned`` items are returned. A paused or terminal item is never
        run again by the automatic runner.
        """
        clock = _clock_from(at or datetime.now())
        rows = self.conn.execute(
            "SELECT * FROM day_plan_items"
            " WHERE day = ? AND status = 'planned' AND starts_at <= ?"
            " ORDER BY starts_at, created_at",
            (day.isoformat(), clock),
        ).fetchall()
        return tuple(_row_to_item(row) for row in rows)

    def interrupt_running(self, day: date) -> tuple[DayPlanItem, ...]:
        """Mark rows left ``running`` by a stopped process as interrupted.

        A runner is a process, not a persistent worker. When it starts, any
        ``running`` row for its day belongs to a previous process that ended
        before recording an outcome. It must be reviewed, never replayed: the
        request may have reached a write before the crash. This deliberately
        assumes one runner per data directory; concurrent runners are not a
        supported mode.
        """
        rows = self.conn.execute(
            "SELECT * FROM day_plan_items WHERE day = ? AND status = 'running'"
            " ORDER BY starts_at, created_at",
            (day.isoformat(),),
        ).fetchall()
        return tuple(
            self.update_status(
                row["id"],
                "interrupted",
                error="previous run ended before completion; review before retrying",
            )
            for row in rows
        )

    def update_status(
        self,
        item_id: str,
        new_status: str,
        *,
        task_id: str | None = None,
        approval_ref: str | None = None,
        error: str | None = None,
    ) -> DayPlanItem:
        if new_status not in PLAN_STATUSES:
            raise ValueError(f"unknown day-plan status {new_status!r}")
        item = self.get(item_id)
        if item is None:
            raise LookupError(f"no day-plan item with id {item_id}")
        if new_status not in ALLOWED_TRANSITIONS[item.status]:
            raise ValueError(
                f"cannot move day-plan item {item_id} from {item.status!r} to {new_status!r}"
            )
        now = utc_now()
        finished_at = now if new_status in TERMINAL_STATUSES else None
        self.conn.execute(
            "UPDATE day_plan_items SET status = ?, updated_at = ?, finished_at = ?,"
            " task_id = COALESCE(?, task_id), approval_ref = COALESCE(?, approval_ref),"
            " error = COALESCE(?, error) WHERE id = ?",
            (new_status, now, finished_at, task_id, approval_ref, error, item_id),
        )
        updated = self.get(item_id)
        assert updated is not None
        return updated


def render_day_plan(items: Iterable[DayPlanItem], day: date) -> str:
    material = tuple(items)
    if not material:
        return f"No tasks planned for {describe_day(day)}."
    lines = [f"Plan for {describe_day(day)}:"]
    lines.extend(f"- {item.as_line()}" for item in material)
    return "\n".join(lines)


__all__ = [
    "ALLOWED_TRANSITIONS",
    "DayPlanItem",
    "DayPlanRepository",
    "PLAN_STATUSES",
    "render_day_plan",
]
