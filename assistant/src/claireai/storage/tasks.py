"""Task records and their explicit state machine.

Every user request becomes a task row so that the daily summary can be built
from the task log alone (roadmap 6.3). Invalid transitions are rejected here
rather than left to the caller: a cancelled or finished task must never later
appear to be running.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

STATUSES = frozenset(
    {
        "queued",
        "planning",
        "awaiting_approval",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "interrupted",
    }
)

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"planning", "running", "failed", "cancelled"}),
    "planning": frozenset({"awaiting_approval", "running", "failed", "cancelled"}),
    "awaiting_approval": frozenset({"running", "failed", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled", "interrupted"}),
    # An interrupted task must be reconciled by a human-driven retry, never
    # silently replayed (roadmap 7A.9).
    "interrupted": frozenset({"running", "failed", "cancelled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class InvalidTransition(ValueError):
    """Raised when a task is moved between states the state machine forbids."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Task:
    id: str
    created_at: str
    updated_at: str
    finished_at: str | None
    device: str
    actor: str
    kind: str
    request: str
    status: str
    current_step: int | None
    error: str | None


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
        device=row["device"],
        actor=row["actor"],
        kind=row["kind"],
        request=row["request"],
        status=row["status"],
        current_step=row["current_step"],
        error=row["error"],
    )


class TaskRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(
        self,
        *,
        request: str,
        kind: str = "chat",
        device: str = "desktop",
        actor: str = "user",
    ) -> Task:
        now = utc_now()
        task_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO tasks (id, created_at, updated_at, device, actor, kind, request, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')",
            (task_id, now, now, device, actor, kind, request),
        )
        task = self.get(task_id)
        assert task is not None  # just inserted
        return task

    def get(self, task_id: str) -> Task | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def recent(self, limit: int = 20) -> list[Task]:
        rows = self._conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_task(row) for row in rows]

    def update_status(
        self,
        task_id: str,
        new_status: str,
        *,
        error: str | None = None,
        current_step: int | None = None,
    ) -> Task:
        if new_status not in STATUSES:
            raise InvalidTransition(f"unknown task status {new_status!r}")
        task = self.get(task_id)
        if task is None:
            raise LookupError(f"no task with id {task_id}")
        if new_status not in ALLOWED_TRANSITIONS[task.status]:
            raise InvalidTransition(f"cannot move task {task_id} from {task.status!r} to {new_status!r}")

        now = utc_now()
        finished_at = now if new_status in TERMINAL_STATUSES else task.finished_at
        self._conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ?, finished_at = ?,"
            " current_step = COALESCE(?, current_step), error = COALESCE(?, error)"
            " WHERE id = ?",
            (new_status, now, finished_at, current_step, error, task_id),
        )
        updated = self.get(task_id)
        assert updated is not None
        return updated

    def set_current_step(self, task_id: str, step: int) -> None:
        """Record how far into the plan a running task is.

        Kept separate from :meth:`update_status` because a step counter is not a
        state change: ``running`` does not legally transition to ``running``, and
        loosening the state machine to allow it would hide real bugs.
        """
        cursor = self._conn.execute(
            "UPDATE tasks SET current_step = ?, updated_at = ? WHERE id = ?",
            (step, utc_now(), task_id),
        )
        if cursor.rowcount == 0:
            raise LookupError(f"no task with id {task_id}")
