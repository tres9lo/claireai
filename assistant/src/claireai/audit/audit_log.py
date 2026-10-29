"""Append-only audit logging.

The application only ever inserts here; the schema triggers reject UPDATE and
DELETE. Details are redacted and truncated on the way in so credentials and
full private payloads are never stored by default (roadmap 7A.4).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping

from claireai.audit.redaction import redact
from claireai.storage.tasks import utc_now

LOG_STATUSES = frozenset({"done", "failed", "skipped", "pending"})


@dataclass(frozen=True)
class AuditEntry:
    id: int
    created_at: str
    device: str
    task_id: str | None
    actor: str
    tool: str | None
    action: str
    target: str | None
    status: str
    approval_ref: str | None
    detail: str | None


def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
    return AuditEntry(
        id=row["id"],
        created_at=row["created_at"],
        device=row["device"],
        task_id=row["task_id"],
        actor=row["actor"],
        tool=row["tool"],
        action=row["action"],
        target=row["target"],
        status=row["status"],
        approval_ref=row["approval_ref"],
        detail=row["detail"],
    )


class AuditLog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def record(
        self,
        *,
        action: str,
        status: str,
        actor: str = "user",
        device: str = "desktop",
        task_id: str | None = None,
        tool: str | None = None,
        target: str | None = None,
        approval_ref: str | None = None,
        detail: Mapping[str, Any] | str | None = None,
    ) -> AuditEntry:
        if status not in LOG_STATUSES:
            raise ValueError(f"unknown audit status {status!r}; expected one of {sorted(LOG_STATUSES)}")
        if not action.strip():
            raise ValueError("audit action must not be empty")

        serialized_detail: str | None
        if detail is None:
            serialized_detail = None
        elif isinstance(detail, str):
            serialized_detail = redact(detail)
        else:
            serialized_detail = redact(json.dumps(dict(detail), default=str, sort_keys=True))

        cursor = self._conn.execute(
            "INSERT INTO task_log (created_at, device, task_id, actor, tool, action, target,"
            " status, approval_ref, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                utc_now(),
                device,
                task_id,
                actor,
                tool,
                action,
                redact(target) if target else None,
                status,
                approval_ref,
                serialized_detail,
            ),
        )
        row = self._conn.execute("SELECT * FROM task_log WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _row_to_entry(row)

    def for_task(self, task_id: str) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT * FROM task_log WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def recent(self, limit: int = 20) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT * FROM task_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_entry(row) for row in rows]
