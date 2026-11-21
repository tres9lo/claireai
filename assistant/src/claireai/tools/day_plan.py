"""Tools for assigning and reading the owner's full-day plan.

``plan_day`` writes local rows and therefore needs approval. ``list_day_plan`` is
read-only and safe. The tools resolve phrases like ``tomorrow`` with the same
deterministic date parser as the calendar; the model never invents the day.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from claireai.audit import redact
from claireai.dates import describe_day, parse_when
from claireai.storage.database import Database, transaction
from claireai.storage.day_plan import DayPlanRepository, render_day_plan
from claireai.tools.base import ToolContext, ToolError, ToolResult

_CLOCK = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanItemInput(_Args):
    starts_at: str = Field(description="Local start time, 24-hour HH:MM.")
    request: str = Field(min_length=1, max_length=2000, description="What to do.")


class PlanDayArgs(_Args):
    day: str = Field(
        default="today",
        description="Which day: 'today', 'tomorrow', '2026-11-21' or a weekday name.",
    )
    items: list[PlanItemInput] = Field(
        min_length=1,
        max_length=24,
        description="Ordered tasks with local start times.",
    )


class ListDayPlanArgs(_Args):
    day: str = Field(
        default="today",
        description="Which day: 'today', 'tomorrow', '2026-11-21' or a weekday name.",
    )


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


def _validate_clock(value: str) -> str:
    text = " ".join(str(value or "").split())
    if not _CLOCK.match(text):
        raise ToolError(f"starts_at must be a 24-hour time like 09:30, got {value!r}")
    return text


def plan_day(context: ToolContext, args: PlanDayArgs) -> ToolResult:
    day = _resolve_day(args.day)
    items = [(_validate_clock(item.starts_at), redact(item.request)) for item in args.items]
    if context.dry_run:
        lines = [f"[dry run] would plan for {describe_day(day)}:"]
        lines.extend(f"- {starts_at} {request}" for starts_at, request in items)
        return ToolResult(
            "\n".join(lines),
            {"day": day.isoformat(), "count": len(items), "dry_run": True},
        )

    conn = _connect(context)
    try:
        with transaction(conn):
            created = DayPlanRepository(conn).add_many(day=day, items=items)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise ToolError(
            "the day_plan_items table is missing; start ClaireAI once so migrations run"
        ) from exc
    finally:
        conn.close()
    return ToolResult(
        render_day_plan(created, day),
        {"day": day.isoformat(), "count": len(created), "ids": [item.id for item in created]},
    )


def list_day_plan(context: ToolContext, args: ListDayPlanArgs) -> ToolResult:
    day = _resolve_day(args.day)
    conn = _connect(context)
    try:
        items = DayPlanRepository(conn).for_day(day)
    except sqlite3.Error as exc:
        raise ToolError(
            "the day_plan_items table is missing; start ClaireAI once so migrations run"
        ) from exc
    finally:
        conn.close()
    return ToolResult(
        render_day_plan(items, day),
        {"day": day.isoformat(), "count": len(items)},
    )


__all__ = [
    "ListDayPlanArgs",
    "PlanDayArgs",
    "PlanItemInput",
    "list_day_plan",
    "plan_day",
]
