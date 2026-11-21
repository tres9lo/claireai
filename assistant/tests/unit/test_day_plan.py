"""The full-day planner: local rows, safe tools and a write-pausing runner."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from claireai.brain import BrainHealth, BrainResponse, ToolCall
from claireai.config import Settings
from claireai.permissions import Sandbox
from claireai.planner import PauseApprovalBroker, run_due_items
from claireai.routing import route
from claireai.storage import DayPlanRepository
from claireai.storage.database import Database
from claireai.summary import build as build_summary
from claireai.tools import default_registry
from claireai.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from claireai.tools.day_plan import (
    ListDayPlanArgs,
    PlanDayArgs,
    PlanItemInput,
    list_day_plan,
    plan_day,
)

DB = Path("db") / "claireai.sqlite3"
DAY = date(2026, 11, 21)


def _context(tmp_path: Path) -> ToolContext:
    Database(tmp_path / DB).migrate()
    return ToolContext(sandbox=Sandbox(roots=()), data_dir=tmp_path)


def test_plan_items_are_sorted_and_dry_run_does_not_write(tmp_path: Path) -> None:
    context = _context(tmp_path)
    preview = plan_day(
        ToolContext(sandbox=Sandbox(roots=()), data_dir=tmp_path, dry_run=True),
        PlanDayArgs(
            day="2026-11-21",
            items=[
                PlanItemInput(starts_at="14:00", request="Review notes"),
                PlanItemInput(starts_at="09:00", request="Read inbox"),
            ],
        ),
    )
    assert preview.detail["dry_run"] is True
    assert preview.detail["count"] == 2
    assert list_day_plan(context, ListDayPlanArgs(day="2026-11-21")).detail["count"] == 0

    created = plan_day(
        context,
        PlanDayArgs(
            day="2026-11-21",
            items=[
                PlanItemInput(starts_at="14:00", request="Review notes"),
                PlanItemInput(starts_at="09:00", request="Read inbox"),
            ],
        ),
    )
    assert created.detail["count"] == 2
    lines = list_day_plan(context, ListDayPlanArgs(day="2026-11-21")).text.splitlines()
    assert lines[1].startswith("- 09:00 Read inbox")
    assert lines[2].startswith("- 14:00 Review notes")


def test_due_only_returns_planned_items_that_have_arrived(conn) -> None:
    repo = DayPlanRepository(conn)
    early, late = repo.add_many(
        day=DAY,
        items=[("09:00", "Early"), ("15:00", "Late")],
    )
    assert [item.request for item in repo.due(DAY, at=datetime(2026, 11, 21, 10, 0))] == [
        "Early"
    ]
    repo.update_status(early.id, "running")
    repo.update_status(early.id, "awaiting_approval", approval_ref="ref-1")
    assert repo.due(DAY, at=datetime(2026, 11, 21, 16, 0)) == (late,)


def test_the_plan_tool_needs_approval_and_reading_is_safe() -> None:
    registry = default_registry()
    assert registry.get("plan_day").risk == "needs_approval"
    assert registry.get("list_day_plan").risk == "safe"
    result = route("plan my day with reading and a review", registry)
    assert result.intent == "day_plan"
    assert "plan_day" in result.tools
    assert "list_day_plan" in result.tools


class _NoArgs(BaseModel):
    pass


class QueueBrain:
    name = "queue"

    def __init__(self, responses: list[BrainResponse]) -> None:
        self.responses = list(responses)

    async def health(self) -> BrainHealth:
        return BrainHealth(available=True, model_present=True, models=("queue",), detail="ok")

    async def generate(self, messages, tools=None, *, temperature=None) -> BrainResponse:
        return self.responses.pop(0)


async def test_runner_runs_safe_items_and_pauses_writes(
    conn, settings: Settings
) -> None:
    writes: list[str] = []

    def read(context: ToolContext, arguments: _NoArgs) -> ToolResult:
        return ToolResult("a note")

    def write(context: ToolContext, arguments: _NoArgs) -> ToolResult:
        writes.append("write")
        return ToolResult("wrote")

    registry = ToolRegistry(
        [
            Tool(
                name="read_note",
                description="read a note",
                risk="safe",
                args_model=_NoArgs,
                handler=read,
            ),
            Tool(
                name="write_note",
                description="write a note",
                risk="needs_approval",
                args_model=_NoArgs,
                handler=write,
            ),
        ]
    )
    brain = QueueBrain(
        [
            BrainResponse(content="", model="queue", tool_calls=(ToolCall(name="read_note"),)),
            BrainResponse(content="read done", model="queue"),
            BrainResponse(content="", model="queue", tool_calls=(ToolCall(name="write_note"),)),
            BrainResponse(content="write not approved", model="queue"),
        ]
    )
    repo = DayPlanRepository(conn)
    repo.add_many(
        day=DAY,
        items=[("09:00", "read the note"), ("09:01", "write the note")],
    )

    report = await run_due_items(
        settings=settings,
        conn=conn,
        brain=brain,
        day=DAY,
        now=datetime(2026, 11, 21, 10, 0),
        registry=registry,
        sandbox=Sandbox(roots=()),
    )

    statuses = [item.status for item in repo.for_day(DAY)]
    assert statuses == ["succeeded", "awaiting_approval"]
    assert writes == []
    assert [item.status for item in report.items] == ["succeeded", "awaiting_approval"]
    paused = repo.for_day(DAY)[1]
    assert paused.approval_ref
    # The evening summary must expose the action that did not happen, even
    # though the task itself finished normally.
    summary = build_summary(conn)
    skipped = [reason for line in summary.failed for reason in line.skipped]
    assert "write_note (not approved)" in skipped


def test_plan_state_machine_refuses_to_replay_a_finished_item(conn) -> None:
    repo = DayPlanRepository(conn)
    item = repo.add(day=DAY, starts_at="09:00", request="read the note")
    repo.update_status(item.id, "running")
    repo.update_status(item.id, "succeeded")

    with pytest.raises(ValueError, match="cannot move"):
        repo.update_status(item.id, "running")


async def test_the_runner_reports_stale_running_items_instead_of_replaying(
    conn, settings: Settings
) -> None:
    repo = DayPlanRepository(conn)
    item = repo.add(day=DAY, starts_at="09:00", request="possibly ran before a crash")
    repo.update_status(item.id, "running")

    report = await run_due_items(
        settings=settings,
        conn=conn,
        brain=QueueBrain([]),
        day=DAY,
        now=datetime(2026, 11, 21, 10, 0),
        registry=default_registry(),
        sandbox=Sandbox(roots=()),
    )

    stale = repo.get(item.id)
    assert stale is not None and stale.status == "interrupted"
    assert "review before retrying" in (stale.error or "")
    assert [entry.status for entry in report.items] == ["interrupted"]
    assert report.failed is True
    assert repo.due(DAY, at=datetime(2026, 11, 21, 10, 0)) == ()


def test_pause_broker_denies_and_records() -> None:
    broker = PauseApprovalBroker()
    request = type(
        "Request",
        (),
        {"ref": "r", "tool": "write_note"},
    )()
    import asyncio

    decision = asyncio.run(broker.decide(request))  # type: ignore[arg-type]
    assert decision.approved is False
    assert broker.seen == [request]


def test_bad_plan_time_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="24-hour"):
        plan_day(
            _context(tmp_path),
            PlanDayArgs(day="today", items=[PlanItemInput(starts_at="9am", request="Read")]),
        )
