"""Run due day-plan items without granting unattended writes.

The runner uses the normal :func:`claireai.agent.run_task` loop, so the registry,
sandbox, risk policy and audit log all still apply. It attaches a broker that
records an approval request and refuses it. A task that reaches such a request
is marked ``awaiting_approval`` and is not picked up again automatically.

This is deliberately not an "always approve" worker: an unattended process must
not authorise its own file writes, downloads or web actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from claireai.agent import AgentError, run_task
from claireai.approval import ApprovalDecision, ApprovalRequest
from claireai.brain import Brain, BrainError, BrainStopped
from claireai.config import Settings
from claireai.dates import describe_day
from claireai.permissions import Sandbox
from claireai.storage.day_plan import DayPlanItem, DayPlanRepository
from claireai.tools.base import ToolRegistry


@dataclass
class PauseApprovalBroker:
    """Record an approval request, deny it, and let the plan item pause."""

    name: str = "day-plan-pause"
    seen: list[ApprovalRequest] = field(default_factory=list)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.seen.append(request)
        return ApprovalDecision(
            approved=False,
            reason=(
                "the day-plan runner pauses actions that need approval; "
                "run the request manually and approve it in chat"
            ),
            decided_by="policy",
        )


@dataclass(frozen=True)
class PlanRunItem:
    item: DayPlanItem
    status: str
    task_id: str | None = None
    message: str = ""


@dataclass(frozen=True)
class DayPlanRun:
    day: date
    items: tuple[PlanRunItem, ...] = ()

    @property
    def failed(self) -> bool:
        return any(item.status in {"failed", "interrupted"} for item in self.items)


def _set_status(
    repo: DayPlanRepository,
    item: DayPlanItem,
    status: str,
    *,
    task_id: str | None = None,
    approval_ref: str | None = None,
    error: str | None = None,
) -> DayPlanItem:
    return repo.update_status(
        item.id,
        status,
        task_id=task_id,
        approval_ref=approval_ref,
        error=error,
    )


async def run_due_items(
    *,
    settings: Settings,
    conn,
    brain: Brain,
    day: date | None = None,
    now: datetime | None = None,
    registry: ToolRegistry | None = None,
    sandbox: Sandbox | None = None,
    device: str = "desktop",
) -> DayPlanRun:
    """Run planned items whose local start time has arrived.

    Safe actions may execute. If the model asks for approval, the broker refuses
    it and the item is paused; no mutation happens without a human decision.
    """
    target_day = day or (now.date() if now else date.today())
    repo = DayPlanRepository(conn)
    # A fresh runner process is evidence that a row still marked ``running``
    # belongs to a process that stopped before recording an outcome. Report it
    # for review instead of replaying a request that may already have reached a
    # write.
    reports: list[PlanRunItem] = [
        PlanRunItem(
            item=item,
            status="interrupted",
            message=item.error or "previous run ended before completion",
        )
        for item in repo.interrupt_running(target_day)
    ]
    due = repo.due(target_day, at=now)

    for item in due:
        try:
            _set_status(repo, item, "running")
        except (ValueError, LookupError) as exc:
            reports.append(PlanRunItem(item=item, status="failed", message=str(exc)))
            continue

        broker = PauseApprovalBroker()
        try:
            outcome = await run_task(
                settings=settings,
                conn=conn,
                brain=brain,
                request=item.request,
                registry=registry,
                sandbox=sandbox,
                approvals=broker,
                device=device,
            )
        except BrainStopped as exc:
            _set_status(repo, item, "interrupted", error=str(exc))
            reports.append(
                PlanRunItem(item=item, status="interrupted", message="ClaireAI was stopped")
            )
            break
        except (AgentError, BrainError) as exc:
            if broker.seen:
                first = broker.seen[0]
                _set_status(
                    repo,
                    item,
                    "awaiting_approval",
                    approval_ref=first.ref,
                    error="action needs owner approval",
                )
                reports.append(
                    PlanRunItem(
                        item=item,
                        status="awaiting_approval",
                        message=f"{first.tool} needs approval",
                    )
                )
            else:
                _set_status(repo, item, "failed", error=str(exc))
                reports.append(PlanRunItem(item=item, status="failed", message=str(exc)))
            continue

        if broker.seen:
            first = broker.seen[0]
            _set_status(
                repo,
                item,
                "awaiting_approval",
                task_id=outcome.task_id,
                approval_ref=first.ref,
                error="action needs owner approval",
            )
            reports.append(
                PlanRunItem(
                    item=item,
                    status="awaiting_approval",
                    task_id=outcome.task_id,
                    message=f"{first.tool} needs approval",
                )
            )
            continue

        _set_status(repo, item, "succeeded", task_id=outcome.task_id)
        reports.append(
            PlanRunItem(
                item=item,
                status="succeeded",
                task_id=outcome.task_id,
                message=outcome.content,
            )
        )

    return DayPlanRun(day=target_day, items=tuple(reports))


def render_day_plan_run(report: DayPlanRun) -> str:
    if not report.items:
        return f"No day-plan tasks were due for {describe_day(report.day)}."
    lines = [f"Day-plan run for {describe_day(report.day)}:"]
    for result in report.items:
        detail = f" — {result.message}" if result.message else ""
        lines.append(f"- {result.item.starts_at} {result.item.request} -> {result.status}{detail}")
    return "\n".join(lines)


__all__ = [
    "DayPlanRun",
    "PauseApprovalBroker",
    "PlanRunItem",
    "render_day_plan_run",
    "run_due_items",
]
