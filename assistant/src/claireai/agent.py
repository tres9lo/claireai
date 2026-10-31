"""The agent loop (Month 1, Weeks 2-3).

The model proposes, this module decides. For every proposed call the runtime:

1. looks the tool up in the registry (unknown → refused, logged);
2. validates the arguments against the tool's pydantic model (invalid → refused,
   logged, and the error is fed back so the model can correct itself);
3. asks the owner for anything above ``safe`` (``needs_approval`` /
   ``dangerous``) through an :mod:`claireai.approval` broker, showing the exact
   tool, arguments and target — what is approved is exactly what runs;
4. runs the handler through the folder allow-list, then logs the outcome.

The loop stops when the model replies without tool calls, or after
``agent_max_steps`` steps (default 8), whichever comes first. Every step writes a
``task_log`` row, so the daily summary can be built from the log alone. Approval
decisions are logged too, and the tool row carries the same ``approval_ref``.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from claireai.approval import ApprovalBroker, ApprovalRequest, DenyAll, new_ref
from claireai.audit import AuditLog, redact
from claireai.brain import Brain, BrainError, Message, ToolCall
from claireai.config import Settings
from claireai.permissions import PermissionDenied, Sandbox, check_risk, requires_approval
from claireai.presence import PresenceStream
from claireai.routing import Route, route as choose_route, schema_bytes
from claireai.storage import TaskRepository, transaction
from claireai.tools import ToolError, ToolRegistry, UnknownTool
from claireai.tools.base import ToolContext

SYSTEM_PROMPT = (
    "You are ClaireAI, a local personal assistant on the owner's own computer. "
    "You can read files with the provided tools, and only inside the allowed folders. "
    "Call a tool when the request needs real data; do not guess file names or sizes. "
    "For the biggest or largest files, call find_largest_files and answer only "
    "from the sizes it returns; never invent a size. "
    "Lines starting with 'dir' are folders, not files; only 'file' lines have sizes. "
    "Write tools exist: they create, write, rename or move files, and every one needs "
    "the owner's approval before it runs. Propose them; never assume approval was given. "
    "Nothing you have deletes permanently: move_to_trash moves a file to a recoverable bin. "
    "When you have the answer, reply with a short final answer and no tool call. "
    "If a tool returns an error or is denied, explain it or correct the arguments."
)

MAX_TOOL_TEXT = 4000


class AgentError(Exception):
    """The agent could not finish the request."""


class AgentStepLimit(AgentError):
    """The model kept calling tools past the configured step limit."""


class AgentEmptyReply(AgentError):
    """The model returned neither text nor a tool call.

    Observed for real with ``qwen2.5:1.5b``: an empty assistant message is not a
    valid answer, and treating it as one shows the owner a blank screen while the
    task claims success.
    """


@dataclass(frozen=True)
class AgentStep:
    index: int
    tool: str
    arguments: Mapping[str, Any]
    status: str
    summary: str
    duration_ms: int
    result_text: str = ""
    approval_ref: str | None = None


@dataclass(frozen=True)
class AgentOutcome:
    task_id: str
    content: str
    model: str
    steps: tuple[AgentStep, ...] = field(default_factory=tuple)
    duration_ms: int = 0
    route: str | None = None


def _clip(text: str, limit: int = MAX_TOOL_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated, {len(text) - limit} characters omitted]"


def system_message(sandbox: Sandbox) -> Message:
    return Message(role="system", content=f"{SYSTEM_PROMPT} Allowed folders: {sandbox.describe()}.")


async def run_task(
    *,
    settings: Settings,
    conn: sqlite3.Connection,
    brain: Brain,
    request: str,
    registry: ToolRegistry | None = None,
    sandbox: Sandbox | None = None,
    approvals: ApprovalBroker | None = None,
    presence: PresenceStream | None = None,
    device: str = "desktop",
    dry_run: bool = False,
    max_steps: int | None = None,
) -> AgentOutcome:
    """Run one request to completion, logging every step."""
    from claireai.tools import default_registry  # local import keeps import graph flat

    registry = registry or default_registry()
    sandbox = sandbox or Sandbox.from_settings(settings)
    broker: ApprovalBroker = approvals or DenyAll()
    step_limit = max_steps or settings.agent_max_steps
    data_dir = settings.resolved_data_dir

    tasks = TaskRepository(conn)
    audit = AuditLog(conn)

    # Which tools the model is shown. A route never grants or removes
    # permission: validation and approval still run against the full registry,
    # so a filtered-out call that arrives anyway simply executes.
    plan: Route | None = choose_route(request, registry) if settings.tool_routing == "auto" else None
    schemas = plan.schemas(registry) if plan else registry.schemas()

    with transaction(conn):
        task = tasks.create(request=redact(request), kind="agent", device=device)
    with transaction(conn):
        tasks.update_status(task.id, "planning")
    with transaction(conn):
        tasks.update_status(task.id, "running")
        if plan:
            audit.record(
                task_id=task.id,
                device=device,
                action="agent.route",
                status="done",
                target=plan.intent,
                detail={
                    "intent": plan.intent,
                    "tools": list(plan.tools),
                    "tools_total": len(registry),
                    "narrowed": plan.narrowed,
                    "schema_chars": schema_bytes(schemas),
                    "schema_chars_all": schema_bytes(registry.schemas()),
                },
            )
    if presence:
        presence.emit("thinking", task_id=task.id)

    messages: list[Message] = [system_message(sandbox), Message(role="user", content=request)]
    steps: list[AgentStep] = []
    started = time.perf_counter()

    try:
        for step_number in range(1, step_limit + 1):
            response = await brain.generate(messages, tools=schemas)

            if not response.tool_calls:
                if not response.content.strip():
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    with transaction(conn):
                        audit.record(
                            task_id=task.id,
                            device=device,
                            action="agent.answer",
                            status="failed",
                            target=response.model,
                            detail={
                                "model": response.model,
                                "steps": len(steps),
                                "duration_ms": duration_ms,
                                "reason": "empty reply",
                            },
                        )
                        tasks.update_status(
                            task.id, "failed", error="the model returned an empty reply"
                        )
                    if presence:
                        presence.emit("error", detail={"reason": "empty reply"})
                    raise AgentEmptyReply(
                        "the model returned an empty reply (no text and no tool call); try again"
                    )
                duration_ms = int((time.perf_counter() - started) * 1000)
                with transaction(conn):
                    audit.record(
                        task_id=task.id,
                        device=device,
                        action="agent.answer",
                        status="done",
                        target=response.model,
                        detail={
                            "model": response.model,
                            "steps": len(steps),
                            "duration_ms": duration_ms,
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                            "dry_run": dry_run,
                            "route": plan.intent if plan else None,
                        },
                    )
                    tasks.update_status(task.id, "succeeded", current_step=len(steps))
                if presence:
                    presence.emit("idle", task_id=task.id)
                return AgentOutcome(
                    task_id=task.id,
                    content=response.content.strip(),
                    model=response.model,
                    steps=tuple(steps),
                    duration_ms=duration_ms,
                    route=plan.intent if plan else None,
                )

            with transaction(conn):
                tasks.set_current_step(task.id, step_number)
            messages.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )
            for position, call in enumerate(response.tool_calls):
                step = await _execute(
                    call=call,
                    index=len(steps) + 1,
                    task_id=task.id,
                    device=device,
                    registry=registry,
                    sandbox=sandbox,
                    data_dir=data_dir,
                    approvals=broker,
                    presence=presence,
                    audit=audit,
                    conn=conn,
                    dry_run=dry_run,
                )
                steps.append(step)
                messages.append(
                    Message(role="tool", tool_name=call.name, content=_clip(step.result_text))
                )
                if len(steps) >= step_limit and position < len(response.tool_calls) - 1:
                    break

        duration_ms = int((time.perf_counter() - started) * 1000)
        error = f"step limit ({step_limit}) reached without a final answer"
        with transaction(conn):
            audit.record(
                task_id=task.id,
                device=device,
                action="agent.step_limit",
                status="failed",
                target=None,
                detail={"steps": len(steps), "limit": step_limit, "duration_ms": duration_ms},
            )
            tasks.update_status(task.id, "failed", error=error)
        if presence:
            presence.emit("error", detail={"reason": "step limit"})
        raise AgentStepLimit(error)

    except BrainError as exc:
        with transaction(conn):
            audit.record(
                task_id=task.id,
                device=device,
                action="agent.run",
                status="failed",
                target=None,
                detail={"error": str(exc), "steps": len(steps)},
            )
            tasks.update_status(task.id, "failed", error=str(exc))
        if presence:
            presence.emit("offline" if "reach" in str(exc).lower() else "error")
        raise


async def _execute(
    *,
    call: ToolCall,
    index: int,
    task_id: str,
    device: str,
    registry: ToolRegistry,
    sandbox: Sandbox,
    data_dir: Path,
    approvals: ApprovalBroker,
    presence: PresenceStream | None,
    audit: AuditLog,
    conn: sqlite3.Connection,
    dry_run: bool,
) -> AgentStep:
    """Validate, authorize, run and log one proposed call."""
    started = time.perf_counter()
    arguments = dict(call.arguments)
    target = arguments.get("path") if isinstance(arguments.get("path"), str) else None
    if target is None and isinstance(arguments.get("source"), str):
        target = arguments["source"]
    context = ToolContext(sandbox=sandbox, dry_run=dry_run, data_dir=data_dir)

    status = "failed"
    approval_ref: str | None = None
    detail: dict[str, Any] = {"arguments": arguments, "dry_run": dry_run}
    text = ""
    try:
        tool = registry.get(call.name)
        check_risk(tool.risk)
        validated = tool.validate(arguments)
        rendered = validated.model_dump(mode="json")

        if dry_run:
            status = "skipped"
            text = f"[dry run] would call {tool.name} with {rendered}"
            detail["reason"] = "dry_run"
        else:
            if requires_approval(tool.risk):
                request = ApprovalRequest(
                    ref=new_ref(),
                    task_id=task_id,
                    tool=tool.name,
                    risk=tool.risk,
                    arguments=rendered,
                    target=target,
                    preview=f"would {tool.name} with {rendered}",
                )
                approval_ref = request.ref
                if presence:
                    presence.emit(
                        "awaiting_approval",
                        task_id=task_id,
                        detail={"tool": tool.name, "ref": request.ref},
                    )
                decision = await approvals.decide(request)
                with transaction(conn):
                    audit.record(
                        task_id=task_id,
                        device=device,
                        actor=decision.decided_by,
                        tool=tool.name,
                        action="approval.decision",
                        target=target,
                        status="done" if decision.approved else "skipped",
                        approval_ref=request.ref,
                        detail={
                            "risk": tool.risk,
                            "arguments": rendered,
                            "approved": decision.approved,
                            "reason": decision.reason,
                        },
                    )
                if not decision.approved:
                    status = "skipped"
                    detail["reason"] = "approval_denied"
                    detail["approval_ref"] = request.ref
                    text = (
                        "the owner did not approve this action"
                        + (f": {decision.reason}" if decision.reason else "")
                    )
                    return _finish(
                        audit=audit,
                        conn=conn,
                        task_id=task_id,
                        device=device,
                        call=call,
                        index=index,
                        arguments=arguments,
                        target=target,
                        status=status,
                        text=text,
                        detail=detail,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        approval_ref=approval_ref,
                        presence=presence,
                    )
            # Emitted for every tool that actually executes, approved or safe.
            if presence:
                presence.emit("tool_running", task_id=task_id, detail={"tool": tool.name})
            result = tool.handler(context, validated)
            status = "done"
            text = result.text
            detail.update(dict(result.detail))
            if approval_ref:
                detail["approval_ref"] = approval_ref

    except UnknownTool as exc:
        status = "skipped"
        text = f"error: {exc}"
    except PermissionDenied as exc:
        status = "skipped"
        text = f"permission denied: {exc}"
        detail["reason"] = "permission_denied"
    except ToolError as exc:
        status = "failed"
        text = f"error: {exc}"
        detail["error"] = str(exc)
    except OSError as exc:
        status = "failed"
        text = f"error: the operating system refused the request ({exc.strerror or exc})"
        detail["error"] = text

    return _finish(
        audit=audit,
        conn=conn,
        task_id=task_id,
        device=device,
        call=call,
        index=index,
        arguments=arguments,
        target=target,
        status=status,
        text=text,
        detail=detail,
        duration_ms=int((time.perf_counter() - started) * 1000),
        approval_ref=approval_ref,
        presence=presence,
    )


def _finish(
    *,
    audit: AuditLog,
    conn: sqlite3.Connection,
    task_id: str,
    device: str,
    call: ToolCall,
    index: int,
    arguments: Mapping[str, Any],
    target: str | None,
    status: str,
    text: str,
    detail: dict[str, Any],
    duration_ms: int,
    approval_ref: str | None,
    presence: PresenceStream | None,
) -> AgentStep:
    """Write the tool row and build the step record."""
    summary = text if len(text) <= 160 else text[:157] + "..."
    detail["duration_ms"] = duration_ms
    detail["status"] = status
    detail["result_preview"] = summary

    with transaction(conn):
        audit.record(
            task_id=task_id,
            device=device,
            actor="agent",
            tool=call.name,
            action="tool.call",
            target=target,
            status=status,
            approval_ref=approval_ref,
            detail=detail,
        )

    return AgentStep(
        index=index,
        tool=call.name,
        arguments=arguments,
        status=status,
        summary=summary,
        duration_ms=duration_ms,
        result_text=text,
        approval_ref=approval_ref,
    )


def describe_steps(steps: Sequence[AgentStep]) -> str:
    """Human-facing one line per step, used by the CLI."""
    lines = []
    for step in steps:
        marker = f" (approval {step.approval_ref})" if step.approval_ref else ""
        lines.append(
            f"  {step.index}. {step.tool} [{step.status}] {step.duration_ms} ms{marker} — {step.summary}"
        )
    return "\n".join(lines)
