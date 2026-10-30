"""The agent loop (Month 1, Week 2).

The model proposes, this module decides. For every proposed call the runtime:

1. looks the tool up in the registry (unknown → refused, logged);
2. validates the arguments against the tool's pydantic model (invalid → refused,
   logged, and the error is fed back so the model can correct itself);
3. applies the risk policy (anything not ``safe`` is refused until approval
   exists — see :func:`claireai.permissions.check_risk`);
4. runs the handler through the folder allow-list, then logs the outcome.

The loop stops when the model replies without tool calls, or after
``agent_max_steps`` steps (default 8), whichever comes first. Every step writes a
``task_log`` row, so the daily summary can be built from the log alone.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from claireai.audit import AuditLog, redact
from claireai.brain import Brain, BrainError, Message, ToolCall
from claireai.config import Settings
from claireai.permissions import PermissionDenied, Sandbox, check_risk
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
    "When you have the answer, reply with a short final answer and no tool call. "
    "If a tool returns an error, correct the arguments or explain the problem. "
    "Never claim you changed, deleted or moved anything: you have no write tools."
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


@dataclass(frozen=True)
class AgentOutcome:
    task_id: str
    content: str
    model: str
    steps: tuple[AgentStep, ...] = field(default_factory=tuple)
    duration_ms: int = 0


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
    device: str = "desktop",
    dry_run: bool = False,
    max_steps: int | None = None,
    approval_available: bool = False,
) -> AgentOutcome:
    """Run one request to completion, logging every step."""
    from claireai.tools import default_registry  # local import keeps import graph flat

    registry = registry or default_registry()
    sandbox = sandbox or Sandbox.from_settings(settings)
    step_limit = max_steps or settings.agent_max_steps

    tasks = TaskRepository(conn)
    audit = AuditLog(conn)

    with transaction(conn):
        task = tasks.create(request=redact(request), kind="agent", device=device)
    with transaction(conn):
        tasks.update_status(task.id, "planning")
    with transaction(conn):
        tasks.update_status(task.id, "running")

    messages: list[Message] = [system_message(sandbox), Message(role="user", content=request)]
    steps: list[AgentStep] = []
    started = time.perf_counter()

    try:
        for step_number in range(1, step_limit + 1):
            response = await brain.generate(messages, tools=registry.schemas())

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
                        },
                    )
                    tasks.update_status(task.id, "succeeded", current_step=len(steps))
                return AgentOutcome(
                    task_id=task.id,
                    content=response.content.strip(),
                    model=response.model,
                    steps=tuple(steps),
                    duration_ms=duration_ms,
                )

            with transaction(conn):
                tasks.set_current_step(task.id, step_number)
            messages.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )
            for position, call in enumerate(response.tool_calls):
                step = _execute(
                    call=call,
                    index=len(steps) + 1,
                    task_id=task.id,
                    device=device,
                    registry=registry,
                    sandbox=sandbox,
                    audit=audit,
                    conn=conn,
                    dry_run=dry_run,
                    approval_available=approval_available,
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
        raise


def _execute(
    *,
    call: ToolCall,
    index: int,
    task_id: str,
    device: str,
    registry: ToolRegistry,
    sandbox: Sandbox,
    audit: AuditLog,
    conn: sqlite3.Connection,
    dry_run: bool,
    approval_available: bool,
) -> AgentStep:
    """Validate, authorize, run and log one proposed call."""
    started = time.perf_counter()
    arguments = dict(call.arguments)
    target = arguments.get("path") if isinstance(arguments.get("path"), str) else None
    context = ToolContext(sandbox=sandbox, dry_run=dry_run)

    status = "failed"
    detail: dict[str, Any] = {"arguments": arguments, "dry_run": dry_run}
    try:
        tool = registry.get(call.name)
        check_risk(tool.risk, approval_available=approval_available)
        validated = tool.validate(arguments)

        if dry_run:
            status = "skipped"
            text = f"[dry run] would call {tool.name} with {validated.model_dump(mode='json')}"
            detail["reason"] = "dry_run"
        else:
            result = tool.handler(context, validated)
            status = "done"
            text = result.text
            detail.update(dict(result.detail))

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

    duration_ms = int((time.perf_counter() - started) * 1000)
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
    )


def describe_steps(steps: Sequence[AgentStep]) -> str:
    """Human-facing one line per step, used by the CLI."""
    lines = []
    for step in steps:
        lines.append(f"  {step.index}. {step.tool} [{step.status}] {step.duration_ms} ms — {step.summary}")
    return "\n".join(lines)
