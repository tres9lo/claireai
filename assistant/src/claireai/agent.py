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
from claireai.brain import Brain, BrainError, BrainResponse, BrainStopped, Message, ToolCall
from claireai.config import Settings
from claireai.permissions import PermissionDenied, Sandbox, check_risk, requires_approval
from claireai.presence import PresenceStream
from claireai.routing import Route, route as choose_route, schema_bytes
from claireai.runtime_control import RuntimeControl
from claireai.storage import TaskRepository, transaction
from claireai.tools import ToolError, ToolRegistry, UnknownTool
from claireai.tools.base import ToolContext
from claireai.tools.browser import BrowserSession

SYSTEM_PROMPT = (
    "You are ClaireAI, a friendly local personal assistant on the owner's own computer. "
    "You are assisting the owner named {owner_name}; address them naturally, without overusing their name. "
    "Use light humor when it fits or when asked, and keep status reports structured: "
    "completed, blocked, and next. When asked for ideas, offer practical options and state assumptions. "
    "Never claim consciousness or claim an action succeeded "
    "until a tool result confirms it. "
    "You can read files with the provided tools, and only inside the allowed folders. "
    "Call a tool when the request needs real data; do not guess file names or sizes. "
    "For the biggest or largest files, call find_largest_files and answer only "
    "from the sizes it returns; never invent a size. "
    "Lines starting with 'dir' are folders, not files; only 'file' lines have sizes. "
    "Write tools exist: they create, write, rename or move files, and every one needs "
    "the owner's approval before it runs. Propose them; never assume approval was given. "
    "Start every path with one of the allowed folders; never invent an absolute path. "
    "The current contents of those folders are listed for you with each request: use "
    "those real names, and call list_folder or get_file_info for anything not shown. "
    "Nothing you have deletes permanently: move_to_trash moves a file to a recoverable bin. "
    "When you have the answer, reply with a short final answer and no tool call. "
    "If a tool returns an error or is denied, explain it or correct the arguments."
)

MAX_TOOL_TEXT = 4000
# How many entries the runtime lists when it grounds a write. Enough to hold the
# names a request mentions, short enough not to cost real prompt time.
MAX_LISTED = 20


def _retry_worthwhile(response: BrainResponse, temperature: float) -> bool:
    """Is one more attempt at the same conversation likely to differ?

    The empty replies we have measured come from a *generated but malformed*
    tool call: a runtime reports generated tokens while delivering neither
    text nor a call, so the tokens exist and a fresh sample can differ. Nothing
    generated means there is nothing to re-roll, and a temperature of 0 means
    the same input yields the same empty reply again — in both cases a retry
    would spend a full prompt evaluation (seconds of CPU on this hardware) to
    reach a known answer, so it is refused here rather than silently.
    """
    if not response.completion_tokens:
        return False
    return temperature != 0


class AgentError(Exception):
    """The agent could not finish the request."""


class AgentStepLimit(AgentError):
    """The model kept calling tools past the configured step limit."""


class AgentEmptyReply(AgentError):
    """The model returned neither text nor a tool call.

    An empty assistant message is not a valid answer, and treating it as one
    shows the owner a blank screen while the task claims success.
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
    # True when the runtime read the folder on the model's behalf because a
    # write arrived with nothing read yet (see ``_discovery``).
    discovered: bool = False


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


def system_message(sandbox: Sandbox, owner_name: str = "Bikorimana Ange Tresor") -> Message:
    prompt = SYSTEM_PROMPT.format(owner_name=owner_name)
    return Message(role="system", content=f"{prompt} Allowed folders: {sandbox.describe()}.")


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

    controls = RuntimeControl(settings.control_state_path)
    control_state = controls.read()
    if control_state["shutdown"]:
        raise BrainStopped("ClaireAI is stopped. Resume it from the local control panel.")
    registry = registry or default_registry(browser_enabled=control_state["browser"])
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

    # A registry with no read tool at all starts verified: there is nothing to
    # require, and a gate that can never be satisfied would block every write.
    can_verify = any(registry.get(name).risk == "safe" for name in registry.names())
    verified = not can_verify

    # For a request that can write, read the allowed folders *before the model
    # speaks*. Small models can go straight to a write without grounding.
    # Grounding it up front costs no model call and removes the guess.
    context_note = ""
    if can_verify and plan is not None and any(
        registry.get(name).mutates for name in plan.tools
    ):
        listing, read_ok = _discovery(sandbox, registry, {})
        if read_ok:
            context_note = f"\n\nWhat those folders contain right now:\n{listing}"
            verified = True

    messages: list[Message] = [
        Message(role="system", content=system_message(sandbox, settings.owner_name).content + context_note),
        Message(role="user", content=request),
    ]
    steps: list[AgentStep] = []
    browser = BrowserSession() if "open_url" in registry else None
    started = time.perf_counter()

    try:
        for step_number in range(1, step_limit + 1):
            if controls.read()["shutdown"]:
                raise BrainStopped("ClaireAI was stopped from the local control panel.")
            # Dynamic decoding. Until a step has succeeded the model is choosing
            # what to do, so it gets the low temperature; once it has real data
            # it is phrasing an answer, so it gets the configured one.
            decided = any(step.status == "done" for step in steps)
            temperature = settings.temperature if decided else settings.tool_temperature
            response = await brain.generate(messages, tools=schemas, temperature=temperature)

            if not response.tool_calls and not response.content.strip() and _retry_worthwhile(
                response, temperature
            ):
                # One bounded second attempt at the same messages, recorded as
                # its own audit row: the first attempt is a real event the owner
                # should be able to see, and a model that only ever recovers by
                # retrying must not look like one that answered first time.
                with transaction(conn):
                    audit.record(
                        task_id=task.id,
                        device=device,
                        action="agent.retry",
                        status="skipped",
                        target=response.model,
                        detail={
                            "reason": "empty reply",
                            "step": step_number,
                            "completion_tokens": response.completion_tokens,
                        },
                    )
                if presence:
                    presence.emit("thinking", task_id=task.id)
                response = await brain.generate(
                    messages, tools=schemas, temperature=temperature
                )

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
                state = controls.read()
                if state["shutdown"]:
                    raise BrainStopped("ClaireAI was stopped before the proposed action could run.")
                if call.name in {"open_url", "get_page_text", "search_web", "download_file"} and not state["browser"]:
                    raise BrainResponseError("browser access was disabled before the action could run")
                step = await _execute(
                    call=call,
                    index=len(steps) + 1,
                    task_id=task.id,
                    device=device,
                    shown=plan.tools if plan else registry.names(),
                    registry=registry,
                    sandbox=sandbox,
                    data_dir=data_dir,
                    approvals=broker,
                    presence=presence,
                    audit=audit,
                    conn=conn,
                    dry_run=dry_run,
                    verified=verified,
                    browser=browser,
                )
                steps.append(step)
                if step.discovered or (
                    step.status == "done"
                    and step.tool in registry
                    and registry.get(step.tool).risk == "safe"
                ):
                    # A read happened, whether the model asked for it or the
                    # runtime did it on the model's behalf.
                    verified = True
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
    finally:
        if browser is not None:
            browser.close()


def _discovery(
    sandbox: Sandbox, registry: ToolRegistry, arguments: Mapping[str, Any]
) -> tuple[str, bool]:
    """Read a folder the model should have looked at, and describe it.

    Called only when a write arrives with nothing read yet. The runtime does the
    read instead of asking again: small models can go straight to ``move_file``
    without grounding. A local folder listing costs no model call, and putting the real
    names in the transcript is the only way this model gets them.

    Returns the text to feed back and whether a read actually happened.
    """
    if "list_folder" not in registry:
        return "", False

    folders: list[Path] = []
    for key in ("source", "path", "destination"):
        raw = arguments.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            resolved = sandbox.resolve(raw)
        except PermissionDenied:
            continue
        folders.append(resolved if resolved.is_dir() else resolved.parent)
        break
    if not folders:
        # The named path is not reachable at all. Show what *is* reachable,
        # which is exactly the fact a model that invented /Users/... lacks.
        folders = list(sandbox.roots)

    # Absolute paths, one per line, each on its own: a small model copies a
    # whole line and gets a working path. Fed the human-readable listing
    # ("file 13 B quarterly-report.pdf") it pasted that fragment in as a name.
    parts: list[str] = []
    for folder in folders[:2]:
        try:
            entries = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name))[:MAX_LISTED]
        except OSError:
            continue
        block = [f"{folder} contains:"]
        for entry in entries:
            if entry.is_dir():
                block.append(f"  {entry}  (folder)")
            else:
                try:
                    block.append(f"  {entry}  (file, {entry.stat().st_size} bytes)")
                except OSError:
                    block.append(f"  {entry}  (file)")
        try:
            total = sum(1 for _ in folder.iterdir())
        except OSError:
            total = len(entries)
        if total > len(entries):
            block.append(f"  ... and {total - len(entries)} more")
        parts.append("\n".join(block))
    return "\n".join(parts), bool(parts)


async def _execute(
    *,
    call: ToolCall,
    index: int,
    task_id: str,
    device: str,
    shown: Sequence[str],
    registry: ToolRegistry,
    sandbox: Sandbox,
    data_dir: Path,
    approvals: ApprovalBroker,
    presence: PresenceStream | None,
    audit: AuditLog,
    conn: sqlite3.Connection,
    dry_run: bool,
    verified: bool,
    browser: Any = None,
) -> AgentStep:
    """Validate, authorize, run and log one proposed call."""
    started = time.perf_counter()
    arguments = dict(call.arguments)
    target = arguments.get("path") if isinstance(arguments.get("path"), str) else None
    if target is None and isinstance(arguments.get("source"), str):
        target = arguments["source"]
    context = ToolContext(
        sandbox=sandbox, dry_run=dry_run, data_dir=data_dir, browser=browser
    )

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
        elif tool.mutates and not verified:
            # Nothing has been read in this task, so this target has never been
            # checked. Offering the read tools was measured not to be enough, so
            # the runtime reads the folder itself and hands back real names.
            status = "skipped"
            detail["reason"] = "verify_first"
            found, did_read = _discovery(sandbox, registry, arguments)
            detail["discovered"] = did_read
            if did_read:
                # Factual, with no instruction the model can mistake for an
                # answer: it copied "call move_file again" back as its reply.
                text = f"{tool.name} did not run. The folder really contains:\n{found}"
            else:
                text = (
                    "nothing has been read in this task yet, so this target is "
                    "unchecked; call list_folder or get_file_info on the allowed "
                    f"folder first, then call {tool.name} again"
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
                approval_ref=None,
                presence=presence,
                discovered=did_read,
            )
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
        # Name only the tools this request was offered. The registry's own list
        # names every tool, including ones routing deliberately hid, so feeding
        # it back hands the model a menu it was never shown and invites it to
        # try them anyway. This is a display choice: a tool outside `shown` that
        # the model calls anyway is still validated and executed normally.
        # The full registry listing stays in the audit row, not in the reply.
        detail["error"] = str(exc)
        text = (
            f"error: unknown tool {call.name!r}; "
            f"offered for this request: {', '.join(shown) or '(none)'}"
        )
    except PermissionDenied as exc:
        status = "skipped"
        # Naming the allowed folders in the refusal gives the model the one fact
        # its next attempt needs. Measured: after a bare refusal the 1.5B had
        # invented a path outside the sandbox, been refused, and then stopped
        # instead of correcting itself — the correction was never made cheap.
        text = f"permission denied: {exc}. Allowed folders: {sandbox.describe()}"
        detail["reason"] = "permission_denied"
    except ToolError as exc:
        status = "failed"
        text = f"error: {exc}"
        detail["error"] = str(exc)
        if "does not exist" in str(exc):
            # A missing path is the failure the model has to recover from, and
            # it does not recover from a bare "does not exist" — it stops. Give
            # it the real names, absolute paths, in the same breath.
            found, did_read = _discovery(sandbox, registry, arguments)
            if did_read:
                text = f"error: {exc}\n{found}"
                detail["discovered"] = True
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
    discovered: bool = False,
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
        discovered=discovered,
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
