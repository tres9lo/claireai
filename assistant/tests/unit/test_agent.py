"""The agent loop: validation, permissions, logging and the step limit.

The brain is scripted, so these tests assert the runtime's decisions — not model
quality. The real 1.5B run is recorded in PROGRESS.md.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel

from claireai.agent import AgentEmptyReply, AgentStepLimit, run_task, system_message
from claireai.approval import ApprovalDecision
from claireai.audit import AuditLog
from claireai.brain import BrainResponse, BrainResponseError, BrainStopped, BrainUnavailable, ToolCall
from claireai.config import Settings
from claireai.permissions import PermissionDenied, Sandbox
from claireai.presence import PresenceStream
from claireai.runtime_control import RuntimeControl
from claireai.storage import TaskRepository
from claireai.tools import default_registry
from claireai.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


class ScriptedBrain:
    name = "scripted"

    def __init__(self, responses) -> None:  # noqa: ANN001
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def health(self):  # noqa: ANN201
        raise NotImplementedError

    async def generate(self, messages, tools=None, *, temperature=None) -> BrainResponse:  # noqa: ANN001
        self.calls.append({"messages": list(messages), "tools": tools, "temperature": temperature})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def trail(logged):  # noqa: ANN001, ANN201
    """Audit rows after route selection.

    ``agent.route`` records which tools the model was offered, before the task
    starts. Assertions about what then ran read the trail after that row, so
    adding routing never weakens them to a substring match.
    """
    return [entry for entry in logged if entry.action != "agent.route"]


def tool_call(name: str, **arguments) -> BrainResponse:
    return BrainResponse(content="", model="scripted", tool_calls=(ToolCall(name=name, arguments=arguments),))


def answer(text: str = "Here is the answer.") -> BrainResponse:
    return BrainResponse(content=text, model="scripted")


@pytest.fixture()
def downloads(tmp_path: Path) -> Path:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    (folder / "big.iso").write_bytes(b"z" * 4000)
    (folder / "small.txt").write_text("tiny", encoding="utf-8")
    return folder


@pytest.fixture()
def agent_settings(tmp_path: Path, downloads: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", allowed_folders=[downloads])


class NoArgs(BaseModel):
    pass


class FixedBroker:
    """Stand-in for the human: records the request and answers the same way."""

    name = "fixed"

    def __init__(self, approved: bool, reason: str = "") -> None:
        self.approved = approved
        self.reason = reason
        self.requests = []

    async def decide(self, request):  # noqa: ANN001, ANN201
        self.requests.append(request)
        return ApprovalDecision(approved=self.approved, reason=self.reason, decided_by="test")


def test_system_message_names_the_allowed_folders(downloads: Path) -> None:
    message = system_message(Sandbox(roots=(downloads,)))
    assert str(downloads) in message.content
    assert "approval" in message.content


@pytest.mark.asyncio
async def test_owner_stop_cancels_the_task_and_logs_it_as_skipped(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain([BrainStopped("stopped by owner")])

    with pytest.raises(BrainStopped, match="stopped by owner"):
        await run_task(
            settings=agent_settings,
            conn=conn,
            brain=brain,
            request="List my files",
        )

    task = TaskRepository(conn).recent(1)[0]
    assert task.status == "cancelled"
    rows = AuditLog(conn).for_task(task.id)
    assert rows[-1].action == "agent.run"
    assert rows[-1].status == "skipped"


def test_the_system_prompt_tells_the_model_to_check_paths_before_writing(
    downloads: Path,
) -> None:
    # The refusal is only useful if the model knows a check exists. Measured:
    # after being refused for an invented path, the 1.5B stopped instead of
    # correcting, because nothing told it how to get the fact it was missing.
    content = system_message(Sandbox(roots=(downloads,))).content
    assert "never invent an absolute path" in content
    assert "list_folder" in content
    assert "get_file_info" in content


async def test_one_tool_call_then_an_answer(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads)), answer("Two files.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="what is in Downloads?"
    )

    assert outcome.content == "Two files."
    assert len(outcome.steps) == 1
    assert outcome.steps[0].tool == "list_folder"
    assert outcome.steps[0].status == "done"
    assert "big.iso" in outcome.steps[0].result_text

    task = TaskRepository(conn).get(outcome.task_id)
    assert task is not None
    assert task.status == "succeeded"
    assert task.kind == "agent"
    assert task.current_step == 1

    logged = AuditLog(conn).for_task(outcome.task_id)
    assert logged[0].action == "agent.route"
    assert logged[0].target == "list"
    rows = trail(logged)
    assert [entry.action for entry in rows] == ["tool.call", "agent.answer"]
    assert rows[0].tool == "list_folder"
    assert rows[0].status == "done"
    assert rows[0].target == str(downloads)


async def test_the_tool_result_is_fed_back_to_the_model(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads)), answer()])

    await run_task(settings=agent_settings, conn=conn, brain=brain, request="list it")

    # Two model calls: the second one must carry the tool result back.
    assert len(brain.calls) == 2
    second = brain.calls[1]["messages"]
    assert second[-1].role == "tool"
    assert second[-1].tool_name == "list_folder"
    assert "big.iso" in second[-1].content
    # The assistant message that proposed the call stays in the model history.
    assert any(message.tool_calls for message in second)


async def test_disabled_browser_before_a_proposed_action_fails_the_task(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    """Regression: the guard raised an undefined name, so nothing recorded the failure.

    The model may propose a browser call and the owner may switch browser access
    off between the proposal and the action. That must end as a failed task with
    an audit row, not as a ``NameError`` escaping the ``except BrainError``
    handler with the task stuck in ``running``.
    """
    RuntimeControl(agent_settings.control_state_path).update({"browser": False})
    brain = ScriptedBrain([tool_call("open_url", url="https://example.com")])

    with pytest.raises(BrainResponseError, match="browser access was disabled"):
        await run_task(settings=agent_settings, conn=conn, brain=brain, request="open example.com")

    task = TaskRepository(conn).recent(1)[0]
    assert task.status == "failed"
    assert "browser access was disabled" in (task.error or "")
    rows = AuditLog(conn).for_task(task.id)
    assert rows[-1].action == "agent.run"
    assert rows[-1].status == "failed"


async def test_tool_schemas_are_offered_to_the_model(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    RuntimeControl(agent_settings.control_state_path).update({"browser": True})
    brain = ScriptedBrain([answer()])
    await run_task(settings=agent_settings, conn=conn, brain=brain, request="hi")

    tools = brain.calls[0]["tools"]
    names = {tool["function"]["name"] for tool in tools}
    assert {
        "list_folder",
        "get_file_info",
        "read_text_file",
        "search_files",
        "find_largest_files",
        "create_folder",
        "create_text_file",
        "write_to_file",
        "rename_file",
        "move_file",
        "move_to_trash",
        "open_url",
        "get_page_text",
        "search_web",
        "download_file",
        "identify_image",
        "see_screen",
    } == names


async def test_unknown_tool_is_refused_and_logged(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain([tool_call("delete_everything"), answer("I cannot do that.")])

    outcome = await run_task(settings=agent_settings, conn=conn, brain=brain, request="delete it")

    assert outcome.steps[0].status == "skipped"
    assert "unknown tool" in outcome.steps[0].result_text
    logged = trail(AuditLog(conn).for_task(outcome.task_id))
    assert logged[0].status == "skipped"


async def test_the_unknown_tool_error_names_the_tools_that_were_offered(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    # "show me my downloads" routes to `list`, so the model is shown one tool.
    # Naming the eleven tools it was never shown would send it back to a menu the
    # route deliberately hid.
    brain = ScriptedBrain([tool_call("delete_everything"), answer("I cannot do that.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="show me my downloads"
    )

    text = outcome.steps[0].result_text
    assert "unknown tool" in text
    assert "offered for this request: list_folder" in text
    assert "move_to_trash" not in text
    # The full registry listing is still recorded, just not shown to the model.
    detail = trail(AuditLog(conn).for_task(outcome.task_id))[0].detail or ""
    assert "move_to_trash" in detail
    # Routing narrowed the menu, it did not remove a permission: an offered-out
    # tool is still executed if called anyway.
    assert len(brain.calls) == 2
    assert brain.calls[0]["tools"] and len(brain.calls[0]["tools"]) == 1


async def test_with_routing_off_the_error_lists_every_tool(
    conn: sqlite3.Connection, agent_settings: Settings, tmp_path: Path, downloads: Path
) -> None:
    un_routed = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        allowed_folders=[downloads],
        tool_routing="off",
    )
    brain = ScriptedBrain([tool_call("delete_everything"), answer("no")])

    outcome = await run_task(settings=un_routed, conn=conn, brain=brain, request="delete it")

    assert "list_folder" in outcome.steps[0].result_text
    assert "move_to_trash" in outcome.steps[0].result_text


async def test_invalid_arguments_are_refused_and_the_error_goes_back(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain([tool_call("list_folder"), answer("Sorry, I need a path.")])

    outcome = await run_task(settings=agent_settings, conn=conn, brain=brain, request="list")

    assert outcome.steps[0].status == "failed"
    assert "invalid arguments" in outcome.steps[0].result_text
    fed_back = brain.calls[1]["messages"][-1]
    assert fed_back.role == "tool"
    assert "invalid arguments" in fed_back.content


async def test_unexpected_argument_keys_are_refused(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads), all=True), answer()])
    outcome = await run_task(settings=agent_settings, conn=conn, brain=brain, request="list")
    assert outcome.steps[0].status == "failed"


async def test_a_dangerous_tool_is_refused_without_running_it(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    calls: list[str] = []

    def handler(context: ToolContext, arguments: NoArgs) -> ToolResult:
        calls.append("ran")
        return ToolResult(text="should not happen")

    registry = ToolRegistry(
        [
            Tool(
                name="wipe_disk",
                description="Pretend",
                risk="dangerous",
                args_model=NoArgs,
                handler=handler,
            )
        ]
    )
    brain = ScriptedBrain([tool_call("wipe_disk"), answer("I need approval for that.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="wipe it", registry=registry
    )

    assert calls == []
    assert outcome.steps[0].status == "skipped"
    assert "no approval channel" in outcome.steps[0].result_text


async def test_a_needs_approval_tool_runs_after_an_approving_broker(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    def handler(context: ToolContext, arguments: NoArgs) -> ToolResult:
        return ToolResult(text="moved")

    registry = ToolRegistry(
        [
            Tool(
                name="move_file",
                description="Pretend",
                risk="needs_approval",
                args_model=NoArgs,
                handler=handler,
            )
        ]
    )
    broker = FixedBroker(approved=True)
    brain = ScriptedBrain([tool_call("move_file"), answer("Moved.")])

    outcome = await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="move it",
        registry=registry,
        approvals=broker,
    )
    assert outcome.steps[0].status == "done"
    assert len(broker.requests) == 1
    assert broker.requests[0].tool == "move_file"
    assert broker.requests[0].ref == outcome.steps[0].approval_ref


async def test_dry_run_reports_without_executing(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads)), answer("That is the plan.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="list", dry_run=True
    )

    assert outcome.steps[0].status == "skipped"
    assert outcome.steps[0].result_text.startswith("[dry run] would call list_folder")
    assert "big.iso" not in outcome.steps[0].result_text
    logged = trail(AuditLog(conn).for_task(outcome.task_id))
    assert logged[0].status == "skipped"
    assert '"dry_run": true' in (logged[0].detail or "")


async def test_a_path_outside_the_allow_list_is_refused(
    conn: sqlite3.Connection, agent_settings: Settings, tmp_path: Path
) -> None:
    secret = tmp_path / "outside.txt"
    secret.write_text("private", encoding="utf-8")
    brain = ScriptedBrain([tool_call("read_text_file", path=str(secret)), answer()])

    outcome = await run_task(settings=agent_settings, conn=conn, brain=brain, request="read it")

    assert outcome.steps[0].status == "skipped"
    assert "permission denied" in outcome.steps[0].result_text
    assert "private" not in outcome.steps[0].result_text
    # The refusal must also say where the model *can* go, or a corrected
    # attempt has nothing new to work with.
    assert str(agent_settings.allowed_folders[0]) in outcome.steps[0].result_text
    fed_back = brain.calls[1]["messages"][-1]
    assert fed_back.role == "tool"
    assert "Allowed folders" in fed_back.content


async def test_permission_denied_never_reaches_the_handler(
    conn: sqlite3.Connection, agent_settings: Settings, tmp_path: Path
) -> None:
    touched: list[Path] = []

    def handler(context: ToolContext, arguments: NoArgs) -> ToolResult:
        touched.append(context.sandbox.resolve(str(tmp_path)))
        return ToolResult(text="leak")

    registry = ToolRegistry(
        [Tool(name="peek", description="Pretend", risk="safe", args_model=NoArgs, handler=handler)]
    )
    brain = ScriptedBrain([tool_call("peek"), answer()])
    await run_task(settings=agent_settings, conn=conn, brain=brain, request="peek", registry=registry)
    assert touched == []


async def test_step_limit_stops_the_loop_and_fails_the_task(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain(
        [tool_call("list_folder", path=str(downloads)) for _ in range(5)]
    )

    with pytest.raises(AgentStepLimit, match="step limit"):
        await run_task(
            settings=agent_settings,
            conn=conn,
            brain=brain,
            request="loop forever",
            max_steps=2,
        )

    task_id = TaskRepository(conn).recent(1)[0].id
    task = TaskRepository(conn).get(task_id)
    assert task is not None
    assert task.status == "failed"
    assert "step limit" in (task.error or "")

    logged = AuditLog(conn).for_task(task_id)
    assert [entry.action for entry in trail(logged)] == [
        "tool.call",
        "tool.call",
        "agent.step_limit",
    ]
    assert logged[-1].status == "failed"


async def test_default_step_limit_comes_from_settings(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    assert agent_settings.agent_max_steps == 8
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads)) for _ in range(9)])
    with pytest.raises(AgentStepLimit, match=r"\(8\)"):
        await run_task(settings=agent_settings, conn=conn, brain=brain, request="loop")


async def test_an_empty_reply_is_a_failure_not_an_answer(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    # Real behaviour of qwen2.5:1.5b: sometimes it returns "" with no tool call.
    brain = ScriptedBrain([BrainResponse(content="   ", model="scripted")])

    with pytest.raises(AgentEmptyReply, match="empty reply"):
        await run_task(settings=agent_settings, conn=conn, brain=brain, request="list it")

    task = TaskRepository(conn).recent(1)[0]
    assert task.status == "failed"
    logged = AuditLog(conn).for_task(task.id)
    assert logged[-1].status == "failed"
    assert '"reason": "empty reply"' in (logged[-1].detail or "")


def _empty_reply(**kwargs) -> BrainResponse:
    """An empty reply carrying the signature of a generated-but-swallowed call.

    ``completion_tokens`` is what the earlier runtime reported: 23-40 tokens
    generated, nothing delivered. It is also the field that decides whether a
    retry is worth attempting.
    """
    return BrainResponse(content="", model="scripted", **kwargs)


async def test_the_temperature_is_low_while_deciding_and_normal_once_decided(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    # Dynamic decoding: repeatable while the model is choosing a tool, natural
    # once it has real data to phrase. Both values are configurable.
    brain = ScriptedBrain(
        [tool_call("list_folder", path=str(downloads)), answer("Two files.")]
    )

    await run_task(settings=agent_settings, conn=conn, brain=brain, request="list it")

    assert agent_settings.temperature_tools < agent_settings.temperature
    assert brain.calls[0]["temperature"] == agent_settings.temperature_tools
    assert brain.calls[1]["temperature"] == agent_settings.temperature


async def test_an_empty_reply_with_generated_tokens_gets_one_retry(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain(
        [_empty_reply(completion_tokens=32), answer("Recovered on the second attempt.")]
    )

    outcome = await run_task(settings=agent_settings, conn=conn, brain=brain, request="list it")

    assert outcome.content == "Recovered on the second attempt."
    assert len(brain.calls) == 2
    rows = trail(AuditLog(conn).for_task(outcome.task_id))
    assert [entry.action for entry in rows] == ["agent.retry", "agent.answer"]
    retry = rows[0]
    assert retry.status == "skipped"
    assert retry.target == "scripted"
    assert '"reason": "empty reply"' in (retry.detail or "")
    assert TaskRepository(conn).get(outcome.task_id).status == "succeeded"


async def test_the_empty_reply_retry_is_bounded_to_one_attempt(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain(
        [_empty_reply(completion_tokens=32), _empty_reply(completion_tokens=32)]
    )

    with pytest.raises(AgentEmptyReply, match="empty reply"):
        await run_task(settings=agent_settings, conn=conn, brain=brain, request="list it")

    # Two attempts, never a third: a model that keeps emitting nothing must fail
    # honestly instead of burning the step budget on re-rolls.
    assert len(brain.calls) == 2
    rows = trail(AuditLog(conn).for_task(TaskRepository(conn).recent(1)[0].id))
    assert sum(1 for entry in rows if entry.action == "agent.retry") == 1


async def test_no_retry_when_nothing_was_generated(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain([_empty_reply()])

    with pytest.raises(AgentEmptyReply, match="empty reply"):
        await run_task(settings=agent_settings, conn=conn, brain=brain, request="list it")

    assert len(brain.calls) == 1


async def test_no_retry_when_the_decoding_is_pinned(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    # Temperature 0 redraws the same messages to the same answer, so a retry
    # would spend a full prompt evaluation to reach the same empty reply.
    pinned = agent_settings.model_copy(update={"temperature": 0.0, "temperature_tools": 0.0})
    brain = ScriptedBrain([_empty_reply(completion_tokens=32)])

    with pytest.raises(AgentEmptyReply, match="empty reply"):
        await run_task(settings=pinned, conn=conn, brain=brain, request="list it")

    assert len(brain.calls) == 1
    assert brain.calls[0]["temperature"] == 0.0


async def test_a_brain_failure_fails_the_task(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain([BrainUnavailable("runtime down")])

    with pytest.raises(BrainUnavailable):
        await run_task(settings=agent_settings, conn=conn, brain=brain, request="hello")

    task = TaskRepository(conn).recent(1)[0]
    assert task.status == "failed"
    assert "runtime down" in (task.error or "")
    logged = AuditLog(conn).for_task(task.id)
    assert logged[-1].action == "agent.run"
    assert logged[-1].status == "failed"


async def test_request_is_redacted_before_storage(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain([answer()])
    await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="my api key is sk-abcdefghijklmnopqrstuvwxyz123456",
    )
    task = TaskRepository(conn).recent(1)[0]
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in task.request


async def test_empty_allow_list_refuses_file_access(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    brain = ScriptedBrain([tool_call("list_folder", path=str(tmp_path)), answer("I cannot read anything.")])

    outcome = await run_task(settings=settings, conn=conn, brain=brain, request="list")

    assert outcome.steps[0].status == "skipped"
    assert "no folders are allow-listed" in outcome.steps[0].result_text


def test_sandbox_and_permission_denied_are_exported_where_expected() -> None:
    # Guards against a future refactor moving the permission type silently.
    assert PermissionDenied.__module__ == "claireai.permissions"


# --- Week 3: approvals --------------------------------------------------------


async def test_an_approval_request_carries_the_exact_target(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    new_folder = downloads / "reports"
    broker = FixedBroker(approved=True)
    # A write needs a read first; the gate is what makes that true here.
    brain = ScriptedBrain(
        [
            tool_call("list_folder", path=str(downloads)),
            tool_call("create_folder", path=str(new_folder)),
            answer("Created."),
        ]
    )

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="make a folder", approvals=broker
    )

    assert outcome.steps[0].tool == "list_folder"
    assert outcome.steps[1].status == "done"
    assert new_folder.is_dir()
    request = broker.requests[0]
    assert request.target == str(new_folder)
    assert request.arguments["path"] == str(new_folder)
    assert request.risk == "needs_approval"
    assert request.task_id == outcome.task_id


async def test_a_denied_approval_is_logged_and_the_tool_never_runs(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    calls: list[str] = []

    def handler(context: ToolContext, arguments: NoArgs) -> ToolResult:
        calls.append("ran")
        return ToolResult(text="should not happen")

    registry = ToolRegistry(
        [Tool(name="wipe", description="Pretend", risk="needs_approval", args_model=NoArgs, handler=handler)]
    )
    broker = FixedBroker(approved=False, reason="not this time")
    brain = ScriptedBrain([tool_call("wipe"), answer("Understood, I will not.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="wipe it", registry=registry, approvals=broker
    )

    assert calls == []
    assert outcome.steps[0].status == "skipped"
    assert "did not approve" in outcome.steps[0].result_text
    assert "not this time" in outcome.steps[0].result_text

    logged = trail(AuditLog(conn).for_task(outcome.task_id))
    assert [entry.action for entry in logged] == ["approval.decision", "tool.call", "agent.answer"]
    assert logged[0].status == "skipped"
    assert logged[0].approval_ref == outcome.steps[0].approval_ref
    assert logged[1].approval_ref == logged[0].approval_ref
    assert logged[1].status == "skipped"


async def test_an_approved_action_records_the_same_reference_twice(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    broker = FixedBroker(approved=True)
    brain = ScriptedBrain(
        [
            tool_call("list_folder", path=str(downloads)),
            tool_call("create_folder", path=str(downloads / "x")),
            answer("Done."),
        ]
    )
    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="make x", approvals=broker
    )

    logged = trail(AuditLog(conn).for_task(outcome.task_id))
    decision = next(row for row in logged if row.action == "approval.decision")
    tool_row = next(
        row for row in logged if row.action == "tool.call" and row.tool == "create_folder"
    )
    assert decision.action == "approval.decision" and decision.status == "done"
    assert tool_row.action == "tool.call" and tool_row.status == "done"
    assert decision.approval_ref == tool_row.approval_ref
    assert '"approved": true' in (decision.detail or "")


async def test_a_write_request_is_grounded_in_a_real_listing_first(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    # Measured: qwen2.5:1.5b goes straight to move_file and invents a path, and
    # a refusal afterwards makes it answer instead of retry. So the runtime reads
    # the allowed folders *before* the model speaks, at no extra model call.
    target = downloads / "reports"
    broker = FixedBroker(approved=True)
    brain = ScriptedBrain([tool_call("create_folder", path=str(target)), answer("Created.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="make a folder", approvals=broker
    )

    # Two calls, which is the floor here: propose the tool, then answer. A gate
    # that refused the write would make it three.
    assert len(brain.calls) == 2
    system = brain.calls[0]["messages"][0]
    assert system.role == "system"
    assert "What those folders contain right now" in system.content
    assert "big.iso" in system.content  # a real entry, not a guess

    assert [step.tool for step in outcome.steps] == ["create_folder"]
    assert outcome.steps[0].status == "done"
    assert target.is_dir()


async def test_a_read_request_pays_nothing_for_that_listing(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    # Only routes that can write get the context: reads must not pay prompt
    # tokens (and a read is not a write target that needs checking).
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads)), answer("Two files.")])

    await run_task(settings=agent_settings, conn=conn, brain=brain, request="what is in my downloads")

    system = brain.calls[0]["messages"][0]
    assert "What those folders contain right now" not in system.content


async def test_a_write_with_nothing_read_is_refused_when_no_listing_is_possible(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    # Fallback when the runtime cannot list anything (no list_folder tool): the
    # write is refused rather than run blind, and a read by the model unblocks it.
    ran: list[str] = []

    def probe(context: ToolContext, arguments: NoArgs) -> ToolResult:
        ran.append("probe")
        return ToolResult(text="checked")

    def writer(context: ToolContext, arguments: NoArgs) -> ToolResult:
        ran.append("write")
        return ToolResult(text="wrote")

    registry = ToolRegistry(
        [
            Tool(name="get_file_info", description="Check", risk="safe", args_model=NoArgs, handler=probe),
            Tool(
                name="create_folder",
                description="Create",
                risk="needs_approval",
                args_model=NoArgs,
                handler=writer,
                mutates=True,
            ),
        ]
    )
    broker = FixedBroker(approved=True)
    brain = ScriptedBrain(
        [
            tool_call("create_folder"),
            tool_call("get_file_info"),
            tool_call("create_folder"),
            answer("Done."),
        ]
    )

    outcome = await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="make a folder",
        registry=registry,
        approvals=broker,
    )

    refused = outcome.steps[0]
    assert refused.status == "skipped"
    assert "nothing has been read in this task yet" in refused.result_text
    assert refused.approval_ref is None  # the human was never asked for it
    assert ran == ["probe", "write"]  # the write never ran blind
    assert [step.tool for step in outcome.steps] == ["create_folder", "get_file_info", "create_folder"]
    assert outcome.steps[2].status == "done"
    assert len(broker.requests) == 1

    rows = trail(AuditLog(conn).for_task(outcome.task_id))
    reason = next(row for row in rows if row.action == "tool.call" and row.status == "skipped")
    assert '"reason": "verify_first"' in (reason.detail or "")


async def test_validation_happens_before_asking_a_human(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    broker = FixedBroker(approved=True)
    brain = ScriptedBrain([tool_call("create_folder"), answer("Need a path.")])

    outcome = await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="make it", approvals=broker
    )

    assert broker.requests == []  # nobody is asked to approve a malformed call
    assert outcome.steps[0].status == "failed"


async def test_dry_run_never_asks_for_approval(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    broker = FixedBroker(approved=True)
    target = downloads / "dry-run-folder"
    brain = ScriptedBrain([tool_call("create_folder", path=str(target)), answer("Plan only.")])

    outcome = await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="make a folder",
        approvals=broker,
        dry_run=True,
    )

    assert broker.requests == []
    assert outcome.steps[0].status == "skipped"
    assert outcome.steps[0].result_text.startswith("[dry run]")
    assert not target.exists()


async def test_the_denial_is_fed_back_to_the_model(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    broker = FixedBroker(approved=False, reason="no")
    brain = ScriptedBrain(
        [
            tool_call("list_folder", path=str(downloads)),
            tool_call("create_folder", path=str(downloads / "y")),
            answer("Not approved."),
        ]
    )
    await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="make y", approvals=broker
    )

    fed_back = brain.calls[2]["messages"][-1]
    assert fed_back.role == "tool"
    assert "did not approve" in fed_back.content


async def test_presence_follows_the_run_and_ends_idle(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    stream = PresenceStream()
    seen: list[str] = []
    brain = ScriptedBrain([tool_call("list_folder", path=str(downloads)), answer("Two files.")])

    # PresenceStream has no subscription API: poll seq between runs is enough to
    # assert ordering, and the reporter hook records every state change.
    stream = PresenceStream(reporter=lambda event: seen.append(event.state))
    await run_task(
        settings=agent_settings, conn=conn, brain=brain, request="list", presence=stream
    )

    assert seen[0] == "thinking"
    assert "tool_running" in seen
    assert seen[-1] == "idle"
    assert stream.snapshot().seq >= 3


async def test_presence_reports_waiting_for_approval(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    seen: list[str] = []
    stream = PresenceStream(reporter=lambda event: seen.append(event.state))
    brain = ScriptedBrain(
        [
            tool_call("list_folder", path=str(downloads)),
            tool_call("create_folder", path=str(downloads / "z")),
            answer("Done."),
        ]
    )
    await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="make z",
        approvals=FixedBroker(approved=True),
        presence=stream,
    )
    assert "awaiting_approval" in seen


async def test_routing_hides_tools_but_never_blocks_one(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    """The safety property of routing: it changes what is offered, not what is
    allowed. A tool filtered out of the route still runs when the model asks."""
    touched: list[str] = []

    def peek_handler(_context, _args):  # noqa: ANN001, ANN202
        touched.append("peek")
        return ToolResult(text="peeked")

    registry = ToolRegistry(
        [
            Tool(name="list_folder", description="List", risk="safe", args_model=NoArgs, handler=peek_handler),
            Tool(name="peek", description="Peek", risk="safe", args_model=NoArgs, handler=peek_handler),
        ]
    )
    # "list files in downloads" routes to list_folder only, so `peek` is hidden.
    brain = ScriptedBrain([tool_call("peek"), answer("Peeked.")])
    outcome = await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="list the files in downloads",
        registry=registry,
    )

    assert outcome.route == "list"
    assert len(brain.calls[0]["tools"]) == 1  # only list_folder was offered
    assert touched == ["peek"]  # ...and the hidden tool still ran
    assert outcome.steps[0].status == "done"


async def test_routing_can_be_switched_off(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    settings = agent_settings.model_copy(update={"tool_routing": "off"})
    brain = ScriptedBrain([answer("Nothing to do.")])

    outcome = await run_task(
        settings=settings, conn=conn, brain=brain, request="what is in Downloads?"
    )

    assert outcome.route is None
    assert len(brain.calls[0]["tools"]) == len(default_registry())
    assert all(entry.action != "agent.route" for entry in AuditLog(conn).for_task(outcome.task_id))


async def test_the_route_row_records_the_prompt_saving(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    import json

    brain = ScriptedBrain([tool_call("find_largest_files", path=str(downloads)), answer("big.iso")])
    outcome = await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="List the biggest files in my Downloads folder",
    )

    row = AuditLog(conn).for_task(outcome.task_id)[0]
    assert row.action == "agent.route"
    assert row.target == "largest"
    detail = json.loads(row.detail or "{}")
    assert detail["narrowed"] is True
    assert detail["tools_total"] == len(default_registry())
    assert detail["schema_chars"] < detail["schema_chars_all"]
