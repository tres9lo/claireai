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
from claireai.audit import AuditLog
from claireai.brain import BrainResponse, BrainUnavailable, ToolCall
from claireai.config import Settings
from claireai.permissions import PermissionDenied, Sandbox
from claireai.storage import TaskRepository
from claireai.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


class ScriptedBrain:
    name = "scripted"

    def __init__(self, responses) -> None:  # noqa: ANN001
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def health(self):  # noqa: ANN201
        raise NotImplementedError

    async def generate(self, messages, tools=None) -> BrainResponse:  # noqa: ANN001
        self.calls.append({"messages": list(messages), "tools": tools})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


def test_system_message_names_the_allowed_folders(downloads: Path) -> None:
    message = system_message(Sandbox(roots=(downloads,)))
    assert str(downloads) in message.content
    assert "no write tools" in message.content


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
    assert [entry.action for entry in logged] == ["tool.call", "agent.answer"]
    assert logged[0].tool == "list_folder"
    assert logged[0].status == "done"
    assert logged[0].target == str(downloads)


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
    # The assistant message that proposed the call is echoed, as Ollama requires.
    assert any(message.tool_calls for message in second)


async def test_tool_schemas_are_offered_to_the_model(
    conn: sqlite3.Connection, agent_settings: Settings, downloads: Path
) -> None:
    brain = ScriptedBrain([answer()])
    await run_task(settings=agent_settings, conn=conn, brain=brain, request="hi")

    tools = brain.calls[0]["tools"]
    names = {tool["function"]["name"] for tool in tools}
    assert names == {
        "list_folder",
        "get_file_info",
        "read_text_file",
        "search_files",
        "find_largest_files",
    }


async def test_unknown_tool_is_refused_and_logged(
    conn: sqlite3.Connection, agent_settings: Settings
) -> None:
    brain = ScriptedBrain([tool_call("delete_everything"), answer("I cannot do that.")])

    outcome = await run_task(settings=agent_settings, conn=conn, brain=brain, request="delete it")

    assert outcome.steps[0].status == "skipped"
    assert "unknown tool" in outcome.steps[0].result_text
    logged = AuditLog(conn).for_task(outcome.task_id)
    assert logged[0].status == "skipped"


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
    assert "no approval flow" in outcome.steps[0].result_text


async def test_a_needs_approval_tool_runs_when_approval_exists(
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
    brain = ScriptedBrain([tool_call("move_file"), answer("Moved.")])

    outcome = await run_task(
        settings=agent_settings,
        conn=conn,
        brain=brain,
        request="move it",
        registry=registry,
        approval_available=True,
    )
    assert outcome.steps[0].status == "done"


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
    logged = AuditLog(conn).for_task(outcome.task_id)
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
    assert [entry.action for entry in logged] == [
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
