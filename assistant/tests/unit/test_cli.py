"""CLI turn tests with a stub brain.

The brain is replaced so these tests assert the task/audit bookkeeping, not the
model's answer. The real Ollama path is exercised by the manual smoke test.
"""

from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from claireai.approval import ApprovalRequest
from claireai.audit import AuditLog
from claireai.brain import BrainHealth, BrainResponse, BrainUnavailable, ToolCall
from claireai.cli import CliApprovalBroker, _handle_turn, ask
from claireai.config import Settings
from claireai.storage import TaskRepository


class StubBrain:
    name = "stub"

    def __init__(self, *, content: str = "stub answer", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[list] = []

    async def health(self) -> BrainHealth:
        return BrainHealth(available=True, model_present=True, models=("stub",), detail="ok")

    async def generate(self, messages, tools=None) -> BrainResponse:
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        return BrainResponse(content=self.content, model="stub-model", prompt_tokens=4, completion_tokens=2)


async def test_successful_turn_creates_task_and_audit_row(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    turn = await ask(settings=settings, conn=conn, brain=StubBrain(content="42"), question="life?")

    task = TaskRepository(conn).get(turn.task_id)
    assert task is not None
    assert task.status == "succeeded"
    assert task.kind == "chat"
    assert task.request == "life?"

    entries = AuditLog(conn).for_task(turn.task_id)
    assert len(entries) == 1
    assert entries[0].action == "brain.generate"
    assert entries[0].status == "done"
    assert entries[0].target == "stub-model"
    assert '"completion_tokens": 2' in (entries[0].detail or "")


async def test_turn_sends_system_prompt_then_question(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    brain = StubBrain()
    await ask(settings=settings, conn=conn, brain=brain, question="hi")
    roles = [message.role for message in brain.calls[0]]
    assert roles == ["system", "user"]
    assert brain.calls[0][1].content == "hi"


async def test_failure_marks_task_failed_and_audits_redacted_error(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    brain = StubBrain(error=BrainUnavailable("cannot reach Ollama (ConnectError)"))
    with pytest.raises(BrainUnavailable):
        await ask(settings=settings, conn=conn, brain=brain, question="hi")

    tasks = TaskRepository(conn)
    failed = tasks.recent(limit=1)[0]
    assert failed.status == "failed"
    assert "cannot reach Ollama" in (failed.error or "")

    entries = AuditLog(conn).for_task(failed.id)
    assert [entry.status for entry in entries] == ["failed"]


async def test_secrets_in_request_are_not_stored(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    await ask(settings=settings, conn=conn, brain=StubBrain(), question="my api_key=sk-live-1234567890abc")
    stored = TaskRepository(conn).recent(limit=1)[0].request
    assert "sk-live-1234567890abc" not in stored
    assert "[REDACTED]" in stored


# --- agent wiring: the CLI must use the tool loop by default ------------------


class ScriptedBrain:
    """Two turns: a tool call, then a final answer."""

    name = "scripted"

    def __init__(self, call: ToolCall, answer: str) -> None:
        self._responses = [
            BrainResponse(content="", model="scripted", tool_calls=(call,)),
            BrainResponse(content=answer, model="scripted"),
        ]
        self.tools_seen: list = []

    async def health(self) -> BrainHealth:
        return BrainHealth(available=True, model_present=True, models=("scripted",))

    async def generate(self, messages, tools=None) -> BrainResponse:
        self.tools_seen.append(tools)
        return self._responses.pop(0)


def _args(**overrides) -> Namespace:  # noqa: ANN003
    base = {"no_tools": False, "dry_run": False, "device": "desktop"}
    base.update(overrides)
    return Namespace(**base)


async def test_cli_uses_the_agent_loop_by_default(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, capsys
) -> None:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    (folder / "a.txt").write_text("hello", encoding="utf-8")
    settings = settings.model_copy(update={"allowed_folders": [folder]})
    brain = ScriptedBrain(ToolCall(name="list_folder", arguments={"path": str(folder)}), "One file.")

    code = await _handle_turn(
        args=_args(), settings=settings, conn=conn, brain=brain, question="list it", database_path=tmp_path / "db"
    )

    assert code == 0
    assert brain.tools_seen[0]  # tools were offered
    out = capsys.readouterr().out
    assert "steps:" in out
    assert "list_folder" in out
    assert "One file." in out
    assert TaskRepository(conn).recent(1)[0].kind == "agent"


async def test_cli_no_tools_keeps_plain_chat(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, capsys
) -> None:
    brain = StubBrain(content="plain answer")

    code = await _handle_turn(
        args=_args(no_tools=True), settings=settings, conn=conn, brain=brain, question="hi", database_path=tmp_path / "db"
    )

    assert code == 0
    assert "plain answer" in capsys.readouterr().out
    assert TaskRepository(conn).recent(1)[0].kind == "chat"


async def test_cli_dry_run_passes_through(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, capsys
) -> None:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    settings = settings.model_copy(update={"allowed_folders": [folder]})
    brain = ScriptedBrain(ToolCall(name="list_folder", arguments={"path": str(folder)}), "Plan only.")

    await _handle_turn(
        args=_args(dry_run=True),
        settings=settings,
        conn=conn,
        brain=brain,
        question="list it",
        database_path=tmp_path / "db",
    )

    task = TaskRepository(conn).recent(1)[0]
    logged = [
        entry for entry in AuditLog(conn).for_task(task.id) if entry.action != "agent.route"
    ]
    assert logged[0].status == "skipped"
    assert "dry run" in capsys.readouterr().out


# --- Week 3: the CLI approval prompt -----------------------------------------


def _approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        ref="abc123def456",
        task_id="task-1",
        tool="create_folder",
        risk="needs_approval",
        arguments={"path": "/home/tresor/Downloads/reports"},
        target="/home/tresor/Downloads/reports",
        preview="would create_folder with {'path': ...}",
    )


async def test_the_prompt_approves_only_on_explicit_yes() -> None:
    for answer in ["y", "Y", "yes", "YES", " yes "]:
        broker = CliApprovalBroker(interactive=True, read=lambda _prompt, a=answer: a, write=lambda _line: None)
        decision = await broker.decide(_approval_request())
        assert decision.approved is True, answer

    for answer in ["", "n", "no", "later", "yess"]:
        broker = CliApprovalBroker(interactive=True, read=lambda _prompt, a=answer: a, write=lambda _line: None)
        decision = await broker.decide(_approval_request())
        assert decision.approved is False, answer


async def test_the_prompt_shows_the_exact_target_and_reference() -> None:
    printed: list[str] = []
    broker = CliApprovalBroker(
        interactive=True, read=lambda _prompt: "n", write=printed.append
    )
    await broker.decide(_approval_request())
    text = "\n".join(printed)
    assert "create_folder" in text
    assert "/home/tresor/Downloads/reports" in text
    assert "abc123def456" in text
    assert "needs_approval" in text


async def test_a_non_interactive_run_cannot_approve_anything() -> None:
    def explode(_prompt: str) -> str:
        raise AssertionError("a non-interactive run must not try to prompt")

    broker = CliApprovalBroker(interactive=False, read=explode, write=lambda _line: None)
    decision = await broker.decide(_approval_request())
    assert decision.approved is False
    assert "not an interactive terminal" in decision.reason
    assert decision.decided_by == "policy"


async def test_end_of_input_counts_as_a_refusal() -> None:
    def eof(_prompt: str) -> str:
        raise EOFError

    broker = CliApprovalBroker(interactive=True, read=eof, write=lambda _line: None)
    decision = await broker.decide(_approval_request())
    assert decision.approved is False
    assert "no answer" in decision.reason


async def test_denied_write_tool_leaves_the_filesystem_alone(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    folder = tmp_path / "Downloads"
    folder.mkdir()
    settings = settings.model_copy(update={"allowed_folders": [folder]})
    target = folder / "should-not-exist"
    brain = ScriptedBrain(ToolCall(name="create_folder", arguments={"path": str(target)}), "Not allowed.")

    await _handle_turn(
        args=_args(),
        settings=settings,
        conn=conn,
        brain=brain,
        question="make a folder",
        database_path=tmp_path / "db",
    )

    assert not target.exists()
    task = TaskRepository(conn).recent(1)[0]
    logged = [
        entry for entry in AuditLog(conn).for_task(task.id) if entry.action != "agent.route"
    ]
    assert [entry.action for entry in logged] == ["approval.decision", "tool.call", "agent.answer"]
    assert logged[0].status == "skipped"
