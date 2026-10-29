"""CLI turn tests with a stub brain.

The brain is replaced so these tests assert the task/audit bookkeeping, not the
model's answer. The real Ollama path is exercised by the manual smoke test.
"""

from __future__ import annotations

import sqlite3

import pytest

from claireai.audit import AuditLog
from claireai.brain import BrainHealth, BrainResponse, BrainUnavailable
from claireai.cli import ask
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
