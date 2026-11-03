from __future__ import annotations

import asyncio
import time

import pytest

from claireai.approval import ApprovalRequest
from claireai.ui_approval import LocalUIApprovalBroker


def request(ref: str = "ref-1") -> ApprovalRequest:
    return ApprovalRequest(
        ref=ref, task_id="task-1", tool="create_folder", risk="needs_approval",
        arguments={"path": "/tmp/example"}, target="/tmp/example", preview="Create folder",
    )


def test_ui_decision_is_bound_to_exact_pending_reference() -> None:
    broker = LocalUIApprovalBroker(should_stop=lambda: False)

    async def run():
        task = asyncio.create_task(broker.decide(request()))
        deadline = time.monotonic() + 1
        while not broker.pending() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        pending = broker.pending()
        assert len(pending) == 1
        assert pending[0]["arguments"] == {"path": "/tmp/example"}
        with pytest.raises(ValueError, match="no longer pending"):
            broker.decide_in_ui("other-ref", approved=True)
        broker.decide_in_ui("ref-1", approved=True)
        decision = await task
        assert decision.approved is True
        assert decision.decided_by == "owner"
        assert broker.pending() == []

    asyncio.run(run())


def test_ui_approval_expires_and_denies() -> None:
    broker = LocalUIApprovalBroker(should_stop=lambda: False, timeout=0.02)
    decision = asyncio.run(broker.decide(request()))
    assert decision.approved is False
    assert decision.decided_by == "policy"
    assert "expired" in decision.reason


def test_shutdown_denies_pending_ui_approval() -> None:
    stopped = False
    broker = LocalUIApprovalBroker(should_stop=lambda: stopped)

    async def run():
        nonlocal stopped
        task = asyncio.create_task(broker.decide(request()))
        deadline = time.monotonic() + 1
        while not broker.pending() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        stopped = True
        decision = await task
        assert decision.approved is False
        assert decision.decided_by == "control"

    asyncio.run(run())
