"""One-shot owner approvals surfaced by the loopback UI."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Callable

from claireai.approval import ApprovalDecision, ApprovalRequest

APPROVAL_TIMEOUT_SECONDS = 120


@dataclass
class _Pending:
    request: ApprovalRequest
    deadline: float
    decision: ApprovalDecision | None = None


class LocalUIApprovalBroker:
    """Wait for a same-origin UI decision on the exact pending request."""

    name = "local-ui"

    def __init__(
        self,
        *,
        should_stop: Callable[[], bool],
        timeout: float = APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        self._should_stop = should_stop
        self._timeout = timeout
        self._condition = threading.Condition()
        self._pending: dict[str, _Pending] = {}

    def pending(self) -> list[dict[str, object]]:
        now = time.monotonic()
        with self._condition:
            requests = [
                {
                    "ref": item.request.ref,
                    "task_id": item.request.task_id,
                    "tool": item.request.tool,
                    "risk": item.request.risk,
                    "arguments": dict(item.request.arguments),
                    "target": item.request.target,
                    "preview": item.request.preview,
                    "description": item.request.describe(),
                    "expires_in": max(0, int(item.deadline - now)),
                }
                for item in self._pending.values()
                if item.decision is None and item.deadline > now
            ]
        return requests

    def decide_in_ui(self, ref: str, *, approved: bool) -> None:
        if not isinstance(ref, str) or not ref:
            raise ValueError("approval reference is required")
        if type(approved) is not bool:
            raise ValueError("approval decision must be true or false")
        with self._condition:
            item = self._pending.get(ref)
            if item is None or item.decision is not None:
                raise ValueError("approval is no longer pending or was already decided")
            if time.monotonic() >= item.deadline:
                item.decision = ApprovalDecision(
                    approved=False,
                    reason="approval expired; ask ClaireAI to try again",
                    decided_by="policy",
                )
                self._condition.notify_all()
                raise ValueError("approval expired; ask ClaireAI to try again")
            item.decision = ApprovalDecision(
                approved=approved,
                reason="owner decision from local UI",
                decided_by="owner",
            )
            self._condition.notify_all()

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        with self._condition:
            if request.ref in self._pending:
                raise ValueError("duplicate approval reference")
            item = _Pending(request=request, deadline=time.monotonic() + self._timeout)
            self._pending[request.ref] = item
            self._condition.notify_all()
        try:
            return await asyncio.to_thread(self._wait_for_decision, item)
        finally:
            with self._condition:
                self._pending.pop(request.ref, None)
                self._condition.notify_all()

    def _wait_for_decision(self, item: _Pending) -> ApprovalDecision:
        with self._condition:
            while item.decision is None:
                remaining = item.deadline - time.monotonic()
                if remaining <= 0:
                    item.decision = ApprovalDecision(
                        approved=False,
                        reason="approval expired; ask ClaireAI to try again",
                        decided_by="policy",
                    )
                    break
                if self._should_stop():
                    item.decision = ApprovalDecision(
                        approved=False,
                        reason="ClaireAI was stopped while awaiting approval",
                        decided_by="control",
                    )
                    break
                self._condition.wait(min(remaining, 0.25))
            self._condition.notify_all()
            return item.decision
