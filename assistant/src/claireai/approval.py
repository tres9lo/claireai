"""Approval: the human gate for tools above ``safe`` (Month 1, Week 3).

The model can propose a destructive action; only a human can authorise it. Every
request carries the exact tool, the validated arguments and the target path, so
what the owner approves is what runs — a changed plan needs a new approval
(roadmap 7.10).

Decision records are written to ``task_log`` (``approval.decision``), and the
tool row that follows carries the same ``approval_ref``. That keeps the log
append-only and self-contained: an auditor can see exactly which human decision
allowed which action, without a second table that could drift.

Non-interactive runs get :class:`DenyAll`. There is deliberately no "always
approve" broker in the product: an unattended run must not be able to authorise
its own writes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

APPROVED = "approved"
DENIED = "denied"
EXPIRED = "expired"


def new_ref() -> str:
    """Short, unique reference for one approval decision."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class ApprovalRequest:
    ref: str
    task_id: str
    tool: str
    risk: str
    arguments: Mapping[str, Any]
    target: str | None = None
    preview: str = ""

    def describe(self) -> str:
        lines = [
            f"ClaireAI wants to run `{self.tool}` ({self.risk}).",
        ]
        if self.target:
            lines.append(f"  target: {self.target}")
        if self.arguments:
            rendered = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
            lines.append(f"  arguments: {rendered}")
        if self.preview:
            lines.append(f"  effect: {self.preview}")
        lines.append(f"  approval reference: {self.ref}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    reason: str = ""
    decided_by: str = "owner"


class ApprovalBroker(Protocol):
    name: str

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


@dataclass
class DenyAll:
    """Refuse everything. The default for programmatic and unattended runs."""

    reason: str = (
        "no approval channel is attached; run the interactive CLI to approve "
        "actions, or do not ask for write tools in automation"
    )
    name: str = "deny-all"
    seen: list[ApprovalRequest] = field(default_factory=list)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.seen.append(request)
        return ApprovalDecision(approved=False, reason=self.reason, decided_by="policy")
