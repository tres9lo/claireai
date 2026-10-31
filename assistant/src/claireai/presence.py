"""Presence state: what the renderer is allowed to show.

The renderer is a read-only consumer of this state (docs/architecture/
presence-renderer.md). It holds no capability, cannot invoke a tool, and cannot
approve anything — it just draws what the assistant reports.

The same state is emitted while the agent works, so what the HUD shows and what
the task log records come from one place. When they disagree, the log is right.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

STATES = (
    "idle",
    "listening",
    "thinking",
    "tool_running",
    "awaiting_approval",
    "speaking",
    "error",
    "offline",
)

ACTIVE_KEYS = ("microphone", "camera", "screen", "input")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PresenceEvent:
    seq: int
    ts: str
    state: str
    task_id: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)
    active: Mapping[str, bool] = field(default_factory=lambda: {key: False for key in ACTIVE_KEYS})

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "taskId": self.task_id,
            "state": self.state,
            "detail": dict(self.detail),
            "active": dict(self.active),
        }


def idle_event() -> PresenceEvent:
    return PresenceEvent(seq=0, ts=utc_now(), state="idle")


class PresenceStream:
    """A tiny, thread-safe state holder with a monotonic sequence number.

    Kept deliberately dumb: the renderer polls or streams the snapshot, and
    duplicate or out-of-order events are detectable because ``seq`` only grows.
    """

    def __init__(self, reporter: Callable[[PresenceEvent], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._event = idle_event()
        # Optional sink for humans: the CLI prints state changes with --presence.
        self._reporter = reporter

    def emit(
        self,
        state: str,
        *,
        task_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
        active: Mapping[str, bool] | None = None,
        ts: str | None = None,
    ) -> PresenceEvent:
        if state not in STATES:
            raise ValueError(f"unknown presence state {state!r}; expected one of {', '.join(STATES)}")
        with self._lock:
            merged_active = dict(self._event.active)
            if active:
                for key, value in active.items():
                    if key not in ACTIVE_KEYS:
                        raise ValueError(f"unknown activity flag {key!r}; expected {', '.join(ACTIVE_KEYS)}")
                    merged_active[key] = bool(value)
            event = PresenceEvent(
                seq=self._event.seq + 1,
                ts=ts or utc_now(),
                state=state,
                task_id=task_id if task_id is not None else self._event.task_id,
                detail=dict(detail or {}),
                active=merged_active,
            )
            self._event = event
        if self._reporter is not None:
            self._reporter(event)
        return event

    def snapshot(self) -> PresenceEvent:
        with self._lock:
            return self._event

    def set_active(self, **flags: bool) -> PresenceEvent:
        return self.emit(self.snapshot().state, active=flags)

    def reset(self) -> PresenceEvent:
        with self._lock:
            self._event = PresenceEvent(
                seq=self._event.seq + 1,
                ts=utc_now(),
                state="idle",
                active={key: False for key in ACTIVE_KEYS},
            )
            return self._event


def sse_frame(event: PresenceEvent) -> str:
    """One Server-Sent Events frame for the renderer."""
    import json

    return f"event: presence\ndata: {json.dumps(event.as_dict(), sort_keys=True)}\n\n"
