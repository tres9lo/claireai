"""Headless core of the native desktop app.

Every decision lives here so it can be tested without a display: the Tk shell
only draws what these methods return. Connections are opened per call, so the
worker thread that runs the model never shares a SQLite handle with the UI
thread that lists conversations.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from claireai.agent import AgentError, run_task
from claireai.brain import BrainError, BrainStopped, build_brain
from claireai.config import Settings
from claireai.presence import PresenceStream
from claireai.runtime_control import RuntimeControl
from claireai.storage import (
    ConversationRepository,
    Database,
    DirectiveRepository,
    PreferenceRepository,
)
from claireai.tools import default_registry
from claireai.ui_approval import LocalUIApprovalBroker
from claireai.voice import (
    VoiceStopped,
    VoiceUnavailable,
    available_voices,
    speak_text,
)


@dataclass(frozen=True)
class Turn:
    conversation_id: str
    task_id: str
    content: str
    model: str
    duration_ms: int = 0
    route: str | None = None


class DesktopRuntime:
    """The assistant, without a window."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        presence: PresenceStream | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.migrate()
        self.presence = presence or PresenceStream()
        self._speech_stop = threading.Event()
        self.approvals = LocalUIApprovalBroker(should_stop=self.stopped)

    # -- plumbing ---------------------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self.database.connect()
        try:
            yield conn
        finally:
            conn.close()

    @property
    def data_dir(self) -> Path:
        return self.settings.resolved_data_dir

    def stopped(self) -> bool:
        return RuntimeControl(self.settings.control_state_path).read()["shutdown"]

    def control(self) -> dict[str, bool]:
        return RuntimeControl(self.settings.control_state_path).read()

    def set_control(self, **flags: bool) -> dict[str, bool]:
        return RuntimeControl(self.settings.control_state_path).update(flags)

    # -- status -----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        checkpoint = self.settings.resolved_model_checkpoint
        control = self.control()
        return {
            "data_dir": str(self.data_dir),
            "database": str(self.settings.database_path),
            "checkpoint": str(checkpoint),
            "model_ready": checkpoint.is_file(),
            "voices": available_voices(),
            "browser": control["browser"],
            "shutdown": control["shutdown"],
            "self_improvement": control["self_improvement"],
            "allowed_folders": [str(path) for path in self.settings.allowed_folder_roots],
        }

    # -- preferences and directives --------------------------------------

    def preferences(self) -> dict[str, Any]:
        with self._conn() as conn:
            return PreferenceRepository(conn).effective(self.settings)

    def update_preferences(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._conn() as conn:
            return PreferenceRepository(conn).update(values)

    def directives(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return DirectiveRepository(conn).list()

    def save_directive(
        self, *, title: str, content: str, enabled: bool = True, directive_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            repository = DirectiveRepository(conn)
            repository.save(
                directive_id=directive_id, title=title, content=content, enabled=enabled
            )
            return repository.list()

    def delete_directive(self, directive_id: str) -> bool:
        with self._conn() as conn:
            return DirectiveRepository(conn).delete(directive_id)

    # -- conversations and activity --------------------------------------

    def conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return ConversationRepository(conn).recent(limit=limit)

    def new_conversation(self, title: str = "New conversation") -> str:
        with self._conn() as conn:
            return ConversationRepository(conn).create(title=title)

    def history(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return [message.__dict__ for message in ConversationRepository(conn).messages(conversation_id)]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._conn() as conn:
            return ConversationRepository(conn).delete(conversation_id)

    def recent_tasks(self, limit: int = 60) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, status, request, error FROM tasks"
                " ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- approvals --------------------------------------------------------

    def pending_approvals(self) -> list[dict[str, object]]:
        return self.approvals.pending()

    def decide_approval(self, ref: str, *, approved: bool) -> None:
        self.approvals.decide_in_ui(ref, approved=approved)

    # -- talking to the model --------------------------------------------

    def send(
        self,
        text: str,
        *,
        conversation_id: str | None = None,
        use_tools: bool = True,
    ) -> Turn:
        """One turn. With tools it goes through the agent; without, plain chat."""
        message = text.strip()
        if not message:
            raise ValueError("the message is empty")
        if self.stopped():
            raise BrainStopped("ClaireAI is stopped. Turn it back on in Settings.")
        return asyncio.run(self._send(message, conversation_id, use_tools))

    async def _send(self, message: str, conversation_id: str | None, use_tools: bool) -> Turn:
        brain = build_brain(self.settings)
        try:
            health = await brain.health()
            if not health.available or not health.model_present:
                raise BrainError(health.detail or "the configured brain is unavailable")
            if not use_tools:
                from claireai.cli import ask

                with self._conn() as conn:
                    turn = await ask(
                        settings=self.settings,
                        conn=conn,
                        brain=brain,
                        question=message,
                        device="desktop-native",
                        conversation_id=conversation_id,
                    )
                return Turn(
                    conversation_id=turn.conversation_id,
                    task_id=turn.task_id,
                    content=turn.content,
                    model=turn.model,
                    duration_ms=turn.duration_ms,
                )

            with self._conn() as conn:
                outcome = await run_task(
                    settings=self.settings,
                    conn=conn,
                    brain=brain,
                    request=message,
                    registry=default_registry(browser_enabled=self.control()["browser"]),
                    approvals=self.approvals,
                    presence=self.presence,
                    device="desktop-native",
                    conversation_id=conversation_id,
                )
            return Turn(
                conversation_id=outcome.conversation_id,
                task_id=outcome.task_id,
                content=outcome.content,
                model=outcome.model,
                duration_ms=outcome.duration_ms,
                route=outcome.route,
            )
        finally:
            await brain.aclose()

    # -- speech -----------------------------------------------------------

    def speak(
        self,
        text: str,
        *,
        language: str | None = None,
        pitch: int | None = None,
        rate: int | None = None,
    ) -> str:
        self._speech_stop.clear()
        preferences = self.preferences()
        return speak_text(
            text,
            pitch=preferences["voice_pitch"] if pitch is None else pitch,
            rate=preferences["voice_rate"] if rate is None else rate,
            language=language,
            should_stop=self._speech_stop.is_set,
        )

    def stop_speaking(self) -> None:
        self._speech_stop.set()


__all__ = ["DesktopRuntime", "Turn", "VoiceStopped", "VoiceUnavailable", "AgentError", "BrainError"]
