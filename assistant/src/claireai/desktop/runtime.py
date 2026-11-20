"""Headless core of the native desktop app.

Every decision lives here so it can be tested without a display: the Tk shell
only draws what these methods return. Connections are opened per call, so the
worker thread that runs the model never shares a SQLite handle with the UI
thread that lists conversations.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from claireai.agent import AgentError, run_task
from claireai.brain import BrainError, BrainStopped, build_brain
from claireai.config import Settings
from claireai.identity import answer_self_question
from claireai.learning import BackgroundLearner, LearningError, LearningJournal
from claireai.training import TrainingWorker
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


class ModelMissingError(BrainError):
    """No checkpoint is installed, which the window can offer to fix."""


def _checkpoint_manifest(path: Path) -> None:
    """Refuse a file that is not a brain-lab checkpoint before installing it.

    ``torch.load`` is already required to run a scratch checkpoint, so checking
    here costs one load during an explicit install and prevents a 28 MB copy from
    silently replacing the model with something that cannot start.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            'PyTorch is needed to install a checkpoint: pip install "claireai[model]"'
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - torch raises its own error types
        raise ValueError(f"this file is not a readable checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or "state_dict" not in payload or "manifest" not in payload:
        raise ValueError("this file is missing the state_dict/manifest pair a checkpoint needs")


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
        self.journal = LearningJournal(self.settings.resolved_data_dir)
        self._learner: BackgroundLearner | None = None
        self._trainer: TrainingWorker | None = None

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

    def model_status(self) -> dict[str, Any]:
        """What the window needs to explain, or fix, a missing model."""
        provider = self.settings.brain_provider
        checkpoint = self.settings.resolved_model_checkpoint
        if provider == "scratch":
            ready = checkpoint.is_file()
            hint = (
                ""
                if ready
                else "Install a checkpoint from a brain-lab run, or set CLAIREAI_MODEL_CHECKPOINT."
            )
        else:
            ready = True
            hint = f"Using the {provider} connector; conversations go to that endpoint."
        facts = None
        if provider == "scratch" and ready:
            from claireai.models import describe_checkpoint

            facts = describe_checkpoint(checkpoint)
        return {
            "provider": provider,
            "checkpoint": str(checkpoint),
            "ready": ready,
            "hint": hint,
            "can_install": provider == "scratch",
            "name": (facts or {}).get("name"),
            "parameters": (facts or {}).get("parameters"),
            "context": (facts or {}).get("context"),
        }

    def ensure_model(self) -> dict[str, Any]:
        """Adopt a locally trained checkpoint if this data directory has none."""
        from claireai.models import ensure_installed_checkpoint

        ensure_installed_checkpoint(self.settings)
        return self.model_status()

    def install_checkpoint(self, source: Path) -> dict[str, Any]:
        """Validate a checkpoint and install it atomically as the local model."""
        origin = Path(source).expanduser()
        if not origin.is_file() or origin.suffix != ".pt":
            raise ValueError("choose a .pt checkpoint produced by brain-lab")
        target = self.settings.model_directory / "claire-scratch.pt"
        if origin.resolve() != target.resolve():
            _checkpoint_manifest(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.resolve() != target.resolve():
            temporary = target.with_name(target.name + ".installing")
            shutil.copyfile(origin, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        sidecar = origin.with_suffix(".json")
        if sidecar.is_file():
            shutil.copyfile(sidecar, target.with_suffix(".json"))
        return self.model_status()

    def learning_status(self) -> dict[str, Any]:
        try:
            status = self.journal.status()
        except (LearningError, OSError):
            status = {"total": 0, "compiled": 0, "pending": 0, "last_at": None, "corpus": None}
        return {
            **status,
            "enabled": bool(self.control()["self_improvement"]),
            "worker_error": getattr(self._learner, "last_error", None),
            "training": (
                self._trainer.status()
                if self._trainer is not None
                else {"enabled": False, "status": "not started"}
            ),
        }

    def start_background_learning(self) -> None:
        """Compile journals while the window is open; inert until the flag is on."""
        if self._learner is not None:
            return
        allowed = lambda: bool(self.control()["self_improvement"])  # noqa: E731
        self._learner = BackgroundLearner(self.journal, enabled=allowed)
        self._learner.start()
        self._trainer = TrainingWorker(self.data_dir, enabled=allowed)
        self._trainer.start()

    def stop_background_learning(self) -> None:
        for worker in (self._learner, self._trainer):
            if worker is not None:
                worker.stop()
        self._learner = None
        self._trainer = None

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
        control = self.control()
        # An identity question does not need a model: answer it from the real
        # configuration so it still works before a checkpoint is installed.
        about_itself = answer_self_question(
            message, self.settings, control=control, learning=self.learning_status()
        )
        brain = build_brain(self.settings)
        try:
            if about_itself is not None:
                # Identity, greeting and capability turns are answered by the
                # runtime itself (and recorded through ``ask``), so they work
                # with no checkpoint and never become a model hallucination.
                use_tools = False
            else:
                health = await brain.health()
                if not health.available or not health.model_present:
                    if self.settings.brain_provider == "scratch":
                        raise ModelMissingError(
                            "No local checkpoint is installed yet, so I cannot answer that."
                        )
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
                # ``run_task`` takes a conversation id but does not return it, so
                # an absent one has to be created here. The previous version read
                # ``outcome.conversation_id`` (it does not exist) and every
                # tool-enabled turn crashed after the model had answered.
                if conversation_id is None:
                    conversation_id = ConversationRepository(conn).create()
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
                conversation_id=conversation_id,
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


__all__ = [
    "DesktopRuntime",
    "Turn",
    "VoiceStopped",
    "VoiceUnavailable",
    "AgentError",
    "BrainError",
    "ModelMissingError",
]
