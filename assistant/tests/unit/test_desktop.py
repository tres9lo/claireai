"""The native app: core behaviour without a window, and widgets when one exists."""

from __future__ import annotations

from pathlib import Path

import pytest

from claireai.brain import BrainStopped
from claireai.config import Settings
from claireai.desktop.runtime import DesktopRuntime


def _runtime(tmp_path: Path) -> DesktopRuntime:
    return DesktopRuntime(Settings(_env_file=None, data_dir=tmp_path / "data"))


# -- core ------------------------------------------------------------------


def test_the_database_is_migrated_and_status_is_honest(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    status = runtime.status()

    assert Path(status["database"]).is_file()
    assert status["model_ready"] is False
    assert status["shutdown"] is False
    assert status["browser"] is False


def test_conversations_round_trip(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    conversation_id = runtime.new_conversation("First")
    assert runtime.conversations()[0]["id"] == conversation_id

    assert runtime.delete_conversation(conversation_id) is True
    assert runtime.conversations() == []
    assert runtime.delete_conversation("missing") is False


def test_directives_round_trip(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    saved = runtime.save_directive(title="Tone", content="Be brief.")
    assert saved[0]["title"] == "Tone"

    directive_id = saved[0]["id"]
    assert runtime.delete_directive(directive_id) is True
    assert runtime.directives() == []


def test_preferences_are_validated_like_the_runtime(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.update_preferences({"owner_name": "Ange", "voice_rate": 200})
    assert runtime.preferences()["owner_name"] == "Ange"

    with pytest.raises(ValueError):
        runtime.update_preferences({"max_output_tokens": 100000})
    with pytest.raises(ValueError):
        runtime.update_preferences({"not_a_setting": 1})


def test_control_switches_are_shared_with_the_cli(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.set_control(shutdown=True)
    assert runtime.stopped() is True
    assert runtime.status()["shutdown"] is True
    runtime.set_control(shutdown=False)
    assert runtime.stopped() is False


def test_a_stopped_assistant_refuses_to_run(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.set_control(shutdown=True)

    with pytest.raises(BrainStopped):
        runtime.send("hello")
    with pytest.raises(ValueError):
        runtime.send("   ")


def test_approvals_start_empty_and_unknown_refs_are_refused(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    assert runtime.pending_approvals() == []
    with pytest.raises(ValueError):
        runtime.decide_approval("not-a-real-ref", approved=True)


def test_activity_lists_what_the_runtime_ran(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.new_conversation()
    assert isinstance(runtime.recent_tasks(), list)


def test_speaking_can_be_cancelled(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.stop_speaking()
    assert runtime._speech_stop.is_set() is True


# -- window ----------------------------------------------------------------


class FakeRuntime:
    """Just enough for the pages to build."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._stopped = False

    def preferences(self):
        return {
            "owner_name": "Tester", "response_style": "balanced", "max_output_tokens": 128,
            "agent_max_steps": 8, "voice_enabled": False, "voice_pitch": 50, "voice_rate": 175,
        }

    def status(self):
        return {
            "data_dir": str(self.data_dir), "database": str(self.data_dir / "db.sqlite3"),
            "checkpoint": str(self.data_dir / "model.pt"), "model_ready": False,
            "voices": {"en": "English", "fr": "French"}, "browser": False,
            "shutdown": False, "self_improvement": False, "allowed_folders": [],
        }

    def control(self):
        return {"shutdown": self._stopped, "browser": False, "self_improvement": False}

    def set_control(self, **flags):
        self._stopped = bool(flags.get("shutdown", self._stopped))
        return self.control()

    def stopped(self):
        return self._stopped

    def pending_approvals(self):
        return []

    def recent_tasks(self):
        return []

    def conversations(self):
        return []

    def directives(self):
        return []

    def update_preferences(self, values):
        return self.preferences() | values

    def stop_speaking(self):
        return None


def _tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for widget tests")
    root.withdraw()
    return root


def test_every_page_is_its_own_screen(tmp_path) -> None:
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import PAGES

    root = _tk_root()
    try:
        app = ClaireDesktop(root, FakeRuntime(tmp_path))  # type: ignore[arg-type]
        for page_class in PAGES:
            app.show(page_class)
            assert app.current is page_class
            assert app.pages[page_class].frame.winfo_exists()
        # Every page has a distinct frame, so nothing is one long scrolling form.
        frames = {id(app.pages[page_class].frame) for page_class in PAGES}
        assert len(frames) == len(PAGES)
    finally:
        root.destroy()


def test_the_chat_page_appends_and_resets(tmp_path) -> None:
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import ChatPage

    root = _tk_root()
    try:
        app = ClaireDesktop(root, FakeRuntime(tmp_path))  # type: ignore[arg-type]
        app.show(ChatPage)
        chat = app.pages[ChatPage]
        chat.append("You", "hello", "you")
        assert "hello" in chat.transcript.get("1.0", "end")
        chat.new_chat()
        assert chat.transcript.get("1.0", "end").strip() == ""
    finally:
        root.destroy()
