"""The native app: core behaviour without a window, and widgets when one exists."""

from __future__ import annotations

import gc
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


def test_day_plan_round_trip(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    saved = runtime.add_day_plan_item(
        when="today", starts_at="09:30", request="read the note"
    )

    assert saved[0]["starts_at"] == "09:30"
    assert saved[0]["status"] == "planned"
    assert runtime.day_plan("today")[0]["request"] == "read the note"


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
        self.preference_updates: list[dict[str, object]] = []
        self.plan_items: list[dict[str, object]] = []
        self.speech_stops = 0

    def preferences(self):
        return {
            "owner_name": "Tester", "response_style": "balanced", "max_output_tokens": 128,
            "agent_max_steps": 8, "voice_enabled": False, "voice_pitch": 50, "voice_rate": 175,
            "allowed_folders": [],
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

    def day_plan(self, when="today"):
        return list(self.plan_items)

    def add_day_plan_item(self, *, when, starts_at, request):
        item = {
            "id": f"plan-{len(self.plan_items) + 1}",
            "day": when,
            "starts_at": starts_at,
            "request": request,
            "status": "planned",
        }
        self.plan_items.append(item)
        return list(self.plan_items)

    def conversations(self):
        return []

    def directives(self):
        return []

    def update_preferences(self, values):
        self.preference_updates.append(dict(values))
        return self.preferences() | values

    def stop_speaking(self):
        self.speech_stops += 1

    def start_background_learning(self):
        return None

    def stop_background_learning(self):
        return None

    def learning_status(self):
        return {"enabled": False, "total": 0, "compiled": 0, "pending": 0,
                "last_at": None, "corpus": None, "worker_error": None}

    def model_status(self):
        return {"provider": "scratch", "checkpoint": str(self.data_dir / "model.pt"),
                "ready": True, "hint": "", "can_install": True}

    def install_checkpoint(self, source):
        return self.model_status()


@pytest.fixture
def tk_root():
    """A withdrawn Tk root, torn down before its interpreter is collected.

    A Tk object that outlives its interpreter calls into Tcl from whatever thread
    happens to run the destructor.  Tcl is not thread safe, so that call blocks
    forever instead of failing: a later presence-server request that imported a
    module at that moment held the request thread in ``tkinter/font.py __del__``
    and the client timed out.  Collecting here, while the interpreter is alive and
    on the thread that created it, keeps Tk from leaking into other tests.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for widget tests")
    root.withdraw()
    try:
        yield root
    finally:
        gc.collect()  # widgets and fonts are unreachable once the test returned
        root.destroy()
        tk._default_root = None
        gc.collect()


def test_every_page_is_its_own_screen(tmp_path, tk_root) -> None:
    pytest.importorskip("customtkinter")
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import PAGES

    app = ClaireDesktop(tk_root, FakeRuntime(tmp_path))  # type: ignore[arg-type]
    for page_class in PAGES:
        app.show(page_class)
        assert app.current is page_class
        assert app.pages[page_class].frame.winfo_exists()
    # Every page has a distinct frame, so nothing is one long scrolling form.
    frames = {id(app.pages[page_class].frame) for page_class in PAGES}
    assert len(frames) == len(PAGES)


def test_saving_voice_tuning_does_not_turn_speech_on(tmp_path) -> None:
    from types import SimpleNamespace

    from claireai.desktop.pages import VoicePage

    page = object.__new__(VoicePage)
    page.app = SimpleNamespace(runtime=FakeRuntime(tmp_path))
    page.pitch = SimpleNamespace(get=lambda: 60)
    page.rate = SimpleNamespace(get=lambda: 190)
    page.transcript_note = SimpleNamespace(configure=lambda **_kwargs: None)
    page.save()

    assert page.runtime.preference_updates == [{"voice_pitch": 60, "voice_rate": 190}]


def test_turning_speak_replies_off_persists_and_stops_audio(tmp_path) -> None:
    from types import SimpleNamespace

    from claireai.desktop.pages import ChatPage

    page = object.__new__(ChatPage)
    page.app = SimpleNamespace(runtime=FakeRuntime(tmp_path))
    page.speak_replies = SimpleNamespace(get=lambda: False)
    page.append = lambda *_args: None
    page._toggle_voice()

    assert page.runtime.preference_updates[-1] == {"voice_enabled": False}
    assert page.runtime.speech_stops == 1


def test_the_chat_page_appends_and_resets(tmp_path, tk_root) -> None:
    pytest.importorskip("customtkinter")
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import ChatPage

    app = ClaireDesktop(tk_root, FakeRuntime(tmp_path))  # type: ignore[arg-type]
    app.show(ChatPage)
    chat = app.pages[ChatPage]
    chat.append("You", "hello", "you")
    assert "hello" in chat.transcript.get("1.0", "end")
    chat.new_chat()
    assert chat.transcript.get("1.0", "end").strip() == ""


def test_approvals_page_shows_pending_actions_immediately(tmp_path, tk_root) -> None:
    pytest.importorskip("customtkinter")
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import ApprovalsPage

    class PendingRuntime(FakeRuntime):
        def pending_approvals(self):
            return [
                {
                    "ref": "ref-1",
                    "task_id": "task-1",
                    "tool": "create_folder",
                    "risk": "needs_approval",
                    "arguments": {"path": "/tmp/example"},
                    "target": "/tmp/example",
                    "preview": "Create folder",
                    "description": "ClaireAI wants to run `create_folder`",
                    "expires_in": 120,
                }
            ]

    app = ClaireDesktop(tk_root, PendingRuntime(tmp_path))  # type: ignore[arg-type]
    app.show(ApprovalsPage)

    page = app.pages[ApprovalsPage]
    assert page.tree.get_children() == ("ref-1",)
    assert "1 waiting" in page.note.cget("text")


def test_day_plan_page_adds_and_lists_a_task(tmp_path, tk_root) -> None:
    pytest.importorskip("customtkinter")
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import DayPlanPage

    runtime = FakeRuntime(tmp_path)
    app = ClaireDesktop(tk_root, runtime)  # type: ignore[arg-type]
    app.show(DayPlanPage)
    page = app.pages[DayPlanPage]
    assert page.tree.get_children() == ()

    page.time_entry.insert(0, "09:30")
    page.request_entry.insert(0, "read the note")
    page._add()

    rows = page.tree.get_children()
    assert len(rows) == 1
    assert page.tree.item(rows[0], "values") == ("09:30", "planned", "read the note")
    assert page.request_entry.get() == ""


def test_settings_page_adds_and_saves_a_folder(tmp_path, tk_root, monkeypatch) -> None:
    pytest.importorskip("customtkinter")
    from claireai.desktop import pages as pages_module
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import SettingsPage

    runtime = FakeRuntime(tmp_path)
    app = ClaireDesktop(tk_root, runtime)  # type: ignore[arg-type]
    app.show(SettingsPage)
    page = app.pages[SettingsPage]
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setattr(pages_module.filedialog, "askdirectory", lambda **_kwargs: str(folder))

    page._add_folder()

    expected = str(folder.resolve())
    assert page.allowed_folders == [expected]
    assert runtime.preference_updates[-1] == {"allowed_folders": [expected]}
    assert page.folder_list.get(0, "end") == (expected,)


def test_a_failed_folder_save_does_not_show_unstored_access(tmp_path, tk_root, monkeypatch) -> None:
    pytest.importorskip("customtkinter")
    from claireai.desktop import pages as pages_module
    from claireai.desktop.app import ClaireDesktop
    from claireai.desktop.pages import SettingsPage

    class RefusingRuntime(FakeRuntime):
        def update_preferences(self, values):
            raise OSError("database is read-only")

    app = ClaireDesktop(tk_root, RefusingRuntime(tmp_path))  # type: ignore[arg-type]
    app.show(SettingsPage)
    page = app.pages[SettingsPage]
    folder = tmp_path / "work"
    folder.mkdir()
    monkeypatch.setattr(pages_module.filedialog, "askdirectory", lambda **_kwargs: str(folder))

    page._add_folder()

    assert page.allowed_folders == []
    assert page.folder_list.get(0, "end") == ()
    assert "not added" in page.folder_note.cget("text")
