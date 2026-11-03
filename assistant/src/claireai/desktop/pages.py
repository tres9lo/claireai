"""The pages of the native ClaireAI window.

Each page is its own screen with its own job -- chat is not a dashboard, the
activity log is not squeezed under the transcript. Pages never talk to the model
directly: they call :class:`claireai.desktop.runtime.DesktopRuntime` through the
shell's worker, so the window stays responsive.
"""

from __future__ import annotations

import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from claireai.desktop.app import ClaireDesktop


class Page:
    """Base page: owns one frame in the stack."""

    title = "Page"
    subtitle = ""

    def __init__(self, app: "ClaireDesktop", parent: tk.Widget) -> None:
        self.app = app
        self.frame = ttk.Frame(parent, padding=18)
        self.build()

    @property
    def runtime(self):  # noqa: ANN201
        return self.app.runtime

    def build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def on_show(self) -> None:
        """Called every time the page becomes visible."""

    def heading(self, text: str, subtitle: str = "") -> None:
        ttk.Label(self.frame, text=text, style="Heading.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(self.frame, text=subtitle, style="Muted.TLabel").pack(anchor="w", pady=(2, 10))


class ChatPage(Page):
    title = "Chat"
    subtitle = "Talk to ClaireAI. Tools run through the same approvals as always."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        self.conversation_id: str | None = None
        self.busy = False

        bar = ttk.Frame(self.frame)
        bar.pack(fill="x", pady=(0, 8))
        self.use_tools = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Let it use tools", variable=self.use_tools).pack(side="left")
        self.speak_replies = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Speak replies", variable=self.speak_replies).pack(side="left", padx=12)
        ttk.Button(bar, text="Stop speaking", command=self._stop_speaking).pack(side="right")
        ttk.Button(bar, text="New chat", command=self.new_chat).pack(side="right", padx=6)

        body = ttk.Frame(self.frame)
        body.pack(fill="both", expand=True)
        self.transcript = tk.Text(body, wrap="word", state="disabled", padx=12, pady=10,
                                  relief="flat", background="#12161f", foreground="#e8ecf4")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=scroll.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.transcript.tag_configure("you", foreground="#8bd5ff", spacing1=8)
        self.transcript.tag_configure("claire", foreground="#e8ecf4", spacing1=8)
        self.transcript.tag_configure("meta", foreground="#7c8798", spacing3=10)
        self.transcript.tag_configure("error", foreground="#ff8a8a", spacing3=10)

        compose = ttk.Frame(self.frame)
        compose.pack(fill="x", pady=(10, 0))
        self.entry = tk.Text(compose, height=3, wrap="word", relief="flat",
                             background="#1a2030", foreground="#e8ecf4", insertbackground="#e8ecf4")
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Control-Return>", lambda _event: self.send())
        self.send_button = ttk.Button(compose, text="Send", command=self.send)
        self.send_button.pack(side="right", padx=(10, 0), fill="y")
        ttk.Label(self.frame, text="Ctrl+Enter sends", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

    # -- helpers ---------------------------------------------------------

    def append(self, kind: str, text: str, tag: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{kind}\n", (tag,))
        self.transcript.insert("end", f"{text}\n", (tag,))
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def new_chat(self) -> None:
        self.conversation_id = None
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _stop_speaking(self) -> None:
        self.runtime.stop_speaking()

    def _set_busy(self, busy: bool, note: str = "") -> None:
        self.busy = busy
        self.send_button.configure(state="disabled" if busy else "normal")
        if note:
            self.append("·", note, "meta")

    # -- flow ------------------------------------------------------------

    def send(self) -> None:
        if self.busy:
            return
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return
        self.entry.delete("1.0", "end")
        self.append("You", text, "you")
        self._set_busy(True, "thinking …")
        conversation = self.conversation_id
        use_tools = bool(self.use_tools.get())

        def call():  # noqa: ANN202
            return self.runtime.send(text, conversation_id=conversation, use_tools=use_tools)

        self.app.submit("chat", call, self._replied, self._failed)

    def _replied(self, turn) -> None:  # noqa: ANN001
        self.conversation_id = turn.conversation_id
        self._set_busy(False)
        self.append("ClaireAI", turn.content, "claire")
        meta = f"{turn.model} · {turn.duration_ms / 1000:.1f}s"
        if turn.route:
            meta += f" · route {turn.route}"
        self.append("", meta, "meta")
        if self.speak_replies.get():
            self.app.submit("speak", lambda: self.runtime.speak(turn.content), lambda _: None,
                            lambda error: self.append("·", f"voice: {error}", "error"))

    def _failed(self, error: Exception) -> None:
        self._set_busy(False)
        self.append("!", f"{type(error).__name__}: {error}", "error")

    def on_show(self) -> None:
        self.speak_replies.set(bool(self.runtime.preferences().get("voice_enabled")))


class ActivityPage(Page):
    title = "Activity"
    subtitle = "Every task this assistant has run, newest first."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        bar = ttk.Frame(self.frame)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        self.summary = ttk.Label(bar, text="", style="Muted.TLabel")
        self.summary.pack(side="left", padx=12)

        columns = ("when", "status", "request")
        self.tree = ttk.Treeview(self.frame, columns=columns, show="headings", height=14)
        for column, label, width in (("when", "When", 150), ("status", "Status", 110), ("request", "Request", 520)):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._show_detail)
        self.detail = tk.Text(self.frame, height=5, wrap="word", relief="flat",
                              background="#1a2030", foreground="#e8ecf4", state="disabled")
        self.detail.pack(fill="x", pady=(10, 0))

    def refresh(self) -> None:
        tasks = self.runtime.recent_tasks()
        self.tree.delete(*self.tree.get_children())
        for task in tasks:
            when = str(task.get("created_at", ""))[:19].replace("T", " ")
            self.tree.insert("", "end", iid=task["id"], values=(when, task.get("status", ""), task.get("request", "")))
        failed = sum(1 for task in tasks if task.get("status") in {"failed", "interrupted"})
        self.summary.configure(text=f"{len(tasks)} tasks · {failed} failed or interrupted")

    def _show_detail(self, _event=None) -> None:  # noqa: ANN001
        selection = self.tree.selection()
        if not selection:
            return
        for task in self.runtime.recent_tasks():
            if task["id"] == selection[0]:
                body = f"{task.get('request', '')}\n\nstatus: {task.get('status', '')}"
                if task.get("error"):
                    body += f"\nerror: {task['error']}"
                self.detail.configure(state="normal")
                self.detail.delete("1.0", "end")
                self.detail.insert("1.0", body)
                self.detail.configure(state="disabled")
                return

    def on_show(self) -> None:
        self.refresh()


class ApprovalsPage(Page):
    title = "Approvals"
    subtitle = "Actions that change your disk wait here. Nothing runs until you decide."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        bar = ttk.Frame(self.frame)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="Approve", command=lambda: self._decide(True)).pack(side="left")
        ttk.Button(bar, text="Deny", command=lambda: self._decide(False)).pack(side="left", padx=8)
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="left")
        self.note = ttk.Label(bar, text="", style="Muted.TLabel")
        self.note.pack(side="left", padx=12)

        self.tree = ttk.Treeview(self.frame, columns=("tool", "target", "risk", "expires"),
                                 show="headings", height=12)
        for column, label, width in (("tool", "Tool", 160), ("target", "Target", 420),
                                     ("risk", "Risk", 90), ("expires", "Expires in", 100)):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.preview = tk.Text(self.frame, height=4, wrap="word", relief="flat",
                               background="#1a2030", foreground="#e8ecf4", state="disabled")
        self.preview.pack(fill="x", pady=(10, 0))
        self.tree.bind("<<TreeviewSelect>>", self._show_preview)
        self.frame.after(1000, self._tick)

    def refresh(self) -> None:
        pending = self.runtime.pending_approvals()
        self.tree.delete(*self.tree.get_children())
        for item in pending:
            self.tree.insert("", "end", iid=str(item["ref"]),
                             values=(item["tool"], item.get("target") or "", item.get("risk", ""),
                                     f"{item.get('expires_in', 0)}s"))
        self.note.configure(text="nothing waiting" if not pending else f"{len(pending)} waiting")

    def _show_preview(self, _event=None) -> None:  # noqa: ANN001
        selection = self.tree.selection()
        if not selection:
            return
        for item in self.runtime.pending_approvals():
            if str(item["ref"]) == selection[0]:
                self.preview.configure(state="normal")
                self.preview.delete("1.0", "end")
                self.preview.insert("1.0", str(item.get("description") or item.get("preview") or ""))
                self.preview.configure(state="disabled")
                return

    def _decide(self, approved: bool) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("ClaireAI", "Select an action first.", parent=self.frame)
            return
        try:
            self.runtime.decide_approval(selection[0], approved=approved)
        except ValueError as error:
            messagebox.showwarning("ClaireAI", str(error), parent=self.frame)
        self.refresh()

    def _tick(self) -> None:
        if self.frame.winfo_ismapped():
            self.refresh()
        self.frame.after(1500, self._tick)


class ConversationsPage(Page):
    title = "History"
    subtitle = "Saved conversations. Deleting one removes its stored messages."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        split = ttk.PanedWindow(self.frame, orient="horizontal")
        split.pack(fill="both", expand=True)

        left = ttk.Frame(split)
        self.listbox = tk.Listbox(left, exportselection=False, relief="flat")
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._open)
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(buttons, text="Delete", command=self._delete).pack(side="left", padx=8)
        split.add(left, weight=1)

        right = ttk.Frame(split)
        self.view = tk.Text(right, wrap="word", state="disabled", relief="flat",
                            background="#12161f", foreground="#e8ecf4")
        self.view.pack(fill="both", expand=True)
        split.add(right, weight=2)
        self.ids: list[str] = []

    def refresh(self) -> None:
        conversations = self.runtime.conversations()
        self.listbox.delete(0, "end")
        self.ids = []
        for conversation in conversations:
            title = conversation.get("title") or "Conversation"
            when = str(conversation.get("updated_at") or conversation.get("created_at") or "")[:16]
            self.listbox.insert("end", f"{when}  {title}")
            self.ids.append(str(conversation.get("id") or conversation.get("conversation_id") or ""))

    def _open(self, _event=None) -> None:  # noqa: ANN001
        selection = self.listbox.curselection()
        if not selection or selection[0] >= len(self.ids):
            return
        messages = self.runtime.history(self.ids[selection[0]])
        self.view.configure(state="normal")
        self.view.delete("1.0", "end")
        for message in messages:
            self.view.insert("end", f"{message.get('role', '?')}: {message.get('content', '')}\n\n")
        self.view.configure(state="disabled")

    def _delete(self) -> None:
        selection = self.listbox.curselection()
        if not selection or selection[0] >= len(self.ids):
            return
        if not messagebox.askyesno("ClaireAI", "Delete this conversation?", parent=self.frame):
            return
        self.runtime.delete_conversation(self.ids[selection[0]])
        self.refresh()

    def on_show(self) -> None:
        self.refresh()


class VoicePage(Page):
    title = "Voice"
    subtitle = "Local speech, English and French. No audio leaves this machine."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        status = ttk.LabelFrame(self.frame, text="Speech output", padding=12)
        status.pack(fill="x")
        voices = self.runtime.status()["voices"]
        self.engine_note = ttk.Label(
            status,
            text=(
                "eSpeak NG found · voices: " + ", ".join(voices.values())
                if voices
                else "No local speech engine found (install espeak-ng)"
            ),
            style="Muted.TLabel",
        )
        self.engine_note.pack(anchor="w")

        row = ttk.Frame(status)
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Language").pack(side="left")
        self.language = ttk.Combobox(row, values=["Automatic", "English", "French"], width=12, state="readonly")
        self.language.set("Automatic")
        self.language.pack(side="left", padx=8)
        ttk.Button(row, text="Test the voice", command=self.test_voice).pack(side="left")
        ttk.Button(row, text="Stop", command=self.runtime.stop_speaking).pack(side="left", padx=8)

        preferences = self.runtime.preferences()
        self.pitch = tk.IntVar(value=int(preferences["voice_pitch"]))
        self.rate = tk.IntVar(value=int(preferences["voice_rate"]))
        for label, variable, start, end in (("Pitch", self.pitch, 0, 99), ("Rate", self.rate, 80, 450)):
            line = ttk.Frame(status)
            line.pack(fill="x", pady=(8, 0))
            ttk.Label(line, text=label, width=8).pack(side="left")
            ttk.Scale(line, from_=start, to=end, variable=variable, orient="horizontal").pack(
                side="left", fill="x", expand=True)
            ttk.Label(line, textvariable=variable, width=5).pack(side="left")

        input_box = ttk.LabelFrame(self.frame, text="Speech input", padding=12)
        input_box.pack(fill="x", pady=(14, 0))
        self.mic_note = ttk.Label(input_box, text=self._microphone_note(), style="Muted.TLabel")
        self.mic_note.pack(anchor="w")
        row2 = ttk.Frame(input_box)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Button(row2, text="Record 5 seconds", command=self.record).pack(side="left")
        self.transcript_note = ttk.Label(input_box, text="", style="Muted.TLabel", wraplength=760)
        self.transcript_note.pack(anchor="w", pady=(8, 0))

        ttk.Button(self.frame, text="Save voice settings", command=self.save).pack(anchor="w", pady=(14, 0))

    def _microphone_note(self) -> str:
        from claireai.voice_input import transcriber_for

        transcriber = transcriber_for(str(self.runtime.data_dir / "models" / "voice" / "asr"))
        if not Path("/dev/snd").exists():
            return "No audio capture device (/dev/snd) — recording is unavailable on this host."
        if not transcriber.runtime_available:
            return "Microphone present, but speech recognition is not installed (pip install faster-whisper)."
        if not transcriber.installed:
            return "Microphone present; install the speech model before recording."
        return "Ready: microphone and local speech model are available."

    def _language_code(self) -> str | None:
        choice = self.language.get()
        return {"English": "en", "French": "fr"}.get(choice)

    def test_voice(self) -> None:
        sample = (
            "Bonjour, je suis Claire. Voici un aperçu de ma voix française."
            if self._language_code() == "fr"
            else "Hello, I am Claire. This is what my voice sounds like."
        )
        language = self._language_code()
        self.app.submit(
            "speak",
            lambda: self.runtime.speak(sample, language=language),
            lambda engine: self.transcript_note.configure(text=f"Spoken with {engine}."),
            lambda error: self.transcript_note.configure(text=f"Voice error: {error}"),
        )

    def record(self) -> None:
        from claireai.voice_input import SpeechRecognitionUnavailable, transcriber_for

        transcriber = transcriber_for(str(self.runtime.data_dir / "models" / "voice" / "asr"))
        self.transcript_note.configure(text="Recording for five seconds …")

        def capture():  # noqa: ANN202
            import subprocess

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                path = handle.name
            try:
                subprocess.run(
                    ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "5", "-t", "wav", path],
                    capture_output=True, timeout=20, check=False,
                )
                audio = Path(path).read_bytes()
                if len(audio) <= 44:
                    raise SpeechRecognitionUnavailable("no audio was captured from the microphone")
                return transcriber.transcribe(audio, content_type="audio/wav")
            finally:
                Path(path).unlink(missing_ok=True)

        self.app.submit(
            "record",
            capture,
            lambda text: self.transcript_note.configure(text=f"You said: {text}"),
            lambda error: self.transcript_note.configure(text=f"Speech input error: {error}"),
        )

    def save(self) -> None:
        self.runtime.update_preferences(
            {"voice_pitch": int(self.pitch.get()), "voice_rate": int(self.rate.get()), "voice_enabled": True}
        )
        self.transcript_note.configure(text="Voice settings saved.")

    def on_show(self) -> None:
        voices = self.runtime.status()["voices"]
        self.engine_note.configure(
            text="eSpeak NG found · voices: " + ", ".join(voices.values())
            if voices
            else "No local speech engine found (install espeak-ng)"
        )
        self.mic_note.configure(text=self._microphone_note())


class SettingsPage(Page):
    title = "Settings"
    subtitle = "Stored locally. Bounds match the runtime's own validation."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        preferences = self.runtime.preferences()
        control = self.runtime.control()

        form = ttk.LabelFrame(self.frame, text="Assistant", padding=12)
        form.pack(fill="x")
        self.owner = tk.StringVar(value=str(preferences["owner_name"]))
        self.style = tk.StringVar(value=str(preferences["response_style"]))
        self.max_tokens = tk.IntVar(value=int(preferences["max_output_tokens"]))
        self.max_steps = tk.IntVar(value=int(preferences["agent_max_steps"]))
        self.voice_enabled = tk.BooleanVar(value=bool(preferences["voice_enabled"]))

        self._field(form, "Owner name", ttk.Entry(form, textvariable=self.owner, width=36))
        self._field(form, "Response style", ttk.Combobox(form, textvariable=self.style, state="readonly",
                                                        values=["concise", "balanced", "detailed", "playful"]))
        self._field(form, "Max answer tokens", ttk.Spinbox(form, from_=8, to=1024, textvariable=self.max_tokens, width=10))
        self._field(form, "Max tool steps", ttk.Spinbox(form, from_=1, to=24, textvariable=self.max_steps, width=10))
        self._field(form, "Speak answers", ttk.Checkbutton(form, variable=self.voice_enabled))

        switches = ttk.LabelFrame(self.frame, text="Capabilities", padding=12)
        switches.pack(fill="x", pady=(12, 0))
        self.browser = tk.BooleanVar(value=bool(control["browser"]))
        self.improvement = tk.BooleanVar(value=bool(control["self_improvement"]))
        self.shutdown = tk.BooleanVar(value=bool(control["shutdown"]))
        self._field(switches, "Web tools", ttk.Checkbutton(switches, text="allow browser and downloads",
                                                          variable=self.browser))
        self._field(switches, "Self-improvement", ttk.Checkbutton(switches, text="allow training candidates",
                                                                 variable=self.improvement))
        self._field(switches, "Assistant", ttk.Checkbutton(switches, text="stopped (emergency switch)",
                                                           variable=self.shutdown))

        folders = ", ".join(self.runtime.status()["allowed_folders"]) or "none"
        ttk.Label(self.frame, text=f"Allowed folders: {folders}", style="Muted.TLabel",
                  wraplength=800).pack(anchor="w", pady=(12, 0))
        ttk.Label(self.frame, text="Change folders with CLAIREAI_ALLOWED_FOLDERS; they are never widened from here.",
                  style="Muted.TLabel", wraplength=800).pack(anchor="w")

        ttk.Button(self.frame, text="Apply", command=self.apply).pack(anchor="w", pady=(14, 0))
        self.result = ttk.Label(self.frame, text="", style="Muted.TLabel")
        self.result.pack(anchor="w", pady=(6, 0))

    def _field(self, parent: tk.Widget, label: str, widget: tk.Widget) -> None:
        line = ttk.Frame(parent)
        line.pack(fill="x", pady=4)
        ttk.Label(line, text=label, width=20).pack(side="left")
        widget.pack(side="left")

    def apply(self) -> None:
        try:
            self.runtime.update_preferences(
                {
                    "owner_name": self.owner.get().strip() or "Owner",
                    "response_style": self.style.get(),
                    "max_output_tokens": int(self.max_tokens.get()),
                    "agent_max_steps": int(self.max_steps.get()),
                    "voice_enabled": bool(self.voice_enabled.get()),
                }
            )
            self.runtime.set_control(
                browser=bool(self.browser.get()),
                self_improvement=bool(self.improvement.get()),
                shutdown=bool(self.shutdown.get()),
            )
        except (ValueError, OSError) as error:
            self.result.configure(text=f"Not saved: {error}")
            return
        self.result.configure(text="Saved.")

    def on_show(self) -> None:
        control = self.runtime.control()
        self.browser.set(bool(control["browser"]))
        self.improvement.set(bool(control["self_improvement"]))
        self.shutdown.set(bool(control["shutdown"]))


class AboutPage(Page):
    title = "About"
    subtitle = "Where everything lives and what is actually available."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        self.body = tk.Text(self.frame, wrap="word", relief="flat", background="#12161f",
                            foreground="#e8ecf4", height=20)
        self.body.pack(fill="both", expand=True)
        ttk.Button(self.frame, text="Refresh", command=self.refresh).pack(anchor="w", pady=(10, 0))

    def refresh(self) -> None:
        status = self.runtime.status()
        from claireai import __version__

        lines = [
            f"ClaireAI {__version__}",
            "",
            f"Data directory   {status['data_dir']}",
            f"Database         {status['database']}",
            f"Model checkpoint {status['checkpoint']}",
            f"Model trained    {'yes' if status['model_ready'] else 'no — train with brain-lab/train.py'}",
            f"Speech voices    {', '.join(status['voices'].values()) or 'none'}",
            f"Web tools        {'on' if status['browser'] else 'off'}",
            f"Self-improvement {'on' if status['self_improvement'] else 'off'}",
            f"Emergency stop   {'engaged' if status['shutdown'] else 'not engaged'}",
            "",
            "The language model is trained from scratch on this machine. It is a",
            "small research prototype, not a general assistant: expect short,",
            "repetitive answers until it is trained on a much larger corpus.",
        ]
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", "\n".join(lines))
        self.body.configure(state="disabled")

    def on_show(self) -> None:
        self.refresh()


PAGES: list[type[Page]] = [
    ChatPage,
    ActivityPage,
    ApprovalsPage,
    ConversationsPage,
    VoicePage,
    SettingsPage,
    AboutPage,
]
