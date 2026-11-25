"""The pages of the native ClaireAI window.

Each page is its own screen with its own job -- chat is not a dashboard, the
activity log is not squeezed under the transcript. Pages never talk to the model
directly: they call :class:`claireai.desktop.runtime.DesktopRuntime` through the
shell's worker, so the window stays responsive.

Presentation is customtkinter over Tk. Long-form text keeps a plain ``tk.Text``
because message tags and fast appends matter there, and tables keep
``ttk.Treeview`` because it is virtualised; both are restyled to match.
"""

from __future__ import annotations

import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import customtkinter as ctk

from claireai.brain import BrainUnavailable
from claireai.desktop import theme
from claireai.desktop.runtime import ModelMissingError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from claireai.desktop.app import ClaireDesktop

FONT_BODY = (theme.FONT, 12)
FONT_SMALL = (theme.FONT, 11)
FONT_TITLE = (theme.FONT, 24, "bold")
FONT_SUBTITLE = (theme.FONT, 12)


def _whole_pc_roots() -> tuple[Path, ...]:
    """Filesystem roots for the explicit whole-PC allow-list choice.

    On POSIX the root is the whole filesystem. Windows has one root per drive,
    so only the drives that currently exist are offered; a missing A: drive must
    not turn into a phantom allow-list entry.
    """
    if os.name != "nt":
        return (Path("/"),)
    roots: list[Path] = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{letter}:\\")
        try:
            if root.exists():
                roots.append(root)
        except OSError:
            continue
    return tuple(roots)


def _style_tables(root: tk.Misc) -> None:
    """One dark table style, applied once per process."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:  # pragma: no cover - theme availability varies
        pass
    style.configure(
        "Claire.Treeview",
        background=theme.SURFACE,
        fieldbackground=theme.SURFACE,
        foreground=theme.TEXT,
        rowheight=30,
        borderwidth=0,
        font=FONT_SMALL,
    )
    style.configure(
        "Claire.Treeview.Heading",
        background=theme.SURFACE_HIGH,
        foreground=theme.MUTED,
        relief="flat",
        borderwidth=0,
        font=(theme.FONT, 10, "bold"),
        padding=(8, 6),
    )
    style.map(
        "Claire.Treeview",
        background=[("selected", theme.ACCENT_DEEP)],
        foreground=[("selected", theme.TEXT)],
    )
    style.layout("Claire.Treeview", [("Claire.Treeview.treearea", {"sticky": "nswe"})])


class Page:
    """Base page: owns one frame in the stack."""

    title = "Page"
    glyph = "•"
    subtitle = ""

    def __init__(self, app: "ClaireDesktop", parent: tk.Widget) -> None:
        self.app = app
        self.frame = ctk.CTkFrame(parent, corner_radius=0, fg_color=theme.BACKGROUND)
        self.build()

    @property
    def runtime(self):  # noqa: ANN201
        return self.app.runtime

    @property
    def animator(self):  # noqa: ANN201
        return self.app.animator

    def build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def on_show(self) -> None:
        """Called every time the page becomes visible."""

    # -- shared building blocks -----------------------------------------

    def heading(self, text: str, subtitle: str = "") -> None:
        ctk.CTkLabel(
            self.frame,
            text=text,
            font=ctk.CTkFont(family=theme.FONT, size=24, weight="bold"),
            text_color=theme.TEXT,
            anchor="w",
        ).pack(anchor="w", padx=theme.GUTTER, pady=(theme.GUTTER, 0))
        if subtitle:
            ctk.CTkLabel(
                self.frame,
                text=subtitle,
                font=ctk.CTkFont(family=theme.FONT, size=12),
                text_color=theme.MUTED,
                anchor="w",
                justify="left",
                wraplength=820,
            ).pack(anchor="w", padx=theme.GUTTER, pady=(2, 14))

    def card(self, parent: tk.Misc | None = None, *, pady: tuple[int, int] = (0, 12)) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(
            parent or self.frame,
            corner_radius=theme.RADIUS,
            fg_color=theme.SURFACE,
            border_width=1,
            border_color=theme.BORDER,
        )
        frame.pack(fill="x", padx=theme.GUTTER, pady=pady)
        return frame

    def button(self, parent: tk.Misc, text: str, command, *, primary: bool = False) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=34,
            corner_radius=theme.RADIUS_SMALL,
            fg_color=theme.ACCENT if primary else theme.SURFACE_HIGH,
            hover_color=theme.ACCENT_ALT if primary else theme.SURFACE_HOVER,
            text_color=theme.BACKGROUND if primary else theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=12, weight="bold" if primary else "normal"),
        )

    def text_area(self, parent: tk.Misc, *, height: int = 6) -> tk.Text:
        return tk.Text(
            parent,
            height=height,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
            background=theme.SURFACE_HIGH,
            foreground=theme.TEXT,
            insertbackground=theme.ACCENT,
            selectbackground=theme.ACCENT_DEEP,
            font=FONT_BODY,
            spacing1=2,
            spacing3=6,
        )

    def scroll_text(self, parent: tk.Misc, *, height: int = 6) -> tuple[tk.Text, ctk.CTkScrollbar]:
        widget = self.text_area(parent, height=height)
        bar = ctk.CTkScrollbar(parent, command=widget.yview, fg_color="transparent")
        widget.configure(yscrollcommand=bar.set)
        return widget, bar


class ChatPage(Page):
    title = "Chat"
    glyph = "✦"
    subtitle = "Talk to ClaireAI. Tools run through the same approvals as always."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        self.conversation_id: str | None = None
        self.busy = False
        self._dots = 0
        self._build_notice()

        bar = ctk.CTkFrame(self.frame, fg_color="transparent")
        bar.pack(fill="x", padx=theme.GUTTER, pady=(0, 10))
        self.use_tools = tk.BooleanVar(value=True)
        self.speak_replies = tk.BooleanVar(value=False)
        self.tools_switch = ctk.CTkSwitch(
            bar, text="Let it use tools", variable=self.use_tools, progress_color=theme.ACCENT,
            text_color=theme.TEXT, font=ctk.CTkFont(family=theme.FONT, size=12),
        )
        self.tools_switch.pack(side="left")
        self.voice_switch = ctk.CTkSwitch(
            bar, text="Speak replies", variable=self.speak_replies, progress_color=theme.ACCENT,
            text_color=theme.TEXT, font=ctk.CTkFont(family=theme.FONT, size=12),
            command=self._toggle_voice,
        )
        self.voice_switch.pack(side="left", padx=18)
        self.button(bar, "Stop speaking", self._stop_speaking).pack(side="right")
        self.button(bar, "New chat", self.new_chat).pack(side="right", padx=8)
        self.model_chip = ctk.CTkLabel(
            self.frame, text="", text_color=theme.MUTED, anchor="w",
            font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.model_chip.pack(anchor="w", padx=theme.GUTTER, pady=(0, 8))

        body = ctk.CTkFrame(
            self.frame, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        self.body = body
        # Height 1: the transcript takes whatever the window leaves after the
        # composer and controls. A fixed line count is what pushed the composer
        # off the bottom of a 768px screen.
        self.transcript = self.text_area(body, height=1)
        scroll = ctk.CTkScrollbar(body, command=self.transcript.yview, fg_color="transparent")
        self.transcript.configure(yscrollcommand=scroll.set)
        self.transcript.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self.transcript.tag_configure("you", foreground=theme.ACCENT, spacing1=10)
        self.transcript.tag_configure("claire", foreground=theme.TEXT, spacing1=10)
        self.transcript.tag_configure("meta", foreground=theme.MUTED, spacing3=12)
        self.transcript.tag_configure("error", foreground=theme.DANGER, spacing3=12)
        self.transcript.configure(state="disabled")
        self._welcome()

        # Bottom-up packing: the hint, composer and status line claim their
        # space before the transcript expands, so a short window shrinks the
        # transcript instead of hiding the box you type into.
        ctk.CTkLabel(
            self.frame, text="Ctrl+Enter sends", text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=10), anchor="w",
        ).pack(side="bottom", anchor="w", padx=theme.GUTTER, pady=(2, 10))

        compose = ctk.CTkFrame(self.frame, fg_color="transparent")
        compose.pack(side="bottom", fill="x", padx=theme.GUTTER, pady=(4, 0))
        self.entry = ctk.CTkTextbox(
            compose,
            height=64,
            corner_radius=theme.RADIUS_SMALL,
            fg_color=theme.SURFACE_HIGH,
            border_width=1,
            border_color=theme.BORDER,
            text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=12),
            wrap="word",
        )
        self.entry.pack(side="left", fill="both", expand=True)
        self.entry.bind("<Control-Return>", lambda _event: self.send())
        self.send_button = self.button(compose, "Send", self.send, primary=True)
        self.send_button.configure(width=96)
        self.send_button.pack(side="right", padx=(10, 0), fill="y")

        self.activity = ctk.CTkLabel(
            self.frame,
            text="",
            text_color=theme.ACCENT,
            font=ctk.CTkFont(family=theme.FONT, size=11),
            anchor="w",
        )
        self.activity.pack(side="bottom", anchor="w", padx=theme.GUTTER, pady=(4, 0))
        body.pack(fill="both", expand=True, padx=theme.GUTTER, pady=(0, 8))

    # -- model state -----------------------------------------------------

    def _build_notice(self) -> None:
        """The space above the transcript that offers to fix a missing model."""
        self.notice = ctk.CTkFrame(
            self.frame, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.ACCENT_DEEP,
        )
        self.notice_title = ctk.CTkLabel(
            self.notice, text="", text_color=theme.TEXT, anchor="w", justify="left",
            font=ctk.CTkFont(family=theme.FONT, size=13, weight="bold"),
        )
        self.notice_title.pack(anchor="w", padx=16, pady=(12, 2))
        self.notice_text = ctk.CTkLabel(
            self.notice, text="", text_color=theme.MUTED, anchor="w", justify="left",
            wraplength=760, font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.notice_text.pack(anchor="w", padx=16)
        self.notice_actions = ctk.CTkFrame(self.notice, fg_color="transparent")
        self.notice_actions.pack(anchor="w", padx=16, pady=(8, 12))
        self.notice_button = self.button(
            self.notice_actions, "Install a checkpoint…", self._choose_checkpoint, primary=True
        )
        self.notice_button.pack(side="left")
        self.notice_secondary = self.button(
            self.notice_actions, "What can you do?", lambda: self._ask_itself()
        )
        self.notice_secondary.pack(side="left", padx=8)

    def _show_notice(self, title: str, text: str) -> None:
        self.notice_title.configure(text=title)
        self.notice_text.configure(text=text)
        self.notice.pack(fill="x", padx=theme.GUTTER, pady=(0, 10), before=self.body)

    def _hide_notice(self) -> None:
        self.notice.pack_forget()

    def refresh_model_state(self) -> None:
        status = self.runtime.model_status()
        if status["ready"]:
            self._hide_notice()
            if status["provider"] == "scratch":
                name = status.get("name") or Path(status["checkpoint"]).stem
                facts = name
                if status.get("parameters"):
                    facts += f" · {status['parameters'] / 1_000_000:.1f}M params"
                if status.get("context"):
                    facts += f" · {status['context']} context"
                self.model_chip.configure(text=f"model {facts} · trained from scratch")
            elif status["provider"] == "ollama":
                self.model_chip.configure(
                    text=f"ollama · {status.get('model') or 'Qwen'} · local"
                )
            else:
                self.model_chip.configure(text=f"{status['provider']} connector")
            return
        self.model_chip.configure(text="no model")
        self._show_notice(
            "No checkpoint installed yet",
            "I look for the checkpoint you promoted in brain-lab (runtime-data/models, "
            "or the models folder in this data directory) and start with it automatically. "
            f"If none is found, install one here — expecting {status['checkpoint']}. "
            "Only from-scratch checkpoints are accepted; nothing is downloaded. "
            "Until then I can still answer questions about myself.",
        )

    def _choose_checkpoint(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.frame,
            title="Choose a brain-lab checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")],
        )
        if not chosen:
            return
        path = Path(chosen)
        self.notice_text.configure(text=f"Checking and installing {path.name} …")

        def install():  # noqa: ANN202
            return self.runtime.install_checkpoint(path)

        self.app.submit("model", install, self._installed, self._install_failed)

    def _installed(self, status: dict) -> None:
        self.refresh_model_state()
        name = status.get("name") or Path(status["checkpoint"]).stem
        self.append("·", f"Installed {name} (archived scratch checkpoint on this machine).", "meta")
        self.append("ClaireAI", "Ready when you are — ask me anything, or try ‘what can you do?’.", "claire")

    def _install_failed(self, error: Exception) -> None:
        self.notice_text.configure(text=f"Not installed: {error}")

    def _ask_itself(self) -> None:
        self.entry.configure(state="normal")
        self.entry.delete("1.0", "end")
        self.entry.insert("1.0", "What can you do?")
        self.send()

    def _welcome(self) -> None:
        owner = self.runtime.preferences().get("owner_name") or "there"
        self.append(
            "ClaireAI",
            f"Hi {owner}. I run here, on this machine. Ask about your files or calendar, "
            "or ask what I can do — that answer comes from my real settings, not a guess.",
            "claire",
        )
        self.append("", "Ctrl+Enter sends · writes wait in Approvals · nothing is deleted permanently", "meta")

    # -- helpers ---------------------------------------------------------

    def append(self, kind: str, text: str, tag: str) -> None:
        self.transcript.configure(state="normal")
        if kind:
            self.transcript.insert("end", f"{kind}\n", (tag,))
        self.transcript.insert("end", f"{text}\n", (tag,))
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def new_chat(self) -> None:
        self.conversation_id = None
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self.activity.configure(text="")

    def _stop_speaking(self) -> None:
        self.runtime.stop_speaking()

    def _set_busy(self, busy: bool, note: str = "") -> None:
        self.busy = busy
        self.send_button.configure(state="disabled" if busy else "normal")
        if busy:
            self._start_thinking()
        else:
            self._stop_thinking()
        if note:
            self.append("·", note, "meta")

    def _start_thinking(self) -> None:
        """Three dots that actually move, so a 30-second turn looks alive."""
        self._dots = 0
        self.activity.configure(text="thinking")
        self.animator.tween_color(
            "chat-activity", theme.MUTED, theme.ACCENT, 260, lambda colour: self.activity.configure(text_color=colour)
        )

        def tick() -> None:
            if not self.busy:
                return
            self._dots = (self._dots + 1) % 4
            self.activity.configure(text="thinking" + "·" * self._dots)
            self.frame.after(320, tick)

        self.frame.after(320, tick)

    def _stop_thinking(self) -> None:
        self.activity.configure(text="")

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
        if isinstance(error, ModelMissingError) or (
            isinstance(error, BrainUnavailable)
            and ("checkpoint" in str(error) or "No such file" in str(error))
        ):
            self.refresh_model_state()
            self.append("·", "I need a local checkpoint before I can answer that — see the card above.", "meta")
            return
        self.append("!", f"{type(error).__name__}: {error}", "error")

    def _toggle_voice(self) -> None:
        """Persist the switch, and never let "off" finish with speech.

        Previously only ``on_show`` read the preference, so this switch changed
        the current window but not the saved setting (and the Voice page wrote
        ``voice_enabled: True`` whenever pitch or rate was saved). A reply could
        therefore still be spoken after the owner turned this off.
        """
        enabled = bool(self.speak_replies.get())
        try:
            self.runtime.update_preferences({"voice_enabled": enabled})
        except (ValueError, OSError) as error:
            self.append("!", f"could not save the voice setting: {error}", "error")
            return
        if not enabled:
            self.runtime.stop_speaking()

    def on_show(self) -> None:
        self.speak_replies.set(bool(self.runtime.preferences().get("voice_enabled")))
        self.refresh_model_state()


class ActivityPage(Page):
    title = "Activity"
    glyph = "◷"
    subtitle = "Every task this assistant has run, newest first."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        _style_tables(self.frame)
        bar = ctk.CTkFrame(self.frame, fg_color="transparent")
        bar.pack(fill="x", padx=theme.GUTTER, pady=(0, 10))
        self.button(bar, "Refresh", self.refresh).pack(side="left")
        self.summary = ctk.CTkLabel(
            bar, text="", text_color=theme.MUTED, font=ctk.CTkFont(family=theme.FONT, size=11)
        )
        self.summary.pack(side="left", padx=12)

        columns = ("when", "status", "request")
        self.tree = ttk.Treeview(self.frame, columns=columns, show="headings", height=6, style="Claire.Treeview")
        for column, label, width in (("when", "When", 150), ("status", "Status", 110), ("request", "Request", 520)):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=theme.GUTTER)
        self.tree.bind("<<TreeviewSelect>>", self._show_detail)

        holder = ctk.CTkFrame(
            self.frame, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        holder.pack(fill="x", padx=theme.GUTTER, pady=(10, theme.GUTTER))
        self.detail = self.text_area(holder, height=3)
        self.detail.pack(fill="x", padx=6, pady=6)
        self.detail.configure(state="disabled")

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


class DayPlanPage(Page):
    title = "Day plan"
    glyph = "▤"
    subtitle = "Assign tasks to local times. Safe actions can run; writes wait for approval."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        _style_tables(self.frame)
        self.busy = False

        bar = ctk.CTkFrame(self.frame, fg_color="transparent")
        bar.pack(fill="x", padx=theme.GUTTER, pady=(0, 8))
        ctk.CTkLabel(
            bar, text="Day", text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11),
        ).pack(side="left", padx=(0, 6))
        self.day_entry = ctk.CTkEntry(
            bar, width=150, height=34, corner_radius=theme.RADIUS_SMALL,
            fg_color=theme.SURFACE_HIGH, border_color=theme.BORDER,
            placeholder_text="today",
        )
        self.day_entry.insert(0, "today")
        self.day_entry.pack(side="left")
        self.button(bar, "Show", self.refresh).pack(side="left", padx=8)
        self.run_button = self.button(bar, "Run due now", self._run, primary=True)
        self.run_button.pack(side="left")
        self.note = ctk.CTkLabel(
            bar, text="", text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.note.pack(side="left", padx=12)

        add = ctk.CTkFrame(
            self.frame, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        add.pack(fill="x", padx=theme.GUTTER, pady=(0, 10))
        self.time_entry = ctk.CTkEntry(
            add, width=82, height=34, corner_radius=theme.RADIUS_SMALL,
            fg_color=theme.SURFACE_HIGH, border_color=theme.BORDER,
            placeholder_text="09:30",
        )
        self.time_entry.pack(side="left", padx=(10, 8), pady=10)
        self.request_entry = ctk.CTkEntry(
            add, height=34, corner_radius=theme.RADIUS_SMALL,
            fg_color=theme.SURFACE_HIGH, border_color=theme.BORDER,
            placeholder_text="What should ClaireAI do?",
        )
        self.request_entry.pack(side="left", fill="x", expand=True, pady=10)
        self.request_entry.bind("<Return>", lambda _event: self._add())
        self.button(add, "Add task", self._add, primary=True).pack(side="left", padx=8, pady=10)

        self.tree = ttk.Treeview(
            self.frame, columns=("time", "status", "request"), show="headings",
            height=6, style="Claire.Treeview",
        )
        for column, label, width in (
            ("time", "Time", 80), ("status", "Status", 130), ("request", "Request", 540)
        ):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=theme.GUTTER)
        self.refresh()

    def refresh(self) -> None:
        when = self.day_entry.get().strip() or "today"
        try:
            items = self.runtime.day_plan(when)
        except Exception as error:  # noqa: BLE001 - the page must remain usable
            self.tree.delete(*self.tree.get_children())
            self.note.configure(text=f"could not read the plan: {error}", text_color=theme.DANGER)
            return
        self.tree.delete(*self.tree.get_children())
        for item in items:
            self.tree.insert(
                "", "end", iid=str(item["id"]),
                values=(item.get("starts_at", ""), item.get("status", ""), item.get("request", "")),
            )
        self.note.configure(
            text="nothing planned" if not items else f"{len(items)} task(s)",
            text_color=theme.MUTED if not items else theme.TEXT,
        )

    def _add(self) -> None:
        when = self.day_entry.get().strip() or "today"
        starts_at = self.time_entry.get().strip()
        request = self.request_entry.get().strip()
        try:
            self.runtime.add_day_plan_item(when=when, starts_at=starts_at, request=request)
        except Exception as error:  # noqa: BLE001 - show the owner the actual validation error
            messagebox.showwarning("ClaireAI", str(error), parent=self.frame)
            return
        self.time_entry.delete(0, "end")
        self.request_entry.delete(0, "end")
        self.refresh()

    def _run(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.run_button.configure(state="disabled", text="Running…")
        self.note.configure(text="running due tasks…", text_color=theme.ACCENT)
        when = self.day_entry.get().strip() or "today"
        self.app.submit(
            "day-plan",
            lambda: self.runtime.run_day_plan(when),
            self._ran,
            self._run_failed,
        )

    def _ran(self, report) -> None:  # noqa: ANN001
        self.busy = False
        self.run_button.configure(state="normal", text="Run due now")
        self.refresh()
        items = tuple(getattr(report, "items", ()))
        if not items:
            self.note.configure(text="nothing was due", text_color=theme.MUTED)
            return
        paused = sum(1 for item in items if getattr(item, "status", "") == "awaiting_approval")
        self.note.configure(
            text=f"{len(items)} due · {paused} waiting for approval",
            text_color=theme.WARN if paused else theme.TEXT,
        )

    def _run_failed(self, error: Exception) -> None:
        self.busy = False
        self.run_button.configure(state="normal", text="Run due now")
        self.note.configure(text=f"run failed: {error}", text_color=theme.DANGER)

    def on_show(self) -> None:
        self.refresh()


class ApprovalsPage(Page):
    title = "Approvals"
    glyph = "✓"
    subtitle = "Actions that change your disk wait here. Nothing runs until you decide."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        _style_tables(self.frame)
        bar = ctk.CTkFrame(self.frame, fg_color="transparent")
        bar.pack(fill="x", padx=theme.GUTTER, pady=(0, 10))
        self.button(bar, "Approve", lambda: self._decide(True), primary=True).pack(side="left")
        self.button(bar, "Deny", lambda: self._decide(False)).pack(side="left", padx=8)
        self.button(bar, "Refresh", self.refresh).pack(side="left")
        self.note = ctk.CTkLabel(
            bar, text="", text_color=theme.MUTED, font=ctk.CTkFont(family=theme.FONT, size=11)
        )
        self.note.pack(side="left", padx=12)

        self.tree = ttk.Treeview(
            self.frame, columns=("tool", "target", "risk", "expires"),
            show="headings", height=6, style="Claire.Treeview",
        )
        for column, label, width in (("tool", "Tool", 160), ("target", "Target", 420),
                                     ("risk", "Risk", 90), ("expires", "Expires in", 100)):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=theme.GUTTER)
        self.tree.bind("<<TreeviewSelect>>", self._show_preview)

        holder = ctk.CTkFrame(
            self.frame, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        holder.pack(fill="x", padx=theme.GUTTER, pady=(10, theme.GUTTER))
        self.preview = self.text_area(holder, height=3)
        self.preview.pack(fill="x", padx=6, pady=6)
        self.preview.configure(state="disabled")
        self.refresh()
        self.frame.after(1000, self._tick)

    def refresh(self) -> None:
        pending = self.runtime.pending_approvals()
        self.tree.delete(*self.tree.get_children())
        for item in pending:
            self.tree.insert("", "end", iid=str(item["ref"]),
                             values=(item["tool"], item.get("target") or "", item.get("risk", ""),
                                     f"{item.get('expires_in', 0)}s"))
        self.note.configure(
            text="nothing waiting" if not pending else f"{len(pending)} waiting",
            text_color=theme.MUTED if not pending else theme.WARN,
        )

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

    def on_show(self) -> None:
        # The Activity page already refreshes when opened. Without this, a
        # request arriving while the page was hidden could leave the list empty
        # until the next timer tick — and a failed tick used to stop the timer.
        self.refresh()

    def _tick(self) -> None:
        try:
            if self.frame.winfo_ismapped():
                self.refresh()
        except Exception as error:  # noqa: BLE001 - the page must keep polling
            self.note.configure(text=f"refresh failed: {error}", text_color=theme.DANGER)
        if self.frame.winfo_exists():
            self.frame.after(1500, self._tick)


class ConversationsPage(Page):
    title = "History"
    glyph = "↺"
    subtitle = "Saved conversations. Deleting one removes its stored messages."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        split = ctk.CTkFrame(self.frame, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=theme.GUTTER, pady=(0, theme.GUTTER))
        split.grid_columnconfigure(0, weight=1, uniform="history")
        split.grid_columnconfigure(1, weight=2, uniform="history")
        split.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(
            split, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.listbox = tk.Listbox(
            left,
            exportselection=False,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            background=theme.SURFACE,
            foreground=theme.TEXT,
            selectbackground=theme.ACCENT_DEEP,
            selectforeground=theme.TEXT,
            activestyle="none",
            font=FONT_SMALL,
            # A Listbox has no padx/pady options; the frame around it does the spacing.
        )
        self.listbox.pack(fill="both", expand=True, padx=10, pady=8)
        self.listbox.bind("<<ListboxSelect>>", self._open)
        buttons = ctk.CTkFrame(left, fg_color="transparent")
        buttons.pack(fill="x", padx=6, pady=(0, 6))
        self.button(buttons, "Refresh", self.refresh).pack(side="left")
        self.button(buttons, "Delete", self._delete).pack(side="left", padx=8)

        right = ctk.CTkFrame(
            split, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        right.grid(row=0, column=1, sticky="nsew")
        self.view = self.text_area(right)
        view_bar = ctk.CTkScrollbar(right, command=self.view.yview, fg_color="transparent")
        self.view.configure(yscrollcommand=view_bar.set)
        self.view.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        view_bar.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self.view.configure(state="disabled")
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
    glyph = "◉"
    subtitle = "Local speech, English and French. No audio leaves this machine."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        body = ctk.CTkScrollableFrame(self.frame, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=theme.GUTTER - 6, pady=(0, 8))

        output = self.card(body)
        ctk.CTkLabel(
            output, text="Speech output", text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=16, pady=(14, 2))

        voices = self.runtime.status()["voices"]
        self.engine_note = ctk.CTkLabel(
            output,
            text=(
                "eSpeak NG found · voices: " + ", ".join(voices.values())
                if voices
                else "No local speech engine found (install espeak-ng)"
            ),
            text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11),
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self.engine_note.pack(anchor="w", padx=16)

        row = ctk.CTkFrame(output, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(12, 0))
        ctk.CTkLabel(row, text="Language", text_color=theme.TEXT,
                     font=ctk.CTkFont(family=theme.FONT, size=12)).pack(side="left")
        self.language = ctk.CTkSegmentedButton(
            row,
            values=["Automatic", "English", "French"],
            selected_color=theme.ACCENT_DEEP,
            selected_hover_color=theme.ACCENT_DEEP,
            unselected_color=theme.SURFACE_HIGH,
            unselected_hover_color=theme.SURFACE_HOVER,
            text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.language.set("Automatic")
        self.language.pack(side="left", padx=10)
        self.button(row, "Test the voice", self.test_voice, primary=True).pack(side="left")
        self.button(row, "Stop", self.runtime.stop_speaking).pack(side="left", padx=8)

        preferences = self.runtime.preferences()
        self.pitch = tk.IntVar(value=int(preferences["voice_pitch"]))
        self.rate = tk.IntVar(value=int(preferences["voice_rate"]))
        for label, variable, start, end in (("Pitch", self.pitch, 0, 99), ("Rate", self.rate, 80, 450)):
            line = ctk.CTkFrame(output, fg_color="transparent")
            line.pack(fill="x", padx=16, pady=(10, 0))
            ctk.CTkLabel(line, text=label, width=54, anchor="w", text_color=theme.MUTED,
                         font=ctk.CTkFont(family=theme.FONT, size=11)).pack(side="left")
            ctk.CTkSlider(
                line, from_=start, to=end, variable=variable,
                progress_color=theme.ACCENT, button_color=theme.ACCENT,
                button_hover_color=theme.ACCENT_ALT, fg_color=theme.SURFACE_HIGH,
            ).pack(side="left", fill="x", expand=True, padx=10)
            ctk.CTkLabel(line, textvariable=variable, width=40, text_color=theme.TEXT,
                         font=ctk.CTkFont(family=theme.FONT, size=11)).pack(side="left")

        input_box = self.card(body, pady=(0, 12))
        ctk.CTkLabel(
            input_box, text="Speech input", text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=16, pady=(14, 2))
        self.mic_note = ctk.CTkLabel(
            input_box, text=self._microphone_note(), text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11), anchor="w", justify="left", wraplength=760,
        )
        self.mic_note.pack(anchor="w", padx=16)
        row2 = ctk.CTkFrame(input_box, fg_color="transparent")
        row2.pack(fill="x", padx=16, pady=(10, 0))
        self.button(row2, "Record 5 seconds", self.record).pack(side="left")
        self.transcript_note = ctk.CTkLabel(
            input_box, text="", text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11), anchor="w", justify="left", wraplength=760,
        )
        self.transcript_note.pack(anchor="w", padx=16, pady=(10, 14))

        self.button(body, "Save voice settings", self.save, primary=True).pack(anchor="w", pady=(0, 14))

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
        self.transcript_note.configure(text="Speaking …", text_color=theme.ACCENT)
        self.app.submit(
            "speak",
            lambda: self.runtime.speak(sample, language=language),
            lambda engine: self.transcript_note.configure(text=f"Spoken with {engine}.", text_color=theme.OK),
            lambda error: self.transcript_note.configure(text=f"Voice error: {error}", text_color=theme.DANGER),
        )

    def record(self) -> None:
        from claireai.voice_input import SpeechRecognitionUnavailable, transcriber_for

        transcriber = transcriber_for(str(self.runtime.data_dir / "models" / "voice" / "asr"))
        self.transcript_note.configure(text="Recording for five seconds …", text_color=theme.ACCENT)

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
            lambda text: self.transcript_note.configure(text=f"You said: {text}", text_color=theme.TEXT),
            lambda error: self.transcript_note.configure(text=f"Speech input error: {error}", text_color=theme.DANGER),
        )

    def save(self) -> None:
        # Saving pitch/rate is not permission to speak. The old unconditional
        # ``voice_enabled: True`` overrode the Chat switch and the Settings page.
        self.runtime.update_preferences(
            {"voice_pitch": int(self.pitch.get()), "voice_rate": int(self.rate.get())}
        )
        self.transcript_note.configure(text="Voice settings saved.", text_color=theme.OK)

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
    glyph = "⚙"
    subtitle = "Stored locally. Bounds match the runtime's own validation."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        body = ctk.CTkScrollableFrame(self.frame, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=theme.GUTTER - 6, pady=(0, 8))

        preferences = self.runtime.preferences()
        control = self.runtime.control()

        form = self.card(body)
        ctk.CTkLabel(
            form, text="Assistant", text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=16, pady=(14, 6))

        self.owner = tk.StringVar(value=str(preferences["owner_name"]))
        self.style = tk.StringVar(value=str(preferences["response_style"]))
        self.max_tokens = tk.IntVar(value=int(preferences["max_output_tokens"]))
        self.max_steps = tk.IntVar(value=int(preferences["agent_max_steps"]))
        self.voice_enabled = tk.BooleanVar(value=bool(preferences["voice_enabled"]))

        self._field(form, "Owner name", lambda row: ctk.CTkEntry(
            row, textvariable=self.owner, width=280, height=32,
            corner_radius=theme.RADIUS_SMALL, fg_color=theme.SURFACE_HIGH, border_color=theme.BORDER))
        self._field(form, "Response style", lambda row: ctk.CTkOptionMenu(
            row, variable=self.style, values=["concise", "balanced", "detailed", "playful"],
            width=180, height=32, corner_radius=theme.RADIUS_SMALL,
            fg_color=theme.SURFACE_HIGH, button_color=theme.SURFACE_HOVER,
            button_hover_color=theme.ACCENT_DEEP, text_color=theme.TEXT,
        ))
        self._field(form, "Max answer tokens", lambda row: ctk.CTkEntry(
            row, textvariable=self.max_tokens, width=110, height=32,
            corner_radius=theme.RADIUS_SMALL, fg_color=theme.SURFACE_HIGH, border_color=theme.BORDER))
        self._field(form, "Max tool steps", lambda row: ctk.CTkEntry(
            row, textvariable=self.max_steps, width=110, height=32,
            corner_radius=theme.RADIUS_SMALL, fg_color=theme.SURFACE_HIGH, border_color=theme.BORDER))
        self._field(form, "Speak answers", lambda row: ctk.CTkSwitch(
            row, variable=self.voice_enabled, text="", progress_color=theme.ACCENT))

        switches = self.card(body)
        ctk.CTkLabel(
            switches, text="Capabilities", text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=16, pady=(14, 6))
        self.browser = tk.BooleanVar(value=bool(control["browser"]))
        self.improvement = tk.BooleanVar(value=bool(control["self_improvement"]))
        self.shutdown = tk.BooleanVar(value=bool(control["shutdown"]))
        self._field(switches, "Web tools", lambda row: ctk.CTkSwitch(
            row, text="allow browser and downloads", variable=self.browser, progress_color=theme.ACCENT,
            text_color=theme.TEXT, font=ctk.CTkFont(family=theme.FONT, size=11)))
        self._field(switches, "Background learning", lambda row: ctk.CTkSwitch(
            row, text="journal exchanges and build training material", variable=self.improvement,
            progress_color=theme.ACCENT, text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=11)))
        self.learning_note = ctk.CTkLabel(
            switches, text="", text_color=theme.MUTED, anchor="w", justify="left",
            wraplength=780, font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.learning_note.pack(anchor="w", padx=16, pady=(2, 12))
        self._field(switches, "Assistant", lambda row: ctk.CTkSwitch(
            row, text="stopped (emergency switch)", variable=self.shutdown,
            progress_color=theme.DANGER, text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=11)))

        access = self.card(body)
        ctk.CTkLabel(
            access, text="File access", text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=13, weight="bold"), anchor="w",
        ).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkLabel(
            access,
            text=(
                "ClaireAI can read only these folders. Writes still ask for approval "
                "every time. Choose specific folders, or explicitly allow the whole PC."
            ),
            text_color=theme.MUTED, font=ctk.CTkFont(family=theme.FONT, size=11),
            anchor="w", justify="left", wraplength=780,
        ).pack(anchor="w", padx=16, pady=(0, 8))

        saved_folders = preferences.get("allowed_folders")
        if saved_folders is None:
            saved_folders = self.runtime.status().get("allowed_folders", [])
        self.allowed_folders = [
            str(Path(path).expanduser().resolve(strict=False)) for path in saved_folders
        ]
        self.folder_list = tk.Listbox(
            access,
            height=4,
            exportselection=False,
            selectmode=tk.EXTENDED,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=theme.BORDER,
            background=theme.SURFACE_HIGH,
            foreground=theme.TEXT,
            selectbackground=theme.ACCENT_DEEP,
            selectforeground=theme.TEXT,
            activestyle="none",
            font=FONT_SMALL,
        )
        self.folder_list.pack(fill="x", padx=16, pady=(0, 8))
        folder_buttons = ctk.CTkFrame(access, fg_color="transparent")
        folder_buttons.pack(fill="x", padx=16, pady=(0, 6))
        self.button(folder_buttons, "Add folder…", self._add_folder, primary=True).pack(side="left")
        self.button(folder_buttons, "Allow whole PC…", self._allow_whole_pc).pack(side="left", padx=8)
        self.button(folder_buttons, "Remove selected", self._remove_folder).pack(side="left")
        self.folder_note = ctk.CTkLabel(
            access, text="", text_color=theme.MUTED, anchor="w", justify="left",
            wraplength=780, font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.folder_note.pack(anchor="w", padx=16, pady=(0, 14))
        self._render_folders()

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(anchor="w", pady=(14, 0))
        self.button(actions, "Apply", self.apply, primary=True).pack(side="left")
        self.button(actions, "Update app", self.update_app).pack(side="left", padx=8)
        self.result = ctk.CTkLabel(
            actions, text="", text_color=theme.MUTED, font=ctk.CTkFont(family=theme.FONT, size=11)
        )
        self.result.pack(side="left", padx=12)
        self.update_result = ctk.CTkLabel(
            body, text="", text_color=theme.MUTED, anchor="w", justify="left",
            wraplength=780, font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        self.update_result.pack(anchor="w", padx=16, pady=(8, 14))

    def _field(self, parent: tk.Misc, label: str, factory) -> None:  # noqa: ANN001
        """One label + control row. The control must be created in the row:
        a Tk widget cannot be reparented, and packing it into a row it does not
        belong to leaves it floating over the card (the old Settings bug)."""
        line = ctk.CTkFrame(parent, fg_color="transparent")
        line.pack(fill="x", padx=16, pady=5)
        ctk.CTkLabel(
            line, text=label, width=140, anchor="w", text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11),
        ).pack(side="left")
        factory(line).pack(side="left")

    def _render_folders(self) -> None:
        self.folder_list.delete(0, "end")
        for folder in self.allowed_folders:
            self.folder_list.insert("end", folder)
        if self.allowed_folders:
            self.folder_note.configure(
                text=f"{len(self.allowed_folders)} folder(s) allowed.", text_color=theme.MUTED
            )
        else:
            self.folder_note.configure(
                text="No folders yet — file tools are disabled until you add one.",
                text_color=theme.WARN,
            )

    def _save_folders(self) -> bool:
        try:
            saved = self.runtime.update_preferences({"allowed_folders": self.allowed_folders})
        except (ValueError, OSError) as error:
            self.folder_note.configure(text=f"Not saved: {error}", text_color=theme.DANGER)
            return False
        self.allowed_folders = [
            str(Path(path).expanduser().resolve(strict=False))
            for path in saved.get("allowed_folders", self.allowed_folders)
        ]
        self._render_folders()
        self.folder_note.configure(text="Folders saved.", text_color=theme.OK)
        return True

    def _add_folders(self, candidates: list[Path]) -> None:
        previous = list(self.allowed_folders)
        known = {Path(folder).expanduser().resolve(strict=False) for folder in self.allowed_folders}
        added = False
        for candidate in candidates:
            resolved = candidate.expanduser().resolve(strict=False)
            if resolved in known:
                continue
            known.add(resolved)
            self.allowed_folders.append(str(resolved))
            added = True
        if not added:
            self.folder_note.configure(text="Already allowed.", text_color=theme.MUTED)
            return
        if not self._save_folders():
            # A permission list must not show a folder that was not persisted.
            self.allowed_folders = previous
            self._render_folders()
            self.folder_note.configure(
                text="Not saved; the folder was not added.", text_color=theme.DANGER
            )

    def _add_folder(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self.frame,
            title="Choose a folder ClaireAI may use",
            mustexist=True,
        )
        if chosen:
            self._add_folders([Path(chosen)])

    def _allow_whole_pc(self) -> None:
        roots = _whole_pc_roots()
        if not roots:
            self.folder_note.configure(text="No filesystem root was found.", text_color=theme.DANGER)
            return
        names = ", ".join(str(root) for root in roots)
        if not messagebox.askyesno(
            "Allow the whole PC?",
            (
                f"ClaireAI will be able to read every file under {names}.\n\n"
                "Writes still require your approval for each action. "
                "Only continue if you understand this."
            ),
            parent=self.frame,
        ):
            return
        self._add_folders(list(roots))

    def _remove_folder(self) -> None:
        selected = list(self.folder_list.curselection())
        if not selected:
            self.folder_note.configure(text="Select a folder first.", text_color=theme.WARN)
            return
        previous = list(self.allowed_folders)
        for index in reversed(selected):
            del self.allowed_folders[index]
        if not self._save_folders():
            self.allowed_folders = previous
            self._render_folders()
            self.folder_note.configure(
                text="Not saved; the folder was not removed.", text_color=theme.DANGER
            )

    def apply(self) -> None:
        try:
            self.runtime.update_preferences(
                {
                    "owner_name": self.owner.get().strip() or "Owner",
                    "response_style": self.style.get(),
                    "max_output_tokens": int(self.max_tokens.get()),
                    "agent_max_steps": int(self.max_steps.get()),
                    "voice_enabled": bool(self.voice_enabled.get()),
                    "allowed_folders": self.allowed_folders,
                }
            )
            self.runtime.set_control(
                browser=bool(self.browser.get()),
                self_improvement=bool(self.improvement.get()),
                shutdown=bool(self.shutdown.get()),
            )
        except (ValueError, OSError) as error:
            self.result.configure(text=f"Not saved: {error}", text_color=theme.DANGER)
            return
        self.result.configure(text="Saved.", text_color=theme.OK)
        self._refresh_learning()

    def update_app(self) -> None:
        """Rebuild this checkout and install it into the running environment."""
        from claireai.update import run_update

        self.update_result.configure(
            text="Building this checkout and installing it … this can take a minute.",
            text_color=theme.ACCENT,
        )

        def build():  # noqa: ANN202
            return run_update()

        self.app.submit("update", build, self._updated, self._update_failed)

    def _updated(self, result: dict) -> None:
        text = f"Installed {result['wheel']} from {result['source']}. Restart ClaireAI to run it."
        colour = theme.OK
        if result.get("warning"):
            text += f" A dependency is still missing: {result['warning']}"
            colour = theme.WARN
        self.update_result.configure(text=text, text_color=colour)

    def _update_failed(self, error: Exception) -> None:
        self.update_result.configure(text=f"Update failed: {error}", text_color=theme.DANGER)

    def _refresh_learning(self) -> None:
        status = self.runtime.learning_status()
        if not status["enabled"]:
            self.learning_note.configure(
                text=(
                    f"Off — {status['total']} exchanges are kept from earlier sessions. "
                    "While on, new exchanges are journaled here and compiled in the background; "
                    "candidates are evaluated automatically and only a passing one is installed."
                )
            )
            return
        note = (
            f"On — {status['total']} exchanges journaled, {status['compiled']} compiled, "
            f"{status['pending']} waiting."
        )
        if status.get("last_at"):
            note += f" Last: {str(status['last_at'])[:16].replace('T', ' ')}."
        if status.get("worker_error"):
            note += f" Last compile problem: {status['worker_error']}"
        training = status.get("training") or {}
        if training:
            when = str(training.get("last_success") or training.get("last_attempt") or "")[:16]
            note += f" Training: {training.get('status', 'not started')}"
            if when:
                note += f" ({when.replace('T', ' ')})"
            if training.get("candidate"):
                note += f"; candidate {Path(str(training['candidate'])).name}"
            if training.get("worker_error"):
                note += f"; last training problem: {training['worker_error']}"
            gate = training.get("gate") or {}
            if gate:
                if gate.get("promoted"):
                    note += "; candidate passed the gate and was installed"
                elif gate.get("passed") is True:
                    note += "; candidate passed the gate"
                else:
                    reason = str(gate.get("reason") or "not evaluated")[:170]
                    note += f"; candidate not installed — {reason}"
        self.learning_note.configure(text=note)

    def on_show(self) -> None:
        control = self.runtime.control()
        self.browser.set(bool(control["browser"]))
        self.improvement.set(bool(control["self_improvement"]))
        self.shutdown.set(bool(control["shutdown"]))
        preferences = self.runtime.preferences()
        saved_folders = preferences.get("allowed_folders")
        if saved_folders is None:
            saved_folders = self.runtime.status().get("allowed_folders", [])
        self.allowed_folders = [
            str(Path(path).expanduser().resolve(strict=False)) for path in saved_folders
        ]
        self._render_folders()
        self._refresh_learning()


class AboutPage(Page):
    title = "About"
    glyph = "ⓘ"
    subtitle = "Where everything lives and what is actually available."

    def build(self) -> None:
        self.heading(self.title, self.subtitle)
        holder = ctk.CTkFrame(
            self.frame, corner_radius=theme.RADIUS, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        holder.pack(fill="both", expand=True, padx=theme.GUTTER, pady=(0, 10))
        self.body = self.text_area(holder, height=6)
        self.body.pack(fill="both", expand=True, padx=6, pady=6)
        self.body.configure(state="disabled")
        self.button(self.frame, "Refresh", self.refresh).pack(anchor="w", padx=theme.GUTTER,
                                                              pady=(0, theme.GUTTER))

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
            f"Learning         {'on' if status['self_improvement'] else 'off'}",
            f"Emergency stop   {'engaged' if status['shutdown'] else 'not engaged'}",
            "",
            "Identity and capability questions are answered from this configuration,",
            "so they stay accurate even before a model is installed.",
            "",
            "The language model is trained from scratch on this machine. It is a",
            "small research prototype, not a general assistant: expect short,",
            "repetitive answers until it is trained on a much larger corpus.",
        ]
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", "\n".join(lines))
        self.body.configure(state="disabled")


PAGES: list[type[Page]] = [
    ChatPage,
    ActivityPage,
    DayPlanPage,
    ApprovalsPage,
    ConversationsPage,
    VoicePage,
    SettingsPage,
    AboutPage,
]
