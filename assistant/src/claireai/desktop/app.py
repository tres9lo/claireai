"""Native Tk shell for ClaireAI: a real window, no browser, no web server.

Only this module touches widgets. Work that can block (the model, speech,
recording) runs on a worker thread and hands its result back through a queue,
so the window never freezes and Tk is only ever used from the main thread.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

from claireai.desktop.pages import PAGES, Page
from claireai.desktop.runtime import DesktopRuntime

BACKGROUND = "#0d1117"
SURFACE = "#12161f"
TEXT = "#e8ecf4"
MUTED = "#7c8798"
ACCENT = "#5cc8ff"


class ClaireDesktop:
    """The window and its navigation."""

    def __init__(self, root: tk.Tk, runtime: DesktopRuntime) -> None:
        self.root = root
        self.runtime = runtime
        self.queue: queue.Queue[tuple[Callable[[Any], None], Any]] = queue.Queue()
        self.pages: dict[type[Page], Page] = {}
        self.buttons: dict[type[Page], tk.Button] = {}
        self.current: type[Page] | None = None

        self._configure()
        self.show(PAGES[0])
        self.root.after(80, self._pump)
        self.root.after(400, self._poll_approvals)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # -- construction ----------------------------------------------------

    def _configure(self) -> None:
        owner = self.runtime.preferences().get("owner_name") or "ClaireAI"
        self.root.title(f"ClaireAI — {owner}")
        self.root.geometry("1160x780")
        self.root.minsize(920, 620)
        self.root.configure(background=BACKGROUND)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover - theme availability varies
            pass
        style.configure(".", background=SURFACE, foreground=TEXT, fieldbackground="#1a2030")
        style.configure("TFrame", background=SURFACE)
        style.configure("TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Heading.TLabel", background=SURFACE, foreground=TEXT,
                        font=("DejaVu Sans", 20, "bold"))
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED)
        style.configure("TButton", padding=(12, 6))
        style.configure("TLabelframe", background=SURFACE, foreground=MUTED)
        style.configure("TLabelframe.Label", background=SURFACE, foreground=MUTED)
        style.configure("TCheckbutton", background=SURFACE, foreground=TEXT)
        style.configure("Treeview", background="#12161f", fieldbackground="#12161f", foreground=TEXT)
        style.configure("Treeview.Heading", background="#1a2030", foreground=MUTED)

        self.sidebar = tk.Frame(self.root, background=BACKGROUND, width=208)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Label(self.sidebar, text="ClaireAI", background=BACKGROUND, foreground=TEXT,
                 font=("DejaVu Sans", 20, "bold")).pack(anchor="w", padx=18, pady=(22, 0))
        tk.Label(self.sidebar, text="local · offline · yours", background=BACKGROUND,
                 foreground=MUTED, font=("DejaVu Sans", 9)).pack(anchor="w", padx=18, pady=(0, 18))

        for page_class in PAGES:
            button = tk.Button(
                self.sidebar,
                text=page_class.title,
                command=lambda cls=page_class: self.show(cls),
                relief="flat",
                anchor="w",
                padx=18,
                pady=8,
                background=BACKGROUND,
                foreground=TEXT,
                activebackground="#1a2030",
                activeforeground=ACCENT,
                borderwidth=0,
                highlightthickness=0,
                font=("DejaVu Sans", 11),
            )
            button.pack(fill="x")
            self.buttons[page_class] = button

        self.badge = tk.Label(self.sidebar, text="", background=BACKGROUND, foreground="#ffcf6b",
                              justify="left", font=("DejaVu Sans", 9), wraplength=180)
        self.badge.pack(anchor="w", side="bottom", padx=18, pady=16)

        self.content = ttk.Frame(self.root)
        self.content.pack(side="left", fill="both", expand=True)

    # -- navigation ------------------------------------------------------

    def show(self, page_class: type[Page]) -> None:
        if self.current is not None:
            self.pages[self.current].frame.pack_forget()
        if page_class not in self.pages:
            self.pages[page_class] = page_class(self, self.content)
        self.current = page_class
        page = self.pages[page_class]
        page.frame.pack(fill="both", expand=True)
        for cls, button in self.buttons.items():
            button.configure(foreground=ACCENT if cls is page_class else TEXT,
                             background="#161d2b" if cls is page_class else BACKGROUND)
        page.on_show()

    # -- threading -------------------------------------------------------

    def submit(
        self,
        kind: str,
        call: Callable[[], Any],
        on_ok: Callable[[Any], None],
        on_error: Callable[[Exception], None] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Run ``call`` off the UI thread, deliver the outcome on it."""

        def work() -> None:
            try:
                self.queue.put((on_ok, call()))
            except Exception as error:  # noqa: BLE001 - surfaced to the page
                self.queue.put((on_error or self.report_error, error))

        threading.Thread(target=work, name=name or f"claireai-{kind}", daemon=True).start()

    def _pump(self) -> None:
        while True:
            try:
                callback, value = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(value)
            except Exception:  # pragma: no cover - a broken callback must not kill the app
                traceback.print_exc()
        self.root.after(80, self._pump)

    def report_error(self, error: Exception) -> None:
        self.badge.configure(text=f"{type(error).__name__}: {error}"[:220])

    def _poll_approvals(self) -> None:
        try:
            pending = self.runtime.pending_approvals()
        except Exception as error:  # noqa: BLE001 - keep polling
            pending = []
            self.report_error(error)
        if pending:
            self.badge.configure(text=f"{len(pending)} action(s) waiting for approval\nOpen Approvals")
        elif self.runtime.stopped():
            self.badge.configure(text="stopped — turn it back on in Settings")
        else:
            self.badge.configure(text="")
        self.root.after(1200, self._poll_approvals)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self.runtime.stop_speaking()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from claireai.config import Settings

    parser = argparse.ArgumentParser(prog="claireai-desktop", description="Open the native ClaireAI window.")
    parser.add_argument("--data-dir", help="data directory (default: the OS application-data directory)")
    args = parser.parse_args(argv)

    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"ERROR: no graphical display available ({error})", file=sys.stderr)
        return 2

    settings = Settings(data_dir=Path(args.data_dir).expanduser()) if args.data_dir else Settings()
    runtime = DesktopRuntime(settings)
    ClaireDesktop(root, runtime)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - window entry point
    raise SystemExit(main())
