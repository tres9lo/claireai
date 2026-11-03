"""Native window for ClaireAI: a real application window, no browser involved.

Only this module and :mod:`claireai.desktop.pages` touch widgets. Work that can
block (the model, speech, recording) runs on a worker thread and hands its result
back through a queue, so the window never freezes and Tk is only ever used from
the main thread.

The look is customtkinter over Tk: rounded surfaces, one accent colour, and
motion that is timed from the wall clock (see :mod:`claireai.desktop.theme`).
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from typing import Any, Callable

import customtkinter as ctk

from claireai.desktop import theme
from claireai.desktop.pages import PAGES, Page
from claireai.desktop.runtime import DesktopRuntime


class ClaireDesktop:
    """The window, its navigation, and the animation clock."""

    def __init__(self, root: tk.Tk, runtime: DesktopRuntime) -> None:
        self.root = root
        self.runtime = runtime
        self.queue: queue.Queue[tuple[Callable[[Any], None], Any]] = queue.Queue()
        self.pages: dict[type[Page], Page] = {}
        self.buttons: dict[type[Page], ctk.CTkButton] = {}
        # customtkinter reports "transparent" rather than a colour for unpainted
        # widgets, so the tween endpoints are tracked here instead of read back.
        self._nav_fill: dict[type[Page], str] = {}
        self._nav_text: dict[type[Page], str] = {}
        self.current: type[Page] | None = None
        self.animator = theme.Animator(root)
        self._closing = False

        self._configure()
        self.show(PAGES[0])
        self.root.after(60, self._pump)
        self.root.after(400, self._poll_approvals)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # -- construction ----------------------------------------------------

    def _configure(self) -> None:
        ctk.set_appearance_mode("dark")
        owner = self.runtime.preferences().get("owner_name") or "ClaireAI"
        self.root.title(f"ClaireAI — {owner}")
        self.root.geometry("1220x820")
        self.root.minsize(980, 660)
        self.root.configure(background=theme.BACKGROUND)

        self.sidebar = ctk.CTkFrame(
            self.root, width=236, corner_radius=0, fg_color=theme.SURFACE, border_width=0
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar,
            text="ClaireAI",
            font=ctk.CTkFont(family=theme.FONT, size=23, weight="bold"),
            text_color=theme.TEXT,
            anchor="w",
        ).pack(anchor="w", padx=theme.GUTTER, pady=(26, 0))

        ctk.CTkLabel(
            self.sidebar,
            text="local · offline · yours",
            font=ctk.CTkFont(family=theme.FONT, size=11),
            text_color=theme.MUTED,
            anchor="w",
        ).pack(anchor="w", padx=theme.GUTTER, pady=(0, 20))

        # The selected pill slides: each button owns a rail frame whose colour is
        # tweened, and the accent bar is packed into whichever page is current.
        self.nav = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.nav.pack(fill="x", padx=12)
        for page_class in PAGES:
            button = ctk.CTkButton(
                self.nav,
                text=f"  {page_class.title}",
                command=lambda cls=page_class: self.show(cls),
                anchor="w",
                height=40,
                corner_radius=theme.RADIUS_SMALL,
                fg_color="transparent",
                hover_color=theme.SURFACE_HOVER,
                text_color=theme.MUTED,
                font=ctk.CTkFont(family=theme.FONT, size=13),
                border_spacing=6,
            )
            button.pack(fill="x", pady=2)
            self.buttons[page_class] = button
            self._nav_fill[page_class] = theme.SURFACE
            self._nav_text[page_class] = theme.MUTED

        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=theme.GUTTER, pady=18)

        self.status_dot = ctk.CTkLabel(
            footer, text="●", text_color=theme.OK, font=ctk.CTkFont(family=theme.FONT, size=12)
        )
        self.status_dot.pack(side="left")

        self.status_text = ctk.CTkLabel(
            footer,
            text="ready",
            text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11),
            anchor="w",
        )
        self.status_text.pack(side="left", padx=(6, 0))

        self.badge = ctk.CTkLabel(
            self.sidebar,
            text="",
            text_color=theme.WARN,
            font=ctk.CTkFont(family=theme.FONT, size=11),
            justify="left",
            wraplength=200,
            anchor="w",
        )
        self.badge.pack(side="bottom", anchor="w", padx=theme.GUTTER, pady=(0, 4))

        self.content = ctk.CTkFrame(self.root, corner_radius=0, fg_color=theme.BACKGROUND)
        self.content.pack(side="left", fill="both", expand=True)

        self._breathe_status()

    # -- animation -------------------------------------------------------

    def _breathe_status(self) -> None:
        """A slow breathing dot, so the window never looks frozen."""
        self.animator.pulse("status-dot", theme.OK, theme.ACCENT, 2600, self._paint_status)

    def _paint_status(self, colour: str) -> None:
        try:
            self.status_dot.configure(text_color=colour)
        except tk.TclError:  # pragma: no cover - window closing
            pass

    def set_status(self, text: str, colour: str = theme.MUTED) -> None:
        """Sidebar status line; also retargets the breathing dot."""
        try:
            self.status_text.configure(text=text, text_color=colour)
        except tk.TclError:  # pragma: no cover - window closing
            pass
        self.animator.cancel("status-dot")
        self.animator.pulse("status-dot", colour, theme.SURFACE_HIGH, 2600, self._paint_status)

    def flash_badge(self, text: str) -> None:
        """Show a message in the sidebar footer with a short fade-in."""
        self.badge.configure(text=text)

    # -- navigation ------------------------------------------------------

    def show(self, page_class: type[Page]) -> None:
        previous = self.current
        if previous is not None and previous in self.pages:
            self.pages[previous].frame.pack_forget()
        if page_class not in self.pages:
            self.pages[page_class] = page_class(self, self.content)
        self.current = page_class
        page = self.pages[page_class]
        page.frame.pack(fill="both", expand=True)

        for cls, button in self.buttons.items():
            selected = cls is page_class
            target_fill = theme.ACCENT_DEEP if selected else theme.SURFACE
            target_text = theme.ACCENT if selected else theme.MUTED
            self.animator.tween_color(
                ("nav", cls),
                self._nav_fill[cls],
                target_fill,
                170,
                lambda colour, widget=button: widget.configure(fg_color=colour),
            )
            self.animator.tween_color(
                ("nav-text", cls),
                self._nav_text[cls],
                target_text,
                170,
                lambda colour, widget=button: widget.configure(text_color=colour),
            )
            self._nav_fill[cls] = target_fill
            self._nav_text[cls] = target_text

        self._fade_in(page.frame)
        page.on_show()

    def _fade_in(self, frame) -> None:  # noqa: ANN001 - a customtkinter frame
        """Content rises into place: a short offset plus a colour wash."""
        try:
            start = theme.SURFACE
            end = frame.cget("fg_color")
            if not isinstance(end, str):
                return
        except (tk.TclError, ValueError):  # pragma: no cover - frame gone
            return
        self.animator.tween_color(
            ("fade", id(frame)),
            start,
            end,
            190,
            lambda colour, widget=frame: widget.configure(fg_color=colour),
            easing="out_cubic",
        )

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
        if not self._closing:
            self.root.after(60, self._pump)

    def report_error(self, error: Exception) -> None:
        self.flash_badge(f"{type(error).__name__}: {error}"[:200])
        self.set_status("problem", theme.DANGER)

    def _poll_approvals(self) -> None:
        try:
            pending = self.runtime.pending_approvals()
        except Exception as error:  # noqa: BLE001 - keep polling
            pending = []
            self.report_error(error)
        if pending:
            self.flash_badge(f"{len(pending)} action(s) waiting for approval")
            self.set_status("needs you", theme.WARN)
        elif self.runtime.stopped():
            self.flash_badge("")
            self.set_status("stopped", theme.DANGER)
        else:
            self.flash_badge("")
            self.set_status("ready", theme.OK)
        if not self._closing:
            self.root.after(1200, self._poll_approvals)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._closing = True
        self.animator.stop_all()
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
        root = ctk.CTk()
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
