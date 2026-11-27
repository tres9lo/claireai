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
import tkinter.font as tkfont
import traceback
from pathlib import Path
from typing import Any, Callable

import customtkinter as ctk

from claireai.desktop import theme
from claireai.desktop.pages import PAGES, ChatPage, Page, VoicePage
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
        self._approval_dialog: ctk.CTkToplevel | None = None
        self._approval_prompted: set[str] = set()

        self._sidebar_width = 0
        self._window_icon = None
        self._brand_logo = None
        self._configure()
        self.runtime.start_background_learning()
        self.show(PAGES[0])
        self.root.after(60, self._pump)
        self.root.after(400, self._poll_approvals)
        self.root.after(700, self._restore_wakeword)
        # First token on this CPU can include a full model load. Do that while
        # the owner is looking at the window, not after they press Send.
        self.root.after(1400, self._warm_model)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # -- construction ----------------------------------------------------

    def _configure(self) -> None:
        ctk.set_appearance_mode("dark")
        self.root.configure(background=theme.BACKGROUND)
        self.root.resizable(True, True)

        # A Tk window on a scaled display is drawn at 1x and then stretched by the
        # compositor, which is what makes an otherwise fine layout look chunky and
        # pixelated. Ask the display for its real DPI and let Tk and customtkinter
        # scale from it before any widget exists.
        try:
            dpi = float(self.root.winfo_fpixels("1i"))
            if dpi > 0:
                ctk.set_widget_scaling(max(1.0, min(2.0, dpi / 96.0)))
                self.root.tk.call("tk", "scaling", dpi / 72.0)
        except (tk.TclError, ValueError):  # pragma: no cover - display dependent
            pass
        try:
            theme.use_family(theme.pick_family(tkfont.families(self.root)))
        except tk.TclError:  # pragma: no cover - no font list on a broken display
            pass

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width, height, x, y = theme.window_geometry(screen_width, screen_height)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(*theme.minimum_size(screen_width, screen_height))
        self._draw_window_icon()

        owner = self.runtime.preferences().get("owner_name") or "ClaireAI"
        self.root.title(f"ClaireAI — {owner}")

        self.sidebar = ctk.CTkFrame(
            self.root, width=theme.sidebar_width(width), corner_radius=0,
            fg_color=theme.SURFACE, border_width=0,
        )
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.brand.pack(fill="x", padx=theme.GUTTER, pady=(24, 0))
        self._brand_logo = self._brand_image(48)
        if self._brand_logo is not None:
            self.brand_logo = tk.Label(self.brand, image=self._brand_logo, bd=0,
                                       highlightthickness=0, background=theme.SURFACE)
            self.brand_logo.pack(side="left")

        self.brand_text = ctk.CTkFrame(self.brand, fg_color="transparent")
        self.brand_text.pack(side="left", padx=10)
        ctk.CTkLabel(
            self.brand_text,
            text="ClaireAI",
            font=ctk.CTkFont(family=theme.FONT, size=20, weight="bold"),
            text_color=theme.TEXT,
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            self.brand_text,
            text="private assistant",
            font=ctk.CTkFont(family=theme.FONT, size=11),
            text_color=theme.MUTED,
            anchor="w",
        ).pack(anchor="w")

        # The selected pill slides: the button colour is tweened rather than
        # swapped, and the accent marks exactly one row.
        self.nav = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.nav.pack(fill="x", padx=12, pady=(22, 0))
        for page_class in PAGES:
            button = ctk.CTkButton(
                self.nav,
                text=f"  {page_class.glyph}   {page_class.title}",
                command=lambda cls=page_class: self.show(cls),
                anchor="w",
                height=38,
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
        self.mic_badge = ctk.CTkLabel(
            footer, text="", text_color=theme.ACCENT,
            font=ctk.CTkFont(family=theme.FONT, size=10, weight="bold"),
        )
        self.mic_badge.pack(side="left", padx=(8, 0))

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

        self.root.bind("<Configure>", self._on_resize)
        self._apply_layout(width)
        self._breathe_status()

    def _draw_window_icon(self) -> None:
        """The taskbar/dock icon, from the same mark as the sidebar and favicon."""
        path = theme.ASSETS / "logo-256.png"
        try:
            self._window_icon = tk.PhotoImage(file=str(path), master=self.root)
            self.root.iconphoto(True, self._window_icon)
        except (tk.TclError, OSError):  # pragma: no cover - assets missing on an odd install
            self._window_icon = None

    def _brand_image(self, size: int):  # noqa: ANN201 - a tk image or None
        path = theme.ASSETS / f"logo-{size}.png"
        try:
            return tk.PhotoImage(file=str(path), master=self.root)
        except (tk.TclError, OSError):  # pragma: no cover - assets missing
            return None

    def _on_resize(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is not self.root:
            return
        self._apply_layout(event.width)

    def _apply_layout(self, width: int) -> None:
        """Follow the window: full rail, tight rail, glyph-only rail."""
        target = theme.sidebar_width(width)
        if target == self._sidebar_width:
            return
        self._sidebar_width = target
        compact = target < 120
        self.sidebar.configure(width=target)
        self.badge.configure(wraplength=max(60, target - 2 * theme.GUTTER))
        for page_class, button in self.buttons.items():
            if compact:
                button.configure(text=page_class.glyph, anchor="center", border_spacing=0)
            else:
                button.configure(
                    text=f"  {page_class.glyph}   {page_class.title}", anchor="w", border_spacing=6
                )
        if compact:
            self.brand_text.pack_forget()
            self.brand.pack_configure(padx=10)
            if self._brand_logo is not None:
                self.brand_logo.pack_configure(side="top", anchor="center")
            self.status_text.pack_forget()
        else:
            if self._brand_logo is not None:
                self.brand_logo.pack_configure(side="left", anchor="w")
            self.brand_text.pack(side="left", padx=10)
            self.brand.pack_configure(padx=theme.GUTTER)
            self.status_text.pack(side="left", padx=(6, 0))

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

    def _warm_model(self) -> None:
        if self._closing:
            return
        warm = getattr(self.runtime, "warm_model", None)
        if not callable(warm):
            return
        self.submit("warm", warm, lambda _result: None)

    def _poll_approvals(self) -> None:
        try:
            pending = self.runtime.pending_approvals()
        except Exception as error:  # noqa: BLE001 - keep polling
            pending = []
            self.report_error(error)
        wake = self.runtime.wakeword_status()
        self.mic_badge.configure(
            text="● mic" if wake["running"] else "",
            text_color=theme.ACCENT if wake["running"] else theme.MUTED,
        )
        if pending:
            self.flash_badge(f"{len(pending)} action(s) waiting for approval")
            self.set_status("needs you", theme.WARN)
            first = pending[0]
            ref = str(first.get("ref") or "")
            if (
                ref
                and self._approval_dialog is None
                and ref not in self._approval_prompted
            ):
                self._show_approval_dialog(first)
        elif self.runtime.stopped():
            self.flash_badge("")
            self.set_status("stopped", theme.DANGER)
        elif wake["running"]:
            self.flash_badge(f"Wake word on · say {wake.get('label') or wake['model'].replace('_', ' ')}")
            self.set_status("listening", theme.ACCENT)
        elif wake["enabled"] and wake["error"]:
            self.flash_badge(wake["error"][:200])
            self.set_status("wake word error", theme.DANGER)
        else:
            self.flash_badge("")
            self.set_status("ready", theme.OK)
        if not pending:
            self._approval_prompted.clear()
        if not self._closing:
            self.root.after(1200, self._poll_approvals)

    def _show_approval_dialog(self, item: dict[str, Any]) -> None:
        """A modern, in-window confirmation for one exact action."""
        if self._approval_dialog is not None or self._closing:
            return
        ref = str(item.get("ref") or "")
        if not ref:
            return
        self._approval_prompted.add(ref)

        dialog = ctk.CTkToplevel(self.root)
        self._approval_dialog = dialog
        dialog.title("ClaireAI permission")
        dialog.configure(fg_color=theme.BACKGROUND)
        # A normal decorated window: the title bar must keep its minimise and
        # close controls. The old fixed, non-resizable, transient dialog clipped
        # the action row on smaller screens and could not be minimised.
        dialog.resizable(True, True)
        dialog.minsize(520, 430)
        dialog.geometry("580x500")
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - 580) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - 500) // 2)
        dialog.geometry(f"+{x}+{y}")

        shell = ctk.CTkFrame(
            dialog, corner_radius=22, fg_color=theme.SURFACE,
            border_width=1, border_color=theme.BORDER,
        )
        shell.pack(fill="both", expand=True, padx=14, pady=14)

        # Reserve the action row at the bottom before long descriptions can
        # claim the height. This is what keeps Allow/Deny visible.
        actions = ctk.CTkFrame(shell, fg_color="transparent")
        actions.pack(side="bottom", fill="x", padx=22, pady=(12, 18))

        top = ctk.CTkFrame(shell, fg_color="transparent")
        top.pack(fill="x", padx=22, pady=(20, 0))
        orb = ctk.CTkLabel(
            top, text="◈", width=44, height=44, corner_radius=22,
            fg_color=theme.ACCENT_DEEP, text_color=theme.ACCENT,
            font=ctk.CTkFont(family=theme.FONT, size=22),
        )
        orb.pack(side="left")
        heading = ctk.CTkFrame(top, fg_color="transparent")
        heading.pack(side="left", padx=12)
        ctk.CTkLabel(
            heading, text="Permission needed", anchor="w", text_color=theme.ACCENT,
            font=ctk.CTkFont(family=theme.FONT, size=11, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            heading, text=str(item.get("tool") or "action").replace("_", " "),
            anchor="w", text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=20, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            top, text=str(item.get("risk") or "needs approval").replace("_", " "),
            text_color=theme.WARN, fg_color=theme.SURFACE_HIGH, corner_radius=10,
            padx=10, pady=3, font=ctk.CTkFont(family=theme.FONT, size=10, weight="bold"),
        ).pack(side="right", pady=10)

        target = str(item.get("target") or "")
        if target:
            ctk.CTkLabel(
                shell, text=target, anchor="w", justify="left", wraplength=480,
                text_color=theme.TEXT, font=ctk.CTkFont(family=theme.FONT, size=12, weight="bold"),
            ).pack(fill="x", padx=26, pady=(18, 0))

        description = str(item.get("description") or item.get("preview") or "")
        detail = ctk.CTkLabel(
            shell, text=description, anchor="w", justify="left", wraplength=500,
            height=58, text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=11),
        )
        detail.pack(fill="x", padx=26, pady=(8, 0))

        expires = int(item.get("expires_in") or 0)
        ctk.CTkLabel(
            shell,
            text=f"This exact action expires in {expires}s. Nothing else is approved.",
            anchor="w", text_color=theme.MUTED,
            font=ctk.CTkFont(family=theme.FONT, size=10),
        ).pack(fill="x", padx=26, pady=(14, 0))

        def finish(approved: bool) -> None:
            try:
                self.runtime.decide_approval(ref, approved=approved)
            except ValueError as error:
                self.flash_badge(str(error)[:200])
                self.set_status("approval expired", theme.WARN)
            else:
                self.flash_badge("Approved" if approved else "Denied")
                self.set_status("ready", theme.OK)
            close()

        def close() -> None:
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            if dialog.winfo_exists():
                dialog.destroy()
            if self._approval_dialog is dialog:
                self._approval_dialog = None

        ctk.CTkButton(
            actions, text="Deny", command=lambda: finish(False), height=38,
            corner_radius=12, fg_color=theme.SURFACE_HIGH,
            hover_color=theme.SURFACE_HOVER, text_color=theme.TEXT,
            font=ctk.CTkFont(family=theme.FONT, size=12, weight="bold"),
        ).pack(side="left")
        ctk.CTkButton(
            actions, text="Allow once", command=lambda: finish(True), height=38,
            corner_radius=12, fg_color=theme.ACCENT, hover_color=theme.ACCENT_ALT,
            text_color=theme.BACKGROUND,
            font=ctk.CTkFont(family=theme.FONT, size=12, weight="bold"),
        ).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.lift()
        dialog.focus_force()

    # -- wake word -------------------------------------------------------

    def _restore_wakeword(self) -> None:
        if not self._closing and self.runtime.wakeword_requested():
            self.start_wakeword(announce=False)

    def start_wakeword(self, *, announce: bool = True) -> None:
        try:
            self.runtime.start_wakeword(self._voice_command_ready)
        except Exception as error:  # noqa: BLE001 - the Voice page explains the fix
            self.flash_badge(f"Wake word: {error}"[:200])
            self.set_status("wake word error", theme.DANGER)
            if VoicePage in self.pages:
                self.pages[VoicePage].refresh_wakeword()
            return
        wake = self.runtime.wakeword_status()
        self.flash_badge(
            f"Wake word on · say {wake.get('label') or wake['model'].replace('_', ' ')}"
        )
        self.set_status("listening", theme.ACCENT)
        if announce:
            self.show(ChatPage)
        if VoicePage in self.pages:
            self.pages[VoicePage].refresh_wakeword()

    def stop_wakeword(self) -> None:
        self.runtime.stop_wakeword()
        self.flash_badge("Wake word off")
        self.set_status("ready", theme.OK)
        if VoicePage in self.pages:
            self.pages[VoicePage].refresh_wakeword()

    def _voice_command_ready(self, text: str) -> None:
        """Called by the listener thread; hand Tk work to the main-loop queue.

        ``root.after`` is not safe from a non-Tk thread. ``_pump`` already drains
        this queue on the main loop, so a recognized command uses the same path
        as every other background result.
        """
        self.queue.put((self._deliver_voice_command, text))

    def _deliver_voice_command(self, text: str) -> None:
        if self._closing:
            return
        self.show(ChatPage)
        page = self.pages[ChatPage]
        page.send_text(text)

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._closing = True
        self.animator.stop_all()
        self.runtime.stop_wakeword(persist=False)
        self.runtime.stop_background_learning()
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
        # A distinct WM class keeps window-manager maximise state from being
        # inherited by other Tk apps, and lets the menu entry match the window.
        root = ctk.CTk(className="ClaireAI")
    except TypeError:  # pragma: no cover - older customtkinter
        root = ctk.CTk()
    except tk.TclError as error:
        print(f"ERROR: no graphical display available ({error})", file=sys.stderr)
        return 2

    settings = Settings(data_dir=Path(args.data_dir).expanduser()) if args.data_dir else Settings()
    # A fresh data directory has no checkpoint while a checkout may already
    # hold the promoted one. Adopt it before the window opens, so the first
    # message is answered instead of pointing at a file dialog.
    try:
        from claireai.models import ensure_installed_checkpoint

        ensure_installed_checkpoint(settings)
    except Exception:  # noqa: BLE001 - the install card remains the fallback
        pass
    runtime = DesktopRuntime(settings)
    ClaireDesktop(root, runtime)
    root.mainloop()
    return 0


if __name__ == "__main__":  # pragma: no cover - window entry point
    raise SystemExit(main())
