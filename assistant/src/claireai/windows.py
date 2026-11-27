"""Small X11 window arranger for the local desktop.

The owner asked ClaireAI to move, resize and tile open application windows.  A
full window manager is not something this project should replace; this module
uses the EWMH hints that the existing desktop already publishes.  It is
optional and reports a clear error on a headless host or a native-Wayland
window that XWayland does not expose.

No window is touched until the tool's approval has been granted.  The module
itself has no permission decision; :mod:`claireai.tools.apps` checks the
``apps`` capability before it is called.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from claireai.cua import _display_auth_file


class WindowError(RuntimeError):
    """The desktop window list or one requested window is unavailable."""


@dataclass(frozen=True)
class WindowInfo:
    id: int
    title: str
    x: int
    y: int
    width: int
    height: int
    pid: int | None = None
    active: bool = False


def _clean_title(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = " ".join(str(value or "").split())
    return text[:180] or "Untitled window"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class WindowManager:
    """A thin EWMH client, with the display opened only when first used."""

    def __init__(self, display: Any = None) -> None:
        self._display = display
        self._atoms: dict[str, Any] = {}

    def _connect(self) -> Any:
        if self._display is not None:
            return self._display
        try:
            from Xlib import display as xdisplay
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise WindowError(
                "window arrangement needs python-xlib in the desktop environment"
            ) from exc

        try:
            self._display = xdisplay.Display()
        except Exception as first:
            # Xwayland on this host stores its cookie under a different auth
            # family. Reuse the same private-authority helper as CUA instead of
            # editing the owner's Xauthority file.
            target = _display_auth_file(
                os.environ.get("DISPLAY", ""), os.environ.get("XAUTHORITY")
            )
            if target is None:
                raise WindowError(
                    "the display refused window access; run ClaireAI inside the "
                    "signed-in desktop session"
                ) from first
            os.environ["XAUTHORITY"] = str(target)
            try:
                self._display = xdisplay.Display()
            except Exception as exc:
                raise WindowError(
                    "window arrangement is unavailable in this display session"
                ) from exc
        return self._display

    @property
    def display(self) -> Any:
        return self._connect()

    def _atom(self, name: str) -> Any:
        from Xlib import X

        if name not in self._atoms:
            self._atoms[name] = self.display.intern_atom(name)
        return self._atoms[name]

    def _property(self, window: Any, name: str) -> Any:
        from Xlib import X

        try:
            return window.get_full_property(self._atom(name), X.AnyPropertyType)
        except Exception:
            return None

    def _text_property(self, window: Any, name: str) -> str:
        prop = self._property(window, name)
        if prop is None:
            return ""
        return _clean_title(prop.value)

    def _title(self, window: Any) -> str:
        title = self._text_property(window, "_NET_WM_NAME")
        if title:
            return title
        try:
            return _clean_title(window.get_wm_name())
        except Exception:
            return "Untitled window"

    def _window_type(self, window: Any) -> set[int]:
        prop = self._property(window, "_NET_WM_WINDOW_TYPE")
        if prop is None:
            return set()
        try:
            return {int(item) for item in prop.value}
        except (TypeError, ValueError):
            return set()

    def _pid(self, window: Any) -> int | None:
        prop = self._property(window, "_NET_WM_PID")
        if prop is None or not prop.value:
            return None
        return _as_int(prop.value[0])

    def _is_normal_window(self, window: Any) -> bool:
        from Xlib import X

        try:
            if window.get_attributes().map_state != X.IsViewable:
                return False
        except Exception:
            return False
        types = self._window_type(window)
        if not types:
            return True
        allowed = {
            self._atom("_NET_WM_WINDOW_TYPE_NORMAL"),
            self._atom("_NET_WM_WINDOW_TYPE_DIALOG"),
        }
        return bool(types & allowed)

    def _active_id(self) -> int:
        from Xlib import X

        prop = self._property(self.display.screen().root, "_NET_ACTIVE_WINDOW")
        if prop is None or not prop.value:
            return 0
        return _as_int(prop.value[0]) or 0

    def list(self, query: str = "") -> list[WindowInfo]:
        from Xlib import X

        root = self.display.screen().root
        prop = self._property(root, "_NET_CLIENT_LIST")
        ids: list[int] = []
        if prop is not None:
            try:
                ids = [int(item) for item in prop.value]
            except (TypeError, ValueError):
                ids = []
        if not ids:
            # A bare X server may not publish EWMH's client list.
            try:
                ids = [int(item) for item in root.query_tree().children]
            except Exception:
                ids = []
        active = self._active_id()
        needle = " ".join(query.split()).casefold()
        rows: list[WindowInfo] = []
        seen: set[int] = set()
        for window_id in ids:
            if not window_id or window_id in seen:
                continue
            seen.add(window_id)
            try:
                window = self.display.create_resource_object("window", window_id)
                if not self._is_normal_window(window):
                    continue
                geometry = window.get_geometry()
                if geometry.width <= 0 or geometry.height <= 0:
                    continue
                title = self._title(window)
                if needle and needle not in title.casefold():
                    continue
                rows.append(
                    WindowInfo(
                        id=window_id,
                        title=title,
                        x=int(geometry.x),
                        y=int(geometry.y),
                        width=int(geometry.width),
                        height=int(geometry.height),
                        pid=self._pid(window),
                        active=window_id == active,
                    )
                )
            except Exception:
                continue
        return sorted(rows, key=lambda row: (not row.active, row.title.casefold()))

    def find(self, target: str) -> WindowInfo:
        wanted = " ".join(str(target or "").split())
        if not wanted:
            raise WindowError("choose a window by title or id; use list_windows first")
        if wanted.lower().startswith("0x"):
            try:
                wanted_id = int(wanted, 16)
            except ValueError:
                wanted_id = -1
            if wanted_id >= 0:
                for row in self.list():
                    if row.id == wanted_id:
                        return row
        rows = self.list()
        exact = [row for row in rows if row.title.casefold() == wanted.casefold()]
        if len(exact) == 1:
            return exact[0]
        matches = [row for row in rows if wanted.casefold() in row.title.casefold()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            names = ", ".join(f"{row.title} (0x{row.id:x})" for row in matches[:5])
            raise WindowError(f"more than one window matches {wanted!r}: {names}")
        raise WindowError(f"no open window matches {wanted!r}; use list_windows first")

    def _workarea(self) -> tuple[int, int, int, int]:
        prop = self._property(self.display.screen().root, "_NET_WORKAREA")
        if prop is not None and len(prop.value) >= 4:
            values = [_as_int(item) for item in prop.value[:4]]
            if all(value is not None for value in values):
                x, y, width, height = (int(value or 0) for value in values)
                if width > 0 and height > 0:
                    return x, y, width, height
        geometry = self.display.screen().root.get_geometry()
        return 0, 0, int(geometry.width), int(geometry.height)

    def _has_window_manager(self) -> bool:
        return self._property(self.display.screen().root, "_NET_SUPPORTING_WM_CHECK") is not None

    def _configure(self, window: Any, x: int, y: int, width: int, height: int) -> None:
        from Xlib import X
        from Xlib.protocol import event

        flags = 2 | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (2 << 12)
        message = event.ClientMessage(
            window=window,
            client_type=self._atom("_NET_MOVERESIZE_WINDOW"),
            data=(32, [flags, x, y, width, height]),
        )
        try:
            self.display.screen().root.send_event(
                message,
                event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
            )
            self.display.flush()
        except Exception:
            pass
        if not self._has_window_manager():
            window.configure(x=x, y=y, width=width, height=height)
            self.display.flush()

    def move(self, target: str, x: int, y: int) -> WindowInfo:
        row = self.find(target)
        self._configure(
            self.display.create_resource_object("window", row.id),
            x, y, row.width, row.height,
        )
        return WindowInfo(row.id, row.title, x, y, row.width, row.height, row.pid, True)

    def resize(self, target: str, width: int, height: int) -> WindowInfo:
        if width < 120 or height < 80:
            raise WindowError("a window must be at least 120 by 80 pixels")
        row = self.find(target)
        self._configure(
            self.display.create_resource_object("window", row.id),
            row.x, row.y, width, height,
        )
        return WindowInfo(row.id, row.title, row.x, row.y, width, height, row.pid, True)

    def focus(self, target: str) -> WindowInfo:
        from Xlib import X
        from Xlib.protocol import event

        row = self.find(target)
        window = self.display.create_resource_object("window", row.id)
        message = event.ClientMessage(
            window=window,
            client_type=self._atom("_NET_ACTIVE_WINDOW"),
            data=(32, [2, X.CurrentTime, 0]),
        )
        try:
            self.display.screen().root.send_event(
                message,
                event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
            )
            window.raise_window()
            window.set_input_focus(X.RevertToParent, X.CurrentTime)
            self.display.flush()
        except Exception as exc:
            raise WindowError(f"could not focus {row.title!r}: {exc}") from exc
        return WindowInfo(row.id, row.title, row.x, row.y, row.width, row.height, row.pid, True)

    def tile(self, target: str, side: str) -> WindowInfo:
        x, y, width, height = self._workarea()
        if side == "left":
            box = (x, y, max(160, width // 2), height)
        elif side == "right":
            box = (x + width // 2, y, max(160, width - width // 2), height)
        elif side == "maximize":
            box = (x, y, width, height)
        else:
            raise WindowError("side must be left, right or maximize")
        row = self.find(target)
        self._configure(
            self.display.create_resource_object("window", row.id),
            box[0], box[1], box[2], box[3],
        )
        return WindowInfo(row.id, row.title, *box, row.pid, True)

    def center(self, target: str) -> WindowInfo:
        x, y, width, height = self._workarea()
        box_width = max(320, int(width * 0.7))
        box_height = max(240, int(height * 0.8))
        box = (
            x + max(0, (width - box_width) // 2),
            y + max(0, (height - box_height) // 2),
            box_width,
            box_height,
        )
        row = self.find(target)
        self._configure(
            self.display.create_resource_object("window", row.id),
            box[0], box[1], box[2], box[3],
        )
        return WindowInfo(row.id, row.title, *box, row.pid, True)

    def tile_all(self) -> int:
        rows = [
            row for row in self.list()
            if row.pid != os.getpid() and row.title.casefold() != "claireai"
        ]
        if not rows:
            raise WindowError("there are no other open application windows to arrange")
        x, y, width, height = self._workarea()
        gap = 8
        columns = 2 if len(rows) > 1 else 1
        lines = math.ceil(len(rows) / columns)
        cell_width = max(160, (width - gap * (columns + 1)) // columns)
        cell_height = max(120, (height - gap * (lines + 1)) // lines)
        for index, row in enumerate(rows):
            column = index % columns
            line = index // columns
            box = (
                x + gap + column * (cell_width + gap),
                y + gap + line * (cell_height + gap),
                cell_width,
                cell_height,
            )
            self._configure(
                self.display.create_resource_object("window", row.id),
                box[0], box[1], box[2], box[3],
            )
        return len(rows)

    def close(self) -> None:
        if self._display is not None:
            try:
                self._display.close()
            except Exception:
                pass
            self._display = None


def format_windows(rows: list[WindowInfo]) -> str:
    if not rows:
        return "No open application windows were found."
    lines = [
        f"{index}. {row.title} — {row.width}x{row.height} at ({row.x}, {row.y}) "
        f"[0x{row.id:x}]"
        for index, row in enumerate(rows, 1)
    ]
    return "Open windows:\n" + "\n".join(lines)


__all__ = ["WindowError", "WindowInfo", "WindowManager", "format_windows"]
