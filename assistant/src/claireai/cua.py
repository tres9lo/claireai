"""Optional computer-use driver.

The driver is deliberately small: it exposes the actions the permission layer
can describe and approve, and it imports the GUI packages only when an action is
actually attempted. A headless machine, a missing X display or Wayland without a
portal therefore gets a clear :class:`CuaError` instead of an import crash.

It is not a separate permission system. Callers must check
``CLAIREAI_ALLOWED_CAPABILITIES`` with :func:`claireai.permissions.check_capability`
and the tool is ``dangerous``, so every call still goes through approval.
"""

from __future__ import annotations

import io
import time
from typing import Any

MAX_COORDINATE = 100_000
MAX_TEXT = 2_000
MAX_KEY = 80


class CuaError(RuntimeError):
    """The desktop could not perform this action."""


def _backend() -> Any:
    try:
        import pyautogui  # type: ignore[import-not-found]
    except Exception as exc:  # ImportError, and pyautogui's display probe
        raise CuaError(
            "computer control is not installed or no graphical session is available; "
            "install the optional cua extra (pyautogui) and run inside the desktop session"
        ) from exc
    # pyautogui's fail-safe is valuable: moving the pointer to a screen corner
    # aborts the next action. Keep it explicitly on.
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.03
    return pyautogui


def _coordinate(value: int, name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_COORDINATE:
        raise CuaError(f"{name} must be an integer from 0 to {MAX_COORDINATE}")
    return value


class CuaDriver:
    """Thin, testable wrapper around the desktop input APIs."""

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend

    @property
    def available(self) -> bool:
        if self._backend is not None:
            return True
        try:
            import pyautogui  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def gui(self) -> Any:
        if self._backend is None:
            self._backend = _backend()
        return self._backend

    def position(self) -> tuple[int, int]:
        point = self.gui.position()
        return int(point.x), int(point.y)

    def move(self, x: int, y: int, *, duration: float = 0.2) -> tuple[int, int]:
        _coordinate(x, "x")
        _coordinate(y, "y")
        duration = max(0.0, min(float(duration), 2.0))
        self.gui.moveTo(x, y, duration=duration)
        return x, y

    def click(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        clicks: int = 1,
    ) -> tuple[int, int]:
        if button not in {"left", "middle", "right"}:
            raise CuaError("button must be left, middle or right")
        if type(clicks) is not int or not 1 <= clicks <= 3:
            raise CuaError("clicks must be an integer from 1 to 3")
        if (x is None) != (y is None):
            raise CuaError("provide both x and y, or neither")
        if x is not None and y is not None:
            _coordinate(x, "x")
            _coordinate(y, "y")
        self.gui.click(x=x, y=y, button=button, clicks=clicks)
        return self.position()

    def type_text(self, text: str, *, interval: float = 0.02) -> int:
        text = str(text)
        if not text or len(text) > MAX_TEXT:
            raise CuaError(f"text must contain 1-{MAX_TEXT} characters")
        interval = max(0.0, min(float(interval), 0.2))
        # pyautogui.write does not understand newline; press Enter between lines
        # so a multi-line note behaves the way the owner expects.
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if line:
                self.gui.write(line, interval=interval)
            if index < len(lines) - 1:
                self.gui.press("enter")
        return len(text)

    def press(self, key: str) -> str:
        key = str(key).strip().lower()
        if not key or len(key) > MAX_KEY:
            raise CuaError("a key name is required")
        if not all(part.replace("_", "").replace("-", "").isalnum() for part in key.split("+")):
            raise CuaError("key names may contain letters, digits, underscore and plus")
        if "+" in key:
            self.gui.hotkey(*key.split("+"))
        else:
            self.gui.press(key)
        return key

    def scroll(self, amount: int) -> int:
        if type(amount) is not int or not -20 <= amount <= 20:
            raise CuaError("scroll amount must be an integer from -20 to 20")
        # Positive means up, matching the tool schema and pyautogui.
        self.gui.scroll(amount)
        return amount

    def screenshot(self) -> bytes:
        """PNG bytes of the primary display, kept in memory.

        ``mss`` is used when present because it works on X11 and Windows without
        a toolkit dependency. The result is returned to the caller; this method
        never writes a file.
        """
        try:
            import mss  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except Exception as exc:
            raise CuaError(
                "screen capture needs the optional cua extra (mss and Pillow)"
            ) from exc
        try:
            with mss.mss() as screen:
                raw = screen.grab(screen.monitors[1])
                image = Image.frombytes("RGB", raw.size, raw.rgb)
                output = io.BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()
        except Exception as exc:
            raise CuaError(f"screen capture failed: {type(exc).__name__}") from exc


def wait_for_stop(stop: Any, seconds: float) -> bool:
    """Small helper for multi-step control loops; returns True when stopped."""
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if stop.is_set():
            return True
        time.sleep(0.01)
    return bool(stop.is_set())
