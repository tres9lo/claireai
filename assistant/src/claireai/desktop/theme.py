"""Design tokens and animation helpers for the native window.

This module deliberately imports neither ``tkinter`` nor ``customtkinter``, so
the colour maths and the easing curves can be tested on a machine with no
display. The window modules ask it for tokens and for tweens; it never touches
a widget itself beyond calling the ``apply`` callback it is handed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

# -- palette ---------------------------------------------------------------
# The same ramp the browser app uses, so the two surfaces read as one product:
# a near-black shell, one violet accent, state colours only for state.

BACKGROUND = "#08080b"
SURFACE = "#121217"
SURFACE_HIGH = "#191920"
SURFACE_HOVER = "#212129"
BORDER = "#26262f"
TEXT = "#f5f5f8"
MUTED = "#8b8b99"
ACCENT = "#8b7cff"
ACCENT_DEEP = "#241d52"
ACCENT_ALT = "#a99dff"
OK = "#3ddc97"
WARN = "#f5b455"
DANGER = "#ff6b6b"

# -- metrics ---------------------------------------------------------------

# Tk cannot load a font file, so the window uses the best family the machine
# already has. Inter/Segoe/Ubuntu/Cantarell are all proportional UI faces;
# DejaVu Sans is the guaranteed fallback on Linux.
FONT_CANDIDATES = (
    "Inter",
    "SF Pro Display",
    "Segoe UI Variable Text",
    "Segoe UI",
    "Ubuntu",
    "Cantarell",
    "Noto Sans",
    "DejaVu Sans",
)
FONT = "DejaVu Sans"
RADIUS = 16
RADIUS_SMALL = 10
GUTTER = 24

ASSETS = Path(__file__).resolve().parent / "assets"


def pick_family(available) -> str:
    """The first preferred UI font the system actually has."""
    names = {str(name).strip().lower() for name in available}
    for candidate in FONT_CANDIDATES:
        if candidate.lower() in names:
            return candidate
    return FONT


def use_family(name: str) -> None:
    """Set the module-wide family before pages are built."""
    global FONT
    FONT = name


def window_geometry(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    """A centred window that fits the screen instead of assuming 1220x820.

    Small laptops and 1366x768 panels are common; a fixed window opens partly
    off-screen there. The height is capped just under the screen so the title
    bar and panel chrome stay visible, and the minimums shrink with it.
    """
    width = min(1240, max(760, int(screen_width * 0.86)))
    height = min(860, max(560, int(screen_height * 0.84)))
    width = min(width, max(480, screen_width))
    height = min(height, max(420, screen_height))
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 3)
    return width, height, x, y


def minimum_size(screen_width: int, screen_height: int) -> tuple[int, int]:
    """Never demand more room than the display has."""
    return min(940, max(480, int(screen_width * 0.7))), min(600, max(420, int(screen_height * 0.7)))


def sidebar_width(window_width: int) -> int:
    """Full rail, then a compact glyph rail, as the window narrows."""
    if window_width >= 1080:
        return 236
    if window_width >= 900:
        return 200
    return 84


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """``#rrggbb`` to an ``(r, g, b)`` triple."""
    text = value.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected #rrggbb, got {value!r}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """An ``(r, g, b)`` triple to ``#rrggbb``, clamped to the byte range."""
    red, green, blue = (max(0, min(255, int(round(channel)))) for channel in rgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


def lerp_color(start: str, end: str, t: float) -> str:
    """Blend two colours; ``t=0`` is ``start``, ``t=1`` is ``end``."""
    t = max(0.0, min(1.0, t))
    a = hex_to_rgb(start)
    b = hex_to_rgb(end)
    return rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float) -> float:
    """Slight overshoot, for the selection pill."""
    t = max(0.0, min(1.0, t))
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda t: max(0.0, min(1.0, t)),
    "out_cubic": ease_out_cubic,
    "in_out_cubic": ease_in_out_cubic,
    "out_back": ease_out_back,
}


class Animator:
    """Time-based tweens driven by ``root.after``.

    Each tween is keyed, so starting a new tween on the same key replaces the
    old one instead of fighting it. Values are computed from the wall clock, not
    from a frame count, so a slow machine drops frames rather than playing the
    animation in slow motion.
    """

    def __init__(self, root) -> None:  # noqa: ANN001 - tkinter root or any after-capable widget
        self.root = root
        self._jobs: dict[object, str] = {}

    def tween(
        self,
        key: object,
        start: float,
        end: float,
        milliseconds: int,
        apply: Callable[[float], None],
        *,
        easing: str = "out_cubic",
        done: Callable[[], None] | None = None,
    ) -> None:
        self.cancel(key)
        curve = EASINGS.get(easing, ease_out_cubic)
        started = time.monotonic()
        duration = max(1, milliseconds) / 1000.0

        def frame() -> None:
            elapsed = time.monotonic() - started
            progress = min(1.0, elapsed / duration)
            value = start + (end - start) * curve(progress)
            try:
                apply(value)
            except Exception:  # noqa: BLE001 - a widget may vanish mid-tween
                self._jobs.pop(key, None)
                return
            if progress >= 1.0:
                self._jobs.pop(key, None)
                if done is not None:
                    done()
                return
            self._jobs[key] = self.root.after(16, frame)

        # Apply the first frame immediately so nothing flickers at the start.
        # It is guarded like every later frame: the widget may already be gone.
        try:
            apply(start)
        except Exception:  # noqa: BLE001 - a destroyed widget is not an error here
            return
        self._jobs[key] = self.root.after(16, frame)

    def tween_color(
        self,
        key: object,
        start: str,
        end: str,
        milliseconds: int,
        apply: Callable[[str], None],
        *,
        easing: str = "out_cubic",
        done: Callable[[], None] | None = None,
    ) -> None:
        self.tween(
            key,
            0.0,
            1.0,
            milliseconds,
            lambda value: apply(lerp_color(start, end, value)),
            easing=easing,
            done=done,
        )

    def pulse(
        self,
        key: object,
        base: str,
        peak: str,
        milliseconds: int,
        apply: Callable[[str], None],
    ) -> None:
        """Breathing loop: ``base`` to ``peak`` and back, forever until cancelled."""
        half = max(1, milliseconds // 2)

        def outward() -> None:
            self.tween_color(key, base, peak, half, apply, easing="in_out_cubic", done=inward)

        def inward() -> None:
            self.tween_color(key, peak, base, half, apply, easing="in_out_cubic", done=outward)

        outward()

    def cancel(self, key: object) -> None:
        job = self._jobs.pop(key, None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:  # noqa: BLE001 - the widget may already be gone
                pass

    def stop_all(self) -> None:
        for key in list(self._jobs):
            self.cancel(key)
