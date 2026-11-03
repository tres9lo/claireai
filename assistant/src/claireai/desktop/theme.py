"""Design tokens and animation helpers for the native window.

This module deliberately imports neither ``tkinter`` nor ``customtkinter``, so
the colour maths and the easing curves can be tested on a machine with no
display. The window modules ask it for tokens and for tweens; it never touches
a widget itself beyond calling the ``apply`` callback it is handed.
"""

from __future__ import annotations

import time
from typing import Callable

# -- palette ---------------------------------------------------------------
# A dark, low-contrast shell with one saturated accent. The accents are used
# for state only (selected, waiting, refused), never as decoration.

BACKGROUND = "#080b12"
SURFACE = "#0f1520"
SURFACE_HIGH = "#151d2b"
SURFACE_HOVER = "#1b2536"
BORDER = "#1e293a"
TEXT = "#e9eef7"
MUTED = "#8592a6"
ACCENT = "#5ee0ff"
ACCENT_DEEP = "#0e3a47"
ACCENT_ALT = "#a78bfa"
OK = "#6ee7a8"
WARN = "#ffc46b"
DANGER = "#ff7b7b"

# -- metrics ---------------------------------------------------------------

FONT = "DejaVu Sans"
RADIUS = 14
RADIUS_SMALL = 9
GUTTER = 22


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
