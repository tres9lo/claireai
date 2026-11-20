"""The `claireai-desktop` entry point, kept importable without a toolkit.

:mod:`claireai.desktop.app` and :mod:`claireai.desktop.pages` import ``tkinter``
and ``customtkinter`` at module level. That is the right place for them, but it
means the console script used to die with a ``ModuleNotFoundError`` traceback on
a core install: the failure happened while importing the module that was meant
to report it.

This module imports nothing optional. It checks the two dependencies in the
order the window needs them, prints one actionable line when one is missing,
and only then imports the real window.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

EXIT_CONFIG = 2

# customtkinter is in the `desktop` extra; Tk ships with Python on most
# platforms but is a separate OS package on Debian and its derivatives.
TOOLKIT_HINT = 'the native window needs customtkinter: pip install "claireai[desktop]"'
TK_HINT = (
    "Python's Tk bindings are missing: install python3-tk (Debian/Ubuntu) or the "
    'tk package, then pip install "claireai[desktop]"'
)


def _bitmap_fonts() -> bool:
    """Whether this Tk has fontconfig/Xft, or only the old X11 bitmap fonts.

    Some conda Tk 9 builds report ``fontsystem x11``; every glyph then comes
    from a bitmap face, so the whole window looks pixelated regardless of the
    chosen family. Better to explain that before drawing it.
    """
    if not sys.platform.startswith("linux"):
        return False
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import tkinter

        root = tkinter.Tk()
    except Exception:  # noqa: BLE001 - the window itself will report this
        return False
    try:
        root.withdraw()
        return root.tk.call("::tk::pkgconfig", "get", "fontsystem") == "x11"
    except Exception:  # noqa: BLE001 - old Tk without pkgconfig draws fine
        return False
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def main(argv: Sequence[str] | None = None) -> int:
    """Open the window, or explain in one line why it cannot open."""
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(f"ERROR: {TK_HINT}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        import customtkinter  # noqa: F401
    except ImportError:
        print(f"ERROR: {TOOLKIT_HINT}", file=sys.stderr)
        return EXIT_CONFIG

    allow_bitmap = os.environ.get("CLAIREAI_ALLOW_BITMAP_FONTS", "").strip().lower()
    if allow_bitmap not in {"1", "true", "yes"} and _bitmap_fonts():
        print(
            "ERROR: this Python's Tk has no font antialiasing (fontsystem x11), so "
            "the window would render with bitmap fonts.\n"
            "Run ./install.sh to install into a Python with Xft Tk, or use "
            "`claireai --cli`.\n"
            "Set CLAIREAI_ALLOW_BITMAP_FONTS=1 to open it anyway.",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    from claireai.desktop.app import main as window_main

    return window_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
