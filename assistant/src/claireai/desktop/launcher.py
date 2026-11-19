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

    from claireai.desktop.app import main as window_main

    return window_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
