"""Native (Tk) desktop app: no browser, no local web server.

Run with ``claireai-desktop``, ``python -m claireai.desktop``, or
``claireai --install-autostart`` to have it start with the session.
"""

from claireai.desktop.runtime import DesktopRuntime, Turn

__all__ = ["DesktopRuntime", "Turn"]
