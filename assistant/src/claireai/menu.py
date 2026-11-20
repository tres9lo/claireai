"""Install a desktop menu entry that runs the *working* interpreter.

A `pip install` into an environment whose Tk lacks Xft still starts, but every
glyph is a bitmap and the window looks pixelated. The installer chooses a Python
with a usable Tk and creates its own environment; the menu entry uses that
interpreter explicitly, so launching from the applications menu does not depend
on whichever `claireai` happens to be first on PATH.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

FILE_NAME = "claireai.desktop"


def entry_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "applications" / FILE_NAME


def _icon() -> Path:
    return Path(__file__).resolve().parent / "desktop" / "assets" / "logo-256.png"


def icon_path() -> Path:
    """The icon in the hicolor theme, where every desktop environment looks."""
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "icons" / "hicolor" / "256x256" / "apps" / "claireai.png"


def _command() -> list[str]:
    return [sys.executable, "-m", "claireai.app"]


def render(command: list[str] | None = None) -> str:
    argv = command or _command()
    exec_line = " ".join(f'"{part}"' if " " in part else part for part in argv)
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Name=ClaireAI",
            "Comment=Local-first assistant: desktop window, no browser",
            f"Exec={exec_line}",
            "Icon=claireai",
            "Terminal=false",
            "Categories=Utility;",
            # Tk capitalises the class it reports, so this is what the window
            # manager actually sees; the older ClaireAI spelling missed the entry.
            "StartupWMClass=Claireai",
            "StartupNotify=true",
            "",
        ]
    )


def install(command: list[str] | None = None) -> Path:
    path = entry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # An absolute Icon= path is allowed by the spec but is often not picked up
    # when the file lives inside a virtual environment. The hicolor name is the
    # portable form and covers GNOME, KDE, XFCE and docks.
    target = icon_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_icon(), target)
        target.chmod(0o644)
    except OSError:
        pass
    path.write_text(render(command), encoding="utf-8")
    try:
        path.chmod(0o644)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return path


def uninstall() -> bool:
    path = entry_path()
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    icon_path().unlink(missing_ok=True)
    return True


def status() -> dict[str, str | bool]:
    path = entry_path()
    return {
        "installed": path.is_file(),
        "path": str(path),
        "icon": str(icon_path()),
        "command": " ".join(_command()),
    }


__all__ = ["FILE_NAME", "entry_path", "icon_path", "install", "render", "status", "uninstall"]
