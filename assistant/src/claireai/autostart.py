"""Start ClaireAI with the desktop session.

Linux uses the XDG autostart directory (``~/.config/autostart``), which GNOME,
KDE, XFCE and most other sessions honour without root. Windows and macOS are
explicitly refused here rather than writing a file that does nothing.

The launcher file is plain text, paths are absolute, and uninstalling only ever
deletes the file this module wrote.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

DESKTOP_FILE_NAME = "claireai.desktop"


class AutostartUnsupported(RuntimeError):
    """This platform has no autostart mechanism implemented."""


def autostart_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "autostart"


def desktop_file() -> Path:
    return autostart_dir() / DESKTOP_FILE_NAME


def _command() -> list[str]:
    """The command the session should run: this interpreter, this package.

    ``sys.executable`` is used as-is: resolving the symlink would point at the
    system Python and drop the virtual environment that holds ClaireAI.
    """
    return [sys.executable, "-m", "claireai.desktop"]


def _quote(argument: str) -> str:
    """Escape for a Desktop Entry Exec key, which is not a shell string."""
    if any(character in argument for character in ' \t"\'\\$`'):
        escaped = argument.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return argument


def render() -> str:
    if os.name == "nt" or sys.platform == "darwin":
        raise AutostartUnsupported("autostart is implemented for Linux desktop sessions only")
    exec_line = " ".join(_quote(part) for part in _command())
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Name=ClaireAI",
            "Comment=Local-first personal assistant",
            f"Exec={exec_line}",
            "Path=" + str(Path.home()),
            "Terminal=false",
            "X-GNOME-Autostart-enabled=true",
            "StartupNotify=false",
            "Categories=Utility;",
            "",
        ]
    )


def install(exec_command: Sequence[str] | None = None) -> Path:
    """Write the autostart entry, returning its path."""
    content = render()
    if exec_command is not None:
        exec_line = " ".join(_quote(part) for part in exec_command)
        content = "\n".join(
            f"Exec={exec_line}" if line.startswith("Exec=") else line
            for line in content.splitlines()
        ) + "\n"
    target = desktop_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    target.chmod(0o644)
    return target


def uninstall() -> bool:
    """Remove only our file; report whether there was one."""
    target = desktop_file()
    if not target.is_file():
        return False
    target.unlink()
    return True


def status() -> dict[str, object]:
    target = desktop_file()
    return {
        "supported": os.name != "nt" and sys.platform != "darwin",
        "path": str(target),
        "installed": target.is_file(),
        "command": " ".join(_command()),
        "desktop_available": shutil.which("xdg-open") is not None,
    }


__all__ = [
    "AutostartUnsupported",
    "autostart_dir",
    "desktop_file",
    "install",
    "render",
    "status",
    "uninstall",
]
