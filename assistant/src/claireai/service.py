"""Install the daily training worker as a systemd user timer.

The in-window worker is useful while ClaireAI is open. This module makes the
same worker run without a window: a user-level ``oneshot`` service, triggered by
a daily ``systemd`` timer. It needs no root, no display and no network listener.

The service respects the same **Background learning** switch and emergency stop.
If the switch is off, the unit exits successfully without training, so installing
the timer is not a hidden permission grant. The timer is deliberately daily and
persistent; a missed run happens after the machine comes back.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

SERVICE_NAME = "claireai-training.service"
TIMER_NAME = "claireai-training.timer"
SERVICE_LABEL = "ClaireAI local training"


class ServiceError(Exception):
    """The user timer could not be installed or queried."""


class ServiceUnsupported(ServiceError):
    """This host has no usable systemd user manager."""


def unit_directory() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "systemd" / "user"


def service_path() -> Path:
    return unit_directory() / SERVICE_NAME


def timer_path() -> Path:
    return unit_directory() / TIMER_NAME


def _quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_service(
    *,
    python: str | None = None,
    data_dir: Path,
    working_directory: Path | None = None,
) -> str:
    interpreter = python or sys.executable
    workdir = Path(working_directory).expanduser() if working_directory else Path.home()
    return "\n".join(
        [
            "[Unit]",
            f"Description={SERVICE_LABEL}",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=oneshot",
            f"ExecStart={_quote(interpreter)} -m claireai.app --training-run --data-dir {_quote(str(Path(data_dir).expanduser()))}",
            # systemd does not accept a quoted WorkingDirectory; it treats the
            # quote as part of the path and rejects the unit as non-absolute.
            f"WorkingDirectory={workdir}",
            "Nice=10",
            "IOSchedulingClass=idle",
            "NoNewPrivileges=true",
            "UMask=0077",
            "TimeoutStartSec=6h",
            "StandardOutput=journal",
            "StandardError=journal",
            "",
        ]
    )


def render_timer() -> str:
    return "\n".join(
        [
            "[Unit]",
            f"Description=Daily {SERVICE_LABEL}",
            "",
            "[Timer]",
            "OnCalendar=*-*-* 03:17:00",
            "RandomizedDelaySec=45m",
            "Persistent=true",
            "AccuracySec=5m",
            f"Unit={SERVICE_NAME}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def _systemctl(args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess:
    executable = shutil.which("systemctl")
    if executable is None:
        raise ServiceUnsupported("systemd is not available on this machine")
    completed = subprocess.run(
        [executable, "--user", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise ServiceError(f"systemctl --user {' '.join(args)} failed: {detail[-1] if detail else 'unknown error'}")
    return completed


def install(
    data_dir: Path,
    *,
    python: str | None = None,
    working_directory: Path | None = None,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess] | None = None,
) -> dict[str, Path]:
    """Write the units and enable the timer for the current user."""
    if shutil.which("systemctl") is None:
        raise ServiceUnsupported(
            "systemd is not available. The window worker still exists; "
            "use `claireai --training-run` from cron/Task Scheduler instead."
        )
    workdir = Path(working_directory).expanduser() if working_directory else Path.home()
    directory = unit_directory()
    directory.mkdir(parents=True, exist_ok=True)
    service = service_path()
    timer = timer_path()
    service.write_text(
        render_service(python=python, data_dir=data_dir, working_directory=workdir),
        encoding="utf-8",
    )
    timer.write_text(render_timer(), encoding="utf-8")
    for path in (service, timer):
        try:
            path.chmod(0o644)
        except OSError:
            pass
    execute = runner or (lambda args: _systemctl(args, check=True))
    execute(["daemon-reload"])
    execute(["enable", "--now", TIMER_NAME])
    return {"service": service, "timer": timer}


def uninstall(*, runner: Callable[[Sequence[str]], subprocess.CompletedProcess] | None = None) -> bool:
    execute = runner or (lambda args: _systemctl(args, check=False))
    removed = False
    if timer_path().is_file():
        try:
            execute(["disable", "--now", TIMER_NAME])
        except (OSError, ServiceError):
            pass
        removed = True
    for path in (service_path(), timer_path()):
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            pass
    if removed and shutil.which("systemctl") is not None:
        try:
            execute(["daemon-reload"])
        except (OSError, ServiceError):
            pass
    return removed


def status() -> dict[str, str | bool]:
    installed = service_path().is_file() and timer_path().is_file()
    state: dict[str, str | bool] = {
        "installed": installed,
        "service": str(service_path()),
        "timer": str(timer_path()),
        "active": "not installed",
        "enabled": "not installed",
        "next": "not installed",
    }
    if not installed:
        return state
    if shutil.which("systemctl") is None:
        state["active"] = "systemd unavailable"
        return state
    try:
        active = _systemctl(["is-active", TIMER_NAME])
        enabled = _systemctl(["is-enabled", TIMER_NAME])
        next_run = _systemctl(["show", TIMER_NAME, "--property=NextElapseUSecRealtime", "--value"])
        state["active"] = (active.stdout or active.stderr or "unknown").strip() or "unknown"
        state["enabled"] = (enabled.stdout or enabled.stderr or "unknown").strip() or "unknown"
        state["next"] = (next_run.stdout or next_run.stderr or "unknown").strip() or "unknown"
    except (OSError, ServiceError, subprocess.SubprocessError) as exc:
        state["active"] = f"query failed: {exc}"
    return state


__all__ = [
    "SERVICE_NAME",
    "TIMER_NAME",
    "ServiceError",
    "ServiceUnsupported",
    "install",
    "render_service",
    "render_timer",
    "service_path",
    "status",
    "timer_path",
    "uninstall",
    "unit_directory",
]
