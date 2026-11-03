"""Operating-system facts for platform-aware tool implementations."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingSystem:
    family: str
    name: str
    release: str
    machine: str
    python_platform: str


def detect_os() -> OperatingSystem:
    """Return the detected host; callers still must check feature support."""
    system = platform.system().lower()
    family = {"windows": "windows", "linux": "linux", "darwin": "macos"}.get(system, "other")
    return OperatingSystem(
        family=family,
        name=platform.system() or "Unknown",
        release=platform.release(),
        machine=platform.machine(),
        python_platform=sys.platform,
    )


def supported_host() -> bool:
    """Whether the current OS is one of the project's primary desktop targets."""
    return detect_os().family in {"linux", "windows"}
