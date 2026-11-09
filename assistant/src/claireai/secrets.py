"""Where a connector key is allowed to live.

The settings file and the environment are the documented places; this module adds
the OS keyring as the *preferred* one and a mode check for the file, so a key that
was pasted into ``.env`` can be noticed instead of quietly staying world-readable.

Nothing here ever prints a secret value: ``keyring_secret`` returns it to the
caller that needs it, and the warnings only name files and modes.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

KEYRING_SERVICE = "claireai"


def keyring_secret(account: str) -> str | None:
    """The stored secret for ``account``, or None when keyring is unusable.

    ``keyring`` is optional: a machine without it keeps working from the
    environment, which is what the connectors did before this existed.
    """
    try:
        import keyring  # noqa: PLC0415 - optional dependency
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return None
    try:
        value = keyring.get_password(KEYRING_SERVICE, account)
    except Exception:  # noqa: BLE001 - a locked/unavailable backend is not fatal
        return None
    return value or None


def stored_secret(account: str) -> str | None:
    """Keyring first, then the CLAIREAI_ environment variable for that account."""
    value = keyring_secret(account)
    if value:
        return value
    env_name = "CLAIREAI_" + account.upper().replace(":", "_").replace("-", "_")
    return os.environ.get(env_name) or None


def loose_permissions(path: Path) -> str | None:
    """Describe a permission problem on a file that may hold a key, or None.

    Only the mode is reported, never the contents.
    """
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if not stat.S_ISREG(mode):
        return None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return (
            f"{path} is accessible by group/others ({stat.S_IMODE(mode):03o}); "
            "it may hold a connector key. chmod 600 it."
        )
    return None
