"""Small cross-process controls shared by the local UI and agent runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_STATE = {"shutdown": False, "browser": False, "self_improvement": False}


class RuntimeControl:
    """Atomic, local-only state file for runtime and opt-in switches."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, bool]:
        try:
            payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return dict(DEFAULT_STATE)
        except (OSError, ValueError):
            # A corrupted or unreadable control file must fail closed.
            return {"shutdown": True, "browser": False, "self_improvement": False}
        if not isinstance(payload, dict):
            return {"shutdown": True, "browser": False, "self_improvement": False}
        return {key: payload.get(key) is True for key in DEFAULT_STATE}

    def update(self, changes: dict[str, Any]) -> dict[str, bool]:
        if not changes or set(changes) - set(DEFAULT_STATE):
            raise ValueError("control update must contain shutdown, browser, and/or self_improvement")
        if any(type(value) is not bool for value in changes.values()):
            raise ValueError("control values must be booleans")
        state = self.read()
        state.update(changes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, self.path)
        return state
