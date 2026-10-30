"""Permission layer: the folder allow-list and the risk policy.

Two separate questions, deliberately kept apart:

* **Where may a tool touch?** :class:`Sandbox` answers this. It resolves a path,
  follows symlinks, and refuses anything that is not inside an allow-listed
  folder. This is code, never a model decision (roadmap 7.2).
* **May this tool run at all?** :func:`check_risk` answers this. Week 2 ships
  only read-only ``safe`` tools; anything needing approval is refused until the
  approval flow exists (roadmap 7.1).

Nothing here reads model output for permission decisions: the model proposes
arguments, this module decides whether they are allowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from claireai.config import Settings

RISK_LEVELS = ("safe", "needs_approval", "dangerous")


class PermissionDenied(Exception):
    """A path or tool the owner's policy does not allow."""


def check_risk(risk: str, *, approval_available: bool = False) -> None:
    """Refuse tools that are not safe yet.

    ``approval_available`` exists so Week 3 can pass ``True`` once the approval
    prompt is real. Until then every non-safe tool is refused rather than
    silently run: an unimplemented approval flow must never mean "allowed".
    """
    if risk not in RISK_LEVELS:
        raise PermissionDenied(f"unknown risk level {risk!r}")
    if risk == "safe":
        return
    if risk == "needs_approval" and approval_available:
        return
    raise PermissionDenied(
        f"{risk} tools are not available yet: this build has no approval flow, "
        "so the action was refused rather than run unapproved"
    )


@dataclass(frozen=True)
class Sandbox:
    """Resolved allow-list. An empty sandbox allows nothing."""

    roots: tuple[Path, ...] = ()

    @classmethod
    def from_settings(cls, settings: Settings) -> "Sandbox":
        return cls(roots=settings.allowed_folder_roots)

    @property
    def is_empty(self) -> bool:
        return not self.roots

    def describe(self) -> str:
        if self.is_empty:
            return "(no folders are allow-listed)"
        return ", ".join(str(root) for root in self.roots)

    def resolve(self, raw: str | os.PathLike[str]) -> Path:
        """Return the absolute path if it is inside an allow-listed folder.

        Symlinks are followed *before* the containment check, so a link inside a
        trusted folder that points outside it is refused. Relative names are
        accepted only when they clearly match one allow-listed folder or one of
        its children, because a bare ``Downloads`` from a model is common and
        resolving it against the process working directory would surprise the
        owner.
        """
        if not self.roots:
            raise PermissionDenied(
                "no folders are allow-listed, so file tools are disabled. "
                "Set CLAIREAI_ALLOWED_FOLDERS to the folders ClaireAI may read."
            )
        text = str(raw).strip()
        if not text:
            raise PermissionDenied("a path is required")

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self._relative_to_root(candidate)

        resolved = candidate.resolve(strict=False)
        for root in self.roots:
            if resolved == root or root in resolved.parents:
                return resolved
        raise PermissionDenied(
            f"{text!r} is outside the allow-listed folders ({self.describe()}). "
            "Ask the owner to add the folder to CLAIREAI_ALLOWED_FOLDERS."
        )

    def _relative_to_root(self, candidate: Path) -> Path:
        relative = Path(candidate)
        for root in self.roots:
            # `Downloads` for a root of `~/Downloads`, or `Downloads/report.pdf`.
            if relative == Path(root.name):
                return root
            if relative.parts and relative.parts[0] == root.name:
                return root.joinpath(*relative.parts[1:])
            joined = root / relative
            if joined.exists():
                return joined
        raise PermissionDenied(
            f"{str(candidate)!r} is a relative path that matches no allow-listed "
            f"folder ({self.describe()}); use an absolute path"
        )

    def iter_roots(self) -> Iterable[Path]:
        return iter(self.roots)
