"""Permission layer: the folder allow-list and the risk policy.

Two separate questions, deliberately kept apart:

* **Where may a tool touch?** :class:`Sandbox` answers this. It resolves a path,
  follows symlinks, and refuses anything that is not inside an allow-listed
  folder. This is code, never a model decision (roadmap 7.2).
* **May this tool run at all?** :func:`requires_approval` answers whether a
  human decision is needed, and :func:`check_risk` rejects nonsense levels.
  ``safe`` tools run freely; everything else goes through
  :mod:`claireai.approval`, and a run with no approval channel refuses
  (roadmap 7.1).

Nothing here reads model output for permission decisions: the model proposes
arguments, this module decides whether they are allowed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from claireai.config import KNOWN_CAPABILITIES, Settings

RISK_LEVELS = ("safe", "needs_approval", "dangerous")


class PermissionDenied(Exception):
    """A path or tool the owner's policy does not allow."""


def check_risk(risk: str) -> None:
    """Reject a risk level that is not part of the policy."""
    if risk not in RISK_LEVELS:
        raise PermissionDenied(f"unknown risk level {risk!r}")


def requires_approval(risk: str) -> bool:
    """True when a human must approve before the tool runs.

    ``dangerous`` and ``needs_approval`` both need a decision. They differ in
    how often: trust may later be granted for named ``needs_approval`` actions,
    while ``dangerous`` always asks (roadmap 7.1).
    """
    check_risk(risk)
    return risk != "safe"


def check_capability(capability: str, allowed: Iterable[str]) -> None:
    """Refuse a machine-access capability the owner has not enabled.

    Same shape as the folder allow-list: absent from
    ``CLAIREAI_ALLOWED_CAPABILITIES`` (empty by default), the call is refused in
    code before any device or display is touched. Enabling a capability only
    makes it *available*; each use still needs approval while no named action is
    trusted (docs/architecture/machine-access.md).
    """
    name = str(capability).strip().lower()
    if name not in KNOWN_CAPABILITIES:
        raise PermissionDenied(
            f"unknown capability {capability!r}; known: {', '.join(sorted(KNOWN_CAPABILITIES))}"
        )
    if name not in {str(item).strip().lower() for item in allowed}:
        raise PermissionDenied(
            f"capability {name!r} is not enabled. Add it to "
            "CLAIREAI_ALLOWED_CAPABILITIES to allow it for this assistant."
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
