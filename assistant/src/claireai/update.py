"""Rebuild this checkout and reinstall it over the running copy.

An installed wheel does not see later edits in the source tree. This module is
the one place that knows how to close that gap: find the checkout, build a wheel
from it, and install that wheel into the interpreter that is running ClaireAI.
The wheel goes in with ``--no-deps`` so unrelated packages are not upgraded;
the ``desktop`` extra is then reconciled explicitly, so a dependency added by
the update is installed instead of leaving a half-working app. The checkout is
remembered in the data directory, which is what makes the Settings button work
when the window was started from a desktop icon rather than the repository.

Nothing here runs on its own. It is reached from ``claireai --update`` and the
Settings page, and the caller is told to restart so the new modules are the ones
Python imports.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlparse


class UpdateError(Exception):
    """There is no checkout to build, or the build/install failed."""


SOURCE_MARKER = "update-source"


def _is_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "claireai").is_dir()


def _as_checkout(path: Path) -> Path | None:
    """Accept either the assistant directory or the repository root."""
    for candidate in (path, path / "assistant"):
        if _is_checkout(candidate):
            return candidate.resolve()
    return None


def _saved_source(data_dir: Path | None = None) -> Path | None:
    if data_dir is None:
        try:
            from claireai.config import Settings

            data_dir = Settings().resolved_data_dir
        except Exception:  # noqa: BLE001 - config is optional for this lookup
            return None
    marker = Path(data_dir) / SOURCE_MARKER
    try:
        return Path(marker.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _installed_from() -> list[Path]:
    """Paths recorded by pip for a local install, if any.

    A wheel built by Update into /tmp leaves a direct_url pointing at a file
    that is already gone; a local directory install points at the checkout.
    """
    try:
        raw = metadata.distribution("claireai").read_text("direct_url.json")
    except (metadata.PackageNotFoundError, OSError):
        return []
    if not raw:
        return []
    try:
        url = json.loads(raw).get("url", "")
    except (ValueError, AttributeError):
        return []
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return []
    path = Path(unquote(parsed.path))
    return [path, path.parent]


def _discovered_checkouts() -> list[Path]:
    """A bounded look for the checkout, for installs whose cwd is elsewhere.

    A desktop icon starts in the home directory, not in the repository. These
    roots cover the usual checkout locations without walking the whole home
    directory; the saved marker is always tried first.
    """
    roots = [Path.cwd(), *Path.cwd().parents]
    home = Path.home()
    roots += [home / name for name in ("Downloads", "Projects", "code", "src")]
    patterns = ("assistant", "*/assistant", "*/*/assistant", "*/*/*/assistant")
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir() or root in seen:
            continue
        seen.add(root)
        for pattern in patterns:
            for candidate in root.glob(pattern):
                checkout = _as_checkout(candidate)
                if checkout is not None:
                    found.append(checkout)
        if found:
            break
    return found


def remember_source(root: Path) -> Path:
    """Record the checkout, so the Settings button works from any cwd later."""
    from claireai.config import Settings

    data_dir = Settings().resolved_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / SOURCE_MARKER
    marker.write_text(str(root.resolve()), encoding="utf-8")
    try:
        marker.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return marker


def checkout_root(data_dir: Path | None = None) -> Path | None:
    """The ``assistant`` directory of a source checkout, if one can be found.

    Order: ``CLAIREAI_SOURCE``, the path remembered by the last successful
    update/install, pip's ``direct_url.json``, this file's own location (which
    is a checkout only for an editable install), then a bounded search from the
    cwd and the usual project folders. Inside a wheel this file lives in
    ``site-packages``, where no sibling ``pyproject.toml`` exists.
    """
    candidates: list[Path] = []
    override = os.environ.get("CLAIREAI_SOURCE")
    if override:
        candidates.append(Path(override).expanduser())
    saved = _saved_source(data_dir)
    if saved is not None:
        candidates.append(saved)
    candidates.extend(_installed_from())
    # This file is <repo>/assistant/src/claireai/update.py, so parents[2] is
    # the assistant package directory itself.
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.extend(_discovered_checkouts())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        checkout = _as_checkout(candidate)
        if checkout is not None:
            return checkout
    return None


def _uv() -> str | None:
    return shutil.which("uv")


def build_command(source: Path, output: Path) -> list[str]:
    uv = _uv()
    if uv:
        return [uv, "build", "--wheel", "--out-dir", str(output)]
    return [
        sys.executable, "-m", "pip", "wheel", "--no-deps",
        "--wheel-dir", str(output), str(source),
    ]


def install_command(wheel: Path) -> list[str]:
    uv = _uv()
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, "--reinstall", str(wheel)]
    return [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)]


def desktop_requirements(root: Path) -> list[str]:
    """Requirements the ``desktop`` extra promises, read from the checkout.

    A code update can add a dependency (faster-whisper, for example). Installing
    only the wheel with ``--no-deps`` would silently leave the app half-installed,
    so the extra is reconciled after the code is replaced.
    """
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UpdateError(f"cannot read {root / 'pyproject.toml'}: {exc}") from exc
    extras = data.get("project", {}).get("optional-dependencies", {})
    requirements = extras.get("desktop", [])
    return [str(item) for item in requirements]


def dependency_command(requirements: Sequence[str]) -> list[str]:
    uv = _uv()
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *requirements]
    return [sys.executable, "-m", "pip", "install", *requirements]


def _run(command: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        list(command), cwd=str(cwd), capture_output=True, text=True, timeout=600
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        tail = " | ".join(output.strip().splitlines()[-3:])
        raise UpdateError(f"command failed ({' '.join(command[:3])} …): {tail}")
    return output


def run_update(*, source: Path | None = None) -> dict[str, str]:
    """Build the checkout and install it over this interpreter's copy."""
    if source is not None:
        requested = Path(source).expanduser()
        root = _as_checkout(requested)
        if root is None:
            raise UpdateError(f"{requested} does not look like the assistant package")
    else:
        root = checkout_root()
        if root is None:
            raise UpdateError(
                "no source checkout found. Run `claireai --update --source /path/to/claireai` "
                "once (or set CLAIREAI_SOURCE); the path is remembered for the Settings button."
            )

    with tempfile.TemporaryDirectory(prefix="claireai-update-") as temporary:
        output = Path(temporary)
        _run(build_command(root, output), cwd=root)
        wheels = sorted(output.glob("claireai-*.whl"))
        if not wheels:
            raise UpdateError("the build produced no wheel")
        wheel = wheels[-1]
        _run(install_command(wheel), cwd=output)

    requirements = desktop_requirements(root)
    warning = ""
    if requirements:
        try:
            _run(dependency_command(requirements), cwd=root)
        except (UpdateError, OSError) as exc:
            # The code is already updated; a dependency that could not be fetched
            # is reported instead of pretending the install is complete.
            warning = str(exc)
    try:
        marker = remember_source(root)
    except OSError:
        marker = None
    return {
        "source": str(root),
        "remembered": str(marker) if marker else "",
        "wheel": wheel.name,
        "python": sys.executable,
        "extras": "desktop",
        "warning": warning,
    }


__all__ = [
    "SOURCE_MARKER",
    "UpdateError",
    "build_command",
    "checkout_root",
    "remember_source",
    "dependency_command",
    "desktop_requirements",
    "install_command",
    "run_update",
]
