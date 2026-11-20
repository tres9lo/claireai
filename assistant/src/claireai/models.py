"""Find and install checkpoints that were trained here, not downloaded.

Nothing in this module accepts a pretrained model. A checkpoint qualifies only
if it carries the manifest a ``brain-lab`` run writes: ``tokenizer ==
corpus-bpe-v1`` plus the ``state_dict``/``manifest`` pair. That is the same
check the loader performs, done early enough to explain a refusal.

Discovery exists because a development checkout and an installed app keep their
data in different directories: the app's data directory starts empty while
``runtime-data/models`` already holds the owner's promoted checkpoint. On first
start the promoted file is copied in (never moved, never overwritten in place),
so the app answers with a model trained on this machine instead of asking the
owner to find a file.
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

TOKENIZER = "corpus-bpe-v1"
CANONICAL_NAME = "claire-scratch.pt"


class ModelInstallError(Exception):
    """A checkpoint is missing, unreadable, or not one of ours."""


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ModelInstallError(
            'PyTorch is needed to run a from-scratch checkpoint: pip install "claireai[model]"'
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - torch raises its own error types
        raise ModelInstallError(f"{path.name} is not a readable checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or "manifest" not in payload or "state_dict" not in payload:
        raise ModelInstallError(f"{path.name} is not a brain-lab checkpoint")
    manifest = payload["manifest"]
    if not isinstance(manifest, dict) or manifest.get("tokenizer") != TOKENIZER:
        raise ModelInstallError(
            f"{path.name} was not trained by brain-lab (expected tokenizer {TOKENIZER!r})"
        )
    return payload


@lru_cache(maxsize=16)
def _describe_cached(path: str, modified: float) -> dict[str, Any]:
    payload = _load_payload(Path(path))
    manifest = payload["manifest"]
    architecture = manifest.get("architecture") or {}
    parameters = sum(
        int(tensor.numel())
        for tensor in payload["state_dict"].values()
        if hasattr(tensor, "numel")
    )
    sidecar = Path(path).with_suffix(".json")
    evaluation = {}
    if sidecar.is_file():
        try:
            evaluation = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            evaluation = {}
    return {
        "name": str(manifest.get("name") or Path(path).stem),
        "parameters": parameters,
        "context": architecture.get("context"),
        "width": architecture.get("width"),
        "layers": architecture.get("layers"),
        "vocabulary": architecture.get("vocab_size"),
        "steps": evaluation.get("steps") if isinstance(evaluation, dict) else None,
    }


def describe_checkpoint(path: Path) -> dict[str, Any] | None:
    """Manifest facts for a from-scratch checkpoint, or ``None`` if it is not one."""
    checkpoint = Path(path)
    try:
        return _describe_cached(str(checkpoint), checkpoint.stat().st_mtime)
    except (OSError, ModelInstallError):
        return None


def candidate_directories(settings) -> list[Path]:
    """Where a locally trained checkpoint may already live, in priority order."""
    directories: list[Path] = []
    override = os.environ.get("CLAIREAI_MODEL_DIRS")
    if override:
        directories.extend(Path(item).expanduser() for item in override.split(os.pathsep) if item)
    directories.append(Path(settings.resolved_data_dir) / "models")
    # A checkout: this file is <repo>/assistant/src/claireai/models.py.
    checkout = Path(__file__).resolve().parents[3]
    directories.append(checkout / "runtime-data" / "models")
    directories.append(Path.cwd() / "runtime-data" / "models")
    directories.append(Path.cwd() / "models")
    # An installed copy lives in site-packages, so its own path says nothing
    # about the checkout. The update module knows the remembered/discovered
    # source; use the runtime-data next to it. This is still a local, trained
    # checkpoint, never a download.
    try:
        from claireai.update import checkout_root

        source = checkout_root()
    except Exception:  # noqa: BLE001 - discovery is best-effort
        source = None
    if source is not None:
        directories.append(source.parent / "runtime-data" / "models")
    seen: set[Path] = set()
    unique: list[Path] = []
    for directory in directories:
        resolved = directory.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def find_local_checkpoint(settings) -> Path | None:
    """The promoted checkpoint if present, else the newest complete training run."""
    for directory in candidate_directories(settings):
        canonical = directory / CANONICAL_NAME
        if canonical.is_file():
            return canonical
    newest: list[Path] = []
    for directory in candidate_directories(settings):
        if not directory.is_dir():
            continue
        newest.extend(
            path
            for path in directory.glob("claire-*.pt")
            if path.is_file() and not path.name.endswith(".partial")
        )
    if not newest:
        return None
    return max(newest, key=lambda path: path.stat().st_mtime)


def ensure_installed_checkpoint(settings) -> Path | None:
    """Make the configured checkpoint usable, copying in a local run if needed.

    Returns the path that is ready, or ``None`` when nothing on this machine
    qualifies. A checkpoint that is already in place is never touched.
    """
    target = settings.resolved_model_checkpoint
    if target.is_file():
        return target
    source = find_local_checkpoint(settings)
    if source is None:
        return None
    if describe_checkpoint(source) is None:
        return None
    destination = Path(settings.resolved_data_dir) / "models" / CANONICAL_NAME
    if source.resolve() == destination.resolve():
        return source if source.is_file() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".installing")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    sidecar = source.with_suffix(".json")
    if sidecar.is_file():
        shutil.copyfile(sidecar, destination.with_suffix(".json"))
    return destination


__all__ = [
    "CANONICAL_NAME",
    "ModelInstallError",
    "TOKENIZER",
    "candidate_directories",
    "describe_checkpoint",
    "ensure_installed_checkpoint",
    "find_local_checkpoint",
]
