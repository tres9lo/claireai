#!/usr/bin/env python3
"""Install a verified context-only expansion of the scratch checkpoint.

This is not a trained-candidate promotion and it does not relax the gate. The
script proves that every old weight is byte-for-byte equal and that positions
0..old_context-1 are unchanged. Only the new position rows differ, so behaviour
inside the old window is identical; the larger window is the new capability.

    assistant/.venv/bin/python brain-lab/install_context_upgrade.py \
        --candidate runtime-data/models/candidates/claire-context512-base.pt \
        --installed ~/.local/share/ClaireAI/models/claire-scratch.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--yes", action="store_true", help="required to replace the installed checkpoint")
    args = parser.parse_args()
    if not args.yes:
        parser.error("pass --yes after reading what this does")

    candidate = args.candidate.expanduser()
    installed = args.installed.expanduser()
    for path in (candidate, candidate.with_suffix(".json"), installed, installed.with_suffix(".json")):
        if not path.is_file():
            parser.error(f"missing checkpoint or sidecar: {path}")

    new = torch.load(candidate, map_location="cpu", weights_only=True)
    old = torch.load(installed, map_location="cpu", weights_only=True)
    new_state, old_state = new.get("state_dict"), old.get("state_dict")
    new_manifest = new.get("manifest") or {}
    old_manifest = old.get("manifest") or {}
    if not isinstance(new_state, dict) or not isinstance(old_state, dict):
        parser.error("checkpoints must contain state_dict")

    old_context = int((old_manifest.get("architecture") or {}).get("context") or 0)
    new_context = int((new_manifest.get("architecture") or {}).get("context") or 0)
    if new_context <= old_context:
        parser.error(f"new context {new_context} is not larger than {old_context}")
    if set(new_state) != set(old_state):
        parser.error("checkpoint keys differ; this is not a context-only expansion")
    for key in old_state:
        if key == "position_embedding.weight":
            continue
        if not torch.equal(old_state[key], new_state[key]):
            parser.error(f"{key} changed; refusing a non-context upgrade")
    if not torch.equal(
        old_state["position_embedding.weight"],
        new_state["position_embedding.weight"][:old_context],
    ):
        parser.error("the old position rows changed; refusing the upgrade")
    if not torch.isfinite(new_state["position_embedding.weight"]).all():
        parser.error("the expanded position table contains NaN or infinity")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = installed.parent / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    backup = archive / f"{installed.stem}-context{old_context}-{stamp}.pt"
    backup_json = backup.with_suffix(".json")
    shutil.copy2(installed, backup)
    shutil.copy2(installed.with_suffix(".json"), backup_json)
    for path in (backup, backup_json):
        path.chmod(0o600)
    shutil.copyfile(candidate, installed)
    shutil.copyfile(candidate.with_suffix(".json"), installed.with_suffix(".json"))
    installed.chmod(0o600)
    installed.with_suffix(".json").chmod(0o600)
    print(f"installed context {old_context} -> {new_context}: {installed}")
    print(f"old checkpoint archived at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
