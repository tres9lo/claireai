#!/usr/bin/env python3
"""Expand a checkpoint's position table without changing its learned weights.

Positions 0..old_context-1 are copied exactly, so outputs inside the old window
are unchanged. New positions start from the mean of the last few learned rows and
are trained on the new corpus. The installed checkpoint is never overwritten.

    assistant/.venv/bin/python brain-lab/expand_context.py \
        --source ~/.local/share/ClaireAI/models/claire-scratch.pt \
        --context 512 \
        --output runtime-data/models/candidates/claire-context512-base.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.is_file():
        parser.error(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    manifest = payload.get("manifest")
    if not isinstance(state, dict) or not isinstance(manifest, dict):
        parser.error("checkpoint has no state_dict/manifest")
    architecture = dict(manifest.get("architecture") or {})
    old_context = int(architecture.get("context") or 0)
    width = int(architecture.get("width") or 0)
    new_context = int(args.context)
    if new_context <= old_context:
        parser.error(f"--context {new_context} must be larger than {old_context}")
    positions = state.get("position_embedding.weight")
    if positions is None or tuple(positions.shape) != (old_context, width):
        parser.error("checkpoint position table does not match its manifest")

    expanded = dict(state)
    table = torch.empty((new_context, width), dtype=positions.dtype)
    table[:old_context] = positions
    tail = positions[-min(8, old_context):]
    table[old_context:] = tail.mean(dim=0, keepdim=True)
    expanded["position_embedding.weight"] = table

    architecture["context"] = new_context
    manifest = dict(manifest)
    manifest["name"] = args.name or f"{manifest.get('name', source.stem)}-ctx{new_context}"
    manifest["architecture"] = architecture
    manifest["expanded_from"] = str(source)
    manifest["expanded_from_context"] = old_context
    manifest["notes"] = "Position rows beyond the old window are seeded from the last rows and require training."

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": expanded, "manifest": manifest}, output)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"expanded {old_context} -> {new_context} positions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
