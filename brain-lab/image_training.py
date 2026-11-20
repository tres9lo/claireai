#!/usr/bin/env python3
"""Placeholder for the future image/vision training stage.

The shipped chat transformer is text-only. This module defines the manifest
shape that a future vision run will consume and validates it, but it does not
train a model, read pixels or claim visual capability. Keeping the schema here
means the data contract can be reviewed before any images are collected.

Example manifest:

    {
      "version": 1,
      "dataset": "image-training-v1",
      "examples": [
        {
          "image": "images/example.png",
          "instruction": "Click the Save button.",
          "target": {"kind": "point", "x": 412, "y": 188},
          "split": "train",
          "source": "owned",
          "license": "MIT"
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "data" / "image-training" / "manifest.json"
TARGET_KINDS = {"point", "box", "text"}
SPLITS = {"train", "validation", "test"}


def validate_manifest(payload: dict[str, Any], *, manifest_path: Path) -> list[str]:
    """Return human-readable problems; an empty list means the manifest is usable."""
    problems: list[str] = []
    if payload.get("version") != 1:
        problems.append("version must be 1")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        return problems + ["examples must be a list"]
    for index, item in enumerate(examples):
        where = f"examples[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{where} must be an object")
            continue
        for key in ("image", "instruction", "target", "split", "source", "license"):
            if key not in item:
                problems.append(f"{where} is missing {key}")
        if item.get("split") not in SPLITS:
            problems.append(f"{where}.split must be one of {sorted(SPLITS)}")
        target = item.get("target")
        if not isinstance(target, dict) or target.get("kind") not in TARGET_KINDS:
            problems.append(f"{where}.target.kind must be one of {sorted(TARGET_KINDS)}")
        image = item.get("image")
        if isinstance(image, str) and not (manifest_path.parent / image).is_file():
            problems.append(f"{where}.image does not exist: {image}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not read {args.manifest}: {exc}")
        return 2
    problems = validate_manifest(payload, manifest_path=args.manifest)
    if problems:
        print("Invalid image-training manifest:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    count = len(payload.get("examples") or [])
    print(f"Image-training manifest is valid: {count} example(s).")
    print("Training is not implemented; the text model does not read images yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
