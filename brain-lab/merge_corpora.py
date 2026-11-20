#!/usr/bin/env python3
"""Merge the curated corpus with downloaded public instruction samples.

The base corpus (chat-v12) carries the assistant's own formats and the licensed
Dolly/TinyStories material. This script adds bounded external records, removes
held-out evaluation prompts, and writes a new corpus. It never downloads
anything; run ``fetch_corpora.py`` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
SEED = 20261120


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("messages"):
            records.append(item)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def held_out_prompts() -> set[str]:
    prompts: set[str] = set()
    for name in ("model_eval_cases.json", "conversation_cases.json"):
        path = ROOT.parent / "scripts" / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for case in payload.get("cases", []):
            if case.get("prompt"):
                prompts.add(str(case["prompt"]).strip())
            for turn in case.get("turns", []):
                if turn.get("user"):
                    prompts.add(str(turn["user"]).strip())
    return prompts


def _key(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages") or []
    user = str(messages[0].get("content") or "").strip() if messages else ""
    assistant = str(messages[-1].get("content") or "").strip() if messages else ""
    return user, assistant


def load_external(directory: Path, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for path in sorted(directory.glob("*.jsonl")):
        sidecar = path.with_suffix(".manifest.json")
        if not sidecar.is_file():
            continue
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            continue
        count = 0
        for record in read_jsonl(path):
            if count >= limit:
                break
            records.append(record)
            count += 1
        manifests[path.stem] = {**manifest, "used": count, "file": str(path)}
    return records, manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "data" / "chat-v12")
    parser.add_argument("--external", type=Path, default=ROOT / "data" / "external")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "chat-v13")
    parser.add_argument("--limit", type=int, default=4000, help="maximum records per external source")
    parser.add_argument("--base-limit", type=int, default=0, help="optional cap on base training records (0 = all)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    base_train = read_jsonl(args.base / "train.jsonl")
    base_validation = read_jsonl(args.base / "validation.jsonl")
    if not base_train or not base_validation:
        parser.error(f"{args.base} needs train.jsonl and validation.jsonl")
    if args.base_limit and len(base_train) > args.base_limit:
        rng0 = random.Random(args.seed)
        base_train = rng0.sample(base_train, args.base_limit)
    external, manifests = load_external(args.external, max(1, args.limit))
    if not external:
        parser.error(f"no licensed external JSONL with sidecars under {args.external}; run fetch_corpora.py")

    held_out = held_out_prompts()
    rng = random.Random(args.seed)
    seen: set[tuple[str, str]] = set()
    clean_external: list[dict[str, Any]] = []
    for record in external:
        messages = record.get("messages") or []
        if len(messages) < 2:
            continue
        user, assistant = _key(record)
        if not user or not assistant or user in held_out:
            continue
        if (user, assistant) in seen:
            continue
        seen.add((user, assistant))
        clean_external.append(record)

    rng.shuffle(clean_external)
    split = max(1, len(clean_external) // 10)
    ext_validation = clean_external[:split]
    ext_train = clean_external[split:]

    train = base_train + ext_train
    validation = base_validation + ext_validation
    rng.shuffle(train)
    rng.shuffle(validation)
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "validation.jsonl", validation)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "dataset": "claireai-chat-v13",
        "built_from": str(args.base),
        "external": manifests,
        "records_by_split": {"train": len(train), "validation": len(validation)},
        "external_records": len(clean_external),
        "base_train_records": len(base_train),
        "held_out_prompts_removed": len(held_out),
        "seed": args.seed,
        "licenses": {
            name: {"license": item.get("license"), "url": item.get("url"), "sha256": item.get("sha256")}
            for name, item in manifests.items()
        },
        "notes": [
            "External records are bounded samples, not a full corpus.",
            "Evaluation prompts are removed by exact match; run the eval suites after training.",
            "The gate, not this manifest, decides whether a checkpoint may be installed.",
        ],
        "files": {
            "train": {"sha256": digest(args.output / "train.jsonl"), "records": len(train)},
            "validation": {"sha256": digest(args.output / "validation.jsonl"), "records": len(validation)},
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(train)} train / {len(validation)} validation records to {args.output}")
    print(f"external: {len(clean_external)} from {', '.join(manifests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
